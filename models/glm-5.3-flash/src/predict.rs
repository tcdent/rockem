//! The decode prefetch's learned expert predictor (docs/expert-cache-architecture.md phase 3; queue 082, tcdent's
//! design: the route-ahead score plus a correction that starts at zero, learned online, never frozen).
//!
//! For each MoE layer pair (l → l + 1), per token, the next layer's expert logits are
//!
//!   z = ca · d + cc + bias + Σ co[picks at l]
//!
//! where d = 10 · (sigmoid(route-ahead logit) + router bias − the 8th best of that), the route-ahead logit being layer
//! l + 1's router over layer l's normed FFN input, and `co` a 288 × 288 co-routing table summed over the token's
//! picks at l. A step's experts are ranked by Σ over its tokens of softplus(z) = −log Π(1 − p). Every pair learns from
//! the next layer's picks a layer later (per-expert binary cross-entropy, Adam), so pair l − 1 is updated while
//! layer l runs and is used again only next step.
//!
//! Measured offline (bringup/expert-cache/phase3, the causal simulation over 4,986 decode steps at 18k, 2026-10-21),
//! settled recall of the next layer's picks at a cap of 13: the route-ahead alone .585, + bias .608, + co-routing
//! .621; a low-rank map of the token's input on top (rank 64 a layer) .628, but its dense update (~14M parameters,
//! ~330 MB of memory traffic a step) took 195 ms a step on one core against a 25 ms step, so it is left out. Only the
//! picked rows of `co` have a gradient, and only they are updated (Adam's moments with the pair's step count); the
//! rest is the simulation's math in f32: the loss summed over a step's tokens divided by tokens × `PAIRS_SCALE`,
//! torch's Adam (β 0.9 / 0.999, ε 1e-8). The prediction changes only which bytes the prefetch warms, never the output.

use crate::{E, TOP_K};

/// Adam's learning rates: the co-routing table; the calibration (ca, cc) and the bias.
pub const LR_CO: f32 = 3e-3;
pub const LR_B: f32 = 1e-2;
/// The simulation's loss divided by the 40 layer pairs it trained together; kept so its learning rates carry over.
const PAIRS_SCALE: f32 = 40.0;

/// torch.optim.Adam's step on one parameter: p −= lr / (1 − β1ᵗ) · m / (√v / √(1 − β2ᵗ) + ε).
#[inline(always)]
fn adam(p: &mut f32, m: &mut f32, v: &mut f32, g: f32, lr_c1: f32, c2: f32) {
    *m = 0.9 * *m + 0.1 * g;
    *v = 0.999 * *v + 0.001 * g * g;
    *p -= lr_c1 * *m / (v.sqrt() / c2 + 1e-8);
}

/// One layer pair's predictor, and what its last prediction leaves for its update.
struct Pair {
    cal: [f32; 2], // ca, cc
    bias: [f32; E],
    co: Vec<f32>, // [E][E]
    m_cal: [f32; 2],
    v_cal: [f32; 2],
    m_bias: [f32; E],
    v_bias: [f32; E],
    m_co: Vec<f32>,
    v_co: Vec<f32>,
    t: i32,
    n: usize,
    d: Vec<f32>, // [n][E] the route-ahead margin
    z: Vec<f32>, // [n][E] the logits
    picks: Vec<[usize; TOP_K]>,
}

/// Every MoE layer pair's predictor and the layers' router biases.
pub struct Predictor {
    pairs: Vec<Option<Box<Pair>>>, // indexed by layer l (the pair l → l + 1)
    router_bias: Vec<Vec<f32>>,    // indexed by layer: the router's selection bias (empty where dense)
}

impl Predictor {
    /// `router_bias[l]`: layer l's router selection bias (`e_score_correction_bias`), empty for a dense layer. A pair
    /// exists for every MoE layer l whose next layer is MoE; every correction starts at zero.
    pub fn new(router_bias: Vec<Vec<f32>>) -> Self {
        let layers = router_bias.len();
        let pairs = (0..layers)
            .map(|l| {
                (l + 1 < layers && !router_bias[l].is_empty() && !router_bias[l + 1].is_empty()).then(|| {
                    Box::new(Pair {
                        cal: [3.0, 0.0],
                        bias: [0.0; E],
                        co: vec![0.0; E * E],
                        m_cal: [0.0; 2],
                        v_cal: [0.0; 2],
                        m_bias: [0.0; E],
                        v_bias: [0.0; E],
                        m_co: vec![0.0; E * E],
                        v_co: vec![0.0; E * E],
                        t: 0,
                        n: 0,
                        d: Vec::new(),
                        z: Vec::new(),
                        picks: Vec::new(),
                    })
                })
            })
            .collect();
        Predictor { pairs, router_bias }
    }

    /// Whether layer l has a pair (predicts layer l + 1).
    pub fn has(&self, l: usize) -> bool {
        self.pairs.get(l).is_some_and(|p| p.is_some())
    }

    /// Layer l + 1's logits for the step's `n` tokens into `z` ([n][E]), from layer l + 1's route-ahead logits (`ahead`
    /// [n][E], before the sigmoid) and each token's picks at l; kept for `learn`.
    pub fn predict(&mut self, l: usize, n: usize, ahead: &[f32], picks: &[[usize; TOP_K]], z: &mut [f32]) {
        let bias_next = &self.router_bias[l + 1];
        let p = self.pairs[l].as_mut().expect("no predictor for this layer");
        p.n = n;
        p.d.resize(n * E, 0.0);
        p.z.resize(n * E, 0.0);
        p.picks.clear();
        p.picks.extend_from_slice(&picks[..n]);
        for i in 0..n {
            let (lg, d) = (&ahead[i * E..(i + 1) * E], &mut p.d[i * E..(i + 1) * E]);
            for e in 0..E {
                d[e] = 1.0 / (1.0 + (-lg[e]).exp()) + bias_next[e];
            }
            let mut top = [f32::NEG_INFINITY; TOP_K];
            for &g in d.iter() {
                if g > top[TOP_K - 1] {
                    let mut j = TOP_K - 1;
                    while j > 0 && g > top[j - 1] {
                        top[j] = top[j - 1];
                        j -= 1;
                    }
                    top[j] = g;
                }
            }
            let zt = &mut p.z[i * E..(i + 1) * E];
            for e in 0..E {
                d[e] = (d[e] - top[TOP_K - 1]) * 10.0;
                zt[e] = p.cal[0] * d[e] + p.cal[1] + p.bias[e];
            }
            for &q in &picks[i] {
                let row = &p.co[q * E..(q + 1) * E];
                for e in 0..E {
                    zt[e] += row[e];
                }
            }
        }
        z[..n * E].copy_from_slice(&p.z[..n * E]);
    }

    /// Pair l's update from its last prediction and the next layer's picks (`labels`, a token's 8 at l + 1).
    pub fn learn(&mut self, l: usize, labels: &[[usize; TOP_K]]) {
        let Some(p) = self.pairs.get_mut(l).and_then(|p| p.as_mut()) else { return };
        let n = std::mem::take(&mut p.n);
        if n == 0 || labels.len() < n {
            return;
        }
        // dL/dz = (sigmoid(z) − y) / (n · PAIRS_SCALE), into z's place
        let s = 1.0 / (n as f32 * PAIRS_SCALE);
        for i in 0..n {
            let g = &mut p.z[i * E..(i + 1) * E];
            g.iter_mut().for_each(|v| *v = s / (1.0 + (-*v).exp()));
            for &q in &labels[i] {
                g[q] -= s;
            }
        }
        let gz = &p.z;
        p.t += 1;
        let (c1, c2) = (1.0 - 0.9f32.powi(p.t), (1.0 - 0.999f32.powi(p.t)).sqrt());
        // calibration and bias (dense)
        let (mut gcal, mut gb) = ([0f32; 2], [0f32; E]);
        for i in 0..n {
            for e in 0..E {
                let g = gz[i * E + e];
                gcal[0] += g * p.d[i * E + e];
                gcal[1] += g;
                gb[e] += g;
            }
        }
        for k in 0..2 {
            adam(&mut p.cal[k], &mut p.m_cal[k], &mut p.v_cal[k], gcal[k], LR_B / c1, c2);
        }
        for e in 0..E {
            adam(&mut p.bias[e], &mut p.m_bias[e], &mut p.v_bias[e], gb[e], LR_B / c1, c2);
        }
        // co-routing: each picked row's gradient is the sum of its tokens' gz
        let mut rows: [usize; 4 * TOP_K] = [usize::MAX; 4 * TOP_K];
        let mut nr = 0;
        for i in 0..n {
            for &q in &p.picks[i] {
                if !rows[..nr].contains(&q) {
                    if nr == rows.len() {
                        break;
                    }
                    rows[nr] = q;
                    nr += 1;
                }
            }
        }
        let mut g = [0f32; E];
        for &q in &rows[..nr] {
            g.iter_mut().for_each(|v| *v = 0.0);
            for i in 0..n {
                if p.picks[i].contains(&q) {
                    for e in 0..E {
                        g[e] += gz[i * E + e];
                    }
                }
            }
            let r = q * E..(q + 1) * E;
            let (co, m, v) = (&mut p.co[r.clone()], &mut p.m_co[r.clone()], &mut p.v_co[r]);
            for e in 0..E {
                adam(&mut co[e], &mut m[e], &mut v[e], g[e], LR_CO / c1, c2);
            }
        }
    }
}

/// The step's experts ranked from its tokens' logits ([n][E]): by Σ softplus(z) (the union's −log Π(1 − p)), best
/// first, `skip`ped experts left out, at most `cap`.
pub fn rank(z: &[f32], n: usize, skip: &dyn Fn(usize) -> bool, cap: usize, out: &mut Vec<usize>) {
    let mut score = [0f32; E];
    for i in 0..n {
        for e in 0..E {
            let v = z[i * E + e];
            score[e] += if v > 20.0 { v } else { (1.0 + v.exp()).ln() };
        }
    }
    out.clear();
    let mut ix: Vec<usize> = (0..E).filter(|&e| !skip(e)).collect();
    ix.sort_by(|&a, &b| score[b].partial_cmp(&score[a]).unwrap());
    out.extend_from_slice(&ix[..cap.min(ix.len())]);
}
