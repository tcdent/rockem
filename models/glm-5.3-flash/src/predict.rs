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
//!
//! The decode slot predicts on the main thread from the router's scores `ffn::picks` already made (no exponentials
//! there), and a housekeeping core learns: on the main thread the learning cost 50 µs a layer, the verify step
//! 56.3 → 59.7 ms (2026-10-21). Each pair is behind its own lock, and a learning job names the prediction it is for.
//! Prefill trains it too (`train`, phase 4): every prompt position is a label, in decode-sized batches.

use crate::{E, TOP_K};
use std::sync::Mutex;

use crate::plan::{LR_B, LR_CO, PAIRS_SCALE};

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
    seq: u64,
    n: usize,
    d: Vec<f32>, // [n][E] the route-ahead margin
    z: Vec<f32>, // [n][E] the logits
    picks: Vec<[usize; TOP_K]>,
}

/// Every MoE layer pair's predictor and the layers' router biases.
pub struct Predictor {
    pairs: Vec<Option<Mutex<Box<Pair>>>>, // indexed by layer l (the pair l → l + 1)
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
                    Mutex::new(Box::new(Pair {
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
                        seq: 0,
                        n: 0,
                        d: Vec::new(),
                        z: Vec::new(),
                        picks: Vec::new(),
                    }))
                })
            })
            .collect();
        Predictor { pairs, router_bias }
    }

    /// Whether layer l has a pair (predicts layer l + 1).
    pub fn has(&self, l: usize) -> bool {
        self.pairs.get(l).is_some_and(|p| p.is_some())
    }

    /// Layer l + 1's logits for the step's `n` tokens into `z` ([n][E]), from layer l + 1's route-ahead scores (token
    /// i's E sigmoids at `scores[i * stride..]`), the router's picks over them (`ahead`, best first: the 8th is the
    /// margin's threshold) and each token's picks at l; kept for `learn`. Returns the prediction's number for `learn`.
    pub fn predict(&self, l: usize, n: usize, scores: &[f32], stride: usize, ahead: &[[usize; TOP_K]], picks: &[[usize; TOP_K]], z: &mut [f32]) -> u64 {
        let bias_next = &self.router_bias[l + 1][..E];
        let mut g = self.pairs[l].as_ref().expect("no predictor for this layer").lock().unwrap();
        let p = &mut **g;
        p.seq += 1;
        p.n = n;
        p.d.resize(n * E, 0.0);
        p.z.resize(n * E, 0.0);
        p.picks.clear();
        p.picks.extend_from_slice(&picks[..n]);
        let (ca, cc) = (p.cal[0], p.cal[1]);
        for i in 0..n {
            let sc = &scores[i * stride..i * stride + E];
            let k = ahead[i][TOP_K - 1];
            let thr = sc[k] + bias_next[k];
            let (d, zt, bias) = (&mut p.d[i * E..(i + 1) * E], &mut p.z[i * E..(i + 1) * E], &p.bias);
            for e in 0..E {
                d[e] = (sc[e] + bias_next[e] - thr) * 10.0;
                zt[e] = ca * d[e] + cc + bias[e];
            }
            for &q in &picks[i] {
                let row = &p.co[q * E..(q + 1) * E];
                for e in 0..E {
                    zt[e] += row[e];
                }
            }
        }
        z[..n * E].copy_from_slice(&p.z[..n * E]);
        p.seq
    }

    /// Pair l's update from its prediction number `seq` and the next layer's picks (`labels`, a token's 8 at l + 1);
    /// nothing when a later prediction has replaced it.
    pub fn learn(&self, l: usize, seq: u64, labels: &[[usize; TOP_K]]) {
        let Some(m) = self.pairs.get(l).and_then(|p| p.as_ref()) else { return };
        let mut g = m.lock().unwrap();
        let p = &mut **g;
        if p.seq != seq {
            return;
        }
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

impl Predictor {
    /// Pair l trained on a prefill chunk's `n` tokens: layer l + 1's route-ahead logits (`logits` [n][E], before the
    /// sigmoid), the picks at l and at l + 1 (`labels`), in batches of `batch` tokens as decode's steps learn — a
    /// prediction and its update a batch, nothing ranked.
    pub fn train(&self, l: usize, n: usize, logits: &[f32], picks: &[[usize; TOP_K]], labels: &[[usize; TOP_K]], batch: usize) {
        if !self.has(l) {
            return;
        }
        let bias_next = &self.router_bias[l + 1][..E];
        let mut sc = vec![0f32; batch * E];
        let mut z = vec![0f32; batch * E];
        let mut ahead = vec![[0usize; TOP_K]; batch];
        let mut a = 0;
        while a < n {
            let m = batch.min(n - a);
            for i in 0..m {
                let (lg, s) = (&logits[(a + i) * E..(a + i + 1) * E], &mut sc[i * E..(i + 1) * E]);
                let (mut idx, mut val) = ([0usize; TOP_K], [f32::NEG_INFINITY; TOP_K]);
                for e in 0..E {
                    s[e] = 1.0 / (1.0 + (-lg[e]).exp());
                    let v = s[e] + bias_next[e];
                    if v > val[TOP_K - 1] {
                        let mut j = TOP_K - 1;
                        while j > 0 && v > val[j - 1] {
                            val[j] = val[j - 1];
                            idx[j] = idx[j - 1];
                            j -= 1;
                        }
                        val[j] = v;
                        idx[j] = e;
                    }
                }
                ahead[i] = idx;
            }
            let seq = self.predict(l, m, &sc, E, &ahead[..m], &picks[a..a + m], &mut z);
            self.learn(l, seq, &labels[a..a + m]);
            a += m;
        }
    }
}

/// The step's experts ranked from its tokens' logits ([n][E]) by Σ softplus(z) (the union's −log Π(1 − p)), best
/// first, `skip`ped experts left out, at most `cap`: as Π(1 + eᶻ), the same order without a logarithm (z clamped to
/// 20 so four tokens' product stays finite), eᶻ to ~1e-4 relative.
pub fn rank(z: &[f32], n: usize, skip: &dyn Fn(usize) -> bool, cap: usize, out: &mut Vec<usize>) {
    let mut score = [1f32; E];
    for i in 0..n {
        let zt = &z[i * E..(i + 1) * E];
        for e in 0..E {
            score[e] *= 1.0 + exp(zt[e].min(20.0));
        }
    }
    // one pass keeping the best `cap` (an expert goes after every kept one at least as large: the lower id first
    // between equals)
    out.clear();
    let k = cap.min(E);
    if k == 0 {
        return;
    }
    let mut val: [f32; E] = [0.0; E];
    for e in 0..E {
        let v = score[e];
        if (out.len() == k && !(v > val[k - 1])) || skip(e) {
            continue;
        }
        let mut j = out.len().min(k - 1);
        if out.len() < k {
            out.push(e);
        }
        while j > 0 && v > val[j - 1] {
            val[j] = val[j - 1];
            out[j] = out[j - 1];
            j -= 1;
        }
        val[j] = v;
        out[j] = e;
    }
}

/// eˣ for x in [−87, 20]: 2^(x log2 e) as an exponent and a degree-5 polynomial over [0, 1) (~1e-4 relative; a branchless
/// loop the compiler vectorizes).
#[inline(always)]
fn exp(x: f32) -> f32 {
    // floor without a call (the crates build for baseline x86-64, where `floor` is libm's): truncate, then one down
    // where that rounded up
    let t = x.max(-87.0) * std::f32::consts::LOG2_E;
    let i = t as i32;
    let i = i - (t < i as f32) as i32;
    let r = t - i as f32;
    let p = 1.0 + r * (0.693_147_2 + r * (0.240_226_5 + r * (0.055_504_11 + r * (0.009_618_129 + r * 0.001_333_355_8))));
    f32::from_bits(((i + 127) << 23) as u32) * p
}
