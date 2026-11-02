//! The adapters' optimizer and the one control on it: Adam per parameter, and per expert a
//! ceiling on how far the expert's **effective** weights move in a step, measured against the
//! base expert's own size (`continual-adaptation.md` §6: "a ceiling, not a normaliser" — the
//! derived rate caps movement and leaves small steps small). The same measurement is the drift
//! instrument: each expert's distance from the base, `‖B·A‖_F / ‖W‖_F` per matrix, after every
//! step.
//!
//! An adapter's effective update is `Δ(B·A) = (B+ΔB)(A+ΔA) − BA`, rank ≤ 2r; its Frobenius norm
//! comes from r×r Gram matrices, never from the `[m][k]` product:
//! `‖BA‖² = tr((BᵀB)(AAᵀ))`, `⟨BA, B'A'⟩ = tr((BᵀB')(A'Aᵀ))`. When the step's relative movement
//! over the expert's three matrices exceeds the cap, both deltas are scaled by one factor `s`
//! found by bisection so the movement lands on the cap exactly.
//!
//! Each expert keeps its own step count (its gradient arrives only when a token routes to it),
//! as the inference crate's predictor does per pair.

use crate::expert::Adapter;
use glm_5_3_flash::expert::Weights;
use glm_5_3_flash::{D_FF, H};
use train_common::adam::{self, Adam};

/// The optimizer's settings: Adam's, and the cap on an expert's relative movement in one step
/// (`‖Δ(BA)‖_F / ‖W‖_F`, summed in squares over gate, up and down).
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub adam: Adam,
    pub cap: f32,
}

/// One expert's state: the moments of its six matrices, its step count, and the base expert's
/// Frobenius norms (gate, up, down) the movement is measured against.
pub struct ExpertState {
    pub t: u32,
    pub m: Adapter,
    pub v: Adapter,
    pub base_norm: [f32; 3],
}

impl ExpertState {
    pub fn new(r: usize, base_norm: [f32; 3]) -> Self {
        ExpertState { t: 0, m: Adapter::zeros(r), v: Adapter::zeros(r), base_norm }
    }
}

/// What a step reports for one expert.
#[derive(Clone, Copy, Debug, Default)]
pub struct Movement {
    /// The step's relative movement before the cap, and the factor applied (1 when uncapped).
    pub proposed: f32,
    pub scale: f32,
    /// Distance from the base after the step: `√Σ ‖BA‖² / √Σ ‖W‖²` over the three matrices.
    pub distance: f32,
}

/// `‖W‖_F` of the base expert's gate, up and down from its q4 record: the dequantized values
/// squared, read tile by tile through `common::q4`'s byte layout.
pub fn base_norms(w: &Weights) -> [f32; 3] {
    use common::q4::{GROUP, TILE, WBIAS};
    use glm_5_3_flash::expert::{DOWN_PARTS, DOWN_ROWS, GU_PARTS, GU_ROWS};
    fn sumsq(k: usize, m: usize, w: &[u8], ws: &[f32]) -> f64 {
        let ng = k / GROUP;
        let mut acc = 0f64;
        for t in 0..m / TILE {
            for g in 0..ng {
                let base = t * k * 2 + g * (GROUP / 2) * TILE;
                for r in 0..TILE {
                    let s = ws[t * ng * TILE + g * TILE + r] as f64;
                    let mut ssq = 0i64;
                    for b in 0..GROUP / 2 {
                        let byte = w[base + r * (GROUP / 2) + b];
                        let (lo, hi) = ((byte & 0x0F) as i64 - WBIAS as i64, (byte >> 4) as i64 - WBIAS as i64);
                        ssq += lo * lo + hi * hi;
                    }
                    acc += ssq as f64 * s * s;
                }
            }
        }
        acc
    }
    let (mut g, mut u, mut d) = (0f64, 0f64, 0f64);
    for c in 0..GU_PARTS {
        let (gw, uw, gs, us) = w.gu(c);
        g += sumsq(H, GU_ROWS, gw, gs);
        u += sumsq(H, GU_ROWS, uw, us);
    }
    for c in 0..DOWN_PARTS {
        let (dw, ds) = w.down(c);
        d += sumsq(D_FF, DOWN_ROWS, dw, ds);
    }
    [g.sqrt() as f32, u.sqrt() as f32, d.sqrt() as f32]
}

/// `G = Xᵀ·Y` for `X: [m][r]`, `Y: [m][r]` → `[r][r]` (f64).
fn gram_cols(m: usize, r: usize, x: &[f32], y: &[f32]) -> Vec<f64> {
    let mut g = vec![0f64; r * r];
    for i in 0..m {
        let (xi, yi) = (&x[i * r..(i + 1) * r], &y[i * r..(i + 1) * r]);
        for a in 0..r {
            for b in 0..r {
                g[a * r + b] += xi[a] as f64 * yi[b] as f64;
            }
        }
    }
    g
}
/// `G = X·Yᵀ` for `X: [r][k]`, `Y: [r][k]` → `[r][r]` (f64).
fn gram_rows(r: usize, k: usize, x: &[f32], y: &[f32]) -> Vec<f64> {
    let mut g = vec![0f64; r * r];
    for a in 0..r {
        for b in 0..r {
            let (xa, yb) = (&x[a * k..(a + 1) * k], &y[b * k..(b + 1) * k]);
            let mut s = 0f64;
            for j in 0..k {
                s += xa[j] as f64 * yb[j] as f64;
            }
            g[a * r + b] = s;
        }
    }
    g
}
/// `⟨B·A, B'·A'⟩_F = tr((BᵀB')(A'Aᵀ))`.
fn inner(r: usize, bb: &[f64], aa: &[f64]) -> f64 {
    let mut s = 0f64;
    for a in 0..r {
        for b in 0..r {
            s += bb[a * r + b] * aa[b * r + a];
        }
    }
    s
}

/// One matrix's `‖B·A‖²_F` and, for a candidate `(B + s·dB, A + s·dA)`, `‖Δ(BA)‖²_F` as a
/// function of `s`, from the Gram matrices of the four factors.
struct Pair {
    r: usize,
    bb: Vec<f64>,   // BᵀB
    bdb: Vec<f64>,  // BᵀdB
    dbdb: Vec<f64>, // dBᵀdB
    aa: Vec<f64>,   // AAᵀ
    ada: Vec<f64>,  // A dAᵀ
    dada: Vec<f64>, // dA dAᵀ
}

impl Pair {
    fn new(m: usize, r: usize, k: usize, b: &[f32], db: &[f32], a: &[f32], da: &[f32]) -> Self {
        Pair { r, bb: gram_cols(m, r, b, b), bdb: gram_cols(m, r, b, db), dbdb: gram_cols(m, r, db, db), aa: gram_rows(r, k, a, a), ada: gram_rows(r, k, a, da), dada: gram_rows(r, k, da, da) }
    }
    #[cfg(test)]
    fn norm2(&self) -> f64 {
        inner(self.r, &self.bb, &self.aa)
    }
    /// `‖(B − s dB)(A − s dA) − BA‖²` (Adam's step subtracts): with `X = −s(dB·A + B·dA) + s²·dB·dA`,
    /// expanded over the Gram matrices.
    fn delta2(&self, s: f64) -> f64 {
        let r = self.r;
        let t = |bb: &[f64], aa: &[f64]| inner(r, bb, aa);
        // transposes of the mixed Grams: (dBᵀB) = (BᵀdB)ᵀ, (dA Aᵀ) = (A dAᵀ)ᵀ
        let tr = |g: &[f64]| -> Vec<f64> { (0..r * r).map(|i| g[(i % r) * r + i / r]).collect() };
        let (dbb, daa) = (tr(&self.bdb), tr(&self.ada));
        // X = −s·P + s²·Q, P = dB·A + B·dA, Q = dB·dA
        // ‖P‖² = ‖dB·A‖² + ‖B·dA‖² + 2⟨dB·A, B·dA⟩
        let pp = t(&self.dbdb, &self.aa) + t(&self.bb, &self.dada) + 2.0 * t(&dbb, &self.ada);
        // ⟨P, Q⟩ = ⟨dB·A, dB·dA⟩ + ⟨B·dA, dB·dA⟩
        let pq = t(&self.dbdb, &daa) + t(&self.bdb, &self.dada);
        let qq = t(&self.dbdb, &self.dada);
        s * s * pp - 2.0 * s * s * s * pq + s * s * s * s * qq
    }
}

/// Adam's step on one expert's adapter under the movement cap. `grads` is consumed (zeroed).
pub fn step(set: &Settings, st: &mut ExpertState, ad: &mut Adapter, grads: &mut Adapter) -> Movement {
    let r = ad.r;
    st.t += 1;
    let mut d = Adapter::zeros(r);
    for (g, m, v, dd) in [(&grads.ga, &mut st.m.ga, &mut st.v.ga, &mut d.ga), (&grads.gb, &mut st.m.gb, &mut st.v.gb, &mut d.gb), (&grads.ua, &mut st.m.ua, &mut st.v.ua, &mut d.ua), (&grads.ub, &mut st.m.ub, &mut st.v.ub, &mut d.ub), (&grads.da, &mut st.m.da, &mut st.v.da, &mut d.da), (&grads.db, &mut st.m.db, &mut st.v.db, &mut d.db)] {
        set.adam.moments(st.t, g, m, v, dd);
    }
    // the effective movement of the three matrices, as a function of the common scale s
    let pairs = [Pair::new(D_FF, r, H, &ad.gb, &d.gb, &ad.ga, &d.ga), Pair::new(D_FF, r, H, &ad.ub, &d.ub, &ad.ua, &d.ua), Pair::new(H, r, D_FF, &ad.db, &d.db, &ad.da, &d.da)];
    let base2: f64 = st.base_norm.iter().map(|&n| n as f64 * n as f64).sum();
    let rel = |s: f64| (pairs.iter().map(|p| p.delta2(s)).sum::<f64>() / base2).sqrt();
    let proposed = rel(1.0);
    let mut s = 1.0f64;
    if proposed > set.cap as f64 {
        // ‖Δ(s)‖ is monotone in s on (0, 1] for any step small against the adapter: bisect
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if rel(mid) > set.cap as f64 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        s = lo;
    }
    for (p, dd) in [(&mut ad.ga, &d.ga), (&mut ad.gb, &d.gb), (&mut ad.ua, &d.ua), (&mut ad.ub, &d.ub), (&mut ad.da, &d.da), (&mut ad.db, &d.db)] {
        adam::apply(p, dd, s as f32);
    }
    for g in [&mut grads.ga, &mut grads.gb, &mut grads.ua, &mut grads.ub, &mut grads.da, &mut grads.db] {
        g.iter_mut().for_each(|v| *v = 0.0);
    }
    Movement { proposed: proposed as f32, scale: s as f32, distance: distance(ad, &st.base_norm) }
}

/// The adapter's distance from the base: `√Σ_matrices ‖BA‖² / √Σ ‖W‖²`.
pub fn distance(ad: &Adapter, base_norm: &[f32; 3]) -> f32 {
    let r = ad.r;
    let n2 = |m: usize, k: usize, b: &[f32], a: &[f32]| inner(r, &gram_cols(m, r, b, b), &gram_rows(r, k, a, a));
    let num = n2(D_FF, H, &ad.gb, &ad.ga) + n2(D_FF, H, &ad.ub, &ad.ua) + n2(H, D_FF, &ad.db, &ad.da);
    let den: f64 = base_norm.iter().map(|&n| n as f64 * n as f64).sum();
    (num / den).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert::tests::{random_adapter, Fixture};
    use common::testing::Rng;
    use glm_5_3_flash::FIRST_K_DENSE_REPLACE;

    fn dense_norm2(m: usize, r: usize, k: usize, b: &[f32], a: &[f32]) -> f64 {
        let mut s = 0f64;
        for i in 0..m {
            for j in 0..k {
                let mut v = 0f32;
                for t in 0..r {
                    v += b[i * r + t] * a[t * k + j];
                }
                s += v as f64 * v as f64;
            }
        }
        s
    }

    /// The Gram-matrix norms against the dense products, and the base norms against the fixture's dense twin.
    #[test]
    fn norms_match_dense() {
        let mut rnd = Rng::new(0x0971_0001);
        let fx = Fixture::new(&mut rnd);
        let st = fx.stream();
        let w = Weights::at(&st, FIRST_K_DENSE_REPLACE, 0);
        let bn = base_norms(&w);
        let want = |rows: &[f32]| rows.iter().map(|v| *v as f64 * *v as f64).sum::<f64>().sqrt() as f32;
        let (g, u) = fx.gate_up.split_at(D_FF * H);
        for (got, w) in [(bn[0], want(g)), (bn[1], want(u)), (bn[2], want(&fx.down))] {
            assert!((got - w).abs() <= 1e-4 * w, "{got} vs {w}");
        }
        let r = 4;
        let ad = random_adapter(r, 0.05, &mut rnd);
        let d = distance(&ad, &bn);
        let num = dense_norm2(D_FF, r, H, &ad.gb, &ad.ga) + dense_norm2(D_FF, r, H, &ad.ub, &ad.ua) + dense_norm2(H, r, D_FF, &ad.db, &ad.da);
        let den: f64 = bn.iter().map(|&n| n as f64 * n as f64).sum();
        let want = (num / den).sqrt() as f32;
        assert!((d - want).abs() <= 1e-4 * want, "{d} vs {want}");
        // the step's movement function against the dense difference at s = 0.3
        let dl = random_adapter(r, 0.01, &mut rnd);
        let p = Pair::new(D_FF, r, H, &ad.gb, &dl.gb, &ad.ga, &dl.ga);
        let s = 0.3f32;
        let b2: Vec<f32> = ad.gb.iter().zip(&dl.gb).map(|(x, y)| x - s * y).collect();
        let a2: Vec<f32> = ad.ga.iter().zip(&dl.ga).map(|(x, y)| x - s * y).collect();
        let mut diff = 0f64;
        for i in 0..D_FF {
            for j in 0..H {
                let (mut v1, mut v2) = (0f32, 0f32);
                for t in 0..r {
                    v1 += ad.gb[i * r + t] * ad.ga[t * H + j];
                    v2 += b2[i * r + t] * a2[t * H + j];
                }
                diff += (v2 - v1) as f64 * (v2 - v1) as f64;
            }
        }
        let got = p.delta2(s as f64);
        assert!((got - diff).abs() <= 1e-3 * diff, "{got} vs {diff}");
        assert!((p.norm2() - dense_norm2(D_FF, r, H, &ad.gb, &ad.ga)).abs() <= 1e-4 * p.norm2());
    }

    /// A small gradient steps uncapped; a huge one lands exactly on the cap; the distance grows accordingly.
    #[test]
    fn cap_binds_only_when_exceeded() {
        let mut rnd = Rng::new(0x0971_0002);
        let bn = [900.0f32, 900.0, 900.0];
        let r = 4;
        let set = Settings { adam: Adam { lr: 1e-3, ..Adam::default() }, cap: 1e-3 };
        let mut ad = random_adapter(r, 0.05, &mut rnd);
        let mut st = ExpertState::new(r, bn);
        let mut g = random_adapter(r, 1e-6, &mut rnd);
        let mv = step(&set, &mut st, &mut ad, &mut g);
        assert!(mv.scale == 1.0 && mv.proposed < set.cap, "{mv:?}");
        assert!(g.ga.iter().all(|v| *v == 0.0), "the gradient is consumed");
        // Adam's first step is lr in every coordinate whatever the gradient's size: a cap of 1e-6 binds
        let set2 = Settings { adam: set.adam, cap: 1e-6 };
        let before = ad.clone();
        let mut g = random_adapter(r, 1.0, &mut rnd);
        let mv = step(&set2, &mut st, &mut ad, &mut g);
        assert!(mv.scale < 1.0 && mv.proposed > set2.cap, "{mv:?}");
        // the realized movement is the cap
        let mut realized = 0f64;
        for (b, a, b0, a0, m, k) in [(&ad.gb, &ad.ga, &before.gb, &before.ga, D_FF, H), (&ad.ub, &ad.ua, &before.ub, &before.ua, D_FF, H), (&ad.db, &ad.da, &before.db, &before.da, H, D_FF)] {
            for i in 0..m {
                for j in 0..k {
                    let (mut v1, mut v0) = (0f32, 0f32);
                    for t in 0..r {
                        v1 += b[i * r + t] * a[t * k + j];
                        v0 += b0[i * r + t] * a0[t * k + j];
                    }
                    realized += (v1 - v0) as f64 * (v1 - v0) as f64;
                }
            }
        }
        let rel = (realized / bn.iter().map(|&n| n as f64 * n as f64).sum::<f64>()).sqrt();
        assert!((rel - set2.cap as f64).abs() <= 1e-2 * set2.cap as f64, "realized {rel} against the cap {}", set2.cap);
    }
}
