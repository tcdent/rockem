//! The hyper-connection (mHC) around a block half, one token: the backward of `hc::pre`
//! and `hc::post`. The forward is the inference crate's own (`glm_5_3_flash::hc::pre`
//! returns the mix; `post` mixes in place); this module keeps what it needs and inverts it.
//!
//! ```text
//! pre:   flat = S · r  (r = rsqrt(mean S² + eps), over all HC·H)     c = fn_ · flat   (24 bf16 rows, frozen)
//!        pre_i = σ(c_i·k0 + b_i) + ε       post_i = 2σ(c_{HC+i}·k1 + b_{HC+i})
//!        comb  = Sinkhorn(softmax rows of c_{2HC+ij}·k2 + b + ε)     x = Σ_i pre_i · S_i
//! block: y = f(x)
//! post:  S'_j = post_j · y + Σ_i comb_ij · S_i
//! ```
//!
//! Backward, in two halves around the block's own:
//!
//! - `post_backward`: given `d S'`, the gradient at `y` (`Σ_j post_j d S'_j`), the residual's
//!   share into `d S` (`Σ_j comb_ij d S'_j`), and the mix's (`d post_j = ⟨y, d S'_j⟩`,
//!   `d comb_ij = ⟨S_i, d S'_j⟩`), kept in a `DMix` for the pre half.
//! - `pre_backward`: given `d x` from the block and the `DMix`: `d pre_i = ⟨S_i, d x⟩`,
//!   `d S_i += pre_i · d x`; then the coefficients — the sigmoids, the softmax and the Sinkhorn
//!   chain unrolled in reverse over its recomputed intermediates — `d flat = fn_ᵀ · d c` over
//!   the 24 rows, and the unweighted norm's backward into `d S`.

use glm_5_3_flash::hc::{sigmoid, Mix};
use glm_5_3_flash::weights;
use glm_5_3_flash::{H, HC, HC_COEFS, HC_EPS, HC_SINKHORN_ITERS, RMS_NORM_EPS};
use train_common::{bf16, rmsnorm};

pub const FLAT: usize = HC * H;

/// What `pre`'s backward needs besides the streams: the coefficients the forward computed
/// (`hc::Scratch::coef` after `pre`) and the mix it returned.
pub struct Saved {
    pub coef: [f32; HC_COEFS],
    pub mix: Mix,
    pub pre: [f32; HC],
}

impl Saved {
    /// From the inference forward's scratch and result: `pre_i` is recomputed from the coefficients.
    pub fn from_forward(w: &weights::Hc, coef: &[f32], mix: Mix) -> Self {
        let mut c = [0f32; HC_COEFS];
        c.copy_from_slice(&coef[..HC_COEFS]);
        let mut pre = [0f32; HC];
        for i in 0..HC {
            pre[i] = sigmoid(c[i] * w.scale[0] + w.base[i]) + HC_EPS;
        }
        Saved { coef: c, mix, pre }
    }
}

/// The mix's gradient, from the post half to the pre half.
#[derive(Clone, Copy, Default)]
pub struct DMix {
    pub post: [f32; HC],
    pub comb: [[f32; HC]; HC],
}

/// `S'_j = post_j · y + Σ_i comb_ij · S_i`: given `d S'` (`[HC][H]`), the block output `y` and the
/// residual streams `S` the post read, accumulate `d y` (`+=`, `[H]`) and the residual's share of
/// `d S` (`+=`, `[HC][H]`); return the mix's gradient.
pub fn post_backward(mix: &Mix, y: &[f32], s: &[f32], ds_out: &[f32], dy: &mut [f32], ds: &mut [f32]) -> DMix {
    let mut dm = DMix::default();
    for j in 0..HC {
        let dj = &ds_out[j * H..(j + 1) * H];
        let mut dp = 0f32;
        for d in 0..H {
            dp += y[d] * dj[d];
            dy[d] += mix.post[j] * dj[d];
        }
        dm.post[j] = dp;
        for i in 0..HC {
            let si = &s[i * H..(i + 1) * H];
            let dsi = &mut ds[i * H..(i + 1) * H];
            let mut dc = 0f32;
            let cij = mix.comb[i][j];
            for d in 0..H {
                dc += si[d] * dj[d];
                dsi[d] += cij * dj[d];
            }
            dm.comb[i][j] = dc;
        }
    }
    dm
}

/// The Sinkhorn chain's intermediates: the softmax rows (with ε) and the matrix after each
/// normalization, columns first, then (rows, columns) × (iters − 1).
fn sinkhorn_forward(c: &[f32; HC_COEFS], w: &weights::Hc) -> ([[f32; HC]; HC], Vec<[[f32; HC]; HC]>) {
    let mut a = [[0f32; HC]; HC];
    for i in 0..HC {
        let mut row = [0f32; HC];
        let mut mx = f32::NEG_INFINITY;
        for j in 0..HC {
            row[j] = c[2 * HC + i * HC + j] * w.scale[2] + w.base[2 * HC + i * HC + j];
            if row[j] > mx {
                mx = row[j];
            }
        }
        let mut sum = 0f32;
        for j in 0..HC {
            row[j] = common::math::exp(row[j] - mx);
            sum += row[j];
        }
        for j in 0..HC {
            a[i][j] = row[j] / sum;
        }
    }
    let soft = a;
    for row in a.iter_mut() {
        for v in row.iter_mut() {
            *v += HC_EPS;
        }
    }
    let mut steps = Vec::with_capacity(2 * HC_SINKHORN_ITERS - 1);
    steps.push(a);
    let cols = |m: &mut [[f32; HC]; HC]| {
        for j in 0..HC {
            let mut sum = 0f32;
            for i in 0..HC {
                sum += m[i][j];
            }
            let sum = sum + HC_EPS;
            for i in 0..HC {
                m[i][j] /= sum;
            }
        }
    };
    cols(&mut a);
    steps.push(a);
    for _ in 0..HC_SINKHORN_ITERS - 1 {
        for row in a.iter_mut() {
            let mut sum = 0f32;
            for v in row.iter() {
                sum += v;
            }
            let sum = sum + HC_EPS;
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
        steps.push(a);
        cols(&mut a);
        steps.push(a);
    }
    (soft, steps)
}

/// The backward of `v' = v / (Σ v + ε)` over one line of `HC` values: `d v_k = (d v'_k − Σ_j d v'_j v'_j) / s`,
/// `s` recomputed from the inputs.
#[inline(always)]
fn norm_line_backward(v: [f32; HC], vn: [f32; HC], dvn: [f32; HC]) -> [f32; HC] {
    let s = v.iter().sum::<f32>() + HC_EPS;
    let dot: f32 = (0..HC).map(|j| dvn[j] * vn[j]).sum();
    core::array::from_fn(|k| (dvn[k] - dot) / s)
}

/// Given `d x` (`[H]`, the gradient at the block's input) and the mix's gradient, accumulate the
/// streams' gradient (`+=`, `[HC][H]`).
pub fn pre_backward(w: &weights::Hc, s: &[f32], saved: &Saved, dx: &[f32], dm: &DMix, ds: &mut [f32]) {
    // the collapse: d pre_i = ⟨S_i, d x⟩, d S_i += pre_i · d x
    let mut dc = [0f32; HC_COEFS];
    for i in 0..HC {
        let si = &s[i * H..(i + 1) * H];
        let dsi = &mut ds[i * H..(i + 1) * H];
        let mut dp = 0f32;
        let p = saved.pre[i];
        for d in 0..H {
            dp += si[d] * dx[d];
            dsi[d] += p * dx[d];
        }
        // pre_i = σ(z) + ε, z = c_i · k0 + b_i
        let sg = p - HC_EPS;
        dc[i] = dp * sg * (1.0 - sg) * w.scale[0];
        // post_i = 2σ(z), z = c_{HC+i} · k1 + b
        let sp = saved.mix.post[i] / 2.0;
        dc[HC + i] = dm.post[i] * 2.0 * sp * (1.0 - sp) * w.scale[1];
    }
    // comb: the Sinkhorn chain in reverse, then the softmax rows
    let (soft, steps) = sinkhorn_forward(&saved.coef, w);
    let mut g = dm.comb;
    // steps[k] is the matrix after normalization k; steps[0] the softmax + ε. Normalizations in
    // order: cols, then (rows, cols) × (iters − 1): odd k are column passes, even k ≥ 2 row passes.
    for k in (1..steps.len()).rev() {
        let (before, after) = (&steps[k - 1], &steps[k]);
        let mut gin = [[0f32; HC]; HC];
        if k % 2 == 1 {
            for j in 0..HC {
                let v: [f32; HC] = core::array::from_fn(|i| before[i][j]);
                let vn: [f32; HC] = core::array::from_fn(|i| after[i][j]);
                let dvn: [f32; HC] = core::array::from_fn(|i| g[i][j]);
                let dv = norm_line_backward(v, vn, dvn);
                for i in 0..HC {
                    gin[i][j] = dv[i];
                }
            }
        } else {
            for i in 0..HC {
                gin[i] = norm_line_backward(before[i], after[i], g[i]);
            }
        }
        g = gin;
    }
    // the + ε passes the gradient; the softmax: d row = a ⊙ (d a − Σ d a · a), then × k2
    for i in 0..HC {
        let dot: f32 = (0..HC).map(|j| g[i][j] * soft[i][j]).sum();
        for j in 0..HC {
            dc[2 * HC + i * HC + j] = soft[i][j] * (g[i][j] - dot) * w.scale[2];
        }
    }
    // d flat = fn_ᵀ · d c over the 24 rows; then the unweighted norm into d S
    let mut dflat = vec![0f32; FLAT];
    let rows: Vec<(usize, f32)> = (0..HC_COEFS).map(|i| (i, dc[i])).collect();
    bf16::rows_t::<FLAT>(w.fn_, &rows, &mut dflat);
    rmsnorm::backward_unweighted::<FLAT>(s, RMS_NORM_EPS, &dflat, ds);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference;
    use common::gemv::widen;
    use common::pool::Serial;
    use common::testing::Rng;

    fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(worst <= tol * scale, "{what}: worst |Δ| {worst} against a scale of {scale}");
    }

    /// The backward over the inference forward against the dense f32 reference, at the real shape,
    /// for one block half with the block as `y = x` (the block's own backward is its own test).
    #[test]
    fn matches_dense_reference() {
        let mut rnd = Rng::new(0x4C0E_0001);
        let fn_: Vec<u16> = (0..HC_COEFS * FLAT).map(|_| ((rnd.unit() * 0.02).to_bits() >> 16) as u16).collect();
        let fn_dense: Vec<f32> = fn_.iter().map(|&b| widen(b)).collect();
        let base: Vec<f32> = (0..HC_COEFS).map(|_| rnd.unit()).collect();
        let scale = [0.7f32, 1.3, 0.9];
        let w = weights::Hc { fn_: &fn_, base: &base, scale: &scale };
        let s: Vec<f32> = (0..FLAT).map(|_| rnd.unit()).collect();
        let ds_out: Vec<f32> = (0..FLAT).map(|_| rnd.unit()).collect();

        // forward: the inference crate's
        let (mut flat, mut coef) = (vec![0f32; FLAT], vec![0f32; HC_COEFS]);
        let mut x = vec![0f32; H];
        let mix = glm_5_3_flash::hc::pre(&w, &s, &mut glm_5_3_flash::hc::Scratch { flat: &mut flat, coef: &mut coef }, &Serial, &mut x);
        let saved = Saved::from_forward(&w, &coef, mix);
        let y = x.clone(); // the block as the identity
        let mut s_out = s.clone();
        glm_5_3_flash::hc::post(&mix, &y, &mut s_out);
        // backward
        let mut dy = vec![0f32; H];
        let mut ds = vec![0f32; FLAT];
        let dm = post_backward(&mix, &y, &s, &ds_out, &mut dy, &mut ds);
        let dx = dy.clone();
        pre_backward(&w, &s, &saved, &dx, &dm, &mut ds);

        let r = reference::HcRef { hc: HC, h: H, fn_: &fn_dense, base: &base, scale, eps: HC_EPS, rms_eps: RMS_NORM_EPS, iters: HC_SINKHORN_ITERS };
        let (rx, rmix) = reference::hc_pre(&r, &s);
        let rs_out = reference::hc_post(&r, &rmix, &rx, &s);
        close(&x, &rx, 1e-4, "x");
        close(&s_out, &rs_out, 1e-4, "S'");
        let (rdy, mut rds, rdm) = reference::hc_post_backward(&r, &rmix, &rx, &s, &ds_out);
        reference::hc_pre_backward(&r, &s, &rmix, &rdy, &rdm, &mut rds);
        close(&ds, &rds, 1e-4, "dS");
    }
}
