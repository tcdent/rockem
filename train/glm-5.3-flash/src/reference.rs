//! The f32 reference of `expert`: dense matrices, plain loops, any shape —
//! the formulas the kernels are diffed against, and the formulas themselves
//! checked by finite differences at a small shape with the quantizers off.
//! The quantizers, when on, are `common::q4::QAct`'s: the kernel's INT8
//! activation path and this reference then see the same `x̂` and `ĥ`.

use common::q4::{QAct, GROUP};
use common::swiglu::silu;

/// A dense expert: `gate_up: [2·ff][h]` (gate rows then up rows), `down: [h][ff]`.
pub struct Dense<'a> {
    pub h: usize,
    pub ff: usize,
    pub gate_up: &'a [f32],
    pub down: &'a [f32],
    pub limit: f32,
}

/// Its adapter: `ga, ua: [r][h]`, `gb, ub: [ff][r]`, `da: [r][ff]`, `db: [h][r]`.
pub struct LowRank<'a> {
    pub r: usize,
    pub ga: &'a [f32],
    pub gb: &'a [f32],
    pub ua: &'a [f32],
    pub ub: &'a [f32],
    pub da: &'a [f32],
    pub db: &'a [f32],
}

/// The adapter's gradient, owned.
pub struct Grads {
    pub ga: Vec<f32>,
    pub gb: Vec<f32>,
    pub ua: Vec<f32>,
    pub ub: Vec<f32>,
    pub da: Vec<f32>,
    pub db: Vec<f32>,
}

pub struct Saved {
    pub xh: Vec<f32>,
    pub g: Vec<f32>,
    pub u: Vec<f32>,
    pub hh: Vec<f32>,
    pub zg: Vec<f32>,
    pub zu: Vec<f32>,
    pub zd: Vec<f32>,
}

fn matvec(m: usize, k: usize, w: &[f32], x: &[f32]) -> Vec<f32> {
    (0..m).map(|i| (0..k).map(|j| w[i * k + j] * x[j]).sum()).collect()
}

fn matvec_t(m: usize, k: usize, w: &[f32], dy: &[f32]) -> Vec<f32> {
    let mut dx = vec![0f32; k];
    for i in 0..m {
        for j in 0..k {
            dx[j] += w[i * k + j] * dy[i];
        }
    }
    dx
}

fn outer_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i * b.len() + j] += ai * bj;
        }
    }
}

/// `deq(Q(x))` with the kernel's quantizer; `x` itself when `quant` is off (then the
/// width need not be a multiple of the group).
fn requantize(x: &[f32], quant: bool) -> Vec<f32> {
    if !quant {
        return x.to_vec();
    }
    assert!(x.len() % GROUP == 0);
    let (mut q, mut s, mut gs) = (vec![0i8; x.len()], vec![0f32; x.len() / GROUP], vec![0i32; x.len() / GROUP]);
    QAct { q: &mut q, scale: &mut s, gsum: &mut gs }.quantize(x);
    (0..x.len()).map(|i| q[i] as f32 * s[i / GROUP]).collect()
}

pub fn forward(d: &Dense, a: &LowRank, x: &[f32], quant: bool) -> (Vec<f32>, Saved) {
    let (h, ff, r) = (d.h, d.ff, a.r);
    let xh = requantize(x, quant);
    let gu = matvec(2 * ff, h, d.gate_up, &xh);
    let zg = matvec(r, h, a.ga, &xh);
    let zu = matvec(r, h, a.ua, &xh);
    let mut g = gu[..ff].to_vec();
    let mut u = gu[ff..].to_vec();
    for (gi, v) in g.iter_mut().zip(matvec(ff, r, a.gb, &zg)) {
        *gi += v;
    }
    for (ui, v) in u.iter_mut().zip(matvec(ff, r, a.ub, &zu)) {
        *ui += v;
    }
    let hv: Vec<f32> = (0..ff).map(|i| silu(g[i].min(d.limit)) * u[i].clamp(-d.limit, d.limit)).collect();
    let hh = requantize(&hv, quant);
    let mut out = matvec(h, ff, d.down, &hh);
    let zd = matvec(r, ff, a.da, &hh);
    for (oi, v) in out.iter_mut().zip(matvec(h, r, a.db, &zd)) {
        *oi += v;
    }
    (out, Saved { xh, g, u, hh, zg, zu, zd })
}

pub fn backward(d: &Dense, a: &LowRank, s: &Saved, dout: &[f32]) -> (Vec<f32>, Grads) {
    let (h, ff, r) = (d.h, d.ff, a.r);
    let mut gr = Grads { ga: vec![0.0; r * h], gb: vec![0.0; ff * r], ua: vec![0.0; r * h], ub: vec![0.0; ff * r], da: vec![0.0; r * ff], db: vec![0.0; h * r] };
    // down
    let mut dh = matvec_t(h, ff, d.down, dout);
    let t = matvec_t(h, r, a.db, dout);
    outer_add(dout, &s.zd, &mut gr.db);
    outer_add(&t, &s.hh, &mut gr.da);
    for (v, w) in dh.iter_mut().zip(matvec_t(r, ff, a.da, &t)) {
        *v += w;
    }
    // the clamped SwiGLU
    let mut dg = vec![0f32; ff];
    let mut du = vec![0f32; ff];
    for i in 0..ff {
        let (g, u) = (s.g[i], s.u[i]);
        let gc = g.min(d.limit);
        let uc = u.clamp(-d.limit, d.limit);
        let sg = 1.0 / (1.0 + common::math::exp(-gc));
        if u.abs() <= d.limit {
            du[i] = dh[i] * silu(gc);
        }
        if g <= d.limit {
            dg[i] = dh[i] * uc * sg * (1.0 + gc * (1.0 - sg));
        }
    }
    // gate and up
    let mut dx = matvec_t(ff, h, &d.gate_up[..ff * h], &dg);
    for (v, w) in dx.iter_mut().zip(matvec_t(ff, h, &d.gate_up[ff * h..], &du)) {
        *v += w;
    }
    let tg = matvec_t(ff, r, a.gb, &dg);
    outer_add(&dg, &s.zg, &mut gr.gb);
    outer_add(&tg, &s.xh, &mut gr.ga);
    for (v, w) in dx.iter_mut().zip(matvec_t(r, h, a.ga, &tg)) {
        *v += w;
    }
    let tu = matvec_t(ff, r, a.ub, &du);
    outer_add(&du, &s.zu, &mut gr.ub);
    outer_add(&tu, &s.xh, &mut gr.ua);
    for (v, w) in dx.iter_mut().zip(matvec_t(r, h, a.ua, &tu)) {
        *v += w;
    }
    (dx, gr)
}

/// The MoE's router, dense: `w: [e][h]`, `bias: [e]`, picks `top_k`, weights scaled by `scale`.
pub struct Router<'a> {
    pub e: usize,
    pub top_k: usize,
    pub h: usize,
    pub w: &'a [f32],
    pub bias: &'a [f32],
    pub scale: f32,
}

/// → (picks ascending, their sigmoid scores, S) — `reference::Moe::route`'s arithmetic.
pub fn route(r: &Router, x: &[f32]) -> (Vec<usize>, Vec<f32>, f32) {
    let logits = matvec(r.e, r.h, r.w, x);
    let s: Vec<f32> = logits.iter().map(|&l| glm_5_3_flash::hc::sigmoid(l)).collect();
    let mut order: Vec<usize> = (0..r.e).collect();
    order.sort_by(|&a, &b| (s[b] + r.bias[b]).total_cmp(&(s[a] + r.bias[a])).then(a.cmp(&b)));
    let mut idx = order[..r.top_k].to_vec();
    idx.sort();
    let sum = idx.iter().map(|&e| s[e]).sum::<f32>() + 1e-20;
    (idx.clone(), idx.iter().map(|&e| s[e]).collect(), sum)
}

/// The MoE: a router, the shared expert (as a `Dense` with a rank-0 adapter), and the routed
/// experts that exist (`(e, dense, adapter)`; a pick without one is a bug in the test).
pub struct MoeRef<'a> {
    pub router: &'a Router<'a>,
    pub shared: &'a Dense<'a>,
    pub shared_lr: &'a LowRank<'a>,
    pub experts: &'a [(usize, Dense<'a>, LowRank<'a>)],
}

pub struct MoeSaved {
    pub idx: Vec<usize>,
    pub s: Vec<f32>,
    pub sum: f32,
    pub experts: Vec<(Vec<f32>, Saved)>, // each pick's raw output and saved state
    pub shared: Saved,
}

fn expert_of<'a, 'b>(m: &'b MoeRef<'a>, e: usize) -> (&'b Dense<'a>, &'b LowRank<'a>) {
    let (_, d, a) = m.experts.iter().find(|(ee, _, _)| *ee == e).expect("a picked expert exists");
    (d, a)
}

pub fn moe_forward(m: &MoeRef, x: &[f32], quant: bool) -> (Vec<f32>, MoeSaved) {
    let (idx, s, sum) = route(m.router, x);
    let h = m.router.h;
    let mut out = vec![0f32; h];
    let mut experts = Vec::new();
    for (k, &e) in idx.iter().enumerate() {
        let (d, a) = expert_of(m, e);
        let (eo, saved) = forward(d, a, x, quant);
        let wt = s[k] / sum * m.router.scale;
        for i in 0..h {
            out[i] += wt * eo[i];
        }
        experts.push((eo, saved));
    }
    let (so, shared) = forward(m.shared, m.shared_lr, x, false);
    for i in 0..h {
        out[i] += so[i];
    }
    (out, MoeSaved { idx, s, sum, experts, shared })
}

/// → (dx, the picked experts' adapter gradients by expert).
pub fn moe_backward(m: &MoeRef, sv: &MoeSaved, dout: &[f32]) -> (Vec<f32>, Vec<(usize, Grads)>) {
    let h = m.router.h;
    let (mut dx, _) = backward(m.shared, m.shared_lr, &sv.shared, dout);
    let mut grads = Vec::new();
    let mut dwt = vec![0f32; sv.idx.len()];
    for (k, &e) in sv.idx.iter().enumerate() {
        let (d, a) = expert_of(m, e);
        let (eo, saved) = &sv.experts[k];
        let wt = sv.s[k] / sv.sum * m.router.scale;
        dwt[k] = eo.iter().zip(dout).map(|(o, g)| o * g).sum();
        let scaled: Vec<f32> = dout.iter().map(|g| wt * g).collect();
        let (edx, g) = backward(d, a, saved, &scaled);
        for i in 0..h {
            dx[i] += edx[i];
        }
        grads.push((e, g));
    }
    // wt_j = c · s_j / S, S = ε + Σ s: d s_e = c/S · (d wt_e − Σ_j d wt_j s_j / S)
    let mixed: f32 = (0..sv.idx.len()).map(|k| dwt[k] * sv.s[k]).sum::<f32>() / sv.sum;
    for (k, &e) in sv.idx.iter().enumerate() {
        let ds = m.router.scale / sv.sum * (dwt[k] - mixed);
        let dl = ds * sv.s[k] * (1.0 - sv.s[k]);
        for i in 0..h {
            dx[i] += dl * m.router.w[e * h + i];
        }
    }
    (dx, grads)
}

/// The hyper-connection, dense, any shape: `hc` streams of `h`, `fn_: [(2 + hc)·hc][hc·h]`.
pub struct HcRef<'a> {
    pub hc: usize,
    pub h: usize,
    pub fn_: &'a [f32],
    pub base: &'a [f32],
    pub scale: [f32; 3],
    pub eps: f32,
    pub rms_eps: f32,
    pub iters: usize,
}

/// `pre`'s mix: `post: [hc]`, `comb: [hc][hc]` row-major, with the coefficients for the backward.
pub struct HcMix {
    pub pre: Vec<f32>,
    pub post: Vec<f32>,
    pub comb: Vec<f32>,
    pub coef: Vec<f32>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + common::math::exp(-x))
}

/// The Sinkhorn chain from the coefficients: the softmax rows and the matrix after each
/// normalization (`[0]` the softmax + ε, then columns, then (rows, columns) × (iters − 1)).
fn hc_sinkhorn(r: &HcRef, c: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
    let hc = r.hc;
    let mut a = vec![0f32; hc * hc];
    for i in 0..hc {
        let row: Vec<f32> = (0..hc).map(|j| c[2 * hc + i * hc + j] * r.scale[2] + r.base[2 * hc + i * hc + j]).collect();
        let mx = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        let e: Vec<f32> = row.iter().map(|&v| common::math::exp(v - mx)).collect();
        let s: f32 = e.iter().sum();
        for j in 0..hc {
            a[i * hc + j] = e[j] / s;
        }
    }
    let soft = a.clone();
    for v in a.iter_mut() {
        *v += r.eps;
    }
    let mut steps = vec![a.clone()];
    let cols = |a: &mut Vec<f32>| {
        for j in 0..hc {
            let s = (0..hc).map(|i| a[i * hc + j]).sum::<f32>() + r.eps;
            for i in 0..hc {
                a[i * hc + j] /= s;
            }
        }
    };
    let rows = |a: &mut Vec<f32>| {
        for i in 0..hc {
            let s = a[i * hc..(i + 1) * hc].iter().sum::<f32>() + r.eps;
            for v in a[i * hc..(i + 1) * hc].iter_mut() {
                *v /= s;
            }
        }
    };
    cols(&mut a);
    steps.push(a.clone());
    for _ in 0..r.iters - 1 {
        rows(&mut a);
        steps.push(a.clone());
        cols(&mut a);
        steps.push(a.clone());
    }
    (soft, steps)
}

pub fn hc_pre(r: &HcRef, s: &[f32]) -> (Vec<f32>, HcMix) {
    let (hc, h) = (r.hc, r.h);
    let n = hc * h;
    let ss: f32 = s[..n].iter().map(|v| v * v).sum();
    let rr = common::math::rsqrt(ss / n as f32 + r.rms_eps);
    let flat: Vec<f32> = s[..n].iter().map(|v| v * rr).collect();
    let nc = (2 + hc) * hc;
    let coef = matvec(nc, n, r.fn_, &flat);
    let pre: Vec<f32> = (0..hc).map(|i| sigmoid(coef[i] * r.scale[0] + r.base[i]) + r.eps).collect();
    let post: Vec<f32> = (0..hc).map(|i| 2.0 * sigmoid(coef[hc + i] * r.scale[1] + r.base[hc + i])).collect();
    let (_, steps) = hc_sinkhorn(r, &coef);
    let comb = steps.last().unwrap().clone();
    let x: Vec<f32> = (0..h).map(|d| (0..hc).map(|i| pre[i] * s[i * h + d]).sum()).collect();
    (x, HcMix { pre, post, comb, coef })
}

/// `S'_j = post_j · y + Σ_i comb_ij · S_i`.
pub fn hc_post(r: &HcRef, m: &HcMix, y: &[f32], s: &[f32]) -> Vec<f32> {
    let (hc, h) = (r.hc, r.h);
    let mut out = vec![0f32; hc * h];
    for j in 0..hc {
        for d in 0..h {
            out[j * h + d] = m.post[j] * y[d] + (0..hc).map(|i| m.comb[i * hc + j] * s[i * h + d]).sum::<f32>();
        }
    }
    out
}

/// → (d y, the residual's share of d S, (d post, d comb)).
pub fn hc_post_backward(r: &HcRef, m: &HcMix, y: &[f32], s: &[f32], ds_out: &[f32]) -> (Vec<f32>, Vec<f32>, (Vec<f32>, Vec<f32>)) {
    let (hc, h) = (r.hc, r.h);
    let mut dy = vec![0f32; h];
    let mut ds = vec![0f32; hc * h];
    let mut dpost = vec![0f32; hc];
    let mut dcomb = vec![0f32; hc * hc];
    for j in 0..hc {
        for d in 0..h {
            let g = ds_out[j * h + d];
            dy[d] += m.post[j] * g;
            dpost[j] += y[d] * g;
            for i in 0..hc {
                dcomb[i * hc + j] += s[i * h + d] * g;
                ds[i * h + d] += m.comb[i * hc + j] * g;
            }
        }
    }
    (dy, ds, (dpost, dcomb))
}

/// Given d x and the mix's gradient, `ds += …`.
pub fn hc_pre_backward(r: &HcRef, s: &[f32], m: &HcMix, dx: &[f32], dm: &(Vec<f32>, Vec<f32>), ds: &mut [f32]) {
    let (hc, h) = (r.hc, r.h);
    let n = hc * h;
    let nc = (2 + hc) * hc;
    let mut dc = vec![0f32; nc];
    for i in 0..hc {
        let dp: f32 = (0..h).map(|d| s[i * h + d] * dx[d]).sum();
        for d in 0..h {
            ds[i * h + d] += m.pre[i] * dx[d];
        }
        let sg = m.pre[i] - r.eps;
        dc[i] = dp * sg * (1.0 - sg) * r.scale[0];
        let sp = m.post[i] / 2.0;
        dc[hc + i] = dm.0[i] * 2.0 * sp * (1.0 - sp) * r.scale[1];
    }
    let (soft, steps) = hc_sinkhorn(r, &m.coef);
    let mut g = dm.1.clone();
    for k in (1..steps.len()).rev() {
        let (before, after) = (&steps[k - 1], &steps[k]);
        let mut gin = vec![0f32; hc * hc];
        let line = |idx: &dyn Fn(usize) -> usize, g: &[f32], gin: &mut [f32]| {
            let sum = (0..hc).map(|t| before[idx(t)]).sum::<f32>() + r.eps;
            let dot: f32 = (0..hc).map(|t| g[idx(t)] * after[idx(t)]).sum();
            for t in 0..hc {
                gin[idx(t)] = (g[idx(t)] - dot) / sum;
            }
        };
        if k % 2 == 1 {
            for j in 0..hc {
                line(&|t| t * hc + j, &g, &mut gin);
            }
        } else {
            for i in 0..hc {
                line(&|t| i * hc + t, &g, &mut gin);
            }
        }
        g = gin;
    }
    for i in 0..hc {
        let dot: f32 = (0..hc).map(|j| g[i * hc + j] * soft[i * hc + j]).sum();
        for j in 0..hc {
            dc[2 * hc + i * hc + j] = soft[i * hc + j] * (g[i * hc + j] - dot) * r.scale[2];
        }
    }
    let dflat = matvec_t(nc, n, r.fn_, &dc);
    let ss: f32 = s[..n].iter().map(|v| v * v).sum();
    let rr = common::math::rsqrt(ss / n as f32 + r.rms_eps);
    let dot: f32 = (0..n).map(|i| dflat[i] * s[i]).sum();
    let c = rr * rr * rr / n as f32 * dot;
    for i in 0..n {
        ds[i] += rr * dflat[i] - s[i] * c;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::testing::Rng;

    /// The hyper-connection's analytic gradient against central differences at a small shape,
    /// the block between pre and post a fixed random linear map so both halves are exercised
    /// through one loss.
    #[test]
    fn hc_backward_is_the_gradient() {
        let (hc, h) = (3usize, 6usize);
        let mut rnd = Rng::new(0xF1D0_0003);
        let nc = (2 + hc) * hc;
        let fn_: Vec<f32> = (0..nc * hc * h).map(|_| rnd.unit() * 0.5).collect();
        let base: Vec<f32> = (0..nc).map(|_| rnd.unit()).collect();
        let r = HcRef { hc, h, fn_: &fn_, base: &base, scale: [0.8, 1.2, 0.9], eps: 1e-6, rms_eps: 1e-5, iters: 4 };
        let block: Vec<f32> = (0..h * h).map(|_| rnd.unit()).collect();
        let mut s: Vec<f32> = (0..hc * h).map(|_| rnd.unit()).collect();
        let ds_out: Vec<f32> = (0..hc * h).map(|_| rnd.unit()).collect();
        let loss = |s: &[f32]| -> f64 {
            let (x, m) = hc_pre(&r, s);
            let y = matvec(h, h, &block, &x);
            hc_post(&r, &m, &y, s).iter().zip(&ds_out).map(|(a, b)| (*a as f64) * (*b as f64)).sum()
        };
        let (x, m) = hc_pre(&r, &s);
        let y = matvec(h, h, &block, &x);
        let (dy, mut ds, dm) = hc_post_backward(&r, &m, &y, &s, &ds_out);
        let dx = matvec_t(h, h, &block, &dy);
        hc_pre_backward(&r, &s, &m, &dx, &dm, &mut ds);
        let eps = 1e-3f32;
        for i in 0..hc * h {
            let base_v = s[i];
            s[i] = base_v + eps;
            let plus = loss(&s);
            s[i] = base_v - eps;
            let minus = loss(&s);
            s[i] = base_v;
            let num = ((plus - minus) / (2.0 * eps as f64)) as f32;
            assert!((num - ds[i]).abs() <= 3e-3 * (1.0 + num.abs()), "dS[{i}]: numeric {num} analytic {}", ds[i]);
        }
    }

    /// The MoE's analytic gradient against central differences at a small shape, quantizers
    /// off: the routing weights' path through the renormalization and the sigmoid into the
    /// router's rows, and the shared expert, on top of the experts' own (checked below).
    #[test]
    fn moe_backward_is_the_gradient() {
        let (e, top_k, h, ff, sff, r) = (6usize, 2usize, 12usize, 10usize, 8usize, 2usize);
        let mut rnd = Rng::new(0xF1D0_0002);
        let rw: Vec<f32> = (0..e * h).map(|_| rnd.unit()).collect();
        let bias: Vec<f32> = (0..e).map(|_| rnd.unit() * 0.1).collect();
        let router = Router { e, top_k, h, w: &rw, bias: &bias, scale: 2.5 };
        let sgu: Vec<f32> = (0..2 * sff * h).map(|_| rnd.unit()).collect();
        let sd: Vec<f32> = (0..h * sff).map(|_| rnd.unit()).collect();
        let shared = Dense { h, ff: sff, gate_up: &sgu, down: &sd, limit: 10.0 };
        let none = LowRank { r: 0, ga: &[], gb: &[], ua: &[], ub: &[], da: &[], db: &[] };
        let dense: Vec<(Vec<f32>, Vec<f32>)> = (0..e).map(|_| ((0..2 * ff * h).map(|_| rnd.unit()).collect(), (0..h * ff).map(|_| rnd.unit()).collect())).collect();
        let mut ads: Vec<Grads> = (0..e).map(|_| Grads { ga: (0..r * h).map(|_| rnd.unit()).collect(), gb: (0..ff * r).map(|_| rnd.unit()).collect(), ua: (0..r * h).map(|_| rnd.unit()).collect(), ub: (0..ff * r).map(|_| rnd.unit()).collect(), da: (0..r * ff).map(|_| rnd.unit()).collect(), db: (0..h * r).map(|_| rnd.unit()).collect() }).collect();
        let mut x: Vec<f32> = (0..h).map(|_| rnd.unit()).collect();
        let dout: Vec<f32> = (0..h).map(|_| rnd.unit()).collect();

        let build = |ads: &[Grads], f: &mut dyn FnMut(&MoeRef) -> f64| -> f64 {
            let experts: Vec<(usize, Dense, LowRank)> = (0..e).map(|i| (i, Dense { h, ff, gate_up: &dense[i].0, down: &dense[i].1, limit: 10.0 }, LowRank { r, ga: &ads[i].ga, gb: &ads[i].gb, ua: &ads[i].ua, ub: &ads[i].ub, da: &ads[i].da, db: &ads[i].db })).collect();
            f(&MoeRef { router: &router, shared: &shared, shared_lr: &none, experts: &experts })
        };
        let loss = |ads: &[Grads], x: &[f32]| build(ads, &mut |m| moe_forward(m, x, false).0.iter().zip(&dout).map(|(o, g)| (*o as f64) * (*g as f64)).sum());
        let (idx0, _, _) = route(&router, &x);
        let (dx, grads) = {
            let mut res = None;
            build(&ads, &mut |m| { let (_, s) = moe_forward(m, &x, false); res = Some(moe_backward(m, &s, &dout)); 0.0 });
            res.unwrap()
        };

        let eps = 1e-3f32;
        let check = |analytic: f32, numeric: f32, what: &str| {
            assert!((numeric - analytic).abs() <= 3e-3 * (1.0 + analytic.abs()), "{what}: numeric {numeric} analytic {analytic}");
        };
        for i in 0..h {
            let base = x[i];
            x[i] = base + eps;
            let (idx1, _, _) = route(&router, &x);
            let plus = loss(&ads, &x);
            x[i] = base - eps;
            let (idx2, _, _) = route(&router, &x);
            let minus = loss(&ads, &x);
            x[i] = base;
            assert!(idx1 == idx0 && idx2 == idx0, "the picks must not flip under the step");
            check(dx[i], ((plus - minus) / (2.0 * eps as f64)) as f32, &format!("dx[{i}]"));
        }
        for (e_, g) in &grads {
            macro_rules! params {
                ($($f:ident),*) => {$(
                    for i in (0..g.$f.len()).step_by(3) {
                        let base = ads[*e_].$f[i];
                        ads[*e_].$f[i] = base + eps;
                        let plus = loss(&ads, &x);
                        ads[*e_].$f[i] = base - eps;
                        let minus = loss(&ads, &x);
                        ads[*e_].$f[i] = base;
                        check(g.$f[i], ((plus - minus) / (2.0 * eps as f64)) as f32, &format!("expert {e_} {}[{i}]", stringify!($f)));
                    }
                )*};
            }
            params!(ga, gb, ua, ub, da, db);
        }
    }

    /// The analytic gradient against central differences, quantizers off, at a small shape:
    /// the loss is `⟨dout, out⟩`, so d loss / d p = the backward's gradient for p.
    #[test]
    fn backward_is_the_gradient() {
        let (h, ff, r) = (24usize, 20usize, 3usize);
        let mut rnd = Rng::new(0xF1D0_0001);
        let gate_up: Vec<f32> = (0..2 * ff * h).map(|_| rnd.unit()).collect();
        let down: Vec<f32> = (0..h * ff).map(|_| rnd.unit()).collect();
        let mut p = Grads { ga: (0..r * h).map(|_| rnd.unit()).collect(), gb: (0..ff * r).map(|_| rnd.unit()).collect(), ua: (0..r * h).map(|_| rnd.unit()).collect(), ub: (0..ff * r).map(|_| rnd.unit()).collect(), da: (0..r * ff).map(|_| rnd.unit()).collect(), db: (0..h * r).map(|_| rnd.unit()).collect() };
        let mut x: Vec<f32> = (0..h).map(|_| rnd.unit() * 3.0).collect();
        let dout: Vec<f32> = (0..h).map(|_| rnd.unit()).collect();
        // a limit the activations actually reach, so the clamps' zero branches are exercised
        let limit = 2.0;
        let d = Dense { h, ff, gate_up: &gate_up, down: &down, limit };

        let loss = |p: &Grads, x: &[f32]| -> f64 {
            let a = LowRank { r, ga: &p.ga, gb: &p.gb, ua: &p.ua, ub: &p.ub, da: &p.da, db: &p.db };
            let (out, _) = forward(&d, &a, x, false);
            out.iter().zip(&dout).map(|(o, g)| (*o as f64) * (*g as f64)).sum()
        };
        let (dx, gr) = {
            let a = LowRank { r, ga: &p.ga, gb: &p.gb, ua: &p.ua, ub: &p.ub, da: &p.da, db: &p.db };
            let (_, s) = forward(&d, &a, &x, false);
            backward(&d, &a, &s, &dout)
        };

        let eps = 1e-3f32;
        let mut checked = 0;
        let mut check = |analytic: f32, get: &mut dyn FnMut(f32) -> f64, what: &str| {
            let plus = get(eps);
            let minus = get(-eps);
            let numeric = ((plus - minus) / (2.0 * eps as f64)) as f32;
            assert!((numeric - analytic).abs() <= 2e-3 * (1.0 + analytic.abs()), "{what}: numeric {numeric} analytic {analytic}");
            checked += 1;
        };
        for i in (0..h).step_by(5) {
            let base = x[i];
            check(dx[i], &mut |e| { x[i] = base + e; let v = loss(&p, &x); x[i] = base; v }, &format!("dx[{i}]"));
        }
        macro_rules! params {
            ($($f:ident),*) => {$(
                let n = p.$f.len();
                for i in (0..n).step_by((n / 7).max(1)) {
                    let base = p.$f[i];
                    check(gr.$f[i], &mut |e| { p.$f[i] = base + e; let v = loss(&p, &x); p.$f[i] = base; v }, &format!("{}[{i}]", stringify!($f)));
                }
            )*};
        }
        params!(ga, gb, ua, ub, da, db);
        assert!(checked > 40);
    }
}
