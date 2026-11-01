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

// ---------------------------------------------------------------------------------------------
// KDA (Kimi delta attention), dense, any shape, over a sequence — `reference::Kda::forward`'s
// arithmetic with every intermediate kept, and its backward.

/// Shapes are `[out][in]`: `nh` heads of `d`; `q, k, v: [nh·d][h]`; `conv: [3·nh·d][taps]`;
/// `f_a, g_a: [d][h]`; `f_b, g_b: [nh·d][d]`; `b: [nh][h]`; `o: [h][nh·d]`.
pub struct KdaRef<'a> {
    pub nh: usize,
    pub d: usize,
    pub h: usize,
    pub taps: usize,
    pub lower: f32,
    pub rms_eps: f32,
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub conv: &'a [f32],
    pub f_a: &'a [f32],
    pub f_b: &'a [f32],
    pub dt_bias: &'a [f32],
    pub a_log: &'a [f32],
    pub b: &'a [f32],
    pub g_a: &'a [f32],
    pub g_b: &'a [f32],
    pub o_norm: &'a [f32],
    pub o: &'a [f32],
}

/// The layer's state: the convolution's last `taps − 1` inputs (`[taps−1][3·nh·d]`, oldest
/// first) and the memory `[nh][d][d]`.
#[derive(Clone)]
pub struct KdaState {
    pub window: Vec<f32>,
    pub mem: Vec<f32>,
}

/// Everything the forward computed, per position and per head.
pub struct KdaSaved {
    pub n: usize,
    pub mixed: Vec<f32>,  // [n][3·nh·d]
    pub pre: Vec<f32>,    // [n][3·nh·d]: the convolution before SiLU
    pub conv: Vec<f32>,   // [n][3·nh·d]
    pub fa: Vec<f32>,     // [n][d]
    pub f: Vec<f32>,      // [n][nh·d]
    pub beta: Vec<f32>,   // [n][nh]
    pub ga: Vec<f32>,     // [n][d]
    pub gate: Vec<f32>,   // [n][nh·d]
    pub qn: Vec<f32>,     // [n][nh·d]: l2-normed q (unscaled)
    pub kn: Vec<f32>,     // [n][nh·d]
    pub e: Vec<f32>,      // [n][nh·d]: the forget gate
    pub s_prev: Vec<f32>, // [n+1][nh][d][d]: the memory before each position, and after the last
    pub mem_read: Vec<f32>, // [n][nh·d]
    pub delta: Vec<f32>,  // [n][nh·d]
    pub o: Vec<f32>,      // [n][nh·d]
    pub on: Vec<f32>,     // [n][nh·d]
    pub normed: Vec<f32>, // [n][nh·d]
    pub window_in: Vec<f32>,
}

fn l2norm_ref(x: &[f32]) -> (Vec<f32>, f32) {
    let r = 1.0 / common::math::sqrt(x.iter().map(|v| v * v).sum::<f32>() + 1e-6);
    (x.iter().map(|v| v * r).collect(), r)
}
/// y = x · r, r = (Σx² + 1e-6)^-½: dx = r·dy − x·r³·⟨dy, x⟩.
fn l2norm_backward_ref(x: &[f32], dy: &[f32]) -> Vec<f32> {
    let r = 1.0 / common::math::sqrt(x.iter().map(|v| v * v).sum::<f32>() + 1e-6);
    let dot: f32 = dy.iter().zip(x).map(|(a, b)| a * b).sum();
    x.iter().zip(dy).map(|(xi, dyi)| r * dyi - xi * r * r * r * dot).collect()
}
fn rms_ref(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let r = common::math::rsqrt(x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps);
    (0..n).map(|i| w[i] * (x[i] * r)).collect()
}
fn rms_backward_ref(x: &[f32], w: &[f32], eps: f32, dy: &[f32]) -> Vec<f32> {
    let n = x.len();
    let r = common::math::rsqrt(x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps);
    let dot: f32 = (0..n).map(|i| w[i] * dy[i] * x[i]).sum();
    let c = r * r * r / n as f32 * dot;
    (0..n).map(|i| r * w[i] * dy[i] - x[i] * c).collect()
}

/// `x: [n][h]` → (`y: [n][h]`, everything kept, the state after).
pub fn kda_forward(r: &KdaRef, x: &[f32], n: usize, st: &KdaState) -> (Vec<f32>, KdaSaved, KdaState) {
    let (nh, d, h, taps) = (r.nh, r.d, r.h, r.taps);
    let (qkv, cd) = (nh * d, 3 * nh * d);
    let scale = 1.0 / common::math::sqrt(d as f32);
    let mut sv = KdaSaved { n, mixed: vec![0.0; n * cd], pre: vec![0.0; n * cd], conv: vec![0.0; n * cd], fa: vec![0.0; n * d], f: vec![0.0; n * qkv], beta: vec![0.0; n * nh], ga: vec![0.0; n * d], gate: vec![0.0; n * qkv], qn: vec![0.0; n * qkv], kn: vec![0.0; n * qkv], e: vec![0.0; n * qkv], s_prev: vec![0.0; (n + 1) * nh * d * d], mem_read: vec![0.0; n * qkv], delta: vec![0.0; n * qkv], o: vec![0.0; n * qkv], on: vec![0.0; n * qkv], normed: vec![0.0; n * qkv], window_in: st.window.clone() };
    // the mixed timeline: the entering window, then each position's projections
    let mut timeline = st.window.clone();
    let mut mem = st.mem.clone();
    sv.s_prev[..nh * d * d].copy_from_slice(&mem);
    let mut y = vec![0f32; n * h];
    for t in 0..n {
        let xt = &x[t * h..(t + 1) * h];
        let (q, k, v) = (matvec(qkv, h, r.q, xt), matvec(qkv, h, r.k, xt), matvec(qkv, h, r.v, xt));
        let mixed = [q, k, v].concat();
        sv.mixed[t * cd..(t + 1) * cd].copy_from_slice(&mixed);
        timeline.extend_from_slice(&mixed);
        let base = timeline.len() - taps * cd; // tap j reads timeline[(base + j·cd) + c]
        for c in 0..cd {
            let mut acc = 0f32;
            for j in 0..taps {
                acc += r.conv[c * taps + j] * timeline[base + j * cd + c];
            }
            sv.pre[t * cd + c] = acc;
            sv.conv[t * cd + c] = silu(acc);
        }
        let fa = matvec(d, h, r.f_a, xt);
        let f = matvec(qkv, d, r.f_b, &fa);
        let beta = matvec(nh, h, r.b, xt);
        let ga = matvec(d, h, r.g_a, xt);
        let gate = matvec(qkv, d, r.g_b, &ga);
        sv.fa[t * d..(t + 1) * d].copy_from_slice(&fa);
        sv.f[t * qkv..(t + 1) * qkv].copy_from_slice(&f);
        sv.beta[t * nh..(t + 1) * nh].copy_from_slice(&beta);
        sv.ga[t * d..(t + 1) * d].copy_from_slice(&ga);
        sv.gate[t * qkv..(t + 1) * qkv].copy_from_slice(&gate);
        let conv = &sv.conv[t * cd..(t + 1) * cd];
        for hh in 0..nh {
            let (qn, _) = l2norm_ref(&conv[hh * d..(hh + 1) * d]);
            let (kn, _) = l2norm_ref(&conv[qkv + hh * d..qkv + (hh + 1) * d]);
            let vh = &conv[2 * qkv + hh * d..2 * qkv + (hh + 1) * d];
            sv.qn[t * qkv + hh * d..t * qkv + (hh + 1) * d].copy_from_slice(&qn);
            sv.kn[t * qkv + hh * d..t * qkv + (hh + 1) * d].copy_from_slice(&kn);
            let decay = common::math::exp(r.a_log[hh]);
            let bt = glm_5_3_flash::hc::sigmoid(beta[hh]);
            let s = &mut mem[hh * d * d..(hh + 1) * d * d];
            for kk in 0..d {
                let e = common::math::exp(r.lower * glm_5_3_flash::hc::sigmoid(decay * (f[hh * d + kk] + r.dt_bias[hh * d + kk])));
                sv.e[t * qkv + hh * d + kk] = e;
                for vv in 0..d {
                    s[kk * d + vv] *= e;
                }
            }
            let mut delta = vec![0f32; d];
            for vv in 0..d {
                let m: f32 = (0..d).map(|kk| s[kk * d + vv] * kn[kk]).sum();
                sv.mem_read[t * qkv + hh * d + vv] = m;
                delta[vv] = (vh[vv] - m) * bt;
            }
            for kk in 0..d {
                for vv in 0..d {
                    s[kk * d + vv] += kn[kk] * delta[vv];
                }
            }
            sv.delta[t * qkv + hh * d..t * qkv + (hh + 1) * d].copy_from_slice(&delta);
            let o: Vec<f32> = (0..d).map(|vv| (0..d).map(|kk| s[kk * d + vv] * qn[kk] * scale).sum()).collect();
            let on = rms_ref(&o, r.o_norm, r.rms_eps);
            for vv in 0..d {
                sv.normed[t * qkv + hh * d + vv] = on[vv] * glm_5_3_flash::hc::sigmoid(gate[hh * d + vv]);
            }
            sv.o[t * qkv + hh * d..t * qkv + (hh + 1) * d].copy_from_slice(&o);
            sv.on[t * qkv + hh * d..t * qkv + (hh + 1) * d].copy_from_slice(&on);
        }
        sv.s_prev[(t + 1) * nh * d * d..(t + 2) * nh * d * d].copy_from_slice(&mem);
        let yt = matvec(h, qkv, r.o, &sv.normed[t * qkv..(t + 1) * qkv]);
        y[t * h..(t + 1) * h].copy_from_slice(&yt);
    }
    let window = timeline[timeline.len() - (taps - 1) * cd..].to_vec();
    (y, sv, KdaState { window, mem })
}

/// Given `dy: [n][h]` and the gradient at the state after (`d window`, `d mem`), → (`dx: [n][h]`,
/// the gradient at the state before).
pub fn kda_backward(r: &KdaRef, x: &[f32], sv: &KdaSaved, dy: &[f32], dst_out: &KdaState) -> (Vec<f32>, KdaState) {
    let (nh, d, h, taps, n) = (r.nh, r.d, r.h, r.taps, sv.n);
    let (qkv, cd) = (nh * d, 3 * nh * d);
    let scale = 1.0 / common::math::sqrt(d as f32);
    let mut dx = vec![0f32; n * h];
    let mut dconv = vec![0f32; n * cd];
    let mut df = vec![0f32; n * qkv];
    let mut dbeta = vec![0f32; n * nh];
    let mut dgate = vec![0f32; n * qkv];
    let mut dnormed = vec![0f32; n * qkv];
    for t in 0..n {
        dnormed[t * qkv..(t + 1) * qkv].copy_from_slice(&matvec_t(h, qkv, r.o, &dy[t * h..(t + 1) * h]));
    }
    let mut dmem_in = vec![0f32; nh * d * d];
    for hh in 0..nh {
        let decay = common::math::exp(r.a_log[hh]);
        let mut ds = dst_out.mem[hh * d * d..(hh + 1) * d * d].to_vec();
        for t in (0..n).rev() {
            fn at<'a>(v: &'a [f32], t: usize, qkv: usize, hh: usize, d: usize) -> &'a [f32] {
                &v[t * qkv + hh * d..t * qkv + (hh + 1) * d]
            }
            let (qn, kn, e, mr, delta, o, on) = (at(&sv.qn, t, qkv, hh, d), at(&sv.kn, t, qkv, hh, d), at(&sv.e, t, qkv, hh, d), at(&sv.mem_read, t, qkv, hh, d), at(&sv.delta, t, qkv, hh, d), at(&sv.o, t, qkv, hh, d), at(&sv.on, t, qkv, hh, d));
            let gate = at(&sv.gate, t, qkv, hh, d);
            let conv = &sv.conv[t * cd..(t + 1) * cd];
            let vh = &conv[2 * qkv + hh * d..2 * qkv + (hh + 1) * d];
            let s_prev = &sv.s_prev[t * nh * d * d + hh * d * d..t * nh * d * d + (hh + 1) * d * d];
            let s2 = &sv.s_prev[(t + 1) * nh * d * d + hh * d * d..(t + 1) * nh * d * d + (hh + 1) * d * d];
            let bt = glm_5_3_flash::hc::sigmoid(sv.beta[t * nh + hh]);
            // the gated norm
            let dout = at(&dnormed, t, qkv, hh, d);
            let mut don = vec![0f32; d];
            for vv in 0..d {
                let sg = glm_5_3_flash::hc::sigmoid(gate[vv]);
                don[vv] = dout[vv] * sg;
                dgate[t * qkv + hh * d + vv] = dout[vv] * on[vv] * sg * (1.0 - sg);
            }
            let do_ = rms_backward_ref(o, r.o_norm, r.rms_eps, &don);
            // o = S2ᵀ (qn · scale): dS2 += (qn·scale) ⊗ do; d qn = scale · S2 · do
            let mut dqn = vec![0f32; d];
            for kk in 0..d {
                let mut acc = 0f32;
                for vv in 0..d {
                    ds[kk * d + vv] += qn[kk] * scale * do_[vv];
                    acc += s2[kk * d + vv] * do_[vv];
                }
                dqn[kk] = acc * scale;
            }
            // S2 = S1 + kn ⊗ delta: d kn += dS · delta; d delta = dSᵀ kn
            let mut dkn = vec![0f32; d];
            let mut ddelta = vec![0f32; d];
            for kk in 0..d {
                for vv in 0..d {
                    dkn[kk] += ds[kk * d + vv] * delta[vv];
                    ddelta[vv] += ds[kk * d + vv] * kn[kk];
                }
            }
            // delta = (v − mem) · b
            let mut dv = vec![0f32; d];
            let mut dmem = vec![0f32; d];
            let mut db = 0f32;
            for vv in 0..d {
                dv[vv] = ddelta[vv] * bt;
                dmem[vv] = -ddelta[vv] * bt;
                db += ddelta[vv] * (vh[vv] - mr[vv]);
            }
            dbeta[t * nh + hh] = db * bt * (1.0 - bt);
            // mem = S1ᵀ kn, S1 = diag(e) S_prev: dS1 += kn ⊗ dmem; d kn += S1 · dmem
            for kk in 0..d {
                let mut acc = 0f32;
                for vv in 0..d {
                    ds[kk * d + vv] += kn[kk] * dmem[vv];
                    acc += e[kk] * s_prev[kk * d + vv] * dmem[vv];
                }
                dkn[kk] += acc;
            }
            // S1 = diag(e) S_prev: d e_kk = Σ_vv dS1[kk][vv] S_prev[kk][vv]; dS_prev = diag(e) dS1
            for kk in 0..d {
                let mut de = 0f32;
                for vv in 0..d {
                    de += ds[kk * d + vv] * s_prev[kk * d + vv];
                    ds[kk * d + vv] *= e[kk];
                }
                // e = exp(lower · σ(z)), z = decay · (f + dt_bias)
                let z = decay * (sv.f[t * qkv + hh * d + kk] + r.dt_bias[hh * d + kk]);
                let sg = glm_5_3_flash::hc::sigmoid(z);
                df[t * qkv + hh * d + kk] = de * e[kk] * r.lower * sg * (1.0 - sg) * decay;
            }
            // the l2 norms, back to the convolution's output
            let dq = l2norm_backward_ref(&conv[hh * d..(hh + 1) * d], &dqn);
            let dk = l2norm_backward_ref(&conv[qkv + hh * d..qkv + (hh + 1) * d], &dkn);
            dconv[t * cd + hh * d..t * cd + (hh + 1) * d].copy_from_slice(&dq);
            dconv[t * cd + qkv + hh * d..t * cd + qkv + (hh + 1) * d].copy_from_slice(&dk);
            dconv[t * cd + 2 * qkv + hh * d..t * cd + 2 * qkv + (hh + 1) * d].copy_from_slice(&dv);
        }
        dmem_in[hh * d * d..(hh + 1) * d * d].copy_from_slice(&ds);
    }
    // the convolution: a timeline of d mixed, the entering window first
    let mut dtl = vec![0f32; (taps - 1 + n) * cd];
    let out0 = (n + taps - 1 - (taps - 1)) * cd; // the out window is the timeline's last taps−1 entries
    dtl[out0..].copy_from_slice(&dst_out.window);
    for t in 0..n {
        for c in 0..cd {
            let da = dconv[t * cd + c] * train_common::swiglu::dsilu(sv.pre[t * cd + c]);
            for j in 0..taps {
                dtl[(t + j) * cd + c] += da * r.conv[c * taps + j];
            }
        }
    }
    let dwindow_in = dtl[..(taps - 1) * cd].to_vec();
    for t in 0..n {
        let dm = &dtl[(taps - 1 + t) * cd..(taps + t) * cd];
        let mut dxt = matvec_t(qkv, h, r.q, &dm[..qkv]);
        for (a, b) in dxt.iter_mut().zip(matvec_t(qkv, h, r.k, &dm[qkv..2 * qkv])) {
            *a += b;
        }
        for (a, b) in dxt.iter_mut().zip(matvec_t(qkv, h, r.v, &dm[2 * qkv..])) {
            *a += b;
        }
        let dfa = matvec_t(qkv, d, r.f_b, &df[t * qkv..(t + 1) * qkv]);
        for (a, b) in dxt.iter_mut().zip(matvec_t(d, h, r.f_a, &dfa)) {
            *a += b;
        }
        for (a, b) in dxt.iter_mut().zip(matvec_t(nh, h, r.b, &dbeta[t * nh..(t + 1) * nh])) {
            *a += b;
        }
        let dga = matvec_t(qkv, d, r.g_b, &dgate[t * qkv..(t + 1) * qkv]);
        for (a, b) in dxt.iter_mut().zip(matvec_t(d, h, r.g_a, &dga)) {
            *a += b;
        }
        dx[t * h..(t + 1) * h].copy_from_slice(&dxt);
    }
    let _ = x;
    (dx, KdaState { window: dwindow_in, mem: dmem_in })
}

#[cfg(test)]
mod kda_tests {
    use super::*;
    use common::testing::Rng;

    /// The KDA backward against central differences at a small shape over three positions, with a
    /// loss on the outputs and on the state after (so the state's gradient is exercised too).
    #[test]
    fn kda_backward_is_the_gradient() {
        let (nh, d, h, taps, n) = (2usize, 4usize, 6usize, 4usize, 3usize);
        let (qkv, cd) = (nh * d, 3 * nh * d);
        let mut rnd = Rng::new(0xF1D0_0004);
        let mut u = |m: usize, s: f32| -> Vec<f32> { (0..m).map(|_| rnd.unit() * s).collect() };
        let (q, k, v) = (u(qkv * h, 0.6), u(qkv * h, 0.6), u(qkv * h, 0.6));
        let conv = u(cd * taps, 0.5);
        let (f_a, f_b, dt_bias, a_log) = (u(d * h, 0.5), u(qkv * d, 0.5), u(qkv, 0.5), u(nh, 0.3));
        let (b, g_a, g_b) = (u(nh * h, 0.5), u(d * h, 0.5), u(qkv * d, 0.5));
        let o_norm: Vec<f32> = u(d, 0.5).iter().map(|v| 1.0 + v).collect();
        let o = u(h * qkv, 0.5);
        let r = KdaRef { nh, d, h, taps, lower: -5.0, rms_eps: 1e-5, q: &q, k: &k, v: &v, conv: &conv, f_a: &f_a, f_b: &f_b, dt_bias: &dt_bias, a_log: &a_log, b: &b, g_a: &g_a, g_b: &g_b, o_norm: &o_norm, o: &o };
        let st = KdaState { window: u((taps - 1) * cd, 0.5), mem: u(nh * d * d, 0.3) };
        let mut x = u(n * h, 1.0);
        let dy = u(n * h, 1.0);
        let dst = KdaState { window: u((taps - 1) * cd, 1.0), mem: u(nh * d * d, 1.0) };
        let loss = |x: &[f32], st: &KdaState| -> f64 {
            let (y, _, out) = kda_forward(&r, x, n, st);
            let a: f64 = y.iter().zip(&dy).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
            let bw: f64 = out.window.iter().zip(&dst.window).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
            let bm: f64 = out.mem.iter().zip(&dst.mem).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
            a + bw + bm
        };
        let (_, sv, _) = kda_forward(&r, &x, n, &st);
        let (dx, dst_in) = kda_backward(&r, &x, &sv, &dy, &dst);
        let eps = 1e-3f32;
        let check = |analytic: f32, numeric: f32, what: &str| {
            assert!((numeric - analytic).abs() <= 3e-3 * (1.0 + analytic.abs()), "{what}: numeric {numeric} analytic {analytic}");
        };
        for i in 0..n * h {
            let base = x[i];
            x[i] = base + eps;
            let p = loss(&x, &st);
            x[i] = base - eps;
            let m = loss(&x, &st);
            x[i] = base;
            check(dx[i], ((p - m) / (2.0 * eps as f64)) as f32, &format!("dx[{i}]"));
        }
        let mut st2 = st.clone();
        for i in 0..st.mem.len() {
            let base = st2.mem[i];
            st2.mem[i] = base + eps;
            let p = loss(&x, &st2);
            st2.mem[i] = base - eps;
            let m = loss(&x, &st2);
            st2.mem[i] = base;
            check(dst_in.mem[i], ((p - m) / (2.0 * eps as f64)) as f32, &format!("dmem[{i}]"));
        }
        for i in 0..st.window.len() {
            let base = st2.window[i];
            st2.window[i] = base + eps;
            let p = loss(&x, &st2);
            st2.window[i] = base - eps;
            let m = loss(&x, &st2);
            st2.window[i] = base;
            check(dst_in.window[i], ((p - m) / (2.0 * eps as f64)) as f32, &format!("dwindow[{i}]"));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// MLA (NoPE, dense causal attention — the selection under DENSE_BELOW positions), any shape,
// over a sequence, in the reference's expanded form (keys and values from the latent per head).

/// `nh` heads; query/key width `dq`, value width `dv`; latent rank `r`, query rank `qr`.
/// `q_a: [qr][h]`, `q_b: [nh·dq][qr]`, `kv_a: [r][h]`, `kv_b: [nh][dq + dv][r]`, `o: [h][nh·dv]`.
pub struct MlaRef<'a> {
    pub nh: usize,
    pub dq: usize,
    pub dv: usize,
    pub r: usize,
    pub qr: usize,
    pub h: usize,
    pub rms_eps: f32,
    pub q_a: &'a [f32],
    pub q_a_ln: &'a [f32],
    pub q_b: &'a [f32],
    pub kv_a: &'a [f32],
    pub kv_a_ln: &'a [f32],
    pub kv_b: &'a [f32],
    pub o: &'a [f32],
    /// Store the latent at bf16 precision, as the node's cache does.
    pub round: bool,
}

pub struct MlaSaved {
    pub n: usize,
    pub qa: Vec<f32>,     // [n][qr]
    pub q: Vec<f32>,      // [n][nh·dq]
    pub kva: Vec<f32>,    // [n][r]
    pub latent: Vec<f32>, // [n][r]
    pub att: Vec<f32>,    // [n][nh][n]: query t, head h, key j ≤ t
    pub heads: Vec<f32>,  // [n][nh·dv]
}

pub fn mla_forward(m: &MlaRef, x: &[f32], n: usize) -> (Vec<f32>, MlaSaved) {
    let (nh, dq, dv, r, qr, h) = (m.nh, m.dq, m.dv, m.r, m.qr, m.h);
    let scale = 1.0 / common::math::sqrt(dq as f32);
    let mut sv = MlaSaved { n, qa: vec![0.0; n * qr], q: vec![0.0; n * nh * dq], kva: vec![0.0; n * r], latent: vec![0.0; n * r], att: vec![0.0; n * nh * n], heads: vec![0.0; n * nh * dv] };
    for t in 0..n {
        let xt = &x[t * h..(t + 1) * h];
        let qa = matvec(qr, h, m.q_a, xt);
        let q = matvec(nh * dq, qr, m.q_b, &rms_ref(&qa, m.q_a_ln, m.rms_eps));
        let kva = matvec(r, h, m.kv_a, xt);
        let mut lat = rms_ref(&kva, m.kv_a_ln, m.rms_eps);
        if m.round {
            for v in lat.iter_mut() {
                *v = common::math::bf16_round(*v);
            }
        }
        sv.qa[t * qr..(t + 1) * qr].copy_from_slice(&qa);
        sv.q[t * nh * dq..(t + 1) * nh * dq].copy_from_slice(&q);
        sv.kva[t * r..(t + 1) * r].copy_from_slice(&kva);
        sv.latent[t * r..(t + 1) * r].copy_from_slice(&lat);
    }
    let mut y = vec![0f32; n * h];
    for t in 0..n {
        for hh in 0..nh {
            let qh = &sv.q[t * nh * dq + hh * dq..t * nh * dq + (hh + 1) * dq];
            let w_uk = &m.kv_b[hh * (dq + dv) * r..(hh * (dq + dv) + dq) * r];
            let w_uv = &m.kv_b[(hh * (dq + dv) + dq) * r..(hh + 1) * (dq + dv) * r];
            let mut sc = vec![0f32; t + 1];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..=t {
                let k = matvec(dq, r, w_uk, &sv.latent[j * r..(j + 1) * r]);
                sc[j] = qh.iter().zip(&k).map(|(a, b)| a * b).sum::<f32>() * scale;
                mx = mx.max(sc[j]);
            }
            let mut sum = 0f32;
            for v in sc.iter_mut() {
                *v = common::math::exp(*v - mx);
                sum += *v;
            }
            let mut oh = vec![0f32; dv];
            for j in 0..=t {
                let a = sc[j] / sum;
                sv.att[(t * nh + hh) * n + j] = a;
                let v = matvec(dv, r, w_uv, &sv.latent[j * r..(j + 1) * r]);
                for d in 0..dv {
                    oh[d] += a * v[d];
                }
            }
            sv.heads[t * nh * dv + hh * dv..t * nh * dv + (hh + 1) * dv].copy_from_slice(&oh);
        }
        let yt = matvec(h, nh * dv, m.o, &sv.heads[t * nh * dv..(t + 1) * nh * dv]);
        y[t * h..(t + 1) * h].copy_from_slice(&yt);
    }
    (y, sv)
}

/// → (dx `[n][h]`, d latent `[n][r]` — every position's, as accumulated from the queries).
pub fn mla_backward(m: &MlaRef, sv: &MlaSaved, dy: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (nh, dq, dv, r, qr, h, n) = (m.nh, m.dq, m.dv, m.r, m.qr, m.h, sv.n);
    let scale = 1.0 / common::math::sqrt(dq as f32);
    let mut dx = vec![0f32; n * h];
    let mut dlatent = vec![0f32; n * r];
    for t in 0..n {
        let dheads = matvec_t(h, nh * dv, m.o, &dy[t * h..(t + 1) * h]);
        let mut dq_all = vec![0f32; nh * dq];
        for hh in 0..nh {
            let qh = &sv.q[t * nh * dq + hh * dq..t * nh * dq + (hh + 1) * dq];
            let w_uk = &m.kv_b[hh * (dq + dv) * r..(hh * (dq + dv) + dq) * r];
            let w_uv = &m.kv_b[(hh * (dq + dv) + dq) * r..(hh + 1) * (dq + dv) * r];
            let doh = &dheads[hh * dv..(hh + 1) * dv];
            let att = &sv.att[(t * nh + hh) * n..(t * nh + hh) * n + t + 1];
            // d att_j = ⟨doh, v_j⟩; d v_j = att_j · doh
            let mut datt = vec![0f32; t + 1];
            let mut dot = 0f32;
            for j in 0..=t {
                let v = matvec(dv, r, w_uv, &sv.latent[j * r..(j + 1) * r]);
                datt[j] = doh.iter().zip(&v).map(|(a, b)| a * b).sum();
                dot += att[j] * datt[j];
                let dv_j: Vec<f32> = doh.iter().map(|d| att[j] * d).collect();
                for (a, b) in dlatent[j * r..(j + 1) * r].iter_mut().zip(matvec_t(dv, r, w_uv, &dv_j)) {
                    *a += b;
                }
            }
            // the softmax; d q += scale · Σ ds_j k_j; d k_j = scale · ds_j · q
            let mut dqh = vec![0f32; dq];
            for j in 0..=t {
                let ds = att[j] * (datt[j] - dot) * scale;
                let k = matvec(dq, r, w_uk, &sv.latent[j * r..(j + 1) * r]);
                for d in 0..dq {
                    dqh[d] += ds * k[d];
                }
                let dk: Vec<f32> = qh.iter().map(|q| ds * q).collect();
                for (a, b) in dlatent[j * r..(j + 1) * r].iter_mut().zip(matvec_t(dq, r, w_uk, &dk)) {
                    *a += b;
                }
            }
            dq_all[hh * dq..(hh + 1) * dq].copy_from_slice(&dqh);
        }
        let dqr = matvec_t(nh * dq, qr, m.q_b, &dq_all);
        let dqa = rms_backward_ref(&sv.qa[t * qr..(t + 1) * qr], m.q_a_ln, m.rms_eps, &dqr);
        for (a, b) in dx[t * h..(t + 1) * h].iter_mut().zip(matvec_t(qr, h, m.q_a, &dqa)) {
            *a += b;
        }
    }
    // every position's latent back through its norm and kv_a (the rounding straight through)
    for j in 0..n {
        let dkva = rms_backward_ref(&sv.kva[j * r..(j + 1) * r], m.kv_a_ln, m.rms_eps, &dlatent[j * r..(j + 1) * r]);
        for (a, b) in dx[j * h..(j + 1) * h].iter_mut().zip(matvec_t(r, h, m.kv_a, &dkva)) {
            *a += b;
        }
    }
    (dx, dlatent)
}

#[cfg(test)]
mod mla_tests {
    use super::*;
    use common::testing::Rng;

    #[test]
    fn mla_backward_is_the_gradient() {
        let (nh, dq, dv, r, qr, h, n) = (2usize, 4usize, 3usize, 5usize, 6usize, 7usize, 4usize);
        let mut rnd = Rng::new(0xF1D0_0005);
        let mut u = |m: usize, s: f32| -> Vec<f32> { (0..m).map(|_| rnd.unit() * s).collect() };
        let (q_a, q_b, kv_a, kv_b, o) = (u(qr * h, 0.7), u(nh * dq * qr, 0.7), u(r * h, 0.7), u(nh * (dq + dv) * r, 0.7), u(h * nh * dv, 0.7));
        let q_a_ln: Vec<f32> = u(qr, 0.3).iter().map(|v| 1.0 + v).collect();
        let kv_a_ln: Vec<f32> = u(r, 0.3).iter().map(|v| 1.0 + v).collect();
        let m = MlaRef { nh, dq, dv, r, qr, h, rms_eps: 1e-5, q_a: &q_a, q_a_ln: &q_a_ln, q_b: &q_b, kv_a: &kv_a, kv_a_ln: &kv_a_ln, kv_b: &kv_b, o: &o, round: false };
        let mut x = u(n * h, 1.0);
        let dy = u(n * h, 1.0);
        let loss = |x: &[f32]| -> f64 { mla_forward(&m, x, n).0.iter().zip(&dy).map(|(a, b)| (*a as f64) * (*b as f64)).sum() };
        let (_, sv) = mla_forward(&m, &x, n);
        let (dx, _) = mla_backward(&m, &sv, &dy);
        let eps = 1e-3f32;
        for i in 0..n * h {
            let base = x[i];
            x[i] = base + eps;
            let p = loss(&x);
            x[i] = base - eps;
            let mm = loss(&x);
            x[i] = base;
            let num = ((p - mm) / (2.0 * eps as f64)) as f32;
            assert!((num - dx[i]).abs() <= 3e-3 * (1.0 + num.abs()), "dx[{i}]: numeric {num} analytic {}", dx[i]);
        }
    }
}
