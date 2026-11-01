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

#[cfg(test)]
mod tests {
    use super::*;
    use common::testing::Rng;

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
