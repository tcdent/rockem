//! One routed expert with its adapter: the forward the node will serve with,
//! keeping what the backward needs, and the backward.
//!
//! The base expert is `glm_5_3_flash::expert::forward`'s arithmetic on its
//! `q4g128` record — the activation quantizer, the INT4×INT8 products, the
//! clamped SwiGLU — unchanged. The adapter adds `B·(A·x̂)` to each of the three
//! projections, where `x̂` is the **dequantized** input the base products see
//! (`q · scale`), so the adapter and the base read the same vector on the CPU
//! and on the cards, which hold only the integers.
//!
//! ```text
//! x̂ = deq(Q(x))              g = Wg·x̂ + Bg·(Ag·x̂)    u = Wu·x̂ + Bu·(Au·x̂)
//! h = silu(min(g, L)) ⊙ clamp(u, ±L)
//! ĥ = deq(Q(h))              out = Wd·ĥ + Bd·(Ad·ĥ)
//! ```
//!
//! Backward: straight-through at both quantizers (`d x = d x̂`, `d h = d ĥ`);
//! the clamps pass the gradient on the closed interval, as torch's `clamp`
//! does; the frozen `W` contribute `Wᵀ·d(out)` through `train_common::q4`
//! and no weight gradient. The expert's own output weight (the router's) is
//! applied by the MoE layer, which scales `dout` before calling this.

use common::q4::{self, GROUP};
use glm_5_3_flash::expert::{Scratch, Weights, DOWN_PARTS, DOWN_ROWS, GU_PARTS, GU_ROWS};
use glm_5_3_flash::{D_FF, H, SWIGLU_LIMIT};
use train_common::q4 as q4t;
use train_common::{lowrank, swiglu};

/// An expert's adapter, f32, rank `r`: `a` is `[r][k]`, `b` is `[m][r]`, for gate and up
/// (`k = H`, `m = D_FF`) and down (`k = D_FF`, `m = H`). The same shape holds its gradient.
#[derive(Clone)]
pub struct Adapter {
    pub r: usize,
    pub ga: Vec<f32>,
    pub gb: Vec<f32>,
    pub ua: Vec<f32>,
    pub ub: Vec<f32>,
    pub da: Vec<f32>,
    pub db: Vec<f32>,
}

impl Adapter {
    pub fn zeros(r: usize) -> Self {
        Adapter { r, ga: vec![0.0; r * H], gb: vec![0.0; D_FF * r], ua: vec![0.0; r * H], ub: vec![0.0; D_FF * r], da: vec![0.0; r * D_FF], db: vec![0.0; H * r] }
    }
    pub fn params(&self) -> usize {
        self.ga.len() + self.gb.len() + self.ua.len() + self.ub.len() + self.da.len() + self.db.len()
    }
}

/// What the forward keeps for the backward, one token.
pub struct Saved {
    pub xh: Vec<f32>, // H: deq(Q(x))
    pub g: Vec<f32>,  // D_FF: the gate projection with its adapter
    pub u: Vec<f32>,  // D_FF
    pub hh: Vec<f32>, // D_FF: deq(Q(h))
    pub zg: Vec<f32>, // r: Ag·x̂
    pub zu: Vec<f32>, // r
    pub zd: Vec<f32>, // r: Ad·ĥ
}

impl Saved {
    pub fn new(r: usize) -> Self {
        Saved { xh: vec![0.0; H], g: vec![0.0; D_FF], u: vec![0.0; D_FF], hh: vec![0.0; D_FF], zg: vec![0.0; r], zu: vec![0.0; r], zd: vec![0.0; r] }
    }
}

#[inline(always)]
fn dequantize(q: &[i8], scale: &[f32], out: &mut [f32]) {
    for (g, chunk) in out.chunks_exact_mut(GROUP).enumerate() {
        let s = scale[g];
        for (i, v) in chunk.iter_mut().enumerate() {
            *v = q[g * GROUP + i] as f32 * s;
        }
    }
}

/// The forward with the adapter, keeping `saved`. With a zero adapter this is
/// `glm_5_3_flash::expert::forward` value for value.
pub fn forward(w: &Weights, ad: &Adapter, x: &[f32], s: &mut Scratch, saved: &mut Saved, out: &mut [f32]) {
    s.xa.quantize(&x[..H]);
    dequantize(s.xa.q, s.xa.scale, &mut saved.xh);
    let xa = s.xa.act();
    for c in 0..GU_PARTS {
        let (gw, uw, gs, us) = w.gu(c);
        q4::gemv::<H>(GU_ROWS, gw, gs, &xa, &mut s.gu[c * GU_ROWS..(c + 1) * GU_ROWS]);
        q4::gemv::<H>(GU_ROWS, uw, us, &xa, &mut s.gu[D_FF + c * GU_ROWS..D_FF + (c + 1) * GU_ROWS]);
    }
    let (g, u) = s.gu.split_at_mut(D_FF);
    lowrank::down(ad.r, H, &ad.ga, &saved.xh, &mut saved.zg);
    lowrank::up(D_FF, ad.r, &ad.gb, &saved.zg, g);
    lowrank::down(ad.r, H, &ad.ua, &saved.xh, &mut saved.zu);
    lowrank::up(D_FF, ad.r, &ad.ub, &saved.zu, u);
    saved.g.copy_from_slice(g);
    saved.u.copy_from_slice(u);
    common::swiglu::swiglu_clamped::<D_FF>(s.gu, SWIGLU_LIMIT, s.h);
    s.ha.quantize(s.h);
    dequantize(s.ha.q, s.ha.scale, &mut saved.hh);
    let ha = s.ha.act();
    for c in 0..DOWN_PARTS {
        let (dw, ds) = w.down(c);
        q4::gemv::<D_FF>(DOWN_ROWS, dw, ds, &ha, &mut out[c * DOWN_ROWS..(c + 1) * DOWN_ROWS]);
    }
    lowrank::down(ad.r, D_FF, &ad.da, &saved.hh, &mut saved.zd);
    lowrank::up(H, ad.r, &ad.db, &saved.zd, out);
}

/// Per-token scratch for the backward.
pub struct Grad {
    pub dh: Vec<f32>, // D_FF: d out / d ĥ
    pub dg: Vec<f32>, // D_FF
    pub du: Vec<f32>, // D_FF
    pub t: Vec<f32>,  // r
}

impl Grad {
    pub fn new(r: usize) -> Self {
        Grad { dh: vec![0.0; D_FF], dg: vec![0.0; D_FF], du: vec![0.0; D_FF], t: vec![0.0; r] }
    }
}

/// Given `dout` (already scaled by the token's routing weight), accumulate the adapter's
/// gradient into `grads` and the input's into `dx` (`+=`, `[H]`).
pub fn backward(w: &Weights, ad: &Adapter, saved: &Saved, dout: &[f32], gr: &mut Grad, grads: &mut Adapter, dx: &mut [f32]) {
    // down: d ĥ = Wdᵀ·dout + Adᵀ·(Bdᵀ·dout); dBd += dout ⊗ zd; dAd += t ⊗ ĥ
    for v in gr.dh.iter_mut() {
        *v = 0.0;
    }
    for c in 0..DOWN_PARTS {
        let (dw, ds) = w.down(c);
        q4t::gemv_t::<D_FF>(DOWN_ROWS, dw, ds, &dout[c * DOWN_ROWS..(c + 1) * DOWN_ROWS], &mut gr.dh);
    }
    lowrank::up_backward(H, ad.r, &ad.db, &saved.zd, dout, &mut gr.t, &mut grads.db);
    lowrank::down_backward(ad.r, D_FF, &ad.da, &saved.hh, &gr.t, &mut grads.da, &mut gr.dh);
    // the clamped SwiGLU: d h = d ĥ (straight through); d g into `dh`'s place, d u into `du`
    let dh = core::mem::take(&mut gr.dh);
    swiglu::clamped_backward(D_FF, &saved.g, &saved.u, SWIGLU_LIMIT, &dh, &mut gr.dg, &mut gr.du);
    gr.dh = dh;
    // gate and up: d x̂ = Wgᵀ·dg + Wuᵀ·du + Agᵀ·(Bgᵀ·dg) + Auᵀ·(Buᵀ·du)
    for c in 0..GU_PARTS {
        let (gw, uw, gs, us) = w.gu(c);
        q4t::gemv_t::<H>(GU_ROWS, gw, gs, &gr.dg[c * GU_ROWS..(c + 1) * GU_ROWS], dx);
        q4t::gemv_t::<H>(GU_ROWS, uw, us, &gr.du[c * GU_ROWS..(c + 1) * GU_ROWS], dx);
    }
    lowrank::up_backward(D_FF, ad.r, &ad.gb, &saved.zg, &gr.dg, &mut gr.t, &mut grads.gb);
    lowrank::down_backward(ad.r, H, &ad.ga, &saved.xh, &gr.t, &mut grads.ga, dx);
    lowrank::up_backward(D_FF, ad.r, &ad.ub, &saved.zu, &gr.du, &mut gr.t, &mut grads.ub);
    lowrank::down_backward(ad.r, H, &ad.ua, &saved.xh, &gr.t, &mut grads.ua, dx);
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::reference;
    use common::testing::Rng;
    use glm_5_3_flash::expert::{pack_items, ScratchBuf, Stream, ITEMS};
    use glm_5_3_flash::layout::{self, DOWN_ITEMS, GU_ITEMS};

    /// A random expert at (FIRST_K_DENSE_REPLACE, 0) inside one block of stream, with its
    /// dequantized dense twin: gate_up `[2·D_FF][H]`, down `[H][D_FF]`.
    pub struct Fixture {
        pub block: Vec<u32>,
        pub addr: Vec<u64>,
        pub gate_up: Vec<f32>,
        pub down: Vec<f32>,
    }

    impl Fixture {
        pub fn new(rnd: &mut Rng) -> Self {
            let mk = |m: usize, k: usize, rnd: &mut Rng| {
                let w: Vec<i8> = (0..m * k).map(|_| rnd.i4()).collect();
                let s: Vec<f32> = (0..m * (k / GROUP)).map(|_| 0.01 + (rnd.next() % 100) as f32 * 1e-4).collect();
                let dense: Vec<f32> = (0..m * k).map(|i| w[i] as f32 * s[(i / k) * (k / GROUP) + (i % k) / GROUP]).collect();
                let (pw, ps) = q4::pack(m, k, &w, &s);
                (pw, ps, dense)
            };
            let (guw, gus, gate_up) = mk(2 * D_FF, H, rnd);
            let (dw, ds, down) = mk(H, D_FF, rnd);
            let mut block = vec![0u32; layout::XBLOCK_BYTES / 4];
            let stream: &mut [u8] = unsafe { core::slice::from_raw_parts_mut(block.as_mut_ptr() as *mut u8, layout::XBLOCK_BYTES) };
            let (l, e) = (glm_5_3_flash::FIRST_K_DENSE_REPLACE, 0);
            for (off, it) in pack_items(l, e, &guw, &gus, &dw, &ds) {
                stream[off..off + it.len()].copy_from_slice(&it);
            }
            let mut addr = Vec::with_capacity(layout::MOE_LAYERS * glm_5_3_flash::E * ITEMS);
            for ll in glm_5_3_flash::FIRST_K_DENSE_REPLACE..layout::LAYERS_IN_FILE {
                for ee in 0..glm_5_3_flash::E {
                    for c in 0..GU_ITEMS {
                        addr.push(layout::gu_item(ll, ee, c).start as u64);
                    }
                    for c in 0..DOWN_ITEMS {
                        addr.push(layout::down_item(ll, ee, c).start as u64);
                    }
                }
            }
            Fixture { block, addr, gate_up, down }
        }
        pub fn stream(&self) -> Stream<'_> {
            let bytes: &[u8] = unsafe { core::slice::from_raw_parts(self.block.as_ptr() as *const u8, layout::XBLOCK_BYTES) };
            Stream::placed(bytes, self.addr.clone(), vec![0; layout::MOE_LAYERS * glm_5_3_flash::E * ITEMS], Vec::new())
        }
    }

    pub fn random_adapter(r: usize, scale: f32, rnd: &mut Rng) -> Adapter {
        let mut ad = Adapter::zeros(r);
        for v in ad.ga.iter_mut().chain(ad.gb.iter_mut()).chain(ad.ua.iter_mut()).chain(ad.ub.iter_mut()).chain(ad.da.iter_mut()).chain(ad.db.iter_mut()) {
            *v = rnd.unit() * scale;
        }
        ad
    }

    fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(worst <= tol * scale, "{what}: worst |Δ| {worst} against a scale of {scale}");
    }

    /// With a zero adapter the forward is the inference crate's, value for value.
    #[test]
    fn zero_adapter_is_the_inference_forward() {
        let mut rnd = Rng::new(0xADA0_0001);
        let fx = Fixture::new(&mut rnd);
        let st = fx.stream();
        let w = Weights::at(&st, glm_5_3_flash::FIRST_K_DENSE_REPLACE, 0);
        let x: Vec<f32> = (0..H).map(|_| rnd.unit()).collect();
        let mut buf = ScratchBuf::new();
        let mut want = vec![0f32; H];
        glm_5_3_flash::expert::forward(&w, &x, &mut buf.scratch(), &mut want);
        let ad = Adapter::zeros(8);
        let mut saved = Saved::new(8);
        let mut got = vec![0f32; H];
        forward(&w, &ad, &x, &mut buf.scratch(), &mut saved, &mut got);
        assert!(got.iter().zip(&want).all(|(a, b)| a == b));
    }

    /// The kernels against the dense f32 reference on the dequantized weights, forward and backward.
    #[test]
    fn kernel_matches_dense_reference() {
        let mut rnd = Rng::new(0xADA0_0002);
        let fx = Fixture::new(&mut rnd);
        let st = fx.stream();
        let w = Weights::at(&st, glm_5_3_flash::FIRST_K_DENSE_REPLACE, 0);
        let r = 16;
        let ad = random_adapter(r, 0.05, &mut rnd);
        let x: Vec<f32> = (0..H).map(|_| rnd.unit()).collect();
        let dout: Vec<f32> = (0..H).map(|_| rnd.unit()).collect();

        let mut buf = ScratchBuf::new();
        let mut saved = Saved::new(r);
        let mut out = vec![0f32; H];
        forward(&w, &ad, &x, &mut buf.scratch(), &mut saved, &mut out);
        let mut grads = Adapter::zeros(r);
        let mut dx = vec![0f32; H];
        let mut gr = Grad::new(r);
        backward(&w, &ad, &saved, &dout, &mut gr, &mut grads, &mut dx);

        let dense = reference::Dense { h: H, ff: D_FF, gate_up: &fx.gate_up, down: &fx.down, limit: SWIGLU_LIMIT };
        let lr = reference::LowRank { r, ga: &ad.ga, gb: &ad.gb, ua: &ad.ua, ub: &ad.ub, da: &ad.da, db: &ad.db };
        let (rout, rsaved) = reference::forward(&dense, &lr, &x, true);
        let (rdx, rgrads) = reference::backward(&dense, &lr, &rsaved, &dout);

        // the same quantized x̂ and ĥ on both sides, so only summation order differs: measured ~2e-6 of scale
        close(&out, &rout, 1e-4, "out");
        close(&dx, &rdx, 1e-4, "dx");
        close(&grads.ga, &rgrads.ga, 1e-4, "dAg");
        close(&grads.gb, &rgrads.gb, 1e-4, "dBg");
        close(&grads.ua, &rgrads.ua, 1e-4, "dAu");
        close(&grads.ub, &rgrads.ub, 1e-4, "dBu");
        close(&grads.da, &rgrads.da, 1e-4, "dAd");
        close(&grads.db, &rgrads.db, 1e-4, "dBd");
    }
}
