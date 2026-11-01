//! The MoE half of a layer, one token, on its normed input `xn`: the forward the node
//! serves with (plus the experts' adapters), keeping what the backward needs, and the
//! backward to `d xn`. Everything but the adapters is frozen, so the router, the shared
//! expert and the base experts contribute only the gradient at their inputs.
//!
//! ```text
//! s = σ(router · xn)         picks = top TOP_K of s + bias (discrete: no gradient)
//! S = 1e-20 + Σ_picks s_e    wt_e = s_e / S · ROUTED_SCALING_FACTOR
//! out = Σ_picks wt_e · expert_e(xn) + shared(xn)
//! ```
//!
//! Backward, given `d out`: each picked expert gets `wt_e · d out` (`expert::backward`);
//! `d wt_e = ⟨expert_e(xn), d out⟩` goes back through the renormalization and the sigmoid
//! into the router's picked rows, `d xn += Σ_e d logit_e · router[e]`; the shared expert is
//! the FP8 transposed products around the clamped SwiGLU. `d xn` accumulates (`+=`).
//!
//! The dense FFN of layers 0–2 has no backward here: nothing below layer
//! `FIRST_K_DENSE_REPLACE` moves, so a backward stops at that layer's input.

use crate::expert::{self, Adapter};
use common::gemv;
use glm_5_3_flash::expert::{ScratchBuf, Stream, Weights};
use glm_5_3_flash::ffn::route_scores;
use glm_5_3_flash::weights::Moe;
use glm_5_3_flash::{D_FF, E, H, N_SHARED_EXPERTS, ROUTED_SCALING_FACTOR, SWIGLU_LIMIT, TOP_K};
use train_common::{bf16, fp8, swiglu};

/// The shared expert's width.
pub const S_FF: usize = N_SHARED_EXPERTS * D_FF;

pub struct Scratch {
    pub xbuf: ScratchBuf,
    pub score: Vec<f32>, // E
    pub gu: Vec<f32>,    // 2·S_FF
    pub h: Vec<f32>,     // S_FF
    pub dh: Vec<f32>,    // S_FF
    pub dg: Vec<f32>,    // S_FF
    pub du: Vec<f32>,    // S_FF
    pub sh: Vec<f32>,    // H: the shared expert's output
    pub grad: expert::Grad,
}

impl Scratch {
    pub fn new(r: usize) -> Self {
        Scratch { xbuf: ScratchBuf::new(), score: vec![0.0; E], gu: vec![0.0; 2 * S_FF], h: vec![0.0; S_FF], dh: vec![0.0; S_FF], dg: vec![0.0; S_FF], du: vec![0.0; S_FF], sh: vec![0.0; H], grad: expert::Grad::new(r) }
    }
}

/// What the forward keeps for the backward, one token.
pub struct Saved {
    pub xn: Vec<f32>, // H
    pub idx: [usize; TOP_K],
    pub wt: [f32; TOP_K],
    pub s: [f32; TOP_K], // the picks' sigmoid scores
    pub sum: f32,        // S
    pub experts: Vec<expert::Saved>,
    pub expert_out: Vec<f32>, // [TOP_K][H], each expert's raw output
    pub sg: Vec<f32>,         // S_FF: the shared expert's gate pre-activation
    pub su: Vec<f32>,         // S_FF
}

impl Saved {
    pub fn new(r: usize) -> Self {
        Saved { xn: vec![0.0; H], idx: [0; TOP_K], wt: [0.0; TOP_K], s: [0.0; TOP_K], sum: 0.0, experts: (0..TOP_K).map(|_| expert::Saved::new(r)).collect(), expert_out: vec![0.0; TOP_K * H], sg: vec![0.0; S_FF], su: vec![0.0; S_FF] }
    }
}

/// `ad[e]` is expert `e`'s adapter (`ad.len() == E`); `out` is written (`=`): the routed sum plus the shared expert.
pub fn forward(l: usize, w: &Moe, stream: &Stream, ad: &[Adapter], xn: &[f32], s: &mut Scratch, saved: &mut Saved, out: &mut [f32]) {
    saved.xn.copy_from_slice(&xn[..H]);
    // the router, as `ffn::route`: logits, sigmoids in place, the picks ascending, the weights over the reference mass
    gemv::bf16::<E, H>(w.router, xn, &mut s.score);
    let (idx, wt) = route_scores(w, &mut s.score, None);
    saved.idx = idx;
    saved.wt = wt;
    saved.sum = 1e-20;
    for k in 0..TOP_K {
        debug_assert!(idx[k] < E, "ROUTED_K is TOP_K: every pick runs");
        saved.s[k] = s.score[idx[k]];
        saved.sum += saved.s[k];
    }
    // the experts, each on xn with its adapter, the weighted sum in pick order
    for v in out[..H].iter_mut() {
        *v = 0.0;
    }
    for k in 0..TOP_K {
        let e = idx[k];
        let eo = &mut saved.expert_out[k * H..(k + 1) * H];
        expert::forward(&Weights::at(stream, l, e), &ad[e], xn, &mut s.xbuf.scratch(), &mut saved.experts[k], eo);
        for i in 0..H {
            out[i] += wt[k] * eo[i];
        }
    }
    // the shared expert, as `ffn::moe_shared` (one core's rows at a time: the same bits a row)
    gemv::fp8::<S_FF, H>(w.shared_gate.w, w.shared_gate.s, xn, &mut s.gu[..S_FF]);
    gemv::fp8::<S_FF, H>(w.shared_up.w, w.shared_up.s, xn, &mut s.gu[S_FF..2 * S_FF]);
    saved.sg.copy_from_slice(&s.gu[..S_FF]);
    saved.su.copy_from_slice(&s.gu[S_FF..2 * S_FF]);
    common::swiglu::swiglu_clamped::<S_FF>(&s.gu, SWIGLU_LIMIT, &mut s.h);
    gemv::fp8::<H, S_FF>(w.shared_down.w, w.shared_down.s, &s.h, &mut s.sh);
    for i in 0..H {
        out[i] += s.sh[i];
    }
}

/// `grads[e]` accumulates expert `e`'s adapter gradient; `dxn` accumulates (`+=`, `[H]`).
pub fn backward(l: usize, w: &Moe, stream: &Stream, ad: &[Adapter], saved: &Saved, dout: &[f32], s: &mut Scratch, grads: &mut [Adapter], dxn: &mut [f32]) {
    // the shared expert: d h = Wdᵀ·dout; the SwiGLU; d xn += Wgᵀ·dg + Wuᵀ·du
    for v in s.dh.iter_mut() {
        *v = 0.0;
    }
    fp8::rows_t::<S_FF>(w.shared_down.w, w.shared_down.s, dout, &mut s.dh, 0, H);
    swiglu::clamped_backward(S_FF, &saved.sg, &saved.su, SWIGLU_LIMIT, &s.dh, &mut s.dg, &mut s.du);
    fp8::rows_t::<H>(w.shared_gate.w, w.shared_gate.s, &s.dg, dxn, 0, S_FF);
    fp8::rows_t::<H>(w.shared_up.w, w.shared_up.s, &s.du, dxn, 0, S_FF);
    // the experts: each on wt_e · dout; d wt_e = ⟨expert_e, dout⟩
    let mut dwt = [0f32; TOP_K];
    let mut scaled = vec![0f32; H];
    for k in 0..TOP_K {
        let e = saved.idx[k];
        let eo = &saved.expert_out[k * H..(k + 1) * H];
        let mut d = 0f32;
        for i in 0..H {
            d += eo[i] * dout[i];
            scaled[i] = saved.wt[k] * dout[i];
        }
        dwt[k] = d;
        expert::backward(&Weights::at(stream, l, e), &ad[e], &saved.experts[k], &scaled, &mut s.grad, &mut grads[e], dxn);
    }
    // the router: wt_j = c · s_j / S  ⇒  d s_e = c / S · (d wt_e − Σ_j d wt_j · s_j / S); d logit = d s · s · (1 − s)
    let mut mixed = 0f32;
    for k in 0..TOP_K {
        mixed += dwt[k] * saved.s[k];
    }
    mixed /= saved.sum;
    let mut rows = [(0usize, 0f32); TOP_K];
    for k in 0..TOP_K {
        let sk = saved.s[k];
        let ds = ROUTED_SCALING_FACTOR / saved.sum * (dwt[k] - mixed);
        rows[k] = (saved.idx[k], ds * sk * (1.0 - sk));
    }
    bf16::rows_t::<H>(w.router, &rows, dxn);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert::tests::random_adapter;
    use crate::reference;
    use common::gemv::{e4m3, widen, FP8_BLOCK};
    use common::pool::Serial;
    use common::q4::{self, GROUP};
    use common::testing::Rng;
    use common::trace::NoTrace;
    use glm_5_3_flash::expert::{pack_items, ITEMS};
    use glm_5_3_flash::ffn::Routed;
    use glm_5_3_flash::layout::{self, DOWN_ITEMS, GU_ITEMS};
    use glm_5_3_flash::weights::Fp8;
    use glm_5_3_flash::{FIRST_K_DENSE_REPLACE, INTERMEDIATE_SIZE};

    const L: usize = FIRST_K_DENSE_REPLACE;

    struct Fp8Buf {
        w: Vec<u8>,
        s: Vec<f32>,
        dense: Vec<f32>,
    }
    fn fp8_record(m: usize, k: usize, rnd: &mut Rng) -> Fp8Buf {
        let w: Vec<u8> = (0..m * k).map(|_| { let b = (rnd.next() % 256) as u8; if b & 0x7f == 0x7f { b ^ 1 } else { b } }).collect();
        let kb = k / FP8_BLOCK;
        let s: Vec<f32> = (0..m.div_ceil(FP8_BLOCK) * kb).map(|_| 0.0005 + (rnd.next() % 100) as f32 * 2e-5).collect();
        let dense: Vec<f32> = (0..m * k).map(|i| e4m3(w[i]) * s[(i / k / FP8_BLOCK) * kb + (i % k) / FP8_BLOCK]).collect();
        Fp8Buf { w, s, dense }
    }

    /// The layer's MoE weights, random, with their dense twins; the routed experts a block of
    /// stream holds (experts 0..XBLOCK of layer L), the picked ones packed with dense twins kept.
    struct Fixture {
        router: Vec<u16>,
        router_dense: Vec<f32>,
        bias: Vec<f32>,
        sg: Fp8Buf,
        su: Fp8Buf,
        sd: Fp8Buf,
        block: Vec<u32>,
        addr: Vec<u64>,
        /// (e, gate_up dense, down dense) for the experts packed
        experts: Vec<(usize, Vec<f32>, Vec<f32>)>,
    }

    impl Fixture {
        fn new(xn: &[f32], rnd: &mut Rng) -> Self {
            let router_dense: Vec<f32> = (0..E * H).map(|_| widen((rnd.unit().to_bits() >> 16) as u16 & 0xFFFF) * 0.05).collect();
            let router: Vec<u16> = router_dense.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            let router_dense: Vec<f32> = router.iter().map(|&b| widen(b)).collect();
            // the picks must land in the block: experts past XBLOCK get a bias no score overcomes
            let bias: Vec<f32> = (0..E).map(|e| if e < layout::XBLOCK { rnd.unit() * 0.1 } else { -100.0 }).collect();
            let (sg, su, sd) = (fp8_record(S_FF, H, rnd), fp8_record(S_FF, H, rnd), fp8_record(H, S_FF, rnd));
            // route once, the reference's way, to know which experts to build
            let w_ref = reference::Router { e: E, top_k: TOP_K, h: H, w: &router_dense, bias: &bias, scale: ROUTED_SCALING_FACTOR };
            let (idx, _, _) = reference::route(&w_ref, xn);
            let mut block = vec![0u32; layout::XBLOCK_BYTES / 4];
            let bytes: &mut [u8] = unsafe { core::slice::from_raw_parts_mut(block.as_mut_ptr() as *mut u8, layout::XBLOCK_BYTES) };
            let mut experts = Vec::new();
            for &e in &idx {
                let mk = |m: usize, k: usize, rnd: &mut Rng| {
                    let w: Vec<i8> = (0..m * k).map(|_| rnd.i4()).collect();
                    let s: Vec<f32> = (0..m * (k / GROUP)).map(|_| 0.002 + (rnd.next() % 100) as f32 * 2e-5).collect();
                    let dense: Vec<f32> = (0..m * k).map(|i| w[i] as f32 * s[(i / k) * (k / GROUP) + (i % k) / GROUP]).collect();
                    let (pw, ps) = q4::pack(m, k, &w, &s);
                    (pw, ps, dense)
                };
                let (guw, gus, gate_up) = mk(2 * D_FF, H, rnd);
                let (dw, ds, down) = mk(H, D_FF, rnd);
                for (off, it) in pack_items(L, e, &guw, &gus, &dw, &ds) {
                    bytes[off..off + it.len()].copy_from_slice(&it);
                }
                experts.push((e, gate_up, down));
            }
            let mut addr = Vec::with_capacity(layout::MOE_LAYERS * E * ITEMS);
            for ll in L..layout::LAYERS_IN_FILE {
                for ee in 0..E {
                    for c in 0..GU_ITEMS {
                        addr.push(layout::gu_item(ll, ee, c).start as u64);
                    }
                    for c in 0..DOWN_ITEMS {
                        addr.push(layout::down_item(ll, ee, c).start as u64);
                    }
                }
            }
            Fixture { router, router_dense, bias, sg, su, sd, block, addr, experts }
        }
        fn stream(&self) -> Stream<'_> {
            let bytes: &[u8] = unsafe { core::slice::from_raw_parts(self.block.as_ptr() as *const u8, layout::XBLOCK_BYTES) };
            Stream::placed(bytes, self.addr.clone(), vec![0; layout::MOE_LAYERS * E * ITEMS], Vec::new())
        }
        fn moe(&self) -> Moe<'_> {
            Moe { router: &self.router, bias: &self.bias, shared_gate: Fp8 { w: &self.sg.w, s: &self.sg.s }, shared_up: Fp8 { w: &self.su.w, s: &self.su.s }, shared_down: Fp8 { w: &self.sd.w, s: &self.sd.s } }
        }
        /// The shared expert as one dense expert: gate rows then up rows.
        fn shared_dense(&self) -> Vec<f32> {
            let mut gu = self.sg.dense.clone();
            gu.extend_from_slice(&self.su.dense);
            gu
        }
    }

    fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(worst <= tol * scale, "{what}: worst |Δ| {worst} against a scale of {scale}");
    }

    /// The inference crate's routed experts over the stream, one expert at a time.
    struct Direct<'a> {
        stream: &'a Stream<'a>,
        buf: ScratchBuf,
    }
    impl Routed for Direct<'_> {
        fn run(&mut self, l: usize, idx: &[usize; TOP_K], x: &[f32], out: &mut [f32]) {
            for k in 0..TOP_K {
                glm_5_3_flash::expert::forward(&Weights::at(self.stream, l, idx[k]), x, &mut self.buf.scratch(), &mut out[k * H..(k + 1) * H]);
            }
        }
    }

    /// With zero adapters the forward is the inference crate's MoE (`moe_routed` + `moe_shared`), value for value.
    #[test]
    fn zero_adapters_are_the_inference_moe() {
        let mut rnd = Rng::new(0xF0E0_0001);
        let xn: Vec<f32> = (0..H).map(|_| rnd.unit()).collect();
        let fx = Fixture::new(&xn, &mut rnd);
        let st = fx.stream();
        let w = fx.moe();
        // inference
        let (mut gu, mut h, mut eo, mut sh, mut sc) = (vec![0f32; 2 * INTERMEDIATE_SIZE], vec![0f32; INTERMEDIATE_SIZE], vec![0f32; TOP_K * H], vec![0f32; H], vec![0f32; E]);
        let mut is = glm_5_3_flash::ffn::Scratch { gu: &mut gu, h: &mut h, expert_out: &mut eo, shared: &mut sh, score: &mut sc };
        let mut direct = Direct { stream: &st, buf: ScratchBuf::new() };
        let mut want = vec![0f32; H];
        glm_5_3_flash::ffn::moe_routed(L, &w, &xn, &mut is, &Serial, &mut direct, None, &mut NoTrace, &mut want);
        glm_5_3_flash::ffn::moe_shared(&w, &xn, &mut is, &Serial);
        for i in 0..H {
            want[i] += is.shared[i];
        }
        // training
        let ad: Vec<Adapter> = (0..E).map(|_| Adapter::zeros(0)).collect();
        let mut s = Scratch::new(0);
        let mut saved = Saved::new(0);
        let mut got = vec![0f32; H];
        forward(L, &w, &st, &ad, &xn, &mut s, &mut saved, &mut got);
        assert!(got.iter().zip(&want).all(|(a, b)| a == b), "worst Δ {}", got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max));
    }

    /// Forward and backward against the dense f32 reference at the real shape.
    #[test]
    fn kernel_matches_dense_reference() {
        let mut rnd = Rng::new(0xF0E0_0002);
        let xn: Vec<f32> = (0..H).map(|_| rnd.unit()).collect();
        let dout: Vec<f32> = (0..H).map(|_| rnd.unit()).collect();
        let fx = Fixture::new(&xn, &mut rnd);
        let st = fx.stream();
        let w = fx.moe();
        let r = 4;
        let mut ad: Vec<Adapter> = (0..E).map(|_| Adapter::zeros(0)).collect();
        for &(e, _, _) in &fx.experts {
            ad[e] = random_adapter(r, 0.05, &mut rnd);
        }
        let mut s = Scratch::new(r);
        let mut saved = Saved::new(r);
        let mut out = vec![0f32; H];
        forward(L, &w, &st, &ad, &xn, &mut s, &mut saved, &mut out);
        let mut grads: Vec<Adapter> = ad.iter().map(|a| Adapter::zeros(a.r)).collect();
        let mut dxn = vec![0f32; H];
        backward(L, &w, &st, &ad, &saved, &dout, &mut s, &mut grads, &mut dxn);

        let router = reference::Router { e: E, top_k: TOP_K, h: H, w: &fx.router_dense, bias: &fx.bias, scale: ROUTED_SCALING_FACTOR };
        let shared_gu = fx.shared_dense();
        let shared = reference::Dense { h: H, ff: S_FF, gate_up: &shared_gu, down: &fx.sd.dense, limit: SWIGLU_LIMIT };
        let none = reference::LowRank { r: 0, ga: &[], gb: &[], ua: &[], ub: &[], da: &[], db: &[] };
        let experts: Vec<(usize, reference::Dense, reference::LowRank)> = fx
            .experts
            .iter()
            .map(|(e, gu, d)| {
                let a = &ad[*e];
                (*e, reference::Dense { h: H, ff: D_FF, gate_up: gu, down: d, limit: SWIGLU_LIMIT }, reference::LowRank { r: a.r, ga: &a.ga, gb: &a.gb, ua: &a.ua, ub: &a.ub, da: &a.da, db: &a.db })
            })
            .collect();
        let moe = reference::MoeRef { router: &router, shared: &shared, shared_lr: &none, experts: &experts };
        let (rout, rsaved) = reference::moe_forward(&moe, &xn, true);
        assert_eq!(rsaved.idx, saved.idx.to_vec(), "the picks");
        let (rdx, rgrads) = reference::moe_backward(&moe, &rsaved, &dout);

        close(&out, &rout, 1e-4, "out");
        close(&dxn, &rdx, 1e-4, "dxn");
        for (k, (e, g)) in rgrads.into_iter().enumerate() {
            close(&grads[e].ga, &g.ga, 1e-4, "dAg");
            close(&grads[e].gb, &g.gb, 1e-4, "dBg");
            close(&grads[e].ua, &g.ua, 1e-4, "dAu");
            close(&grads[e].ub, &g.ub, 1e-4, "dBu");
            // dAd = t ⊗ ĥ and dBd = dout ⊗ (Ad·ĥ), and ĥ is the quantized h: the two sides' h differ by summation order (INT8
            // integer dots against f32 products), so an element at a rounding boundary lands one INT8
            // step apart (and a group whose amax moved by an ulp has every element off by ~1e-7 — not
            // a flip). The flips are few and each is exactly a step of the group's scale (amax / 127);
            // dAd's gate is that step relative to its scale; dBd sees a flip through one row of Ad.
            let (kh, rh) = (&saved.experts[k].hh, &rsaved.experts[k].1.hh);
            let mut flips = 0;
            for g in 0..D_FF / GROUP {
                let step = rh[g * GROUP..(g + 1) * GROUP].iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
                for j in g * GROUP..(g + 1) * GROUP {
                    let d = (kh[j] - rh[j]).abs();
                    if d > 0.5 * step {
                        flips += 1;
                        assert!(d <= 1.001 * step, "ĥ[{j}] off by more than a quantizer step: {} vs {}", kh[j], rh[j]);
                    }
                }
            }
            assert!(flips * 100 < D_FF, "expert {e}: {flips} of {D_FF} quantized activations flipped");
            close(&grads[e].da, &g.da, 1.0 / 127.0, "dAd");
            close(&grads[e].db, &g.db, 2e-3, "dBd");
        }
    }
}
