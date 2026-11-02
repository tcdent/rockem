//! Fixtures a test or a bench builds a model from without a bundle: random weights in the
//! bundle's formats for one of each layer kind (the dense layers share one KDA and one FFN so a
//! five-layer model stays in memory), and a stream holding experts `0..TOP_K` of two MoE layers
//! in one block. The router's bias pins every pick to those experts.

use crate::testing::*;
use common::q4::{self, GROUP};
use common::testing::Rng;
use glm_5_3_flash::expert::{pack_items, Stream, ITEMS};
use glm_5_3_flash::layout::{self, DOWN_ITEMS, GU_ITEMS};
use glm_5_3_flash::weights::{Attn, DenseFfn, Ffn, Hc, Kda, Layer, Mla, Moe, Weights};
use glm_5_3_flash::*;

    /// Random weights in the bundle's formats for one of each layer kind, held so the `Weights`
/// can borrow them; the dense layers share one KDA and one FFN so the test stays in memory.
pub struct Bufs {
    embed: Vec<u16>,
    norm: Vec<f32>,
    lm_head: Vec<u16>,
    ln: Vec<f32>,
    hc_fn: Vec<u16>,
    hc_base: Vec<f32>,
    hc_scale: [f32; 3],
    kda: (Vec<u16>, Vec<u16>, Vec<u16>, [Vec<u16>; 3], Vec<u16>, Vec<u16>, Vec<f32>, Vec<f32>, Vec<u16>, Vec<u16>, Vec<u16>, Vec<f32>, Vec<u16>),
    mla: (Fp8Buf, Vec<f32>, Fp8Buf, Fp8Buf, Vec<f32>, Vec<u16>, Fp8Buf, Vec<u16>, Vec<u16>, Vec<f32>, Vec<f32>, Vec<u16>, Vec<f32>, Vec<u16>),
    dense: (Fp8Buf, Fp8Buf, Fp8Buf),
    router: Vec<u16>,
    bias: Vec<f32>,
    shared: (Fp8Buf, Fp8Buf, Fp8Buf),
}

impl Bufs {
    pub fn new(vocab: usize, rnd: &mut Rng) -> Self {
        let fp = |m, k, sc, rnd: &mut Rng| fp8_record(m, k, sc, rnd);
        Bufs {
            embed: bf(vocab * H, 1.0, rnd),
            norm: (0..H).map(|_| 1.0 + rnd.unit() * 0.2).collect(),
            lm_head: bf(vocab * H, 0.05, rnd),
            ln: (0..H).map(|_| 1.0 + rnd.unit() * 0.2).collect(),
            hc_fn: bf(HC_COEFS * HC * H, 0.02, rnd),
            hc_base: unit(HC_COEFS, 1.0, rnd),
            hc_scale: [0.7, 1.3, 0.9],
            kda: (bf(KDA_QKV * H, 0.03, rnd), bf(KDA_QKV * H, 0.03, rnd), bf(KDA_QKV * H, 0.03, rnd), [bf(KDA_QKV * CONV_TAPS, 0.5, rnd), bf(KDA_QKV * CONV_TAPS, 0.5, rnd), bf(KDA_QKV * CONV_TAPS, 0.5, rnd)], bf(KDA_DIM * H, 0.03, rnd), bf(KDA_QKV * KDA_DIM, 0.1, rnd), unit(KDA_QKV, 1.0, rnd), unit(KDA_HEADS, 0.5, rnd), bf(KDA_HEADS * H, 0.03, rnd), bf(KDA_DIM * H, 0.03, rnd), bf(KDA_QKV * KDA_DIM, 0.1, rnd), (0..KDA_DIM).map(|_| 1.0 + rnd.unit() * 0.3).collect(), bf(H * KDA_QKV, 0.03, rnd)),
            mla: (fp(Q_LORA_RANK, H, 0.002, rnd), (0..Q_LORA_RANK).map(|_| 1.0 + rnd.unit() * 0.3).collect(), fp(MLA_HEADS * Q_HEAD, Q_LORA_RANK, 0.002, rnd), fp(KV_LORA_RANK, H, 0.002, rnd), (0..KV_LORA_RANK).map(|_| 1.0 + rnd.unit() * 0.3).collect(), bf(MLA_HEADS * KVB_HEAD * KV_LORA_RANK, 0.03, rnd), fp(H, MLA_HEADS * V_HEAD_DIM, 0.001, rnd), bf(INDEX_N_HEADS * INDEX_HEAD_DIM * Q_LORA_RANK, 0.03, rnd), bf(INDEX_HEAD_DIM * H, 0.03, rnd), vec![1.0; INDEX_HEAD_DIM], vec![0.0; INDEX_HEAD_DIM], bf(INDEX_N_HEADS * H, 0.03, rnd), unit(INDEX_KPOOL * INDEX_HEAD_DIM, 0.1, rnd), bf(INDEX_HEAD_DIM * H, 0.03, rnd)),
            dense: (fp(INTERMEDIATE_SIZE, H, 0.002, rnd), fp(INTERMEDIATE_SIZE, H, 0.002, rnd), fp(H, INTERMEDIATE_SIZE, 0.001, rnd)),
            router: bf(E * H, 0.05, rnd),
            // the picks are always experts 0..TOP_K: those are the ones packed
            bias: (0..E).map(|e| if e < TOP_K { 0.0 } else { -100.0 }).collect(),
            shared: (fp(D_FF, H, 0.002, rnd), fp(D_FF, H, 0.002, rnd), fp(H, D_FF, 0.001, rnd)),
        }
    }
    fn hc(&self) -> Hc<'_> {
        Hc { fn_: &self.hc_fn, base: &self.hc_base, scale: &self.hc_scale }
    }
    fn kda(&self) -> Attn<'_> {
        let k = &self.kda;
        Attn::Kda(Kda { q: &k.0, k: &k.1, v: &k.2, conv: [&k.3[0], &k.3[1], &k.3[2]], f_a: &k.4, f_b: &k.5, dt_bias: &k.6, a_log: &k.7, b: &k.8, g_a: &k.9, g_b: &k.10, o_norm: &k.11, o: &k.12 })
    }
    fn mla(&self) -> Attn<'_> {
        let m = &self.mla;
        Attn::Mla(Mla { q_a: m.0.view(), q_a_ln: &m.1, q_b: m.2.view(), kv_a: m.3.view(), kv_a_ln: &m.4, kv_b: &m.5, o: m.6.view(), ix_wq_b: &m.7, ix_wk: &m.8, ix_k_norm_w: &m.9, ix_k_norm_b: &m.10, ix_weights: &m.11, ix_ape: &m.12, ix_gate: &m.13 })
    }
    fn dense(&self) -> Ffn<'_> {
        Ffn::Dense(DenseFfn { gate: self.dense.0.view(), up: self.dense.1.view(), down: self.dense.2.view() })
    }
    fn moe(&self) -> Ffn<'_> {
        Ffn::Moe(Moe { router: &self.router, bias: &self.bias, shared_gate: self.shared.0.view(), shared_up: self.shared.1.view(), shared_down: self.shared.2.view() })
    }
    /// Layers 0..FIRST_K_DENSE_REPLACE dense over KDA (one weight set), then an MLA MoE layer and a KDA MoE layer.
    pub fn weights(&self) -> Weights<'_> {
        let mut layers = Vec::new();
        for _ in 0..FIRST_K_DENSE_REPLACE {
            layers.push(Layer { input_ln: &self.ln, post_ln: &self.ln, hc_attn: self.hc(), hc_ffn: self.hc(), attn: self.kda(), ffn: self.dense() });
        }
        layers.push(Layer { input_ln: &self.ln, post_ln: &self.ln, hc_attn: self.hc(), hc_ffn: self.hc(), attn: self.mla(), ffn: self.moe() });
        layers.push(Layer { input_ln: &self.ln, post_ln: &self.ln, hc_attn: self.hc(), hc_ffn: self.hc(), attn: self.kda(), ffn: self.moe() });
        Weights { embed: &self.embed, norm: &self.norm, lm_head: &self.lm_head, layers, mtp: None }
    }
}

/// A stream holding experts `0..TOP_K` of the two MoE layers in one block: layer 3's in its
/// own slots, layer 4's in slots `TOP_K..2·TOP_K` of the same block, the table pointing there.
pub struct Experts {
    block: Vec<u32>,
    addr: Vec<u64>,
}
impl Experts {
    pub fn new(rnd: &mut Rng) -> Self {
        let mut block = vec![0u32; layout::XBLOCK_BYTES / 4];
        let bytes: &mut [u8] = unsafe { core::slice::from_raw_parts_mut(block.as_mut_ptr() as *mut u8, layout::XBLOCK_BYTES) };
        let mut addr = vec![0u64; layout::MOE_LAYERS * E * ITEMS];
        for (li, l) in [FIRST_K_DENSE_REPLACE, FIRST_K_DENSE_REPLACE + 1].into_iter().enumerate() {
            for e in 0..TOP_K {
                let slot = li * TOP_K + e;
                let mk = |m: usize, k: usize, rnd: &mut Rng| {
                    let w: Vec<i8> = (0..m * k).map(|_| rnd.i4()).collect();
                    let s: Vec<f32> = (0..m * (k / GROUP)).map(|_| 0.002 + (rnd.next() % 100) as f32 * 2e-5).collect();
                    q4::pack(m, k, &w, &s)
                };
                let (guw, gus) = mk(2 * D_FF, H, rnd);
                let (dw, ds) = mk(H, D_FF, rnd);
                // the items at the block positions of expert `slot` of the first MoE layer
                for (c, (off, it)) in pack_items(FIRST_K_DENSE_REPLACE, slot, &guw, &gus, &dw, &ds).into_iter().enumerate() {
                    bytes[off..off + it.len()].copy_from_slice(&it);
                    addr[glm_5_3_flash::expert::item_index(l, e, c)] = off as u64;
                }
            }
        }
        let _ = (GU_ITEMS, DOWN_ITEMS);
        Experts { block, addr }
    }
    pub fn stream(&self) -> Stream<'_> {
        let bytes: &[u8] = unsafe { core::slice::from_raw_parts(self.block.as_ptr() as *const u8, layout::XBLOCK_BYTES) };
        Stream::placed(bytes, self.addr.clone(), vec![0; layout::MOE_LAYERS * E * ITEMS], Vec::new())
    }
}
