//! This model's bundle formats. The dense weights ship exactly as the
//! checkpoint stores them (tcdent, 2026-10-15: native precision first): bf16,
//! f32, and FP8 e4m3 with an f32 scale per 128×128 block, one file, every
//! tensor at a 64-byte boundary in the order of the tables below. The routed
//! experts are a stream of fixed-size `q4g128` records in (layer, expert)
//! order for the MoE layers, as V2-Lite's are. The prefix state replaces the
//! sink: the per-position rows of the `FIRST_POS` prefix positions, then the
//! recurrent layers' snapshot after them.
//!
//! Both the bundle builder and the loader walk these tables, so the file has
//! one definition; the names are the checkpoint's own, relative to the layer.

use crate::*;
use common::q4::GROUP;

// --- the routed experts ---
pub const GATE_UP_W: usize = 2 * D_FF * H / 2;
pub const GATE_UP_S: usize = 2 * D_FF * (H / GROUP); // f32 count
pub const DOWN_W: usize = H * D_FF / 2;
pub const DOWN_S: usize = H * (D_FF / GROUP); // f32 count
pub const RECORD: usize = GATE_UP_W + 4 * GATE_UP_S + DOWN_W + 4 * DOWN_S;
pub const LAYER: usize = E * RECORD;
pub const MOE_LAYERS: usize = NUM_HIDDEN_LAYERS - FIRST_K_DENSE_REPLACE;
pub const STREAM: usize = MOE_LAYERS * LAYER;
pub const RECIPE_RTN: &str = "experts:q4g128-a8-rtn dense:native";

const _: () = assert!(RECORD % 64 == 0 && H % GROUP == 0 && D_FF % GROUP == 0);

/// Byte range of expert `e` in MoE layer `l` within the stream.
#[inline(always)]
pub const fn record(l: usize, e: usize) -> core::ops::Range<usize> {
    let start = (l - FIRST_K_DENSE_REPLACE) * LAYER + e * RECORD;
    start..start + RECORD
}

// --- the dense weights, native ---
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Bf16,
    F32,
    /// `rows·cols` e4m3 bytes, then `ceil(rows/128)·ceil(cols/128)` f32 scales (`<name>_scale_inv`).
    Fp8,
}
#[derive(Clone, Copy, Debug)]
pub struct Tensor {
    pub name: &'static str,
    pub kind: Kind,
    pub rows: usize,
    pub cols: usize,
}
pub const BLOCK: usize = 128;
pub const ALIGN: usize = 64;

impl Tensor {
    const fn new(name: &'static str, kind: Kind, rows: usize, cols: usize) -> Self {
        Tensor { name, kind, rows, cols }
    }
    pub const fn len(&self) -> usize {
        self.rows * self.cols
    }
    /// Bytes of the weights alone (the scales follow them for `Fp8`).
    pub const fn weight_bytes(&self) -> usize {
        match self.kind {
            Kind::Bf16 => 2 * self.len(),
            Kind::F32 => 4 * self.len(),
            Kind::Fp8 => self.len(),
        }
    }
    pub const fn scales(&self) -> usize {
        match self.kind {
            Kind::Fp8 => self.rows.div_ceil(BLOCK) * self.cols.div_ceil(BLOCK),
            _ => 0,
        }
    }
    pub const fn bytes(&self) -> usize {
        self.weight_bytes() + 4 * self.scales()
    }
    /// What the tensor occupies in the file: its bytes, padded to the alignment.
    pub const fn slot(&self) -> usize {
        self.bytes().div_ceil(ALIGN) * ALIGN
    }
}

use Kind::*;
pub const HEAD: [Tensor; 3] = [
    Tensor::new("model.language_model.embed_tokens.weight", Bf16, VOCAB_SIZE, H),
    Tensor::new("model.language_model.norm.weight", Bf16, 1, H),
    Tensor::new("lm_head.weight", Bf16, VOCAB_SIZE, H),
];
/// Every layer: the two norms and the two hyper-connections.
pub const COMMON: [Tensor; 8] = [
    Tensor::new("input_layernorm.weight", Bf16, 1, H),
    Tensor::new("post_attention_layernorm.weight", Bf16, 1, H),
    Tensor::new("hc_attn_fn", Bf16, HC_COEFS, HC * H),
    Tensor::new("hc_attn_base", F32, 1, HC_COEFS),
    Tensor::new("hc_attn_scale", F32, 1, 3),
    Tensor::new("hc_ffn_fn", Bf16, HC_COEFS, HC * H),
    Tensor::new("hc_ffn_base", F32, 1, HC_COEFS),
    Tensor::new("hc_ffn_scale", F32, 1, 3),
];
pub const KDA: [Tensor; 15] = [
    Tensor::new("self_attn.q_proj.weight", Bf16, KDA_QKV, H),
    Tensor::new("self_attn.k_proj.weight", Bf16, KDA_QKV, H),
    Tensor::new("self_attn.v_proj.weight", Bf16, KDA_QKV, H),
    Tensor::new("self_attn.q_conv1d.weight", Bf16, KDA_QKV, CONV_TAPS),
    Tensor::new("self_attn.k_conv1d.weight", Bf16, KDA_QKV, CONV_TAPS),
    Tensor::new("self_attn.v_conv1d.weight", Bf16, KDA_QKV, CONV_TAPS),
    Tensor::new("self_attn.f_a_proj.weight", Bf16, KDA_DIM, H),
    Tensor::new("self_attn.f_b_proj.weight", Bf16, KDA_QKV, KDA_DIM),
    Tensor::new("self_attn.dt_bias", F32, 1, KDA_QKV),
    Tensor::new("self_attn.A_log", F32, 1, KDA_HEADS),
    Tensor::new("self_attn.b_proj.weight", Bf16, KDA_HEADS, H),
    Tensor::new("self_attn.g_a_proj.weight", Bf16, KDA_DIM, H),
    Tensor::new("self_attn.g_b_proj.weight", Bf16, KDA_QKV, KDA_DIM),
    Tensor::new("self_attn.o_norm.weight", Bf16, 1, KDA_DIM),
    Tensor::new("self_attn.o_proj.weight", Bf16, H, KDA_QKV),
];
pub const MLA: [Tensor; 14] = [
    Tensor::new("self_attn.q_a_proj.weight", Fp8, Q_LORA_RANK, H),
    Tensor::new("self_attn.q_a_layernorm.weight", Bf16, 1, Q_LORA_RANK),
    Tensor::new("self_attn.q_b_proj.weight", Fp8, MLA_HEADS * Q_HEAD, Q_LORA_RANK),
    Tensor::new("self_attn.kv_a_proj_with_mqa.weight", Fp8, KV_LORA_RANK + QK_ROPE_HEAD_DIM, H),
    Tensor::new("self_attn.kv_a_layernorm.weight", Bf16, 1, KV_LORA_RANK),
    Tensor::new("self_attn.kv_b_proj.weight", Bf16, MLA_HEADS * KVB_HEAD, KV_LORA_RANK),
    Tensor::new("self_attn.o_proj.weight", Fp8, H, MLA_HEADS * V_HEAD_DIM),
    Tensor::new("self_attn.indexer.wq_b.weight", Bf16, INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA_RANK),
    Tensor::new("self_attn.indexer.wk.weight", Bf16, INDEX_HEAD_DIM, H),
    Tensor::new("self_attn.indexer.k_norm.weight", Bf16, 1, INDEX_HEAD_DIM),
    Tensor::new("self_attn.indexer.k_norm.bias", Bf16, 1, INDEX_HEAD_DIM),
    Tensor::new("self_attn.indexer.weights_proj.weight", Bf16, INDEX_N_HEADS, H),
    Tensor::new("self_attn.indexer.index_kpool_compress_ape", Bf16, INDEX_KPOOL, INDEX_HEAD_DIM),
    Tensor::new("self_attn.indexer.index_kpool_compress_gate", Bf16, INDEX_HEAD_DIM, H),
];
pub const DENSE_FFN: [Tensor; 3] = [
    Tensor::new("mlp.gate_proj.weight", Fp8, INTERMEDIATE_SIZE, H),
    Tensor::new("mlp.up_proj.weight", Fp8, INTERMEDIATE_SIZE, H),
    Tensor::new("mlp.down_proj.weight", Fp8, H, INTERMEDIATE_SIZE),
];
pub const MOE: [Tensor; 5] = [
    Tensor::new("mlp.gate.weight", Bf16, E, H),
    Tensor::new("mlp.gate.e_score_correction_bias", F32, 1, E),
    Tensor::new("mlp.shared_experts.gate_proj.weight", Fp8, N_SHARED_EXPERTS * D_FF, H),
    Tensor::new("mlp.shared_experts.up_proj.weight", Fp8, N_SHARED_EXPERTS * D_FF, H),
    Tensor::new("mlp.shared_experts.down_proj.weight", Fp8, H, N_SHARED_EXPERTS * D_FF),
];

pub const fn attn(l: usize) -> &'static [Tensor] {
    if LINEAR_ATTENTION[l] { &KDA } else { &MLA }
}
pub const fn ffn(l: usize) -> &'static [Tensor] {
    if DENSE_MLP[l] { &DENSE_FFN } else { &MOE }
}
pub const fn section(ts: &[Tensor]) -> usize {
    let mut b = 0;
    let mut i = 0;
    while i < ts.len() {
        b += ts[i].slot();
        i += 1;
    }
    b
}
/// Byte offset of tensor `i` within its section.
pub const fn within(ts: &[Tensor], i: usize) -> usize {
    section(ts.split_at(i).0)
}
pub const HEAD_BYTES: usize = section(&HEAD);
/// Byte offset of layer `l`: its `COMMON`, then `attn(l)`, then `ffn(l)`.
pub const fn layer(l: usize) -> usize {
    let mut off = HEAD_BYTES;
    let mut i = 0;
    while i < l {
        off += section(&COMMON) + section(attn(i)) + section(ffn(i));
        i += 1;
    }
    off
}
pub const DENSE_BYTES: usize = layer(NUM_HIDDEN_LAYERS);

/// A tensor's place in the file: (byte offset, the tensor), by layer and checkpoint name
/// (`None` for the head's). Construction only: a linear scan of the layer's tables.
pub fn locate(layer_: Option<usize>, name: &str) -> Option<(usize, Tensor)> {
    match layer_ {
        None => HEAD.iter().enumerate().find(|(_, t)| t.name == name).map(|(i, t)| (within(&HEAD, i), *t)),
        Some(l) => {
            let base = layer(l);
            let sections: [&[Tensor]; 3] = [&COMMON, attn(l), ffn(l)];
            let mut off = base;
            for s in sections {
                if let Some((i, t)) = s.iter().enumerate().find(|(_, t)| t.name == name) {
                    return Some((off + within(s, i), *t));
                }
                off += section(s);
            }
            None
        }
    }
}

// --- files ---
pub const DENSE_FILE: &str = "dense.bin";
pub const EXPERTS_FILE: &str = "experts.q4g128.bin";
pub const VOCAB_FILE: &str = "vocab.bin";
/// The gfx1100 code object (`models/glm-5.3-flash/gpu`), carried as data.
pub const KERNELS_FILE: &str = "kernels.gfx1100.elf";
pub const TOKENIZER_FILES: [&str; 2] = ["tokenizer.json", "tokenizer_config.json"];
pub const CHAT_TEMPLATE_FILE: &str = "chat_template.jinja";
/// The prefix's state, f32: `[FIRST_POS][POSITION]` rows, then `[SNAPSHOT]`.
/// Computed once at bundle build at full precision (the f32 reference); every
/// sequence is seeded from it and computed from `FIRST_POS`.
pub const PREFIX_FILE: &str = "prefix.f32";

// --- sequence state ---
pub const fn count(flags: &[bool], want: bool) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < flags.len() {
        if flags[i] == want {
            n += 1;
        }
        i += 1;
    }
    n
}
pub const KDA_LAYERS: usize = count(&LINEAR_ATTENTION, true);
pub const MLA_LAYERS: usize = count(&LINEAR_ATTENTION, false);
/// One MLA layer's per-position row: the normed latent, the indexer's key, the indexer's pool gate.
pub const MLA_ROW: usize = KV_LORA_RANK + INDEX_HEAD_DIM + INDEX_HEAD_DIM;
/// One position's rows, every MLA layer's in layer order: `[MLA_LAYERS][MLA_ROW]`.
pub const POSITION: usize = MLA_LAYERS * MLA_ROW;
/// One KDA layer's state: the convolution's last `CONV_TAPS - 1` inputs, then the `[head][k][v]` memory.
pub const KDA_WINDOW: usize = (CONV_TAPS - 1) * CONV_DIM;
pub const KDA_MEMORY: usize = KDA_HEADS * KDA_DIM * KDA_DIM;
pub const KDA_STATE: usize = KDA_WINDOW + KDA_MEMORY;
/// The sequence state that is not per position: every KDA layer's, in layer order.
pub const SNAPSHOT: usize = KDA_LAYERS * KDA_STATE;
pub const PREFIX_LEN: usize = FIRST_POS * POSITION + SNAPSHOT;

const _: () = assert!(KDA_LAYERS + MLA_LAYERS == NUM_HIDDEN_LAYERS);
const _: () = assert!(count(&DENSE_MLP, true) == FIRST_K_DENSE_REPLACE);
