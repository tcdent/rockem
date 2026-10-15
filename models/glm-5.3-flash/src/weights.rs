//! The dense weights as slices into the bundle's dense file (`layout`), by
//! layer, in the types the kernels take: bf16 rows as `&[u16]`, FP8 rows with
//! their block scales as `Fp8`, and the small vectors (norms, biases, the
//! hyper-connection coefficients' base and scale, the pool offsets) widened
//! to f32 once. The routed experts are not here: they belong to a `Routed`
//! realization over the expert stream. `std` only; construction only.

use crate::layout::{self, Kind, Tensor};
use crate::*;
use common::gemv::widen;

/// FP8 e4m3 rows with one f32 scale per 128×128 block (`common::gemv::fp8`).
#[derive(Clone, Copy)]
pub struct Fp8<'a> {
    pub w: &'a [u8],
    pub s: &'a [f32],
}

pub struct Hc<'a> {
    pub fn_: &'a [u16], // [HC_COEFS][HC·H]
    pub base: &'a [f32],
    pub scale: &'a [f32],
}

pub struct Kda<'a> {
    pub q: &'a [u16], // [KDA_QKV][H]
    pub k: &'a [u16],
    pub v: &'a [u16],
    /// q's, k's, v's taps: each `[KDA_QKV][CONV_TAPS]`.
    pub conv: [&'a [u16]; 3],
    pub f_a: &'a [u16], // [KDA_DIM][H]
    pub f_b: &'a [u16], // [KDA_QKV][KDA_DIM]
    pub dt_bias: &'a [f32],
    pub a_log: &'a [f32],
    pub b: &'a [u16], // [KDA_HEADS][H]
    pub g_a: &'a [u16],
    pub g_b: &'a [u16],
    pub o_norm: Vec<f32>,
    pub o: &'a [u16], // [H][KDA_QKV]
}

pub struct Mla<'a> {
    pub q_a: Fp8<'a>, // [Q_LORA_RANK][H]
    pub q_a_ln: Vec<f32>,
    pub q_b: Fp8<'a>, // [MLA_HEADS·Q_HEAD][Q_LORA_RANK]
    pub kv_a: Fp8<'a>, // [KV_LORA_RANK][H]
    pub kv_a_ln: Vec<f32>,
    /// `[MLA_HEADS][KVB_HEAD][KV_LORA_RANK]`: per head the key rows, then the value rows.
    pub kv_b: &'a [u16],
    pub o: Fp8<'a>, // [H][MLA_HEADS·V_HEAD_DIM]
    pub ix_wq_b: &'a [u16], // [INDEX_N_HEADS·INDEX_HEAD_DIM][Q_LORA_RANK]
    pub ix_wk: &'a [u16],   // [INDEX_HEAD_DIM][H]
    pub ix_k_norm_w: Vec<f32>,
    pub ix_k_norm_b: Vec<f32>,
    pub ix_weights: &'a [u16], // [INDEX_N_HEADS][H]
    pub ix_ape: Vec<f32>,      // [INDEX_KPOOL][INDEX_HEAD_DIM]
    pub ix_gate: &'a [u16],    // [INDEX_HEAD_DIM][H]
}

pub enum Attn<'a> {
    Kda(Kda<'a>),
    Mla(Mla<'a>),
}

pub struct DenseFfn<'a> {
    pub gate: Fp8<'a>, // [INTERMEDIATE_SIZE][H]
    pub up: Fp8<'a>,
    pub down: Fp8<'a>, // [H][INTERMEDIATE_SIZE]
}

pub struct Moe<'a> {
    pub router: &'a [u16], // [E][H]
    pub bias: &'a [f32],   // [E]
    pub shared_gate: Fp8<'a>, // [D_FF][H]
    pub shared_up: Fp8<'a>,
    pub shared_down: Fp8<'a>, // [H][D_FF]
}

pub enum Ffn<'a> {
    Dense(DenseFfn<'a>),
    Moe(Moe<'a>),
}

pub struct Layer<'a> {
    pub input_ln: Vec<f32>,
    pub post_ln: Vec<f32>,
    pub hc_attn: Hc<'a>,
    pub hc_ffn: Hc<'a>,
    pub attn: Attn<'a>,
    pub ffn: Ffn<'a>,
}

pub struct Weights<'a> {
    pub embed: &'a [u16], // [VOCAB_SIZE][H]
    pub norm: Vec<f32>,
    pub lm_head: &'a [u16], // [VOCAB_SIZE][H]
    pub layers: Vec<Layer<'a>>,
}

/// A tensor's bytes in the file, by layer and name.
struct At<'a> {
    dense: &'a [u8],
}
impl<'a> At<'a> {
    fn slot(&self, l: Option<usize>, name: &str) -> (&'a [u8], Tensor) {
        let (off, t) = layout::locate(l, name).unwrap_or_else(|| panic!("layout has no {name} in layer {l:?}"));
        (&self.dense[off..off + t.bytes()], t)
    }
    fn bf16(&self, l: Option<usize>, name: &str) -> &'a [u16] {
        let (b, t) = self.slot(l, name);
        assert_eq!(t.kind, Kind::Bf16, "{name}");
        let (p, m, q) = unsafe { b.align_to::<u16>() };
        assert!(p.is_empty() && q.is_empty(), "{name}: alignment");
        m
    }
    fn f32(&self, l: Option<usize>, name: &str) -> &'a [f32] {
        let (b, t) = self.slot(l, name);
        assert_eq!(t.kind, Kind::F32, "{name}");
        let (p, m, q) = unsafe { b.align_to::<f32>() };
        assert!(p.is_empty() && q.is_empty(), "{name}: alignment");
        m
    }
    fn fp8(&self, l: Option<usize>, name: &str) -> Fp8<'a> {
        let (b, t) = self.slot(l, name);
        assert_eq!(t.kind, Kind::Fp8, "{name}");
        let (w, s) = b.split_at(t.weight_bytes());
        let (p, s, q) = unsafe { s.align_to::<f32>() };
        assert!(p.is_empty() && q.is_empty() && s.len() == t.scales(), "{name}: scales");
        Fp8 { w, s }
    }
    fn widened(&self, l: Option<usize>, name: &str) -> Vec<f32> {
        self.bf16(l, name).iter().map(|&v| widen(v)).collect()
    }
}

impl<'a> Weights<'a> {
    /// The bundle's dense file, as loaded (its base must be 4-byte aligned; the slots are 64-byte).
    pub fn from_dense(dense: &'a [u8]) -> Self {
        assert_eq!(dense.len(), layout::DENSE_BYTES, "dense file size");
        assert_eq!(dense.as_ptr() as usize % ALIGN_MIN, 0, "dense file alignment");
        let at = At { dense };
        let mut layers = Vec::with_capacity(NUM_HIDDEN_LAYERS);
        for l in 0..NUM_HIDDEN_LAYERS {
            let l_ = Some(l);
            let hc = |which: &str| Hc { fn_: at.bf16(l_, &format!("hc_{which}_fn")), base: at.f32(l_, &format!("hc_{which}_base")), scale: at.f32(l_, &format!("hc_{which}_scale")) };
            let attn = if LINEAR_ATTENTION[l] {
                Attn::Kda(Kda {
                    q: at.bf16(l_, "self_attn.q_proj.weight"),
                    k: at.bf16(l_, "self_attn.k_proj.weight"),
                    v: at.bf16(l_, "self_attn.v_proj.weight"),
                    conv: [at.bf16(l_, "self_attn.q_conv1d.weight"), at.bf16(l_, "self_attn.k_conv1d.weight"), at.bf16(l_, "self_attn.v_conv1d.weight")],
                    f_a: at.bf16(l_, "self_attn.f_a_proj.weight"),
                    f_b: at.bf16(l_, "self_attn.f_b_proj.weight"),
                    dt_bias: at.f32(l_, "self_attn.dt_bias"),
                    a_log: at.f32(l_, "self_attn.A_log"),
                    b: at.bf16(l_, "self_attn.b_proj.weight"),
                    g_a: at.bf16(l_, "self_attn.g_a_proj.weight"),
                    g_b: at.bf16(l_, "self_attn.g_b_proj.weight"),
                    o_norm: at.widened(l_, "self_attn.o_norm.weight"),
                    o: at.bf16(l_, "self_attn.o_proj.weight"),
                })
            } else {
                Attn::Mla(Mla {
                    q_a: at.fp8(l_, "self_attn.q_a_proj.weight"),
                    q_a_ln: at.widened(l_, "self_attn.q_a_layernorm.weight"),
                    q_b: at.fp8(l_, "self_attn.q_b_proj.weight"),
                    kv_a: at.fp8(l_, "self_attn.kv_a_proj_with_mqa.weight"),
                    kv_a_ln: at.widened(l_, "self_attn.kv_a_layernorm.weight"),
                    kv_b: at.bf16(l_, "self_attn.kv_b_proj.weight"),
                    o: at.fp8(l_, "self_attn.o_proj.weight"),
                    ix_wq_b: at.bf16(l_, "self_attn.indexer.wq_b.weight"),
                    ix_wk: at.bf16(l_, "self_attn.indexer.wk.weight"),
                    ix_k_norm_w: at.widened(l_, "self_attn.indexer.k_norm.weight"),
                    ix_k_norm_b: at.widened(l_, "self_attn.indexer.k_norm.bias"),
                    ix_weights: at.bf16(l_, "self_attn.indexer.weights_proj.weight"),
                    ix_ape: at.widened(l_, "self_attn.indexer.index_kpool_compress_ape"),
                    ix_gate: at.bf16(l_, "self_attn.indexer.index_kpool_compress_gate"),
                })
            };
            let ffn = if DENSE_MLP[l] {
                Ffn::Dense(DenseFfn { gate: at.fp8(l_, "mlp.gate_proj.weight"), up: at.fp8(l_, "mlp.up_proj.weight"), down: at.fp8(l_, "mlp.down_proj.weight") })
            } else {
                Ffn::Moe(Moe {
                    router: at.bf16(l_, "mlp.gate.weight"),
                    bias: at.f32(l_, "mlp.gate.e_score_correction_bias"),
                    shared_gate: at.fp8(l_, "mlp.shared_experts.gate_proj.weight"),
                    shared_up: at.fp8(l_, "mlp.shared_experts.up_proj.weight"),
                    shared_down: at.fp8(l_, "mlp.shared_experts.down_proj.weight"),
                })
            };
            layers.push(Layer { input_ln: at.widened(l_, "input_layernorm.weight"), post_ln: at.widened(l_, "post_attention_layernorm.weight"), hc_attn: hc("attn"), hc_ffn: hc("ffn"), attn, ffn });
        }
        Weights {
            embed: at.bf16(None, "model.language_model.embed_tokens.weight"),
            norm: at.widened(None, "model.language_model.norm.weight"),
            lm_head: at.bf16(None, "lm_head.weight"),
            layers,
        }
    }
}

/// The file's base alignment the slices need: f32 scales and bf16 rows are read in place.
const ALIGN_MIN: usize = 4;
