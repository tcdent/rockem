//! GLM-5.3-Flash — `zai-org/GLM-5.3-Flash`, the text model. This directory is
//! the model: the pinned config, the constants transcribed from it, and —
//! behind the `std` feature — `reference`, the f32 transcription of
//! transformers' `modeling_glm5_next.py` that bring-up diffs edge by edge
//! against the oracle (`bringup/glm/oracle.py`). The fused kernels come after
//! it, and are diffed against it and the same goldens.
//!
//! Per layer, four streams in and four out (mHC):
//!
//! ```text
//! attn_hc   streams → (pre, post, comb); x = Σ pre_i · stream_i
//! attn      linear layers: KDA over rmsnorm(x) — short conv, gated delta rule, gated norm
//!           every fourth layer: NoPE MLA over a sparse key set the indexer picks
//! mix       stream_j = post_j · attn(x) + Σ_i comb_ij · stream_i
//! ffn_hc    the same around the FFN: dense SwiGLU (layers 0–2) or the MoE —
//!           sigmoid + bias router, top 8 of 288, one shared expert
//! ```
//! then the mean of the four streams, norm, lm_head.
#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![cfg_attr(target_arch = "amdgpu", feature(abi_gpu_kernel, stdarch_amdgpu))]
#![allow(clippy::too_many_arguments)]

include!(concat!(env!("OUT_DIR"), "/config_consts.rs"));

pub const H: usize = HIDDEN_SIZE;
/// The residual streams, and the mixer's coefficients: `HC` pre, `HC` post, `HC×HC` comb.
pub const HC: usize = HC_MULT;
pub const HC_COEFS: usize = (2 + HC) * HC;
/// KDA: 64 heads of 128; q, k and v are each `KDA_QKV` wide, and the short
/// convolution runs over the three concatenated.
pub const KDA_HEADS: usize = LINEAR_NUM_HEADS;
pub const KDA_DIM: usize = LINEAR_HEAD_DIM;
pub const KDA_QKV: usize = KDA_HEADS * KDA_DIM;
pub const CONV_DIM: usize = 3 * KDA_QKV;
pub const CONV_TAPS: usize = LINEAR_CONV_KERNEL;
/// MLA without rotary: a query head is all nope, `QK_NOPE_HEAD_DIM` wide; `kv_b_proj`
/// emits, per head, that many key rows then `V_HEAD_DIM` value rows over the latent.
pub const MLA_HEADS: usize = NUM_ATTENTION_HEADS;
pub const Q_HEAD: usize = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;
pub const KVB_HEAD: usize = QK_NOPE_HEAD_DIM + V_HEAD_DIM;
/// Up to this many positions the indexer selects every visible key (all
/// `INDEX_TOPK / INDEX_KPOOL` pools plus the incomplete tail), so sparse
/// attention is dense causal attention exactly — the oracle-free test of the sparse path.
pub const DENSE_BELOW: usize = INDEX_TOPK + INDEX_KPOOL - 1;
pub const TOP_K: usize = NUM_EXPERTS_PER_TOK;
pub const E: usize = N_ROUTED_EXPERTS;
pub const D_FF: usize = MOE_INTERMEDIATE_SIZE;

const _: () = assert!(QK_ROPE_HEAD_DIM == 0);
/// The prefix every sequence begins with, `[gMASK]<sop>` (tokenizer.json's added
/// tokens): the model's own start, whose state the bundle ships — `<sop>` is the
/// sink (analysis.md §5). Positions `0..FIRST_POS` are never computed at runtime.
pub const PREFIX_IDS: [u32; 2] = [154822, 154824];
pub const FIRST_POS: usize = PREFIX_IDS.len();
/// generation_config.json: the ids that end a turn — `<|endoftext|>`, `<|user|>`, `<|observation|>`.
pub const EOS_TOKENS: [u32; 3] = [154820, 154827, 154829];
/// Sampling, from generation_config.json: temperature 1.0, top_p 0.95; top_k is the
/// default of the transformers it names (5.16), 50. The order is `common::sample`'s.
/// The seed is a constant of the model: the draw at position p uses `sample::uniform_at(SEED, p)`.
pub const SAMPLING: common::sample::Sampling = common::sample::Sampling { temperature: 1.0, top_k: 50, top_p: 0.95 };
pub const SEED: u64 = 0x726f_636b_656d_0002;
/// The most tokens one step carries (`common::gemv::TOKENS`): the card's launch width.
pub const TOKENS: usize = common::gemv::TOKENS;
/// Tokens a prefill step takes at most: the card's buffers are this deep, the rows kernels run it in
/// passes of `TOKENS`, `hc_pre` in rounds of `PRE_TOKENS`, the KDA step loops it in the kernel, the
/// host's experts group it by expert. Swept 2026-10-17 on metatron, a full step's prompt token:
/// 128 → 9.1 ms, 256 → 8.1, 512 → 7.8 (experts 4.0 / 2.9 / 2.4: no longer falling as 1/n, so the
/// card's 5.0 and the experts' compute are what is left). The kernels do not depend on it.
pub const STEP: usize = 512;

pub mod expert;
pub mod ffn;
pub mod hc;
pub mod kda;
pub mod layout;
pub mod mla;
pub mod probe;
#[cfg(feature = "std")]
pub mod construct;
pub mod weights;

const _: () = assert!(LINEAR_ATTENTION.len() == NUM_HIDDEN_LAYERS);
const _: () = assert!(DENSE_MLP.len() == NUM_HIDDEN_LAYERS);

#[cfg(feature = "std")]
pub mod reference;
