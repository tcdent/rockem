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

const _: () = assert!(LINEAR_ATTENTION.len() == NUM_HIDDEN_LAYERS);
const _: () = assert!(DENSE_MLP.len() == NUM_HIDDEN_LAYERS);

#[cfg(feature = "std")]
pub mod reference;
