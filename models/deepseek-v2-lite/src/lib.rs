//! DeepSeek-V2-Lite — `deepseek-ai/DeepSeek-V2-Lite` at the revision pinned in
//! `checkpoint.toml`. This directory is the whole model: the pinned config and
//! oracle file, the constants transcribed from them, the fused per-layer
//! kernels (`attn`, `ffn`, `expert`), the bundle record layout, and — behind
//! the `std` feature — construction: weights, tables, state, the decode step.
//!
//! Transcribed from `modeling_deepseek.py` (`DeepseekV2DecoderLayer`,
//! `DeepseekV2Attention`, `DeepseekV2MoE`, `MoEGate`). Per layer:
//!
//! ```text
//! attn::step   hn = rmsnorm(h) → q = W_q hn → kva = W_kva hn → c = rmsnorm(latent)
//!              → rope(q_pe, k_pe) [DeepSeek de-interleaved] → cache[pos] = c | k_pe
//!              → absorbed MLA over the cache → h += W_o · heads
//! ffn::dense   (layer 0)   h += down(silu(gate hn) ⊙ up hn)
//! ffn::moe     (layers ≥1) hn = rmsnorm(h) → softmax(W_g hn) top-6
//!              → h += Σ_k w_k expert_{e_k}(hn) + shared(hn)
//! ```
//! then norm, lm_head, argmax. Prefill is the decode step applied to each
//! prompt token in order — at batch 1 the causal forward and the sequential
//! one are the same math.
#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![cfg_attr(target_arch = "amdgpu", feature(abi_gpu_kernel, stdarch_amdgpu))]
#![allow(clippy::too_many_arguments)]

include!(concat!(env!("OUT_DIR"), "/config_consts.rs"));

pub const H: usize = HIDDEN_SIZE;
pub const HEADS: usize = NUM_ATTENTION_HEADS;
/// Per-head query width: 128 nope + 64 rope. `q_proj` emits `HEADS * Q_HEAD_DIM`.
pub const Q_HEAD_DIM: usize = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;
pub const Q_DIM: usize = HEADS * Q_HEAD_DIM; // 3072
/// `kv_a_proj_with_mqa` emits the 512-wide latent followed by the 64-wide shared rope key.
pub const KVA_DIM: usize = KV_LORA_RANK + QK_ROPE_HEAD_DIM; // 576
/// `kv_b_proj` emits, per head, 128 nope-key rows then 128 value rows over the latent.
pub const KVB_HEAD_DIM: usize = QK_NOPE_HEAD_DIM + V_HEAD_DIM; // 256
pub const KVB_DIM: usize = HEADS * KVB_HEAD_DIM; // 4096
pub const O_IN_DIM: usize = HEADS * V_HEAD_DIM; // 2048
/// The shared experts are one MLP of `N_SHARED_EXPERTS * MOE_INTERMEDIATE_SIZE`.
pub const SHARED_FF: usize = N_SHARED_EXPERTS * MOE_INTERMEDIATE_SIZE; // 2816
/// One cache row per token per layer: the post-norm latent and the roped shared key.
pub const KV_ROW: usize = KV_LORA_RANK + QK_ROPE_HEAD_DIM; // 576
pub const TOP_K: usize = NUM_EXPERTS_PER_TOK;
pub const D_FF: usize = MOE_INTERMEDIATE_SIZE;
/// From generation_config.json / tokenizer added tokens: begin and end of sentence.
pub const BOS_TOKEN: usize = 100000;
pub const EOS_TOKEN: usize = 100001;
/// The first position the forward computes. Position 0 is always BOS, and BOS
/// has no context, so its state is a constant of the model: the bundle ships
/// its cache rows (`layout::SINK_FILE`), attention is seeded from them, and
/// every sequence is computed from here. See analysis.md §5.
pub const FIRST_POS: usize = 1;
/// The slot names of each kernel's statistics (`common::stats`), for the host's report.
pub fn stat_names(kernel: &str) -> [&str; 4] {
    match kernel {
        "route" => ffn::gfx11_consts::ROUTE_STATS,
        "dense_down" | "shared_down" => ffn::gfx11_consts::DOWN_STATS,
        "sample" => ffn::gfx11_consts::SAMPLE_STATS,
        "attn_mla" => attn::gfx11_consts::MLA_STATS,
        _ => ["", "", "", ""],
    }
}
/// Sampling, from generation_config.json: `temperature: 0.3`, `top_p: 0.95`,
/// and `top_k: 50`, which the config leaves to the default of the transformers
/// it names (4.39.3), where the order is temperature, top_k, top_p, then the
/// multinomial draw (`common::sample`). The seed is a constant of the model
/// too (tcdent, 2026-09-17: "bake seed and temp with the model config"): the
/// draw at position p uses `sample::uniform_at(SEED, p)`, so the same
/// transcript gives the same tokens on every build of it, however its prefix
/// arrived.
pub const SAMPLING: Sampling = Sampling { temperature: 0.3, top_k: 50, top_p: 0.95 };
/// YaRN's inverse frequencies, the oracle's own f32 values
/// (`bringup/rope_tables.py`, `DeepseekV2YarnRotaryEmbedding`). Computing
/// them from the config in f64 and rounding differed from the oracle's f32
/// arithmetic by an ulp in 12 of the 32, which is nothing at the goldens'
/// positions and 0.0038 in a cosine at position 100,000 (2026-09-19). The
/// table is a constant of the model, checked against the oracle in
/// `construct::tests`.
pub const ROPE_INV_FREQ: [f32; QK_ROPE_HEAD_DIM / 2] = [
    1.0000000e+00, 7.4989420e-01, 5.6234133e-01, 4.2169651e-01,
    3.1622776e-01, 2.3713736e-01, 1.7782794e-01, 1.3335215e-01,
    1.0000000e-01, 7.4989416e-02, 5.6234129e-02, 3.9006926e-02,
    2.6879361e-02, 1.8378144e-02, 1.2447956e-02, 8.3345091e-03,
    5.5000004e-03, 3.5619973e-03, 2.2493652e-03, 1.3705135e-03,
    7.9056941e-04, 4.1499041e-04, 1.7782794e-04, 3.3338034e-05,
    2.4999999e-05, 1.8747354e-05, 1.40585325e-05, 1.0542412e-05,
    7.9056945e-06, 5.9284343e-06, 4.4456983e-06, 3.3338035e-06,
];
/// The most tokens one step carries (`common::gemv::TOKENS`): the card's
/// launches and LDS are sized by it and the CPU's loops run to it.
pub const TOKENS: usize = common::gemv::TOKENS;
pub const SEED: u64 = 0x726f_636b_656d_0001;

const _: () = assert!(Q_DIM == 3072 && KVA_DIM == 576 && KVB_DIM == 4096 && H == 2048);
const _: () = assert!(!NORM_TOPK_PROB, "the router assumes unnormalized top-k weights");

pub mod attn;
pub mod expert;
pub mod ffn;
pub mod layout;

#[cfg(feature = "std")]
pub mod construct;
pub use attn::Attention;
pub use common::sample::Sampling;
#[cfg(feature = "std")]
pub use construct::{Bf16Experts, CpuAttention, CpuExperts, CpuFfn, CpuModel, Ffn, Model, Rope, State, Weights};
pub use ffn::{FeedForward, FfnHead, Head, Routed};
