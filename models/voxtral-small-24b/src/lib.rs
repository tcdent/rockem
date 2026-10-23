//! Voxtral-Small-24B-2507 — `mistralai/Voxtral-Small-24B-2507`, the transcription
//! model (docs/queue/094). This directory is the model: the pinned config, the
//! constants transcribed from it, and — behind the `std` feature — the f32
//! transcription of transformers' `modeling_voxtral.py` that bring-up diffs
//! edge by edge against the oracle. The host kernels come after it, and are
//! diffed against it and the same goldens.
//!
//! ```text
//! encoder    one 30 s window: log-mel [128][3000] → conv1 (k3, p1) GELU → conv2 (k3, s2, p1) GELU
//!            → [1500][1280] + sinusoidal positions → 32 pre-LN layers (bidirectional attention,
//!            20 heads of 64, q scaled by 1/8, k without bias; GELU FFN 1280 → 5120 → 1280) → LN
//! projector  4 consecutive frames concatenated [375][5120] → linear GELU linear (no bias) → [375][5120]
//! decoder    Mistral Small 3.1 (`llama`): 40 layers, RMSNorm, GQA 32 query / 8 kv heads of 128 with
//!            RoPE θ 1e8, SiLU gate · up (32768) down; a placeholder id's row replaces its embedding
//! ```
#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![allow(clippy::too_many_arguments)]

include!(concat!(env!("OUT_DIR"), "/config_consts.rs"));

pub const H: usize = HIDDEN_SIZE;
pub const D_FF: usize = INTERMEDIATE_SIZE;
pub const HEADS: usize = NUM_ATTENTION_HEADS;
pub const KV_HEADS: usize = NUM_KEY_VALUE_HEADS;
/// Query heads a key/value head serves (not `q4::GROUP`, the quantization group a kernel file also imports: the clash
/// sent every query head to kv head 0 for an afternoon, 2026-10-23).
pub const KV_GROUP: usize = HEADS / KV_HEADS;
pub const Q_DIM: usize = HEADS * HEAD_DIM;
pub const KV_DIM: usize = KV_HEADS * HEAD_DIM;

/// The encoder: 20 heads of 64 over 1280, a 30 s window's 3000 mel frames halved by `conv2` to 1500 positions.
pub const ENC_H: usize = ENC_HIDDEN_SIZE;
pub const ENC_HEADS: usize = ENC_NUM_ATTENTION_HEADS;
pub const ENC_POSITIONS: usize = MAX_SOURCE_POSITIONS;
pub const MEL_FRAMES: usize = 2 * ENC_POSITIONS;
/// Encoder positions concatenated into one row for the projector; rows a window becomes.
pub const DOWNSAMPLE: usize = ENC_INTERMEDIATE_SIZE / ENC_H;
pub const ROWS_PER_WINDOW: usize = ENC_POSITIONS / DOWNSAMPLE;
/// The id each row fills (`[AUDIO]`).
pub const AUDIO_ID: u32 = AUDIO_TOKEN_ID as u32;

const _: () = assert!(HEADS % KV_HEADS == 0 && ENC_HEADS * ENC_HEAD_DIM == ENC_H && DOWNSAMPLE * ENC_H == ENC_INTERMEDIATE_SIZE && ENC_INTERMEDIATE_SIZE == H && ENC_POSITIONS % DOWNSAMPLE == 0);
const _: () = assert!(ROWS_PER_WINDOW == 375 && MEL_FRAMES == 3000 && KV_GROUP == 4);

/// tekken: `<s>` 1, `</s>` 2, `[INST]` 3, `[/INST]` 4 (the instruct tokenizer's control ids; `[AUDIO]` 24).
pub const BOS_TOKEN: u32 = 1;
pub const EOS_TOKEN: u32 = 2;
pub const PREFIX_IDS: &[u32] = &[BOS_TOKEN];
pub const EOS_TOKENS: &[u32] = &[EOS_TOKEN];

pub mod layout;
#[cfg(feature = "std")]
pub mod weights;
#[cfg(feature = "std")]
pub mod decoder;
#[cfg(feature = "std")]
pub mod encoder;
