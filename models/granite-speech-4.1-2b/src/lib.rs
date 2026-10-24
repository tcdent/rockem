//! Granite Speech 4.1 2B — `ibm-granite/granite-speech-4.1-2b`, the transcription model chosen for word
//! accuracy on the CPU (docs/queue/096). This directory is the model: the pinned config, the constants
//! transcribed from it, and — behind the `std` feature — the host transcription of transformers'
//! `modeling_granite_speech.py` that bring-up diffs edge by edge against the oracle.
//!
//! ```text
//! features   16 kHz → 80 log-mels (n_fft 512, window 400, hop 160; log10, the clip's max − 8 floor, /4 + 1),
//!            two frames stacked: [T][160] at 20 ms
//! encoder    input_linear 160 → 1024, 16 Conformer blocks (½·FF, block attention over 200 frames with Shaw
//!            relative positions, convolution module k15 with GLU and a folded batch norm, ½·FF, a post norm);
//!            after block 8 the self-conditioned CTC: h += out_mid(softmax(out(h)))
//! projector  windows of 15 frames → 3 learned queries through a 2-layer BLIP-2 Q-Former (self-attention,
//!            cross-attention to the window, GELU FFN, post-LN at 1e-12) → linear 1024 → 2048: 10 Hz rows
//! decoder    Granite 4.0 1B: 40 layers, RMSNorm, GQA 16 query / 4 kv heads of 128, RoPE θ 1e4, SiLU
//!            gate · up (4096) down, with three scalar multipliers (embedding ×12, attention 1/128 in place of
//!            1/√d, residual ×0.22; the config's logits_scaling the served composite model does not apply);
//!            a placeholder id's row replaces its embedding, then ×12 as well
//! ```
#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![allow(clippy::too_many_arguments)]

include!(concat!(env!("OUT_DIR"), "/config_consts.rs"));

pub const H: usize = HIDDEN_SIZE;
pub const D_FF: usize = INTERMEDIATE_SIZE;
pub const HEADS: usize = NUM_ATTENTION_HEADS;
pub const KV_HEADS: usize = NUM_KEY_VALUE_HEADS;
/// The config gives no head_dim: hidden / heads.
pub const HEAD_DIM: usize = H / HEADS;
/// Query heads a key/value head serves (not a quantization group).
pub const KV_GROUP: usize = HEADS / KV_HEADS;
pub const Q_DIM: usize = HEADS * HEAD_DIM;
pub const KV_DIM: usize = KV_HEADS * HEAD_DIM;
/// The vocabulary padded to whole lm_head items of every run (`layout::LM_HEAD`): rows past `VOCAB_SIZE` are zero and
/// their logits are set to −∞ by the decoder.
pub const VOCAB_PAD: usize = VOCAB_SIZE.div_ceil(256) * 256;

/// The encoder.
pub const ENC_H: usize = ENC_HIDDEN_SIZE;
pub const ENC_HEADS: usize = ENC_NUM_ATTENTION_HEADS;
pub const ENC_FF: usize = ENC_H * ENC_FF_MULT;
/// The convolution module's width after the GLU (`up_conv` makes twice this).
pub const ENC_CONV: usize = ENC_H * ENC_CONV_EXPANSION;
/// Shaw relative positions: distances clipped to ±`ENC_CONTEXT`, the table `2 · ENC_MAX_POS_EMB + 1` rows.
pub const ENC_REL_ROWS: usize = 2 * ENC_MAX_POS_EMB + 1;
/// Log-mel bins a 10 ms frame; two make an encoder frame.
pub const NUM_MEL_BINS: usize = ENC_INPUT_DIM / 2;
/// The projector: queries a window.
pub const QUERIES: usize = WINDOW_SIZE / DOWNSAMPLE_RATE;
pub const PROJ_H: usize = PROJ_HIDDEN_SIZE;
pub const PROJ_HEAD_DIM: usize = PROJ_H / PROJ_NUM_ATTENTION_HEADS;

/// The wire's window: 3 s of 10 ms log-mel frames, `[NUM_MEL_BINS][MEL_FRAMES]` bin-major, which is 150 encoder
/// frames = 10 projector windows = 30 rows. The encoder runs over every window of a request at once (its attention
/// blocks and convolutions span them), so the windows are queued and encoded when the first placeholder arrives.
pub const MEL_FRAMES: usize = 300;
pub const ENC_FRAMES_PER_WINDOW: usize = MEL_FRAMES / 2;
pub const ROWS_PER_WINDOW: usize = ENC_FRAMES_PER_WINDOW / WINDOW_SIZE * QUERIES;
/// The id each row fills (`<|audio|>`).
pub const AUDIO_ID: u32 = AUDIO_TOKEN_ID as u32;

const _: () = assert!(HEADS % KV_HEADS == 0 && HEAD_DIM == 128 && ENC_HEADS * ENC_HEAD_DIM == ENC_H && ENC_INPUT_DIM % 2 == 0);
const _: () = assert!(WINDOW_SIZE % DOWNSAMPLE_RATE == 0 && ENC_FRAMES_PER_WINDOW % WINDOW_SIZE == 0 && ROWS_PER_WINDOW == 30 && QUERIES == 3);
const _: () = assert!(ENC_CONTEXT <= ENC_MAX_POS_EMB && PROJ_H == ENC_H && VOCAB_PAD % 256 == 0);

/// The GPT-2 BPE tokenizer with `<|end_of_text|>` 100257 as BOS and EOS; the chat template is plain text
/// (`USER: <|audio|>…\n ASSISTANT:`), so a sequence's first id is the template's `USER` and the prefix the
/// bundle ships is that one id's state. `<|audio|>` is 100352.
pub const EOS_TOKEN: u32 = EOS_TOKEN_ID as u32;
/// `USER` in the GPT-2 vocabulary (the oracle's `meta.json` ids[0]: `USER`, `:`, ` `, then the placeholders).
pub const PREFIX_IDS: &[u32] = &[6584];
pub const EOS_TOKENS: &[u32] = &[EOS_TOKEN];

pub mod layout;
#[cfg(feature = "std")]
pub mod weights;
#[cfg(feature = "std")]
pub mod decoder;
#[cfg(feature = "std")]
pub mod encoder;
