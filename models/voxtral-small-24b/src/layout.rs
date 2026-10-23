//! This model's bundle formats. The decoder's matrices — every layer's fused
//! q|k|v projection, its o projection, its gate|up and down, and the lm_head —
//! are a stream of fixed-size `q4g128` items (`common::q4`: int4 rows, an f32
//! scale per 128), laid out for the memory they are read from: a layer is
//! `RUNS` runs of whole 2 MiB pages, run `r` holding share `r` of every matrix's
//! items, loaded on node `r mod nodes`, so every node streams its share of every
//! matrix and a layer's GEMVs end on every node at once (the expert stream's
//! rule, GLM layout). An item is the unit one core streams: a chunk of output
//! rows with their scales. The runs are padded to the page, under 2 % of the
//! stream. Everything else — the token embedding, the norms, the whole audio
//! encoder and the projector — ships as the checkpoint stores it (bf16), one
//! file, every tensor at a 64-byte boundary in the order of the tables below.
//!
//! Both the bundle builder and the loader walk these tables, so the file has
//! one definition; the names are the checkpoint's own, relative to the layer.

use crate::*;
use common::q4::{GROUP, TILE};

pub const PAGE: usize = 2 << 20;
pub const RUNS: usize = 4;
pub const RECIPE: &str = "decoder:q4g128-a8-rtn encoder:native";

// --- the q4 stream: a layer's four matrices, then the lm_head ---

/// A matrix of the stream: `rows` output rows of `k` inputs, streamed `chunk` rows an item.
#[derive(Clone, Copy, Debug)]
pub struct Mat {
    pub rows: usize,
    pub k: usize,
    pub chunk: usize,
}
impl Mat {
    pub const fn items(&self) -> usize {
        self.rows / self.chunk
    }
    pub const fn per_run(&self) -> usize {
        self.items() / RUNS
    }
    /// An item's int4 rows, then its f32 scales.
    pub const fn item_w(&self) -> usize {
        self.chunk * self.k / 2
    }
    pub const fn item_s(&self) -> usize {
        self.chunk * (self.k / GROUP)
    }
    pub const fn item(&self) -> usize {
        self.item_w() + 4 * self.item_s()
    }
    /// Its share of one run.
    pub const fn run(&self) -> usize {
        self.per_run() * self.item()
    }
    const fn check(&self) -> bool {
        self.rows % self.chunk == 0 && self.items() % RUNS == 0 && self.chunk % TILE == 0 && self.k % GROUP == 0 && self.item() % 64 == 0
    }
}

/// The fused projection: q's rows, then k's, then v's (`[Q_DIM + 2·KV_DIM][H]`).
pub const QKV: Mat = Mat { rows: Q_DIM + 2 * KV_DIM, k: H, chunk: 64 };
pub const O: Mat = Mat { rows: H, k: Q_DIM, chunk: 64 };
/// Gate and up fused an item: `GU.chunk` gate rows then the matching up rows, their scales likewise
/// (`rows` counts both: `2 · D_FF`, `chunk` the rows of each).
pub const GU: Mat = Mat { rows: 2 * D_FF, k: H, chunk: 2 * 64 };
pub const DOWN: Mat = Mat { rows: H, k: D_FF, chunk: 256 };
pub const LM_HEAD: Mat = Mat { rows: VOCAB_SIZE, k: H, chunk: 64 };
const _: () = assert!(QKV.check() && O.check() && GU.check() && DOWN.check() && LM_HEAD.check());

/// A layer's run: its shares of the four matrices, whole pages.
pub const LAYER_RUN: usize = (QKV.run() + O.run() + GU.run() + DOWN.run()).div_ceil(PAGE) * PAGE;
pub const LAYER: usize = RUNS * LAYER_RUN;
pub const HEAD_RUN: usize = LM_HEAD.run().div_ceil(PAGE) * PAGE;
pub const HEAD_BYTES: usize = RUNS * HEAD_RUN;
pub const STREAM: usize = NUM_HIDDEN_LAYERS * LAYER + HEAD_BYTES;

/// The four matrices of a layer, in run order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Which {
    Qkv,
    O,
    Gu,
    Down,
}
pub const fn mat(w: Which) -> Mat {
    match w {
        Which::Qkv => QKV,
        Which::O => O,
        Which::Gu => GU,
        Which::Down => DOWN,
    }
}
/// Byte offset of matrix `w`'s share within a run.
const fn within_run(w: Which) -> usize {
    match w {
        Which::Qkv => 0,
        Which::O => QKV.run(),
        Which::Gu => QKV.run() + O.run(),
        Which::Down => QKV.run() + O.run() + GU.run(),
    }
}
/// Byte range in the stream of item `c` of matrix `w` in layer `l`: rows, then scales (gate rows, up
/// rows, gate scales, up scales for `Gu`).
#[inline(always)]
pub const fn item(l: usize, w: Which, c: usize) -> core::ops::Range<usize> {
    let m = mat(w);
    let start = l * LAYER + (c / m.per_run()) * LAYER_RUN + within_run(w) + (c % m.per_run()) * m.item();
    start..start + m.item()
}
/// Byte range of lm_head item `c`.
#[inline(always)]
pub const fn head_item(c: usize) -> core::ops::Range<usize> {
    let start = NUM_HIDDEN_LAYERS * LAYER + (c / LM_HEAD.per_run()) * HEAD_RUN + (c % LM_HEAD.per_run()) * LM_HEAD.item();
    start..start + LM_HEAD.item()
}
/// The run a page of the stream belongs to: the node that holds it is `run mod nodes`.
#[inline(always)]
pub const fn page_run(page: usize) -> usize {
    let off = page * PAGE;
    if off < NUM_HIDDEN_LAYERS * LAYER { (off % LAYER) / LAYER_RUN } else { ((off - NUM_HIDDEN_LAYERS * LAYER) % HEAD_BYTES) / HEAD_RUN }
}

// --- the dense weights, native ---
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Bf16,
    /// The encoder's sinusoidal positions are stored as f32.
    F32,
}
#[derive(Clone, Copy, Debug)]
pub struct Tensor {
    pub name: &'static str,
    pub kind: Kind,
    pub rows: usize,
    pub cols: usize,
}
pub const ALIGN: usize = 64;
impl Tensor {
    const fn new(name: &'static str, rows: usize, cols: usize) -> Self {
        Tensor { name, kind: Kind::Bf16, rows, cols }
    }
    const fn f32(name: &'static str, rows: usize, cols: usize) -> Self {
        Tensor { name, kind: Kind::F32, rows, cols }
    }
    pub const fn len(&self) -> usize {
        self.rows * self.cols
    }
    pub const fn bytes(&self) -> usize {
        match self.kind {
            Kind::Bf16 => 2 * self.len(),
            Kind::F32 => 4 * self.len(),
        }
    }
    /// What the tensor occupies in the file: its bytes, padded to the alignment.
    pub const fn slot(&self) -> usize {
        self.bytes().div_ceil(ALIGN) * ALIGN
    }
}

/// The decoder's: the token embedding and the final norm (its matrices are in the stream).
pub const HEAD: [Tensor; 2] = [Tensor::new("language_model.model.embed_tokens.weight", VOCAB_SIZE, H), Tensor::new("language_model.model.norm.weight", 1, H)];
/// A decoder layer's norms (relative to `language_model.model.layers.{l}.`).
pub const LAYER_NORMS: [Tensor; 2] = [Tensor::new("input_layernorm.weight", 1, H), Tensor::new("post_attention_layernorm.weight", 1, H)];
/// The encoder's frontend and its final norm (relative to `audio_tower.`): the convolutions' weights
/// are `[out][in · 3]` as Conv1d stores them, the taps innermost.
pub const ENC_HEAD: [Tensor; 7] = [
    Tensor::new("conv1.weight", ENC_H, NUM_MEL_BINS * 3),
    Tensor::new("conv1.bias", 1, ENC_H),
    Tensor::new("conv2.weight", ENC_H, ENC_H * 3),
    Tensor::new("conv2.bias", 1, ENC_H),
    Tensor::f32("embed_positions.weight", ENC_POSITIONS, ENC_H),
    Tensor::new("layer_norm.weight", 1, ENC_H),
    Tensor::new("layer_norm.bias", 1, ENC_H),
];
/// An encoder layer (relative to `audio_tower.layers.{l}.`): k has no bias, the rest do.
pub const ENC_LAYER: [Tensor; 15] = [
    Tensor::new("self_attn_layer_norm.weight", 1, ENC_H),
    Tensor::new("self_attn_layer_norm.bias", 1, ENC_H),
    Tensor::new("self_attn.q_proj.weight", ENC_H, ENC_H),
    Tensor::new("self_attn.q_proj.bias", 1, ENC_H),
    Tensor::new("self_attn.k_proj.weight", ENC_H, ENC_H),
    Tensor::new("self_attn.v_proj.weight", ENC_H, ENC_H),
    Tensor::new("self_attn.v_proj.bias", 1, ENC_H),
    Tensor::new("self_attn.out_proj.weight", ENC_H, ENC_H),
    Tensor::new("self_attn.out_proj.bias", 1, ENC_H),
    Tensor::new("final_layer_norm.weight", 1, ENC_H),
    Tensor::new("final_layer_norm.bias", 1, ENC_H),
    Tensor::new("fc1.weight", ENC_INTERMEDIATE_SIZE, ENC_H),
    Tensor::new("fc1.bias", 1, ENC_INTERMEDIATE_SIZE),
    Tensor::new("fc2.weight", ENC_H, ENC_INTERMEDIATE_SIZE),
    Tensor::new("fc2.bias", 1, ENC_H),
];
/// The projector (relative to `multi_modal_projector.`), no biases.
pub const PROJECTOR: [Tensor; 2] = [Tensor::new("linear_1.weight", H, ENC_INTERMEDIATE_SIZE), Tensor::new("linear_2.weight", H, H)];

/// Bytes a table occupies, and the offset of its `i`th tensor within it.
pub const fn section(t: &[Tensor]) -> usize {
    let mut s = 0;
    let mut i = 0;
    while i < t.len() {
        s += t[i].slot();
        i += 1;
    }
    s
}
pub const fn within(t: &[Tensor], i: usize) -> usize {
    let mut s = 0;
    let mut j = 0;
    while j < i {
        s += t[j].slot();
        j += 1;
    }
    s
}
/// The dense file: the decoder's head, its layers' norms, the encoder's head, its layers, the projector.
pub const DENSE_HEAD: usize = 0;
pub const DENSE_LAYER_NORMS: usize = section(&HEAD);
pub const DENSE_ENC_HEAD: usize = DENSE_LAYER_NORMS + NUM_HIDDEN_LAYERS * section(&LAYER_NORMS);
pub const DENSE_ENC_LAYERS: usize = DENSE_ENC_HEAD + section(&ENC_HEAD);
pub const DENSE_PROJECTOR: usize = DENSE_ENC_LAYERS + ENC_NUM_HIDDEN_LAYERS * section(&ENC_LAYER);
pub const DENSE_BYTES: usize = DENSE_PROJECTOR + section(&PROJECTOR);
/// Offset of decoder layer `l`'s norms, and of encoder layer `l`.
pub const fn layer_norms(l: usize) -> usize {
    DENSE_LAYER_NORMS + l * section(&LAYER_NORMS)
}
pub const fn enc_layer(l: usize) -> usize {
    DENSE_ENC_LAYERS + l * section(&ENC_LAYER)
}

pub const DENSE_FILE: &str = "dense.bin";
pub const STREAM_FILE: &str = "decoder.q4g128.bin";
pub const VOCAB_FILE: &str = "vocab.bin";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const MEL_FILE: &str = "mel_filters.f32";
/// The mel filterbank the edge computes the spectrogram with: `[NUM_MEL_BINS][N_FFT / 2 + 1]` f32.
pub const N_FFT: usize = 400;
pub const MEL_BYTES: usize = 4 * NUM_MEL_BINS * (N_FFT / 2 + 1);

const _: () = assert!(STREAM % PAGE == 0 && LAYER_RUN >= QKV.run() + O.run() + GU.run() + DOWN.run());

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sizes() {
        let mb = |b: usize| b as f64 / 1e6;
        println!("qkv item {} ({} items, {} a run), o {} ({}), gu {} ({}), down {} ({}), lm_head {} ({})", QKV.item(), QKV.items(), QKV.per_run(), O.item(), O.items(), GU.item(), GU.items(), DOWN.item(), DOWN.items(), LM_HEAD.item(), LM_HEAD.items());
        println!("layer run {:.1} MB of {:.1} used; layer {:.1} MB; head {:.1} MB; stream {:.2} GB; dense {:.2} GB", mb(LAYER_RUN), mb(QKV.run() + O.run() + GU.run() + DOWN.run()), mb(LAYER), mb(HEAD_BYTES), STREAM as f64 / 1e9, DENSE_BYTES as f64 / 1e9);
        assert_eq!(item(0, Which::Qkv, 0).start, 0);
        assert_eq!(item(1, Which::Qkv, 0).start, LAYER);
        assert_eq!(item(0, Which::Down, DOWN.items() - 1).end <= LAYER_RUN * RUNS, true);
        assert_eq!(head_item(LM_HEAD.items() - 1).end <= STREAM, true);
        assert_eq!(page_run(LAYER_RUN / PAGE), 1);
    }
}
