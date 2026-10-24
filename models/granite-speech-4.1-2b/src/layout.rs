//! This model's bundle formats. The decoder's matrices — every layer's fused q|k|v, its o, its gate|up and
//! down, and the lm_head — are a stream of fixed-size `q4g128` items (`common::q4`: int4 rows, an f32 scale
//! per 128), in `RUNS` regions: region `r` holds share `r` of every matrix of every layer, then of the
//! lm_head, contiguous, padded to the 2 MiB page once at its end, and is loaded on node `r mod nodes` — so
//! every node streams its share of every matrix and a layer's GEMVs end on every node at once. (Voxtral pads
//! every layer's run to the page; at 4.7 MB a run this model would waste a quarter of its stream that way.) An
//! item is the unit one core streams: a chunk of output rows with their scales. Everything else — the token
//! embedding, the norms, the whole encoder and the projector — ships as the checkpoint stores it (bf16),
//! one file, every tensor at a 64-byte boundary in the order of the tables below.
//!
//! Both the bundle builder and the loader walk these tables, so the file has one definition; the names are
//! the checkpoint's own, relative to the layer.

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
    /// Its share of one region.
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
/// Gate and up fused an item: `GU.chunk / 2` gate rows then the matching up rows, their scales likewise
/// (`rows` counts both: `2 · D_FF`).
pub const GU: Mat = Mat { rows: 2 * D_FF, k: H, chunk: 2 * 64 };
pub const DOWN: Mat = Mat { rows: H, k: D_FF, chunk: 64 };
/// The vocabulary padded to `VOCAB_PAD` rows (zero rows past `VOCAB_SIZE`).
pub const LM_HEAD: Mat = Mat { rows: VOCAB_PAD, k: H, chunk: 64 };
const _: () = assert!(QKV.check() && O.check() && GU.check() && DOWN.check() && LM_HEAD.check());

/// A layer's share of a region: its shares of the four matrices.
pub const LAYER_SHARE: usize = QKV.run() + O.run() + GU.run() + DOWN.run();
/// A region: every layer's share, the lm_head's, whole pages.
pub const REGION: usize = (NUM_HIDDEN_LAYERS * LAYER_SHARE + LM_HEAD.run()).div_ceil(PAGE) * PAGE;
pub const STREAM: usize = RUNS * REGION;

/// The four matrices of a layer, in share order.
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
/// Byte offset of matrix `w`'s share within a layer's share.
const fn within_share(w: Which) -> usize {
    match w {
        Which::Qkv => 0,
        Which::O => QKV.run(),
        Which::Gu => QKV.run() + O.run(),
        Which::Down => QKV.run() + O.run() + GU.run(),
    }
}
/// Byte range in the stream of item `c` of matrix `w` in layer `l`: rows, then scales (gate rows, up
/// rows, gate scales, up scales for `Gu`). Item `c` is in region `c / per_run`.
#[inline(always)]
pub const fn item(l: usize, w: Which, c: usize) -> core::ops::Range<usize> {
    let m = mat(w);
    let start = (c / m.per_run()) * REGION + l * LAYER_SHARE + within_share(w) + (c % m.per_run()) * m.item();
    start..start + m.item()
}
/// Byte range of lm_head item `c`.
#[inline(always)]
pub const fn head_item(c: usize) -> core::ops::Range<usize> {
    let start = (c / LM_HEAD.per_run()) * REGION + NUM_HIDDEN_LAYERS * LAYER_SHARE + (c % LM_HEAD.per_run()) * LM_HEAD.item();
    start..start + LM_HEAD.item()
}
/// The region a page of the stream belongs to: the node that holds it is `region mod nodes`.
#[inline(always)]
pub const fn page_run(page: usize) -> usize {
    page / (REGION / PAGE)
}

// --- the dense weights, native bf16 ---
#[derive(Clone, Copy, Debug)]
pub struct Tensor {
    pub name: &'static str,
    pub rows: usize,
    pub cols: usize,
}
pub const ALIGN: usize = 64;
impl Tensor {
    const fn new(name: &'static str, rows: usize, cols: usize) -> Self {
        Tensor { name, rows, cols }
    }
    pub const fn len(&self) -> usize {
        self.rows * self.cols
    }
    pub const fn bytes(&self) -> usize {
        2 * self.len()
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
/// The encoder's input projection and its CTC heads (relative to `encoder.`).
pub const ENC_HEAD: [Tensor; 6] = [
    Tensor::new("input_linear.weight", ENC_H, ENC_INPUT_DIM),
    Tensor::new("input_linear.bias", 1, ENC_H),
    Tensor::new("out.weight", ENC_OUTPUT_DIM, ENC_H),
    Tensor::new("out.bias", 1, ENC_OUTPUT_DIM),
    Tensor::new("out_mid.weight", ENC_H, ENC_OUTPUT_DIM),
    Tensor::new("out_mid.bias", 1, ENC_H),
];
/// A Conformer block (relative to `encoder.layers.{l}.`); the pointwise convolutions' weights are `[out][in][1]` as
/// Conv1d stores them (the same bytes as `[out][in]`), the depthwise one `[ENC_CONV][1][K]`.
pub const ENC_LAYER: [Tensor; 32] = [
    Tensor::new("ff1.pre_norm.weight", 1, ENC_H),
    Tensor::new("ff1.pre_norm.bias", 1, ENC_H),
    Tensor::new("ff1.up_proj.weight", ENC_FF, ENC_H),
    Tensor::new("ff1.up_proj.bias", 1, ENC_FF),
    Tensor::new("ff1.down_proj.weight", ENC_H, ENC_FF),
    Tensor::new("ff1.down_proj.bias", 1, ENC_H),
    Tensor::new("attn.pre_norm.weight", 1, ENC_H),
    Tensor::new("attn.pre_norm.bias", 1, ENC_H),
    Tensor::new("attn.to_q.weight", ENC_H, ENC_H),
    Tensor::new("attn.to_kv.weight", 2 * ENC_H, ENC_H),
    Tensor::new("attn.to_out.weight", ENC_H, ENC_H),
    Tensor::new("attn.to_out.bias", 1, ENC_H),
    Tensor::new("attn.rel_pos_emb.weight", ENC_REL_ROWS, ENC_HEAD_DIM),
    Tensor::new("conv.norm.weight", 1, ENC_H),
    Tensor::new("conv.norm.bias", 1, ENC_H),
    Tensor::new("conv.up_conv.weight", 2 * ENC_CONV, ENC_H),
    Tensor::new("conv.up_conv.bias", 1, 2 * ENC_CONV),
    Tensor::new("conv.depth_conv.conv.weight", ENC_CONV, ENC_CONV_KERNEL),
    Tensor::new("conv.batch_norm.weight", 1, ENC_CONV),
    Tensor::new("conv.batch_norm.bias", 1, ENC_CONV),
    Tensor::new("conv.batch_norm.running_mean", 1, ENC_CONV),
    Tensor::new("conv.batch_norm.running_var", 1, ENC_CONV),
    Tensor::new("conv.down_conv.weight", ENC_H, ENC_CONV),
    Tensor::new("conv.down_conv.bias", 1, ENC_H),
    Tensor::new("ff2.pre_norm.weight", 1, ENC_H),
    Tensor::new("ff2.pre_norm.bias", 1, ENC_H),
    Tensor::new("ff2.up_proj.weight", ENC_FF, ENC_H),
    Tensor::new("ff2.up_proj.bias", 1, ENC_FF),
    Tensor::new("ff2.down_proj.weight", ENC_H, ENC_FF),
    Tensor::new("ff2.down_proj.bias", 1, ENC_H),
    Tensor::new("post_norm.weight", 1, ENC_H),
    Tensor::new("post_norm.bias", 1, ENC_H),
];
/// The projector's head (relative to `projector.`): the queries, the Q-Former's input norm, the output linear.
pub const PROJ_HEAD: [Tensor; 5] = [
    Tensor::new("query", QUERIES, PROJ_H),
    Tensor::new("qformer.layernorm.weight", 1, PROJ_H),
    Tensor::new("qformer.layernorm.bias", 1, PROJ_H),
    Tensor::new("linear.weight", H, PROJ_H),
    Tensor::new("linear.bias", 1, H),
];
/// A Q-Former layer (relative to `projector.qformer.encoder.layer.{l}.`).
pub const PROJ_LAYER: [Tensor; 26] = [
    Tensor::new("attention.attention.query.weight", PROJ_H, PROJ_H),
    Tensor::new("attention.attention.query.bias", 1, PROJ_H),
    Tensor::new("attention.attention.key.weight", PROJ_H, PROJ_H),
    Tensor::new("attention.attention.key.bias", 1, PROJ_H),
    Tensor::new("attention.attention.value.weight", PROJ_H, PROJ_H),
    Tensor::new("attention.attention.value.bias", 1, PROJ_H),
    Tensor::new("attention.output.dense.weight", PROJ_H, PROJ_H),
    Tensor::new("attention.output.dense.bias", 1, PROJ_H),
    Tensor::new("attention.output.LayerNorm.weight", 1, PROJ_H),
    Tensor::new("attention.output.LayerNorm.bias", 1, PROJ_H),
    Tensor::new("crossattention.attention.query.weight", PROJ_H, PROJ_H),
    Tensor::new("crossattention.attention.query.bias", 1, PROJ_H),
    Tensor::new("crossattention.attention.key.weight", PROJ_H, ENC_H),
    Tensor::new("crossattention.attention.key.bias", 1, PROJ_H),
    Tensor::new("crossattention.attention.value.weight", PROJ_H, ENC_H),
    Tensor::new("crossattention.attention.value.bias", 1, PROJ_H),
    Tensor::new("crossattention.output.dense.weight", PROJ_H, PROJ_H),
    Tensor::new("crossattention.output.dense.bias", 1, PROJ_H),
    Tensor::new("crossattention.output.LayerNorm.weight", 1, PROJ_H),
    Tensor::new("crossattention.output.LayerNorm.bias", 1, PROJ_H),
    Tensor::new("intermediate_query.dense.weight", PROJ_INTERMEDIATE_SIZE, PROJ_H),
    Tensor::new("intermediate_query.dense.bias", 1, PROJ_INTERMEDIATE_SIZE),
    Tensor::new("output_query.dense.weight", PROJ_H, PROJ_INTERMEDIATE_SIZE),
    Tensor::new("output_query.dense.bias", 1, PROJ_H),
    Tensor::new("output_query.LayerNorm.weight", 1, PROJ_H),
    Tensor::new("output_query.LayerNorm.bias", 1, PROJ_H),
];

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
/// The dense file: the decoder's head, its layers' norms, the encoder's head, its blocks, the projector's head, its layers.
pub const DENSE_HEAD: usize = 0;
pub const DENSE_LAYER_NORMS: usize = section(&HEAD);
pub const DENSE_ENC_HEAD: usize = DENSE_LAYER_NORMS + NUM_HIDDEN_LAYERS * section(&LAYER_NORMS);
pub const DENSE_ENC_LAYERS: usize = DENSE_ENC_HEAD + section(&ENC_HEAD);
pub const DENSE_PROJ_HEAD: usize = DENSE_ENC_LAYERS + ENC_NUM_HIDDEN_LAYERS * section(&ENC_LAYER);
pub const DENSE_PROJ_LAYERS: usize = DENSE_PROJ_HEAD + section(&PROJ_HEAD);
pub const DENSE_BYTES: usize = DENSE_PROJ_LAYERS + PROJ_NUM_HIDDEN_LAYERS * section(&PROJ_LAYER);
pub const fn layer_norms(l: usize) -> usize {
    DENSE_LAYER_NORMS + l * section(&LAYER_NORMS)
}
pub const fn enc_layer(l: usize) -> usize {
    DENSE_ENC_LAYERS + l * section(&ENC_LAYER)
}
pub const fn proj_layer(l: usize) -> usize {
    DENSE_PROJ_LAYERS + l * section(&PROJ_LAYER)
}

pub const DENSE_FILE: &str = "dense.bin";
pub const STREAM_FILE: &str = "decoder.q4g128.bin";
pub const VOCAB_FILE: &str = "vocab.bin";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const MEL_FILE: &str = "mel_filters.f32";
/// The mel filterbank the edge computes the spectrogram with: `[NUM_MEL_BINS][N_FFT / 2 + 1]` f32 (torchaudio's
/// HTK-scale bank, unnormalized, 0–8 kHz), over a 400-sample Hann window zero-padded to the 512-point FFT.
pub const N_FFT: usize = 512;
pub const FFT_WINDOW: usize = 400;
pub const HOP: usize = 160;
pub const MEL_BYTES: usize = 4 * NUM_MEL_BINS * (N_FFT / 2 + 1);

const _: () = assert!(STREAM % PAGE == 0 && REGION % PAGE == 0);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sizes() {
        let mb = |b: usize| b as f64 / 1e6;
        println!("qkv item {} ({} items), o {} ({}), gu {} ({}), down {} ({}), lm_head {} ({})", QKV.item(), QKV.items(), O.item(), O.items(), GU.item(), GU.items(), DOWN.item(), DOWN.items(), LM_HEAD.item(), LM_HEAD.items());
        println!("layer share {:.2} MB; region {:.1} MB of {:.1} used; stream {:.3} GB; dense {:.3} GB", mb(LAYER_SHARE), mb(REGION), mb(NUM_HIDDEN_LAYERS * LAYER_SHARE + LM_HEAD.run()), STREAM as f64 / 1e9, DENSE_BYTES as f64 / 1e9);
        assert_eq!(item(0, Which::Qkv, 0).start, 0);
        assert_eq!(item(1, Which::Qkv, 0).start, LAYER_SHARE);
        assert_eq!(item(0, Which::Qkv, QKV.per_run()).start, REGION);
        assert!(item(NUM_HIDDEN_LAYERS - 1, Which::Down, DOWN.items() - 1).end <= STREAM);
        assert!(head_item(LM_HEAD.items() - 1).end <= STREAM);
        assert_eq!(page_run(REGION / PAGE), 1);
        assert_eq!(page_run(STREAM / PAGE - 1), RUNS - 1);
    }
}
