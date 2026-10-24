//! The weights as slices into the bundle's two files (`layout`): the native dense tensors as bf16 (`&[u16]`)
//! and the decoder's q4 stream as items — an item's int4 rows and its f32 scales, by (layer, matrix, item).
//! Plain slices, no copies: the files are mapped where the loader put them (the stream's pages on the node
//! that streams them).

use crate::layout::{self, Mat, Tensor, Which};

/// The dense file.
#[derive(Clone, Copy)]
pub struct Dense<'a> {
    pub bytes: &'a [u8],
}
impl<'a> Dense<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        assert_eq!(bytes.len(), layout::DENSE_BYTES, "dense.bin: not this model's");
        Dense { bytes }
    }
    fn at(&self, off: usize, t: &Tensor) -> &'a [u16] {
        let b = &self.bytes[off..off + t.bytes()];
        let (p, v, q) = unsafe { b.align_to::<u16>() };
        assert!(p.is_empty() && q.is_empty());
        v
    }
    /// `[VOCAB_SIZE][H]`.
    pub fn embed_tokens(&self) -> &'a [u16] {
        self.at(layout::DENSE_HEAD + layout::within(&layout::HEAD, 0), &layout::HEAD[0])
    }
    pub fn final_norm(&self) -> &'a [u16] {
        self.at(layout::DENSE_HEAD + layout::within(&layout::HEAD, 1), &layout::HEAD[1])
    }
    /// Decoder layer `l`'s input norm (0) and post-attention norm (1).
    pub fn layer_norm(&self, l: usize, i: usize) -> &'a [u16] {
        self.at(layout::layer_norms(l) + layout::within(&layout::LAYER_NORMS, i), &layout::LAYER_NORMS[i])
    }
    pub fn enc_head(&self, i: usize) -> &'a [u16] {
        self.at(layout::DENSE_ENC_HEAD + layout::within(&layout::ENC_HEAD, i), &layout::ENC_HEAD[i])
    }
    pub fn enc_layer(&self, l: usize, i: usize) -> &'a [u16] {
        self.at(layout::enc_layer(l) + layout::within(&layout::ENC_LAYER, i), &layout::ENC_LAYER[i])
    }
    pub fn proj_head(&self, i: usize) -> &'a [u16] {
        self.at(layout::DENSE_PROJ_HEAD + layout::within(&layout::PROJ_HEAD, i), &layout::PROJ_HEAD[i])
    }
    pub fn proj_layer(&self, l: usize, i: usize) -> &'a [u16] {
        self.at(layout::proj_layer(l) + layout::within(&layout::PROJ_LAYER, i), &layout::PROJ_LAYER[i])
    }
}

/// The decoder's q4 stream.
#[derive(Clone, Copy)]
pub struct Stream<'a> {
    pub bytes: &'a [u8],
}
/// One item: `rows` int4 rows of `k` as `q4::gemv` reads them, then their scales; for the gate|up items the
/// gate's rows and scales come first, the up's after (`half` rows each).
#[derive(Clone, Copy)]
pub struct Item<'a> {
    pub w: &'a [u8],
    pub s: &'a [f32],
}
impl<'a> Stream<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        assert_eq!(bytes.len(), layout::STREAM, "decoder.q4g128.bin: not this model's");
        Stream { bytes }
    }
    fn at(&self, r: core::ops::Range<usize>, m: &Mat) -> Item<'a> {
        let b = &self.bytes[r];
        let (w, s) = b.split_at(m.item_w());
        let (p, s, q) = unsafe { s.align_to::<f32>() };
        debug_assert!(p.is_empty() && q.is_empty());
        Item { w, s }
    }
    pub fn item(&self, l: usize, which: Which, c: usize) -> Item<'a> {
        self.at(layout::item(l, which, c), &layout::mat(which))
    }
    /// Gate|up item `c` of layer `l`: (gate rows, up rows, gate scales, up scales), `GU.chunk / 2` rows each.
    pub fn gu(&self, l: usize, c: usize) -> (&'a [u8], &'a [u8], &'a [f32], &'a [f32]) {
        let it = self.item(l, Which::Gu, c);
        let (gw, uw) = it.w.split_at(it.w.len() / 2);
        let (gs, us) = it.s.split_at(it.s.len() / 2);
        (gw, uw, gs, us)
    }
    pub fn head_item(&self, c: usize) -> Item<'a> {
        self.at(layout::head_item(c), &layout::LM_HEAD)
    }
}

/// bf16 → f32.
#[inline(always)]
pub fn bf(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}
/// bf16 → f32 widened once, for the small vectors the kernels want as f32 (norms, biases).
pub fn widen(w: &[u16]) -> std::vec::Vec<f32> {
    w.iter().map(|&v| bf(v)).collect()
}
