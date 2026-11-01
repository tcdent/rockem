//! The head, one position: the mean of the four streams, the final norm, `lm_head`. Frozen, so
//! the backward is the transposed product over `lm_head`'s bf16 rows (the one op that streams
//! the whole 1.27 GB matrix a position, as the forward does), the norm's backward, and the
//! mean's — `d stream_j = d mean / HC`.

use common::rmsnorm::rmsnorm;
use common::gemv;
use glm_5_3_flash::{H, HC, RMS_NORM_EPS};
use train_common::{bf16, rmsnorm as rms_t};

/// What the backward needs: the mean of the streams (the norm's input).
pub struct Saved {
    pub mean: Vec<f32>, // H
}

impl Saved {
    pub fn new() -> Self {
        Saved { mean: vec![0.0; H] }
    }
}
impl Default for Saved {
    fn default() -> Self {
        Self::new()
    }
}

/// `streams: [HC][H]` → `logits[0..vocab)` over the first `vocab` rows of `lm_head` (`[vocab][H]` bf16).
pub fn forward(norm: &[f32], lm_head: &[u16], vocab: usize, streams: &[f32], saved: &mut Saved, logits: &mut [f32]) {
    for d in 0..H {
        let mut acc = 0f32;
        for j in 0..HC {
            acc += streams[j * H + d];
        }
        saved.mean[d] = acc / HC as f32;
    }
    let mut xn = vec![0f32; H];
    rmsnorm::<H>(&saved.mean, norm, RMS_NORM_EPS, &mut xn);
    // the inference's `bf16_par` over the pool, a row's bits the same whichever core takes it
    gemv::bf16_rows::<H>(lm_head, &xn, logits, 0, vocab);
}

/// Given `d logits[0..vocab)`, accumulate `d streams` (`+=`, `[HC][H]`).
pub fn backward(norm: &[f32], lm_head: &[u16], vocab: usize, saved: &Saved, dlogits: &[f32], dstreams: &mut [f32]) {
    let mut dxn = vec![0f32; H];
    bf16::dense_t::<H>(vocab, lm_head, &dlogits[..vocab], &mut dxn);
    let mut dmean = vec![0f32; H];
    rms_t::backward::<H>(&saved.mean, norm, RMS_NORM_EPS, &dxn, &mut dmean);
    for j in 0..HC {
        for d in 0..H {
            dstreams[j * H + d] += dmean[d] / HC as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::*;
    use common::testing::Rng;

    /// Against the dense arithmetic at a 4096-row head (the op is linear in the row count).
    #[test]
    fn matches_dense() {
        let mut rnd = Rng::new(0x4EAD_0001);
        let vocab = 4096;
        let lm = bf(vocab * H, 0.05, &mut rnd);
        let lmd = wide(&lm);
        let norm: Vec<f32> = (0..H).map(|_| 1.0 + rnd.unit() * 0.3).collect();
        let streams = unit(HC * H, 1.0, &mut rnd);
        let dlogits = unit(vocab, 1.0, &mut rnd);
        let mut saved = Saved::new();
        let mut logits = vec![0f32; vocab];
        forward(&norm, &lm, vocab, &streams, &mut saved, &mut logits);
        let mut ds = vec![0f32; HC * H];
        backward(&norm, &lm, vocab, &saved, &dlogits, &mut ds);

        let mean: Vec<f32> = (0..H).map(|d| (0..HC).map(|j| streams[j * H + d]).sum::<f32>() / HC as f32).collect();
        let ss: f32 = mean.iter().map(|v| v * v).sum();
        let r = common::math::rsqrt(ss / H as f32 + RMS_NORM_EPS);
        let xn: Vec<f32> = (0..H).map(|d| norm[d] * (mean[d] * r)).collect();
        let want: Vec<f32> = (0..vocab).map(|v| (0..H).map(|d| lmd[v * H + d] * xn[d]).sum()).collect();
        close(&logits, &want, 1e-4, "logits");
        let dxn: Vec<f32> = (0..H).map(|d| (0..vocab).map(|v| lmd[v * H + d] * dlogits[v]).sum()).collect();
        let dot: f32 = (0..H).map(|d| norm[d] * dxn[d] * mean[d]).sum();
        let c = r * r * r / H as f32 * dot;
        let dmean: Vec<f32> = (0..H).map(|d| r * norm[d] * dxn[d] - mean[d] * c).collect();
        let want_ds: Vec<f32> = (0..HC * H).map(|i| dmean[i % H] / HC as f32).collect();
        close(&ds, &want_ds, 1e-4, "d streams");
    }
}
