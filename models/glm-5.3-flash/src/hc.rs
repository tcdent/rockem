//! The hyper-connection (mHC): from one position's four residual streams, the
//! block's input and the mix that puts the block's output back. The
//! coefficients are one bf16 GEMV over the unweighted-normed flattened
//! streams; `pre` and `post` are sigmoids, `comb` a softmax per row then
//! Sinkhorn-Knopp toward doubly stochastic — the reference's arithmetic
//! (`reference::Hc::pre`), one token at a time.

use crate::weights;
use crate::*;
use common::gemv;
use common::pool::Pool;
use common::math::{exp, sqrt};

pub struct Scratch<'a> {
    pub flat: &'a mut [f32], // HC·H
    pub coef: &'a mut [f32], // HC_COEFS
}

#[derive(Clone, Copy)]
pub struct Mix {
    pub post: [f32; HC],
    /// `comb[i][j]`: how much of input stream `i` goes to output stream `j`.
    pub comb: [[f32; HC]; HC],
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

/// `streams: [HC][H]` → the collapsed block input `x: [H]` and the mix.
#[inline(always)]
pub fn pre(w: &weights::Hc, streams: &[f32], s: &mut Scratch, pool: &dyn Pool, x: &mut [f32]) -> Mix {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avx2::norm_flat(streams.as_ptr(), s.flat.as_mut_ptr())
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let n = HC * H;
        let mut ss = 0f32;
        for &v in &streams[..n] {
            ss += v * v;
        }
        let r = 1.0 / sqrt(ss / n as f32 + RMS_NORM_EPS);
        for i in 0..n {
            s.flat[i] = streams[i] * r;
        }
    }
    // 24 rows of 40 KB: a core each
    gemv::bf16_par_r::<HC_COEFS, { HC * H }, 1>(w.fn_, s.flat, s.coef, pool);
    let c = &*s.coef;
    let mut pre = [0f32; HC];
    let mut mix = Mix { post: [0.0; HC], comb: [[0.0; HC]; HC] };
    for i in 0..HC {
        pre[i] = sigmoid(c[i] * w.scale[0] + w.base[i]) + HC_EPS;
        mix.post[i] = 2.0 * sigmoid(c[HC + i] * w.scale[1] + w.base[HC + i]);
    }
    let cb = &mut mix.comb;
    for i in 0..HC {
        let mut row = [0f32; HC];
        let mut mx = f32::NEG_INFINITY;
        for j in 0..HC {
            row[j] = c[2 * HC + i * HC + j] * w.scale[2] + w.base[2 * HC + i * HC + j];
            if row[j] > mx {
                mx = row[j];
            }
        }
        let mut sum = 0f32;
        for j in 0..HC {
            row[j] = exp(row[j] - mx);
            sum += row[j];
        }
        for j in 0..HC {
            cb[i][j] = row[j] / sum + HC_EPS;
        }
    }
    // Sinkhorn-Knopp: columns first, then rows and columns `HC_SINKHORN_ITERS - 1` times
    let cols = |cb: &mut [[f32; HC]; HC]| {
        for j in 0..HC {
            let mut sum = 0f32;
            for i in 0..HC {
                sum += cb[i][j];
            }
            let sum = sum + HC_EPS;
            for i in 0..HC {
                cb[i][j] /= sum;
            }
        }
    };
    cols(cb);
    for _ in 0..HC_SINKHORN_ITERS - 1 {
        for row in cb.iter_mut() {
            let mut sum = 0f32;
            for v in row.iter() {
                sum += v;
            }
            let sum = sum + HC_EPS;
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
        cols(cb);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avx2::collapse(&pre, streams.as_ptr(), x.as_mut_ptr())
    }
    #[cfg(not(target_arch = "x86_64"))]
    for d in 0..H {
        let mut acc = 0f32;
        for i in 0..HC {
            acc += pre[i] * streams[i * H + d];
        }
        x[d] = acc;
    }
    mix
}

/// `stream_j = post_j · y + Σ_i comb_ij · stream_i`, in place: each `d` reads
/// its four inputs before writing its four outputs.
#[inline(always)]
pub fn post(mix: &Mix, y: &[f32], streams: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avx2::post(mix, y.as_ptr(), streams.as_mut_ptr())
    }
    #[cfg(not(target_arch = "x86_64"))]
    for d in 0..H {
        let mut res = [0f32; HC];
        for i in 0..HC {
            res[i] = streams[i * H + d];
        }
        for j in 0..HC {
            let mut acc = 0f32;
            for i in 0..HC {
                acc += mix.comb[i][j] * res[i];
            }
            streams[j * H + d] = mix.post[j] * y[d] + acc;
        }
    }
}

/// The stream-wide loops eight floats at a time; the per-element products and
/// sums in the scalar form's order (multiply, then add), so the results match it.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use core::arch::x86_64::*;
    const N: usize = HC * H;
    const _: () = assert!(N % 32 == 0 && H % 8 == 0);

    /// `flat = streams · r`, `r` the rms-norm scale over every stream's every channel.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn norm_flat(streams: *const f32, flat: *mut f32) {
        let (mut a0, mut a1, mut a2, mut a3) = (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        while i < N {
            let (v0, v1, v2, v3) = (_mm256_loadu_ps(streams.add(i)), _mm256_loadu_ps(streams.add(i + 8)), _mm256_loadu_ps(streams.add(i + 16)), _mm256_loadu_ps(streams.add(i + 24)));
            a0 = _mm256_fmadd_ps(v0, v0, a0);
            a1 = _mm256_fmadd_ps(v1, v1, a1);
            a2 = _mm256_fmadd_ps(v2, v2, a2);
            a3 = _mm256_fmadd_ps(v3, v3, a3);
            i += 32;
        }
        let ss = gemv::avx2::hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        let r = _mm256_set1_ps(1.0 / sqrt(ss / N as f32 + RMS_NORM_EPS));
        let mut i = 0;
        while i < N {
            _mm256_storeu_ps(flat.add(i), _mm256_mul_ps(_mm256_loadu_ps(streams.add(i)), r));
            i += 8;
        }
    }

    /// `x[d] = Σ_i pre[i] · streams[i][d]`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn collapse(pre: &[f32; HC], streams: *const f32, x: *mut f32) {
        let p: [__m256; HC] = core::array::from_fn(|i| _mm256_set1_ps(pre[i]));
        let mut d = 0;
        while d < H {
            let mut acc = _mm256_setzero_ps();
            for i in 0..HC {
                acc = _mm256_add_ps(acc, _mm256_mul_ps(p[i], _mm256_loadu_ps(streams.add(i * H + d))));
            }
            _mm256_storeu_ps(x.add(d), acc);
            d += 8;
        }
    }

    /// `stream_j = post_j · y + Σ_i comb_ij · stream_i`, eight channels at a time.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn post(mix: &Mix, y: *const f32, streams: *mut f32) {
        let comb: [[__m256; HC]; HC] = core::array::from_fn(|i| core::array::from_fn(|j| _mm256_set1_ps(mix.comb[i][j])));
        let post: [__m256; HC] = core::array::from_fn(|j| _mm256_set1_ps(mix.post[j]));
        let mut d = 0;
        while d < H {
            let res: [__m256; HC] = core::array::from_fn(|i| _mm256_loadu_ps(streams.add(i * H + d)));
            let yv = _mm256_loadu_ps(y.add(d));
            for j in 0..HC {
                let mut acc = _mm256_setzero_ps();
                for i in 0..HC {
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(comb[i][j], res[i]));
                }
                _mm256_storeu_ps(streams.add(j * H + d), _mm256_add_ps(_mm256_mul_ps(post[j], yv), acc));
            }
            d += 8;
        }
    }
}
