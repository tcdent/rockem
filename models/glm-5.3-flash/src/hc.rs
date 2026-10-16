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

pub mod gfx11_consts {
    use crate::*;
    /// `hc_coef`: the coefficient rows over the flattened streams, `rows_grid(HC_COEFS)` workgroups.
    pub const COEF_ROWS: usize = HC_COEFS;
    /// The mix the card keeps between a block's two halves: pre `[HC]`, post `[HC]`, comb `[HC][HC]`.
    pub const MIX: usize = 2 * HC + HC * HC;
    /// `hc_post` and `hc_mean`: a thread a channel.
    pub const CHANNEL_GRID: u32 = (H / 256) as u32;
}

#[cfg(target_arch = "amdgpu")]
pub mod gfx11 {
    //! The hyper-connections on the card, where the residual streams live
    //! (`[HC][H]` f32): `hc_pre` (the coefficient GEMV over the flattened
    //! streams rms-normed without a weight as they are staged — `rows_h`'s
    //! skeleton with a row of ones — then, in the last workgroup done, the
    //! sigmoids, the softmax rows and Sinkhorn into the mix and the collapse
    //! to the block input, written to VRAM and, when asked, to host memory for
    //! the routed experts), `hc_post` (the stream
    //! update from the block output), `hc_mean` (the streams' mean, the head's
    //! input), and `hc_handoff` (the words between the cards and the host).
    //! The per-element order is the host's (`pre`, `post`): multiply, then
    //! add, streams in order.
    pub use super::gfx11_consts::*;
    use crate::kda::gfx11_consts::{normed_rows_lds, rows_grid};
    use crate::*;
    use common::gemv::gfx11::rows_over;
    use common::gfx11::{lds, workgroup_sync};
    use common::math::exp;
    use core::arch::amdgpu::{workgroup_id_x, workitem_id_x};
    use core::sync::atomic::{fence, AtomicU32, Ordering};

    #[inline(always)]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + exp(-x))
    }

    /// The hyper-connection's input half as one launch: `coef[row] = W[row] · rmsnorm(streams)`
    /// over all `HC·H` channels (`ones` the norm's weight), `rows_grid(COEF_ROWS)` workgroups;
    /// the last one done (`count`, a VRAM word back at 0 when it leaves) reads the coefficients
    /// past its CU's caches and makes the mix (`hc::pre`'s arithmetic) and the collapsed block
    /// input `x[d] = Σ_i pre_i · stream_i[d]`, into `x` and, when `x_host` is not null, there too
    /// (the routed experts' input; the host's word follows in a later handoff). LDS:
    /// `normed_rows_lds(HC·H, 1)` + 64.
    #[no_mangle]
    pub extern "gpu-kernel" fn hc_pre(streams: *const f32, ones: *const u16, w: *const u16, coef: *mut f32, m: u32, n: u32, base: *const f32, scale: *const f32, mix: *mut f32, x: *mut f32, x_host: *mut f32, count: *mut u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            rows_over::<{ HC * H }, true>(tid, w, ones, m, n, RMS_NORM_EPS, |i, s| *streams.add(i * HC * H + s * 256 + t), || {}, |row, i, y| *coef.add(i * m + row) = y);
            // the launch's arrival: this workgroup's coefficients to L2, one atomic, `last` through LDS
            let last = (lds::<u8>() as *mut u8).add(normed_rows_lds(HC * H, 1)) as *mut u32;
            fence(Ordering::Release);
            workgroup_sync();
            if t == 0 {
                let arrived = AtomicU32::from_ptr(count).fetch_add(1, Ordering::AcqRel);
                *last = (arrived == rows_grid(COEF_ROWS) - 1) as u32;
            }
            workgroup_sync();
            if *last == 0 {
                return;
            }
            fence(Ordering::Acquire);
            if t == 0 {
                *count = 0; // nothing else touches it until the next launch
            }
            // every thread computes the whole mix: a few hundred operations, no exchange needed; the
            // coefficients read past this CU's caches (a plain read after the acquire came back stale
            // one run in ~15 on V2-Lite's router, 2026-09-19)
            let c = |i: usize| f32::from_bits(AtomicU32::from_ptr(coef.add(i) as *mut u32).load(Ordering::Relaxed));
            let (s0, s1, s2) = (*scale, *scale.add(1), *scale.add(2));
            let mut pre = [0f32; HC];
            let mut post = [0f32; HC];
            let mut cb = [[0f32; HC]; HC];
            let mut i = 0;
            while i < HC {
                pre[i] = sigmoid(c(i) * s0 + *base.add(i)) + HC_EPS;
                post[i] = 2.0 * sigmoid(c(HC + i) * s1 + *base.add(HC + i));
                i += 1;
            }
            let mut i = 0;
            while i < HC {
                let mut row = [0f32; HC];
                let mut mx = f32::NEG_INFINITY;
                let mut j = 0;
                while j < HC {
                    row[j] = c(2 * HC + i * HC + j) * s2 + *base.add(2 * HC + i * HC + j);
                    if row[j] > mx {
                        mx = row[j];
                    }
                    j += 1;
                }
                let mut sum = 0f32;
                let mut j = 0;
                while j < HC {
                    row[j] = exp(row[j] - mx);
                    sum += row[j];
                    j += 1;
                }
                let mut j = 0;
                while j < HC {
                    cb[i][j] = row[j] / sum + HC_EPS;
                    j += 1;
                }
                i += 1;
            }
            // Sinkhorn-Knopp: columns first, then rows and columns HC_SINKHORN_ITERS - 1 times
            let mut it = 0;
            while it < HC_SINKHORN_ITERS {
                if it > 0 {
                    let mut i = 0;
                    while i < HC {
                        let mut sum = 0f32;
                        let mut j = 0;
                        while j < HC {
                            sum += cb[i][j];
                            j += 1;
                        }
                        let sum = sum + HC_EPS;
                        let mut j = 0;
                        while j < HC {
                            cb[i][j] /= sum;
                            j += 1;
                        }
                        i += 1;
                    }
                }
                let mut j = 0;
                while j < HC {
                    let mut sum = 0f32;
                    let mut i = 0;
                    while i < HC {
                        sum += cb[i][j];
                        i += 1;
                    }
                    let sum = sum + HC_EPS;
                    let mut i = 0;
                    while i < HC {
                        cb[i][j] /= sum;
                        i += 1;
                    }
                    j += 1;
                }
                it += 1;
            }
            // one thread writes the mix, with constant indices (indexing the arrays by thread spilled them)
            if t == 0 {
                let mut i = 0;
                while i < HC {
                    *mix.add(i) = pre[i];
                    *mix.add(HC + i) = post[i];
                    let mut j = 0;
                    while j < HC {
                        *mix.add(2 * HC + i * HC + j) = cb[i][j];
                        j += 1;
                    }
                    i += 1;
                }
            }
            // the collapse: this thread's channels
            let mut d = t;
            while d < H {
                let mut acc = 0f32;
                let mut i = 0;
                while i < HC {
                    acc += pre[i] * *streams.add(i * H + d);
                    i += 1;
                }
                *x.add(d) = acc;
                if !x_host.is_null() {
                    *x_host.add(d) = acc;
                }
                d += 256;
            }
        }
    }

    /// `stream_j[d] = post_j · y[d] + Σ_i comb_ij · stream_i[d]`, a thread a channel, the block
    /// output `y` the sum of `ya` and `yb` (`yb` null for one source): the parts of a split
    /// attention block, or the shared expert's rows and the host's routed experts. Grid: CHANNEL_GRID.
    #[no_mangle]
    pub extern "gpu-kernel" fn hc_post(mix: *const f32, ya: *const f32, yb: *const f32, streams: *mut f32) {
        unsafe {
            let d = workgroup_id_x() as usize * 256 + workitem_id_x() as usize;
            let mut res = [0f32; HC];
            let mut i = 0;
            while i < HC {
                res[i] = *streams.add(i * H + d);
                i += 1;
            }
            let yd = *ya.add(d) + if yb.is_null() { 0.0 } else { *yb.add(d) };
            let mut j = 0;
            while j < HC {
                let mut acc = 0f32;
                let mut i = 0;
                while i < HC {
                    acc += *mix.add(2 * HC + i * HC + j) * res[i];
                    i += 1;
                }
                *streams.add(j * H + d) = *mix.add(HC + j) * yd + acc;
                j += 1;
            }
        }
    }

    /// `mean[d] = Σ_j stream_j[d] / HC`, a thread a channel. Grid: CHANNEL_GRID.
    #[no_mangle]
    pub extern "gpu-kernel" fn hc_mean(streams: *const f32, mean: *mut f32) {
        unsafe {
            let d = workgroup_id_x() as usize * 256 + workitem_id_x() as usize;
            let mut acc = 0f32;
            let mut j = 0;
            while j < HC {
                acc += *streams.add(j * H + d);
                j += 1;
            }
            *mean.add(d) = acc / HC as f32;
        }
    }
    /// The handoff between the cards and the host, one launch: once everything launched before it
    /// has completed (the barrier bit), `n` floats from `src` (VRAM, a block output a kernel just
    /// emitted there) to `dst` (uncached host memory every card sees) in wide coalesced stores —
    /// a kernel emitting into uncached host memory row by row paid 200–300 µs a launch (timeline,
    /// 2026-10-15) — then `value` into `mine` (when not null: this card's data is in), then the
    /// queue holds until `w0` and `w1` (those not null) read `value`: the card waits for data,
    /// not the host for the card. The words are uncached host memory too, every load a trip over
    /// the link; one thread polls. Grid: 1.
    #[no_mangle]
    pub extern "gpu-kernel" fn hc_handoff(src: *const f32, dst: *mut f32, n: u32, mine: *mut u32, w0: *const u32, w1: *const u32, value: u32) {
        unsafe {
            let t = workitem_id_x();
            if !src.is_null() {
                let mut i = t;
                while i < n {
                    *dst.add(i as usize) = *src.add(i as usize);
                    i += 256;
                }
                fence(Ordering::Release);
                workgroup_sync();
            }
            if t == 0 {
                if !mine.is_null() {
                    AtomicU32::from_ptr(mine).store(value, Ordering::Release);
                }
                if !w0.is_null() {
                    while AtomicU32::from_ptr(w0 as *mut u32).load(Ordering::Relaxed) != value {}
                }
                if !w1.is_null() {
                    while AtomicU32::from_ptr(w1 as *mut u32).load(Ordering::Relaxed) != value {}
                }
            }
        }
    }
    const _: () = assert!(H % 256 == 0 && (HC * H) % 256 == 0 && HC * HC <= 256);
}

