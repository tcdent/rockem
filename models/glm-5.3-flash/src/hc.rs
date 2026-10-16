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
    /// `hc_pre`: the coefficient rows over the flattened streams, each over `SLICES` workgroups of `K_SLICE` channels.
    pub const COEF_ROWS: usize = HC_COEFS;
    /// Workgroups a row. 24 workgroups each norming the streams took 60 µs a launch, 192 took 49
    /// (timeline 2026-10-16): the norm is the cost, so now only the last workgroup computes it,
    /// and arrivals count per row before the row counts arrive, so no atomic sees more than 24.
    pub const SLICES: usize = 8;
    pub const K_SLICE: usize = HC * H / SLICES;
    pub const PRE_GRID: u32 = (COEF_ROWS * SLICES) as u32;
    pub const PRE_LDS: u32 = 512;
    /// `hc_pre`'s VRAM words: an arrival count a row, the launch's, the mix's ready word, the collapse's
    /// arrival count and ready word, the router's arrival count; the counts 0 between launches.
    pub const PRE_COUNTS: usize = COEF_ROWS + 5;
    /// Router rows a workgroup takes in `hc_pre`'s tail.
    pub const ROUTER_ROWS_PER_WG: usize = E.div_ceil(COEF_ROWS * SLICES);
    /// `hc_pre`'s partials in VRAM: the coefficient dots `[COEF_ROWS][SLICES]`, then every workgroup's sum of squares of its channel slice.
    pub const PRE_PARTS: usize = COEF_ROWS * SLICES + COEF_ROWS * SLICES;
    /// Channels of the collapse a workgroup takes.
    pub const COLLAPSE: usize = H.div_ceil(COEF_ROWS * SLICES);
    const _: () = assert!(HC * H % SLICES == 0 && K_SLICE % 256 == 0 && COEF_ROWS <= 256 && (16 + COEF_ROWS + 1) * 4 <= PRE_LDS as usize && COLLAPSE <= 256 && H % 256 == 0);
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
    //! input). The handoffs ride inside `hc_pre` (the routers' logits and words) and
    //! `hc_post` (block outputs out, the words, the wait).
    //! The per-element order is the host's (`pre`, `post`): multiply, then
    //! add, streams in order.
    pub use super::gfx11_consts::*;
    use crate::*;
    use common::gemv::widen;
    use common::gfx11::{lds, wave_sum, workgroup_sync};
    use common::math::{exp, sqrt};
    use core::arch::amdgpu::{workgroup_id_x, workitem_id_x};
    use core::sync::atomic::{fence, AtomicU32, Ordering};

    #[inline(always)]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + exp(-x))
    }

    /// The hyper-connection's input half as one launch over `COEF_ROWS · SLICES` workgroups.
    /// Workgroup (row, slice) takes the dot of its `K_SLICE` of row `row` with the raw stream
    /// channels into `part[row][slice]`, and the sum of squares of those channels alongside
    /// (the norm is a scalar), then counts its arrival (`count[row]`; the last of a row counts
    /// in `count[COEF_ROWS]`). The last workgroup of all sums the squares and each row's
    /// partials in fixed order, makes the mix (`hc::pre`'s arithmetic) into `mix` and publishes
    /// `value` in `count[COEF_ROWS + 1]`; every workgroup waits for that word (all resident:
    /// the grid is a fraction of the card) and collapses its `COLLAPSE` channels,
    /// `x[d] = Σ_i pre_i · stream_i[d]`, into `x` and, when `x_host` is not null, there too (the
    /// routed experts' input). One workgroup doing the collapse alone was the 50 µs (timeline
    /// 2026-10-16). Then, when `router` is not null (the FFN card: the early router over the
    /// attention block's input, or the FFN router over its own), the workgroups arrive again,
    /// each norms `x` with `ln` and takes its `ROUTER_ROWS_PER_WG` rows of the `[E][H]` bf16
    /// router straight into `scores` (host memory), and the last of them writes `value` to
    /// `word`: the two launches and two handoffs a layer the routers cost are inside this one.
    /// Counts back at 0 when the launch leaves. LDS: 512 bytes.
    #[no_mangle]
    pub extern "gpu-kernel" fn hc_pre(streams: *const f32, w: *const u16, part: *mut f32, base: *const f32, scale: *const f32, mix: *mut f32, x: *mut f32, x_host: *mut f32, count: *mut u32, value: u32, ln: *const u16, router: *const u16, scores: *mut f32, word: *mut u32) {
        unsafe {
            let t = workitem_id_x() as usize;
            let (wave, lane) = (t / 32, (t % 32) as u32);
            let wg = workgroup_id_x() as usize;
            let (row, slice) = (wg / SLICES, wg % SLICES);
            let red = lds::<f32>(); // [8] dot partials | [8] square partials | [COEF_ROWS] coefficients | last
            let red2 = red.add(8);
            let coefs = red.add(16);
            let last = red.add(16 + COEF_ROWS) as *mut u32;
            let ready = count.add(COEF_ROWS + 1);
            // this slice of the row against the raw channels, and the channels' squares
            let (w_row, c0) = (w.add(row * HC * H), slice * K_SLICE);
            let (mut acc, mut ss) = (0f32, 0f32);
            let mut i = t;
            while i < K_SLICE {
                let v = *streams.add(c0 + i);
                acc += v * widen(*w_row.add(c0 + i));
                ss += v * v;
                i += 256;
            }
            acc = wave_sum(lane, acc);
            ss = wave_sum(lane, ss);
            if lane == 0 {
                *red.add(wave) = acc;
                *red2.add(wave) = ss;
            }
            workgroup_sync();
            if t == 0 {
                let (mut p, mut q) = (0f32, 0f32);
                let mut k = 0;
                while k < 8 {
                    p += *red.add(k);
                    q += *red2.add(k);
                    k += 1;
                }
                // the partials past this CU's caches, then the arrivals: this row's, and for its last, the launch's
                AtomicU32::from_ptr(part.add(row * SLICES + slice) as *mut u32).store(p.to_bits(), Ordering::Release);
                AtomicU32::from_ptr(part.add(COEF_ROWS * SLICES + wg) as *mut u32).store(q.to_bits(), Ordering::Release);
                let row_done = AtomicU32::from_ptr(count.add(row)).fetch_add(1, Ordering::AcqRel) == SLICES as u32 - 1;
                let mut is_last = 0u32;
                if row_done {
                    *count.add(row) = 0;
                    is_last = (AtomicU32::from_ptr(count.add(COEF_ROWS)).fetch_add(1, Ordering::AcqRel) == COEF_ROWS as u32 - 1) as u32;
                }
                *last = is_last;
            }
            workgroup_sync();
            if *last != 0 {
                if t == 0 {
                    *count.add(COEF_ROWS) = 0; // nothing else touches it until the next launch
                }
                // the norm from one row's slices' squares (every row's slices read the same channels; one row
                // covers them all), the coefficients from every row's partials: fixed order
                let mut tot = 0f32;
                let mut k = 0;
                while k < SLICES {
                    tot += f32::from_bits(AtomicU32::from_ptr(part.add(COEF_ROWS * SLICES + k) as *mut u32).load(Ordering::Acquire));
                    k += 1;
                }
                let r = 1.0 / sqrt(tot / (HC * H) as f32 + RMS_NORM_EPS);
                if t < COEF_ROWS {
                    let mut sum = 0f32;
                    let mut k = 0;
                    while k < SLICES {
                        sum += f32::from_bits(AtomicU32::from_ptr(part.add(t * SLICES + k) as *mut u32).load(Ordering::Acquire));
                        k += 1;
                    }
                    *coefs.add(t) = sum * r;
                }
                workgroup_sync();
                // every thread computes the whole mix: a few hundred operations, no exchange needed
                let c = |i: usize| *coefs.add(i);
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
                // one thread writes the mix past its caches, with constant indices, then publishes it
                if t == 0 {
                    let mut i = 0;
                    while i < HC {
                        AtomicU32::from_ptr(mix.add(i) as *mut u32).store(pre[i].to_bits(), Ordering::Relaxed);
                        AtomicU32::from_ptr(mix.add(HC + i) as *mut u32).store(post[i].to_bits(), Ordering::Relaxed);
                        let mut j = 0;
                        while j < HC {
                            AtomicU32::from_ptr(mix.add(2 * HC + i * HC + j) as *mut u32).store(cb[i][j].to_bits(), Ordering::Relaxed);
                            j += 1;
                        }
                        i += 1;
                    }
                    AtomicU32::from_ptr(ready).store(value, Ordering::Release);
                }
            }
            // every workgroup: the mix, then its channels of the collapse
            if t == 0 {
                while AtomicU32::from_ptr(ready).load(Ordering::Acquire) != value {}
            }
            workgroup_sync();
            let mut pre = [0f32; HC];
            let mut i = 0;
            while i < HC {
                pre[i] = f32::from_bits(AtomicU32::from_ptr(mix.add(i) as *mut u32).load(Ordering::Relaxed));
                i += 1;
            }
            let d0 = wg * COLLAPSE;
            let mut d = d0 + t;
            while d < d0 + COLLAPSE && d < H {
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
            if router.is_null() {
                return;
            }
            // the collapse's arrival: this workgroup's channels to L2, the last publishes; every workgroup waits
            let (arrive2, ready2, arrive3) = (count.add(COEF_ROWS + 2), count.add(COEF_ROWS + 3), count.add(COEF_ROWS + 4));
            fence(Ordering::Release);
            workgroup_sync();
            if t == 0 {
                if AtomicU32::from_ptr(arrive2).fetch_add(1, Ordering::AcqRel) == (COEF_ROWS * SLICES) as u32 - 1 {
                    *arrive2 = 0;
                    AtomicU32::from_ptr(ready2).store(value, Ordering::Release);
                }
                while AtomicU32::from_ptr(ready2).load(Ordering::Acquire) != value {}
            }
            workgroup_sync();
            fence(Ordering::Acquire);
            // the router over the normed block input: the norm from every channel, this workgroup's rows
            let mut ss = 0f32;
            let mut k = t;
            while k < H {
                let v = *x.add(k);
                ss += v * v;
                k += 256;
            }
            ss = wave_sum(lane, ss);
            if lane == 0 {
                *red.add(wave) = ss;
            }
            workgroup_sync();
            let mut tot = 0f32;
            let mut k = 0;
            while k < 8 {
                tot += *red.add(k);
                k += 1;
            }
            let rr = 1.0 / sqrt(tot / H as f32 + RMS_NORM_EPS);
            workgroup_sync(); // red is reused below
            let mut r = 0;
            while r < ROUTER_ROWS_PER_WG {
                let row = wg + r * COEF_ROWS * SLICES;
                if row < E {
                    let wr = router.add(row * H);
                    let mut acc = 0f32;
                    let mut k = t;
                    while k < H {
                        acc += *x.add(k) * widen(*ln.add(k)) * widen(*wr.add(k));
                        k += 256;
                    }
                    acc = wave_sum(lane, acc);
                    if lane == 0 {
                        *red.add(wave) = acc;
                    }
                    workgroup_sync();
                    if t == 0 {
                        let mut sum = 0f32;
                        let mut k = 0;
                        while k < 8 {
                            sum += *red.add(k);
                            k += 1;
                        }
                        *scores.add(row) = sum * rr;
                    }
                    workgroup_sync();
                }
                r += 1;
            }
            // the router's arrival: this workgroup's scores out over the link, the last says so
            fence(Ordering::Release);
            workgroup_sync();
            if t == 0 && AtomicU32::from_ptr(arrive3).fetch_add(1, Ordering::AcqRel) == (COEF_ROWS * SLICES) as u32 - 1 {
                *arrive3 = 0;
                AtomicU32::from_ptr(word).store(value, Ordering::Release);
            }
        }
    }

    /// `stream_j[d] = post_j · y[d] + Σ_i comb_ij · stream_i[d]`, a thread a channel, the block
    /// output `y` the sum of `ya` and `yb` (`yb` null for one source): the parts of a split
    /// attention block, or the shared expert's rows and the host's routed experts. The handoff
    /// first, so the update is one launch, not two: workgroup 0 copies this
    /// card's `n` floats from `src` out to `dst`, writes `value` to `mine`, holds until `w0` and
    /// `w1` (those not null) read `value`, then releases the other workgroups through `go`, a
    /// VRAM word (one poller over the link a card: sixteen slowed the host's experts by 0.5 ms a
    /// token, timeline 2026-10-16). Grid: CHANNEL_GRID.
    #[no_mangle]
    pub extern "gpu-kernel" fn hc_post(mix: *const f32, ya: *const f32, yb: *const f32, streams: *mut f32, src: *const f32, dst: *mut f32, n: u32, mine: *mut u32, w0: *const u32, w1: *const u32, value: u32, go: *mut u32) {
        unsafe {
            let t = workitem_id_x();
            if workgroup_id_x() == 0 {
                if !src.is_null() {
                    let mut i = t;
                    while i < n {
                        *dst.add(i as usize) = *src.add(i as usize);
                        i += 256;
                    }
                    fence(Ordering::Release);
                    workgroup_sync();
                }
                // one poller a card over the link: this card's own word included, so no workgroup reads the
                // shared buffer before the copy is out; then the others released through a VRAM word
                if t == 0 {
                    if !mine.is_null() {
                        AtomicU32::from_ptr(mine).store(value, Ordering::Release);
                        while AtomicU32::from_ptr(mine).load(Ordering::Relaxed) != value {}
                    }
                    if !w0.is_null() {
                        while AtomicU32::from_ptr(w0 as *mut u32).load(Ordering::Relaxed) != value {}
                    }
                    if !w1.is_null() {
                        while AtomicU32::from_ptr(w1 as *mut u32).load(Ordering::Relaxed) != value {}
                    }
                    AtomicU32::from_ptr(go).store(value, Ordering::Release);
                }
            } else if t == 0 {
                while AtomicU32::from_ptr(go).load(Ordering::Acquire) != value {}
            }
            workgroup_sync();
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
    const _: () = assert!(H % 256 == 0 && (HC * H) % 256 == 0 && HC * HC <= 256);
}

