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
    /// Tokens of a step `hc_pre` holds in LDS at a time: a step of more goes through in rounds of this
    /// many (a thread a token in the partials' sum). The LDS sets the grid's residency, which the
    /// kernel's wait for its last workgroup needs: 192 workgroups at 21 KB are 3 a CU on 84 CUs; at a
    /// step of 256 tokens in one round (41 KB, 1 a CU) the grid could never all be resident and the
    /// first launch never completed (2026-10-16, a card reset).
    pub const PRE_TOKENS: usize = 128;
    /// `hc_pre`'s LDS: per token of a round, 8 wave partials of the dots and 8 of the squares, the
    /// coefficients; then a transpose scratch per wave.
    pub const PRE_LDS: u32 = (4 * (PRE_TOKENS * (16 + COEF_ROWS) + 8 * HC * HC)) as u32;
    /// `hc_pre`'s partials in VRAM, per token: the coefficient dots `[COEF_ROWS][SLICES]`, then every
    /// workgroup's sum of squares of its stream channels. The host allocates `STEP · PRE_STRIDE`.
    pub const PRE_STRIDE: usize = PRE_PARTS;
    /// `hc_pre`'s VRAM words: for each of its three device barriers, an arrival count a row (0 between
    /// launches) and a done word a row; then the mix's ready word and the collapse's.
    pub const PRE_BARRIERS: usize = 3;
    pub const PRE_COUNTS: usize = PRE_BARRIERS * 2 * COEF_ROWS + 2;
    /// Router rows a workgroup takes in `hc_pre`'s tail.
    /// Rows the FFN half's router launch carries: layer l's router over the FFN input, then layer l+1's
    /// (the prefetch's predictor, named a layer ahead); `scores` is `[tokens][ROUTE_ROWS]`.
    pub const ROUTE_ROWS: usize = 2 * E;
    pub const ROUTER_ROWS_PER_WG: usize = ROUTE_ROWS.div_ceil(COEF_ROWS * SLICES);
    /// `hc_pre`'s partials in VRAM: the coefficient dots `[COEF_ROWS][SLICES]`, then every workgroup's sum of squares of its channel slice.
    pub const PRE_PARTS: usize = COEF_ROWS * SLICES + COEF_ROWS * SLICES;
    /// Channels of the collapse a workgroup takes.
    pub const COLLAPSE: usize = H.div_ceil(COEF_ROWS * SLICES);
    const _: () = assert!(HC * H % SLICES == 0 && K_SLICE % 256 == 0 && COEF_ROWS <= 256 && PRE_LDS <= 60 * 1024 && COEF_ROWS <= 32 && COLLAPSE <= 256 && H % 256 == 0 && HC * HC <= 32 && HC.is_power_of_two() && PRE_TOKENS <= 256 && K_SLICE / 256 == 8);
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
    use common::gfx11::{await_word, group_max, group_sum, lane_read, lds, nap, wave_sum, workgroup_sync};
    use common::math::{exp, sqrt};
    use core::arch::amdgpu::{workgroup_id_x, workitem_id_x};
    use core::sync::atomic::{fence, AtomicU32, Ordering};

    #[inline(always)]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + exp(-x))
    }

    /// A device-wide barrier over `hc_pre`'s `COEF_ROWS · SLICES` workgroups with no serialized step:
    /// the workgroups of a row count on the row's word (`SLICES` arrivals, the rows in parallel), the
    /// row's last resets it and writes `value` to the row's done word, and workgroup 0's wave 0 watches
    /// the done words, a lane each, until all read `value`. Returns on workgroup 0 only, once every
    /// workgroup has arrived; the others return from `await_word` on whatever workgroup 0 publishes.
    /// `words`: `[COEF_ROWS]` counts then `[COEF_ROWS]` done words. Twenty-four arrivals serialized
    /// on one word cost more than the dots they followed (bench-hc, 2026-10-16).
    #[inline(always)]
    unsafe fn arrive(words: *mut u32, row: usize, t: usize, wave: usize, lane: u32, value: u32) -> bool {
        let wg0 = workgroup_id_x() == 0;
        workgroup_sync();
        if t == 0 {
            if AtomicU32::from_ptr(words.add(row)).fetch_add(1, Ordering::AcqRel) == SLICES as u32 - 1 {
                *words.add(row) = 0;
                AtomicU32::from_ptr(words.add(COEF_ROWS + row)).store(value, Ordering::Release);
            }
        }
        if wg0 && wave == 0 {
            let w = words.add(COEF_ROWS + if (lane as usize) < COEF_ROWS { lane as usize } else { 0 });
            loop {
                let seen = AtomicU32::from_ptr(w).load(Ordering::Acquire) == value;
                if wave_sum(lane, if seen { 1.0 } else { 0.0 }) >= 32.0 {
                    break;
                }
                nap();
            }
        }
        wg0
    }

    /// The hyper-connection's input half as one launch over `COEF_ROWS · SLICES` workgroups, for
    /// the `n` tokens of a step (`streams: [n][HC][H]`, `mix: [n][MIX]`, `x`, `x_host: [n][H]`,
    /// `scores: [n][E]`, `part: [n][PRE_STRIDE]`). Workgroup (row, slice) takes, for every token,
    /// the dot of its `K_SLICE` of row `row` with the raw stream channels and the sum of squares of
    /// those channels (its eight weights a thread read once, the tokens looped), the wave partials
    /// kept in LDS and summed after one barrier, then counts its arrival (`count[row]`). The last
    /// workgroup of all makes every token's mix (`hc::pre`'s arithmetic: the norm from one row's
    /// slices' squares, the coefficients from every row's partials in fixed order, the sigmoids,
    /// softmax rows and Sinkhorn on a wave a token, eight tokens at a time) into `mix` and
    /// publishes `value` in `count`'s ready word; every workgroup waits for it (all resident: the
    /// grid is a fraction of the card) and collapses its `COLLAPSE` channels of every token,
    /// `x[i][d] = Σ_j pre_ij · stream_ij[d]`, into `x` and, when `x_host` is not null, there too
    /// (the routed experts' input). Then, when `router` is not null (the FFN card: the early router over the
    /// attention block's input, or the FFN router over its own), the workgroups arrive again, each
    /// takes its `ROUTER_ROWS_PER_WG` rows of the `[ROUTE_ROWS][H]` bf16 router (the first `rows` live) over every token's `x`
    /// normed with `ln` (a wave a token, no reduction through LDS) straight into `scores` (host
    /// memory), and the last of them writes `value` to `word`. One token is n = 1 through the
    /// same launch; the tokens go through the LDS `PRE_TOKENS` at a time (the partials, then the
    /// coefficients), a token's arithmetic the same whichever round it is in. Counts back at 0 when
    /// the launch leaves. LDS: PRE_LDS.
    #[no_mangle]
    pub extern "gpu-kernel" fn hc_pre(streams: *const f32, w: *const u16, part: *mut f32, base: *const f32, scale: *const f32, mix: *mut f32, x: *mut f32, x_host: *mut f32, count: *mut u32, value: u32, ln: *const u16, router: *const u16, scores: *mut f32, word: *mut u32, rows: u32, n: u32) {
        unsafe {
            let n = n as usize;
            let t = workitem_id_x() as usize;
            let (wave, lane) = (t / 32, (t % 32) as u32);
            let wg = workgroup_id_x() as usize;
            let (row, slice) = (wg / SLICES, wg % SLICES);
            let l = lds::<f32>();
            let red = l; // [PRE_TOKENS][8] dot partials
            let red2 = l.add(PRE_TOKENS * 8); // [PRE_TOKENS][8] square partials
            let coefs = l.add(PRE_TOKENS * 16); // [PRE_TOKENS][COEF_ROWS]
            let trs = l.add(PRE_TOKENS * (16 + COEF_ROWS)); // [8 waves][HC·HC] transpose scratch
            let (b0, b1, b2) = (count, count.add(2 * COEF_ROWS), count.add(4 * COEF_ROWS));
            let ready = count.add(PRE_BARRIERS * 2 * COEF_ROWS);
            // this slice of the row: the thread's eight weights once, every token's channels against them
            let (w_row, c0) = (w.add(row * HC * H), slice * K_SLICE);
            let mut wv = [0f32; 8];
            let mut k = 0;
            while k < 8 {
                wv[k] = widen(*w_row.add(c0 + k * 256 + t));
                k += 1;
            }
            // the tokens a round at a time: each one's wave partials into LDS, then a thread a token
            let mut r0 = 0;
            while r0 < n {
                let m = if n - r0 < PRE_TOKENS { n - r0 } else { PRE_TOKENS };
                let mut i = 0;
                while i < m {
                    let s_tok = streams.add((r0 + i) * HC * H + c0);
                    let (mut acc, mut ss) = (0f32, 0f32);
                    let mut k = 0;
                    while k < 8 {
                        let v = *s_tok.add(k * 256 + t);
                        acc += v * wv[k];
                        ss += v * v;
                        k += 1;
                    }
                    acc = wave_sum(lane, acc);
                    ss = wave_sum(lane, ss);
                    if lane == 0 {
                        *red.add(i * 8 + wave) = acc;
                        *red2.add(i * 8 + wave) = ss;
                    }
                    i += 1;
                }
                workgroup_sync();
                // a thread a token: the eight partials summed in order, past this CU's caches
                if t < m {
                    let (mut p, mut q) = (0f32, 0f32);
                    let mut k = 0;
                    while k < 8 {
                        p += *red.add(t * 8 + k);
                        q += *red2.add(t * 8 + k);
                        k += 1;
                    }
                    let pt = part.add((r0 + t) * PRE_STRIDE);
                    AtomicU32::from_ptr(pt.add(row * SLICES + slice) as *mut u32).store(p.to_bits(), Ordering::Release);
                    AtomicU32::from_ptr(pt.add(COEF_ROWS * SLICES + wg) as *mut u32).store(q.to_bits(), Ordering::Release);
                }
                r0 += PRE_TOKENS;
                if r0 < n {
                    workgroup_sync(); // the partials' LDS is the next round's
                }
            }
            if arrive(b0, row, t, wave, lane, value) {
                // every token's coefficients, a round of tokens at a time: the norm from one row's slices'
                // squares (every row's slices read the same channels; one row covers them all), the rows'
                // partials in fixed order
                let mut r0 = 0;
                while r0 < n {
                    let m = if n - r0 < PRE_TOKENS { n - r0 } else { PRE_TOKENS };
                    if t < COEF_ROWS {
                        let mut i = 0;
                        while i < m {
                            let pt = part.add((r0 + i) * PRE_STRIDE);
                            let mut tot = 0f32;
                            let mut k = 0;
                            while k < SLICES {
                                tot += f32::from_bits(AtomicU32::from_ptr(pt.add(COEF_ROWS * SLICES + k) as *mut u32).load(Ordering::Acquire));
                                k += 1;
                            }
                            let rn = 1.0 / sqrt(tot / (HC * H) as f32 + RMS_NORM_EPS);
                            let mut sum = 0f32;
                            let mut k = 0;
                            while k < SLICES {
                                sum += f32::from_bits(AtomicU32::from_ptr(pt.add(t * SLICES + k) as *mut u32).load(Ordering::Acquire));
                                k += 1;
                            }
                            *coefs.add(i * COEF_ROWS + t) = sum * rn;
                            i += 1;
                        }
                    }
                    workgroup_sync();
                    // the mix on a wave a token, a lane an entry (lane = i·HC + j): the sigmoids on lanes 0..2·HC,
                    // the softmax rows and the Sinkhorn passes as group sums over the row's four lanes and the
                    // column's four (a transpose through LDS turns columns into rows), eight tokens at a time
                    let (s0, s1, s2) = (*scale, *scale.add(1), *scale.add(2));
                    let tr = trs.add(wave * HC * HC);
                    let mut i = wave;
                    while i < m {
                        let c = |k: usize| *coefs.add(i * COEF_ROWS + k);
                        let lu = lane as usize;
                        let (li, lj) = (lu / HC, lu % HC);
                        let inb = lu < HC * HC;
                        let small = if lu < HC { lu } else { 0 };
                        let pre_v = sigmoid(c(small) * s0 + *base.add(small)) + HC_EPS;
                        let post_v = 2.0 * sigmoid(c(HC + small) * s1 + *base.add(HC + small));
                        let mut e = if inb { c(2 * HC + li * HC + lj) * s2 + *base.add(2 * HC + li * HC + lj) } else { f32::NEG_INFINITY };
                        let mx = group_max(lane, e, HC as u32);
                        e = if inb { exp(e - mx) } else { 0.0 };
                        let sum = group_sum(lane, e, HC as u32);
                        let mut cbv = if inb { e / sum + HC_EPS } else { 0.0 };
                        let mut it = 0;
                        while it < HC_SINKHORN_ITERS {
                            if it > 0 {
                                let rs = group_sum(lane, cbv, HC as u32) + HC_EPS;
                                cbv /= rs;
                            }
                            if inb {
                                *tr.add(lj * HC + li) = cbv;
                            }
                            let mirror = if inb { *tr.add(lu) } else { 0.0 };
                            let cs_t = group_sum(lane, mirror, HC as u32) + HC_EPS;
                            let cs = lane_read((lj * HC) as u32, cs_t);
                            cbv /= cs;
                            it += 1;
                        }
                        let mt = mix.add((r0 + i) * MIX);
                        if inb {
                            AtomicU32::from_ptr(mt.add(2 * HC + lu) as *mut u32).store(cbv.to_bits(), Ordering::Relaxed);
                        }
                        if lu < HC {
                            AtomicU32::from_ptr(mt.add(lu) as *mut u32).store(pre_v.to_bits(), Ordering::Relaxed);
                            AtomicU32::from_ptr(mt.add(HC + lu) as *mut u32).store(post_v.to_bits(), Ordering::Relaxed);
                        }
                        i += 8;
                    }
                    workgroup_sync(); // the coefficients' LDS is the next round's; after the last, the mix is written
                    r0 += PRE_TOKENS;
                }
                if t == 0 {
                    AtomicU32::from_ptr(ready).store(value, Ordering::Release);
                }
            }
            // every workgroup: each token's mix, then its channels of the token's collapse, and their squares
            if t == 0 {
                await_word(ready, value);
            }
            workgroup_sync();
            let d0 = wg * COLLAPSE;
            let d = d0 + t;
            if t < COLLAPSE && d < H {
                let mut i = 0;
                while i < n {
                    let (mt, st) = (mix.add(i * MIX), streams.add(i * HC * H));
                    let mut acc = 0f32;
                    let mut j = 0;
                    while j < HC {
                        acc += f32::from_bits(AtomicU32::from_ptr(mt.add(j) as *mut u32).load(Ordering::Relaxed)) * *st.add(j * H + d);
                        j += 1;
                    }
                    *x.add(i * H + d) = acc;
                    if !x_host.is_null() {
                        *x_host.add(i * H + d) = acc;
                    }
                    i += 1;
                }
            }
            if router.is_null() {
                return;
            }
            // the collapse's arrival: this workgroup's channels to L2; workgroup 0 publishes; every workgroup waits
            let ready2 = ready.add(1);
            fence(Ordering::Release);
            if arrive(b1, row, t, wave, lane, value) && t == 0 {
                AtomicU32::from_ptr(ready2).store(value, Ordering::Release);
            }
            if t == 0 {
                await_word(ready2, value);
            }
            workgroup_sync();
            fence(Ordering::Acquire);
            // the router over every token's normed block input, this workgroup's rows. A few tokens: the
            // 256-thread reduction a (token, row) at a time (two barriers each) — a token at a time
            // keeps every wave busy. Many: a wave a token, the norm and every dot in that same order
            // (thread lane + 32·w over elements 256 apart, each wave's lanes summed, the eight waves in
            // order), so a score's bits are the same either way.
            if n < 8 {
                let mut i = 0;
                while i < n {
                    let xt = x.add(i * H);
                    let mut ss = 0f32;
                    let mut k = t;
                    while k < H {
                        let v = *xt.add(k);
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
                        let rrow = wg + r * COEF_ROWS * SLICES;
                        if rrow < rows as usize {
                            let wr = router.add(rrow * H);
                            let mut acc = 0f32;
                            let mut k = t;
                            while k < H {
                                acc += *xt.add(k) * widen(*ln.add(k)) * widen(*wr.add(k));
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
                                *scores.add(i * ROUTE_ROWS + rrow) = sum * rr;
                            }
                            workgroup_sync();
                        }
                        r += 1;
                    }
                    i += 1;
                }
            }
            let mut i = wave;
            while i < n && n >= 8 {
                let xt = x.add(i * H);
                let mut tot = 0f32;
                let mut w = 0;
                while w < 8 {
                    let mut part_ss = 0f32;
                    let mut k = lane as usize + 32 * w;
                    while k < H {
                        let v = *xt.add(k);
                        part_ss += v * v;
                        k += 256;
                    }
                    tot += wave_sum(lane, part_ss);
                    w += 1;
                }
                let rr = 1.0 / sqrt(tot / H as f32 + RMS_NORM_EPS);
                let mut r = 0;
                while r < ROUTER_ROWS_PER_WG {
                    let rrow = wg + r * COEF_ROWS * SLICES;
                    if rrow < rows as usize {
                        let wr = router.add(rrow * H);
                        // the workgroup's order on one wave, so a score's bits are what the 256-thread
                        // reduction gave: thread (lane + 32·w) summed elements lane + 32·w + 256·j in
                        // order, each wave's lanes summed, the eight waves summed in order
                        let mut acc = 0f32;
                        let mut w = 0;
                        while w < 8 {
                            let mut part = 0f32;
                            let mut k = lane as usize + 32 * w;
                            while k < H {
                                part += *xt.add(k) * widen(*ln.add(k)) * widen(*wr.add(k));
                                k += 256;
                            }
                            acc += wave_sum(lane, part);
                            w += 1;
                        }
                        if lane == 0 {
                            *scores.add(i * ROUTE_ROWS + rrow) = acc * rr;
                        }
                    }
                    r += 1;
                }
                i += 8;
            }
            // the router's arrival: this workgroup's scores out over the link; workgroup 0 says so
            fence(Ordering::Release);
            if arrive(b2, row, t, wave, lane, value) && t == 0 {
                AtomicU32::from_ptr(word).store(value, Ordering::Release);
            }
        }
    }

    /// `stream_ij[d] = post_ij · y_i[d] + Σ_k comb_ikj · stream_ik[d]` for the `tokens` tokens of a
    /// step (`mix: [tokens][MIX]`, `streams: [tokens][HC][H]`, `ya`, `yb: [tokens][H]`), a thread a
    /// channel looping the tokens, the block output `y` the sum of `ya` and `yb` (`yb` null for one
    /// source): the parts of a split attention block, or the shared expert's rows and the host's
    /// routed experts. The handoff first, so the update is one launch, not two: workgroup 0 copies
    /// this card's `n` floats from `src` out to `dst`, writes `value` to `mine`, holds until `w0`
    /// and `w1` (those not null) read `value`, then releases the other workgroups through `go`, a
    /// VRAM word (one poller over the link a card: sixteen slowed the host's experts by 0.5 ms a
    /// token, timeline 2026-10-16). Grid: CHANNEL_GRID.
    #[no_mangle]
    pub extern "gpu-kernel" fn hc_post(mix: *const f32, ya: *const f32, yb: *const f32, streams: *mut f32, src: *const f32, dst: *mut f32, n: u32, mine: *mut u32, w0: *const u32, w1: *const u32, value: u32, go: *mut u32, tokens: u32) {
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
                await_word(go, value);
            }
            workgroup_sync();
            let d = workgroup_id_x() as usize * 256 + workitem_id_x() as usize;
            let mut i = 0;
            while i < tokens as usize {
                let (mt, st) = (mix.add(i * MIX), streams.add(i * HC * H));
                let mut res = [0f32; HC];
                let mut k = 0;
                while k < HC {
                    res[k] = *st.add(k * H + d);
                    k += 1;
                }
                let yd = *ya.add(i * H + d) + if yb.is_null() { 0.0 } else { *yb.add(i * H + d) };
                let mut j = 0;
                while j < HC {
                    let mut acc = 0f32;
                    let mut k = 0;
                    while k < HC {
                        acc += *mt.add(2 * HC + k * HC + j) * res[k];
                        k += 1;
                    }
                    *st.add(j * H + d) = *mt.add(HC + j) * yd + acc;
                    j += 1;
                }
                i += 1;
            }
        }
    }

    /// `x[i] += a[i] + b[i]` over `n` floats (`b` null for one addend): the draft layer's plain
    /// residual. Grid: n.div_ceil(256).
    #[no_mangle]
    pub extern "gpu-kernel" fn add2(x: *mut f32, a: *const f32, b: *const f32, n: u32) {
        unsafe {
            let i = workgroup_id_x() as usize * 256 + workitem_id_x() as usize;
            if i < n as usize {
                *x.add(i) += *a.add(i) + if b.is_null() { 0.0 } else { *b.add(i) };
            }
        }
    }
    /// `mean[i][d] = Σ_j stream_i,j[d] / HC` for `n` tokens (`streams: [n][HC][H]`), a thread a
    /// channel looping the tokens. Grid: CHANNEL_GRID.
    #[no_mangle]
    pub extern "gpu-kernel" fn hc_mean(streams: *const f32, mean: *mut f32, n: u32) {
        unsafe {
            let d = workgroup_id_x() as usize * 256 + workitem_id_x() as usize;
            let mut i = 0;
            while i < n as usize {
                let st = streams.add(i * HC * H);
                let mut acc = 0f32;
                let mut j = 0;
                while j < HC {
                    acc += *st.add(j * H + d);
                    j += 1;
                }
                *mean.add(i * H + d) = acc / HC as f32;
                i += 1;
            }
        }
    }
    const _: () = assert!(H % 256 == 0 && (HC * H) % 256 == 0 && HC * HC <= 256);
}

