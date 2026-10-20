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
    /// The LDS of the coefficient rounds, or of the FFN input's act phase (a token's 256 partials, the sigmoid
    /// scores and the bias an expert, the picks, the norm's scale and the mass), whichever is larger.
    const PRE_COEF_LDS: usize = 4 * (PRE_TOKENS * (16 + COEF_ROWS) + 8 * HC * HC);
    pub const ACT_LDS: usize = 4 * (256 + E + 2 * TOP_K + 8) + 8 * E;
    pub const PRE_LDS: u32 = (if PRE_COEF_LDS > ACT_LDS { PRE_COEF_LDS } else { ACT_LDS }) as u32;
    /// `hc_pre`'s act table (a MoE layer on its FFN card): the device addresses its act phase reads and writes,
    /// in this order — the post-norm (f32), the router's bias (f32), the card's quantized activation (`xq`, `xs`,
    /// `xg`), the host's copies of it, the host's picks (u32) and weights (f32), the card's copy of the weights
    /// (the leader's routed sum, queue 002) — each `[STEP]` tokens deep, the
    /// launch's tokens written from row `t0` (a prefill chunk's first, 037); the host's word is the launch's `act_word`.
    pub const ACT_SLOTS: usize = 11;
    /// `hc_pre`'s partials in VRAM, per token: the coefficient dots `[COEF_ROWS][SLICES]`, then every
    /// workgroup's sum of squares of its stream channels. The host allocates `STEP · PRE_STRIDE`.
    pub const PRE_STRIDE: usize = PRE_PARTS;
    /// `hc_pre`'s VRAM words: for each of its four device barriers (the dots, the collapse, the router, the mixes), an
    /// arrival count a row (0 between launches) and a done word a row; then the ready words.
    pub const PRE_BARRIERS: usize = 4;
    /// The arrival counts, then the ready words: the dots (every workgroup's mixes may start), the collapse, the scores (the
    /// act phase's start), the act phase's arrivals, `hc_route_ahead`'s arrivals, the mixes (the collapse may start).
    pub const PRE_COUNTS: usize = PRE_BARRIERS * 2 * COEF_ROWS + 6;
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
    const _: () = assert!(common::gemv::gfx11::WMMA_LDS <= PRE_LDS as usize, "the router's matrix path stages in hc_pre's LDS");

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
    /// the `n` tokens of a step (`streams: [n][HC][H]`, `mix: [n][MIX]`, `x: [n][H]`,
    /// `scores: [n][E]`, `part: [n][PRE_STRIDE]`). Workgroup (row, slice) takes, for every token,
    /// the dot of its `K_SLICE` of row `row` with the raw stream channels and the sum of squares of
    /// those channels (its eight weights a thread read once, the tokens looped), the wave partials
    /// kept in LDS and summed after one barrier, then counts its arrival (`count[row]`). The last
    /// workgroup of all makes every token's mix (`hc::pre`'s arithmetic: the norm from one row's
    /// slices' squares, the coefficients from every row's partials in fixed order, the sigmoids,
    /// softmax rows and Sinkhorn on a wave a token, eight tokens at a time) into `mix` and
    /// publishes `value` in `count`'s ready word; every workgroup waits for it (all resident: the
    /// grid is a fraction of the card) and collapses its `COLLAPSE` channels of every token,
    /// `x[i][d] = Σ_j pre_ij · stream_ij[d]`, into `x` (bring-up's host copy is a copy launch when it asks: the
    /// collapse once wrote 8 MB a prefill chunk to host memory that serving never read). Then, when `router` is not null (the FFN card: the early router over the
    /// attention block's input, or the FFN router over its own), the workgroups arrive again, each
    /// takes its `ROUTER_ROWS_PER_WG` rows of the `[ROUTE_ROWS][H]` bf16 router (the first `rows` live) over every token's `x`
    /// normed with `ln` (a wave a token, no reduction through LDS) straight into `scores` (host
    /// memory), and the last of them writes `value` to `word`. With `act` (a MoE layer's FFN card), every
    /// workgroup with a token then makes its FFN input (`act_token`) into `act`'s table at rows `t0..t0 + n`
    /// (the pointers above are the chunk's own, 037), and the last workgroup out writes `value` to
    /// `act_word`. One token is n = 1 through the
    /// same launch; the tokens go through the LDS `PRE_TOKENS` at a time (the partials, then the
    /// coefficients), a token's arithmetic the same whichever round it is in. Counts back at 0 when
    /// the launch leaves. LDS: PRE_LDS.
    #[no_mangle]
    pub extern "gpu-kernel" fn hc_pre(streams: *const f32, w: *const u16, part: *mut f32, base: *const f32, scale: *const f32, mix: *mut f32, x: *mut f32, count: *mut u32, value: u32, ln: *const u16, router: *const u16, scores: *mut f32, scores_v: *mut f32, word: *mut u32, act: *const u64, act_word: *mut u32, t0: u32, rows: u32, n: u32) {
        unsafe {
            let n = n as usize;
            // stats (a stats code object only), the launch's phases as workgroup 0 sees them: slot 0 the coefficient dots to
            // the grid's first arrival, 1 the mix and the collapse to the router's start (the plain launch's end), 2 the
            // router to every score in, 3 the workgroup's tokens' act phase
            let t = workitem_id_x() as usize;
            let (wave, lane) = (t / 32, (t % 32) as u32);
            let wg = workgroup_id_x() as usize;
            let (row, slice) = (wg / SLICES, wg % SLICES);
            let l = lds::<f32>();
            let red = l; // [PRE_TOKENS][8] dot partials
            let red2 = l.add(PRE_TOKENS * 8); // [PRE_TOKENS][8] square partials
            let coefs = l.add(PRE_TOKENS * 16); // [PRE_TOKENS][COEF_ROWS]
            let trs = l.add(PRE_TOKENS * (16 + COEF_ROWS)); // [8 waves][HC·HC] transpose scratch
            let (b0, b1, b2, b3) = (count, count.add(2 * COEF_ROWS), count.add(4 * COEF_ROWS), count.add(6 * COEF_ROWS));
            let ready = count.add(PRE_BARRIERS * 2 * COEF_ROWS);
            let kt0 = common::stat_clock!();
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
            let a0 = arrive(b0, row, t, wave, lane, value);
            let kt1 = common::stat_clock!();
            common::stat!(0, kt1.wrapping_sub(kt0) & common::stats::CLOCK_MASK);
            // the dots are in: workgroup 0 says so, every workgroup then makes its own tokens' mixes (tokens wg, wg + the grid,
            // ...: one workgroup made every token's, ~1.5–1.8 ms of a 512-token chunk while 191 waited, 2026-10-20)
            if a0 && t == 0 {
                AtomicU32::from_ptr(ready).store(value, Ordering::Release);
            }
            if t == 0 {
                await_word(ready, value);
            }
            workgroup_sync();
            {
                const GRID: usize = COEF_ROWS * SLICES;
                let tok = |k: usize| wg + k * GRID;
                let m = if wg < n { (n - wg).div_ceil(GRID) } else { 0 };
                {
                    // every token's coefficients: the norm from one row's slices' squares (every row's slices read
                    // the same channels; one row covers them all), the rows' partials in fixed order
                    if t < COEF_ROWS {
                        let mut i = 0;
                        while i < m {
                            let pt = part.add(tok(i) * PRE_STRIDE);
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
                        let mt = mix.add(tok(i) * MIX);
                        if inb {
                            AtomicU32::from_ptr(mt.add(2 * HC + lu) as *mut u32).store(cbv.to_bits(), Ordering::Relaxed);
                        }
                        if lu < HC {
                            AtomicU32::from_ptr(mt.add(lu) as *mut u32).store(pre_v.to_bits(), Ordering::Relaxed);
                            AtomicU32::from_ptr(mt.add(HC + lu) as *mut u32).store(post_v.to_bits(), Ordering::Relaxed);
                        }
                        i += 8;
                    }
                }
                // the mixes' arrival: every workgroup's tokens written; workgroup 0 says so
                fence(Ordering::Release);
                let ready_mix = ready.add(5);
                if arrive(b3, row, t, wave, lane, value) && t == 0 {
                    AtomicU32::from_ptr(ready_mix).store(value, Ordering::Release);
                }
                if t == 0 {
                    await_word(ready_mix, value);
                }
            }
            // every workgroup: each token's mix, then its channels of the token's collapse, and their squares
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
                    i += 1;
                }
            }
            if router.is_null() {
                let kt2 = common::stat_clock!();
                common::stat!(1, kt2.wrapping_sub(kt1) & common::stats::CLOCK_MASK);
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
            let kt2 = common::stat_clock!();
            common::stat!(1, kt2.wrapping_sub(kt1) & common::stats::CLOCK_MASK);
            // the router's rows `0..rows` (this layer's: the next layer's, the prefetch's predictor, run in their own launch
            // after this one, `hc_route_ahead` — a store to host memory leaves only at a launch's end-of-kernel release
            // here, so work behind the act phase's word in this launch held the word too, 2026-10-20)
            // a step past SPEC_TOKENS on the matrix path (the rows kernels' twin, the activations bf16 as staged): the first
            // `wmma_grid(rows, n)` workgroups a 128-row × 128-token block each; the decode-shaped loop re-read every token's
            // input a row pair a workgroup, 4.1 ms of a 512-token chunk (2026-10-20)
            if n > SPEC_TOKENS {
                use common::gemv::gfx11::{rows_wmma_into, wmma_grid, WBf16};
                let m = rows as usize;
                if wg < wmma_grid(m, n) as usize {
                    rows_wmma_into::<H, true, WBf16>(t as u32, wg, WBf16(router), ln, m, n, RMS_NORM_EPS, |i, k| *x.add(i * H + k), |tok, r, y| {
                        *scores.add(tok * ROUTE_ROWS + r) = y;
                        *scores_v.add(tok * ROUTE_ROWS + r) = y;
                    });
                }
            } else {
                router_rows(0, rows as usize, n, x, ln, router, scores, scores_v, red, t, wave, lane, wg);
            }
            // the router's arrival: this workgroup's scores out over the link; workgroup 0 says so
            fence(Ordering::Release);
            let ready3 = ready.add(2);
            if arrive(b2, row, t, wave, lane, value) && t == 0 {
                if !word.is_null() {
                    AtomicU32::from_ptr(word).store(value, Ordering::Release);
                }
                if !act.is_null() {
                    AtomicU32::from_ptr(ready3).store(value, Ordering::Release);
                }
            }
            if !act.is_null() {
                // the FFN input's host-side work (queue 001), once every score is in: a workgroup a token, the grid's
                // workgroups taking tokens wg, wg + PRE_GRID, ... — the post-norm, the quantization, the router's picks
                if wg < n {
                    if t == 0 {
                        await_word(ready3, value);
                    }
                    workgroup_sync();
                    fence(Ordering::Acquire);
                    let kt3 = common::stat_clock!();
                    common::stat!(2, kt3.wrapping_sub(kt2) & common::stats::CLOCK_MASK);
                    let mut tok = wg;
                    while tok < n {
                        act_token(tok, t0 as usize + tok, t, wave, lane, x, scores_v, act);
                        tok += COEF_ROWS * SLICES;
                    }
                    let kt4 = common::stat_clock!();
                    common::stat!(3, kt4.wrapping_sub(kt3) & common::stats::CLOCK_MASK);
                }
                // the act phase's arrival: the last workgroup out writes the host's word
                fence(Ordering::Release);
                workgroup_sync();
                if t == 0 {
                    let done = ready.add(3);
                    if AtomicU32::from_ptr(done).fetch_add(1, Ordering::AcqRel) == (COEF_ROWS * SLICES) as u32 - 1 {
                        AtomicU32::from_ptr(done).store(0, Ordering::Relaxed);
                        AtomicU32::from_ptr(act_word).store(value, Ordering::Release);
                    }
                }
            }
        }
    }

    /// The next MoE layer's router rows (`E..ROUTE_ROWS` of the FFN half's `[ROUTE_ROWS][H]` router) over the block input
    /// `x` the FFN card's `hc_pre` collapsed, into `scores` and `scores_v` as `hc_pre`'s own rows: the prefetch's
    /// predictor (005), in a launch of its own behind `hc_pre` so the act phase's word leaves with `hc_pre`'s end; the
    /// last workgroup out writes `value` to `word` (the router's, for the host). Grid PRE_GRID, `count` `hc_pre`'s.
    #[no_mangle]
    pub extern "gpu-kernel" fn hc_route_ahead(x: *const f32, ln: *const u16, router: *const u16, scores: *mut f32, scores_v: *mut f32, word: *mut u32, count: *mut u32, value: u32, n: u32) {
        unsafe {
            let t = workitem_id_x() as usize;
            let (wave, lane) = (t / 32, (t % 32) as u32);
            let wg = workgroup_id_x() as usize;
            router_rows(E, ROUTE_ROWS, n as usize, x, ln, router, scores, scores_v, lds::<f32>(), t, wave, lane, wg);
            fence(Ordering::Release);
            workgroup_sync();
            if t == 0 {
                let done = count.add(PRE_BARRIERS * 2 * COEF_ROWS + 4);
                if AtomicU32::from_ptr(done).fetch_add(1, Ordering::AcqRel) == PRE_GRID - 1 {
                    AtomicU32::from_ptr(done).store(0, Ordering::Relaxed);
                    AtomicU32::from_ptr(word).store(value, Ordering::Release);
                }
            }
        }
    }

    /// The router's rows `lo..hi` this workgroup takes (`wg + r · PRE_GRID`) over the launch's `n` tokens' block
    /// input `x` normed with `ln`, into `scores` (host memory) and `scores_v` (VRAM), `[n][ROUTE_ROWS]`.
    #[inline(always)]
    unsafe fn router_rows(lo: usize, hi: usize, n: usize, x: *const f32, ln: *const u16, router: *const u16, scores: *mut f32, scores_v: *mut f32, red: *mut f32, t: usize, wave: usize, lane: u32, wg: usize) {
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
                    if rrow >= lo && rrow < hi {
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
                            *scores_v.add(i * ROUTE_ROWS + rrow) = sum * rr;
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
                if rrow >= lo && rrow < hi {
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
                        *scores_v.add(i * ROUTE_ROWS + rrow) = acc * rr;
                    }
                }
                r += 1;
            }
            i += 8;
        }
    }

    /// One token's FFN input, the host's arithmetic exactly (whole workgroup): the post-norm's scale as
    /// `common::rmsnorm::sumsq_tree` (thread j sums x[j + 256i] in order, lanes of wave 0 add eight of those
    /// in order, lane 0 the 32 in order), each group quantized as `QAct::quantize` (a wave four groups, their
    /// loads first), and the router's picks and weights as `ffn::route_scores` (each expert's rank counted
    /// against every key: the greater key first, between equals the lower expert; the mass in id order). Writes the card's activation buffers and the host's copies, picks and weights (`act`'s table) at row `ot`
    /// (the launch's token `tok` past the chunk's first).
    #[inline(always)]
    unsafe fn act_token(tok: usize, ot: usize, t: usize, wave: usize, lane: u32, x: *const f32, scores_v: *const f32, act: *const u64) {
        use common::q4::GROUP;
        const _: () = assert!(H == 256 * 16 && E % 32 == 0 && E / 32 <= 32 && crate::ROUTED_K == TOP_K && (H / GROUP) % 8 == 0);
        let at = |k: usize| *act.add(k) as usize;
        let (pln, bias) = (at(0) as *const f32, at(1) as *const f32);
        let (xq, xs, xg) = (at(2) as *mut i8, at(3) as *mut f32, at(4) as *mut i32);
        let (xq_h, xs_h, xg_h) = (at(5) as *mut i8, at(6) as *mut f32, at(7) as *mut i32);
        let (picks, wts, wv) = (at(8) as *mut u32, at(9) as *mut f32, at(10) as *mut f32);
        let xt = x.add(tok * H);
        let l = lds::<f32>();
        let part = l; // [256] the threads' sums of squares
        let sig = l.add(256); // [E] the sigmoid scores
        let idx = sig.add(E) as *mut u32; // [TOP_K] the picks in id order
        let ranked = idx.add(TOP_K); // [TOP_K] the picks by rank
        let misc = ranked.add(TOP_K) as *mut f32; // [0] the norm's scale, [1] the mass
        let ck = misc.add(8) as *mut u64; // [E] the selection order's keys (16-byte aligned: 4·(256 + E + 2·TOP_K + 8))
        let lu = lane as usize;
        // thread t's squares, in order; the sigmoids and the bias a thread an expert
        let mut p = 0f32;
        let mut i = 0;
        while i < 16 {
            let v = *xt.add(t + 256 * i);
            p += v * v;
            i += 1;
        }
        *part.add(t) = p;
        let mut e = t;
        while e < E {
            let sv = sigmoid(*scores_v.add(tok * ROUTE_ROWS + e));
            *sig.add(e) = sv;
            // the selection key (`top`'s sum) as one u64 in the host's order: the key's bits mapped to an unsigned
            // order above, the complement of the expert below, so between equal keys the lower expert is greater
            let kb = (sv + *bias.add(e)).to_bits();
            let ko = if kb >> 31 != 0 { !kb } else { kb | 0x8000_0000 };
            *ck.add(e) = ((ko as u64) << 32) | (!(e as u32)) as u64;
            e += 256;
        }
        workgroup_sync();
        if wave == 0 {
            let mut q = 0f32;
            let mut k = 0;
            while k < 8 {
                q += *part.add(8 * lu + k);
                k += 1;
            }
            let mut ss = 0f32;
            let mut j = 0;
            while j < 32 {
                ss += lane_read(j, q);
                j += 1;
            }
            if lane == 0 {
                *misc = common::math::rsqrt(ss / H as f32 + RMS_NORM_EPS);
            }
        }
        workgroup_sync();
        let r = *misc;
        // the groups: wave w quantizes groups w, w + 8, w + 16, w + 24, the four groups' loads issued first and their
        // butterflies in lockstep
        const GW: usize = H / GROUP / 8;
        let mut v = [[0f32; 4]; GW];
        let mut gi = 0;
        while gi < GW {
            let k0 = (wave + 8 * gi) * GROUP + lu * 4;
            let xv: [f32; 4] = core::ptr::read(xt.add(k0) as *const [f32; 4]);
            let lv: [f32; 4] = core::ptr::read(pln.add(k0) as *const [f32; 4]);
            let mut jj = 0;
            while jj < 4 {
                v[gi][jj] = lv[jj] * (xv[jj] * r);
                jj += 1;
            }
            gi += 1;
        }
        let (packed, sc, sum) = common::q4::gfx11::quantize_groups(lane as u32, v);
        let mut gi = 0;
        while gi < GW {
            let g = wave + 8 * gi;
            let k0 = g * GROUP + lu * 4;
            *(xq.add(ot * H + k0) as *mut u32) = packed[gi];
            *(xq_h.add(ot * H + k0) as *mut u32) = packed[gi];
            if lane == 0 {
                let o = ot * (H / GROUP) + g;
                *xs.add(o) = sc[gi];
                *xg.add(o) = sum[gi];
                *xs_h.add(o) = sc[gi];
                *xg_h.add(o) = sum[gi];
            }
            gi += 1;
        }
        // the router's picks: the host's `top` takes the greatest key first and, between equal keys, the lower expert,
        // which is the u64 order of `ck` (the one order the mapping changes, −0 below +0, needs a sum both of whose
        // terms are −0, and the sigmoid is not); an expert's rank is the count of keys above its own, the TOP_K ranked
        // below TOP_K are the picks, the rank their order taken. A thread ranks expert t and, the 32 experts past 256,
        // expert t + 256 in the same pass: a 64-bit compare and a carry-add a key, eight keys' loads issued together
        const _: () = assert!(E % 8 == 0 && E > 256 && E <= 512 && (4 * (256 + E + 2 * TOP_K + 8)) % 16 == 0);
        let e2 = t + 256;
        let ke = *ck.add(t);
        let ke2 = if e2 < E { *ck.add(e2) } else { u64::MAX };
        let (mut rk, mut rk2) = (0u32, 0u32);
        let mut f = 0;
        while f < E {
            let kf: [u64; 8] = core::ptr::read(ck.add(f) as *const [u64; 8]);
            let mut j = 0;
            while j < 8 {
                rk += (kf[j] > ke) as u32;
                rk2 += (kf[j] > ke2) as u32;
                j += 1;
            }
            f += 8;
        }
        if (rk as usize) < TOP_K {
            *ranked.add(rk as usize) = t as u32;
        }
        if e2 < E && (rk2 as usize) < TOP_K {
            *ranked.add(rk2 as usize) = e2 as u32;
        }
        workgroup_sync();
        if wave == 0 {
            let mine = if lu < TOP_K { *ranked.add(lu) } else { u32::MAX };
            // lane k < TOP_K holds the k-th pick taken; its place in id order is the picks below it
            let mut pos = 0u32;
            let mut j = 0;
            while j < TOP_K as u32 {
                let o = lane_read(j, f32::from_bits(mine)).to_bits();
                if o < mine {
                    pos += 1;
                }
                j += 1;
            }
            if lu < TOP_K {
                *idx.add(pos as usize) = mine;
            }
            common::gfx11::wave_lds_sync();
            if lane == 0 {
                let mut mass = 1e-20f32;
                let mut k = 0;
                while k < TOP_K {
                    mass += *sig.add(*idx.add(k) as usize);
                    k += 1;
                }
                *misc.add(1) = mass;
            }
            common::gfx11::wave_lds_sync();
            if lu < TOP_K {
                let e = *idx.add(lu) as usize;
                *picks.add(ot * TOP_K + lu) = e as u32;
                let wt = *sig.add(e) / *misc.add(1) * ROUTED_SCALING_FACTOR;
                *wts.add(ot * TOP_K + lu) = wt;
                *wv.add(ot * TOP_K + lu) = wt;
            }
        }
        workgroup_sync();
    }

    /// `stream_ij[d] = post_ij · y_i[d] + Σ_k comb_ikj · stream_ik[d]` for the `tokens` tokens of a
    /// step (`mix: [tokens][MIX]`, `streams: [tokens][HC][H]`, `ya`, `yb: [tokens][H]`), a thread a
    /// channel looping the tokens, the block output `y` the sum of `ya` and `yb` (`yb` null for one
    /// source): the parts of a split attention block, or the shared expert's rows and the host's
    /// routed experts. The handoff first, so the update is one launch, not two: workgroup 0 copies
    /// this card's `n` floats from `src` out to `dst` and `dst2` (those not null: a peer card's
    /// receiving buffer in its VRAM, queue 001), writes `value` to `mine` (this card's own word) and
    /// `mine2` (the peer's copy of it), holds until `w0` and `w1` (those not null, in this card's
    /// VRAM: the producers' words as they landed here) read `value`, then releases the other
    /// workgroups through `go`, a VRAM word (one poller a card: sixteen slowed the host's experts by
    /// 0.5 ms a token, timeline 2026-10-16). `rows` (the leader of a MoE layer, queue 002): the routed
    /// experts' output rows `[tokens · TOP_K][H]` in its VRAM, weighted by `wts: [tokens][TOP_K]` and summed in
    /// pick order in place of `yb`, the sums written to `fwd` (the peer's receiving buffer) and, once every
    /// workgroup's are out, `value` to `fwd_word` (the peer's copy of the experts' word). Grid: CHANNEL_GRID.
    #[no_mangle]
    pub extern "gpu-kernel" fn hc_post(mix: *const f32, ya: *const f32, yb: *const f32, streams: *mut f32, src: *const f32, dst: *mut f32, dst2: *mut f32, n: u32, mine: *mut u32, mine2: *mut u32, w0: *const u32, w1: *const u32, value: u32, go: *mut u32, tokens: u32, rows: *const f32, wts: *const f32, fwd: *mut f32, fwd_word: *mut u32) {
        unsafe {
            let t = workitem_id_x();
            if workgroup_id_x() == 0 {
                if !src.is_null() {
                    let mut i = t;
                    while i < n {
                        let v = *src.add(i as usize);
                        if !dst.is_null() {
                            *dst.add(i as usize) = v;
                        }
                        if !dst2.is_null() {
                            *dst2.add(i as usize) = v;
                        }
                        i += 256;
                    }
                    fence(Ordering::Release);
                    workgroup_sync();
                }
                // one poller a card: this card's own word included, so no workgroup reads a buffer before the
                // copy is out; then the others released through a VRAM word
                if t == 0 {
                    if !mine.is_null() {
                        AtomicU32::from_ptr(mine).store(value, Ordering::Release);
                        if !mine2.is_null() {
                            AtomicU32::from_ptr(mine2).store(value, Ordering::Release);
                        }
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
                // the routed experts' part: the leader's weighted sum of the token's rows in pick order (the host's
                // arithmetic, a multiply and an add a pick from zero), forwarded to the peer; else the part given
                let ybv = if !rows.is_null() {
                    let mut s = 0f32;
                    let mut k = 0;
                    while k < TOP_K {
                        s += *wts.add(i * TOP_K + k) * *rows.add((i * TOP_K + k) * H + d);
                        k += 1;
                    }
                    if !fwd.is_null() {
                        *fwd.add(i * H + d) = s;
                    }
                    s
                } else if yb.is_null() {
                    0.0
                } else {
                    *yb.add(i * H + d)
                };
                let yd = *ya.add(i * H + d) + ybv;
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
            // the forwarded sums out: the last workgroup writes the peer's word (`go`'s next word counts arrivals)
            if !fwd_word.is_null() {
                fence(Ordering::Release);
                workgroup_sync();
                if workitem_id_x() == 0 {
                    let count = go.add(1);
                    if AtomicU32::from_ptr(count).fetch_add(1, Ordering::AcqRel) == CHANNEL_GRID - 1 {
                        AtomicU32::from_ptr(count).store(0, Ordering::Relaxed);
                        AtomicU32::from_ptr(fwd_word).store(value, Ordering::Release);
                    }
                }
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

