//! The expert predictor's analog of the next layer (098): a cut-down run of layer l + 1's KDA attention on the
//! cards during the host's expert run of layer l, whose router logits replace the route-ahead as the predictor's
//! first margin. Per card, the part of layer l + 1 it holds: layer l's FFN output stood in by the shared expert's
//! output alone (`fyv`, on the card before the host's run), through layer l's FFN mix into the streams; layer l + 1's
//! attention-half hyper-connection over them (`an_pre`); its q, k, v and gate rows of `NSEL` chosen heads of the part
//! (`an_rows_sel`, `an_fg_sel`: the layer's own exp4 rows, a head table naming the rows); `kda_step_ro` on those heads
//! from the state as it stands; then the router composed into the output side — `C = R·diag(ln)·O` over the chosen
//! heads' columns, `[E][NSEL · KDA_DIM]` bf16, the only new weights — against the heads' readouts, a partial logit
//! vector a card (`an_out`); the leader adds the router over the no-attention input (`an_pre`'s second half: layer
//! l + 1's FFN mix of the streams without the attention's output) and the two partials inside `pred_step`'s first
//! margin: `s = s0 + (c / rms(x0)) · (p_leader + p_peer)` — the composed form, measured offline within .02 of the exact
//! one (098, `-rn`). Nothing here writes what the forward reads.
//!
//! The handoffs between the cards ride the existing word kernels (`word_wait` / `word_post`, 058): the leader's
//! `an_pre` copies `fyv` into the peer's receiving buffer and posts the peer's word; the peer's chain waits for it
//! ahead of its `an_pre`; the peer's `an_out` writes its partial into the leader's receiving buffer and posts the
//! leader's word, which the leader waits for ahead of its `an_out`. No cycle: the leader's post precedes every wait
//! in its queue, the peer's post follows only its own wait, and both chains sit ahead of their `hc_post`.

use crate::*;

/// Heads of a part the analog runs (half of the part's `PART_HEADS`).
pub const NSEL: usize = 16;
/// `an_pre`'s grid: a coefficient row a workgroup.
pub const PRE_GRID: u32 = HC_COEFS as u32;
/// `an_pre`'s scratch a token, f32: the two halves' coefficients `[2][HC_COEFS]` then their mixes `[2][MIX]`.
pub const PRE_SCRATCH: usize = 2 * HC_COEFS + 2 * crate::hc::gfx11_consts::MIX;
/// `an_pre`'s grid barriers: an arrival count and a done word each.
pub const PRE_BARRIERS: usize = 6;
/// `an_pre`'s words: the barriers' counts and done words.
pub const PRE_COUNTS: usize = 2 * PRE_BARRIERS;
/// `an_pre`'s LDS, floats: the waves' partials `[8]` and a transpose scratch a wave `[8][HC·HC]`.
pub const PRE_LDS: u32 = (4 * (16 + 8 * HC * HC)) as u32;
/// The rows of `sq` the selection covers: the small rows, then q, k, v of the chosen heads.
pub const SEL_ROWS: usize = 2 * KDA_DIM + KDA_HEADS + 3 * NSEL * KDA_DIM;
/// The rows of `fg` the selection covers: f_b then g_b of the chosen heads.
pub const SEL_FG_ROWS: usize = 2 * NSEL * KDA_DIM;
/// `an_out`'s grid: a composed row a wave, eight a workgroup; the leader's launch takes the router's rows too.
pub const OUT_WGS: u32 = (E / 8) as u32;
pub const OUT_LDS: u32 = 64;
/// The width of a part's chosen heads' readouts, the composed rows' width.
pub const SEL_W: usize = NSEL * KDA_DIM;
const _: () = assert!(E % 8 == 0 && SEL_W % 256 == 0 && H % 256 == 0 && HC == 4 && (HC * H) % 256 == 0);

#[cfg(target_arch = "amdgpu")]
pub mod gfx11 {
    pub use super::*;
    use crate::hc::gfx11_consts::{COEF_ROWS, MIX};
    use crate::kda::gfx11_consts::{E4_ROWS, E4_WAVES, PART_QKV};
    use common::gemv::gfx11::{exp4_rows_direct_map, widen8_at, Exp4};
    use common::gemv::widen;
    use common::gfx11::{await_word, group_max, group_sum, lane_read, lds, wave_sum, workgroup_sync};
    use common::math::{exp, rsqrt, sqrt};
    use core::arch::amdgpu::{workgroup_id_x, workitem_id_x};
    use core::sync::atomic::{fence, AtomicU32, Ordering};
    const SMALL_ROWS: usize = 2 * KDA_DIM + KDA_HEADS;
    const M_SMALL: usize = SMALL_ROWS + 3 * PART_QKV;
    const D: usize = KDA_DIM;

    #[inline(always)]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + exp(-x))
    }

    /// A grid barrier over `PRE_GRID` workgroups on its (count, done) pair: thread 0 counts in, the last resets the
    /// count and publishes `value` in `done`; every workgroup waits for it. Uniform: every thread of every workgroup.
    #[inline(always)]
    unsafe fn barrier(count: *mut u32, b: usize, t: usize, value: u32) {
        fence(Ordering::Release);
        workgroup_sync();
        let (cnt, done) = (count.add(2 * b), count.add(2 * b + 1));
        if t == 0 {
            if AtomicU32::from_ptr(cnt).fetch_add(1, Ordering::AcqRel) == PRE_GRID - 1 {
                AtomicU32::from_ptr(cnt).store(0, Ordering::Relaxed);
                AtomicU32::from_ptr(done).store(value, Ordering::Release);
            }
            await_word(done, value);
        }
        workgroup_sync();
        fence(Ordering::Acquire);
    }

    /// One token's hyper-connection mix from its coefficients (`hc_pre`'s arithmetic on a wave: the sigmoids, the
    /// softmax rows and Sinkhorn), into `mix[MIX]` (pre | post | comb). The whole wave calls; `tr` is the wave's
    /// `[HC·HC]` transpose scratch.
    #[inline(always)]
    unsafe fn mix_token(c: impl Fn(usize) -> f32, base: *const f32, scale: *const f32, lane: u32, tr: *mut f32, mix: *mut f32) {
        let (s0, s1, s2) = (*scale, *scale.add(1), *scale.add(2));
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
        if inb {
            *mix.add(2 * HC + lu) = cbv;
        }
        if lu < HC {
            *mix.add(lu) = pre_v;
            *mix.add(HC + lu) = post_v;
        }
    }

    /// A coefficient row's dot with a token's flattened streams normed without a weight (`hc::pre`): this workgroup's
    /// 256 threads over the `HC · H` channels, `v(k)` the channel's value; the dot and the sum of squares.
    #[inline(always)]
    unsafe fn coef_row(w_row: *const u16, v: impl Fn(usize) -> f32, t: usize, lane: u32, wave: usize, red: *mut f32) -> (f32, f32) {
        let (mut acc, mut ss) = (0f32, 0f32);
        let mut k = t;
        while k < HC * H {
            let x = v(k);
            acc += x * widen(*w_row.add(k));
            ss += x * x;
            k += 256;
        }
        acc = wave_sum(lane, acc);
        ss = wave_sum(lane, ss);
        if lane == 0 {
            *red.add(wave) = acc;
            *red.add(8 + wave) = ss;
        }
        workgroup_sync();
        let (mut a, mut s) = (0f32, 0f32);
        let mut k = 0;
        while k < 8 {
            a += *red.add(k);
            s += *red.add(8 + k);
            k += 1;
        }
        workgroup_sync();
        (a, s)
    }

    /// The analog's input half (the module): layer l's FFN mix `mix_l` (`[n][MIX]`) applied to its FFN-input
    /// `streams` with `fyv` as the output into `an_streams` (`[n][HC][H]`, this card's), layer l + 1's attention-half
    /// coefficients (`hc_a`: fn rows, base, scale) over them, the mix and the collapse into `x` (`[n][H]`, unnormed);
    /// then the FFN half (`hc_f`) over the attention mix of the same streams with no attention output (`an_streams2`),
    /// its collapse into `x0` and `c[i] = Σ_j pre_f[j] · post_a[j]`. `fyv_dst` not null (the leader): `fyv` copied there (the peer's
    /// receiving buffer) first and `value` posted to `fyv_word`. `scratch`: `[n][PRE_SCRATCH]`; `count`: `PRE_COUNTS`
    /// words, 0 between launches. Grid PRE_GRID, LDS PRE_LDS.
    #[no_mangle]
    pub extern "gpu-kernel" fn an_pre(streams: *const f32, mix_l: *const f32, fyv: *const f32, fyv_dst: *mut f32, fyv_word: *mut u32, a_fn: *const u16, a_base: *const f32, a_scale: *const f32, f_fn: *const u16, f_base: *const f32, f_scale: *const f32, an_streams: *mut f32, an_streams2: *mut f32, x: *mut f32, x0: *mut f32, c: *mut f32, scratch: *mut f32, count: *mut u32, value: u32, n: u32) {
        unsafe {
            let n = n as usize;
            let t = workitem_id_x() as usize;
            let (wave, lane) = (t / 32, (t % 32) as u32);
            let wg = workgroup_id_x() as usize;
            let l = lds::<f32>();
            let (red, trs) = (l, l.add(16));
            let coefs = |i: usize, half: usize| scratch.add(i * PRE_SCRATCH + half * COEF_ROWS);
            let mixes = |i: usize, half: usize| scratch.add(i * PRE_SCRATCH + 2 * COEF_ROWS + half * MIX);
            // phase 0: the leader's copy of fyv to the peer and its word; every workgroup under H / 256 makes its
            // channel's stood-in streams: s'_j[d] = post_j · fyv[d] + Σ_k comb_kj · stream_k[d] (hc_post's arithmetic)
            if wg == 0 && !fyv_dst.is_null() {
                let mut i = t;
                while i < n * H {
                    *fyv_dst.add(i) = *fyv.add(i);
                    i += 256;
                }
                fence(Ordering::Release);
                workgroup_sync();
                if t == 0 {
                    AtomicU32::from_ptr(fyv_word).store(value, Ordering::Release);
                }
            }
            let d = wg * 256 + t;
            if d < H {
                let mut i = 0;
                while i < n {
                    let (mt, st, out) = (mix_l.add(i * MIX), streams.add(i * HC * H), an_streams.add(i * HC * H));
                    let mut res = [0f32; HC];
                    let mut k = 0;
                    while k < HC {
                        res[k] = *st.add(k * H + d);
                        k += 1;
                    }
                    let yd = *fyv.add(i * H + d);
                    let mut j = 0;
                    while j < HC {
                        let mut acc = 0f32;
                        let mut k = 0;
                        while k < HC {
                            acc += *mt.add(2 * HC + k * HC + j) * res[k];
                            k += 1;
                        }
                        *out.add(j * H + d) = *mt.add(HC + j) * yd + acc;
                        j += 1;
                    }
                    i += 1;
                }
            }
            barrier(count, 0, t, value);
            // phase 1: the attention half's coefficient row `wg` of every token
            let mut i = 0;
            while i < n {
                let st = an_streams.add(i * HC * H);
                let (a, s) = coef_row(a_fn.add(wg * HC * H), |k| *st.add(k), t, lane, wave, red);
                if t == 0 {
                    *coefs(i, 0).add(wg) = a * rsqrt(s / (HC * H) as f32 + RMS_NORM_EPS);
                }
                i += 1;
            }
            barrier(count, 1, t, value);
            // phase 2: workgroup 0's waves make the tokens' attention mixes
            if wg == 0 && wave < n {
                let cf = coefs(wave, 0);
                mix_token(|k| *cf.add(k), a_base, a_scale, lane, trs.add(wave * HC * HC), mixes(wave, 0));
            }
            barrier(count, 2, t, value);
            // phase 3: the collapse `x` (the attention's input, normed by its reader), and the streams as the attention
            // mix leaves them with no output, s''_j[d] = Σ_k comb_kj · s'_k[d], into `an_streams2`
            if d < H {
                let mut i = 0;
                while i < n {
                    let (mt, st, out) = (mixes(i, 0), an_streams.add(i * HC * H), an_streams2.add(i * HC * H));
                    let mut res = [0f32; HC];
                    let mut k = 0;
                    while k < HC {
                        res[k] = *st.add(k * H + d);
                        k += 1;
                    }
                    let mut acc = 0f32;
                    let mut j = 0;
                    while j < HC {
                        acc += *mt.add(j) * res[j];
                        let mut sj = 0f32;
                        let mut k = 0;
                        while k < HC {
                            sj += *mt.add(2 * HC + k * HC + j) * res[k];
                            k += 1;
                        }
                        *out.add(j * H + d) = sj;
                        j += 1;
                    }
                    *x.add(i * H + d) = acc;
                    i += 1;
                }
            }
            barrier(count, 3, t, value);
            // phase 3b: the FFN half's coefficient row over them
            let mut i = 0;
            while i < n {
                let st = an_streams2.add(i * HC * H);
                let (a, s) = coef_row(f_fn.add(wg * HC * H), |k| *st.add(k), t, lane, wave, red);
                if t == 0 {
                    *coefs(i, 1).add(wg) = a * rsqrt(s / (HC * H) as f32 + RMS_NORM_EPS);
                }
                i += 1;
            }
            barrier(count, 4, t, value);
            // phase 4: workgroup 0's waves make the FFN mixes and c
            if wg == 0 && wave < n {
                let cf = coefs(wave, 1);
                mix_token(|k| *cf.add(k), f_base, f_scale, lane, trs.add(wave * HC * HC), mixes(wave, 1));
                common::gfx11::wave_lds_sync();
                if lane == 0 {
                    let (ma, mf) = (mixes(wave, 0), mixes(wave, 1));
                    let mut acc = 0f32;
                    let mut j = 0;
                    while j < HC {
                        acc += *mf.add(j) * *ma.add(HC + j);
                        j += 1;
                    }
                    *c.add(wave) = acc;
                }
            }
            barrier(count, 5, t, value);
            // phase 5: the no-attention collapse `x0[i][d] = Σ_j pre_f[j] · s''_j[d]`
            if d < H {
                let mut i = 0;
                while i < n {
                    let (mf, st) = (mixes(i, 1), an_streams2.add(i * HC * H));
                    let mut acc = 0f32;
                    let mut j = 0;
                    while j < HC {
                        acc += *mf.add(j) * *st.add(j * H + d);
                        j += 1;
                    }
                    *x0.add(i * H + d) = acc;
                    i += 1;
                }
            }
        }
    }

    /// The small rows and the chosen heads' q, k, v rows of a part's `sq` (exp4 planes at `w`, `M_SMALL` rows) over
    /// `x` normed with `ln`, into `out` (`[n][M_SMALL]`, the rows' own places: `kda_step_ro` reads it as `kda_step`
    /// reads `sq`'s output). `heads`: `NSEL` u32, the part's head indices. Grid `SEL_ROWS.div_ceil(E4_WAVES · E4_ROWS)`
    /// of `32 · E4_WAVES`, no LDS.
    #[no_mangle]
    pub extern "gpu-kernel" fn an_rows_sel(x: *const f32, ln: *const u16, w: *const u8, out: *mut f32, heads: *const u32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let n = n as usize;
            let map = |r: usize| {
                if r < SMALL_ROWS {
                    r
                } else {
                    let j = r - SMALL_ROWS;
                    let (part, rem) = (j / (NSEL * D), j % (NSEL * D));
                    SMALL_ROWS + part * PART_QKV + *heads.add(rem / D) as usize * D + rem % D
                }
            };
            let emit = |row: usize, i: usize, y: f32| *out.add(i * M_SMALL + row) = y;
            let xs = |_: usize, i: usize| x.add(i * H);
            let e4 = Exp4::<2, 8>::at::<H>(w, M_SMALL);
            match n {
                1 => exp4_rows_direct_map::<H, true, 2, 8, E4_ROWS, E4_WAVES, 1>(tid, e4, ln, SEL_ROWS, n, RMS_NORM_EPS, map, xs, emit),
                2 => exp4_rows_direct_map::<H, true, 2, 8, E4_ROWS, E4_WAVES, 2>(tid, e4, ln, SEL_ROWS, n, RMS_NORM_EPS, map, xs, emit),
                3 => exp4_rows_direct_map::<H, true, 2, 8, E4_ROWS, E4_WAVES, 3>(tid, e4, ln, SEL_ROWS, n, RMS_NORM_EPS, map, xs, emit),
                _ => exp4_rows_direct_map::<H, true, 2, 8, E4_ROWS, E4_WAVES, { common::gemv::FEW }>(tid, e4, ln, SEL_ROWS, n, RMS_NORM_EPS, map, xs, emit),
            }
        }
    }

    /// The chosen heads' f_b and g_b rows of a part's `fg` (exp4 planes, `2 · PART_QKV` rows of `KDA_DIM`) over the
    /// gates' first halves in `smix` (`[n][M_SMALL]`: f_a's output at 0, g_a's at `KDA_DIM`), into `out`
    /// (`[n][2 · PART_QKV]`, the rows' own places). Grid `SEL_FG_ROWS.div_ceil(E4_WAVES · E4_ROWS)`, no LDS.
    #[no_mangle]
    pub extern "gpu-kernel" fn an_fg_sel(smix: *const f32, w: *const u8, out: *mut f32, heads: *const u32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let n = n as usize;
            let map = |j: usize| {
                let (part, rem) = (j / (NSEL * D), j % (NSEL * D));
                part * PART_QKV + *heads.add(rem / D) as usize * D + rem % D
            };
            let emit = |row: usize, i: usize, y: f32| *out.add(i * 2 * PART_QKV + row) = y;
            let xs = |row: usize, i: usize| if row >= PART_QKV { smix.add(D + i * M_SMALL) } else { smix.add(i * M_SMALL) };
            let e4 = Exp4::<2, 8>::at::<KDA_DIM>(w, 2 * PART_QKV);
            match n {
                1 => exp4_rows_direct_map::<KDA_DIM, false, 2, 8, E4_ROWS, E4_WAVES, 1>(tid, e4, core::ptr::null(), SEL_FG_ROWS, n, 0.0, map, xs, emit),
                2 => exp4_rows_direct_map::<KDA_DIM, false, 2, 8, E4_ROWS, E4_WAVES, 2>(tid, e4, core::ptr::null(), SEL_FG_ROWS, n, 0.0, map, xs, emit),
                3 => exp4_rows_direct_map::<KDA_DIM, false, 2, 8, E4_ROWS, E4_WAVES, 3>(tid, e4, core::ptr::null(), SEL_FG_ROWS, n, 0.0, map, xs, emit),
                _ => exp4_rows_direct_map::<KDA_DIM, false, 2, 8, E4_ROWS, E4_WAVES, { common::gemv::FEW }>(tid, e4, core::ptr::null(), SEL_FG_ROWS, n, 0.0, map, xs, emit),
            }
        }
    }

    /// The output side (the module): workgroups under `OUT_WGS` take the composed rows `crows` (`[E][SEL_W]` bf16), a
    /// row a wave, over the chosen heads' readouts gathered from `normed` (`[n][PART_QKV]`) into `part` (`[n][E]`:
    /// this card's own buffer, or the leader's receiving buffer); with `part_word` (the peer), the last workgroup out
    /// posts `value` there (`count` its arrivals). Workgroups from `OUT_WGS` (the leader's launch: grid `2 · OUT_WGS`)
    /// take layer l + 1's router rows `E..2E` (`router`, `[2E][H]` bf16) over `x0` normed with `ln` into
    /// `scores[i · ROUTE_ROWS + row]`, and the first of them writes `k[i] = c[i] / rms(x0[i])`. LDS OUT_LDS.
    #[no_mangle]
    pub extern "gpu-kernel" fn an_out(normed: *const f32, heads: *const u32, crows: *const u16, part: *mut f32, part_word: *mut u32, count: *mut u32, x0: *const f32, ln: *const u16, router: *const u16, scores: *mut f32, c: *const f32, k: *mut f32, value: u32, n: u32) {
        unsafe {
            const RR: usize = crate::hc::gfx11_consts::ROUTE_ROWS;
            let n = n as usize;
            let t = workitem_id_x() as usize;
            let (wave, lane) = (t / 32, t % 32);
            let wg = workgroup_id_x() as usize;
            if wg < OUT_WGS as usize {
                // a composed row a wave: the lane's eight columns of each 256-wide step gathered from the head the
                // step's columns belong to (a head is 128 columns: a step is two heads, a lane's eight within one)
                let row = wg * 8 + wave;
                let wr = crows.add(row * SEL_W);
                let mut acc = [0f32; 4];
                let mut s = 0;
                while s < SEL_W / 256 {
                    let col = s * 256 + lane * 8;
                    let wv = widen8_at(wr.add(col));
                    let src = *heads.add(col / D) as usize * D + col % D;
                    let mut i = 0;
                    while i < n {
                        let xv: [f32; 8] = core::ptr::read(normed.add(i * PART_QKV + src) as *const [f32; 8]);
                        let mut j = 0;
                        while j < 8 {
                            acc[i] += wv[j] * xv[j];
                            j += 1;
                        }
                        i += 1;
                    }
                    s += 1;
                }
                let mut i = 0;
                while i < n {
                    let y = wave_sum(lane as u32, acc[i]);
                    if lane == 0 {
                        *part.add(i * E + row) = y;
                    }
                    i += 1;
                }
                if !part_word.is_null() {
                    // the partials out over the link; the last workgroup posts the leader's word
                    fence(Ordering::Release);
                    workgroup_sync();
                    if t == 0 && AtomicU32::from_ptr(count).fetch_add(1, Ordering::AcqRel) == OUT_WGS - 1 {
                        AtomicU32::from_ptr(count).store(0, Ordering::Relaxed);
                        AtomicU32::from_ptr(part_word).store(value, Ordering::Release);
                    }
                }
            } else {
                // the leader: a router row a wave over the no-attention input, normed; k from the norm
                let row = E + (wg - OUT_WGS as usize) * 8 + wave;
                let wr = router.add(row * H);
                let mut acc = [0f32; 4];
                let mut ss = [0f32; 4];
                let mut s = 0;
                while s < H / 256 {
                    let col = s * 256 + lane * 8;
                    let wv = widen8_at(wr.add(col));
                    let lv = widen8_at(ln.add(col));
                    let mut i = 0;
                    while i < n {
                        let xv: [f32; 8] = core::ptr::read(x0.add(i * H + col) as *const [f32; 8]);
                        let mut j = 0;
                        while j < 8 {
                            acc[i] += xv[j] * lv[j] * wv[j];
                            ss[i] += xv[j] * xv[j];
                            j += 1;
                        }
                        i += 1;
                    }
                    s += 1;
                }
                let mut i = 0;
                while i < n {
                    let y = wave_sum(lane as u32, acc[i]);
                    let rr = 1.0 / sqrt(wave_sum(lane as u32, ss[i]) / H as f32 + RMS_NORM_EPS);
                    if lane == 0 {
                        *scores.add(i * RR + row) = y * rr;
                        if row == E {
                            *k.add(i) = *c.add(i) * rr;
                        }
                    }
                    i += 1;
                }
            }
        }
    }
}
