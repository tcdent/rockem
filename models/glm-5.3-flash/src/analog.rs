//! The expert predictor's analog of the next layer (098): a cut-down run of layer l + 1's KDA attention on the
//! cards during the host's expert run of layer l, whose router logits replace the route-ahead as the predictor's
//! first margin. Per card, the part of layer l + 1 it holds: layer l's FFN output stood in by the shared expert's
//! output alone (`fyv`, on the card before the host's run), through layer l's FFN mix into the streams; layer l + 1's
//! attention-half hyper-connection over them (`an_pre`: without materializing them — every coefficient of both halves
//! is a combination of a row's dots with five base vectors, the streams and `fyv`); its q, k, v and gate rows of `NSEL` chosen heads of the part
//! (`an_rows_sel`, `an_fg_sel`: the layer's own exp4 rows, a head table naming the rows); `kda_step_ro` on those heads
//! from the state as it stands; then the router composed into the output side — `C = R·diag(ln)·O` over the chosen
//! heads' columns, `[E][NSEL · KDA_DIM]` bf16, the only new weights — against the heads' readouts, a partial logit
//! vector a card (`an_out`); the leader adds the router over the no-attention input (`an_pre`'s second half: layer
//! l + 1's FFN mix of the streams without the attention's output) and the two partials inside `pred_step`'s first
//! margin: `s = s0 + (c / rms(x0)) · (p_leader + p_peer)` — the composed form, measured offline within .02 of the exact
//! one (098, `-rn`). Nothing here writes what the forward reads.
//!
//! The handoffs between the cards are two words, waited for in-kernel (as `hc_post` waits): the leader's `an_send`
//! (behind its shared expert, before the run) copies `fyv` into the peer's receiving buffer and posts the peer's word;
//! the peer's `an_pre` waits for it; the peer's `an_out` writes its partial into the leader's receiving buffer and
//! posts the leader's word, which the leader's `pred_step` waits for. No cycle: the leader's send precedes every wait
//! in its queue, the peer posts only after its own wait, and both chains sit ahead of their `hc_post`.

use crate::*;

/// Heads of a part the analog runs: 8 of its 32 (16 of the layer's 64). Measured offline (the first-margin shape, used
/// experts a layer over today's posts, cody / survey): 32 heads +0.44 / +0.31, 24 +0.35 / +0.22, **16 +0.27 / +0.16**,
/// 8 +0.14 / +0.07; the rows read scales with them (~42 MB a card at 32, ~22 at 16), and the host's run the chain
/// must fit is ~170–200 µs at width 2 (2026-10-27): 16 heads is the size that fits.
pub const NSEL: usize = 8;
/// `an_chain`'s grid: two workgroups a CU of the 7900 XT's 84, all resident (its grid barriers); its waves loop over
/// the rows. `bringup/kernel_report.py` holds its registers to what keeps the grid resident.
pub const CHAIN_GRID: u32 = 168;
pub const CHAIN_WAVES: usize = 8;
/// `an_chain`'s grid barriers: each `BARRIER_WORDS` words — the top count, a count a group of `BARRIER_GROUP`
/// workgroups, the done word (`CHAIN_COUNTS` words in all, 0 between launches). Arrivals count on a group's word
/// first, the groups' last on the top word: 168 arrivals serialized on one word cost ~8 µs a barrier (the stats boot
/// of 2026-10-27: the hyper-connection phases 42 µs against 18 predicted).
pub const CHAIN_BARRIERS: usize = 6;
pub const BARRIER_GROUP: usize = 8;
pub const BARRIER_GROUPS: usize = CHAIN_GRID as usize / BARRIER_GROUP;
pub const BARRIER_WORDS: usize = 32;
pub const CHAIN_COUNTS: usize = CHAIN_BARRIERS * BARRIER_WORDS;
const _: () = assert!(CHAIN_GRID as usize % BARRIER_GROUP == 0 && 2 + BARRIER_GROUPS <= BARRIER_WORDS);
/// `an_chain`'s scratch a token, f32 (`gfx11::SCRATCH` lays it out).
pub const PRE_SCRATCH: usize = 2048;
/// `an_chain`'s LDS: phase A's wave partials `[8][4 · DOTS + BASE²]`, or the step's (`kda::STEP_LDS`), the larger.
/// Phase B's LDS a wave, floats: the coefficients, the transpose scratch, the Gram, alpha and alpha''.
pub const B_LDS: usize = 128;
pub const CHAIN_LDS: u32 = {
    let a = 4 * 8 * (4 * 20 + 25);
    let b = crate::kda::gfx11_consts::STEP_LDS;
    let c = 4 * 4 * B_LDS;
    let m = if a > b { a } else { b };
    (if m > c { m } else { c }) as u32 + 4 * NSEL as u32
};
/// The rows of `sq` the selection covers: the small rows, then q, k, v of the chosen heads.
pub const SEL_ROWS: usize = 2 * KDA_DIM + KDA_HEADS + 3 * NSEL * KDA_DIM;
/// The rows of `fg` the selection covers: f_b then g_b of the chosen heads.
pub const SEL_FG_ROWS: usize = 2 * NSEL * KDA_DIM;
/// Phase R's rows a wave: enough that the grid's waves take the selection in one round.
pub const R_ROWS: usize = SEL_ROWS.div_ceil(CHAIN_GRID as usize * CHAIN_WAVES);
const _: () = assert!(R_ROWS * CHAIN_GRID as usize * CHAIN_WAVES >= SEL_ROWS && SEL_FG_ROWS <= crate::kda::gfx11_consts::E4_ROWS * CHAIN_GRID as usize * CHAIN_WAVES);
/// The output phase: a composed row a wave, eight a workgroup; the leader's workgroups past them take the router's
/// rows over the no-attention input, then over the second margin's input.
pub const OUT_WGS: usize = E / 8;
/// The width of a part's chosen heads' readouts, the composed rows' width.
pub const SEL_W: usize = NSEL * KDA_DIM;
/// `an_chain`'s table of device addresses (one a card a pair, built at boot): `TAB_*` index it.
pub const TAB_STREAMS: usize = 0;
pub const TAB_MIX_L: usize = 1;
pub const TAB_FYV: usize = 2;
pub const TAB_FYV_WORD: usize = 3;
pub const TAB_A_FN: usize = 4;
pub const TAB_A_BASE: usize = 5;
pub const TAB_A_SCALE: usize = 6;
pub const TAB_F_FN: usize = 7;
pub const TAB_F_BASE: usize = 8;
pub const TAB_F_SCALE: usize = 9;
pub const TAB_X: usize = 10;
pub const TAB_X0: usize = 11;
pub const TAB_C: usize = 12;
pub const TAB_SCRATCH: usize = 13;
pub const TAB_COUNT: usize = 14;
pub const TAB_INPUT_LN: usize = 15;
pub const TAB_SQ: usize = 16;
pub const TAB_SMIX: usize = 17;
pub const TAB_HEADS: usize = 18;
pub const TAB_FG_W: usize = 19;
pub const TAB_FG: usize = 20;
pub const TAB_TAPS: usize = 21;
pub const TAB_WINDOW: usize = 22;
pub const TAB_DT_BIAS: usize = 23;
pub const TAB_A_LOG: usize = 24;
pub const TAB_MEM: usize = 25;
pub const TAB_O_NORM: usize = 26;
pub const TAB_NORMED: usize = 27;
pub const TAB_CROWS: usize = 28;
pub const TAB_PART: usize = 29;
pub const TAB_PART_WORD: usize = 30;
pub const TAB_OCOUNT: usize = 31;
pub const TAB_LN1: usize = 32;
pub const TAB_ROUTER: usize = 33;
pub const TAB_SCORES: usize = 34;
pub const TAB_K: usize = 35;
pub const TAB_S2: usize = 36;
pub const TAB_XL: usize = 37;
pub const TAB_LN_L: usize = 38;
pub const TAB_LEN: usize = 39;
const _: () = assert!(E % 8 == 0 && SEL_W % 256 == 0 && H % 256 == 0 && HC == 4 && (HC * H) % 256 == 0 && CHAIN_GRID as usize >= OUT_WGS * 3 && CHAIN_GRID as usize >= 2 * HC_COEFS);

#[cfg(target_arch = "amdgpu")]
pub mod gfx11 {
    pub use super::*;
    use crate::hc::gfx11_consts::{COEF_ROWS, MIX};
    use crate::kda::gfx11_consts::{E4_ROWS, PART_QKV};
    use common::gemv::gfx11::{exp4_rows_direct_rows, widen8_at, Exp4};
    use common::gemv::widen;
    use common::gfx11::{await_word, lds, rcp, wave_sum, wave_sum_f32, workgroup_sync};
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

    /// A grid barrier over `CHAIN_GRID` workgroups on its words: thread 0 counts in on its group's word; the group's
    /// last counts on the top word; the last group's publishes `value` in the done word, which every workgroup waits
    /// for. Counts back at 0 when the barrier leaves. Uniform: every thread of every workgroup.
    #[inline(always)]
    unsafe fn barrier(count: *mut u32, b: usize, t: usize, wg: usize, value: u32) {
        fence(Ordering::Release);
        workgroup_sync();
        let base = count.add(b * BARRIER_WORDS);
        let (top, group, done) = (base, base.add(1 + wg / BARRIER_GROUP), base.add(1 + BARRIER_GROUPS));
        if t == 0 {
            if AtomicU32::from_ptr(group).fetch_add(1, Ordering::AcqRel) == BARRIER_GROUP as u32 - 1 {
                AtomicU32::from_ptr(group).store(0, Ordering::Relaxed);
                if AtomicU32::from_ptr(top).fetch_add(1, Ordering::AcqRel) == BARRIER_GROUPS as u32 - 1 {
                    AtomicU32::from_ptr(top).store(0, Ordering::Relaxed);
                    AtomicU32::from_ptr(done).store(value, Ordering::Release);
                }
            }
            await_word(done, value);
        }
        workgroup_sync();
        fence(Ordering::Acquire);
    }

    use common::gfx11::{dpp_xmask, fma, other_row, wave_sum_dpp, wave_sum_dpp_n, XMASK1, XMASK2, XMASK4, XMASK8};
    /// The sum over a lane's quad (lanes `4q..4q + 4`): the Sinkhorn matrix's row, an entry a lane.
    #[inline(always)]
    unsafe fn quad_sum(v: f32) -> f32 {
        let v = v + dpp_xmask::<XMASK1>(v);
        v + dpp_xmask::<XMASK2>(v)
    }
    #[inline(always)]
    unsafe fn quad_max(v: f32) -> f32 {
        let o = dpp_xmask::<XMASK1>(v);
        let v = if o > v { o } else { v };
        let o = dpp_xmask::<XMASK2>(v);
        if o > v { o } else { v }
    }
    /// The sum over lanes `{l, l ^ 4, l ^ 8, l ^ 12}`: the matrix's column.
    #[inline(always)]
    unsafe fn col_sum(v: f32) -> f32 {
        let v = v + dpp_xmask::<XMASK4>(v);
        v + dpp_xmask::<XMASK8>(v)
    }

    /// A mix's inputs for this lane, loaded with phase B's first batch (inside `mix_token` the compiler sank them into
    /// the branch arms that used them: a round trip each, four of B's seven drains in build 12): the lane's base
    /// (coefficient `lane` for the pre/post lanes), its comb base, the three scales.
    #[derive(Clone, Copy)]
    struct MixIn {
        b: f32,
        b8: f32,
        s0: f32,
        s1: f32,
        s2: f32,
    }
    #[inline(always)]
    unsafe fn mix_in(base: *const f32, scale: *const f32, lu: usize) -> MixIn {
        MixIn {
            b: *base.add(if lu < 2 * HC { lu } else { 0 }),
            b8: *base.add(2 * HC + if lu < HC * HC { lu } else { 0 }),
            s0: *scale,
            s1: *scale.add(1),
            s2: *scale.add(2),
        }
    }
    #[inline(always)]
    unsafe fn sigmoid_rcp(x: f32) -> f32 {
        rcp(1.0 + exp(-x))
    }

    /// One token's hyper-connection mix from its coefficients (`hc::pre`'s arithmetic: the sigmoids, the softmax rows
    /// and Sinkhorn), an entry a lane: lanes 0–15 the comb matrix (entry `(lane / 4, lane % 4)`, its coefficient
    /// computed on this very lane — `phase_b`'s lane 0–15 owns row `8 + lane`), lanes 0–3 pre and 4–7 post (their
    /// coefficients rows 0–7, owned by lanes 16–23: one `v_permlanex16` brings them over). Row and column sums as DPP
    /// butterflies within the wave's first row — no LDS, no permute, and the critical wave issues ~12 instructions a
    /// Sinkhorn iteration instead of the ~60 of the form every lane computed alike (31 µs on one wave at ~9 cycles an
    /// instruction, 2026-10-27). Both of pre/post computed and selected (a branch arm is where the compiler sinks
    /// loads). Returns this lane's entries: (pre or post, comb); lanes under `MIX` store a copy.
    #[inline(always)]
    unsafe fn mix_token(coef: f32, m: MixIn, lu: usize, mix: *mut f32) -> (f32, f32) {
        let c_pp = other_row(coef);
        let pre = sigmoid_rcp(fma(c_pp, m.s0, m.b)) + HC_EPS;
        let post = 2.0 * sigmoid_rcp(fma(c_pp, m.s1, m.b));
        let pp = if lu < HC { pre } else { post };
        // the softmax of the row, then Sinkhorn: the columns first, then rows and columns `HC_SINKHORN_ITERS - 1`
        // times (`hc::pre`'s order)
        let e = fma(coef, m.s2, m.b8);
        let e = exp(e - quad_max(e));
        let mut cb = e * rcp(quad_sum(e)) + HC_EPS;
        let mut it = 0;
        while it < HC_SINKHORN_ITERS {
            if it > 0 {
                cb *= rcp(quad_sum(cb) + HC_EPS);
            }
            cb *= rcp(col_sum(cb) + HC_EPS);
            it += 1;
        }
        if lu < 2 * HC {
            *mix.add(lu) = pp;
        }
        if lu < HC * HC {
            *mix.add(2 * HC + lu) = cb;
        }
        (pp, cb)
    }

    /// The stood-in and no-attention streams are linear in five base vectors a token — layer l's four FFN-input streams
    /// and `fyv` — so both hyper-connection halves' coefficients are combinations of base dots (a row's four blocks
    /// against the five vectors) and their norms come from the vectors' Gram; the collapses are five-term sums. One
    /// streaming pass makes the dots and the Gram (phase A), one wave a token makes the two mixes from them (phase B),
    /// every workgroup then collapses (phase C): two grid barriers (the first form materialized the streams twice and
    /// crossed six, 92 µs on the box, 2026-10-26).
    pub const BASE: usize = HC + 1;
    /// A row's dots a token: `[HC blocks][BASE]`.
    pub const DOTS: usize = HC * BASE;
    /// Phase A's rows: the attention half's coefficient rows, then the FFN half's; each over `A_CHUNKS` workgroups,
    /// a chunk of `H / A_CHUNKS` channels each (one row a workgroup ran ~20 µs, a chain of round trips over its two
    /// chunks, with 120 of the 168 workgroups idle; the measurement of 2026-10-27).
    pub const A_ROWS: usize = 2 * HC_COEFS;
    pub const A_CHUNKS: usize = 2;
    pub const A_GRID: usize = A_ROWS * A_CHUNKS;
    /// `an_chain`'s scratch a token, floats: the rows' dots a chunk `[A_CHUNKS][A_ROWS][DOTS]`, the Gram a chunk
    /// `[A_CHUNKS][BASE][BASE]`, the two mixes `[2][MIX]`, the collapses' weights `[2][BASE]`.
    pub const SCR_DOTS: usize = 0;
    pub const SCR_GRAM: usize = SCR_DOTS + A_CHUNKS * A_ROWS * DOTS;
    pub const SCR_MIX: usize = SCR_GRAM + A_CHUNKS * BASE * BASE;
    pub const SCR_BETA: usize = SCR_MIX + 2 * MIX;
    pub const SCRATCH: usize = SCR_BETA + 2 * BASE;
    const _: () = assert!(SCRATCH <= PRE_SCRATCH && A_GRID as u32 + 4 * A_CHUNKS as u32 <= CHAIN_GRID && HC_COEFS <= 32 && BASE * BASE <= 32 && 8 * (4 * DOTS + BASE * BASE) * 4 <= CHAIN_LDS as usize && H % (A_CHUNKS * 2048) == 0);

    /// Phase A for `NP` tokens from `i0`: this workgroup's row's dots with the base vectors over its four blocks, the
    /// partials held in registers, one reduction round. A thread takes eight consecutive channels a chunk — the row's
    /// weights of each block as one 16-byte load, each base vector's as one 32-byte load reused over the four blocks:
    /// the first form's single strided loads (hundreds a thread, a round trip a batch) made the phase latency-bound at
    /// 42–46 µs (2026-10-27). The Gram is `phase_a_gram`'s, on workgroups of its own (it rode on rows 0..n here —
    /// 25 extra dots a channel-eight and two extra reductions on those workgroups: A's slowest at 12.2 µs against the
    /// others' ~6, and the first barrier waits for the slowest; an12s, 2026-10-27).
    #[inline(always)]
    unsafe fn phase_a<const NP: usize>(w_row: *const u16, streams: *const f32, fyv: *const f32, scratch: *mut f32, row: usize, chunk: usize, i0: usize, t: usize, wave: usize, red: *mut f32) {
        const CH: usize = H / (256 * 8) / A_CHUNKS;
        let mut acc = [[[0f32; BASE]; HC]; NP];
        let mut ch = 0;
        while ch < CH {
            let d8 = t * 8 + (chunk * CH + ch) * 2048;
            // the row's four blocks' weights at these channels, held; the base vectors one at a time
            let mut wv = [[0f32; 8]; HC];
            let mut j = 0;
            while j < HC {
                wv[j] = widen8_at(w_row.add(j * H + d8));
                j += 1;
            }
            let mut a = 0;
            while a < BASE {
                let mut i = 0;
                while i < NP {
                    let xv: [f32; 8] = if a < HC { core::ptr::read(streams.add((i0 + i) * HC * H + a * H + d8) as *const [f32; 8]) } else { core::ptr::read(fyv.add((i0 + i) * H + d8) as *const [f32; 8]) };
                    let mut j = 0;
                    while j < HC {
                        let mut d = 0f32;
                        let mut k = 0;
                        while k < 8 {
                            d += wv[j][k] * xv[k];
                            k += 1;
                        }
                        acc[i][j][a] += d;
                        j += 1;
                    }
                    i += 1;
                }
                a += 1;
            }
            ch += 1;
        }
        // one reduction round: every value a wave sum on the VALU (DPP butterflies; the lockstep `ds_bpermute` form
        // before) — lane 0's into LDS `[8 waves][NP·DOTS]`, a thread a value
        const W: usize = 8;
        let stride = NP * DOTS;
        let mut i = 0;
        while i < NP {
            let v = wave_sum_dpp_n::<DOTS>(core::array::from_fn(|k| acc[i][k / BASE][k % BASE]));
            if t % 32 == 0 {
                let mut k = 0;
                while k < DOTS {
                    *red.add(wave * stride + i * DOTS + k) = v[k];
                    k += 1;
                }
            }
            i += 1;
        }
        workgroup_sync();
        if t < stride {
            let mut sum = 0f32;
            let mut w = 0;
            while w < W {
                sum += *red.add(w * stride + t);
                w += 1;
            }
            let (i, k) = (t / DOTS, t % DOTS);
            *scratch.add((i0 + i) * PRE_SCRATCH + SCR_DOTS + (chunk * A_ROWS + row) * DOTS + k) = sum;
        }
        workgroup_sync();
    }

    /// Token `i`'s Gram of its five base vectors over one chunk, on a workgroup of its own (`A_GRID + i · A_CHUNKS +
    /// chunk`): the same loads as a row's base vectors, 25 dots a channel-eight, one reduction round.
    #[inline(always)]
    unsafe fn phase_a_gram(streams: *const f32, fyv: *const f32, scratch: *mut f32, i: usize, chunk: usize, t: usize, wave: usize, red: *mut f32) {
        const CH: usize = H / (256 * 8) / A_CHUNKS;
        let mut gram = [[0f32; BASE]; BASE];
        let st = streams.add(i * HC * H);
        let mut ch = 0;
        while ch < CH {
            let d8 = t * 8 + (chunk * CH + ch) * 2048;
            let mut v = [[0f32; 8]; BASE];
            let mut a = 0;
            while a < HC {
                v[a] = core::ptr::read(st.add(a * H + d8) as *const [f32; 8]);
                a += 1;
            }
            v[HC] = core::ptr::read(fyv.add(i * H + d8) as *const [f32; 8]);
            let mut a = 0;
            while a < BASE {
                let mut b = 0;
                while b < BASE {
                    let mut d = 0f32;
                    let mut k = 0;
                    while k < 8 {
                        d += v[a][k] * v[b][k];
                        k += 1;
                    }
                    gram[a][b] += d;
                    b += 1;
                }
                a += 1;
            }
            ch += 1;
        }
        const W: usize = 8;
        const G: usize = BASE * BASE;
        let g = wave_sum_dpp_n::<G>(core::array::from_fn(|k| gram[k / BASE][k % BASE]));
        if t % 32 == 0 {
            let mut k = 0;
            while k < G {
                *red.add(wave * G + k) = g[k];
                k += 1;
            }
        }
        workgroup_sync();
        if t < G {
            let mut sum = 0f32;
            let mut w = 0;
            while w < W {
                sum += *red.add(w * G + t);
                w += 1;
            }
            *scratch.add(i * PRE_SCRATCH + SCR_GRAM + chunk * G + t) = sum;
        }
        workgroup_sync();
    }

    /// Phase B on one wave for token `i`: the attention half's coefficients from its dots and layer l's FFN mix
    /// (`alpha[j][a]`: how the stood-in stream j is made of the base vectors), its mix; the no-attention streams'
    /// weights, the FFN half's coefficients, its mix; c and the collapses' weights. The lanes do different work: lane r
    /// its row's coefficient (its own dots against the weights), the norm's hundred Gram terms four a lane and one
    /// wave sum, the mix an entry a lane (`mix_token`), the mixes' values read across the lanes where another lane's
    /// is needed. Every operand loaded once as a batch; no LDS.
    #[inline(always)]
    unsafe fn phase_b(i: usize, mix_l: *const f32, a_base: *const f32, a_scale: *const f32, f_base: *const f32, f_scale: *const f32, c: *mut f32, scratch: *mut f32, lane: u32) {
        use common::gfx11::lane_read;
        let lu = lane as usize;
        let sc = scratch.add(i * PRE_SCRATCH);
        let ml = mix_l.add(i * MIX);
        // lane l < 16 owns coefficient row 8 + l (the comb entry it will hold), lanes 16–23 rows 0–7 (pre, post)
        let row = if lu < HC * HC { 2 * HC + lu } else if lu < HC_COEFS { lu - HC * HC } else { 0 };
        let a_in = mix_in(a_base, a_scale, lu);
        let f_in = mix_in(f_base, f_scale, lu);
        // alpha[j][a]: stood-in stream j = Σ_a alpha[j][a] · v_a — comb_kj over the streams, post_j over fyv
        let mut alpha = [[0f32; BASE]; HC];
        let mut j = 0;
        while j < HC {
            let mut a = 0;
            while a < HC {
                alpha[j][a] = *ml.add(2 * HC + a * HC + j);
                a += 1;
            }
            alpha[j][HC] = *ml.add(HC + j);
            j += 1;
        }
        // this lane's Gram term (a, b) for the norms, lanes under BASE²
        let (ga, gb) = if lu < BASE * BASE { (lu / BASE, lu % BASE) } else { (0, 0) };
        let mut g = 0f32;
        let mut ch = 0;
        while ch < A_CHUNKS {
            g += *sc.add(SCR_GRAM + ch * BASE * BASE + ga * BASE + gb);
            ch += 1;
        }
        // the lane's row's dots of a half, the chunks' partials: forty independent loads issued together, then summed
        let load_dots = |h: usize| -> [f32; DOTS] {
            let mut p = [[0f32; DOTS]; A_CHUNKS];
            let mut ch = 0;
            while ch < A_CHUNKS {
                let mut k = 0;
                while k < DOTS {
                    p[ch][k] = *sc.add(SCR_DOTS + (ch * A_ROWS + h * HC_COEFS + row) * DOTS + k);
                    k += 1;
                }
                ch += 1;
            }
            let mut d = [0f32; DOTS];
            let mut k = 0;
            while k < DOTS {
                let mut ch = 0;
                while ch < A_CHUNKS {
                    d[k] += p[ch][k];
                    ch += 1;
                }
                k += 1;
            }
            d
        };
        // a half's coefficient on lane r: its row's dots against the streams' weights `al`, over the norm — the
        // norm's Σ_j al_ja al_jb G_ab as this lane's (a, b) term, summed across the wave
        // (a lane-indexed read of a register array is a stack object in scratch: the entries are selected instead)
        let pick = |row: &[f32; BASE], idx: usize| -> f32 {
            let mut v = row[0];
            let mut a = 1;
            while a < BASE {
                if idx == a {
                    v = row[a];
                }
                a += 1;
            }
            v
        };
        let coef = |al: &[[f32; BASE]; HC], dots: &[f32; DOTS]| -> f32 {
            let mut term = 0f32;
            let mut j = 0;
            while j < HC {
                term = fma(pick(&al[j], ga), pick(&al[j], gb), term);
                j += 1;
            }
            let ss = wave_sum_dpp(if lu < BASE * BASE { term * g } else { 0.0 });
            let mut acc = 0f32;
            let mut j = 0;
            while j < HC {
                let mut a = 0;
                while a < BASE {
                    acc = fma(al[j][a], dots[j * BASE + a], acc);
                    a += 1;
                }
                j += 1;
            }
            acc * common::gfx11::rsq(ss * (1.0 / (HC * H) as f32) + RMS_NORM_EPS)
        };
        let dots = load_dots(0);
        let ca = coef(&alpha, &dots);
        let (pp_a, cb_a) = mix_token(ca, a_in, lu, sc.add(SCR_MIX));
        // the no-attention streams: s''_j = Σ_k comb_a[k][j] · s'_k, so alpha''[j][a] = Σ_k comb_a[k][j] · alpha[k][a];
        // comb_a's entry (k, j) is on lane 4k + j
        let mut comb = [[0f32; HC]; HC];
        let mut k = 0;
        while k < HC {
            let mut j = 0;
            while j < HC {
                comb[k][j] = lane_read((k * HC + j) as u32, cb_a);
                j += 1;
            }
            k += 1;
        }
        let mut alpha2 = [[0f32; BASE]; HC];
        let mut j = 0;
        while j < HC {
            let mut a = 0;
            while a < BASE {
                let mut v = 0f32;
                let mut k = 0;
                while k < HC {
                    v = fma(comb[k][j], alpha[k][a], v);
                    k += 1;
                }
                alpha2[j][a] = v;
                a += 1;
            }
            j += 1;
        }
        let dots = load_dots(1);
        let cf = coef(&alpha2, &dots);
        let (pp_f, _) = mix_token(cf, f_in, lu, sc.add(SCR_MIX + MIX));
        // pre_a[j] on lane j, post_a[j] on lane 4 + j, pre_f[j] on lane j (the FFN mix's): read across
        let mut pre_a = [0f32; HC];
        let mut post_a = [0f32; HC];
        let mut pre_f = [0f32; HC];
        let mut j = 0;
        while j < HC {
            pre_a[j] = lane_read(j as u32, pp_a);
            post_a[j] = lane_read((HC + j) as u32, pp_a);
            pre_f[j] = lane_read(j as u32, pp_f);
            j += 1;
        }
        // c = Σ_j pre_f[j] · post_a[j]; the collapses' weights: beta_a = Σ_j pre_a[j] alpha[j][a], beta0_a = Σ_j pre_f[j]
        // alpha''[j][a] — a lane an entry, constant-indexed selects
        if lu < BASE {
            let (mut beta, mut beta0) = (0f32, 0f32);
            let mut a = 0;
            while a < BASE {
                if lu == a {
                    let mut j = 0;
                    while j < HC {
                        beta = fma(pre_a[j], alpha[j][a], beta);
                        beta0 = fma(pre_f[j], alpha2[j][a], beta0);
                        j += 1;
                    }
                }
                a += 1;
            }
            *sc.add(SCR_BETA + lu) = beta;
            *sc.add(SCR_BETA + BASE + lu) = beta0;
        }
        if lane == 0 {
            let mut cc = 0f32;
            let mut j = 0;
            while j < HC {
                cc = fma(pre_f[j], post_a[j], cc);
                j += 1;
            }
            *c.add(i) = cc;
        }
    }

    /// The whole chain of one card as one launch (the module), `CHAIN_GRID` workgroups all resident, its phases behind
    /// grid barriers on `tab[TAB_COUNT]`'s words: the peer's wait for the leader's shared output; A — the two hyper-
    /// connection halves' base dots and the Gram (workgroups under `A_GRID`); B — the mixes, a wave a token (workgroup
    /// 0); C — the collapses `x`, `x0` (workgroups under H / 256); R — the chosen heads' q, k, v rows of `sq` (every
    /// wave looping the selection, `E4_ROWS` rows at a time); F — their f_b, g_b rows; S — `kda::step_head` read-only
    /// over the chosen heads (workgroups under `NSEL`); O — the composed rows over the readouts into `part`
    /// (workgroups under `OUT_WGS`; the peer's last one posts the leader's word), and on the leader the router's rows
    /// over the no-attention input into `scores` with `k`, and over the second margin's input (`x_l + fyv`, normed with
    /// layer l's post-norm) into `s2` — what `pred_input` + `hc_route_ahead` made. `tab`: `TAB_LEN` device addresses.
    /// LDS CHAIN_LDS.
    #[no_mangle]
    pub extern "gpu-kernel" fn an_chain(tab: *const u64, streams: *const f32, fyv: *const f32, a_fn: *const u16, f_fn: *const u16, sq: *const u8, fg_w: *const u8, crows: *const u16, router: *const u16, x: *mut f32, smix: *mut f32, fg: *mut f32, normed: *mut f32, mem: *const f32, mix_l: *const f32, scratch: *mut f32, c: *mut f32, count: *mut u32, x0: *mut f32, a_base: *const f32, a_scale: *const f32, f_base: *const f32, f_scale: *const f32, head0: u32, value: u32, n: u32, layer: u32) {
        unsafe {
            // the streamed operands and phase B's are arguments (a kernel argument's loads are global; an address read
            // from the table is generic — flat loads, which count on the scalar counter too, so every LDS wait drained
            // them: the an9s ISA cut, 2026-10-27); the rest in `tab`
            const RR: usize = crate::hc::gfx11_consts::ROUTE_ROWS;
            let n = n as usize;
            let t = workitem_id_x() as usize;
            let (wave, lane) = (t / 32, (t % 32) as u32);
            let wg = workgroup_id_x() as usize;
            let l = lds::<f32>();
            let at = |k: usize| *tab.add(k) as usize;
            // the head table in LDS (its reads through a generic pointer were a round trip a wave a row group)
            let heads = l.add((CHAIN_LDS as usize - 4 * NSEL) / 4) as *mut u32;
            if t < NSEL {
                *heads.add(t) = *(at(TAB_HEADS) as *const u32).add(t);
            }
            workgroup_sync();
            let heads = heads as *const u32;
            let lead = at(TAB_PART_WORD) == 0;
            // stats (a stats code object only), ticks of the 100 MHz clock as every wave sees them — the hyper-connection
            // phases apart, after three builds left their sum at 42–48 µs (2026-10-27): slot 0 phase A (after the peer's
            // wait), 1 the first grid barrier's wait, 2 phase B (workgroup 0's waves; 0 elsewhere), 3 the second barrier,
            // phase C and the third barrier. A `stats_tail` object stamps the tail instead: slot 0 phase R alone, 1 the
            // fourth barrier, F and the fifth, 2 S (with the sixth), 3 O to the end.
            const TAIL: bool = cfg!(feature = "stats_tail");
            let kt_w = common::stat_clock!();
            // the peer: the leader's shared output must be in (its word in this card's copy)
            let fyv_word = at(TAB_FYV_WORD) as *const u32;
            if !fyv_word.is_null() {
                if t == 0 {
                    await_word(fyv_word, value);
                }
                workgroup_sync();
                fence(Ordering::Acquire);
            }
            let kt0 = common::stat_clock!();
            let _ = kt_w;
            // phase A: a coefficient row a workgroup (the attention half's rows, then the FFN half's)
            if wg < A_GRID {
                let (row, chunk) = (wg / A_CHUNKS, wg % A_CHUNKS);
                let w_row = if row < HC_COEFS { a_fn.add(row * HC * H) } else { f_fn.add((row - HC_COEFS) * HC * H) };
                let mut i0 = 0;
                while i0 < n {
                    if n - i0 >= 2 {
                        phase_a::<2>(w_row, streams, fyv, scratch, row, chunk, i0, t, wave, l);
                    } else {
                        phase_a::<1>(w_row, streams, fyv, scratch, row, chunk, i0, t, wave, l);
                    }
                    i0 += 2;
                }
            } else if wg < A_GRID + n * A_CHUNKS {
                let g = wg - A_GRID;
                phase_a_gram(streams, fyv, scratch, g / A_CHUNKS, g % A_CHUNKS, t, wave, l);
            }
            let kta = common::stat_clock!();
            if !TAIL {
                common::stat!(0, kta.wrapping_sub(kt0) & common::stats::CLOCK_MASK);
            }
            barrier(count, 0, t, wg, value);
            let ktb0 = common::stat_clock!();
            if !TAIL {
                common::stat!(1, ktb0.wrapping_sub(kta) & common::stats::CLOCK_MASK);
            }
            // phase B: workgroup 0, a wave a token
            if wg == 0 && wave < n {
                phase_b(wave, mix_l, a_base, a_scale, f_base, f_scale, c, scratch, lane);
            }
            let ktb1 = common::stat_clock!();
            if !TAIL {
                common::stat!(2, ktb1.wrapping_sub(ktb0) & common::stats::CLOCK_MASK);
            }
            barrier(count, 1, t, wg, value);
            // phase C: the collapses, a thread eight channels (workgroups under H / 2048), every token
            if wg < H / 2048 {
                let d8 = wg * 2048 + t * 8;
                let mut i = 0;
                while i < n {
                    let (st, sc) = (streams.add(i * HC * H), scratch.add(i * PRE_SCRATCH + SCR_BETA));
                    let mut v = [[0f32; 8]; BASE];
                    let mut a = 0;
                    while a < HC {
                        v[a] = core::ptr::read(st.add(a * H + d8) as *const [f32; 8]);
                        a += 1;
                    }
                    v[HC] = core::ptr::read(fyv.add(i * H + d8) as *const [f32; 8]);
                    let (mut xa, mut x0a) = ([0f32; 8], [0f32; 8]);
                    let mut a = 0;
                    while a < BASE {
                        let (ba, b0a) = (*sc.add(a), *sc.add(BASE + a));
                        let mut k = 0;
                        while k < 8 {
                            xa[k] += ba * v[a][k];
                            x0a[k] += b0a * v[a][k];
                            k += 1;
                        }
                        a += 1;
                    }
                    core::ptr::write(x.add(i * H + d8) as *mut [f32; 8], xa);
                    core::ptr::write(x0.add(i * H + d8) as *mut [f32; 8], x0a);
                    i += 1;
                }
            }
            barrier(count, 2, t, wg, value);
            let kt1 = common::stat_clock!();
            if !TAIL {
                common::stat!(3, kt1.wrapping_sub(ktb1) & common::stats::CLOCK_MASK);
            }
            // phase R: the small rows and the chosen heads' q, k, v rows of `sq` over `x` normed with the input norm,
            // into `smix` at the rows' own places (the step reads it as `kda_step` reads `sq`'s output); every wave
            // loops the selection E4_ROWS rows at a time
            {
                let ln = at(TAB_INPUT_LN) as *const u16;
                let map = |r: usize| {
                    if r < SMALL_ROWS {
                        r
                    } else {
                        let j = r - SMALL_ROWS;
                        let (part, rem) = (j / (NSEL * D), j % (NSEL * D));
                        SMALL_ROWS + part * PART_QKV + *heads.add(rem / D) as usize * D + rem % D
                    }
                };
                let emit = |row: usize, i: usize, y: f32| *smix.add(i * M_SMALL + row) = y;
                let xs = |_: usize, i: usize| x.add(i * H) as *const f32;
                // the rows a wave: the selection in one round of the grid's waves at widths 1–2 (`R_ROWS` = 3; at
                // `E4_ROWS` = 2 the first 704 waves went round twice — twenty serial batches of loads instead of
                // ten, with the rest of the grid idle: the phase is one wave's round trips, not bytes; 2026-10-27);
                // widths 3–4 keep two (three rows at four tokens spill), their host run being the longer
                let e4 = Exp4::<2, 8>::at::<H>(sq, M_SMALL);
                macro_rules! rows_pass {
                    ($np:expr, $r:expr) => {{
                        let mut vrow = (wg * CHAIN_WAVES + wave) * $r;
                        while vrow < SEL_ROWS {
                            exp4_rows_direct_rows::<H, true, 2, 8, $r, $np, true>(t as u32, e4, ln, SEL_ROWS, n, RMS_NORM_EPS, vrow, &map, &xs, &emit);
                            vrow += CHAIN_GRID as usize * CHAIN_WAVES * $r;
                        }
                    }};
                }
                match n {
                    1 => rows_pass!(1, R_ROWS),
                    2 => rows_pass!(2, R_ROWS),
                    3 => rows_pass!(3, E4_ROWS),
                    _ => rows_pass!({ common::gemv::FEW }, E4_ROWS),
                }
            }
            // the tail's first stamp before the barrier: R's own time a wave, the barrier's wait in the next slot
            // (stamped after it, R and its barrier were one number — 45 µs mean and slowest alike, 2026-10-27)
            let ktr = common::stat_clock!();
            if TAIL {
                common::stat!(0, ktr.wrapping_sub(kt1) & common::stats::CLOCK_MASK);
            }
            barrier(count, 3, t, wg, value);
            // phase F: the chosen heads' f_b and g_b rows of `fg` over the gates' first halves in `smix`
            {
                let smix = smix as *const f32;
                let map = |j: usize| {
                    let (part, rem) = (j / (NSEL * D), j % (NSEL * D));
                    part * PART_QKV + *heads.add(rem / D) as usize * D + rem % D
                };
                let emit = |row: usize, i: usize, y: f32| *fg.add(i * 2 * PART_QKV + row) = y;
                let xs = |row: usize, i: usize| if row >= PART_QKV { smix.add(D + i * M_SMALL) } else { smix.add(i * M_SMALL) };
                let e4 = Exp4::<2, 8>::at::<KDA_DIM>(fg_w, 2 * PART_QKV);
                let mut vrow = (wg * CHAIN_WAVES + wave) * E4_ROWS;
                while vrow < SEL_FG_ROWS {
                    match n {
                        1 => exp4_rows_direct_rows::<KDA_DIM, false, 2, 8, E4_ROWS, 1, true>(t as u32, e4, core::ptr::null(), SEL_FG_ROWS, n, 0.0, vrow, &map, &xs, &emit),
                        2 => exp4_rows_direct_rows::<KDA_DIM, false, 2, 8, E4_ROWS, 2, true>(t as u32, e4, core::ptr::null(), SEL_FG_ROWS, n, 0.0, vrow, &map, &xs, &emit),
                        3 => exp4_rows_direct_rows::<KDA_DIM, false, 2, 8, E4_ROWS, 3, true>(t as u32, e4, core::ptr::null(), SEL_FG_ROWS, n, 0.0, vrow, &map, &xs, &emit),
                        _ => exp4_rows_direct_rows::<KDA_DIM, false, 2, 8, E4_ROWS, { common::gemv::FEW }, true>(t as u32, e4, core::ptr::null(), SEL_FG_ROWS, n, 0.0, vrow, &map, &xs, &emit),
                    }
                    vrow += CHAIN_GRID as usize * CHAIN_WAVES * E4_ROWS;
                }
            }
            barrier(count, 4, t, wg, value);
            let ktf = common::stat_clock!();
            if TAIL {
                common::stat!(1, ktf.wrapping_sub(ktr) & common::stats::CLOCK_MASK);
            }
            // phase S: the chosen heads' step from the state as it stands, nothing stored back
            if wg < NSEL {
                let (smix, fg) = (smix as *const f32, fg as *const f32);
                let hl = *heads.add(wg) as usize;
                crate::kda::gfx11::step_head::<false>(smix.add(SMALL_ROWS), at(TAB_TAPS) as *const u16, at(TAB_WINDOW) as *mut f32, fg, at(TAB_DT_BIAS) as *const f32, at(TAB_A_LOG) as *const f32, smix.add(2 * KDA_DIM), fg.add(PART_QKV), mem as *mut f32, at(TAB_O_NORM) as *const f32, normed, head0 as usize + hl, hl, n as u32, M_SMALL as u32, 2 * PART_QKV as u32, M_SMALL as u32, core::ptr::null_mut(), core::ptr::null_mut());
            }
            barrier(count, 5, t, wg, value);
            let kts = common::stat_clock!();
            if TAIL {
                common::stat!(2, kts.wrapping_sub(ktf) & common::stats::CLOCK_MASK);
            }
            // phase O
            let normed = normed as *const f32;
            if wg < OUT_WGS {
                // a composed row a wave: the lane's eight columns of each 256-wide step gathered from the head the
                // step's columns belong to (a head is 128 columns: a step is two heads, a lane's eight within one)
                let part = at(TAB_PART) as *mut f32;
                let row = wg * 8 + wave;
                let wr = crows.add(row * SEL_W);
                // every token's row read whether or not it is in the step (the buffers hold SPEC_TOKENS rows; a
                // load under `i < n` is a branch arm the compiler sinks it into — a round trip a step, 39 drains in
                // the phase, 2026-10-27); the stores keep `i < n`
                let mut acc = [0f32; 4];
                let mut s = 0;
                while s < SEL_W / 256 {
                    let col = s * 256 + lane as usize * 8;
                    let wv = widen8_at(wr.add(col));
                    let src = *heads.add(col / D) as usize * D + col % D;
                    let mut i = 0;
                    while i < 4 {
                        let xv: [f32; 8] = core::ptr::read(normed.add(i * PART_QKV + src) as *const [f32; 8]);
                        let mut j = 0;
                        while j < 8 {
                            acc[i] = fma(wv[j], xv[j], acc[i]);
                            j += 1;
                        }
                        i += 1;
                    }
                    s += 1;
                }
                let mut i = 0;
                while i < 4 {
                    let y = wave_sum_dpp(acc[i]);
                    if lane == 0 && i < n {
                        *part.add(i * E + row) = y;
                    }
                    i += 1;
                }
                if !lead {
                    // the partials out over the link; the last workgroup posts the leader's word
                    fence(Ordering::Release);
                    workgroup_sync();
                    let ocount = at(TAB_OCOUNT) as *mut u32;
                    if t == 0 && AtomicU32::from_ptr(ocount).fetch_add(1, Ordering::AcqRel) == OUT_WGS as u32 - 1 {
                        AtomicU32::from_ptr(ocount).store(0, Ordering::Relaxed);
                        AtomicU32::from_ptr(at(TAB_PART_WORD) as *mut u32).store(value, Ordering::Release);
                    }
                }
            } else if lead && wg < 3 * OUT_WGS {
                // the leader: a router row a wave over the no-attention input normed with layer l + 1's post-norm
                // (k from its norm), then over the second margin's input `x_l + fyv` normed with layer l's
                let second = wg >= 2 * OUT_WGS;
                let row = E + (wg - if second { 2 * OUT_WGS } else { OUT_WGS }) * 8 + wave;
                let wr = router.add(row * H);
                let ln = at(if second { TAB_LN_L } else { TAB_LN1 }) as *const u16;
                let xs = if second { at(TAB_XL) as *const f32 } else { x0 as *const f32 };
                // the shared output added for the second margin's rows: loaded for both (the same SPEC_TOKENS rows)
                // and weighted by a uniform 0 or 1 — no branch for the loads to sink into (see the partial rows)
                let fw = if second { 1.0 } else { 0.0 };
                let mut acc = [0f32; 4];
                let mut ss = [0f32; 4];
                let mut s = 0;
                while s < H / 256 {
                    let col = s * 256 + lane as usize * 8;
                    let wv = widen8_at(wr.add(col));
                    let lv = widen8_at(ln.add(col));
                    let mut i = 0;
                    while i < 4 {
                        let xv: [f32; 8] = core::ptr::read(xs.add(i * H + col) as *const [f32; 8]);
                        let fv: [f32; 8] = core::ptr::read(fyv.add(i * H + col) as *const [f32; 8]);
                        let mut j = 0;
                        while j < 8 {
                            let v = fma(fv[j], fw, xv[j]);
                            acc[i] = fma(v * lv[j], wv[j], acc[i]);
                            ss[i] = fma(v, v, ss[i]);
                            j += 1;
                        }
                        i += 1;
                    }
                    s += 1;
                }
                let out = at(if second { TAB_S2 } else { TAB_SCORES }) as *mut f32;
                let mut i = 0;
                while i < 4 {
                    let y = wave_sum_dpp(acc[i]);
                    let rr = common::gfx11::rsq(wave_sum_dpp(ss[i]) * (1.0 / H as f32) + RMS_NORM_EPS);
                    if lane == 0 && i < n {
                        *out.add(i * RR + row) = y * rr;
                        if !second && row == E {
                            *(at(TAB_K) as *mut f32).add(i) = *c.add(i) * rr;
                        }
                    }
                    i += 1;
                }
            }
            if TAIL {
                common::stat!(3, common::stat_clock!().wrapping_sub(kts) & common::stats::CLOCK_MASK);
            }
            let _ = layer;
        }
    }

    /// The leader's shared output (`[n][H]`, its rows at `fyv`) into the peer's receiving buffer and `value` into the
    /// peer's word, behind the launches before it (`ffn_down_shared`), long before the peer's chain. Grid 1, no LDS.
    #[no_mangle]
    pub extern "gpu-kernel" fn an_send(fyv: *const f32, dst: *mut f32, word: *mut u32, value: u32, n: u32) {
        unsafe {
            let t = workitem_id_x() as usize;
            let mut i = t;
            while i < n as usize * H {
                *dst.add(i) = *fyv.add(i);
                i += 256;
            }
            fence(Ordering::Release);
            workgroup_sync();
            if t == 0 {
                AtomicU32::from_ptr(word).store(value, Ordering::Release);
            }
        }
    }

}
