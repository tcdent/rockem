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
pub const PRE_SCRATCH: usize = 1152;
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

    /// The stood-in and no-attention streams are linear in five base vectors a token — layer l's four FFN-input streams
    /// and `fyv` — so both hyper-connection halves' coefficients are combinations of base dots (a row's four blocks
    /// against the five vectors) and their norms come from the vectors' Gram; the collapses are five-term sums. One
    /// streaming pass makes the dots and the Gram (phase A), one wave a token makes the two mixes from them (phase B),
    /// every workgroup then collapses (phase C): two grid barriers (the first form materialized the streams twice and
    /// crossed six, 92 µs on the box, 2026-10-26).
    pub const BASE: usize = HC + 1;
    /// A row's dots a token: `[HC blocks][BASE]`.
    pub const DOTS: usize = HC * BASE;
    /// Phase A's rows: the attention half's coefficient rows, then the FFN half's.
    pub const A_GRID: usize = 2 * HC_COEFS;
    /// `an_pre`'s scratch a token, floats: the rows' dots `[A_GRID][DOTS]`, the Gram `[BASE][BASE]`, the two mixes
    /// `[2][MIX]`, the collapses' weights `[2][BASE]`.
    pub const SCR_DOTS: usize = 0;
    pub const SCR_GRAM: usize = SCR_DOTS + A_GRID * DOTS;
    pub const SCR_MIX: usize = SCR_GRAM + BASE * BASE;
    pub const SCR_BETA: usize = SCR_MIX + 2 * MIX;
    pub const SCRATCH: usize = SCR_BETA + 2 * BASE;
    const _: () = assert!(SCRATCH <= PRE_SCRATCH && A_GRID as u32 <= CHAIN_GRID && HC_COEFS <= 32 && BASE * BASE <= 32 && 8 * (4 * DOTS + BASE * BASE) * 4 <= CHAIN_LDS as usize);

    /// Phase A for `NP` tokens from `i0`: this workgroup's row's dots with the base vectors over its four blocks, and
    /// (workgroup `i < n`, the first pass) token `i`'s Gram; the partials of all of them held in registers, one
    /// reduction round. A thread takes eight consecutive channels a chunk — the row's weights of each block as one
    /// 16-byte load, each base vector's as one 32-byte load reused over the four blocks: the first form's single
    /// strided loads (hundreds a thread, a round trip a batch) made the phase latency-bound at 42–46 µs (2026-10-27).
    #[inline(always)]
    unsafe fn phase_a<const NP: usize>(w_row: *const u16, streams: *const f32, fyv: *const f32, scratch: *mut f32, wg: usize, n: usize, i0: usize, t: usize, lane: u32, wave: usize, red: *mut f32) {
        const CH: usize = H / (256 * 8);
        let mut acc = [[[0f32; BASE]; HC]; NP];
        let mut gram = [[0f32; BASE]; BASE];
        let mut ch = 0;
        while ch < CH {
            let d8 = t * 8 + ch * 2048;
            // the Gram of token `wg`'s base vectors (workgroups under n), in the first pass, before the row's weights
            // are live (the five vectors and the row's four blocks together spilled)
            if wg < n && i0 == 0 {
                let st = streams.add(wg * HC * H);
                let mut v = [[0f32; 8]; BASE];
                let mut a = 0;
                while a < HC {
                    v[a] = core::ptr::read(st.add(a * H + d8) as *const [f32; 8]);
                    a += 1;
                }
                v[HC] = core::ptr::read(fyv.add(wg * H + d8) as *const [f32; 8]);
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
            }
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
        // one reduction round: every value a wave sum, lane 0's into LDS `[8 waves][NP·DOTS + BASE²]`, a thread a value
        const W: usize = 8;
        let stride = NP * DOTS + BASE * BASE;
        let mut i = 0;
        while i < NP {
            let mut j = 0;
            while j < HC {
                let mut a = 0;
                while a < BASE {
                    let v = wave_sum(lane, acc[i][j][a]);
                    if lane == 0 {
                        *red.add(wave * stride + i * DOTS + j * BASE + a) = v;
                    }
                    a += 1;
                }
                j += 1;
            }
            i += 1;
        }
        let mut a = 0;
        while a < BASE {
            let mut b = 0;
            while b < BASE {
                let v = wave_sum(lane, gram[a][b]);
                if lane == 0 {
                    *red.add(wave * stride + NP * DOTS + a * BASE + b) = v;
                }
                b += 1;
            }
            a += 1;
        }
        workgroup_sync();
        if t < stride {
            let mut sum = 0f32;
            let mut w = 0;
            while w < W {
                sum += *red.add(w * stride + t);
                w += 1;
            }
            if t < NP * DOTS {
                let (i, k) = (t / DOTS, t % DOTS);
                *scratch.add((i0 + i) * PRE_SCRATCH + SCR_DOTS + wg * DOTS + k) = sum;
            } else if wg < n && i0 == 0 {
                *scratch.add(wg * PRE_SCRATCH + SCR_GRAM + (t - NP * DOTS)) = sum;
            }
        }
        workgroup_sync();
    }

    /// Phase B on one wave for token `i`: the attention half's coefficients from its dots and layer l's FFN mix
    /// (`alpha[j][a]`: how the stood-in stream j is made of the base vectors), its mix; the no-attention streams'
    /// weights, the FFN half's coefficients, its mix; c and the collapses' weights. `lds`: `B_LDS` floats of this
    /// wave's — the mix terms, the Gram and alpha'' are read or made once into it (read inside the loops from memory
    /// they were a few hundred dependent loads a lane); the lane's own row's dots are twenty registers a half.
    #[inline(always)]
    unsafe fn phase_b(i: usize, mix_l: *const f32, a_base: *const f32, a_scale: *const f32, f_base: *const f32, f_scale: *const f32, c: *mut f32, scratch: *mut f32, lane: u32, lds: *mut f32) {
        let lu = lane as usize;
        let sc = scratch.add(i * PRE_SCRATCH);
        let ml = mix_l.add(i * MIX);
        let coefs = lds; // [HC_COEFS]
        let tr = lds.add(32); // [HC·HC]
        let gram = lds.add(48); // [BASE·BASE]
        let alpha = lds.add(80); // [HC][BASE]
        let alpha2 = lds.add(100); // [HC][BASE]
        // alpha[j][a]: stood-in stream j = Σ_a alpha[j][a] · v_a — comb_kj over the streams, post_j over fyv; a lane an entry
        if lu < HC * BASE {
            let (j, a) = (lu / BASE, lu % BASE);
            *alpha.add(lu) = if a < HC { *ml.add(2 * HC + a * HC + j) } else { *ml.add(HC + j) };
        }
        if lu < BASE * BASE {
            *gram.add(lu) = *sc.add(SCR_GRAM + lu);
        }
        common::gfx11::wave_lds_sync();
        let row = if lu < HC_COEFS { lu } else { 0 };
        let mut dots = [0f32; DOTS];
        let mut k = 0;
        while k < DOTS {
            dots[k] = *sc.add(SCR_DOTS + row * DOTS + k);
            k += 1;
        }
        // the stood-in streams' sum of squares from the Gram, then lane r's attention coefficient
        let mut ss = 0f32;
        let mut acc = 0f32;
        let mut j = 0;
        while j < HC {
            let mut a = 0;
            while a < BASE {
                let al = *alpha.add(j * BASE + a);
                acc += al * dots[j * BASE + a];
                let mut b = 0;
                while b < BASE {
                    ss += al * *alpha.add(j * BASE + b) * *gram.add(a * BASE + b);
                    b += 1;
                }
                a += 1;
            }
            j += 1;
        }
        if lu < HC_COEFS {
            *coefs.add(lu) = acc * rsqrt(ss / (HC * H) as f32 + RMS_NORM_EPS);
        }
        common::gfx11::wave_lds_sync();
        let mix_a = sc.add(SCR_MIX);
        mix_token(|k| *coefs.add(k), a_base, a_scale, lane, tr, mix_a);
        common::gfx11::wave_lds_sync();
        // the no-attention streams: s''_j = Σ_k comb_a[k][j] · s'_k, so alpha''[j][a] = Σ_k comb_a[k][j] · alpha[k][a]; a lane an entry
        if lu < HC * BASE {
            let (j, a) = (lu / BASE, lu % BASE);
            let mut v = 0f32;
            let mut k = 0;
            while k < HC {
                v += *mix_a.add(2 * HC + k * HC + j) * *alpha.add(k * BASE + a);
                k += 1;
            }
            *alpha2.add(lu) = v;
        }
        common::gfx11::wave_lds_sync();
        let mut k = 0;
        while k < DOTS {
            dots[k] = *sc.add(SCR_DOTS + (HC_COEFS + row) * DOTS + k);
            k += 1;
        }
        let mut ss = 0f32;
        let mut acc = 0f32;
        let mut j = 0;
        while j < HC {
            let mut a = 0;
            while a < BASE {
                let al = *alpha2.add(j * BASE + a);
                acc += al * dots[j * BASE + a];
                let mut b = 0;
                while b < BASE {
                    ss += al * *alpha2.add(j * BASE + b) * *gram.add(a * BASE + b);
                    b += 1;
                }
                a += 1;
            }
            j += 1;
        }
        if lu < HC_COEFS {
            *coefs.add(lu) = acc * rsqrt(ss / (HC * H) as f32 + RMS_NORM_EPS);
        }
        common::gfx11::wave_lds_sync();
        let mix_f = sc.add(SCR_MIX + MIX);
        mix_token(|k| *coefs.add(k), f_base, f_scale, lane, tr, mix_f);
        common::gfx11::wave_lds_sync();
        // c = Σ_j pre_f[j] · post_a[j]; the collapses' weights over the base vectors: beta_a = Σ_j pre_a[j] alpha[j][a],
        // beta0_a = Σ_j pre_f[j] alpha''[j][a]
        if lu < BASE {
            let (mut beta, mut beta0) = (0f32, 0f32);
            let mut j = 0;
            while j < HC {
                beta += *mix_a.add(j) * *alpha.add(j * BASE + lu);
                beta0 += *mix_f.add(j) * *alpha2.add(j * BASE + lu);
                j += 1;
            }
            *sc.add(SCR_BETA + lu) = beta;
            *sc.add(SCR_BETA + BASE + lu) = beta0;
        }
        if lane == 0 {
            let mut cc = 0f32;
            let mut j = 0;
            while j < HC {
                cc += *mix_f.add(j) * *mix_a.add(HC + j);
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
    pub extern "gpu-kernel" fn an_chain(tab: *const u64, streams: *const f32, fyv: *const f32, a_fn: *const u16, f_fn: *const u16, sq: *const u8, fg_w: *const u8, crows: *const u16, router: *const u16, x: *mut f32, smix: *mut f32, fg: *mut f32, normed: *mut f32, mem: *const f32, head0: u32, value: u32, n: u32, layer: u32) {
        unsafe {
            // the streamed operands are arguments (a kernel argument's loads are global; an address read from the
            // table is generic — flat loads, which count on the scalar counter too, 2026-10-25); the rest in `tab`
            const RR: usize = crate::hc::gfx11_consts::ROUTE_ROWS;
            let n = n as usize;
            let t = workitem_id_x() as usize;
            let (wave, lane) = (t / 32, (t % 32) as u32);
            let wg = workgroup_id_x() as usize;
            let l = lds::<f32>();
            let at = |k: usize| *tab.add(k) as usize;
            let mix_l = at(TAB_MIX_L) as *const f32;
            let (x0, c, scratch, count) = (at(TAB_X0) as *mut f32, at(TAB_C) as *mut f32, at(TAB_SCRATCH) as *mut f32, at(TAB_COUNT) as *mut u32);
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
            // phase C and the third barrier
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
                let w_row = if wg < HC_COEFS { a_fn.add(wg * HC * H) } else { f_fn.add((wg - HC_COEFS) * HC * H) };
                let mut i0 = 0;
                while i0 < n {
                    if n - i0 >= 2 {
                        phase_a::<2>(w_row, streams, fyv, scratch, wg, n, i0, t, lane, wave, l);
                    } else {
                        phase_a::<1>(w_row, streams, fyv, scratch, wg, n, i0, t, lane, wave, l);
                    }
                    i0 += 2;
                }
            }
            let kta = common::stat_clock!();
            common::stat!(0, kta.wrapping_sub(kt0) & common::stats::CLOCK_MASK);
            barrier(count, 0, t, wg, value);
            let ktb0 = common::stat_clock!();
            common::stat!(1, ktb0.wrapping_sub(kta) & common::stats::CLOCK_MASK);
            // phase B: workgroup 0, a wave a token
            if wg == 0 && wave < n {
                phase_b(wave, mix_l, at(TAB_A_BASE) as *const f32, at(TAB_A_SCALE) as *const f32, at(TAB_F_BASE) as *const f32, at(TAB_F_SCALE) as *const f32, c, scratch, lane, l.add(wave * B_LDS));
            }
            let ktb1 = common::stat_clock!();
            common::stat!(2, ktb1.wrapping_sub(ktb0) & common::stats::CLOCK_MASK);
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
            common::stat!(3, kt1.wrapping_sub(ktb1) & common::stats::CLOCK_MASK);
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
                let e4 = Exp4::<2, 8>::at::<H>(sq, M_SMALL);
                let mut vrow = (wg * CHAIN_WAVES + wave) * E4_ROWS;
                while vrow < SEL_ROWS {
                    match n {
                        1 => exp4_rows_direct_rows::<H, true, 2, 8, E4_ROWS, 1>(t as u32, e4, ln, SEL_ROWS, n, RMS_NORM_EPS, vrow, &map, &xs, &emit),
                        2 => exp4_rows_direct_rows::<H, true, 2, 8, E4_ROWS, 2>(t as u32, e4, ln, SEL_ROWS, n, RMS_NORM_EPS, vrow, &map, &xs, &emit),
                        3 => exp4_rows_direct_rows::<H, true, 2, 8, E4_ROWS, 3>(t as u32, e4, ln, SEL_ROWS, n, RMS_NORM_EPS, vrow, &map, &xs, &emit),
                        _ => exp4_rows_direct_rows::<H, true, 2, 8, E4_ROWS, { common::gemv::FEW }>(t as u32, e4, ln, SEL_ROWS, n, RMS_NORM_EPS, vrow, &map, &xs, &emit),
                    }
                    vrow += CHAIN_GRID as usize * CHAIN_WAVES * E4_ROWS;
                }
            }
            barrier(count, 3, t, wg, value);
            let _ = kt1;
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
                        1 => exp4_rows_direct_rows::<KDA_DIM, false, 2, 8, E4_ROWS, 1>(t as u32, e4, core::ptr::null(), SEL_FG_ROWS, n, 0.0, vrow, &map, &xs, &emit),
                        2 => exp4_rows_direct_rows::<KDA_DIM, false, 2, 8, E4_ROWS, 2>(t as u32, e4, core::ptr::null(), SEL_FG_ROWS, n, 0.0, vrow, &map, &xs, &emit),
                        3 => exp4_rows_direct_rows::<KDA_DIM, false, 2, 8, E4_ROWS, 3>(t as u32, e4, core::ptr::null(), SEL_FG_ROWS, n, 0.0, vrow, &map, &xs, &emit),
                        _ => exp4_rows_direct_rows::<KDA_DIM, false, 2, 8, E4_ROWS, { common::gemv::FEW }>(t as u32, e4, core::ptr::null(), SEL_FG_ROWS, n, 0.0, vrow, &map, &xs, &emit),
                    }
                    vrow += CHAIN_GRID as usize * CHAIN_WAVES * E4_ROWS;
                }
            }
            barrier(count, 4, t, wg, value);
            // phase S: the chosen heads' step from the state as it stands, nothing stored back
            if wg < NSEL {
                let (smix, fg) = (smix as *const f32, fg as *const f32);
                let hl = *heads.add(wg) as usize;
                crate::kda::gfx11::step_head::<false>(smix.add(SMALL_ROWS), at(TAB_TAPS) as *const u16, at(TAB_WINDOW) as *mut f32, fg, at(TAB_DT_BIAS) as *const f32, at(TAB_A_LOG) as *const f32, smix.add(2 * KDA_DIM), fg.add(PART_QKV), mem as *mut f32, at(TAB_O_NORM) as *const f32, normed, head0 as usize + hl, hl, n as u32, M_SMALL as u32, 2 * PART_QKV as u32, M_SMALL as u32, core::ptr::null_mut(), core::ptr::null_mut());
            }
            barrier(count, 5, t, wg, value);
            // phase O
            let normed = normed as *const f32;
            if wg < OUT_WGS {
                // a composed row a wave: the lane's eight columns of each 256-wide step gathered from the head the
                // step's columns belong to (a head is 128 columns: a step is two heads, a lane's eight within one)
                let part = at(TAB_PART) as *mut f32;
                let row = wg * 8 + wave;
                let wr = crows.add(row * SEL_W);
                let mut acc = [0f32; 4];
                let mut s = 0;
                while s < SEL_W / 256 {
                    let col = s * 256 + lane as usize * 8;
                    let wv = widen8_at(wr.add(col));
                    let src = *heads.add(col / D) as usize * D + col % D;
                    let mut i = 0;
                    while i < 4 {
                        if i < n {
                            let xv: [f32; 8] = core::ptr::read(normed.add(i * PART_QKV + src) as *const [f32; 8]);
                            let mut j = 0;
                            while j < 8 {
                                acc[i] += wv[j] * xv[j];
                                j += 1;
                            }
                        }
                        i += 1;
                    }
                    s += 1;
                }
                let mut i = 0;
                while i < 4 {
                    let y = wave_sum(lane, acc[i]);
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
                let (xl, xr) = (at(TAB_XL) as *const f32, x0 as *const f32);
                let mut acc = [0f32; 4];
                let mut ss = [0f32; 4];
                let mut s = 0;
                while s < H / 256 {
                    let col = s * 256 + lane as usize * 8;
                    let wv = widen8_at(wr.add(col));
                    let lv = widen8_at(ln.add(col));
                    let mut i = 0;
                    while i < 4 {
                        if i < n {
                            let mut xv: [f32; 8] = core::ptr::read((if second { xl } else { xr }).add(i * H + col) as *const [f32; 8]);
                            if second {
                                let fv: [f32; 8] = core::ptr::read(fyv.add(i * H + col) as *const [f32; 8]);
                                let mut j = 0;
                                while j < 8 {
                                    xv[j] += fv[j];
                                    j += 1;
                                }
                            }
                            let mut j = 0;
                            while j < 8 {
                                acc[i] += xv[j] * lv[j] * wv[j];
                                ss[i] += xv[j] * xv[j];
                                j += 1;
                            }
                        }
                        i += 1;
                    }
                    s += 1;
                }
                let out = at(if second { TAB_S2 } else { TAB_SCORES }) as *mut f32;
                let mut i = 0;
                while i < 4 {
                    let y = wave_sum(lane, acc[i]);
                    let rr = 1.0 / sqrt(wave_sum(lane, ss[i]) / H as f32 + RMS_NORM_EPS);
                    if lane == 0 && i < n {
                        *out.add(i * RR + row) = y * rr;
                        if !second && row == E {
                            *(at(TAB_K) as *mut f32).add(i) = *c.add(i) * rr;
                        }
                    }
                    i += 1;
                }
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
