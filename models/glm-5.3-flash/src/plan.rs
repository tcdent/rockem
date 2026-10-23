//! The decode prefetch's learned expert predictor on the cards (queue 082, tcdent's design; phase 3/4 of
//! docs/expert-cache-architecture.md): `predict.rs`'s model and update, run where its inputs already are — the FFN card
//! computes the next layer's route-ahead scores and this layer's picks — in the cards' idle time. The host only reads
//! the plan: a decode step's best `PLAN` experts for the next layer.
//!
//! `pred_step`, in the FFN card's queue right ahead of a MoE layer's output half (which spins until the host's experts
//! are in, ~180 µs at the median, so it runs in that wait): pair l predicts layer l + 1 from the route-ahead scores,
//! keeping each token's margins for its own update; a decode step also ranks the union (Π(1 + eᶻ), the order of
//! Σ softplus) and writes the plan and the host's word. `pred_learn`, once a step's layers are done, a workgroup a pair,
//! every pair at once: per-expert BCE, Adam, in batches of `SPEC_TOKENS` tokens as decode's steps. A pair predicts
//! again only in the next step, so its update at the step's end is as fresh as one a layer after its prediction; in the
//! layers' waits a decode step's update took 23 µs a layer, and a 512-token chunk's 10.6 ms (prefill 1.85 → 2.6 ms a
//! prompt token). A pair's state and margins live on its layer's FFN card.
//!
//! The pair's state, f32, `STATE` long: the calibration and Adam's terms (`HDR`), the next layer's router bias (`BN`),
//! the bias and its moments, the co-routing table `[E][E]` (row: an expert picked here) and its moments.
//!
//! Two margins, each with its own learned weight: `d` from the route-ahead (layer l + 1's router over layer l's FFN
//! input, `hc_pre`'s), `d2` from layer l + 1's router over the FFN input plus the shared expert's output (on the FFN card
//! before `hc_post`: `pred_input` and the router rows, then `pred_step`). The second margin, measured offline: settled
//! recall@13 .6205 → .6255 (bringup/expert-cache/phase3b, 2026-10-21).

use crate::{E, SPEC_TOKENS, TOP_K};
#[cfg(target_arch = "amdgpu")]
use crate::H;

/// Adam's learning rates: the co-routing table; the calibration (ca, cc) and the bias.
pub const LR_CO: f32 = 3e-3;
pub const LR_B: f32 = 1e-2;
/// The simulation's loss divided by the 40 layer pairs it trained together; kept so its learning rates carry over.
pub const PAIRS_SCALE: f32 = 40.0;
/// Experts a decode plan names, best first: the host drops the ones resident on the cards and keeps `PREFETCH_CAP`.
pub const PLAN: usize = 24;

/// The header: the calibration (ca, cc, ca2 on the second margin), their first moments, their second moments (four
/// slots each), β1ᵗ, β2ᵗ.
pub const HDR: usize = 0;
pub const M_CAL: usize = 4;
pub const V_CAL: usize = 8;
pub const B1T: usize = 12;
pub const B2T: usize = 13;
pub const HDR_LEN: usize = 16;
pub const BN: usize = HDR + HDR_LEN;
pub const BIAS: usize = BN + E;
pub const M_BIAS: usize = BIAS + E;
pub const V_BIAS: usize = M_BIAS + E;
pub const CO: usize = V_BIAS + E;
pub const M_CO: usize = CO + E * E;
pub const V_CO: usize = M_CO + E * E;
pub const STATE: usize = V_CO + E * E;

/// A pair's state before any update: ca 3, β1⁰ = β2⁰ = 1, the next layer's router bias, every correction zero.
#[cfg(feature = "std")]
pub fn init(bias_next: &[f32]) -> Vec<f32> {
    let mut s = vec![0f32; STATE];
    s[HDR] = 3.0;
    s[HDR + B1T] = 1.0;
    s[HDR + B2T] = 1.0;
    s[BN..BN + E].copy_from_slice(&bias_next[..E]);
    s
}

/// `pred_learn`'s LDS, in words: the batch's dL/dz and labels `[SPEC_TOKENS][E]` each, its picks, its distinct rows
/// and their token masks, the header, the reductions.
pub const LEARN_LDS: u32 = (4 * (2 * SPEC_TOKENS * E + 3 * SPEC_TOKENS * TOP_K + HDR_LEN + 16)) as u32;
/// `pred_learn`'s workgroup: a thread an expert.
pub const LEARN_WG: u32 = E as u32;
/// `pred_learn`'s table: device addresses a pair.
pub const LEARN_TAB: usize = 5;
/// `pred_step`'s: the reductions, the waves' best 8, the plan's keys `[E]`.
pub const PLAN_LDS: u32 = (4 * (16 + E / 32 * TOP_K + E)) as u32;
/// `pred_step`'s workgroup: a thread an expert.
pub const PLAN_WG: u32 = E as u32;
/// Workgroups that predict a long step's tokens, a token a workgroup at a time.
pub const PLAN_PRED_WGS: usize = 64;
/// `pred_rows`' grid: 8 of the next layer's router rows a workgroup, then a token a workgroup (093).
pub const PRED_ROWS_WGS: u32 = (E / 8) as u32;
const _: () = assert!(E % 8 == 0 && PRED_ROWS_WGS as usize >= SPEC_TOKENS);

#[cfg(target_arch = "amdgpu")]
pub mod gfx11 {
    use super::*;
    use crate::RMS_NORM_EPS;
    use common::gemv::widen;
    use common::gfx11::{ballot, lds, wave_max, wave_sum, workgroup_sync};
    use common::math::{exp, sqrt};
    use core::arch::amdgpu::{workgroup_id_x, workitem_id_x};
    use core::sync::atomic::{fence, AtomicU32, Ordering};

    #[inline(always)]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + exp(-x))
    }

    /// torch.optim.Adam's step on one parameter: p −= lr / (1 − β1ᵗ) · m / (√v / √(1 − β2ᵗ) + ε).
    #[inline(always)]
    unsafe fn adam(p: *mut f32, m: *mut f32, v: *mut f32, g: f32, lr_c1: f32, c2: f32) {
        let mm = 0.9 * *m + 0.1 * g;
        let vv = 0.999 * *v + 0.001 * g * g;
        *m = mm;
        *v = vv;
        *p -= lr_c1 * mm / (sqrt(vv) / c2 + 1e-8);
    }

    /// `out = a + b + Σₖ wts[i][k] · rows[i][k]`, `n` floats over `[tokens][H]`: the second margin's router input — the
    /// FFN input plus the whole FFN output (the shared expert's and the routed experts' rows in pick order, the host's
    /// in the leader's VRAM once its word is in; `rows` null: the shared output alone). Offline, the routed rows as the
    /// margin's input settled recall@13 .6205 → .647 (phase3b, 2026-10-21); run after `hc_post`, they exist (093).
    #[no_mangle]
    pub extern "gpu-kernel" fn pred_input(out: *mut f32, a: *const f32, b: *const f32, rows: *const f32, wts: *const f32, n: u32) {
        unsafe {
            let i = workgroup_id_x() as usize * 256 + workitem_id_x() as usize;
            if i < n as usize {
                let mut s = *a.add(i) + *b.add(i);
                if !rows.is_null() {
                    let (t, d) = (i / H, i % H);
                    for k in 0..TOP_K {
                        s += *wts.add(t * TOP_K + k) * *rows.add((t * TOP_K + k) * H + d);
                    }
                }
                *out.add(i) = s;
            }
        }
    }

    /// The 8th best of the workgroup's `E` values, one a thread (with multiplicity: the value with at most 7 above it
    /// and at least 8 at or above it): each wave's best 8 by eight wave maxima, then the best 8 of those 72 in wave 0.
    /// Every thread gets it. Three barriers. (Each thread comparing its value with all 288 took most of `pred_step`'s
    /// 66 µs a layer, 2026-10-21.)
    #[inline(always)]
    unsafe fn eighth(v: f32, lane: u32, wave: usize, cand: *mut f32, red: *mut f32) -> f32 {
        const NONE: f32 = f32::NEG_INFINITY;
        let mut cur = v;
        let mut j = 0;
        while j < TOP_K {
            let m = wave_max(lane, cur);
            if lane == ballot(cur == m).trailing_zeros() {
                cur = NONE;
            }
            if lane == 0 {
                *cand.add(wave * TOP_K + j) = m;
            }
            j += 1;
        }
        workgroup_sync();
        if wave == 0 {
            const C: usize = E / 32 * TOP_K;
            let (mut a, mut b) = (*cand.add(lane as usize), *cand.add(lane as usize + 32));
            let mut c = if (lane as usize) + 64 < C { *cand.add(lane as usize + 64) } else { NONE };
            let mut m = NONE;
            let mut j = 0;
            while j < TOP_K {
                let loc = if a >= b && a >= c { a } else if b >= c { b } else { c };
                m = wave_max(lane, loc);
                if lane == ballot(loc == m).trailing_zeros() {
                    if a == m {
                        a = NONE;
                    } else if b == m {
                        b = NONE;
                    } else {
                        c = NONE;
                    }
                }
                j += 1;
            }
            if lane == 0 {
                *red = m;
            }
        }
        workgroup_sync();
        let thr = *red;
        workgroup_sync();
        thr
    }

    /// Pair l's prediction (see the module): its state, the route-ahead scores (`[n][ROUTE_ROWS]`, the next layer's
    /// from `E`), the second margin's (likewise), the margins it keeps `[n][E]` each, the picks at l `[n][TOP_K]`
    /// u32. `plan` (null in a long step): the decode plan and the host's `word`. Workgroups of `PLAN_WG`, a thread an
    /// expert; grid 1 for a decode step, `PLAN_PRED_WGS` for a long one.
    #[no_mangle]
    pub extern "gpu-kernel" fn pred_step(pred: *const f32, scores: *const f32, scores2: *const f32, p_d: *mut f32, p_d2: *mut f32, p_picks: *const u32, plan: *mut u32, word: *mut u32, value: u32, n: u32, keys: *mut f32, count: *mut u32) {
        unsafe {
            const RR: usize = crate::hc::gfx11_consts::ROUTE_ROWS;
            let e = workitem_id_x() as usize;
            let (lane, wave) = ((e % 32) as u32, e / 32);
            let wg = workgroup_id_x() as usize;
            let n = n as usize;
            let decode = n <= SPEC_TOKENS;
            let l = lds::<f32>();
            let red = l; // [16]
            let cand = red.add(16); // [E / 32 · TOP_K]
            let sk = cand.add(E / 32 * TOP_K); // [E]
            // the tokens this workgroup predicts: a decode step's one a workgroup (the workgroups' keys multiplied by the
            // last to finish — one workgroup over four tokens was 35 µs on the card's path, 2026-10-22), a long step's striped
            let (first, stride) = if decode { (wg, n) } else { (wg, PLAN_PRED_WGS) };
            let (ca, cc, ca2) = (*pred.add(HDR), *pred.add(HDR + 1), *pred.add(HDR + 2));
            let (bn, bias) = (*pred.add(BN + e), *pred.add(BIAS + e));
            // the plan's key of this thread's expert
            let mut key = 1f32;
            let mut i = first;
            while i < n {
                let g1 = sigmoid(*scores.add(i * RR + E + e)) + bn;
                let t1 = eighth(g1, lane, wave, cand, red);
                let g2 = sigmoid(*scores2.add(i * RR + E + e)) + bn;
                let t2 = eighth(g2, lane, wave, cand, red);
                let (d, d2) = ((g1 - t1) * 10.0, (g2 - t2) * 10.0);
                *p_d.add(i * E + e) = d;
                *p_d2.add(i * E + e) = d2;
                if decode {
                    let mut z = ca * d + ca2 * d2 + cc + bias;
                    let mut q = 0;
                    while q < TOP_K {
                        z += *pred.add(CO + *p_picks.add(i * TOP_K + q) as usize * E + e);
                        q += 1;
                    }
                    key *= 1.0 + exp(if z < 20.0 { z } else { 20.0 });
                }
                i += stride;
            }
            if !decode || plan.is_null() {
                return;
            }
            // this token's key term out; the last workgroup to arrive multiplies the step's terms and ranks
            *keys.add(wg * E + e) = key;
            fence(Ordering::Release);
            workgroup_sync();
            if e == 0 {
                let last = AtomicU32::from_ptr(count).fetch_add(1, Ordering::AcqRel) == n as u32 - 1;
                *red.add(15) = if last { 1.0 } else { 0.0 };
                if last {
                    AtomicU32::from_ptr(count).store(0, Ordering::Relaxed);
                }
            }
            workgroup_sync();
            if *red.add(15) == 0.0 {
                return;
            }
            fence(Ordering::Acquire);
            let mut key = 1f32;
            let mut w = 0;
            while w < n {
                key *= *keys.add(w * E + e);
                w += 1;
            }
            // the plan: each expert's place among the keys (the greater first, between equals the lower expert)
            *sk.add(e) = key;
            workgroup_sync();
            let mut r = 0usize;
            let mut j = 0;
            while j < E {
                let w = *sk.add(j);
                r += (w > key || (w == key && j < e)) as usize;
                j += 1;
            }
            if r < PLAN {
                *plan.add(r) = e as u32;
            }
            fence(Ordering::Release);
            workgroup_sync();
            if e == 0 && !word.is_null() {
                AtomicU32::from_ptr(word).store(value, Ordering::Release);
            }
        }
    }

    /// The rows pass as one launch (093): the next layer's router over `x` (the FFN input plus the whole FFN output,
    /// `hc_post`'s), the margins, the plan. Grid `PRED_ROWS_WGS` of `PLAN_WG`, all resident: phase 1, workgroup w
    /// takes router rows `E + 8w..` over the `n` tokens into `scores2` (`[n][ROUTE_ROWS]`); a grid barrier on
    /// `count[0]`; phase 2, workgroup i < n takes token i — the two margins against `scores` (`hc_pre`'s route-ahead)
    /// and `scores2`, stored for the learning, its key into `keys` — and the last to arrive at `count[1]` multiplies the
    /// keys and ranks into `plan`, then `word`. Three launches (27 + 33 µs and a 10 µs add) were ~60 µs on the card's
    /// critical chain behind `hc_post`; the chain is the point, so the margins run a token a workgroup.
    #[no_mangle]
    pub extern "gpu-kernel" fn pred_rows(x: *const f32, ln: *const u16, router: *const u16, scores2: *mut f32, pred: *const f32, scores: *const f32, p_d: *mut f32, p_d2: *mut f32, p_picks: *const u32, plan: *mut u32, word: *mut u32, value: u32, n: u32, keys: *mut f32, count: *mut u32) {
        unsafe {
            const RR: usize = crate::hc::gfx11_consts::ROUTE_ROWS;
            let e = workitem_id_x() as usize;
            let (lane, wave) = ((e % 32) as u32, e / 32);
            let wg = workgroup_id_x() as usize;
            let n = n as usize;
            let l = lds::<f32>();
            let red = l; // [16]
            let cand = red.add(16); // [E / 32 · TOP_K]
            let sk = cand.add(E / 32 * TOP_K); // [E]
            // phase 1: this workgroup's 8 rows of the next layer's router, over every token's normed input (the first 8
            // waves; the ninth idles)
            let mut i = 0;
            while i < n {
                let xt = x.add(i * H);
                let mut ss = 0f32;
                if e < 256 {
                    let mut k = e;
                    while k < H {
                        let v = *xt.add(k);
                        ss += v * v;
                        k += 256;
                    }
                }
                ss = wave_sum(lane, ss);
                if lane == 0 && wave < 8 {
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
                workgroup_sync();
                let mut r = 0;
                while r < 8 {
                    let row = E + wg * 8 + r;
                    let wr = router.add(row * H);
                    let mut acc = 0f32;
                    if e < 256 {
                        let mut k = e;
                        while k < H {
                            acc += *xt.add(k) * widen(*ln.add(k)) * widen(*wr.add(k));
                            k += 256;
                        }
                    }
                    acc = wave_sum(lane, acc);
                    if lane == 0 && wave < 8 {
                        *red.add(wave) = acc;
                    }
                    workgroup_sync();
                    if e == 0 {
                        let mut sum = 0f32;
                        let mut k = 0;
                        while k < 8 {
                            sum += *red.add(k);
                            k += 1;
                        }
                        *scores2.add(i * RR + row) = sum * rr;
                    }
                    workgroup_sync();
                    r += 1;
                }
                i += 1;
            }
            // the grid barrier: every workgroup's rows are in before any token's margins read them
            fence(Ordering::Release);
            workgroup_sync();
            if e == 0 {
                AtomicU32::from_ptr(count).fetch_add(1, Ordering::AcqRel);
                while AtomicU32::from_ptr(count).load(Ordering::Acquire) < PRED_ROWS_WGS {
                    core::hint::spin_loop();
                }
            }
            workgroup_sync();
            fence(Ordering::Acquire);
            if wg >= n {
                return;
            }
            // phase 2: this workgroup's token
            let i = wg;
            let (ca, cc, ca2) = (*pred.add(HDR), *pred.add(HDR + 1), *pred.add(HDR + 2));
            let (bn, bias) = (*pred.add(BN + e), *pred.add(BIAS + e));
            let g1 = sigmoid(*scores.add(i * RR + E + e)) + bn;
            let t1 = eighth(g1, lane, wave, cand, red);
            let g2 = sigmoid(*scores2.add(i * RR + E + e)) + bn;
            let t2 = eighth(g2, lane, wave, cand, red);
            let (d, d2) = ((g1 - t1) * 10.0, (g2 - t2) * 10.0);
            *p_d.add(i * E + e) = d;
            *p_d2.add(i * E + e) = d2;
            let mut z = ca * d + ca2 * d2 + cc + bias;
            let mut q = 0;
            while q < TOP_K {
                z += *pred.add(CO + *p_picks.add(i * TOP_K + q) as usize * E + e);
                q += 1;
            }
            *keys.add(i * E + e) = 1.0 + exp(if z < 20.0 { z } else { 20.0 });
            fence(Ordering::Release);
            workgroup_sync();
            if e == 0 {
                let last = AtomicU32::from_ptr(count.add(1)).fetch_add(1, Ordering::AcqRel) == n as u32 - 1;
                *red.add(15) = if last { 1.0 } else { 0.0 };
                if last {
                    AtomicU32::from_ptr(count.add(1)).store(0, Ordering::Relaxed);
                    AtomicU32::from_ptr(count).store(0, Ordering::Relaxed);
                }
            }
            workgroup_sync();
            if *red.add(15) == 0.0 {
                return;
            }
            fence(Ordering::Acquire);
            let mut key = 1f32;
            let mut w = 0;
            while w < n {
                key *= *keys.add(w * E + e);
                w += 1;
            }
            // the plan: each expert's place among the keys (the greater first, between equals the lower expert)
            *sk.add(e) = key;
            workgroup_sync();
            let mut r = 0usize;
            let mut j = 0;
            while j < E {
                let w = *sk.add(j);
                r += (w > key || (w == key && j < e)) as usize;
                j += 1;
            }
            if r < PLAN {
                *plan.add(r) = e as u32;
            }
            fence(Ordering::Release);
            workgroup_sync();
            if e == 0 {
                AtomicU32::from_ptr(word).store(value, Ordering::Release);
            }
        }
    }

    /// Pairs' updates over the `n` tokens from row `t0` (see the module), a workgroup a pair: workgroup w takes entry
    /// `first + w` of `tab` (`LEARN_TAB` device addresses a pair: its state, its margins `[STEP][E]` kept by its
    /// prediction, the picks at its layer and at the next, its labels, each `[STEP][TOP_K]` u32). Workgroups of
    /// `LEARN_WG`, a thread an expert.
    #[no_mangle]
    pub extern "gpu-kernel" fn pred_learn(tab: *const u64, first: u32, t0: u32, n: u32) {
        unsafe {
            const R: usize = SPEC_TOKENS * TOP_K;
            const NONE: u32 = u32::MAX;
            let e = workitem_id_x() as usize;
            let n = n as usize;
            let row = tab.add((first as usize + workgroup_id_x() as usize) * LEARN_TAB);
            let t0 = t0 as usize;
            let s = *row as *mut f32;
            let d = (*row.add(1) as *const f32).add(t0 * E);
            let picks = (*row.add(2) as *const u32).add(t0 * TOP_K);
            let labels = (*row.add(3) as *const u32).add(t0 * TOP_K);
            let d2 = (*row.add(4) as *const f32).add(t0 * E);
            let l = lds::<f32>();
            let gz = l; // [SPEC_TOKENS][E]
            let y = gz.add(SPEC_TOKENS * E) as *mut u32; // [SPEC_TOKENS][E]
            let pk = y.add(SPEC_TOKENS * E); // [R] the batch's picks at l − 1
            let rq = pk.add(R); // [R] a distinct row of the co-routing table, or NONE
            let rm = rq.add(R); // [R] its tokens, a bit each
            let cal = rm.add(R) as *mut f32; // [HDR_LEN]
            let red = cal.add(HDR_LEN); // [16]
            if e < HDR_LEN {
                *cal.add(e) = *s.add(HDR + e);
            }
            let mut a = 0;
            while a < n {
                let m = if n - a < SPEC_TOKENS { n - a } else { SPEC_TOKENS };
                let mut i = 0;
                while i < m {
                    *y.add(i * E + e) = 0;
                    i += 1;
                }
                workgroup_sync();
                if e < m * TOP_K {
                    let i = e / TOP_K;
                    *y.add(i * E + *labels.add((a + i) * TOP_K + e % TOP_K) as usize) = 1;
                    *pk.add(e) = *picks.add((a + i) * TOP_K + e % TOP_K);
                }
                workgroup_sync();
                // the batch's distinct rows (each pick's first place) and the tokens that picked each
                if e < m * TOP_K {
                    let q = *pk.add(e);
                    let mut first = true;
                    let mut j = 0;
                    while j < e {
                        first &= *pk.add(j) != q;
                        j += 1;
                    }
                    let mut mask = 0u32;
                    let mut j = 0;
                    while j < m * TOP_K {
                        mask |= ((*pk.add(j) == q) as u32) << (j / TOP_K);
                        j += 1;
                    }
                    *rq.add(e) = if first { q } else { NONE };
                    *rm.add(e) = mask;
                }
                workgroup_sync();
                let (ca, cc, ca2) = (*cal, *cal.add(1), *cal.add(2));
                let sc = 1.0 / (m as f32 * PAIRS_SCALE);
                let (mut gca, mut gcc, mut gca2, mut gb) = (0f32, 0f32, 0f32, 0f32);
                let mut i = 0;
                while i < m {
                    let dv = *d.add((a + i) * E + e);
                    let dv2 = *d2.add((a + i) * E + e);
                    let mut z = ca * dv + ca2 * dv2 + cc + *s.add(BIAS + e);
                    let mut q = 0;
                    while q < TOP_K {
                        z += *s.add(CO + *pk.add(i * TOP_K + q) as usize * E + e);
                        q += 1;
                    }
                    let g = (sigmoid(z) - *y.add(i * E + e) as f32) * sc;
                    *gz.add(i * E + e) = g;
                    gb += g;
                    gca += g * dv;
                    gca2 += g * dv2;
                    gcc += g;
                    i += 1;
                }
                let gca = learn_sum(e, gca, red);
                let gcc = learn_sum(e, gcc, red);
                let gca2 = learn_sum(e, gca2, red);
                let (b1t, b2t) = (*cal.add(B1T) * 0.9, *cal.add(B2T) * 0.999);
                let (c1, c2) = (1.0 - b1t, sqrt(1.0 - b2t));
                adam(s.add(BIAS + e), s.add(M_BIAS + e), s.add(V_BIAS + e), gb, LR_B / c1, c2);
                // the rows, four at a time: their loads issued together
                let lr = LR_CO / c1;
                let mut j = 0;
                while j < m * TOP_K {
                    let mut p = [0f32; 4];
                    let mut mo = [0f32; 4];
                    let mut vo = [0f32; 4];
                    let mut at = [0usize; 4];
                    let mut k = 0;
                    while k < 4 {
                        let q = if j + k < m * TOP_K { *rq.add(j + k) } else { NONE };
                        at[k] = if q == NONE { usize::MAX } else { q as usize * E + e };
                        if q != NONE {
                            p[k] = *s.add(CO + at[k]);
                            mo[k] = *s.add(M_CO + at[k]);
                            vo[k] = *s.add(V_CO + at[k]);
                        }
                        k += 1;
                    }
                    let mut k = 0;
                    while k < 4 {
                        if at[k] != usize::MAX {
                            let mask = *rm.add(j + k);
                            let mut g = 0f32;
                            let mut i = 0;
                            while i < m {
                                if mask >> i & 1 != 0 {
                                    g += *gz.add(i * E + e);
                                }
                                i += 1;
                            }
                            let mm = 0.9 * mo[k] + 0.1 * g;
                            let vv = 0.999 * vo[k] + 0.001 * g * g;
                            *s.add(M_CO + at[k]) = mm;
                            *s.add(V_CO + at[k]) = vv;
                            *s.add(CO + at[k]) = p[k] - lr * mm / (sqrt(vv) / c2 + 1e-8);
                        }
                        k += 1;
                    }
                    j += 4;
                }
                workgroup_sync();
                if e == 0 {
                    adam(cal, cal.add(M_CAL), cal.add(V_CAL), gca, LR_B / c1, c2);
                    adam(cal.add(1), cal.add(M_CAL + 1), cal.add(V_CAL + 1), gcc, LR_B / c1, c2);
                    adam(cal.add(2), cal.add(M_CAL + 2), cal.add(V_CAL + 2), gca2, LR_B / c1, c2);
                    *cal.add(B1T) = b1t;
                    *cal.add(B2T) = b2t;
                }
                workgroup_sync();
                a += m;
            }
            if e < HDR_LEN {
                *s.add(HDR + e) = *cal.add(e);
            }
        }
    }

    /// Sum of one f32 a thread over `pred_learn`'s `LEARN_WG` threads (nine waves), every thread getting it. Two barriers.
    #[inline(always)]
    unsafe fn learn_sum(t: usize, v: f32, red: *mut f32) -> f32 {
        const WAVES: usize = (LEARN_WG as usize).div_ceil(32);
        let (lane, wave) = ((t % 32) as u32, t / 32);
        let w = common::gfx11::wave_sum(lane, v);
        if lane == 0 {
            *red.add(wave) = w;
        }
        workgroup_sync();
        let mut total = 0f32;
        let mut k = 0;
        while k < WAVES {
            total += *red.add(k);
            k += 1;
        }
        workgroup_sync();
        total
    }
}
