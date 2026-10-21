//! The decode prefetch's learned expert predictor on the cards (queue 082, tcdent's design; phase 3/4 of
//! docs/expert-cache-architecture.md): `predict.rs`'s model and update, run where its inputs already are — the FFN card
//! computes the next layer's route-ahead scores and this layer's picks — in the cards' idle time. The host only reads
//! the plan: a decode step's best `PLAN` experts for the next layer.
//!
//! `pred_step`, in the FFN card's queue right ahead of a MoE layer's output half (which spins until the host's experts
//! are in, so it runs in that wait): pair l predicts layer l + 1 from the route-ahead scores, keeping each token's margin
//! `d` for its own update; a decode step also ranks the union (Π(1 + eᶻ), the order of Σ softplus) and writes the plan
//! and the host's word. `pred_learn`: per-expert BCE, Adam, in batches of `SPEC_TOKENS` tokens as decode's steps. A
//! decode step's pair l − 1 learns from layer l's picks right after `pred_step`, in the same wait (the two 56 + 65 µs of
//! a ~550 µs wait, 2026-10-21). A prefill step's pairs learn once its layers are done, a workgroup a pair, all at once:
//! a 512-token chunk's learning a layer took 10.6 ms, more than the chunk's wait (the other chunk's card half fills
//! it): prefill 1.85 → 2.6 ms a prompt token. A pair's state and margins live on its layer's FFN card.
//!
//! The pair's state, f32, `STATE` long: the calibration and Adam's terms (`HDR`), the next layer's router bias (`BN`),
//! the bias and its moments, the co-routing table `[E][E]` (row: an expert picked here) and its moments.
//!
//! Two margins, each with its own learned weight: `d` from the route-ahead (layer l + 1's router over layer l's FFN
//! input, `hc_pre`'s), `d2` from layer l + 1's router over the FFN input plus the shared expert's output (on the FFN card
//! before `hc_post`: `pred_input` and the router rows, then `pred_step`). The second margin, measured offline: settled
//! recall@13 .6205 → .6255 (bringup/expert-cache/phase3b, 2026-10-21).

use crate::{E, H, SPEC_TOKENS, TOP_K};

/// Adam's learning rates: the co-routing table; the calibration (ca, cc) and the bias.
pub const LR_CO: f32 = 3e-3;
pub const LR_B: f32 = 1e-2;
/// The input map's (U, V).
pub const LR_X: f32 = 3e-4;
/// The input map's rank: z += (x̂ · U) · V, x̂ the FFN input RMS-normed (no weight), U `[H][MAP_RANK]`, V `[MAP_RANK][E]`.
pub const MAP_RANK: usize = 64;
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
pub const MAP_U: usize = V_CO + E * E;
pub const M_MAP_U: usize = MAP_U + H * MAP_RANK;
pub const V_MAP_U: usize = M_MAP_U + H * MAP_RANK;
pub const MAP_V: usize = V_MAP_U + H * MAP_RANK;
pub const M_MAP_V: usize = MAP_V + MAP_RANK * E;
pub const V_MAP_V: usize = M_MAP_V + MAP_RANK * E;
pub const STATE: usize = V_MAP_V + MAP_RANK * E;
/// A pair's input-map store, f32: a decode step's normed inputs `[SPEC_TOKENS][H]` and the gradients of their h
/// `[SPEC_TOKENS][MAP_RANK]` (for `pred_map`), then a step's h `[STEP][MAP_RANK]` (for its update).
pub const MS_X: usize = 0;
pub const MS_DH: usize = MS_X + SPEC_TOKENS * H;
pub const MS_RN: usize = MS_DH + SPEC_TOKENS * MAP_RANK;
pub const MS_H: usize = MS_RN + SPEC_TOKENS;
pub const MS_LEN: usize = MS_H + crate::STEP * MAP_RANK;
/// `pred_h`'s workgroups, a slice of the input each, and its partials a card: `[MAP_HWGS][STEP][MAP_RANK + 1]` f32
/// (each slice's x · U, unnormed, and its sum of squares). One workgroup reading U whole took ~490 µs a layer at 4 tokens.
pub const MAP_HWGS: usize = 64;
pub const MAP_SLICE: usize = H / MAP_HWGS;
pub const MAP_PART: usize = MAP_HWGS * crate::STEP * (MAP_RANK + 1);
/// `pred_map`'s workgroups: U's update spread over the card.
pub const MAP_WGS: usize = 64;

/// A pair's state before any update: ca 3, β1⁰ = β2⁰ = 1, the next layer's router bias, U small and random (normal /
/// √H, from `seed`), every correction zero (V zero: the map starts at zero).
#[cfg(feature = "std")]
pub fn init(bias_next: &[f32], seed: u64) -> Vec<f32> {
    let mut s = vec![0f32; STATE];
    s[HDR] = 3.0;
    s[HDR + B1T] = 1.0;
    s[HDR + B2T] = 1.0;
    s[BN..BN + E].copy_from_slice(&bias_next[..E]);
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut u = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    for v in &mut s[MAP_U..MAP_U + H * MAP_RANK] {
        let (a, b) = (u(), u());
        *v = ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32 / (H as f32).sqrt();
    }
    s
}

/// `pred_learn`'s LDS, in words: the batch's dL/dz and labels `[SPEC_TOKENS][E]` each, its picks, its distinct rows
/// and their token masks, the header, the reductions.
pub const LEARN_LDS: u32 = (4 * (2 * SPEC_TOKENS * E + 3 * SPEC_TOKENS * TOP_K + HDR_LEN + 16)) as u32;
/// `pred_learn`'s workgroup: a thread an expert.
pub const LEARN_WG: u32 = E as u32;
/// The map store keeps a decode step's inputs raw (`MS_X`) with their norms' reciprocals (`MS_RN`).
/// `pred_learn`'s table: device addresses a pair (state, margins, picks here, labels, second margins, map store).
pub const LEARN_TAB: usize = 6;
/// `pred_step`'s: the reductions, a token's scores and the plan's keys `[E]` each, the normed input `[H]`, the map's
/// partials `[256]` and its h `[MAP_RANK]`.
pub const PLAN_LDS: u32 = (4 * (16 + 2 * E + MAP_RANK)) as u32;
/// `pred_h`'s: a token's slice partials `[256]`.
pub const MAP_H_LDS: u32 = 4 * 256;
/// Workgroups that predict a long step's tokens, a token a workgroup at a time.
pub const PLAN_PRED_WGS: usize = 64;

#[cfg(target_arch = "amdgpu")]
pub mod gfx11 {
    use super::*;
    use common::gfx11::{lds, workgroup_sync};
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

    /// `out = a + b`, `n` floats: the second margin's router input (the FFN input plus the shared expert's output).
    #[no_mangle]
    pub extern "gpu-kernel" fn pred_input(out: *mut f32, a: *const f32, b: *const f32, n: u32) {
        unsafe {
            let i = workgroup_id_x() as usize * 256 + workitem_id_x() as usize;
            if i < n as usize {
                *out.add(i) = *a.add(i) + *b.add(i);
            }
        }
    }

    /// A token's margins from one set of layer l + 1's router logits (token `i`'s E at `src`): σ + the router's bias,
    /// less the 8th best of those, × 10 — this thread's experts t and t + 256 into `m0`, `m1`.
    #[inline(always)]
    unsafe fn margins(t: usize, src: *const f32, bn: *const f32, gv: *mut f32, red: *mut f32) -> (f32, f32) {
        let mut k = 0;
        while k < 2 {
            let e = t + 256 * k;
            if e < E {
                *gv.add(e) = sigmoid(*src.add(e)) + *bn.add(e);
            }
            k += 1;
        }
        workgroup_sync();
        // the 8th best: the value with at most 7 above it and at least 8 at or above it
        let mut k = 0;
        while k < 2 {
            let e = t + 256 * k;
            if e < E {
                let v = *gv.add(e);
                let (mut above, mut at) = (0u32, 0u32);
                let mut j = 0;
                while j < E {
                    let w = *gv.add(j);
                    above += (w > v) as u32;
                    at += (w >= v) as u32;
                    j += 1;
                }
                if above < TOP_K as u32 && at >= TOP_K as u32 {
                    *red = v;
                }
            }
            k += 1;
        }
        workgroup_sync();
        let thr = *red;
        let m0 = (*gv.add(t) - thr) * 10.0;
        let m1 = if t + 256 < E { (*gv.add(t + 256) - thr) * 10.0 } else { 0.0 };
        workgroup_sync();
        (m0, m1)
    }

    /// The input map's x · U over the `n` tokens' FFN inputs (`xin` `[n][H]`), a slice of `MAP_SLICE` inputs a workgroup
    /// (its U rows read once a token, from cache after the first): each slice's partial h and sum of squares into `part`
    /// (rows from `t0`), a decode step's raw inputs into the map store. Grid `MAP_HWGS` of 256: a quarter of the slice
    /// and a rank a thread.
    #[no_mangle]
    pub extern "gpu-kernel" fn pred_h(xin: *const f32, pred: *const f32, ms: *mut f32, part: *mut f32, t0: u32, n: u32) {
        unsafe {
            const Q: usize = 256 / MAP_RANK;
            const _: () = assert!(256 % MAP_RANK == 0 && MAP_SLICE % Q == 0 && MAP_SLICE <= 256 && H % MAP_HWGS == 0);
            let t = workitem_id_x() as usize;
            let w = workgroup_id_x() as usize;
            let (n, t0) = (n as usize, t0 as usize);
            let lp = lds::<f32>();
            let (qq, rr) = (t / MAP_RANK, t % MAP_RANK);
            let k0 = w * MAP_SLICE;
            let mut i = 0;
            while i < n {
                let x = xin.add(i * H + k0);
                let mut acc = 0f32;
                let mut k = qq * (MAP_SLICE / Q);
                while k < (qq + 1) * (MAP_SLICE / Q) {
                    acc += *x.add(k) * *pred.add(MAP_U + (k0 + k) * MAP_RANK + rr);
                    k += 1;
                }
                *lp.add(t) = acc;
                workgroup_sync();
                let p = part.add((w * crate::STEP + t0 + i) * (MAP_RANK + 1));
                if t < MAP_RANK {
                    let mut h = 0f32;
                    let mut j = 0;
                    while j < Q {
                        h += *lp.add(j * MAP_RANK + t);
                        j += 1;
                    }
                    *p.add(t) = h;
                }
                workgroup_sync();
                if t < MAP_SLICE {
                    let v = *x.add(t);
                    if n <= SPEC_TOKENS {
                        *ms.add(MS_X + i * H + k0 + t) = v;
                    }
                    *lp.add(t) = v * v;
                }
                workgroup_sync();
                if t == 0 {
                    let mut ss = 0f32;
                    let mut j = 0;
                    while j < MAP_SLICE {
                        ss += *lp.add(j);
                        j += 1;
                    }
                    *p.add(MAP_RANK) = ss;
                }
                workgroup_sync();
                i += 1;
            }
        }
    }

    /// Pair l's prediction (see the module): its state, the route-ahead scores (`[n][ROUTE_ROWS]`, the next layer's
    /// from `E`), the second margin's (likewise), the margins it keeps `[n][E]` each, the picks at l `[n][TOP_K]`
    /// u32. `plan` (null in a long step): the decode plan and the host's `word`. Grid: 1 for a decode step,
    /// `PLAN_PRED_WGS` for a long one.
    #[no_mangle]
    pub extern "gpu-kernel" fn pred_step(pred: *const f32, scores: *const f32, scores2: *const f32, part: *const f32, ms: *mut f32, t0: u32, p_d: *mut f32, p_d2: *mut f32, p_picks: *const u32, plan: *mut u32, word: *mut u32, value: u32, n: u32) {
        unsafe {
            let t = workitem_id_x() as usize;
            let wg = workgroup_id_x() as usize;
            let n = n as usize;
            let decode = n <= SPEC_TOKENS;
            let l = lds::<f32>();
            let red = l; // [16]
            let gv = red.add(16); // [E]
            let sk = gv.add(E); // [E]
            let hv = sk.add(E); // [MAP_RANK]
            let t0 = t0 as usize;
            // the tokens this workgroup predicts: a decode step's all in one, a long step's striped
            let (first, stride) = if decode { (0, 1) } else { (wg, PLAN_PRED_WGS) };
            let (ca, cc, ca2) = (*pred.add(HDR), *pred.add(HDR + 1), *pred.add(HDR + 2));
            // the plan's key of this thread's experts t and t + 256 (scalars: an array across the barriers is scratch)
            let (mut s0, mut s1) = (1f32, 1f32);
            let mut i = first;
            while i < n {
                // the input map's h from `pred_h`'s partials: Σ over the slices, then the norm (h is linear in x)
                if t < MAP_RANK {
                    let (mut h, mut ss) = (0f32, 0f32);
                    let mut w = 0;
                    while w < MAP_HWGS {
                        let p = part.add((w * crate::STEP + t0 + i) * (MAP_RANK + 1));
                        h += *p.add(t);
                        ss += *p.add(MAP_RANK);
                        w += 1;
                    }
                    let rn = 1.0 / (sqrt(ss / H as f32) + 1e-6);
                    *hv.add(t) = h * rn;
                    *ms.add(MS_H + (t0 + i) * MAP_RANK + t) = h * rn;
                    if decode && t == 0 {
                        *ms.add(MS_RN + i) = rn;
                    }
                }
                workgroup_sync();
                let (a0, a1) = margins(t, scores.add(i * crate::hc::gfx11_consts::ROUTE_ROWS + E), pred.add(BN), gv, red);
                let (b0, b1) = margins(t, scores2.add(i * crate::hc::gfx11_consts::ROUTE_ROWS + E), pred.add(BN), gv, red);
                let mut k = 0;
                while k < 2 {
                    let e = t + 256 * k;
                    if e < E {
                        let (d, d2) = if k == 0 { (a0, b0) } else { (a1, b1) };
                        *p_d.add(i * E + e) = d;
                        *p_d2.add(i * E + e) = d2;
                        if decode {
                            let mut z = ca * d + ca2 * d2 + cc + *pred.add(BIAS + e);
                            let mut q = 0;
                            while q < TOP_K {
                                z += *pred.add(CO + *p_picks.add(i * TOP_K + q) as usize * E + e);
                                q += 1;
                            }
                            let mut r = 0;
                            while r < MAP_RANK {
                                z += *hv.add(r) * *pred.add(MAP_V + r * E + e);
                                r += 1;
                            }
                            let f = 1.0 + exp(if z < 20.0 { z } else { 20.0 });
                            if k == 0 {
                                s0 *= f;
                            } else {
                                s1 *= f;
                            }
                        }
                    }
                    k += 1;
                }
                i += stride;
            }
            if !decode || plan.is_null() {
                return;
            }
            // the plan: each expert's place among the keys (the greater first, between equals the lower expert)
            let mut k = 0;
            while k < 2 {
                let e = t + 256 * k;
                if e < E {
                    *sk.add(e) = if k == 0 { s0 } else { s1 };
                }
                k += 1;
            }
            workgroup_sync();
            let mut k = 0;
            while k < 2 {
                let e = t + 256 * k;
                if e < E {
                    let v = *sk.add(e);
                    let mut r = 0usize;
                    let mut j = 0;
                    while j < E {
                        let w = *sk.add(j);
                        r += (w > v || (w == v && j < e)) as usize;
                        j += 1;
                    }
                    if r < PLAN {
                        *plan.add(r) = e as u32;
                    }
                }
                k += 1;
            }
            fence(Ordering::Release);
            workgroup_sync();
            if t == 0 && !word.is_null() {
                AtomicU32::from_ptr(word).store(value, Ordering::Release);
            }
        }
    }

    /// Pairs' updates over the `n` tokens from row `t0` (see the module), a workgroup a pair: workgroup w takes entry
    /// `first + w` of `tab` (`LEARN_TAB` device addresses a pair: its state, its margins `[STEP][E]` kept by its
    /// prediction, the picks at its layer and at the next, its labels, each `[STEP][TOP_K]` u32). Workgroups of
    /// `LEARN_WG`, a thread an expert.
    #[no_mangle]
    pub extern "gpu-kernel" fn pred_learn(tab: *const u64, first: u32, t0: u32, n: u32, map: u32) {
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
            let ms = *row.add(5) as *mut f32;
            let hs = ms.add(MS_H + t0 * MAP_RANK) as *const f32;
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
                    let mut r = 0;
                    while r < MAP_RANK {
                        z += *hs.add((a + i) * MAP_RANK + r) * *s.add(MAP_V + r * E + e);
                        r += 1;
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
                // a decode step's input map (`map`): h's gradients for `pred_map` (V as it stood), then V's update
                if map != 0 {
                    if e < m * MAP_RANK {
                        let (i, r) = (e / MAP_RANK, e % MAP_RANK);
                        let mut g = 0f32;
                        let mut j = 0;
                        while j < E {
                            g += *s.add(MAP_V + r * E + j) * *gz.add(i * E + j);
                            j += 1;
                        }
                        *ms.add(MS_DH + i * MAP_RANK + r) = g;
                    }
                    workgroup_sync();
                    let mut r = 0;
                    while r < MAP_RANK {
                        let mut g = 0f32;
                        let mut i = 0;
                        while i < m {
                            g += *hs.add((a + i) * MAP_RANK + r) * *gz.add(i * E + e);
                            i += 1;
                        }
                        let o = r * E + e;
                        adam(s.add(MAP_V + o), s.add(M_MAP_V + o), s.add(V_MAP_V + o), g, LR_X / c1, c2);
                        r += 1;
                    }
                }
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

    /// A decode step's update of a pair's U (after `pred_learn`, which leaves the step's β·ᵗ in the header and h's
    /// gradients in the map store): dU[k][r] = Σ_i x̂_i[k] · dh_i[r], Adam. Entry `entry` of `tab`, `n` tokens. Grid
    /// `MAP_WGS` of 256, U's entries strided over them.
    #[no_mangle]
    pub extern "gpu-kernel" fn pred_map(tab: *const u64, entry: u32, n: u32) {
        unsafe {
            let row = tab.add(entry as usize * LEARN_TAB);
            let s = *row as *mut f32;
            let ms = *row.add(5) as *const f32;
            let n = n as usize;
            let (b1t, b2t) = (*s.add(HDR + B1T), *s.add(HDR + B2T));
            let (lr, c2) = (LR_X / (1.0 - b1t), sqrt(1.0 - b2t));
            let mut o = workgroup_id_x() as usize * 256 + workitem_id_x() as usize;
            while o < H * MAP_RANK {
                let (k, r) = (o / MAP_RANK, o % MAP_RANK);
                let mut g = 0f32;
                let mut i = 0;
                while i < n {
                    g += *ms.add(MS_X + i * H + k) * *ms.add(MS_RN + i) * *ms.add(MS_DH + i * MAP_RANK + r);
                    i += 1;
                }
                adam(s.add(MAP_U + o), s.add(M_MAP_U + o), s.add(V_MAP_U + o), g, lr, c2);
                o += MAP_WGS * 256;
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
