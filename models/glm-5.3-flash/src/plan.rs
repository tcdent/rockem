//! The decode prefetch's learned expert predictor on the cards (queue 082, tcdent's design; phase 3/4 of
//! docs/expert-cache-architecture.md): `predict.rs`'s model and update, run where its inputs already are — the FFN card
//! computes the next layer's route-ahead scores and this layer's picks — in the cards' idle time. The host only reads
//! the plan: a decode step's best `PLAN` experts for the next layer.
//!
//! Two launches a MoE layer l, posted once its picks are in. `pred_step`, in the FFN card's queue behind the resident
//! experts and ahead of the FFN's output half (which spins on the host's experts): pair l predicts layer l + 1 from the
//! route-ahead scores, keeping each token's margin `d` for its own update a layer later; a decode step also ranks the
//! union (Π(1 + eᶻ), the order of Σ softplus) and writes the plan and the host's word. `pred_learn`, in the card's
//! side queue (`Kfd::launch_side`, beside the main queue at the lowest priority, so its milliseconds of a prefill
//! chunk hold nothing): pair l − 1 learns from layer l's picks (per-expert BCE, Adam, in batches of `SPEC_TOKENS`
//! tokens as decode's steps). In the one queue, a prefill chunk's learning held the other chunk's card half: prefill
//! 1.85 → 2.9 ms a prompt token (2026-10-21). A pair's state and margins live on its layer's FFN card.
//!
//! The pair's state, f32, `STATE` long: the calibration and Adam's terms (`HDR`), the next layer's router bias (`BN`),
//! the bias and its moments, the co-routing table `[E][E]` (row: an expert picked here) and its moments.

use crate::{E, SPEC_TOKENS, TOP_K};

/// Adam's learning rates: the co-routing table; the calibration (ca, cc) and the bias.
pub const LR_CO: f32 = 3e-3;
pub const LR_B: f32 = 1e-2;
/// The simulation's loss divided by the 40 layer pairs it trained together; kept so its learning rates carry over.
pub const PAIRS_SCALE: f32 = 40.0;
/// Experts a decode plan names, best first: the host drops the ones resident on the cards and keeps `PREFETCH_CAP`.
pub const PLAN: usize = 24;

/// The header: ca, cc, their first moments, their second moments, β1ᵗ, β2ᵗ.
pub const HDR: usize = 0;
pub const BN: usize = HDR + 8;
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
    s[HDR + 6] = 1.0;
    s[HDR + 7] = 1.0;
    s[BN..BN + E].copy_from_slice(&bias_next[..E]);
    s
}

/// `pred_learn`'s LDS, in words: the batch's dL/dz and labels `[SPEC_TOKENS][E]` each, its picks, the header, the
/// reductions.
pub const LEARN_LDS: u32 = (4 * (2 * SPEC_TOKENS * E + SPEC_TOKENS * TOP_K + 8 + 16)) as u32;
/// `pred_step`'s: the reductions, a token's scores and the plan's keys `[E]` each.
pub const PLAN_LDS: u32 = (4 * (16 + 2 * E)) as u32;
/// Workgroups that predict a long step's tokens, a token a workgroup at a time.
pub const PLAN_PRED_WGS: usize = 64;

#[cfg(target_arch = "amdgpu")]
pub mod gfx11 {
    use super::*;
    use common::gfx11::{lds, workgroup_sum, workgroup_sync};
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

    /// Pair l's prediction (see the module): its state, the route-ahead scores (`[n][ROUTE_ROWS]`, the next layer's
    /// from `E`), the margins it keeps `[n][E]`, the picks at l `[n][TOP_K]` u32. `plan` (null in a long step): the
    /// decode plan and the host's `word`. Grid: 1 for a decode step, `PLAN_PRED_WGS` for a long one.
    #[no_mangle]
    pub extern "gpu-kernel" fn pred_step(pred: *const f32, scores: *const f32, p_d: *mut f32, p_picks: *const u32, plan: *mut u32, word: *mut u32, value: u32, n: u32) {
        unsafe {
            let t = workitem_id_x() as usize;
            let wg = workgroup_id_x() as usize;
            let n = n as usize;
            let decode = n <= SPEC_TOKENS;
            let l = lds::<f32>();
            let red = l; // [16]
            let gv = red.add(16); // [E]
            let sk = gv.add(E); // [E]
            // the tokens this workgroup predicts: a decode step's all in one, a long step's striped
            let (first, stride) = if decode { (0, 1) } else { (wg, PLAN_PRED_WGS) };
            let (ca, cc) = (*pred.add(HDR), *pred.add(HDR + 1));
            // the plan's key of this thread's experts t and t + 256 (scalars: an array across the barriers is scratch)
            let (mut s0, mut s1) = (1f32, 1f32);
            let mut i = first;
            while i < n {
                let mut k = 0;
                while k < 2 {
                    let e = t + 256 * k;
                    if e < E {
                        *gv.add(e) = sigmoid(*scores.add(i * crate::hc::gfx11_consts::ROUTE_ROWS + E + e)) + *pred.add(BN + e);
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
                let mut k = 0;
                while k < 2 {
                    let e = t + 256 * k;
                    if e < E {
                        let d = (*gv.add(e) - thr) * 10.0;
                        *p_d.add(i * E + e) = d;
                        if decode {
                            let mut z = ca * d + cc + *pred.add(BIAS + e);
                            let mut q = 0;
                            while q < TOP_K {
                                z += *pred.add(CO + *p_picks.add(i * TOP_K + q) as usize * E + e);
                                q += 1;
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
                workgroup_sync();
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

    /// Pair l − 1's update over `n` tokens (see the module): its state, its tokens' margins `[n][E]` (kept by its
    /// prediction), the picks at l − 1 and at l (its labels), `[n][TOP_K]` u32. Grid 1.
    #[no_mangle]
    pub extern "gpu-kernel" fn pred_learn(s: *mut f32, d: *const f32, picks: *const u32, labels: *const u32, n: u32) {
        unsafe {
            let l = lds::<f32>();
            let gz = l; // [SPEC_TOKENS][E]
            let y = gz.add(SPEC_TOKENS * E) as *mut u32; // [SPEC_TOKENS][E]
            let pk = y.add(SPEC_TOKENS * E); // [SPEC_TOKENS * TOP_K]
            let cal = pk.add(SPEC_TOKENS * TOP_K) as *mut f32; // [8]
            let red = cal.add(8); // [16]
            learn_pair(workitem_id_x() as usize, s, d, picks, labels, n as usize, gz, y, pk, cal, red);
        }
    }

    /// One pair's update over `n` tokens in batches of `SPEC_TOKENS` (a whole workgroup): each batch's logits from the
    /// state as it stands, dL/dz = (σ(z) − y) / (batch · PAIRS_SCALE), Adam on the calibration, the bias and the rows of
    /// the co-routing table the batch picked.
    #[inline(always)]
    unsafe fn learn_pair(t: usize, s: *mut f32, d: *const f32, picks: *const u32, labels: *const u32, n: usize, gz: *mut f32, y: *mut u32, pk: *mut u32, cal: *mut f32, red: *mut f32) {
        if t < 8 {
            *cal.add(t) = *s.add(HDR + t);
        }
        let mut a = 0;
        while a < n {
            let m = if n - a < SPEC_TOKENS { n - a } else { SPEC_TOKENS };
            let mut j = t;
            while j < m * E {
                *y.add(j) = 0;
                j += 256;
            }
            workgroup_sync();
            if t < m * TOP_K {
                let i = t / TOP_K;
                *y.add(i * E + *labels.add((a + i) * TOP_K + t % TOP_K) as usize) = 1;
                *pk.add(t) = *picks.add((a + i) * TOP_K + t % TOP_K);
            }
            workgroup_sync();
            let (ca, cc) = (*cal, *cal.add(1));
            let sc = 1.0 / (m as f32 * PAIRS_SCALE);
            let (mut gca, mut gcc) = (0f32, 0f32);
            let (mut gb0, mut gb1) = (0f32, 0f32);
            let mut k = 0;
            while k < 2 {
                let e = t + 256 * k;
                if e < E {
                    let mut i = 0;
                    while i < m {
                        let dv = *d.add((a + i) * E + e);
                        let mut z = ca * dv + cc + *s.add(BIAS + e);
                        let mut q = 0;
                        while q < TOP_K {
                            z += *s.add(CO + *pk.add(i * TOP_K + q) as usize * E + e);
                            q += 1;
                        }
                        let g = (sigmoid(z) - *y.add(i * E + e) as f32) * sc;
                        *gz.add(i * E + e) = g;
                        if k == 0 {
                            gb0 += g;
                        } else {
                            gb1 += g;
                        }
                        gca += g * dv;
                        gcc += g;
                        i += 1;
                    }
                }
                k += 1;
            }
            let gca = workgroup_sum(t as u32, gca, red);
            let gcc = workgroup_sum(t as u32, gcc, red);
            let (b1t, b2t) = (*cal.add(6) * 0.9, *cal.add(7) * 0.999);
            let (c1, c2) = (1.0 - b1t, sqrt(1.0 - b2t));
            let mut k = 0;
            while k < 2 {
                let e = t + 256 * k;
                if e < E {
                    adam(s.add(BIAS + e), s.add(M_BIAS + e), s.add(V_BIAS + e), if k == 0 { gb0 } else { gb1 }, LR_B / c1, c2);
                    // each row of the co-routing table the batch picked, once: its gradient the sum of its tokens' dL/dz
                    let mut j = 0;
                    while j < m * TOP_K {
                        let q = *pk.add(j);
                        let mut seen = false;
                        let mut jj = 0;
                        while jj < j {
                            seen |= *pk.add(jj) == q;
                            jj += 1;
                        }
                        if !seen {
                            let mut g = 0f32;
                            let mut i = j / TOP_K;
                            while i < m {
                                let mut hit = false;
                                let mut qq = 0;
                                while qq < TOP_K {
                                    hit |= *pk.add(i * TOP_K + qq) == q;
                                    qq += 1;
                                }
                                if hit {
                                    g += *gz.add(i * E + e);
                                }
                                i += 1;
                            }
                            let r = q as usize * E + e;
                            adam(s.add(CO + r), s.add(M_CO + r), s.add(V_CO + r), g, LR_CO / c1, c2);
                        }
                        j += 1;
                    }
                }
                k += 1;
            }
            workgroup_sync();
            if t == 0 {
                adam(cal, cal.add(2), cal.add(4), gca, LR_B / c1, c2);
                adam(cal.add(1), cal.add(3), cal.add(5), gcc, LR_B / c1, c2);
                *cal.add(6) = b1t;
                *cal.add(7) = b2t;
            }
            workgroup_sync();
            a += m;
        }
        if t < 8 {
            *s.add(HDR + t) = *cal.add(t);
        }
    }
}
