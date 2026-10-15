//! Kimi delta attention, one token: the three projections through the causal
//! short convolution and SiLU, then per head the l2-normed query and key, the
//! bounded forget gate, the delta-rule update of the `[k][v]` memory and the
//! readout, a gated RMSNorm, and the output projection. The reference's
//! arithmetic (`reference::Kda::forward`) in decode order. The layer's state
//! between tokens is the convolution's last `CONV_TAPS - 1` inputs and the
//! memory, both the caller's (`layout::KDA_WINDOW`, `layout::KDA_MEMORY`).
//! The projections split by rows over the pool and the heads' recurrences run
//! one per item, each on its own slice of the memory and the scratch.

use crate::hc::sigmoid;
use crate::weights;
use crate::*;
use common::gemv;
use common::math::{exp, sqrt};
use common::pool::Pool;
use common::rmsnorm::rmsnorm;
use common::swiglu::silu;
use common::trace::Trace;

pub struct Scratch<'a> {
    pub mixed: &'a mut [f32],  // CONV_DIM: q | k | v before the convolution
    pub conv: &'a mut [f32],   // CONV_DIM: after it
    pub fa: &'a mut [f32],     // KDA_DIM
    pub f: &'a mut [f32],      // KDA_QKV
    pub beta: &'a mut [f32],   // KDA_HEADS
    pub ga: &'a mut [f32],     // KDA_DIM
    pub gate: &'a mut [f32],   // KDA_QKV
    pub normed: &'a mut [f32], // KDA_QKV: the heads' gated-normed readouts
    pub heads: &'a mut [f32],  // KDA_HEADS · HEAD_SCRATCH: per head, qh | kh | delta | o | on
}
/// f32s of scratch a head's recurrence needs.
pub const HEAD_SCRATCH: usize = 5 * KDA_DIM;

#[inline(always)]
fn l2norm(x: &mut [f32]) {
    let mut ss = 0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let r = 1.0 / sqrt(ss + 1e-6);
    for v in x.iter_mut() {
        *v *= r;
    }
}

/// `xn` is the normed block input (`[H]`); `window` `[CONV_TAPS - 1][CONV_DIM]`
/// and `mem` `[KDA_HEADS][KDA_DIM][KDA_DIM]` are this layer's state, advanced
/// by one position; `out` is the block output (`[H]`).
#[inline(always)]
pub fn step(l: usize, w: &weights::Kda, xn: &[f32], window: &mut [f32], mem: &mut [f32], s: &mut Scratch, pool: &dyn Pool, tr: &mut impl Trace, out: &mut [f32]) {
    gemv::bf16_par::<KDA_QKV, H>(w.q, xn, &mut s.mixed[..KDA_QKV], pool);
    gemv::bf16_par::<KDA_QKV, H>(w.k, xn, &mut s.mixed[KDA_QKV..2 * KDA_QKV], pool);
    gemv::bf16_par::<KDA_QKV, H>(w.v, xn, &mut s.mixed[2 * KDA_QKV..], pool);
    // the causal depthwise convolution: tap j sees the input TAPS-1-j steps back
    for part in 0..3 {
        let taps = w.conv[part];
        for c in part * KDA_QKV..(part + 1) * KDA_QKV {
            let cc = c - part * KDA_QKV;
            let mut acc = 0f32;
            for j in 0..CONV_TAPS - 1 {
                acc += gemv::widen(taps[cc * CONV_TAPS + j]) * window[j * CONV_DIM + c];
            }
            acc += gemv::widen(taps[cc * CONV_TAPS + CONV_TAPS - 1]) * s.mixed[c];
            s.conv[c] = silu(acc);
        }
    }
    window.copy_within(CONV_DIM.., 0);
    window[(CONV_TAPS - 2) * CONV_DIM..].copy_from_slice(s.mixed);
    tr.edge(l, "self_attn.conv.out", s.conv);

    gemv::bf16::<KDA_DIM, H>(w.f_a, xn, s.fa);
    gemv::bf16_par::<KDA_QKV, KDA_DIM>(w.f_b, s.fa, s.f, pool);
    gemv::bf16::<KDA_HEADS, H>(w.b, xn, s.beta);
    gemv::bf16::<KDA_DIM, H>(w.g_a, xn, s.ga);
    gemv::bf16_par::<KDA_QKV, KDA_DIM>(w.g_b, s.ga, s.gate, pool);
    // the heads, one per item: head h owns mem[h], normed[h] and heads[h] of the scratch
    let (conv, f, beta, gate) = (&*s.conv, &*s.f, &*s.beta, &*s.gate);
    let (mem_p, normed_p, heads_p) = (mem.as_mut_ptr() as usize, s.normed.as_mut_ptr() as usize, s.heads.as_mut_ptr() as usize);
    pool.run(KDA_HEADS, &move |h| {
        let m = unsafe { core::slice::from_raw_parts_mut((mem_p as *mut f32).add(h * KDA_DIM * KDA_DIM), KDA_DIM * KDA_DIM) };
        let normed = unsafe { core::slice::from_raw_parts_mut((normed_p as *mut f32).add(h * KDA_DIM), KDA_DIM) };
        let hs = unsafe { core::slice::from_raw_parts_mut((heads_p as *mut f32).add(h * HEAD_SCRATCH), HEAD_SCRATCH) };
        let (qh, rest) = hs.split_at_mut(KDA_DIM);
        let (kh, rest) = rest.split_at_mut(KDA_DIM);
        let (delta, rest) = rest.split_at_mut(KDA_DIM);
        let (o, on) = rest.split_at_mut(KDA_DIM);
        head(w, h, conv, f, beta, gate, m, qh, kh, delta, o, on, normed);
    });
    tr.edge(l, "self_attn.o_proj.in", s.normed);
    gemv::bf16_par::<H, KDA_QKV>(w.o, s.normed, out, pool);
}

/// One head's recurrence at this position: l2-normed query and key, the bounded
/// forget gate on the memory, the delta-rule update, the readout, the gated norm.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn head(w: &weights::Kda, h: usize, conv: &[f32], f: &[f32], beta: &[f32], gate: &[f32], m: &mut [f32], qh: &mut [f32], kh: &mut [f32], delta: &mut [f32], o: &mut [f32], on: &mut [f32], normed: &mut [f32]) {
    let scale = 1.0 / sqrt(KDA_DIM as f32);
    qh.copy_from_slice(&conv[h * KDA_DIM..(h + 1) * KDA_DIM]);
    kh.copy_from_slice(&conv[KDA_QKV + h * KDA_DIM..KDA_QKV + (h + 1) * KDA_DIM]);
    let vh = &conv[2 * KDA_QKV + h * KDA_DIM..2 * KDA_QKV + (h + 1) * KDA_DIM];
    l2norm(qh);
    l2norm(kh);
    for v in qh.iter_mut() {
        *v *= scale;
    }
    let decay = exp(w.a_log[h]);
    let b = sigmoid(beta[h]);
    // two sweeps of the memory, each row once: the forget gate (log-decay in (LOWER, 0)
    // per key channel) with the read `S·k` fused, then the delta-rule update with the
    // readout `S·q` fused — the same products in the same order as four sweeps would make
    for vv in 0..KDA_DIM {
        delta[vv] = 0.0;
        o[vv] = 0.0;
    }
    for kk in 0..KDA_DIM {
        let g = LINEAR_GATE_LOWER_BOUND * sigmoid(decay * (f[h * KDA_DIM + kk] + w.dt_bias[h * KDA_DIM + kk]));
        let e = exp(g);
        let k = kh[kk];
        let row = &mut m[kk * KDA_DIM..(kk + 1) * KDA_DIM];
        for vv in 0..KDA_DIM {
            row[vv] *= e;
            delta[vv] += row[vv] * k;
        }
    }
    for vv in 0..KDA_DIM {
        delta[vv] = (vh[vv] - delta[vv]) * b;
    }
    for kk in 0..KDA_DIM {
        let (k, qk) = (kh[kk], qh[kk]);
        let row = &mut m[kk * KDA_DIM..(kk + 1) * KDA_DIM];
        for vv in 0..KDA_DIM {
            row[vv] += k * delta[vv];
            o[vv] += row[vv] * qk;
        }
    }
    rmsnorm::<KDA_DIM>(o, &w.o_norm, RMS_NORM_EPS, on);
    for vv in 0..KDA_DIM {
        normed[vv] = on[vv] * sigmoid(gate[h * KDA_DIM + vv]);
    }
}
