//! Kimi delta attention, one token: the three projections through the causal
//! short convolution and SiLU, then per head the l2-normed query and key, the
//! bounded forget gate, the delta-rule update of the `[k][v]` memory and the
//! readout, a gated RMSNorm, and the output projection. The reference's
//! arithmetic (`reference::Kda::forward`) in decode order. The layer's state
//! between tokens is the convolution's last `CONV_TAPS - 1` inputs and the
//! memory, both the caller's (`layout::KDA_WINDOW`, `layout::KDA_MEMORY`).

use crate::hc::sigmoid;
use crate::weights;
use crate::*;
use common::gemv;
use common::math::{exp, sqrt};
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
    pub qh: &'a mut [f32],     // KDA_DIM
    pub kh: &'a mut [f32],     // KDA_DIM
    pub delta: &'a mut [f32],  // KDA_DIM
    pub o: &'a mut [f32],      // KDA_DIM
    pub on: &'a mut [f32],     // KDA_DIM
}

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
pub fn step(l: usize, w: &weights::Kda, xn: &[f32], window: &mut [f32], mem: &mut [f32], s: &mut Scratch, tr: &mut impl Trace, out: &mut [f32]) {
    gemv::bf16::<KDA_QKV, H>(w.q, xn, &mut s.mixed[..KDA_QKV]);
    gemv::bf16::<KDA_QKV, H>(w.k, xn, &mut s.mixed[KDA_QKV..2 * KDA_QKV]);
    gemv::bf16::<KDA_QKV, H>(w.v, xn, &mut s.mixed[2 * KDA_QKV..]);
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
    gemv::bf16::<KDA_QKV, KDA_DIM>(w.f_b, s.fa, s.f);
    gemv::bf16::<KDA_HEADS, H>(w.b, xn, s.beta);
    gemv::bf16::<KDA_DIM, H>(w.g_a, xn, s.ga);
    gemv::bf16::<KDA_QKV, KDA_DIM>(w.g_b, s.ga, s.gate);
    let scale = 1.0 / sqrt(KDA_DIM as f32);
    for h in 0..KDA_HEADS {
        let r = h * KDA_DIM..(h + 1) * KDA_DIM;
        s.qh.copy_from_slice(&s.conv[r.clone()]);
        s.kh.copy_from_slice(&s.conv[KDA_QKV + h * KDA_DIM..KDA_QKV + (h + 1) * KDA_DIM]);
        let vh = &s.conv[2 * KDA_QKV + h * KDA_DIM..2 * KDA_QKV + (h + 1) * KDA_DIM];
        l2norm(s.qh);
        l2norm(s.kh);
        for v in s.qh.iter_mut() {
            *v *= scale;
        }
        let decay = exp(w.a_log[h]);
        let b = sigmoid(s.beta[h]);
        let m = &mut mem[h * KDA_DIM * KDA_DIM..(h + 1) * KDA_DIM * KDA_DIM];
        // the forget gate, bounded below: log-decay in (LOWER, 0) per key channel
        for kk in 0..KDA_DIM {
            let g = LINEAR_GATE_LOWER_BOUND * sigmoid(decay * (s.f[h * KDA_DIM + kk] + w.dt_bias[h * KDA_DIM + kk]));
            let e = exp(g);
            for vv in 0..KDA_DIM {
                m[kk * KDA_DIM + vv] *= e;
            }
        }
        for vv in 0..KDA_DIM {
            let mut acc = 0f32;
            for kk in 0..KDA_DIM {
                acc += m[kk * KDA_DIM + vv] * s.kh[kk];
            }
            s.delta[vv] = (vh[vv] - acc) * b;
        }
        for kk in 0..KDA_DIM {
            let k = s.kh[kk];
            for vv in 0..KDA_DIM {
                m[kk * KDA_DIM + vv] += k * s.delta[vv];
            }
        }
        for vv in 0..KDA_DIM {
            let mut acc = 0f32;
            for kk in 0..KDA_DIM {
                acc += m[kk * KDA_DIM + vv] * s.qh[kk];
            }
            s.o[vv] = acc;
        }
        rmsnorm::<KDA_DIM>(s.o, &w.o_norm, RMS_NORM_EPS, s.on);
        for vv in 0..KDA_DIM {
            s.normed[h * KDA_DIM + vv] = s.on[vv] * sigmoid(s.gate[h * KDA_DIM + vv]);
        }
    }
    tr.edge(l, "self_attn.o_proj.in", s.normed);
    gemv::bf16::<H, KDA_QKV>(w.o, s.normed, out);
}
