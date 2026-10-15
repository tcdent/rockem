//! The FFN half of a layer, one token, on its normed input: the clamped
//! SwiGLU MLP of the first layers, or the MoE — the sigmoid router with its
//! selection bias (the top `TOP_K` of `E` by score plus bias, weighted by their
//! scores renormalized and scaled), the routed experts through a `Routed`
//! realization, and the shared expert. The reference's arithmetic
//! (`reference::Moe`). The dense projections here are FP8 as the checkpoint
//! stores them (`common::gemv::fp8`).

use crate::hc::sigmoid;
use crate::weights;
use crate::*;
use common::gemv;
use common::pool::Pool;
use common::swiglu::swiglu_clamped;
use common::trace::Trace;

pub struct Scratch<'a> {
    pub gu: &'a mut [f32],         // 2·INTERMEDIATE_SIZE (the widest)
    pub h: &'a mut [f32],          // INTERMEDIATE_SIZE
    pub expert_out: &'a mut [f32], // TOP_K·H
    pub shared: &'a mut [f32],     // H
    pub score: &'a mut [f32],      // E
}

/// Where the routed experts run: for MoE layer `l`, the `TOP_K` experts named in
/// `idx` on `x` (`[H]`), each one's raw output to `out[k·H..]` in the order of `idx`.
pub trait Routed {
    fn run(&mut self, l: usize, idx: &[usize; TOP_K], x: &[f32], out: &mut [f32]);
}
#[cfg(feature = "std")]
impl<R: Routed + ?Sized> Routed for Box<R> {
    fn run(&mut self, l: usize, idx: &[usize; TOP_K], x: &[f32], out: &mut [f32]) {
        (**self).run(l, idx, x, out)
    }
}

#[inline(always)]
pub fn dense(l: usize, w: &weights::DenseFfn, xn: &[f32], s: &mut Scratch, pool: &dyn Pool, tr: &mut impl Trace, out: &mut [f32]) {
    const N: usize = INTERMEDIATE_SIZE;
    gemv::fp8_par::<N, H>(w.gate.w, w.gate.s, xn, &mut s.gu[..N], pool);
    gemv::fp8_par::<N, H>(w.up.w, w.up.s, xn, &mut s.gu[N..2 * N], pool);
    swiglu_clamped::<N>(s.gu, SWIGLU_LIMIT, s.h);
    gemv::fp8_par::<H, N>(w.down.w, w.down.s, s.h, out, pool);
    tr.edge(l, "mlp.out", out);
}

/// The router: scores are sigmoids of the logits; the pick is by score plus
/// the selection bias, `TOP_K` of them, ascending (lower index first on a
/// tie); the weights are the picks' scores over their sum, times the scaling
/// factor. `forced` (bring-up only) takes these picks instead, weights still
/// computed: what separates a tie flip from everything else when diffing.
#[inline(always)]
pub fn route(l: usize, w: &weights::Moe, xn: &[f32], s: &mut Scratch, pool: &dyn Pool, forced: Option<&[usize; TOP_K]>, tr: &mut impl Trace) -> ([usize; TOP_K], [f32; TOP_K]) {
    gemv::bf16_par::<E, H>(w.router, xn, s.score, pool);
    for v in s.score.iter_mut() {
        *v = sigmoid(*v);
    }
    let mut idx = [0usize; TOP_K];
    match forced {
        Some(f) => idx = *f,
        None => {
            let mut taken = [false; E];
            for k in 0..TOP_K {
                let mut best = usize::MAX;
                for e in 0..E {
                    if !taken[e] && (best == usize::MAX || s.score[e] + w.bias[e] > s.score[best] + w.bias[best]) {
                        best = e;
                    }
                }
                taken[best] = true;
                idx[k] = best;
            }
        }
    }
    idx.sort_unstable();
    let mut sum = 1e-20f32;
    for &e in &idx {
        sum += s.score[e];
    }
    let mut wt = [0f32; TOP_K];
    for k in 0..TOP_K {
        wt[k] = s.score[idx[k]] / sum * ROUTED_SCALING_FACTOR;
    }
    tr.topk(l, &idx, &wt);
    (idx, wt)
}

#[inline(always)]
pub fn moe(l: usize, w: &weights::Moe, xn: &[f32], s: &mut Scratch, pool: &dyn Pool, routed: &mut impl Routed, forced: Option<&[usize; TOP_K]>, tr: &mut impl Trace, out: &mut [f32]) {
    let (idx, wt) = route(l, w, xn, s, pool, forced, tr);
    routed.run(l, &idx, xn, s.expert_out);
    for i in 0..H {
        out[i] = 0.0;
    }
    for k in 0..TOP_K {
        let eo = &s.expert_out[k * H..(k + 1) * H];
        tr.routed(l, k, eo);
        for i in 0..H {
            out[i] += wt[k] * eo[i];
        }
    }
    tr.edge(l, "routed", out);
    const S: usize = N_SHARED_EXPERTS * D_FF;
    gemv::fp8_par::<S, H>(w.shared_gate.w, w.shared_gate.s, xn, &mut s.gu[..S], pool);
    gemv::fp8_par::<S, H>(w.shared_up.w, w.shared_up.s, xn, &mut s.gu[S..2 * S], pool);
    swiglu_clamped::<S>(s.gu, SWIGLU_LIMIT, &mut s.h[..S]);
    gemv::fp8_par::<H, S>(w.shared_down.w, w.shared_down.s, &s.h[..S], s.shared, pool);
    tr.edge(l, "shared", s.shared);
    for i in 0..H {
        out[i] += s.shared[i];
    }
}
