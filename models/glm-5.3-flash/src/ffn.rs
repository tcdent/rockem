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
    /// A step's tokens at once: token `i`'s picks `idx[i]`, its input `x[i · H..]`, its experts'
    /// outputs `out[(i · TOP_K + k) · H..]`. A realization that streams weights groups the step by
    /// expert so each expert's bytes are read once for every token that picked it; the default is
    /// `run` a token at a time.
    fn run_batch(&mut self, l: usize, idx: &[[usize; TOP_K]], x: &[f32], out: &mut [f32]) {
        for (i, picks) in idx.iter().enumerate() {
            self.run(l, picks, &x[i * H..(i + 1) * H], &mut out[i * TOP_K * H..(i + 1) * TOP_K * H]);
        }
    }
    /// A hint, before `run`: these experts (distinct) are likely layer `l`'s next; a realization may
    /// start pulling them toward the cores that will stream them.
    fn prefetch(&self, l: usize, idx: &[usize]) {
        let _ = (l, idx);
    }
    /// Seconds spent in `run` since the last call of this (bring-up: the token's profile); 0 if unmeasured.
    fn elapsed(&mut self) -> f64 {
        0.0
    }
}
#[cfg(feature = "std")]
impl<R: Routed + ?Sized> Routed for Box<R> {
    fn run(&mut self, l: usize, idx: &[usize; TOP_K], x: &[f32], out: &mut [f32]) {
        (**self).run(l, idx, x, out)
    }
    fn run_batch(&mut self, l: usize, idx: &[[usize; TOP_K]], x: &[f32], out: &mut [f32]) {
        (**self).run_batch(l, idx, x, out)
    }
    fn elapsed(&mut self) -> f64 {
        (**self).elapsed()
    }
    fn prefetch(&self, l: usize, idx: &[usize]) {
        (**self).prefetch(l, idx)
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
    route_logits(l, w, s, forced, tr)
}
/// `route` from the router's logits already in `s.score` (a card ran the GEMV).
#[inline(always)]
pub fn route_logits(l: usize, w: &weights::Moe, s: &mut Scratch, forced: Option<&[usize; TOP_K]>, tr: &mut impl Trace) -> ([usize; TOP_K], [f32; TOP_K]) {
    let (idx, wt) = route_scores(w, s.score, forced);
    tr.topk(l, &idx, &wt);
    (idx, wt)
}
/// `route_logits` on the logits alone (`[E]`, turned into scores in place), nothing traced: a step's
/// tokens route in parallel.
#[inline(always)]
pub fn route_scores(w: &weights::Moe, score: &mut [f32], forced: Option<&[usize; TOP_K]>) -> ([usize; TOP_K], [f32; TOP_K]) {
    // the scores weight the picks whoever makes them (forced picks skipped the sigmoid from
    // 1cc1e03 to ebfe97a: the gate failed, serving — the router's own picks — was unaffected)
    scores(score);
    let mut idx = match forced {
        Some(f) => *f,
        None => top(w, score),
    };
    // the reference mass: every ranked pick's score, whether it runs or not, summed in id order (the
    // order the weights always had: a sum in ranked order moved the SCORE text's NLL by 0.003, 2026-10-18)
    let ranked = idx;
    idx.sort_unstable();
    let mut sum = 1e-20f32;
    for &e in &idx {
        sum += score[e];
    }
    // the picks past ROUTED_K (ranked best first) are dropped: `E` marks a slot no expert runs in
    for &e in &ranked[crate::ROUTED_K..] {
        let k = idx.iter().position(|&x| x == e).unwrap();
        idx[k] = E;
    }
    idx.sort_unstable();
    let mut wt = [0f32; TOP_K];
    for k in 0..TOP_K {
        wt[k] = if idx[k] < E { score[idx[k]] / sum * ROUTED_SCALING_FACTOR } else { 0.0 };
    }
    (idx, wt)
}

/// The routed half of the MoE: the router's picks, the experts through
/// `routed`, their weighted sum into `out`.
/// The router's picks from its logits (`[E]`, turned into scores in place): the top `TOP_K`
/// by score plus the selection bias, in the order taken (the best first).
#[inline(always)]
pub fn picks(w: &weights::Moe, logits: &mut [f32]) -> [usize; TOP_K] {
    scores(logits);
    top(w, logits)
}
/// The router's logits into its scores, in place.
#[inline(always)]
fn scores(logits: &mut [f32]) {
    for v in logits.iter_mut() {
        *v = sigmoid(*v);
    }
}
/// The top `TOP_K` scores plus the selection bias, in the order taken.
#[inline(always)]
fn top(w: &weights::Moe, score: &[f32]) -> [usize; TOP_K] {
    let mut idx = [0usize; TOP_K];
    let mut taken = [false; E];
    for k in 0..TOP_K {
        let mut best = usize::MAX;
        for e in 0..E {
            if !taken[e] && (best == usize::MAX || score[e] + w.bias[e] > score[best] + w.bias[best]) {
                best = e;
            }
        }
        taken[best] = true;
        idx[k] = best;
    }
    idx
}

#[inline(always)]
pub fn moe_routed(l: usize, w: &weights::Moe, xn: &[f32], s: &mut Scratch, pool: &dyn Pool, routed: &mut impl Routed, forced: Option<&[usize; TOP_K]>, tr: &mut impl Trace, out: &mut [f32]) -> [usize; TOP_K] {
    let (idx, wt) = route(l, w, xn, s, pool, forced, tr);
    moe_run(l, xn, idx, wt, s, routed, tr, out)
}
/// `moe_routed` with the router's logits already in `s.score` (a card ran the GEMV over its own copy of the normed input).
#[inline(always)]
pub fn moe_routed_logits(l: usize, w: &weights::Moe, xn: &[f32], s: &mut Scratch, routed: &mut impl Routed, forced: Option<&[usize; TOP_K]>, tr: &mut impl Trace, out: &mut [f32]) -> [usize; TOP_K] {
    let (idx, wt) = route_logits(l, w, s, forced, tr);
    moe_run(l, xn, idx, wt, s, routed, tr, out)
}
#[inline(always)]
fn moe_run(l: usize, xn: &[f32], idx: [usize; TOP_K], wt: [f32; TOP_K], s: &mut Scratch, routed: &mut impl Routed, tr: &mut impl Trace, out: &mut [f32]) -> [usize; TOP_K] {
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
    idx
}

/// The shared expert on `xn`, into `s.shared`; the step adds it to the routed sum.
#[inline(always)]
pub fn moe_shared(w: &weights::Moe, xn: &[f32], s: &mut Scratch, pool: &dyn Pool) {
    const S: usize = N_SHARED_EXPERTS * D_FF;
    gemv::fp8_par::<S, H>(w.shared_gate.w, w.shared_gate.s, xn, &mut s.gu[..S], pool);
    gemv::fp8_par::<S, H>(w.shared_up.w, w.shared_up.s, xn, &mut s.gu[S..2 * S], pool);
    swiglu_clamped::<S>(s.gu, SWIGLU_LIMIT, &mut s.h[..S]);
    gemv::fp8_par::<H, S>(w.shared_down.w, w.shared_down.s, &s.h[..S], s.shared, pool);
}

pub mod gfx11_consts {
    use crate::*;
    /// Rows of the fused gate|up launch: the gate's rows, then the up's.
    pub const SHARED_GU_ROWS: usize = 2 * N_SHARED_EXPERTS * D_FF;
    pub const DENSE_GU_ROWS: usize = 2 * INTERMEDIATE_SIZE;
    /// Rows a wave takes in `ffn_gu` (FP8, normed, width H) past one token: four put it at 192
    /// registers with 20 of scratch (2026-10-17); every other rows kernel takes `common::gemv::ROWS`.
    pub const ROWS_GU: usize = 2;
}

#[cfg(target_arch = "amdgpu")]
pub mod gfx11 {
    //! The FFN's FP8 projections on the card: `ffn_gu` (rows of a `[m][H]`
    //! FP8 matrix over `n` activations rms-normed with `ln` on the way — a
    //! gate|up pair as one launch, the gate's rows first), and the down
    //! projections `ffn_down_shared` (`[H][D_FF]`) and `ffn_down_dense`
    //! (`[H][INTERMEDIATE_SIZE]`), whose activation is the clamped SwiGLU of
    //! the gate|up output computed as it is staged — no epilogue launch.
    //! Scales: one f32 per 128×128 block, `[m/128][K/128]` (`common::gemv::fp8`).
    use crate::*;
    use common::gemv::gfx11::{rows_fmt, Fp8};
    use common::gemv::ROWS;
    use super::gfx11_consts::ROWS_GU;
    use common::swiglu::silu;
    use core::arch::amdgpu::workitem_id_x;

    /// `out[i][row] = W[row] · rmsnorm_ln(x[i])`, FP8 rows of width H. Grid: rows_grid(m).
    #[no_mangle]
    pub extern "gpu-kernel" fn ffn_gu(x: *const f32, ln: *const u16, w: *const u8, s: *const f32, out: *mut f32, m: u32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            rows_fmt::<H, true, Fp8, ROWS_GU>(tid, Fp8 { w, s }, ln, m, n, RMS_NORM_EPS, |i, s| *x.add(i * H + s * 256 + t), || {}, |row, i, y| *out.add(i * m + row) = y);
        }
    }

    /// The FP8 rows kernels on the matrix path for a step of many tokens (`rows_wmma_with`). Grid: wmma_grid(m, n), LDS WMMA_LDS.
    #[no_mangle]
    pub extern "gpu-kernel" fn ffn_gu_w(x: *const f32, ln: *const u16, w: *const u8, s: *const f32, out: *mut f32, m: u32, n: u32) {
        unsafe {
            common::gemv::gfx11::rows_wmma_with::<H, true, _>(workitem_id_x(), common::gemv::gfx11::WFp8 { w, s }, ln, m as usize, n as usize, RMS_NORM_EPS, |i, k| *x.add(i * H + k), out);
        }
    }
    #[no_mangle]
    pub extern "gpu-kernel" fn ffn_down_shared_w(gu: *const f32, w: *const u8, s: *const f32, out: *mut f32, m: u32, n: u32) {
        unsafe {
            const N: usize = N_SHARED_EXPERTS * D_FF;
            common::gemv::gfx11::rows_wmma_with::<N, false, _>(workitem_id_x(), common::gemv::gfx11::WFp8 { w, s }, core::ptr::null(), m as usize, n as usize, 0.0, |i, k| swiglu_at::<N>(gu, i, k), out);
        }
    }
    #[no_mangle]
    pub extern "gpu-kernel" fn ffn_down_dense_w(gu: *const f32, w: *const u8, s: *const f32, out: *mut f32, m: u32, n: u32) {
        unsafe {
            const N: usize = INTERMEDIATE_SIZE;
            common::gemv::gfx11::rows_wmma_with::<N, false, _>(workitem_id_x(), common::gemv::gfx11::WFp8 { w, s }, core::ptr::null(), m as usize, n as usize, 0.0, |i, k| swiglu_at::<N>(gu, i, k), out);
        }
    }

    /// The clamped SwiGLU of element `k` of token `i`'s gate|up output (`[n][2·N]`).
    #[inline(always)]
    unsafe fn swiglu_at<const N: usize>(gu: *const f32, i: usize, k: usize) -> f32 {
        let (g, u) = (*gu.add(i * 2 * N + k), *gu.add(i * 2 * N + N + k));
        let g = if g > SWIGLU_LIMIT { SWIGLU_LIMIT } else { g };
        let u = if u > SWIGLU_LIMIT { SWIGLU_LIMIT } else if u < -SWIGLU_LIMIT { -SWIGLU_LIMIT } else { u };
        silu(g) * u
    }

    /// `out[i][row] = W[row] · swiglu(gu[i])` for a `[m][D_FF]` FP8 matrix: the shared expert's down projection. Grid: rows_grid(m).
    #[no_mangle]
    pub extern "gpu-kernel" fn ffn_down_shared(gu: *const f32, w: *const u8, s: *const f32, out: *mut f32, m: u32, n: u32) {
        unsafe {
            const N: usize = N_SHARED_EXPERTS * D_FF;
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            rows_fmt::<N, false, Fp8, ROWS>(tid, Fp8 { w, s }, core::ptr::null(), m, n, 0.0, |i, s| if s * 256 + t < N { swiglu_at::<N>(gu, i, s * 256 + t) } else { 0.0 }, || {}, |row, i, y| *out.add(i * m + row) = y);
        }
    }

    /// The same over `[m][INTERMEDIATE_SIZE]`: the dense layers' down projection.
    #[no_mangle]
    pub extern "gpu-kernel" fn ffn_down_dense(gu: *const f32, w: *const u8, s: *const f32, out: *mut f32, m: u32, n: u32) {
        unsafe {
            const N: usize = INTERMEDIATE_SIZE;
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            rows_fmt::<N, false, Fp8, ROWS>(tid, Fp8 { w, s }, core::ptr::null(), m, n, 0.0, |i, s| swiglu_at::<N>(gu, i, s * 256 + t), || {}, |row, i, y| *out.add(i * m + row) = y);
        }
    }
    const _: () = assert!(INTERMEDIATE_SIZE % 256 == 0 && (N_SHARED_EXPERTS * D_FF) % 256 == 0);
}
