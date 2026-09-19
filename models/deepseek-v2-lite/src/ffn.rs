//! The FFN half of a layer, fused: post-attention norm → (dense MLP | router
//! → routed experts → shared expert) → residual. Two kernels, not one with a
//! mode: layer 0 is dense and the rest are MoE, decided by a constant at
//! construction.
//!
//! The routed experts are a *realization* chosen at construction — `Routed`
//! — so the same kernel runs them from DRAM on this socket, from VRAM on a
//! card, or, later, on another node. The kernel never asks which; it hands
//! over the six indices and the input and gets six outputs back.

use crate::*;
use common::gemv;
use common::rmsnorm::rmsnorm;
use common::softmax::{softmax, topk};
use common::swiglu::swiglu;
use common::trace::Trace;

pub struct Dense<'a> {
    pub post_ln: &'a [f32],
    pub gate: &'a [u16], // [INTERMEDIATE_SIZE][H]
    pub up: &'a [u16],
    pub down: &'a [u16], // [H][INTERMEDIATE_SIZE]
}

pub struct Moe<'a> {
    pub post_ln: &'a [f32],
    pub router: &'a [u16],      // [N_ROUTED_EXPERTS][H]
    pub shared_gate: &'a [u16], // [SHARED_FF][H]
    pub shared_up: &'a [u16],
    pub shared_down: &'a [u16], // [H][SHARED_FF]
}

pub struct Scratch<'a> {
    pub hn: &'a mut [f32],         // H
    pub gu: &'a mut [f32],         // 2·INTERMEDIATE_SIZE (the widest)
    pub h: &'a mut [f32],          // INTERMEDIATE_SIZE
    pub out: &'a mut [f32],        // H
    pub expert_out: &'a mut [f32], // TOP_K · H: every routed expert's output
    pub router: &'a mut [f32],     // N_ROUTED_EXPERTS
}

/// Where the routed experts run. `run` computes, for MoE layer `l`, the
/// `TOP_K` experts named in `idx` on `x` (H wide) and writes each expert's
/// raw output to `out[k·H..]`, in the order of `idx`. Implementations own
/// their scratch and their copy of the expert stream.
pub trait Routed {
    fn run(&mut self, l: usize, idx: &[usize; TOP_K], x: &[f32], out: &mut [f32]);
}

/// Where the FFN half of a layer runs — `dense` for the dense layers, `moe`
/// for the rest — and where the final norm and LM head run (`Head`). Chosen
/// at construction, like `Attention` and `Routed`.
pub trait FeedForward {
    /// `h` is `[n][H]`, updated in place.
    fn dense(&mut self, l: usize, w: &Dense, h: &mut [f32], n: usize, tr: &mut dyn Trace);
    fn moe(&mut self, l: usize, w: &Moe, h: &mut [f32], n: usize, tr: &mut dyn Trace);
}
pub trait Head {
    /// `logits[i] = W_lm · rmsnorm(h[i])` for `n` tokens: `h` is `[n][H]`, `logits` `[n][VOCAB_SIZE]`.
    fn logits(&mut self, norm: &[f32], lm_head: &[u16], h: &[f32], n: usize, logits: &mut [f32]);
}
/// Both halves on one tier, so a node can hold them as one boxed realization.
pub trait FfnHead: FeedForward + Head {}
impl<T: FeedForward + Head> FfnHead for T {}

/// The router: post-attention norm into `s.hn`, then the greedy top-k with
/// unnormalized softmax weights × the scaling factor. Every FFN realization
/// runs this on the host — it is 64 rows, and the indices are what a tier
/// needs to know which bytes to fetch.
#[inline(always)]
pub fn route(l: usize, w: &Moe, h: &[f32], s: &mut Scratch, tr: &mut impl Trace) -> ([usize; TOP_K], [f32; TOP_K]) {
    rmsnorm::<H>(h, w.post_ln, RMS_NORM_EPS, s.hn);
    tr.edge(l, "post_attention_layernorm.out", s.hn);
    gemv::bf16::<N_ROUTED_EXPERTS, H>(w.router, s.hn, s.router);
    softmax(s.router);
    tr.edge(l, "mlp.gate.probs", s.router);
    let (mut idx, mut wt) = ([0usize; TOP_K], [0f32; TOP_K]);
    topk::<TOP_K>(s.router, &mut idx, &mut wt);
    for v in wt.iter_mut() {
        *v *= ROUTED_SCALING_FACTOR;
    }
    tr.topk(l, &idx, &wt);
    (idx, wt)
}

#[inline(always)]
pub fn dense(l: usize, w: &Dense, h: &mut [f32], s: &mut Scratch, tr: &mut impl Trace) {
    const N: usize = INTERMEDIATE_SIZE;
    rmsnorm::<H>(h, w.post_ln, RMS_NORM_EPS, s.hn);
    tr.edge(l, "post_attention_layernorm.out", s.hn);
    gemv::bf16::<N, H>(w.gate, s.hn, &mut s.gu[..N]);
    gemv::bf16::<N, H>(w.up, s.hn, &mut s.gu[N..2 * N]);
    swiglu::<N>(s.gu, s.h);
    gemv::bf16::<H, N>(w.down, s.h, s.out);
    tr.edge(l, "mlp.out", s.out);
    for i in 0..H {
        h[i] += s.out[i];
    }
}

#[inline(always)]
pub fn moe(l: usize, w: &Moe, h: &mut [f32], s: &mut Scratch, routed: &mut impl Routed, tr: &mut impl Trace) {
    let (idx, wt) = route(l, w, h, s, tr);
    routed.run(l, &idx, s.hn, s.expert_out);
    for i in 0..H {
        s.out[i] = 0.0;
    }
    for k in 0..TOP_K {
        let eo = &s.expert_out[k * H..(k + 1) * H];
        tr.routed(l, k, eo);
        for i in 0..H {
            s.out[i] += wt[k] * eo[i];
        }
    }

    const S: usize = SHARED_FF;
    gemv::bf16::<S, H>(w.shared_gate, s.hn, &mut s.gu[..S]);
    gemv::bf16::<S, H>(w.shared_up, s.hn, &mut s.gu[S..2 * S]);
    swiglu::<S>(s.gu, s.h);
    let shared = &mut s.expert_out[..H];
    gemv::bf16::<H, S>(w.shared_down, s.h, shared);
    tr.edge(l, "mlp.shared_experts.out", shared);
    for i in 0..H {
        s.out[i] += shared[i];
    }
    tr.edge(l, "mlp.out", s.out);
    for i in 0..H {
        h[i] += s.out[i];
    }
}

/// Launch geometry of the gfx11 realization, visible to hosts that launch it.
pub mod gfx11_consts {
    use crate::*;
    pub const WG: u32 = 256;
    /// `bf16_rows` and the down kernels: 8 rows per workgroup.
    pub const fn rows_grid(m: usize) -> u32 {
        m.div_ceil(8) as u32
    }
    /// LDS a rows kernel over `[m][K]` needs for a step of `n` tokens: the
    /// slab holds a round of steps at the pass width (`gemv::gfx11`), one
    /// token's whole activation at n = 1 (as the one-token kernels always
    /// staged) and TOKENS token-steps at most; plus 8 partials.
    pub const fn rows_lds(k: usize, n: usize) -> usize {
        let steps = k.div_ceil(256);
        let flight = common::gemv::FLIGHT;
        let slots = if n <= 1 { if steps < flight { steps } else { flight } } else { 2 * 8 };
        4 * (slots * common::gemv::STEP + 8)
    }
    /// norm = 8 partials.
    pub const NORM_LDS: usize = 4 * 8;
    /// route = normed input (H) + probabilities (N_ROUTED_EXPERTS) + 256 row partials + 8 partials.
    pub const ROUTE_LDS: usize = 4 * (H + N_ROUTED_EXPERTS + 256 + 8);
    /// group = per-expert counts.
    pub const GROUP_LDS: usize = 4 * N_ROUTED_EXPERTS;
    /// `group`'s `count` buffer: a count per expert, then the number of active experts,
    /// then the active experts in index order (what the expert kernels' grids walk).
    pub const COUNT_WORDS: usize = 2 * N_ROUTED_EXPERTS + 1;
    /// The statistics each instrumented kernel writes, by slot (`common::stats`).
    pub const ROUTE_STATS: [&str; 4] = ["rows_ticks", "softmax_ticks", "pick_ticks", "quant_ticks"];
    pub const DOWN_STATS: [&str; 4] = ["h_abs", "", "", ""];
    /// `sample` phases: the max pass, θ₀ (rank over the maxima), count + scan + gather, order + masses + walk.
    pub const SAMPLE_STATS: [&str; 4] = ["max_ticks", "theta0_ticks", "gather_ticks", "order_walk_ticks"];
    /// score = 256 partial sums + 8 partials.
    pub const SCORE_LDS: usize = 4 * (256 + 8);
    /// sample = 256 masses, 256 candidate values and indices, 256 offsets, 8 partials, θ₀.
    pub const SAMPLE_LDS: usize = 4 * (256 * 4 + 8 + 1);
    /// Logits per thread of the one `sample` workgroup.
    pub const SAMPLE_PER: usize = VOCAB_SIZE / 256;
}

#[cfg(target_arch = "amdgpu")]
pub mod gfx11 {
    //! The GPU realization of the FFN half and the head, for `n` tokens at
    //! once: `norm` (a workgroup per token: rmsnorm with a weight, written
    //! out — every rows kernel reads it, so the norm is computed once, not
    //! once per workgroup), `bf16_rows` (rows of a bf16 matrix over `n`
    //! activations staged a step at a time — `gemv::gfx11`; the dense layer's
    //! gate/up, the shared experts' gate/up, the LM head), `dense_down` and
    //! `shared_down` (the same rows over SwiGLU, folded into the staging,
    //! with the residual add as epilogue; the MoE one adds the routed
    //! experts' outputs in slot order), `route` (a workgroup per token: the
    //! router, the pick, the INT8 activation) and `group` (the tokens each
    //! expert serves, for the expert kernels). One token is n = 1 through
    //! the same kernels.
    pub use super::gfx11_consts::*;
    use crate::*;
    use common::gemv::gfx11::{rows_over, slot};
    use common::gemv::widen;
    use common::gfx11::{lds, wave_max, wave_sum, workgroup_max, workgroup_sum, workgroup_sync};
    use common::math::{exp, ln, round, rsqrt};
    use common::swiglu::silu;
    use core::arch::amdgpu::{workgroup_id_x, workitem_id_x};
    const _: () = assert!(TOKENS == common::gemv::gfx11::TOKENS);

    /// `xn[i] = rmsnorm(x[i]) ⊙ ln` for a workgroup per token. Grid: n × 256.
    #[no_mangle]
    pub extern "gpu-kernel" fn norm(x: *const f32, ln: *const u16, xn: *mut f32) {
        unsafe {
            let i = workgroup_id_x() as usize;
            let tid = workitem_id_x();
            let red: *mut f32 = lds();
            let (x, xn) = (x.add(i * H), xn.add(i * H));
            let mut ss = 0f32;
            let mut j = tid as usize;
            while j < H {
                let v = *x.add(j);
                ss += v * v;
                j += 256;
            }
            let r = rsqrt(workgroup_sum(tid, ss, red) / H as f32 + RMS_NORM_EPS);
            let mut j = tid as usize;
            while j < H {
                *xn.add(j) = widen(*ln.add(j)) * (*x.add(j) * r);
                j += 256;
            }
        }
    }

    /// `out[i][row] = W[row] · x[i]` for `m` rows of a `[m][H]` bf16 matrix
    /// and `n` activations `x` (`[n][H]`, already normed). Grid: rows_grid(m).
    #[no_mangle]
    pub extern "gpu-kernel" fn bf16_rows(x: *const f32, w: *const u16, out: *mut f32, m: u32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            rows_over::<H>(
                tid,
                w,
                m,
                n,
                |i, s, dst| *dst.add(slot(t)) = *x.add(i * H + s * 256 + t),
                |row, i, y| *out.add(i * m + row) = y,
            );
        }
    }

    /// Step `s` of `swiglu(gu[i])` for one token into `dst` (256 f32): `gu` is
    /// `[n][2·N]` (gate then up); element `k` is `silu(gate[k]) · up[k]`, as the CPU's.
    #[inline(always)]
    unsafe fn stage_swiglu<const N: usize>(t: usize, gu: *const f32, i: usize, s: usize, dst: *mut f32) {
        let k = s * 256 + t;
        if k < N {
            let g = gu.add(i * 2 * N);
            *dst.add(slot(t)) = silu(*g.add(k)) * *g.add(N + k);
        }
    }

    /// Dense layer: `h[i] += down · swiglu(gu[i])`, the new residual also to
    /// `out` (`[n][H]`, the host's trace of the step). Grid: rows_grid(H).
    #[no_mangle]
    pub extern "gpu-kernel" fn dense_down(gu: *const f32, w: *const u16, h: *mut f32, out: *mut f32, n: u32) {
        unsafe {
            const N: usize = INTERMEDIATE_SIZE;
            let tid = workitem_id_x();
            let t = tid as usize;
            rows_over::<N>(
                tid,
                w,
                H,
                n as usize,
                |i, s, dst| stage_swiglu::<N>(t, gu, i, s, dst),
                |row, i, y| {
                    let v = *h.add(i * H + row) + y;
                    *h.add(i * H + row) = v;
                    *out.add(i * H + row) = v;
                    common::stat!(0, if v < 0.0 { -v } else { v });
                },
            );
        }
    }

    /// MoE layer: `h[i] += shared_down · swiglu(gu[i]) + Σ_k wt[i][k] · expert_out[i][k]`,
    /// the new residual also to `out`. Grid: rows_grid(H).
    #[no_mangle]
    pub extern "gpu-kernel" fn shared_down(gu: *const f32, w: *const u16, expert_out: *const f32, wt: *const f32, h: *mut f32, out: *mut f32, n: u32) {
        unsafe {
            const N: usize = SHARED_FF;
            let tid = workitem_id_x();
            let t = tid as usize;
            rows_over::<N>(
                tid,
                w,
                H,
                n as usize,
                |i, s, dst| stage_swiglu::<N>(t, gu, i, s, dst),
                |row, i, y| {
                    let mut a = y;
                    let mut k = 0;
                    while k < TOP_K {
                        a += *wt.add(i * TOP_K + k) * *expert_out.add((i * TOP_K + k) * H + row);
                        k += 1;
                    }
                    let v = *h.add(i * H + row) + a;
                    *h.add(i * H + row) = v;
                    *out.add(i * H + row) = v;
                    common::stat!(0, if v < 0.0 { -v } else { v });
                },
            );
        }
    }

    /// The router, a workgroup per token, nothing serial: `xn = rmsnorm(h)`,
    /// kept in LDS and written out (`[n][H]`, what the shared experts' rows
    /// read — the norm is this kernel's, not a launch of its own); the
    /// N_ROUTED_EXPERTS router rows with every thread streaming a quarter of
    /// a row at once (one wave per row was latency-bound: 49 µs a layer,
    /// 2026-09-18); softmax in one wave, two logits a lane; the greedy top-k
    /// as a rank — each expert counts the ones that beat it, ties to the
    /// lowest index, exactly `softmax::topk`'s order — so the k chosen write
    /// their own slot: the scaled weight and the expert (for `group` and the
    /// trace); and `xn` quantized to INT8, `QAct::quantize` op for op, so the
    /// integers match the CPU path bit for bit (quantizing in the expert
    /// kernels instead measured a net loss, 2026-09-18). Grid: n × 256.
    #[no_mangle]
    pub extern "gpu-kernel" fn route(h: *const f32, ln: *const u16, w: *const u16, xn_out: *mut f32, wt: *mut f32, xq: *mut i8, xs: *mut f32, xg: *mut i32, expert: *mut u32, topk_out: *mut u32) {
        unsafe {
            const E: usize = N_ROUTED_EXPERTS;
            const G: usize = common::q4::GROUP;
            const Q: usize = H / 4; // a thread's quarter row
            let t0 = common::stat_clock!();
            let i = workgroup_id_x() as usize;
            let tid = workitem_id_x();
            let (t, lane, wave) = (tid as usize, tid % 32, (tid / 32) as usize);
            let xn: *mut f32 = lds();
            let prob = xn.add(H);
            let part = prob.add(E);
            let red = part.add(256);
            let (h, xn_out) = (h.add(i * H), xn_out.add(i * H));
            let (wt, xq, xs, xg, expert, topk_out) = (wt.add(i * TOP_K), xq.add(i * H), xs.add(i * (H / G)), xg.add(i * (H / G)), expert.add(i * TOP_K), topk_out.add(i * TOP_K));
            let mut ss = 0f32;
            let mut k = t;
            while k < H {
                let v = *h.add(k);
                ss += v * v;
                k += 256;
            }
            let r = rsqrt(workgroup_sum(tid, ss, red) / H as f32 + RMS_NORM_EPS);
            let mut k = t;
            while k < H {
                let v = widen(*ln.add(k)) * (*h.add(k) * r);
                *xn.add(k) = v;
                *xn_out.add(k) = v;
                k += 256;
            }
            workgroup_sync();
            // rows: thread t takes quarter t % 4 of row t / 4, eight bf16 per load
            let (row, q) = (t / 4, t % 4);
            let wr = w.add(row * H + q * Q) as *const [u16; 8];
            let xq4 = xn.add(q * Q);
            let (mut a0, mut a1) = (0f32, 0f32);
            let mut k = 0;
            while k < Q / 8 {
                let v = *wr.add(k);
                let x = xq4.add(k * 8);
                a0 += widen(v[0]) * *x + widen(v[1]) * *x.add(1) + widen(v[2]) * *x.add(2) + widen(v[3]) * *x.add(3);
                a1 += widen(v[4]) * *x.add(4) + widen(v[5]) * *x.add(5) + widen(v[6]) * *x.add(6) + widen(v[7]) * *x.add(7);
                k += 1;
            }
            *part.add(t) = a0 + a1;
            workgroup_sync();
            let t1 = common::stat_clock!();
            common::stat!(0, (t1.wrapping_sub(t0)) & common::stats::CLOCK_MASK);
            // softmax over the 64 logits in wave 0, two a lane: wave reductions only
            if wave == 0 {
                let l = lane as usize;
                let (la, lb) = (
                    *part.add(l * 4) + *part.add(l * 4 + 1) + *part.add(l * 4 + 2) + *part.add(l * 4 + 3),
                    *part.add((l + 32) * 4) + *part.add((l + 32) * 4 + 1) + *part.add((l + 32) * 4 + 2) + *part.add((l + 32) * 4 + 3),
                );
                let m = wave_max(lane, if la > lb { la } else { lb });
                let (ea, eb) = (exp(la - m), exp(lb - m));
                let inv = 1.0 / wave_sum(lane, ea + eb);
                *prob.add(l) = ea * inv;
                *prob.add(l + 32) = eb * inv;
            }
            workgroup_sync();
            let t2 = common::stat_clock!();
            common::stat!(1, (t2.wrapping_sub(t1)) & common::stats::CLOCK_MASK);
            // the pick, as a rank
            if t < E {
                let p = *prob.add(t);
                let mut rank = 0usize;
                let mut j = 0;
                while j < E {
                    let pj = *prob.add(j);
                    if pj > p || (pj == p && j < t) {
                        rank += 1;
                    }
                    j += 1;
                }
                if rank < TOP_K {
                    *wt.add(rank) = p * ROUTED_SCALING_FACTOR;
                    *expert.add(rank) = t as u32;
                    *topk_out.add(rank) = t as u32;
                }
            }
            let t3 = common::stat_clock!();
            common::stat!(2, (t3.wrapping_sub(t2)) & common::stats::CLOCK_MASK);
            // the INT8 activation: a wave per two groups of G, a lane per four elements, the
            // four integers packed into one store (four byte stores a lane were 2.7 µs a group)
            let mut g = wave * 2;
            while g < wave * 2 + 2 {
                let b = g * G + lane as usize * 4;
                let mut amax = 0f32;
                let mut j = 0;
                while j < 4 {
                    let v = *xn.add(b + j);
                    let a = if v < 0.0 { -v } else { v };
                    if a > amax {
                        amax = a;
                    }
                    j += 1;
                }
                let amax = wave_max(lane, amax);
                let s = amax / 127.0;
                let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
                let mut sum = 0i32;
                let mut packed = 0u32;
                let mut j = 0;
                while j < 4 {
                    let r = round(*xn.add(b + j) * inv);
                    let r = if r > 127.0 { 127.0 } else if r < -127.0 { -127.0 } else { r };
                    let q = r as i8;
                    packed |= (q as u8 as u32) << (8 * j);
                    sum += q as i32;
                    j += 1;
                }
                *(xq.add(b) as *mut u32) = packed;
                let sum = wave_sum(lane, sum as f32);
                if lane == 0 {
                    *xs.add(g) = s;
                    *xg.add(g) = sum as i32;
                }
                g += 1;
            }
            common::stat!(3, (common::stat_clock!().wrapping_sub(t3)) & common::stats::CLOCK_MASK);
            const _: () = assert!(E == 64 && 4 * E == 256 && Q % 8 == 0 && H / G == 16 && G == 4 * 32);
        }
    }

    /// The tokens each expert serves this step: `count[e]` and `list[e][..]`
    /// of `(token << 3) | slot` from `expert` (`[n][TOP_K]`), then the active
    /// experts compacted in index order after the counts (`COUNT_WORDS`), so
    /// the expert kernels' grids cover only experts with tokens — at one
    /// token, six of sixty-four. One workgroup; the order within a list is
    /// the atomics' and changes nothing, every entry being computed on its
    /// own. Grid: 1 × 256.
    #[no_mangle]
    pub extern "gpu-kernel" fn group(expert: *const u32, n: u32, count: *mut u32, list: *mut u32) {
        unsafe {
            const E: usize = N_ROUTED_EXPERTS;
            let t = workitem_id_x() as usize;
            let cnt: *mut u32 = lds();
            if t < E {
                *cnt.add(t) = 0;
            }
            workgroup_sync();
            if t < n as usize * TOP_K {
                let e = *expert.add(t) as usize;
                let slot = core::sync::atomic::AtomicU32::from_ptr(cnt.add(e)).fetch_add(1, core::sync::atomic::Ordering::Relaxed) as usize;
                *list.add(e * TOKENS + slot) = ((t / TOP_K) << 3 | (t % TOP_K)) as u32;
            }
            workgroup_sync();
            if t < E {
                *count.add(t) = *cnt.add(t);
            }
            if t == 0 {
                let mut k = 0;
                let mut e = 0;
                while e < E {
                    if *cnt.add(e) > 0 {
                        *count.add(E + 1 + k) = e as u32;
                        k += 1;
                    }
                    e += 1;
                }
                *count.add(E) = k as u32;
            }
            const _: () = assert!(TOKENS * TOP_K <= 256);
        }
    }

    /// The draw, one workgroup over the vocabulary: `common::sample::draw`'s
    /// twin, step for step (the recipe is that module's doc). Temperature 0 is
    /// the argmax (lowest index on ties, as `softmax::argmax`). Otherwise the
    /// k-th largest of the 256 thread maxima is θ₀, a lower bound of the k-th
    /// largest logit; the logits at or above it are gathered in index order
    /// (CAP at most) and ordered, which makes the k-th of the list exact. Were
    /// more than CAP above θ₀ — hundreds tied near the top, not real logits —
    /// the exact k-th is found first by bisection over the float's order-
    /// preserving bits. Ordering is by rank: each thread counts the pairs
    /// before its own and that count is its slot, two barriers and no sort.
    /// Thread 0 then walks the masses in f64 in the CPU's order. Grid: 1 × 256.
    #[no_mangle]
    pub extern "gpu-kernel" fn sample(logits: *const f32, out: *mut u32, u: f64, temperature: f32, top_k: u32, top_p: f32) {
        unsafe {
            const N: usize = SAMPLE_PER;
            const CAP: usize = common::sample::CAP;
            let tid = workitem_id_x();
            let t = tid as usize;
            let mass: *mut f32 = lds();
            let cv = mass.add(256);
            let ci = cv.add(256) as *mut u32;
            let off = ci.add(256);
            let red = off.add(256) as *mut f32;
            let theta0 = red.add(8);
            let t0 = common::stat_clock!();
            // this thread's run of the vocabulary: one workgroup cannot hide VRAM latency
            // by occupancy, so each pass reads it in blocks of 80 floats, twenty loads in flight
            const B: usize = 80;
            let mine = logits.add(t * N) as *const [f32; B];
            let (mut m, mut mi) = (f32::NEG_INFINITY, 0usize);
            let mut i = 0;
            while i < N / B {
                let q = *mine.add(i);
                let mut j = 0;
                while j < B {
                    if q[j] > m {
                        m = q[j];
                        mi = i * B + j;
                    }
                    j += 1;
                }
                i += 1;
            }
            let big = workgroup_max(tid, m, red);
            *cv.add(t) = m;
            *ci.add(t) = (t * N + mi) as u32;
            workgroup_sync();
            if temperature <= 0.0 {
                if t == 0 {
                    let mut best = 0usize;
                    let mut j = 1;
                    while j < 256 {
                        if *cv.add(j) > *cv.add(best) {
                            best = j;
                        }
                        j += 1;
                    }
                    *out = *ci.add(best);
                }
                return;
            }
            let t1 = common::stat_clock!();
            common::stat!(0, (t1.wrapping_sub(t0)) & common::stats::CLOCK_MASK);
            let k = (top_k as usize).clamp(1, CAP);
            // θ₀: the maximum of rank k-1 among the 256 (value desc, index asc)
            let mine_i = (t * N + mi) as u32;
            if rank(cv, ci, 256, m, mine_i) == k - 1 {
                *theta0 = m;
            }
            workgroup_sync();
            let mut theta = *theta0;
            let t2 = common::stat_clock!();
            common::stat!(1, (t2.wrapping_sub(t1)) & common::stats::CLOCK_MASK);
            let count = |theta: f32| {
                let mut c = 0u32;
                let mut i = 0;
                while i < N / B {
                    let q = *mine.add(i);
                    let mut j = 0;
                    while j < B {
                        c += (q[j] >= theta) as u32;
                        j += 1;
                    }
                    i += 1;
                }
                c
            };
            // counts are exact in f32 below 2^24
            let mut c = count(theta);
            let mut total = workgroup_sum(tid, c as f32, red) as usize;
            if total > CAP {
                let (mut lo, mut hi) = (key(theta), key(big));
                while lo < hi {
                    let mid = lo + (hi - lo + 1) / 2;
                    if workgroup_sum(tid, count(unkey(mid)) as f32, red) as usize >= k {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }
                theta = unkey(lo);
                c = count(theta);
                total = workgroup_sum(tid, c as f32, red) as usize;
            }
            *off.add(t) = c;
            workgroup_sync();
            if t == 0 {
                let mut acc = 0u32;
                let mut j = 0;
                while j < 256 {
                    let n = *off.add(j);
                    *off.add(j) = acc;
                    acc += n;
                    j += 1;
                }
            }
            workgroup_sync();
            let n = if total < CAP { total } else { CAP };
            let mut o = *off.add(t) as usize;
            let mut i = 0;
            while i < N / B {
                let q = *mine.add(i);
                let mut j = 0;
                while j < B {
                    if q[j] >= theta {
                        if o < CAP {
                            *cv.add(o) = q[j];
                            *ci.add(o) = (t * N + i * B + j) as u32;
                        }
                        o += 1;
                    }
                    j += 1;
                }
                i += 1;
            }
            workgroup_sync();
            let t3 = common::stat_clock!();
            common::stat!(2, (t3.wrapping_sub(t2)) & common::stats::CLOCK_MASK);
            // order the list: thread t places slot t at its rank (barriers stay uniform:
            // every thread ranks, the threads past the list rank a pair that sorts last)
            let (v, ix) = if t < n { (*cv.add(t), *ci.add(t)) } else { (f32::NEG_INFINITY, u32::MAX) };
            let r = rank(cv, ci, n, v, ix);
            workgroup_sync();
            if t < n {
                *cv.add(r) = v;
                *ci.add(r) = ix;
            }
            workgroup_sync();
            // the k-th of the list and its ties, and their masses at the top
            let theta_k = *cv.add(k - 1);
            let n2 = workgroup_sum(tid, (t < n && *cv.add(t) >= theta_k) as u32 as f32, red) as usize;
            let top = *cv;
            let inv_t = 1.0 / temperature;
            if t < n2 {
                *mass.add(t) = exp((*cv.add(t) - top) * inv_t);
            }
            workgroup_sync();
            if t == 0 {
                let mut total = 0f64;
                let mut i = 0;
                while i < n2 {
                    total += *mass.add(i) as f64;
                    i += 1;
                }
                let lim = top_p as f64 * total;
                let mut above = 0f64;
                let mut keep = 0;
                while keep < n2 && above < lim {
                    above += *mass.add(keep) as f64;
                    keep += 1;
                }
                let target = u * above;
                let mut acc = 0f64;
                let mut i = 0;
                while i < keep {
                    acc += *mass.add(i) as f64;
                    if target < acc {
                        break;
                    }
                    i += 1;
                }
                *out = *ci.add(if i < keep { i } else { keep - 1 });
            }
            common::stat!(3, (common::stat_clock!().wrapping_sub(t3)) & common::stats::CLOCK_MASK);
            const _: () = assert!(VOCAB_SIZE % 256 == 0 && N % B == 0 && CAP == 256);
        }
    }
    /// How many of the first `n` (value, index) pairs come before `(v, i)`:
    /// value descending, index ascending.
    #[inline(always)]
    /// Sixteen pairs per LDS round trip; the slots past `n` exist (256) and are masked.
    unsafe fn rank(cv: *const f32, ci: *const u32, n: usize, v: f32, i: u32) -> usize {
        let mut r = 0;
        let mut j = 0;
        while j < n {
            let (ov, oi) = (*(cv.add(j) as *const [f32; 16]), *(ci.add(j) as *const [u32; 16]));
            let mut q = 0;
            while q < 16 {
                r += (j + q < n && (ov[q] > v || (ov[q] == v && oi[q] < i))) as usize;
                q += 1;
            }
            j += 16;
        }
        r
    }
    /// ln p(id): `common::softmax::log_prob`'s twin. Each thread's max and sum
    /// of exponentials over its SAMPLE_PER logits, joined by thread 0 in f64.
    /// Grid: 1 × 256.
    #[no_mangle]
    pub extern "gpu-kernel" fn score(logits: *const f32, out: *mut f32, id: u32) {
        unsafe {
            const N: usize = SAMPLE_PER;
            const B: usize = 16; // 80 here put the block in scratch: exp inlined that many times is past the unroller
            let tid = workitem_id_x();
            let t = tid as usize;
            let part: *mut f32 = lds();
            let red = part.add(256);
            let mine = logits.add(t * N) as *const [f32; B];
            let mut m = f32::NEG_INFINITY;
            let mut i = 0;
            while i < N / B {
                let q = *mine.add(i);
                let mut j = 0;
                while j < B {
                    if q[j] > m {
                        m = q[j];
                    }
                    j += 1;
                }
                i += 1;
            }
            let big = workgroup_max(tid, m, red);
            let mut s = 0f32;
            let mut i = 0;
            while i < N / B {
                let q = *mine.add(i);
                let mut j = 0;
                while j < B {
                    s += exp(q[j] - big);
                    j += 1;
                }
                i += 1;
            }
            *part.add(t) = s;
            workgroup_sync();
            if t == 0 {
                let mut total = 0f64;
                let mut j = 0;
                while j < 256 {
                    total += *part.add(j) as f64;
                    j += 1;
                }
                *out = *logits.add(id as usize) - big - ln(total as f32);
            }
        }
    }
    /// f32 → u32 preserving order, and back.
    #[inline(always)]
    fn key(v: f32) -> u32 {
        let b = v.to_bits();
        if b >> 31 == 1 { !b } else { b | 0x8000_0000 }
    }
    #[inline(always)]
    fn unkey(k: u32) -> f32 {
        f32::from_bits(if k >> 31 == 1 { k & 0x7FFF_FFFF } else { !k })
    }

    const _: () = assert!(H % 256 == 0 && INTERMEDIATE_SIZE % 8 == 0 && SHARED_FF % 8 == 0 && TOP_K < 8);
}
