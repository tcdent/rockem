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
    /// `run_batch` over activations already quantized (`xq: [n][H]` i8, `xs`/`xg: [n][H/GROUP]`, as `QAct`
    /// makes them): the cards quantize a step's FFN input once for both tiers (queue 001/064). The shipped
    /// INT4×INT8 realization takes this form; the others take `run_batch`.
    fn run_batch_act(&mut self, l: usize, idx: &[[usize; TOP_K]], xq: &[i8], xs: &[f32], xg: &[i32], out: &mut [f32]) {
        let _ = (l, idx, xq, xs, xg, out);
        unimplemented!("run_batch_act: this realization takes f32 activations (run_batch)");
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
    fn run_batch_act(&mut self, l: usize, idx: &[[usize; TOP_K]], xq: &[i8], xs: &[f32], xg: &[i32], out: &mut [f32]) {
        (**self).run_batch_act(l, idx, xq, xs, xg, out)
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
    /// `ffn_act`'s LDS: the token's sigmoid scores and a taken flag an expert, and the norm's scale.
    /// `ffn_act`'s LDS: the token's row (the norm's sum reads it in order from here, not VRAM), the sigmoid
    /// scores, the bias and a taken flag an expert, the norm's scale, the picks twice.
    pub const ACT_LDS: u32 = (H * 4 + E * 12 + 16 + 2 * TOP_K * 4) as u32;
    const _: () = assert!(E % 32 == 0 && E / 32 <= 32, "ffn_act: an expert a lane slot");
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

    /// The FFN input's host-side work, on the card (queue 001, 064): for token `t` (a workgroup of 256),
    /// the post-norm of `x[t]` with `ln` exactly as the host's `rmsnorm8` (eight lanes sum the squares of
    /// their stride, the eight added in order), its groups quantized exactly as `QAct::quantize` (a wave a group) into the card's own
    /// activation buffers (`xq`, `xs`, `xg`: the resident experts read them) and the host's copies
    /// (`xq_h`, `xs_h`, `xg_h`), and the router's picks and weights from its logits in `scores[t]` exactly
    /// as `ffn::route_scores` (the sigmoid an expert a thread into LDS, then one lane: the top-k with the
    /// bias, the mass in id order, the ranked picks past ROUTED_K dropped) into `picks` and `wts` (host).
    /// The last workgroup to finish writes `value` to `word`. Grid: n workgroups, LDS ACT_LDS.
    #[no_mangle]
    pub extern "gpu-kernel" fn ffn_act(x: *const f32, ln: *const f32, bias: *const f32, scores: *const f32, xq: *mut i8, xs: *mut f32, xg: *mut i32, xq_h: *mut i8, xs_h: *mut f32, xg_h: *mut i32, picks: *mut u32, wts: *mut f32, count: *mut u32, word: *mut u32, value: u32, n: u32) {
        use common::q4::GROUP;
        use core::arch::amdgpu::{wave_reduce_add, wave_reduce_umax, workgroup_id_x};
        use core::sync::atomic::{fence, AtomicU32, Ordering};
        // stats (a stats code object only): slot 0 the loads to the first barrier (the row, the sigmoids over the
        // host-memory scores, the bias), 1 the norm's sum (wave 0; the others wait), 2 the wave's quantize (its
        // stores to VRAM and host memory issued), 3 the top-k (wave 0) and the release fence's drain of the stores
        unsafe {
            let k0 = common::stat_clock!();
            let t = workgroup_id_x() as usize;
            let tid = workitem_id_x();
            let (wave, lane) = ((tid / 32) as usize, tid as usize % 32);
            let xt = x.add(t * H);
            let l: *mut f32 = common::gfx11::lds();
            let row = l; // [H] the token's row
            let sig = l.add(H); // [E] the sigmoid scores
            let bs = sig.add(E); // [E] the selection bias
            let taken = bs.add(E) as *mut u32; // [E]
            let rr = bs.add(2 * E); // [1] the norm's scale
            let ranked = rr.add(4) as *mut u32; // [TOP_K] the picks in the order taken (LDS: a lane's array indexed at run time would be scratch)
            let idx = ranked.add(TOP_K); // [TOP_K] the picks in id order
            // the row into LDS (coalesced), the sigmoids and the bias a thread an expert (E past a workgroup's 256: strided)
            let mut i = tid as usize;
            while i < H {
                *row.add(i) = *xt.add(i);
                i += 256;
            }
            let mut e = tid as usize;
            while e < E {
                *sig.add(e) = crate::hc::sigmoid(*scores.add(t * crate::hc::gfx11_consts::ROUTE_ROWS + e));
                *bs.add(e) = *bias.add(e);
                *taken.add(e) = 0;
                e += 256;
            }
            common::gfx11::workgroup_sync();
            let k1 = common::stat_clock!();
            common::stat!(0, k1.wrapping_sub(k0) & common::stats::CLOCK_MASK);
            // the norm's scale as `rmsnorm8`: lane j of wave 0 sums the squares of x[8i + j] in order, lane 0 adds the
            // eight in order — the host's `sumsq8` exactly, on eight chains
            if wave == 0 && lane < 8 {
                let mut a = 0f32;
                let mut i = lane;
                while i < H {
                    let v = *row.add(i);
                    a += v * v;
                    i += 8;
                }
                let mut ss = 0f32;
                let mut j = 0;
                while j < 8 {
                    ss += common::gfx11::lane_read(j, a);
                    j += 1;
                }
                if lane == 0 {
                    *rr = common::math::rsqrt(ss / H as f32 + RMS_NORM_EPS);
                }
            }
            common::gfx11::workgroup_sync();
            let k2 = common::stat_clock!();
            common::stat!(1, k2.wrapping_sub(k1) & common::stats::CLOCK_MASK);
            let r = *rr;
            // the groups: wave w takes groups w, w + 8, ...: `ln[i] · (x[i] · r)`, then absmax/127, the shared round, the clamp, Σq
            let mut g = wave;
            while g < H / GROUP {
                let k0 = g * GROUP + lane * 4;
                let mut v = [0f32; 4];
                let mut amax = 0f32;
                let mut jj = 0;
                while jj < 4 {
                    v[jj] = *ln.add(k0 + jj) * (*row.add(k0 + jj) * r);
                    let a = if v[jj] < 0.0 { -v[jj] } else { v[jj] };
                    if a > amax {
                        amax = a;
                    }
                    jj += 1;
                }
                let amax = f32::from_bits(wave_reduce_umax::<0>(amax.to_bits()));
                let sc = amax / 127.0;
                let inv = if sc > 0.0 { 1.0 / sc } else { 0.0 };
                let mut packed = 0u32;
                let mut sum = 0i32;
                let mut jj = 0;
                while jj < 4 {
                    let rd = common::math::round(v[jj] * inv);
                    let rd = if rd > 127.0 { 127.0 } else if rd < -127.0 { -127.0 } else { rd };
                    let qv = rd as i8;
                    packed |= ((qv as u8) as u32) << (8 * jj);
                    sum += qv as i32;
                    jj += 1;
                }
                let sum = wave_reduce_add::<0>(sum as u32) as i32;
                *(xq.add(t * H + k0) as *mut u32) = packed;
                *(xq_h.add(t * H + k0) as *mut u32) = packed;
                if lane == 0 {
                    *xs.add(t * (H / GROUP) + g) = sc;
                    *xg.add(t * (H / GROUP) + g) = sum;
                    *xs_h.add(t * (H / GROUP) + g) = sc;
                    *xg_h.add(t * (H / GROUP) + g) = sum;
                }
                g += 8;
            }
            let k3 = common::stat_clock!();
            common::stat!(2, k3.wrapping_sub(k2) & common::stats::CLOCK_MASK);
            // the router's picks, `route_scores`'s arithmetic on wave 0: expert e sits on lane e % 32, slot e / 32,
            // its key `score + bias` as the host adds it; a pass takes each lane's best untaken key, then the wave's by
            // a butterfly that keeps the greater key and, between equal keys, the lower expert — the host's strict
            // `>` scanning e upward. Then one lane: the mass over the picks in id order, the ranked picks past
            // ROUTED_K dropped, the weights.
            if wave == 0 {
                const SLOTS: usize = E / 32;
                let mut key = [0f32; SLOTS];
                let mut taken_mask = 0u32;
                let mut sl = 0;
                while sl < SLOTS {
                    let e = sl * 32 + lane;
                    key[sl] = *sig.add(e) + *bs.add(e);
                    sl += 1;
                }
                let mut k = 0;
                while k < TOP_K {
                    let (mut bv, mut be) = (f32::NEG_INFINITY, E as u32);
                    let mut sl = 0;
                    while sl < SLOTS {
                        if taken_mask & (1 << sl) == 0 {
                            let e = (sl * 32 + lane) as u32;
                            if key[sl] > bv || (key[sl] == bv && e < be) {
                                bv = key[sl];
                                be = e;
                            }
                        }
                        sl += 1;
                    }
                    let mut m = 16u32;
                    while m >= 1 {
                        let ov = common::gfx11::lane_read(lane as u32 ^ m, bv);
                        let oe = common::gfx11::lane_read(lane as u32 ^ m, f32::from_bits(be)).to_bits();
                        if ov > bv || (ov == bv && oe < be) {
                            bv = ov;
                            be = oe;
                        }
                        m >>= 1;
                    }
                    // every lane holds the pass's pick; its owner marks it taken
                    if be as usize % 32 == lane {
                        taken_mask |= 1 << (be as usize / 32);
                    }
                    if lane == 0 {
                        *ranked.add(k) = be;
                        *idx.add(k) = be;
                    }
                    k += 1;
                }
                if lane == 0 {
                    sort8(idx);
                    let mut mass = 1e-20f32;
                    let mut k = 0;
                    while k < TOP_K {
                        mass += *sig.add(*idx.add(k) as usize);
                        k += 1;
                    }
                    let mut k = ROUTED_K;
                    while k < TOP_K {
                        let e = *ranked.add(k);
                        let mut j = 0;
                        while j < TOP_K {
                            if *idx.add(j) == e {
                                *idx.add(j) = E as u32;
                            }
                            j += 1;
                        }
                        k += 1;
                    }
                    sort8(idx);
                    let mut k = 0;
                    while k < TOP_K {
                        let e = *idx.add(k) as usize;
                        *picks.add(t * TOP_K + k) = e as u32;
                        *wts.add(t * TOP_K + k) = if e < E { *sig.add(e) / mass * ROUTED_SCALING_FACTOR } else { 0.0 };
                        k += 1;
                    }
                }
            }
            // the last workgroup publishes
            fence(Ordering::Release);
            let k4 = common::stat_clock!();
            common::stat!(3, k4.wrapping_sub(k3) & common::stats::CLOCK_MASK);
            common::gfx11::workgroup_sync();
            if tid == 0 {
                let prev = AtomicU32::from_ptr(count).fetch_add(1, Ordering::AcqRel);
                if prev + 1 == n {
                    AtomicU32::from_ptr(count).store(0, Ordering::Relaxed);
                    AtomicU32::from_ptr(word).store(value, Ordering::Release);
                }
            }
        }
    }
    /// `TOP_K` indices in LDS sorted ascending (`sort_unstable` on the host: the order is the result, not the algorithm).
    #[inline(always)]
    unsafe fn sort8(a: *mut u32) {
        let mut i = 1;
        while i < TOP_K {
            let v = *a.add(i);
            let mut j = i;
            while j > 0 && *a.add(j - 1) > v {
                *a.add(j) = *a.add(j - 1);
                j -= 1;
            }
            *a.add(j) = v;
            i += 1;
        }
    }

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
