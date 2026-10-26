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

/// The read-out alone, on the memory as it stands: the query projected and run through the convolution against the
/// window (neither updated), l2-normed and scaled, `S·q` a head, the gated norm, the output projection. No forget
/// gate, no delta update — what layer l + 1's attention would say about an input before the token's own update, the
/// expert predictor's attention-ahead (2026-10-22). `s.mixed[..KDA_QKV]`, `s.conv[..KDA_QKV]`, `s.ga`, `s.gate`,
/// `s.normed` and `s.heads` are used.
pub fn readout(w: &weights::Kda, xn: &[f32], window: &[f32], mem: &[f32], s: &mut Scratch, pool: &dyn Pool, out: &mut [f32]) {
    gemv::bf16_par::<KDA_QKV, H>(w.q, xn, &mut s.mixed[..KDA_QKV], pool);
    let taps = w.conv[0];
    for c in 0..KDA_QKV {
        let mut acc = 0f32;
        for j in 0..CONV_TAPS - 1 {
            acc += gemv::widen(taps[c * CONV_TAPS + j]) * window[j * CONV_DIM + c];
        }
        acc += gemv::widen(taps[c * CONV_TAPS + CONV_TAPS - 1]) * s.mixed[c];
        s.conv[c] = silu(acc);
    }
    gemv::bf16::<KDA_DIM, H>(w.g_a, xn, s.ga);
    gemv::bf16_par::<KDA_QKV, KDA_DIM>(w.g_b, s.ga, s.gate, pool);
    let (conv, gate) = (&*s.conv, &*s.gate);
    let (normed_p, heads_p) = (s.normed.as_mut_ptr() as usize, s.heads.as_mut_ptr() as usize);
    let scale = 1.0 / sqrt(KDA_DIM as f32);
    pool.run(KDA_HEADS, &move |h| {
        let m = &mem[h * KDA_DIM * KDA_DIM..(h + 1) * KDA_DIM * KDA_DIM];
        let normed = unsafe { core::slice::from_raw_parts_mut((normed_p as *mut f32).add(h * KDA_DIM), KDA_DIM) };
        let hs = unsafe { core::slice::from_raw_parts_mut((heads_p as *mut f32).add(h * HEAD_SCRATCH), HEAD_SCRATCH) };
        let (qh, rest) = hs.split_at_mut(KDA_DIM);
        let (o, on) = rest.split_at_mut(KDA_DIM);
        let on = &mut on[..KDA_DIM];
        qh.copy_from_slice(&conv[h * KDA_DIM..(h + 1) * KDA_DIM]);
        l2norm(qh);
        for vv in 0..KDA_DIM {
            o[vv] = 0.0;
        }
        for kk in 0..KDA_DIM {
            let qk = qh[kk] * scale;
            let row = &m[kk * KDA_DIM..(kk + 1) * KDA_DIM];
            for vv in 0..KDA_DIM {
                o[vv] += row[vv] * qk;
            }
        }
        rmsnorm::<KDA_DIM>(o, &w.o_norm, RMS_NORM_EPS, on);
        for vv in 0..KDA_DIM {
            normed[vv] = on[vv] * sigmoid(gate[h * KDA_DIM + vv]);
        }
    });
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

/// Launch geometry of the gfx11 realization, visible to hosts that launch it.
pub mod gfx11_consts {
    use crate::*;
    pub const WG: u32 = 256;
    /// Rows a wave of the exp4 rows kernels at a verify width (`rows_h_e4`, `rows_d2_e4`, `rows_part_e4`; one token
    /// takes one): the sizing sweep of 2026-10-25 (`bench-exp4 --sweep`, node order, 4 tokens) against 4 — `sq` 1.06×,
    /// `o` 1.07× (with 4 steps loaded ahead), `fg` 1.16×; 1 lost to both (0.72–0.93×). At 2 the grids are 788 / 256 / 512
    /// workgroups, past the 84 CUs, so no narrow twin is needed.
    pub const E4_ROWS: usize = 2;
    /// Waves a workgroup of the exp4 rows kernels past one token (`exp4_rows_direct`: no LDS, no barrier, so a workgroup
    /// is only the unit the card places).
    pub const E4_WAVES: usize = 8;
    /// The exp4 rows kernels' launch: (grid, workgroup size) for `m` rows at `n` tokens.
    pub const fn e4_launch(m: usize, n: usize) -> (u32, u32) {
        if n > 1 { (m.div_ceil(E4_WAVES * E4_ROWS) as u32, 32 * E4_WAVES as u32) } else { (rows_grid(m, n, 1), WG) }
    }
    /// The rows kernels: 8 rows per workgroup.
    /// Workgroups of a rows launch over `m` rows for `n` tokens, the kernel's `r` rows a wave
    /// (`common::gemv::rows_per_wg`).
    /// The matrix-path rows kernels (`rows_wmma`): their grid and LDS.
    pub const WMMA_TOKENS: usize = 128;
    pub const WMMA_ROWS: usize = 128;
    pub const WMMA_LDS: u32 = (WMMA_TOKENS * (64 + 8) * 2 + WMMA_TOKENS * 4) as u32;
    pub const fn wmma_grid(m: usize, n: usize) -> u32 {
        (m.div_ceil(WMMA_ROWS) * n.div_ceil(WMMA_TOKENS)) as u32
    }
    pub const fn rows_grid(m: usize, n: usize, r: usize) -> u32 {
        m.div_ceil(common::gemv::rows_per_wg(n, r)) as u32
    }
    /// LDS a rows kernel over `[m][k]` needs for a step of `n` tokens (`common::gemv::gfx11`).
    pub const fn rows_lds(k: usize, n: usize) -> usize {
        let steps = k.div_ceil(256);
        let flight = common::gemv::FLIGHT;
        let slots = if n <= 1 { if steps < flight { steps } else { flight } } else { 2 * 8 };
        4 * (slots * common::gemv::STEP + 8)
    }
    /// A rows kernel that norms its input: the slab, then 8 wave partials per token of the pass.
    pub const fn normed_rows_lds(k: usize, n: usize) -> usize {
        rows_lds(k, n) + 4 * 8 * common::gemv::PASS
    }
    /// The exp4 rows kernels (`rows_h_e4`, `rows_d2_e4`, `rows_part_e4`): one token is the staged pass (its slab, and
    /// the norm's partials with `norm`); a few tokens stage nothing (`exp4_rows_direct`), so no LDS, which would only
    /// hold workgroups off the CUs.
    pub const fn e4_rows_lds(k: usize, n: usize, norm: bool) -> usize {
        if n > 1 { 0 } else if norm { normed_rows_lds(k, n) } else { rows_lds(k, n) }
    }
    /// The FP8 rows kernels: the same, plus their decode table.
    pub const fn fp8_rows_lds(k: usize, n: usize) -> usize {
        normed_rows_lds(k, n) + 4 * common::gemv::gfx11_lds::FP8_LDS
    }
    /// `kda_step`: a workgroup per head of its part.
    pub const STEP_GRID: u32 = (KDA_HEADS / PARTS) as u32;
    /// A KDA layer splits by heads into this many parts, one card each where there are
    /// two: each part projects its heads' q, k, v and gates, steps its heads, and takes its
    /// heads' half of every output projection row; the host sums the parts' outputs.
    pub const PARTS: usize = 2;
    /// A part's heads and its share of the q, k, v (and normed) width.
    pub const PART_HEADS: usize = KDA_HEADS / PARTS;
    pub const PART_QKV: usize = KDA_QKV / PARTS;
    /// `copy_f32`: words a workgroup copies.
    pub const COPY_WG_WORDS: usize = 256 * 16;
    /// `kda_step`'s LDS, in floats: the head's conv outputs (q | k | v), the forget
    /// factors, two partials per value channel, and the reduction scratch.
    pub const CONV: usize = 0;
    pub const FORGET: usize = CONV + 3 * KDA_DIM;
    pub const EX: usize = FORGET + KDA_DIM;
    pub const RED: usize = EX + 2 * KDA_DIM;
    pub const STEP_LDS: usize = 4 * (RED + 16);
    /// The three convolutions' taps as one `[3][KDA_QKV][CONV_TAPS]` bf16 block: q's, k's, v's.
    pub const TAPS: usize = 3 * KDA_QKV * CONV_TAPS;
}

#[cfg(target_arch = "amdgpu")]
pub mod gfx11 {
    //! The GPU realization of the KDA layer, for `n` tokens of a step at once:
    //! `rows_h` (rows of a `[m][H]` bf16 matrix over `n` activations, each
    //! rms-normed with the layer's input weight as it is staged — the q, k, v
    //! projections, the forget and output gates' first halves, the beta rows),
    //! `rows_d2` (`[m][KDA_DIM]` rows over two inputs: the gates' second halves), `kda_step` (a
    //! workgroup per head: the short convolution, the l2 norms, the bounded
    //! forget gate, the delta rule over the head's memory held in registers
    //! across the step's tokens, the gated norm) and `rows_qkv` (`[m][KDA_QKV]`
    //! rows: the output projection). One token is n = 1 through the same kernels.
    pub use super::gfx11_consts::*;
    use crate::*;
    use common::gemv::gfx11::{exp4_rows_direct, rows_fmt_one, rows_over, Exp4};
    use common::gemv::{widen, ROWS};
    use common::gfx11::{lds, wave_sum, workgroup_sync};
    use common::math::{exp, rsqrt};
    use common::swiglu::silu;
    use core::arch::amdgpu::{workgroup_id_x, workitem_id_x};
    const _: () = assert!(TOKENS == common::gemv::gfx11::TOKENS && PART_QKV % (8 * ROWS) == 0);

    /// `out[i][row] = W[row] · rmsnorm_ln(x[i])` for `m` rows of a `[m][H]`
    /// bf16 matrix and `n` activations `x` (`[n][H]`). Grid: rows_grid(m).
    #[no_mangle]
    pub extern "gpu-kernel" fn rows_h(x: *const f32, ln: *const u16, w: *const u16, out: *mut f32, m: u32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            rows_over::<H, true, ROWS>(tid, w, ln, m, n, RMS_NORM_EPS, |i, s| *x.add(i * H + s * 256 + t), || {}, |row, i, y| *out.add(i * m + row) = y);
        }
    }

    /// `rows_h` one row a wave (060): the same launch where the wide grid would be under-occupied —
    /// a small `m` at a verify width leaves the card a couple of workgroups at `ROWS` 4, and the
    /// waves, not the rows, stream the weights. The host picks it when `rows_grid(m, n, ROWS)` is
    /// under ~2 workgroups a CU; the rows' arithmetic is unchanged, only their grouping.
    #[no_mangle]
    pub extern "gpu-kernel" fn rows_h1(x: *const f32, ln: *const u16, w: *const u16, out: *mut f32, m: u32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            rows_over::<H, true, 1>(tid, w, ln, m, n, RMS_NORM_EPS, |i, s| *x.add(i * H + s * 256 + t), || {}, |row, i, y| *out.add(i * m + row) = y);
        }
    }


    /// `rows_h` on the matrix path for a step of many tokens (`rows_wmma`). Grid: wmma_grid(m, n), LDS WMMA_LDS.
    #[no_mangle]
    pub extern "gpu-kernel" fn rows_h_w(x: *const f32, ln: *const u16, w: *const u16, out: *mut f32, m: u32, n: u32) {
        unsafe {
            common::gemv::gfx11::rows_wmma::<H, true>(workitem_id_x(), w, ln, m as usize, n as usize, RMS_NORM_EPS, x, out);
        }
    }
    /// `rows_part` on the matrix path for a step of many tokens (`rows_wmma`). Grid: wmma_grid(m, n), LDS WMMA_LDS.
    #[no_mangle]
    pub extern "gpu-kernel" fn rows_part_w(x: *const f32, w: *const u16, out: *mut f32, m: u32, n: u32) {
        unsafe {
            common::gemv::gfx11::rows_wmma::<PART_QKV, false>(workitem_id_x(), w, core::ptr::null(), m as usize, n as usize, 0.0, x, out);
        }
    }

    /// `out[i][row] = W[row] · x[i]` for a `[m][KDA_QKV]` bf16 matrix over `x: [n][KDA_QKV]`: the output projection. Grid: rows_grid(m).
    #[no_mangle]
    pub extern "gpu-kernel" fn rows_qkv(x: *const f32, w: *const u16, out: *mut f32, m: u32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            rows_over::<KDA_QKV, false, ROWS>(tid, w, core::ptr::null(), m, n, 0.0, |i, s| *x.add(i * KDA_QKV + s * 256 + t), || {}, |row, i, y| *out.add(i * m + row) = y);
        }
    }

    /// `rows_d` over two inputs: rows below `split` read `x[0..KDA_DIM]`, rows from `split` read
    /// `x[KDA_DIM..2·KDA_DIM]` — f_b over f_a's output and g_b over g_a's as one launch (the
    /// weights and the outputs back to back, `split` a multiple of a workgroup's rows); token
    /// `i`'s input at `x[i · xs]` (the stride of the launch that made it).
    #[no_mangle]
    pub extern "gpu-kernel" fn rows_d2(x: *const f32, w: *const u16, out: *mut f32, m: u32, n: u32, split: u32, xs: u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n, xs) = (m as usize, n as usize, xs as usize);
            let t = tid as usize;
            let x = if core::arch::amdgpu::workgroup_id_x() as usize * common::gemv::rows_per_wg(n, ROWS) >= split as usize { x.add(KDA_DIM) } else { x };
            rows_over::<KDA_DIM, false, ROWS>(tid, w, core::ptr::null(), m, n, 0.0, |i, s| if s * 256 + t < KDA_DIM { *x.add(i * xs + s * 256 + t) } else { 0.0 }, || {}, |row, i, y| *out.add(i * m + row) = y);
        }
    }

    /// `out[i][row] = W[row] · x[i]` for a `[m][PART_QKV]` bf16 matrix over `x: [n][PART_QKV]`: a
    /// part's half of the output projection, over its heads' normed readouts. Grid: rows_grid(m).
    #[no_mangle]
    pub extern "gpu-kernel" fn rows_part(x: *const f32, w: *const u16, out: *mut f32, m: u32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            rows_over::<PART_QKV, false, ROWS>(tid, w, core::ptr::null(), m, n, 0.0, |i, s| *x.add(i * PART_QKV + s * 256 + t), || {}, |row, i, y| *out.add(i * m + row) = y);
        }
    }

    /// `rows_part` one row a wave (060): the output projection's narrow twin, picked where the wide
    /// grid would be under-occupied (m H at a verify width is 128 workgroups on 84 CUs).
    #[no_mangle]
    pub extern "gpu-kernel" fn rows_part1(x: *const f32, w: *const u16, out: *mut f32, m: u32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            rows_over::<PART_QKV, false, 1>(tid, w, core::ptr::null(), m, n, 0.0, |i, s| *x.add(i * PART_QKV + s * 256 + t), || {}, |row, i, y| *out.add(i * m + row) = y);
        }
    }

    /// An exp4 `[m][k]` matrix at `w` unpacked to bf16 at `out` (029): a prefill step's KDA matrices, once a step, for the
    /// matrix path's bf16 kernels (`rows_h_w`, `rows_part_w`) and the gates' `rows_d2` — their bits, since the bf16 is
    /// the original exactly. A thread a chunk of eight weights. Grid: (m · k / 8).div_ceil(256).
    #[no_mangle]
    pub extern "gpu-kernel" fn exp4_unpack(w: *const u8, out: *mut u16, m: u32, k: u32) {
        unsafe {
            let (m, k) = (m as usize, k as usize);
            let c = workgroup_id_x() as usize * 256 + workitem_id_x() as usize;
            if c < m * k / 8 {
                common::gemv::gfx11::exp4_unpack_chunk(w as *const u8, m, k, c, out.add(c * 8));
            }
        }
    }

    /// The exp4 twins of the decode path's KDA rows kernels (029): the same rows over the lossless 12-bit form of the
    /// same matrices (`common::exp4`, its planes at `w`), the bf16 kernels' arguments and their bits. Decode and verify
    /// steps only (at most `FEW` tokens); a prefill step unpacks to bf16 and takes the matrix path. One token is the
    /// staged pass (`rows_fmt_one`, eight steps loaded ahead); two to four stage nothing (`exp4_rows_direct`: each wave
    /// reads its activations from L2 and runs its steps with no barrier, at the step's own width — half of the verify
    /// steps are two tokens), `E4_ROWS` rows a wave, two steps loaded ahead. Launch: `e4_launch`, LDS `e4_rows_lds`.
    /// Node order against bf16 (bench-exp4 --node 2026-10-25, µs a layer of sq + fg + o): 1 token 166.4 / 204.1,
    /// 2 tokens 176.4 / 220.6, 3 182.5 / 221.7, 4 189.8 / 223.3.
    #[no_mangle]
    pub extern "gpu-kernel" fn rows_h_e4(x: *const f32, ln: *const u16, w: *const u8, out: *mut f32, m: u32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            let emit = |row: usize, i: usize, y: f32| *out.add(i * m + row) = y;
            if n <= 1 {
                rows_fmt_one::<H, true, Exp4<2, 8>>(tid, Exp4::<2, 8>::at::<H>(w, m), ln, m, n, RMS_NORM_EPS, |i, s| *x.add(i * H + s * 256 + t), emit);
            } else {
                // the step's own width: half of the verify steps are two tokens, a quarter three
                match n {
                    2 => exp4_rows_direct::<H, true, 2, 8, E4_ROWS, E4_WAVES, 2>(tid, Exp4::<2, 8>::at::<H>(w, m), ln, m, n, RMS_NORM_EPS, |_, i| x.add(i * H), emit),
                    3 => exp4_rows_direct::<H, true, 2, 8, E4_ROWS, E4_WAVES, 3>(tid, Exp4::<2, 8>::at::<H>(w, m), ln, m, n, RMS_NORM_EPS, |_, i| x.add(i * H), emit),
                    _ => exp4_rows_direct::<H, true, 2, 8, E4_ROWS, E4_WAVES, { common::gemv::FEW }>(tid, Exp4::<2, 8>::at::<H>(w, m), ln, m, n, RMS_NORM_EPS, |_, i| x.add(i * H), emit),
                }
            }
        }
    }
    #[no_mangle]
    pub extern "gpu-kernel" fn rows_d2_e4(x: *const f32, w: *const u8, out: *mut f32, m: u32, n: u32, split: u32, xs: u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n, xs) = (m as usize, n as usize, xs as usize);
            let t = tid as usize;
            let emit = |row: usize, i: usize, y: f32| *out.add(i * m + row) = y;
            if n <= 1 {
                // a workgroup's rows are on one side of `split` (a multiple of a workgroup's rows)
                let x = if core::arch::amdgpu::workgroup_id_x() as usize * common::gemv::rows_per_wg(n, 1) >= split as usize { x.add(KDA_DIM) } else { x };
                rows_fmt_one::<KDA_DIM, false, Exp4<2, 8>>(tid, Exp4::<2, 8>::at::<KDA_DIM>(w, m), core::ptr::null(), m, n, 0.0, |i, s| if s * 256 + t < KDA_DIM { *x.add(i * xs + s * 256 + t) } else { 0.0 }, emit);
            } else {
                // a wave's rows are on one side of `split` (a multiple of E4_ROWS)
                let split = split as usize;
                // the step's own width: half of the verify steps are two tokens, a quarter three
                match n {
                    2 => exp4_rows_direct::<KDA_DIM, false, 2, 8, E4_ROWS, E4_WAVES, 2>(tid, Exp4::<2, 8>::at::<KDA_DIM>(w, m), core::ptr::null(), m, n, 0.0, |row, i| if row >= split { x.add(KDA_DIM + i * xs) } else { x.add(i * xs) }, emit),
                    3 => exp4_rows_direct::<KDA_DIM, false, 2, 8, E4_ROWS, E4_WAVES, 3>(tid, Exp4::<2, 8>::at::<KDA_DIM>(w, m), core::ptr::null(), m, n, 0.0, |row, i| if row >= split { x.add(KDA_DIM + i * xs) } else { x.add(i * xs) }, emit),
                    _ => exp4_rows_direct::<KDA_DIM, false, 2, 8, E4_ROWS, E4_WAVES, { common::gemv::FEW }>(tid, Exp4::<2, 8>::at::<KDA_DIM>(w, m), core::ptr::null(), m, n, 0.0, |row, i| if row >= split { x.add(KDA_DIM + i * xs) } else { x.add(i * xs) }, emit),
                }
            }
        }
    }
    #[no_mangle]
    pub extern "gpu-kernel" fn rows_part_e4(x: *const f32, w: *const u8, out: *mut f32, m: u32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            let emit = |row: usize, i: usize, y: f32| *out.add(i * m + row) = y;
            if n <= 1 {
                rows_fmt_one::<PART_QKV, false, Exp4<2, 8>>(tid, Exp4::<2, 8>::at::<PART_QKV>(w, m), core::ptr::null(), m, n, 0.0, |i, s| *x.add(i * PART_QKV + s * 256 + t), emit);
            } else {
                // the step's own width: half of the verify steps are two tokens, a quarter three
                match n {
                    2 => exp4_rows_direct::<PART_QKV, false, 2, 8, E4_ROWS, E4_WAVES, 2>(tid, Exp4::<2, 8>::at::<PART_QKV>(w, m), core::ptr::null(), m, n, 0.0, |_, i| x.add(i * PART_QKV), emit),
                    3 => exp4_rows_direct::<PART_QKV, false, 2, 8, E4_ROWS, E4_WAVES, 3>(tid, Exp4::<2, 8>::at::<PART_QKV>(w, m), core::ptr::null(), m, n, 0.0, |_, i| x.add(i * PART_QKV), emit),
                    _ => exp4_rows_direct::<PART_QKV, false, 2, 8, E4_ROWS, E4_WAVES, { common::gemv::FEW }>(tid, Exp4::<2, 8>::at::<PART_QKV>(w, m), core::ptr::null(), m, n, 0.0, |_, i| x.add(i * PART_QKV), emit),
                }
            }
        }
    }

    /// `dst[..n] = src[..n]`: a layer's state out to host memory for a snapshot (the host
    /// reads its BAR at 10–30 MB/s). Grid: n.div_ceil(COPY_WG_WORDS).
    #[no_mangle]
    pub extern "gpu-kernel" fn copy_f32(src: *const f32, dst: *mut f32, n: u32) {
        unsafe {
            let base = workgroup_id_x() as usize * COPY_WG_WORDS + workitem_id_x() as usize;
            let mut s = 0;
            while s < COPY_WG_WORDS / 256 {
                let i = base + s * 256;
                if i < n as usize {
                    *dst.add(i) = *src.add(i);
                }
                s += 1;
            }
        }
    }

    #[inline(always)]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + exp(-x))
    }

    /// One head of the layer through the step's `n` tokens, in order. Inputs
    /// per token i, the part's own: `mixed[i · ms]` (`[3][PART_QKV]`: the q, k, v
    /// projections of the part's heads), `f[i · fs]` and `gate[i · fs]` (`[PART_QKV]`);
    /// the layer's `beta[i · bs]` (`[KDA_HEADS]`); the
    /// layer's `taps` (`[3][KDA_QKV][CONV_TAPS]` bf16), `dt_bias`, `a_log`,
    /// `o_norm`; its state, `window` (`[CONV_TAPS-1][CONV_DIM]`) and `mem`
    /// (`[KDA_HEADS][KDA_DIM][KDA_DIM]`, the layer's, by head), advanced by `n`
    /// positions; output `normed[i]` (`[PART_QKV]`, the part's), the
    /// gated-normed readouts. Thread `t` holds
    /// column `t % KDA_DIM` of the memory over rows `(t / KDA_DIM) · 64 ..+64`,
    /// so a sweep's reduction over rows is 64 products in registers and one
    /// exchange of two partials per column. The sum order differs from the
    /// CPU's (two halves, then their sum), as the card's stream already does.
    /// With `ring` and `ring_w` not null (a speculative verify step), the state after each token `i`
    /// is also written to `ring[i]` (`[SPEC_TOKENS][KDA_HEADS][KDA_DIM][KDA_DIM]`, this part's heads'
    /// entries) and `ring_w[i]` (`[SPEC_TOKENS][CONV_TAPS - 1][CONV_DIM]`), so the state after the
    /// last accepted token is a copy away (`kda_rewind`).
    /// Grid: STEP_GRID, over heads `head0..`; LDS: STEP_LDS.
    #[no_mangle]
    pub extern "gpu-kernel" fn kda_step(mixed: *const f32, taps: *const u16, window: *mut f32, f: *const f32, dt_bias: *const f32, a_log: *const f32, beta: *const f32, gate: *const f32, mem: *mut f32, o_norm: *const f32, normed: *mut f32, head0: u32, n: u32, ms: u32, fs: u32, bs: u32, ring: *mut f32, ring_w: *mut f32) {
        unsafe {
            // token i's inputs: `mixed[i · ms]`, `f[i · fs]`, `gate[i · fs]`, `beta[i · bs]` — the strides of the launches that made them
            let (ms, fs, bs) = (ms as usize, fs as usize, bs as usize);
            let hl = workgroup_id_x() as usize; // the head within the part: its buffers' index
            let h = head0 as usize + hl; // the head within the layer: the state's and the small tensors' index
            let tid = workitem_id_x();
            let t = tid as usize;
            let lane = tid % 32;
            let l: *mut f32 = lds();
            let (conv, e, ex, red) = (l.add(CONV), l.add(FORGET), l.add(EX), l.add(RED));
            const D: usize = KDA_DIM;
            const HALF: usize = D / 2;
            let vv = t % D;
            let kk0 = (t / D) * HALF;
            let decay = exp(*a_log.add(h));
            let scale = rsqrt(D as f32);
            // the head's memory: this thread's column over its half of the rows
            let mut s = [0f32; HALF];
            let mut r = 0;
            while r < HALF {
                s[r] = *mem.add(h * D * D + (kk0 + r) * D + vv);
                r += 1;
            }
            let mut i = 0;
            while i < n as usize {
                // the convolution for this head's 3·D channels (two per thread), the window advanced
                let mut c = t;
                while c < 3 * D {
                    let part = c / D;
                    let g = part * KDA_QKV + h * D + c % D;
                    let tp = taps.add((part * KDA_QKV + h * D + c % D) * CONV_TAPS);
                    let (w0, w1, w2) = (*window.add(g), *window.add(CONV_DIM + g), *window.add(2 * CONV_DIM + g));
                    let x = *mixed.add(i * ms + part * PART_QKV + hl * D + c % D);
                    let acc = widen(*tp) * w0 + widen(*tp.add(1)) * w1 + widen(*tp.add(2)) * w2 + widen(*tp.add(3)) * x;
                    *conv.add(c) = silu(acc);
                    *window.add(g) = w1;
                    *window.add(CONV_DIM + g) = w2;
                    *window.add(2 * CONV_DIM + g) = x;
                    if !ring_w.is_null() {
                        let rw = ring_w.add(i * (CONV_TAPS - 1) * CONV_DIM);
                        *rw.add(g) = w1;
                        *rw.add(CONV_DIM + g) = w2;
                        *rw.add(2 * CONV_DIM + g) = x;
                    }
                    c += 256;
                }
                // the forget factors: log-decay in (LOWER, 0) per key channel
                if t < D {
                    *e.add(t) = exp(LINEAR_GATE_LOWER_BOUND * sigmoid(decay * (*f.add(i * fs + hl * D + t) + *dt_bias.add(h * D + t))));
                }
                workgroup_sync();
                // l2-normed q (wave 0) and k (wave 1): four channels a lane
                if t < 64 {
                    let base = (t / 32) * D + lane as usize * 4;
                    let mut ss = 0f32;
                    let mut j = 0;
                    while j < 4 {
                        let v = *conv.add(base + j);
                        ss += v * v;
                        j += 1;
                    }
                    let rn = 1.0 / common::math::sqrt(wave_sum(lane, ss) + 1e-6);
                    let rn = if t < 32 { rn * scale } else { rn };
                    let mut j = 0;
                    while j < 4 {
                        *conv.add(base + j) *= rn;
                        j += 1;
                    }
                }
                workgroup_sync();
                let b = sigmoid(*beta.add(i * bs + h));
                // sweep one: decay the rows, read S·k
                let mut d = 0f32;
                let mut r = 0;
                while r < HALF {
                    s[r] *= *e.add(kk0 + r);
                    d += s[r] * *conv.add(D + kk0 + r);
                    r += 1;
                }
                *ex.add((t / D) * D + vv) = d;
                workgroup_sync();
                let delta = (*conv.add(2 * D + vv) - (*ex.add(vv) + *ex.add(D + vv))) * b;
                workgroup_sync();
                // sweep two: the update, the readout S·q
                let mut o = 0f32;
                let mut r = 0;
                while r < HALF {
                    let k = *conv.add(D + kk0 + r);
                    s[r] += k * delta;
                    o += s[r] * *conv.add(kk0 + r);
                    r += 1;
                }
                *ex.add((t / D) * D + vv) = o;
                workgroup_sync();
                // the gated norm of the readout, by the first D threads
                if t < D {
                    let ov = *ex.add(vv) + *ex.add(D + vv);
                    let ss = wave_sum(lane, ov * ov);
                    if lane == 0 {
                        *red.add(t / 32) = ss;
                    }
                    workgroup_sync_waves4(red);
                    let total = *red.add(0) + *red.add(1) + *red.add(2) + *red.add(3);
                    let rn = rsqrt(total / D as f32 + RMS_NORM_EPS);
                    *normed.add(i * PART_QKV + hl * D + vv) = *o_norm.add(vv) * (ov * rn) * sigmoid(*gate.add(i * fs + hl * D + vv));
                } else {
                    workgroup_sync_waves4(red);
                }
                workgroup_sync();
                if !ring.is_null() {
                    let rm = ring.add(i * KDA_HEADS * D * D);
                    let mut r = 0;
                    while r < HALF {
                        *rm.add(h * D * D + (kk0 + r) * D + vv) = s[r];
                        r += 1;
                    }
                }
                i += 1;
            }
            let mut r = 0;
            while r < HALF {
                *mem.add(h * D * D + (kk0 + r) * D + vv) = s[r];
                r += 1;
            }
        }
    }

    /// The barrier the readout's four-wave reduction crosses; one for every thread of the group.
    #[inline(always)]
    unsafe fn workgroup_sync_waves4(_red: *mut f32) {
        workgroup_sync();
    }
}
