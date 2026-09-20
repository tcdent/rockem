//! The attention half of a layer, fused: input norm → q and kv projections →
//! latent norm → rope → cache write → absorbed MLA → o_proj → residual.
//! One signature; every primitive inlines. On the GPU this becomes two or
//! three launches split only at grid-wide sync points.
//!
//! Absorbed MLA: for each head the nope query is absorbed through that
//! head's `W_UK` (first `QK_NOPE_HEAD_DIM` rows of its `kv_b` block) into
//! latent space, scored against every cached latent plus the rope term,
//! softmaxed, and the probability-weighted latent is expanded through `W_UV`
//! (the next `V_HEAD_DIM` rows) into the head's output. `scale` already
//! carries the YaRN mscale² factor; this kernel never knows.

use crate::*;
use common::gemv;
use common::rmsnorm::rmsnorm;
use common::rope::deinterleave_rotate_half as rope;
use common::softmax::softmax;
use common::trace::Trace;

pub struct Weights<'a> {
    pub input_ln: &'a [f32], // H
    pub q_proj: &'a [u16],   // [Q_DIM][H] bf16
    pub kv_a: &'a [u16],     // [KVA_DIM][H] bf16
    pub kv_a_ln: &'a [f32],  // KV_LORA_RANK
    pub kv_b: &'a [f32],     // [KVB_DIM][KV_LORA_RANK] f32
    pub o_proj: &'a [u16],   // [H][O_IN_DIM] bf16
}

/// Where attention runs. `step` is `attn::step` on some tier: the CPU with
/// the cache in its own memory, or a card holding the cache and the
/// projection weights, with only the residual crossing. Chosen at
/// construction; the forward never asks which.
pub trait Attention {
    /// `n` tokens at positions `pos..pos + n`: `h` is `[n][H]`, `cos`/`sin`
    /// are the tables from position `pos` (`[n][QK_ROPE_HEAD_DIM]`). The
    /// cache holds positions `0..pos`; token `i` attends to `0..=pos + i`.
    fn step(&mut self, l: usize, w: &Weights, h: &mut [f32], pos: usize, n: usize, cos: &[f32], sin: &[f32], scale: f32, tr: &mut dyn Trace);
    /// Install `rows` — `[pos][layout::POSITION]`, so `rows.len() / POSITION`
    /// positions — as positions `0..n`; the sequence is then computed from
    /// `n`. The sink is the one-position case, and a cached context (the
    /// rows every step of a transcript produced) is the general one.
    fn seed(&mut self, rows: &[f32]);
    /// Token `i` of the last step's rows, every layer's, `[layer][KV_ROW]`:
    /// what a context store keeps.
    fn rows(&self, i: usize, out: &mut [f32]);
}

pub struct Scratch<'a> {
    pub hn: &'a mut [f32],     // H
    pub q: &'a mut [f32],      // Q_DIM
    pub kva: &'a mut [f32],    // KVA_DIM
    pub heads: &'a mut [f32],  // O_IN_DIM
    pub o: &'a mut [f32],      // H
    pub q_abs: &'a mut [f32],  // KV_LORA_RANK
    pub scores: &'a mut [f32], // ≥ pos + 1
    pub o_lat: &'a mut [f32],  // KV_LORA_RANK
}

/// `h` is the residual stream (updated in place); `cache` is this layer's
/// `[max_ctx][KV_ROW]`; `cos`/`sin` are position `pos`'s table rows.
#[inline(always)]
pub fn step(l: usize, w: &Weights, h: &mut [f32], cache: &mut [f32], pos: usize, cos: &[f32], sin: &[f32], scale: f32, s: &mut Scratch, tr: &mut impl Trace) {
    rmsnorm::<H>(h, w.input_ln, RMS_NORM_EPS, s.hn);
    tr.edge(l, "input_layernorm.out", s.hn);
    gemv::bf16::<Q_DIM, H>(w.q_proj, s.hn, s.q);
    tr.edge(l, "self_attn.q_proj.out", s.q);
    gemv::bf16::<KVA_DIM, H>(w.kv_a, s.hn, s.kva);
    tr.edge(l, "self_attn.kv_a_proj_with_mqa.out", s.kva);

    // cache row: normed latent | roped shared key
    let row = &mut cache[pos * KV_ROW..(pos + 1) * KV_ROW];
    let (c, k_pe) = row.split_at_mut(KV_LORA_RANK);
    rmsnorm::<KV_LORA_RANK>(&s.kva[..KV_LORA_RANK], w.kv_a_ln, RMS_NORM_EPS, c);
    tr.edge(l, "self_attn.kv_a_layernorm.out", c);
    k_pe.copy_from_slice(&s.kva[KV_LORA_RANK..]);
    rope::<QK_ROPE_HEAD_DIM>(k_pe, cos, sin);
    for hd in 0..HEADS {
        rope::<QK_ROPE_HEAD_DIM>(&mut s.q[hd * Q_HEAD_DIM + QK_NOPE_HEAD_DIM..(hd + 1) * Q_HEAD_DIM], cos, sin);
    }

    // absorbed MLA over positions 0..=pos
    let len = pos + 1;
    for hd in 0..HEADS {
        let qh = &s.q[hd * Q_HEAD_DIM..(hd + 1) * Q_HEAD_DIM];
        let (q_nope, q_pe) = qh.split_at(QK_NOPE_HEAD_DIM);
        let w_uk = &w.kv_b[hd * KVB_HEAD_DIM * KV_LORA_RANK..(hd * KVB_HEAD_DIM + QK_NOPE_HEAD_DIM) * KV_LORA_RANK];
        let w_uv = &w.kv_b[(hd * KVB_HEAD_DIM + QK_NOPE_HEAD_DIM) * KV_LORA_RANK..(hd + 1) * KVB_HEAD_DIM * KV_LORA_RANK];

        for j in 0..KV_LORA_RANK {
            s.q_abs[j] = 0.0;
        }
        for i in 0..QK_NOPE_HEAD_DIM {
            let wi = &w_uk[i * KV_LORA_RANK..(i + 1) * KV_LORA_RANK];
            let qi = q_nope[i];
            for j in 0..KV_LORA_RANK {
                s.q_abs[j] += wi[j] * qi;
            }
        }
        for t in 0..len {
            let r = &cache[t * KV_ROW..(t + 1) * KV_ROW];
            let (ct, kt) = r.split_at(KV_LORA_RANK);
            let mut acc = 0f32;
            for j in 0..KV_LORA_RANK {
                acc += s.q_abs[j] * ct[j];
            }
            for j in 0..QK_ROPE_HEAD_DIM {
                acc += q_pe[j] * kt[j];
            }
            s.scores[t] = acc * scale;
        }
        softmax(&mut s.scores[..len]);
        for j in 0..KV_LORA_RANK {
            s.o_lat[j] = 0.0;
        }
        for t in 0..len {
            let ct = &cache[t * KV_ROW..t * KV_ROW + KV_LORA_RANK];
            let p = s.scores[t];
            for j in 0..KV_LORA_RANK {
                s.o_lat[j] += p * ct[j];
            }
        }
        let oh = &mut s.heads[hd * V_HEAD_DIM..(hd + 1) * V_HEAD_DIM];
        for i in 0..V_HEAD_DIM {
            let wi = &w_uv[i * KV_LORA_RANK..(i + 1) * KV_LORA_RANK];
            let mut acc = 0f32;
            for j in 0..KV_LORA_RANK {
                acc += wi[j] * s.o_lat[j];
            }
            oh[i] = acc;
        }
    }
    tr.edge(l, "self_attn.o_proj.in", s.heads);
    gemv::bf16::<H, O_IN_DIM>(w.o_proj, s.heads, s.o);
    tr.edge(l, "self_attn.o_proj.out", s.o);
    for i in 0..H {
        h[i] += s.o[i];
    }
}

/// Launch geometry of the gfx11 realization, visible to hosts that launch it.
pub mod gfx11_consts {
    use crate::*;
    /// The card's cache is layer-major, `[layer][max_ctx][KV_ROW]`: each
    /// layer's rows contiguous. Measured against position-major (the store's
    /// layout) at a 2,048 context, 2026-09-18: 11 % slower per token there
    /// (rows at a 62 KB stride) for a 2.3× faster swap-in. Assumed mean: a
    /// context generates more than ~10 tokens per swap-in, so the token wins;
    /// a workload that switches contexts every few tokens would flip this.
    /// Byte offset of layer `l`.
    pub const fn cache_layer(l: usize, max_ctx: usize) -> usize {
        l * max_ctx * KV_ROW * 4
    }
    /// How a context is split across `attn_mla` workgroups: one workgroup
    /// per chunk of positions, every head scored in it against each row it
    /// reads (MLA's absorbed form shares one latent row across the heads; a
    /// workgroup per head read the cache sixteen times a token, 2 GB at
    /// 2,048, until 2026-09-19). As many chunks as fill the card (SPLITS)
    /// without a chunk dropping below CHUNK_MIN positions, and never more
    /// than CHUNK_MAX per chunk (the LDS holds sixteen heads' scores over it).
    /// Assumed means for this card; a card with more CUs wants SPLITS up.
    /// Every token of a step splits by its own length, so a position's
    /// attention is the same computation however many tokens it stepped with.
    pub const CHUNK_MAX: usize = 512;
    pub const CHUNK_MIN: usize = 32;
    pub const SPLITS: usize = 96;
    pub const fn mla_splits(len: usize) -> usize {
        let want = if len.div_ceil(CHUNK_MIN) < SPLITS { len.div_ceil(CHUNK_MIN) } else { SPLITS };
        let want = if want < 1 { 1 } else { want };
        let need = len.div_ceil(CHUNK_MAX);
        if need > want { need } else { want }
    }
    pub const fn mla_chunk(len: usize) -> usize {
        len.div_ceil(mla_splits(len))
    }
    /// Workgroups for tokens at `pos..pos + n`: Σ splits(pos + i + 1).
    pub const fn mla_grid(pos: usize, n: usize) -> u32 {
        let mut g = 0;
        let mut i = 0;
        while i < n {
            g += mla_splits(pos + i + 1);
            i += 1;
        }
        g as u32
    }
    pub const fn max_splits(max_ctx: usize) -> usize {
        let need = max_ctx.div_ceil(CHUNK_MAX);
        if need > SPLITS { need } else { SPLITS }
    }
    /// One (chunk, head) partial: `m`, `l`, pad, then the unnormalized latent output.
    pub const PART: usize = KV_LORA_RANK + 8;
    /// f32s of one token's partials for a cache of `max_ctx` positions: `[chunk][head][PART]`.
    pub const fn partials(max_ctx: usize) -> usize {
        max_splits(max_ctx) * HEADS * PART
    }
    pub const WG: u32 = 256;
    pub const PROJ_ROWS: usize = Q_DIM + KVA_DIM;
    pub const PROJ_GRID: u32 = PROJ_ROWS.div_ceil(8) as u32;
    pub const OUT_GRID: u32 = (H / 8) as u32;
    /// Launch-sized LDS: proj/out = a staged step of TOKENS activations (`gemv::gfx11`) + 8 partials;
    /// kv = 8 partials; mla = q_abs (R) + q_pe (P) + scores (CHUNK_MAX) + 8 per-wave outputs (R each) + 8 partials;
    /// merge = o_lat (R) + 8 partials.
    pub use crate::ffn::gfx11_consts::{normed_rows_lds, rows_lds};
    /// Rows a workgroup stages in LDS at a time: read from memory once, then
    /// every wave's two heads work on them. Four is 9 KB; with the rope
    /// queries, four workgroups a CU, whose waves hide each other's LDS
    /// latency (eight rows, two workgroups a CU, measured 15k cycles a
    /// batch against 3k of LDS bandwidth, 2026-09-19).
    pub const ROWS: usize = 4;
    /// mla = q_pe (HEADS × P) + a batch of ROWS cache rows.
    pub const MLA_LDS: usize = 4 * (HEADS * QK_ROPE_HEAD_DIM + ROWS * KV_ROW);
    /// One token's absorbed queries, `[HEADS][KV_LORA_RANK]`, written by `attn_kv`.
    pub const QABS: usize = HEADS * KV_LORA_RANK;
    /// The statistics `attn_mla` writes, by slot (`common::stats`): its phases in clock ticks.
    pub const MLA_STATS: [&str; 4] = ["prologue_ticks", "loop_ticks", "stage_ticks", "compute_ticks"];
    /// merge = 8 waves' partial outputs (R each) + 8 partials.
    pub const MERGE_LDS: usize = 4 * (8 * KV_LORA_RANK + 8);
    /// absorb = the cache rows' 8 partials.
    pub const ABSORB_LDS: usize = 4 * 8;
    /// `attn_absorb`'s grid: a workgroup per (token, head), then one per token.
    pub const fn absorb_grid(n: usize) -> u32 {
        (n * (HEADS + 1)) as u32
    }
}

#[cfg(target_arch = "amdgpu")]
pub mod gfx11 {
    //! The GPU realization of `step` for `n` tokens at once, five launches
    //! split at the grid-wide dependencies: `attn_proj` (the q and kv
    //! projections over the residual, normed as it is staged a step at a
    //! time, one wave per row — `gemv::gfx11`), `attn_absorb` (a workgroup
    //! per (token, head) for the absorbed query, and one per token for the
    //! latent norm, rope, the cache row and its mirror), `attn_mla` (a
    //! workgroup per (token, head, chunk of that token's
    //! positions): absorbed scores over its chunk from the cache, a local
    //! softmax and latent output kept unnormalized with its max and sum),
    //! `attn_merge` (a workgroup per (token, head): the flash-decoding
    //! combine of the chunks, then value expansion), and `attn_out` (`o_proj`
    //! with the residual add as epilogue). One token is n = 1 through the
    //! same kernels; a token's bits do not depend on how many stepped with
    //! it. Before 2026-09-18 `attn_mla` was one workgroup per head over the
    //! whole context: 16 workgroups on 84 CUs, and at a 2,048 context 7.8 ms
    //! a token to read 127 MB — the split is what lets a long context use the
    //! card. The residual stream lives on the card between launches. Weights
    //! are bf16 straight from the bundle's dense file; norms are widened on
    //! the fly.
    use crate::*;
    use common::gemv::gfx11::{bf16_row, rows_over};
    const _: () = assert!(TOKENS == common::gemv::gfx11::TOKENS);
    use common::gemv::widen;
    use common::gfx11::{lds, workgroup_max, workgroup_sum, workgroup_sync};
    use common::math::{exp, rsqrt};
    use core::arch::amdgpu::{workgroup_id_x, workitem_id_x};

    pub use super::gfx11_consts::*;

    /// `q[i] = W_q · xn[i]`, `kva[i] = W_kva · xn[i]` for `n` tokens, `xn[i]`
    /// the residual `h[i]` normed with `ln` on the way (`rows_over` with
    /// NORM): the rows of the two matrices as one `[PROJ_ROWS][H]` matrix
    /// through the rows skeleton. Grid: PROJ_GRID × 256.
    #[no_mangle]
    pub extern "gpu-kernel" fn attn_proj(h: *const f32, ln: *const u16, w_q: *const u16, w_kva: *const u16, q: *mut f32, kva: *mut f32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let t = tid as usize;
            let row = workgroup_id_x() as usize * 8 + (tid / 32) as usize;
            // `rows_over` reads row `row` at `w + row · H`: point `w` so that lands in the matrix holding it
            let w: *const u16 = if row < Q_DIM { w_q } else { w_kva.sub(Q_DIM * H) };
            rows_over::<H, true>(
                tid,
                w,
                ln,
                PROJ_ROWS,
                n as usize,
                RMS_NORM_EPS,
                |i, s| *h.add(i * H + s * 256 + t),
                || {},
                |row, i, y| {
                    if row < Q_DIM {
                        *q.add(i * Q_DIM + row) = y;
                    } else {
                        *kva.add(i * KVA_DIM + row - Q_DIM) = y;
                    }
                },
            );
        }
    }

    /// Workgroup `i` of the cache rows: the latent norm of `kva[i]` and the
    /// rope of its tail become cache row `pos + i` of this layer and the
    /// token's row of `mirror` (`[n][layer][KV_ROW]`, this layer's slot).
    #[inline(always)]
    unsafe fn kv_row(i: usize, tid: u32, kva: *const f32, kv_a_ln: *const u16, cache: *mut f32, pos: u32, cos: *const f32, sin: *const f32, mirror: *mut f32) {
        {
            const R: usize = KV_LORA_RANK;
            const P: usize = QK_ROPE_HEAD_DIM;
            let t = tid as usize;
            let red: *mut f32 = lds();
            let kva = kva.add(i * KVA_DIM);
            let row = cache.add((pos as usize + i) * KV_ROW);
            let mirror = mirror.add(i * layout::POSITION);
            let mut ss = 0f32;
            let mut j = t;
            while j < R {
                let v = *kva.add(j);
                ss += v * v;
                j += 256;
            }
            let r = rsqrt(workgroup_sum(tid, ss, red) / R as f32 + RMS_NORM_EPS);
            let mut j = t;
            while j < R {
                let v = widen(*kv_a_ln.add(j)) * (*kva.add(j) * r);
                *row.add(j) = v;
                *mirror.add(j) = v;
                j += 256;
            }
            // DeepSeek rope on the 64-wide tail: de-interleave then rotate_half; threads 0..32 do one pair each
            if t < P / 2 {
                let (cos, sin) = (cos.add(i * P), sin.add(i * P));
                let (ka, kb) = (*kva.add(R + 2 * t), *kva.add(R + 2 * t + 1));
                let (a, b) = (ka * *cos.add(t) - kb * *sin.add(t), kb * *cos.add(t + P / 2) + ka * *sin.add(t + P / 2));
                *row.add(R + t) = a;
                *row.add(R + t + P / 2) = b;
                *mirror.add(R + t) = a;
                *mirror.add(R + t + P / 2) = b;
            }
        }
    }

    /// What `attn_proj` leaves for `attn_mla`, one launch: a workgroup per
    /// (token, head) for the absorbed query `q_abs[i][hd][j] = Σ_k
    /// W_UK[hd][k][j] · q_nope[hd][k]`, threads owning `j` and `j + 256`, for
    /// `attn_mla` to read once per chunk; then a workgroup per token for its
    /// cache row (`kv_row`). Both need only the projections, so they were two
    /// launches side by side; one launch is a launch and its gap fewer a
    /// layer (2026-09-19). One workgroup per token did all sixteen heads at
    /// 108 µs a layer (2026-09-19): a single workgroup streaming 2 MB.
    /// Grid: n · (HEADS + 1) × 256.
    #[no_mangle]
    pub extern "gpu-kernel" fn attn_absorb(q: *const f32, kv_b: *const u16, q_abs: *mut f32, n: u32, kva: *const f32, kv_a_ln: *const u16, cache: *mut f32, pos: u32, cos: *const f32, sin: *const f32, mirror: *mut f32) {
        unsafe {
            const R: usize = KV_LORA_RANK;
            const N: usize = QK_NOPE_HEAD_DIM;
            let wg = workgroup_id_x() as usize;
            let tid = workitem_id_x();
            if wg >= n as usize * HEADS {
                kv_row(wg - n as usize * HEADS, tid, kva, kv_a_ln, cache, pos, cos, sin, mirror);
                return;
            }
            let (i, hd) = (wg / HEADS, wg % HEADS);
            let t = tid as usize;
            let w_uk = kv_b.add(hd * KVB_HEAD_DIM * R);
            let q_nope = q.add(i * Q_DIM + hd * Q_HEAD_DIM);
            let (mut a0, mut a1) = (0f32, 0f32);
            let mut k = 0;
            while k < N {
                let qi = *q_nope.add(k);
                a0 += widen(*w_uk.add(k * R + t)) * qi;
                a1 += widen(*w_uk.add(k * R + t + 256)) * qi;
                k += 1;
            }
            let out = q_abs.add(i * QABS + hd * R);
            *out.add(t) = a0;
            *out.add(t + 256) = a1;
        }
    }

    /// The (token, split) a workgroup owns: tokens in order, each with
    /// `mla_splits(pos + i + 1)` workgroups.
    #[inline(always)]
    fn owner(wg: usize, pos: usize) -> (usize, usize, usize) {
        let mut i = 0;
        let mut base = 0;
        loop {
            let splits = mla_splits(pos + i + 1);
            if wg < base + splits {
                return (i, wg - base, splits);
            }
            base += splits;
            i += 1;
        }
    }

    /// A workgroup per (token, chunk of `mla_chunk(len)` positions), len the
    /// token's `pos + i + 1`; grid `mla_grid(pos, n)`. The chunk's rows are
    /// staged in LDS ROWS at a time — each row read from memory once per
    /// token — and every wave takes two heads over the staged rows: scores,
    /// then an online softmax (the running max, the rescale of the sum and
    /// the latent output when it moves, as `attn_merge` does across chunks),
    /// so one pass over the rows does it all. Reads the cache and the
    /// absorbed queries only (`attn_kv` wrote both for the step). Leaves one
    /// `PART` per head in `part` at `((i · max_splits + split) · HEADS +
    /// head)`: its max, its sum of exponentials, and its unnormalized latent
    /// output. A workgroup per head over the whole chunk read the cache
    /// sixteen times a token, and heads in registers with waves over rows
    /// read it twelve times from L2 (2026-09-19); this reads it once.
    ///
    /// A lane owns the latent dims `l, l + 32, …, l + 480` (sixteen, strided
    /// by 32), not sixteen consecutive ones: consecutive lanes then read
    /// consecutive LDS words and the staged row is read conflict-free
    /// (sixteen consecutive dims a lane was a 16-way bank conflict, 9× the
    /// kernel's time, 2026-09-19). The rows themselves are staged as they
    /// are, so their loads from memory coalesce.
    #[no_mangle]
    pub extern "gpu-kernel" fn attn_mla(q_abs: *const f32, cache: *const f32, pos: u32, q: *const f32, cos: *const f32, sin: *const f32, scale: f32, part: *mut f32, max_splits: u32) {
        unsafe {
            const R: usize = KV_LORA_RANK;
            const P: usize = QK_ROPE_HEAD_DIM;
            const N: usize = QK_NOPE_HEAD_DIM;
            const HD: usize = HEADS;
            let t0 = common::stat_clock!();
            let (i, sp, _splits) = owner(workgroup_id_x() as usize, pos as usize);
            let len = pos as usize + i + 1;
            let chunk = mla_chunk(len);
            let (p0, p1) = (sp * chunk, if (sp + 1) * chunk < len { (sp + 1) * chunk } else { len });
            let tid = workitem_id_x();
            let (t, lane, wave) = (tid as usize, tid % 32, (tid / 32) as usize);
            let l = lane as usize;
            let q_pe: *mut f32 = lds(); // [HD][P]
            let rows = q_pe.add(HD * P); // [ROWS][KV_ROW]
            let q = q.add(i * Q_DIM);
            let q_abs = q_abs.add(i * QABS);
            let (cos, sin) = (cos.add(i * P), sin.add(i * P));

            // every head's rope tail: sixteen heads of 32 pairs, two per thread
            let mut task = t;
            while task < HD * 32 {
                let (hd, j) = (task / 32, task % 32);
                let qh = q.add(hd * Q_HEAD_DIM + N);
                let (qa, qb) = (*qh.add(2 * j), *qh.add(2 * j + 1));
                *q_pe.add(hd * P + j) = qa * *cos.add(j) - qb * *sin.add(j);
                *q_pe.add(hd * P + j + P / 2) = qb * *cos.add(j + P / 2) + qa * *sin.add(j + P / 2);
                task += 256;
            }
            workgroup_sync();
            // this wave's two heads: absorbed queries over the lane's 16 dims, and its rope pair
            let (h0, h1) = (2 * wave, 2 * wave + 1);
            let mut qa = [[0f32; 16]; 2];
            let mut d = 0;
            while d < 16 {
                qa[0][d] = *q_abs.add(h0 * R + l + 32 * d);
                qa[1][d] = *q_abs.add(h1 * R + l + 32 * d);
                d += 1;
            }
            let qp = [[*q_pe.add(h0 * P + 2 * l), *q_pe.add(h0 * P + 2 * l + 1)], [*q_pe.add(h1 * P + 2 * l), *q_pe.add(h1 * P + 2 * l + 1)]];
            let t1 = common::stat_clock!();
            common::stat!(0, (t1.wrapping_sub(t0)) & common::stats::CLOCK_MASK);

            let mut m = [f32::NEG_INFINITY; 2];
            let mut lsum = [0f32; 2];
            let mut acc = [[0f32; 16]; 2];
            let (mut stage_ticks, mut compute_ticks) = (0u64, 0u64);
            let mut b0 = p0;
            while b0 < p1 {
                let nb = if p1 - b0 < ROWS { p1 - b0 } else { ROWS };
                let ts = common::stat_clock!();
                // stage the batch: 256 threads over its floats, coalesced, every load issued
                // before any store (one at a time was 670 cycles a load, 2026-09-19)
                const PER: usize = ROWS * KV_ROW / 256;
                let mut v = [0f32; PER];
                let mut k = 0;
                while k < PER {
                    if k * 256 + t < nb * KV_ROW {
                        v[k] = *cache.add(b0 * KV_ROW + k * 256 + t);
                    }
                    k += 1;
                }
                let mut k = 0;
                while k < PER {
                    if k * 256 + t < nb * KV_ROW {
                        *rows.add(k * 256 + t) = v[k];
                    }
                    k += 1;
                }
                workgroup_sync();
                let tc = common::stat_clock!();
                stage_ticks += tc.wrapping_sub(ts) & common::stats::CLOCK_MASK;
                // scores of the batch for the two heads; rows past the batch score −∞ (their
                // exponential is 0) — constant-bound loops, so the arrays stay in registers
                let mut s = [[f32::NEG_INFINITY; ROWS]; 2];
                let mut r = 0;
                while r < ROWS {
                    if r < nb {
                        let row = rows.add(r * KV_ROW);
                        let mut c = [0f32; 16];
                        let mut d = 0;
                        while d < 16 {
                            c[d] = *row.add(l + 32 * d);
                            d += 1;
                        }
                        let (k0, k1) = (*row.add(R + 2 * l), *row.add(R + 2 * l + 1));
                        let mut a = [0f32; 2];
                        let mut k = 0;
                        while k < 2 {
                            let mut d = 0;
                            while d < 16 {
                                a[k] += qa[k][d] * c[d];
                                d += 1;
                            }
                            a[k] += qp[k][0] * k0 + qp[k][1] * k1;
                            k += 1;
                        }
                        // both heads' sums in one transposed reduction, then each to every lane
                        let (s0, s1) = butterfly2(lane, a);
                        s[0][r] = s0 * scale;
                        s[1][r] = s1 * scale;
                    }
                    r += 1;
                }
                // the online softmax: a new max rescales what came before
                let mut e = [[0f32; ROWS]; 2];
                let mut k = 0;
                while k < 2 {
                    let mut mb = m[k];
                    let mut r = 0;
                    while r < ROWS {
                        if s[k][r] > mb {
                            mb = s[k][r];
                        }
                        r += 1;
                    }
                    let f = exp(m[k] - mb);
                    lsum[k] *= f;
                    let mut d = 0;
                    while d < 16 {
                        acc[k][d] *= f;
                        d += 1;
                    }
                    let mut r = 0;
                    while r < ROWS {
                        e[k][r] = exp(s[k][r] - mb);
                        lsum[k] += e[k][r];
                        r += 1;
                    }
                    m[k] = mb;
                    k += 1;
                }
                // the outputs: each staged row read once, both heads accumulate from it
                let mut r = 0;
                while r < ROWS {
                    if r < nb {
                        let row = rows.add(r * KV_ROW + l);
                        let mut d = 0;
                        while d < 16 {
                            let v = *row.add(32 * d);
                            acc[0][d] += e[0][r] * v;
                            acc[1][d] += e[1][r] * v;
                            d += 1;
                        }
                    }
                    r += 1;
                }
                workgroup_sync();
                compute_ticks += common::stat_clock!().wrapping_sub(tc) & common::stats::CLOCK_MASK;
                b0 += ROWS;
            }
            let t2 = common::stat_clock!();
            common::stat!(1, (t2.wrapping_sub(t1)) & common::stats::CLOCK_MASK);
            common::stat!(2, stage_ticks);
            common::stat!(3, compute_ticks);
            let mut k = 0;
            while k < 2 {
                let out = part.add(((i * max_splits as usize + sp) * HD + 2 * wave + k) * PART);
                let mut d = 0;
                while d < 16 {
                    *out.add(8 + l + 32 * d) = acc[k][d];
                    d += 1;
                }
                if lane == 0 {
                    *out = m[k];
                    *out.add(1) = lsum[k];
                }
                k += 1;
            }
            let _ = t2;
            const _: () = assert!(HD == 16 && R == 16 * 32 && P == 64 && CHUNK_MAX % ROWS == 0);
        }
    }

    /// Two per-lane partials summed across the wave, transposed: the first
    /// step trades one value with the partner at distance 1 (the lane keeps
    /// the one its bit selects), four steps add the partners at 2, 4, 8 and
    /// 16, then lanes 0 and 1 hold the two sums and every lane fetches both.
    /// Seven exchanges for two heads where two `wave_sum`s are ten.
    #[inline(always)]
    unsafe fn butterfly2(lane: u32, v: [f32; 2]) -> (f32, f32) {
        use core::arch::amdgpu::ds_bpermute;
        let x = |l: u32, val: f32| f32::from_bits(ds_bpermute(l * 4, val.to_bits()));
        let hi = lane & 1 != 0;
        let (keep, give) = if hi { (v[1], v[0]) } else { (v[0], v[1]) };
        let a = keep + x(lane ^ 1, give);
        let a = a + x(lane ^ 2, a);
        let a = a + x(lane ^ 4, a);
        let a = a + x(lane ^ 8, a);
        let a = a + x(lane ^ 16, a);
        (x(0, a), x(1, a))
    }

    /// One workgroup per (token, head): combine the token's chunk partials
    /// (rescale each by exp(m_s − M), divide by the total), then
    /// `heads[i][hd] = W_UV · o_lat`. The max over the chunks is a workgroup
    /// reduction; the chunks then go round-robin over the eight waves, lanes
    /// owning 16 dims, and the waves' sums meet in LDS — one thread walking
    /// 782 chunks in turn was 557 µs a layer at 100k (2026-09-19).
    /// Grid: n · HEADS × 256.
    #[no_mangle]
    pub extern "gpu-kernel" fn attn_merge(part: *const f32, pos: u32, max_splits: u32, kv_b: *const u16, heads: *mut f32) {
        unsafe {
            const R: usize = KV_LORA_RANK;
            const N: usize = QK_NOPE_HEAD_DIM;
            const V: usize = V_HEAD_DIM;
            let wg = workgroup_id_x() as usize;
            let (i, hd) = (wg / HEADS, wg % HEADS);
            let splits = mla_splits(pos as usize + i + 1);
            let tid = workitem_id_x();
            let (t, lane, wave) = (tid as usize, tid % 32, (tid / 32) as usize);
            let l = lane as usize;
            let ow: *mut f32 = lds(); // [8][R] the waves' outputs, then o_lat in wave 0's row
            let red = ow.add(8 * R);
            let at = |sp: usize| part.add(((i * max_splits as usize + sp) * HEADS + hd) * PART);
            let mut m = f32::NEG_INFINITY;
            let mut sp = t;
            while sp < splits {
                let v = *at(sp);
                if v > m {
                    m = v;
                }
                sp += 256;
            }
            let big = workgroup_max(tid, m, red);
            let mut total = 0f32;
            let mut acc = [0f32; 16];
            let mut sp = wave;
            while sp < splits {
                let ps = at(sp);
                let f = exp(*ps - big);
                total += *ps.add(1) * f;
                let mut d = 0;
                while d < 16 {
                    acc[d] += *ps.add(8 + l + 32 * d) * f;
                    d += 1;
                }
                sp += 8;
            }
            let total = workgroup_sum(tid, if lane == 0 { total } else { 0.0 }, red);
            let mut d = 0;
            while d < 16 {
                *ow.add(wave * R + l + 32 * d) = acc[d];
                d += 1;
            }
            workgroup_sync();
            let inv = 1.0 / total;
            let (mut o0, mut o1) = (0f32, 0f32);
            let mut w = 0;
            while w < 8 {
                o0 += *ow.add(w * R + t);
                o1 += *ow.add(w * R + t + 256);
                w += 1;
            }
            workgroup_sync();
            let o_lat = ow;
            *o_lat.add(t) = o0 * inv;
            *o_lat.add(t + 256) = o1 * inv;
            workgroup_sync();

            // heads[i][hd][r] = Σ_j W_UV[r][j] · o_lat[j]; a wave per row, 16 rows per wave
            let w_uv = kv_b.add((hd * KVB_HEAD_DIM + N) * R);
            let mut r = wave;
            while r < V {
                let y = bf16_row::<R>(lane, w_uv.add(r * R), o_lat);
                if lane == 0 {
                    *heads.add(i * O_IN_DIM + hd * V + r) = y;
                }
                r += 8;
            }
        }
    }

    /// `h[i] += W_o · heads[i]` for `n` tokens. Grid: OUT_GRID × 256.
    #[no_mangle]
    pub extern "gpu-kernel" fn attn_out(heads: *const f32, w_o: *const u16, h: *mut f32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let t = tid as usize;
            rows_over::<O_IN_DIM, false>(
                tid,
                w_o,
                core::ptr::null(),
                H,
                n as usize,
                0.0,
                |i, s| *heads.add(i * O_IN_DIM + s * 256 + t),
                || {},
                |row, i, y| *h.add(i * H + row) += y,
            );
        }
    }

    const _: () = assert!(KV_LORA_RANK == 512 && KV_ROW == 576 && QK_ROPE_HEAD_DIM == 64 && O_IN_DIM % 256 == 0 && H % 256 == 0 && V_HEAD_DIM % 8 == 0);
}
