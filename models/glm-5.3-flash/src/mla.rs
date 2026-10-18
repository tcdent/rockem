//! NoPE multi-latent attention over the keys the indexer selects, one token
//! at position `pos`: the query through its LoRA pair, the latent through its
//! projection and norm into this position's cache row, the indexer's key,
//! pool gate, query and head weights, the pool keys (a pool completes every
//! `INDEX_KPOOL` positions), the selection (every complete pool while there
//! are at most `INDEX_TOPK / INDEX_KPOOL`, else the best-scored that many,
//! plus the incomplete tail), then absorbed MLA over the selected positions —
//! `q_abs = W_ukᵀ q` per head, scores against the latents, the latent mix back
//! through `W_uv` (V2-Lite's form; the reference expands keys and values
//! instead, the same math in another order) — and the output projection.
//!
//! The layer's state is the caller's: `cache` `[max_ctx][layout::MLA_ROW]`
//! (latent | indexer key | indexer gate) and `pools` `[max_ctx / INDEX_KPOOL]
//! [INDEX_HEAD_DIM]`, both filled through position `pos` by this step.

use crate::layout::MLA_ROW;
use crate::weights;
use crate::*;
use common::gemv;
use common::math::bf16_round;
use common::math::{exp, sqrt};
use common::pool::Pool;
use common::rmsnorm::rmsnorm;
use common::softmax::softmax;
use common::trace::Trace;

pub struct Scratch<'a> {
    pub qr: &'a mut [f32],     // Q_LORA_RANK, normed
    pub qa: &'a mut [f32],     // Q_LORA_RANK, before the norm
    pub q: &'a mut [f32],      // MLA_HEADS·Q_HEAD
    pub kva: &'a mut [f32],    // KV_LORA_RANK
    pub ik: &'a mut [f32],     // INDEX_HEAD_DIM, before the LayerNorm
    pub iq: &'a mut [f32],     // INDEX_N_HEADS·INDEX_HEAD_DIM
    pub iw: &'a mut [f32],     // INDEX_N_HEADS
    pub pscore: &'a mut [f32], // ≥ max_ctx / INDEX_KPOOL: a score per complete pool
    pub pidx: &'a mut [u32],   // ≥ max_ctx / INDEX_KPOOL: the selection's pool order
    pub sel: &'a mut [usize],  // ≥ DENSE_BELOW: the positions attended, ascending
    pub heads: &'a mut [f32],  // MLA_HEADS·V_HEAD_DIM: the heads' outputs
    pub hs: &'a mut [f32],     // MLA_HEADS · HEAD_SCRATCH: per head, q_abs | o_lat | att
}
/// f32s of scratch a head's attention needs.
pub const HEAD_SCRATCH: usize = 2 * KV_LORA_RANK + DENSE_BELOW;

/// The complete pool `j`'s key from its positions' indexer keys and gates (`cache` rows):
/// a per-channel softmax over the four gates plus the learned offset per slot.
#[inline(always)]
pub fn pool_key(w: &weights::Mla, cache: &[f32], j: usize, out: &mut [f32]) {
    for c in 0..INDEX_HEAD_DIM {
        let mut logit = [0f32; INDEX_KPOOL];
        let mut mx = f32::NEG_INFINITY;
        for i in 0..INDEX_KPOOL {
            logit[i] = cache[(j * INDEX_KPOOL + i) * MLA_ROW + KV_LORA_RANK + INDEX_HEAD_DIM + c] + w.ix_ape[i * INDEX_HEAD_DIM + c];
            if logit[i] > mx {
                mx = logit[i];
            }
        }
        let mut sum = 0f32;
        for i in 0..INDEX_KPOOL {
            logit[i] = exp(logit[i] - mx);
            sum += logit[i];
        }
        // the pool as the reference forms it: the softmax weights cast to the keys' dtype (bf16), the
        // weighted sum accumulated in f32 and stored at bf16 precision
        let mut acc = 0f32;
        for i in 0..INDEX_KPOOL {
            acc += bf16_round(logit[i] / sum) * cache[(j * INDEX_KPOOL + i) * MLA_ROW + KV_LORA_RANK + c];
        }
        out[c] = bf16_round(acc);
    }
}

/// Pools one scoring item takes (`score_pools` over the thread pool).
pub const SCORE_BLOCK: usize = 128;
/// Pools kept by the selection.
pub const KEEP: usize = INDEX_TOPK / INDEX_KPOOL;

/// The indexer's score of pools `j0..j1` for one query (`iq`, its head weights `iw` scaled by
/// `head_scale`): per pool, the heads' ReLU'd dots with the pool key, weighted. One token's pools
/// go in blocks over the thread pool; the dot's order is the definition's whichever block runs it.
#[inline(always)]
pub fn score_pools(iq: &[f32], iw: &[f32], pools: &[f32], j0: usize, j1: usize, pscore: &mut [f32]) {
    let scale = 1.0 / sqrt(INDEX_HEAD_DIM as f32);
    for j in j0..j1 {
        let key = &pools[j * INDEX_HEAD_DIM..(j + 1) * INDEX_HEAD_DIM];
        let mut sc = 0f32;
        for h in 0..INDEX_N_HEADS {
            let qh = &iq[h * INDEX_HEAD_DIM..(h + 1) * INDEX_HEAD_DIM];
            let mut d = 0f32;
            for c in 0..INDEX_HEAD_DIM {
                d += qh[c] * key[c];
            }
            let d = d * scale;
            sc += iw[h] * if d > 0.0 { d } else { 0.0 };
        }
        pscore[j] = sc;
    }
}
/// The head weights' scale, applied once to a token's `iw` before its scores.
#[inline(always)]
pub fn head_scale(iw: &mut [f32]) {
    let wscale = 1.0 / sqrt(INDEX_N_HEADS as f32);
    for v in iw.iter_mut() {
        *v *= wscale;
    }
}
/// The two scales of the indexer's score (`score_pools`'s), for a card running it: the host's
/// values, so the card never takes its own root.
pub fn score_scales() -> (f32, f32) {
    (1.0 / sqrt(INDEX_HEAD_DIM as f32), 1.0 / sqrt(INDEX_N_HEADS as f32))
}
/// Whether query `pos` selects among its pools (past `KEEP` complete ones) or attends them all.
#[inline(always)]
pub fn selects(pos: usize) -> bool {
    (pos + 1) / INDEX_KPOOL > KEEP
}
/// The positions query `pos` attends, ascending, into `sel`; how many. Among more than `KEEP`
/// complete pools, the `KEEP` best by score, lower index first on a tie (`pidx` is the scratch:
/// one partition at the `KEEP`th, then the picks sorted), then the incomplete tail's positions.
#[inline(always)]
pub fn pick(pos: usize, pscore: &[f32], pidx: &mut [u32], sel: &mut [usize]) -> usize {
    let visible = (pos + 1) / INDEX_KPOOL;
    let mut n = 0;
    if visible <= KEEP {
        for p in 0..visible * INDEX_KPOOL {
            sel[n] = p;
            n += 1;
        }
    } else {
        for j in 0..visible {
            pidx[j] = j as u32;
        }
        // score descending, index ascending: a total order on finite scores
        let by = |a: &u32, b: &u32| pscore[*b as usize].partial_cmp(&pscore[*a as usize]).unwrap_or(core::cmp::Ordering::Equal).then(a.cmp(b));
        pidx[..visible].select_nth_unstable_by(KEEP - 1, by);
        pidx[..KEEP].sort_unstable();
        for &j in &pidx[..KEEP] {
            for p in j as usize * INDEX_KPOOL..(j as usize + 1) * INDEX_KPOOL {
                sel[n] = p;
                n += 1;
            }
        }
    }
    for p in visible * INDEX_KPOOL..=pos {
        sel[n] = p;
        n += 1;
    }
    n
}
/// `score_pools` over every complete pool when the query selects, then `pick`.
#[inline(always)]
fn select(pos: usize, iq: &[f32], iw: &[f32], pools: &[f32], pscore: &mut [f32], pidx: &mut [u32], sel: &mut [usize]) -> usize {
    if selects(pos) {
        score_pools(iq, iw, pools, 0, (pos + 1) / INDEX_KPOOL, pscore);
    }
    pick(pos, pscore, pidx, sel)
}

// The step is its parts, composed by the caller so a placement can take the
// projections and the output while the row, the selection and the attention
// run here: `row`, `project`, `finish_row`, `attend`, `output`.

/// Position `pos`'s cache row: the latent, the indexer's key, the indexer's gate.
#[inline(always)]
pub fn row(cache: &mut [f32], pos: usize) -> (&mut [f32], &mut [f32], &mut [f32]) {
    let row = &mut cache[pos * MLA_ROW..(pos + 1) * MLA_ROW];
    let (latent, rest) = row.split_at_mut(KV_LORA_RANK);
    let (ikey, igate) = rest.split_at_mut(INDEX_HEAD_DIM);
    (latent, ikey, igate)
}

/// The projections of the normed block input: the query through its LoRA pair
/// (`s.q`), the latent before its norm (`s.kva`), the indexer's key before its
/// LayerNorm (`s.ik`), the position's gate (`igate`, the cache row's), the
/// indexer's query (`s.iq`) and head weights (`s.iw`, unscaled).
#[inline(always)]
pub fn project(w: &weights::Mla, xn: &[f32], s: &mut Scratch, igate: &mut [f32], pool: &dyn Pool) {
    gemv::fp8_par::<Q_LORA_RANK, H>(w.q_a.w, w.q_a.s, xn, s.qa, pool);
    rmsnorm::<Q_LORA_RANK>(s.qa, &w.q_a_ln, RMS_NORM_EPS, s.qr);
    gemv::fp8_par::<{ MLA_HEADS * Q_HEAD }, Q_LORA_RANK>(w.q_b.w, w.q_b.s, s.qr, s.q, pool);
    gemv::fp8_par::<KV_LORA_RANK, H>(w.kv_a.w, w.kv_a.s, xn, s.kva, pool);
    gemv::bf16::<INDEX_HEAD_DIM, H>(w.ix_wk, xn, s.ik);
    gemv::bf16::<INDEX_HEAD_DIM, H>(w.ix_gate, xn, igate);
    gemv::bf16_par::<{ INDEX_N_HEADS * INDEX_HEAD_DIM }, Q_LORA_RANK>(w.ix_wq_b, s.qr, s.iq, pool);
    gemv::bf16::<INDEX_N_HEADS, H>(w.ix_weights, xn, s.iw);
}

/// The cache row's normed latent and the indexer's key (LayerNorm, eps 1e-6) from the projections.
#[inline(always)]
pub fn finish_row(w: &weights::Mla, kva: &[f32], ik: &[f32], latent: &mut [f32], ikey: &mut [f32]) {
    // the stored attention state follows the reference's dtype (tcdent, 2026-10-17): the latent
    // and the indexer's key at bf16 precision (the arithmetic on them stays f32), the KDA state f32
    rmsnorm::<KV_LORA_RANK>(kva, &w.kv_a_ln, RMS_NORM_EPS, latent);
    for v in latent.iter_mut() {
        *v = bf16_round(*v);
    }
    let mut mean = 0f32;
    for &v in ik.iter() {
        mean += v;
    }
    let mean = mean / INDEX_HEAD_DIM as f32;
    let mut var = 0f32;
    for &v in ik.iter() {
        var += (v - mean) * (v - mean);
    }
    let rs = 1.0 / sqrt(var / INDEX_HEAD_DIM as f32 + 1e-6);
    for c in 0..INDEX_HEAD_DIM {
        ikey[c] = bf16_round((ik[c] - mean) * rs * w.ix_k_norm_w[c] + w.ix_k_norm_b[c]);
    }
}
/// The pool position `pos` completes, if any, gains its key (`pool_key`).
#[inline(always)]
pub fn pool_done(w: &weights::Mla, cache: &[f32], pools: &mut [f32], pos: usize) {
    if (pos + 1) % INDEX_KPOOL == 0 {
        let j = (pos + 1) / INDEX_KPOOL - 1;
        pool_key(w, cache, j, &mut pools[j * INDEX_HEAD_DIM..(j + 1) * INDEX_HEAD_DIM]);
    }
}

/// The pool key a completed pool gains, and the selection: `s.sel[..n]`, the
/// positions attended, ascending.
#[inline(always)]
pub fn select_at(l: usize, w: &weights::Mla, cache: &[f32], pools: &mut [f32], pos: usize, s: &mut Scratch, tr: &mut impl Trace) -> usize {
    pool_done(w, cache, pools, pos);
    head_scale(s.iw);
    let n = select(pos, s.iq, s.iw, pools, s.pscore, s.pidx, s.sel);
    tr.edge(l, "self_attn.indexer.count", &[n as f32]);
    n
}

/// Absorbed MLA over the `n` selected positions, a head per item on its own scratch, into `s.heads`.
#[inline(always)]
pub fn heads(l: usize, w: &weights::Mla, cache: &[f32], s: &mut Scratch, n: usize, pool: &dyn Pool, tr: &mut impl Trace) {
    let (q, sel) = (&*s.q, &s.sel[..n]);
    let (heads_p, hs_p) = (s.heads.as_mut_ptr() as usize, s.hs.as_mut_ptr() as usize);
    // a head's item streams its W_uk and W_uv: on their node
    pool.run_at(MLA_HEADS, &move |h| w.kv_b[h * KVB_HEAD * KV_LORA_RANK..].as_ptr() as *const u8, &move |h| {
        let oh = unsafe { core::slice::from_raw_parts_mut((heads_p as *mut f32).add(h * V_HEAD_DIM), V_HEAD_DIM) };
        let hs = unsafe { core::slice::from_raw_parts_mut((hs_p as *mut f32).add(h * HEAD_SCRATCH), HEAD_SCRATCH) };
        let (q_abs, rest) = hs.split_at_mut(KV_LORA_RANK);
        let (o_lat, att) = rest.split_at_mut(KV_LORA_RANK);
        head(w, h, q, sel, cache, q_abs, o_lat, att, oh);
    });
    tr.edge(l, "self_attn.o_proj.in", s.heads);
}

/// The output projection of the heads' outputs.
#[inline(always)]
pub fn output(w: &weights::Mla, heads: &[f32], out: &mut [f32], pool: &dyn Pool) {
    gemv::fp8_par::<H, { MLA_HEADS * V_HEAD_DIM }>(w.o.w, w.o.s, heads, out, pool);
}

/// One head over the selected positions: `q_abs = W_ukᵀ q`, scores against the
/// latents, softmax, the latent mix, back through `W_uv`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn head(w: &weights::Mla, h: usize, q: &[f32], sel: &[usize], cache: &[f32], q_abs: &mut [f32], o_lat: &mut [f32], att: &mut [f32], oh: &mut [f32]) {
    let scale = 1.0 / sqrt(Q_HEAD as f32);
    let qh = &q[h * Q_HEAD..(h + 1) * Q_HEAD];
    let w_uk = &w.kv_b[h * KVB_HEAD * KV_LORA_RANK..(h * KVB_HEAD + QK_NOPE_HEAD_DIM) * KV_LORA_RANK];
    let w_uv = &w.kv_b[(h * KVB_HEAD + QK_NOPE_HEAD_DIM) * KV_LORA_RANK..(h + 1) * KVB_HEAD * KV_LORA_RANK];
    for j in 0..KV_LORA_RANK {
        q_abs[j] = 0.0;
    }
    for i in 0..QK_NOPE_HEAD_DIM {
        let wi = &w_uk[i * KV_LORA_RANK..(i + 1) * KV_LORA_RANK];
        lat_ops::axpy_bf16(wi, qh[i], q_abs);
    }
    let n = sel.len();
    for (i, &t) in sel.iter().enumerate() {
        let lat = &cache[t * MLA_ROW..t * MLA_ROW + KV_LORA_RANK];
        att[i] = lat_ops::dot(q_abs, lat) * scale;
    }
    softmax(&mut att[..n]);
    for j in 0..KV_LORA_RANK {
        o_lat[j] = 0.0;
    }
    for (i, &t) in sel.iter().enumerate() {
        let lat = &cache[t * MLA_ROW..t * MLA_ROW + KV_LORA_RANK];
        lat_ops::axpy(lat, att[i], o_lat);
    }
    gemv::bf16::<V_HEAD_DIM, KV_LORA_RANK>(w_uv, o_lat, oh);
}

/// The head's latent-width (`KV_LORA_RANK`) vector ops: `acc += a · w` over a bf16
/// row, `Σ a·b`, `acc += p · x`. Eight wide on x86-64 with the scalar form's
/// per-element order (multiply, then add; the dot's eight lanes summed at the end).
mod lat_ops {
    use crate::KV_LORA_RANK as N;
    #[cfg(not(target_arch = "x86_64"))]
    use common::gemv::widen;

    #[inline(always)]
    pub fn axpy_bf16(w: &[u16], a: f32, acc: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            avx2::axpy_bf16(w.as_ptr(), a, acc.as_mut_ptr())
        }
        #[cfg(not(target_arch = "x86_64"))]
        for j in 0..N {
            acc[j] += widen(w[j]) * a;
        }
    }
    #[inline(always)]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            avx2::dot(a.as_ptr(), b.as_ptr())
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut acc = 0f32;
            for j in 0..N {
                acc += a[j] * b[j];
            }
            acc
        }
    }
    #[inline(always)]
    pub fn axpy(x: &[f32], p: f32, acc: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            avx2::axpy(x.as_ptr(), p, acc.as_mut_ptr())
        }
        #[cfg(not(target_arch = "x86_64"))]
        for j in 0..N {
            acc[j] += p * x[j];
        }
    }

    #[cfg(target_arch = "x86_64")]
    mod avx2 {
        use super::N;
        use core::arch::x86_64::*;
        const _: () = assert!(N % 16 == 0);

        #[inline(always)]
        unsafe fn widen8(p: *const u16) -> __m256 {
            _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(_mm_loadu_si128(p as *const __m128i))))
        }
        #[target_feature(enable = "avx2,fma")]
        pub unsafe fn axpy_bf16(w: *const u16, a: f32, acc: *mut f32) {
            let av = _mm256_set1_ps(a);
            let mut j = 0;
            while j < N {
                _mm256_storeu_ps(acc.add(j), _mm256_add_ps(_mm256_loadu_ps(acc.add(j)), _mm256_mul_ps(widen8(w.add(j)), av)));
                _mm256_storeu_ps(acc.add(j + 8), _mm256_add_ps(_mm256_loadu_ps(acc.add(j + 8)), _mm256_mul_ps(widen8(w.add(j + 8)), av)));
                j += 16;
            }
        }
        #[target_feature(enable = "avx2,fma")]
        pub unsafe fn dot(a: *const f32, b: *const f32) -> f32 {
            let (mut s0, mut s1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            let mut j = 0;
            while j < N {
                s0 = _mm256_add_ps(s0, _mm256_mul_ps(_mm256_loadu_ps(a.add(j)), _mm256_loadu_ps(b.add(j))));
                s1 = _mm256_add_ps(s1, _mm256_mul_ps(_mm256_loadu_ps(a.add(j + 8)), _mm256_loadu_ps(b.add(j + 8))));
                j += 16;
            }
            common::gemv::avx2::hsum(_mm256_add_ps(s0, s1))
        }
        #[target_feature(enable = "avx2,fma")]
        pub unsafe fn axpy(x: *const f32, p: f32, acc: *mut f32) {
            let pv = _mm256_set1_ps(p);
            let mut j = 0;
            while j < N {
                _mm256_storeu_ps(acc.add(j), _mm256_add_ps(_mm256_loadu_ps(acc.add(j)), _mm256_mul_ps(pv, _mm256_loadu_ps(x.add(j)))));
                _mm256_storeu_ps(acc.add(j + 8), _mm256_add_ps(_mm256_loadu_ps(acc.add(j + 8)), _mm256_mul_ps(pv, _mm256_loadu_ps(x.add(j + 8)))));
                j += 16;
            }
        }
    }
}

pub mod gfx11_consts {
    use crate::*;
    /// Rows of the fused indexer launch over the normed block input: the key's, the gate's, the head weights'.
    pub const IX_ROWS: usize = 2 * INDEX_HEAD_DIM + INDEX_N_HEADS;
    /// Rows of the output projection's input: every head's value output.
    pub const O_IN: usize = MLA_HEADS * V_HEAD_DIM;
    /// `mla_attend`: a workgroup per head; its LDS in floats: the head's query, `q_abs`, the
    /// latent mix, the scores over the selected positions (`ATT_SLOTS`), reduction scratch.
    /// Heads a workgroup takes: a selected latent row is read once for their dots and their mixes
    /// (a head a workgroup re-read the token's 4 MB of latents 128 times a layer at 2048 selected
    /// positions, 512 MB a launch from L2: 240 µs, 2026-10-17).
    pub const ATTEND_H: usize = 1;
    /// Rows a wave of `mla_qb` takes past one token: FP8 at four spilled (192 registers, 2026-10-17).
    pub const ROWS_QB: usize = 2;
    pub const ATTEND_GRID: u32 = (MLA_HEADS / ATTEND_H) as u32;
    pub const ATT_SLOTS: usize = DENSE_BELOW.div_ceil(256) * 256;
    pub const AT_Q: usize = 0;
    pub const AT_QABS: usize = AT_Q + ATTEND_H * Q_HEAD;
    /// The mixes take `q_abs`'s room: it is dead once the scores are in.
    pub const AT_OLAT: usize = AT_QABS;
    pub const AT_ATT: usize = AT_QABS + ATTEND_H * KV_LORA_RANK;
    pub const AT_RED: usize = AT_ATT + ATTEND_H * ATT_SLOTS;
    pub const ATTEND_LDS: usize = 4 * (AT_RED + ATTEND_H * 8);
    const _: () = assert!(MLA_HEADS % ATTEND_H == 0 && ATTEND_LDS <= 60 * 1024);
    /// `ix_select`'s LDS: a 256-bin histogram and the scan's eight wave totals, u32.
    pub const SELECT_LDS: usize = 4 * (256 + 8);
}

#[cfg(target_arch = "amdgpu")]
pub mod gfx11 {
    //! The MLA projections on the card. Over the normed block input the FFN's
    //! `ffn_gu` (FP8, width H) and the KDA's `rows_h` (bf16, width H) serve
    //! as they are: q_a, kv_a, and the indexer's key | gate | head weights as
    //! one bf16 launch. Here: `mla_qb` (the query's second half, FP8 rows of
    //! width Q_LORA_RANK over q_a's output rms-normed with q_a's norm as it is
    //! staged), `mla_iq` (the indexer's query, bf16 rows over the same), and
    //! `mla_o` (the output projection, FP8 rows of width MLA_HEADS·V_HEAD_DIM).
    pub use super::gfx11_consts::*;
    use crate::*;
    use common::gemv::gfx11::{rows_fmt, Bf16, Fp8};
    use common::gemv::{widen, ROWS};
    use common::math::exp;
    use core::arch::amdgpu::workitem_id_x;

    /// `out[i][row] = W[row] · rmsnorm_ln(qa[i])` for `[m][Q_LORA_RANK]` FP8 rows. Grid: rows_grid(m).
    #[no_mangle]
    pub extern "gpu-kernel" fn mla_qb(qa: *const f32, ln: *const u16, w: *const u8, s: *const f32, out: *mut f32, m: u32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            rows_fmt::<Q_LORA_RANK, true, Fp8, ROWS_QB>(tid, Fp8 { w, s }, ln, m, n, RMS_NORM_EPS, |i, s| *qa.add(i * Q_LORA_RANK + s * 256 + t), || {}, |row, i, y| *out.add(i * m + row) = y);
        }
    }

    /// The same for bf16 rows: the indexer's query. Grid: rows_grid(m).
    #[no_mangle]
    pub extern "gpu-kernel" fn mla_iq(qa: *const f32, ln: *const u16, w: *const u16, out: *mut f32, m: u32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            rows_fmt::<Q_LORA_RANK, true, Bf16, ROWS>(tid, Bf16(w), ln, m, n, RMS_NORM_EPS, |i, s| *qa.add(i * Q_LORA_RANK + s * 256 + t), || {}, |row, i, y| *out.add(i * m + row) = y);
        }
    }

    /// `out[i][row] = W[row] · heads[i]` for `[m][O_IN]` FP8 rows: the output projection. Grid: rows_grid(m).
    #[no_mangle]
    pub extern "gpu-kernel" fn mla_o(heads: *const f32, w: *const u8, s: *const f32, out: *mut f32, m: u32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            rows_fmt::<O_IN, false, Fp8, ROWS>(tid, Fp8 { w, s }, core::ptr::null(), m, n, 0.0, |i, s| *heads.add(i * O_IN + s * 256 + t), || {}, |row, i, y| *out.add(i * m + row) = y);
        }
    }
    const _: () = assert!(Q_LORA_RANK % 256 == 0 && O_IN % 256 == 0);

    /// Absorbed MLA for one token and `ATTEND_H` heads over the token's selected positions (`sel[i]`,
    /// `counts[i]` of them), the latents resident (`lat: [ctx][KV_LORA_RANK]`): `q_abs = W_ukᵀ q` for
    /// each head (a thread two latent channels, the products summed over the query's channels in
    /// order, as the host does), the scores (a thread a position: the row read once, a dot a head),
    /// each head's softmax (its own thread partition and wave order), the latent mixes (a thread two
    /// channels, the positions in order, each latent value read once for the heads), and `W_uv` over
    /// each mix a wave-row at a time into `heads[i][h]`. Every (head, position) product runs in the
    /// one-head kernel's order: the same bits. Grid: ATTEND_GRID × the step's tokens; LDS: ATTEND_LDS.
    /// `q: [tokens][MLA_HEADS][Q_HEAD]`, `kvb` the layer's `[MLA_HEADS][KVB_HEAD][KV_LORA_RANK]` bf16,
    /// `sel: [tokens][DENSE_BELOW]`, `heads: [tokens][MLA_HEADS][V_HEAD_DIM]`.
    #[no_mangle]
    pub extern "gpu-kernel" fn mla_attend(q: *const f32, kvb: *const u16, lat: *const f32, sel: *const u32, counts: *const u32, heads: *mut f32) {
        unsafe {
            use common::gemv::gfx11::bf16_row;
            use common::gfx11::{lds, wave_max, wave_sum, workgroup_sync};
            use core::arch::amdgpu::workgroup_id_x;
            const D: usize = KV_LORA_RANK;
            const AH: usize = ATTEND_H;
            let wg = workgroup_id_x() as usize;
            let (i, h0) = (wg / (MLA_HEADS / AH), (wg % (MLA_HEADS / AH)) * AH);
            let (q, sel, heads) = (q.add(i * MLA_HEADS * Q_HEAD), sel.add(i * DENSE_BELOW), heads.add(i * MLA_HEADS * V_HEAD_DIM));
            let n = *counts.add(i) as usize;
            let tid = workitem_id_x();
            let t = tid as usize;
            let (lane, wave) = (tid % 32, (tid / 32) as usize);
            let l: *mut f32 = lds();
            let (qh, qabs, olat, att, red) = (l.add(AT_Q), l.add(AT_QABS), l.add(AT_OLAT), l.add(AT_ATT), l.add(AT_RED));
            let w_uk = |h: usize| kvb.add((h0 + h) * KVB_HEAD * D);
            let scale = 1.0 / common::math::sqrt(Q_HEAD as f32);
            // the heads' queries
            let mut h = 0;
            while h < AH {
                if t < Q_HEAD {
                    *qh.add(h * Q_HEAD + t) = *q.add((h0 + h) * Q_HEAD + t);
                }
                h += 1;
            }
            workgroup_sync();
            // q_abs: two latent channels a thread, a head at a time
            let mut h = 0;
            while h < AH {
                let (wk, qk) = (w_uk(h), qh.add(h * Q_HEAD));
                let mut j = t;
                while j < D {
                    let mut acc = 0f32;
                    let mut c = 0;
                    while c < QK_NOPE_HEAD_DIM {
                        acc += widen(*wk.add(c * D + j)) * *qk.add(c);
                        c += 1;
                    }
                    *qabs.add(h * D + j) = acc;
                    j += 256;
                }
                h += 1;
            }
            workgroup_sync();
            // the scores: a position a thread, its row read once for the heads' dots; each head's max
            let mut mx = [f32::NEG_INFINITY; AH];
            let mut p = t;
            while p < n {
                let row = lat.add(*sel.add(p) as usize * D);
                let mut h = 0;
                while h < AH {
                    let qa = qabs.add(h * D);
                    let mut acc = 0f32;
                    let mut j = 0;
                    while j < D {
                        acc += *qa.add(j) * *row.add(j);
                        j += 1;
                    }
                    let sc = acc * scale;
                    *att.add(h * ATT_SLOTS + p) = sc;
                    if sc > mx[h] {
                        mx[h] = sc;
                    }
                    h += 1;
                }
                p += 256;
            }
            // each head's softmax: the workgroup's max, the exponentials, their sum (the one-head kernel's
            // thread partition and wave order), then normalized
            let mut h = 0;
            while h < AH {
                let m = wave_max(lane, mx[h]);
                if lane == 0 {
                    *red.add(h * 8 + wave) = m;
                }
                h += 1;
            }
            workgroup_sync();
            let mut h = 0;
            while h < AH {
                let mut m = *red.add(h * 8);
                let mut w = 1;
                while w < 8 {
                    if *red.add(h * 8 + w) > m {
                        m = *red.add(h * 8 + w);
                    }
                    w += 1;
                }
                let ah = att.add(h * ATT_SLOTS);
                let mut sum = 0f32;
                let mut p = t;
                while p < n {
                    let e = exp(*ah.add(p) - m);
                    *ah.add(p) = e;
                    sum += e;
                    p += 256;
                }
                let s = wave_sum(lane, sum);
                workgroup_sync(); // every wave has read the maxima before the sums land in the same scratch
                if lane == 0 {
                    *red.add(h * 8 + wave) = s;
                }
                workgroup_sync();
                let mut total = 0f32;
                let mut w = 0;
                while w < 8 {
                    total += *red.add(h * 8 + w);
                    w += 1;
                }
                let inv = 1.0 / total;
                let mut p = t;
                while p < n {
                    *ah.add(p) *= inv;
                    p += 256;
                }
                h += 1;
            }
            workgroup_sync();
            // the latent mixes: two channels a thread, the positions in order, each value read once for the heads
            let mut j = t;
            while j < D {
                let mut acc = [0f32; AH];
                let mut p = 0;
                while p < n {
                    let v = *lat.add(*sel.add(p) as usize * D + j);
                    let mut h = 0;
                    while h < AH {
                        acc[h] += *att.add(h * ATT_SLOTS + p) * v;
                        h += 1;
                    }
                    p += 1;
                }
                let mut h = 0;
                while h < AH {
                    *olat.add(h * D + j) = acc[h];
                    h += 1;
                }
                j += 256;
            }
            workgroup_sync();
            // W_uv over each head's mix: each wave its rows
            let mut h = 0;
            while h < AH {
                let w_uv = w_uk(h).add(QK_NOPE_HEAD_DIM * D);
                let mut r = wave * (V_HEAD_DIM / 8);
                while r < (wave + 1) * (V_HEAD_DIM / 8) {
                    let y = bf16_row::<D>(lane, w_uv.add(r * D), olat.add(h * D));
                    if lane == 0 {
                        *heads.add((h0 + h) * V_HEAD_DIM + r) = y;
                    }
                    r += 1;
                }
                h += 1;
            }
        }
    }
    /// `out[i][row] = W[row] · x[i]` for the draft layer's `eh_proj` (`[H][2·H]` bf16) over `x: [n][2·H]`
    /// (the next token's normed embedding, then the normed hidden state). Grid: rows_grid(m, n, ROWS).
    #[no_mangle]
    pub extern "gpu-kernel" fn mtp_eh(x: *const f32, w: *const u16, out: *mut f32, m: u32, n: u32) {
        unsafe {
            use common::gemv::gfx11::rows_over;
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            rows_over::<{ 2 * H }, false, ROWS>(tid, w, core::ptr::null(), m, n, 0.0, |i, s| *x.add(i * 2 * H + s * 256 + t), || {}, |row, i, y| *out.add(i * m + row) = y);
        }
    }
    /// The indexer's scores on the card: a thread a (token, pool) of the step, the host's
    /// `score_pools` arithmetic — the dot over the key's channels in order, the heads in order,
    /// the head weights scaled once — so the picks are the host's (`scale` and `wscale` come from
    /// the host, computed once there). `iq: [tokens][INDEX_N_HEADS][INDEX_HEAD_DIM]`, `ix:
    /// [tokens][IX_ROWS]` (the head weights at `2 · INDEX_HEAD_DIM`), `pools: [pools]
    /// [INDEX_HEAD_DIM]`, `score: [tokens][stride]`; token `i` at `pos0 + i` scores its complete
    /// pools when it selects among them. Grid: tokens × `nb` (256 pools a workgroup). One host
    /// thread scored a token's 5.4k pools at 21k positions in 6 ms (2026-10-17).
    #[no_mangle]
    pub extern "gpu-kernel" fn ix_score(iq: *const f32, ix: *const f32, pools: *const f32, score: *mut f32, pos0: u32, nb: u32, stride: u32, scale: f32, wscale: f32) {
        unsafe {
            use core::arch::amdgpu::workgroup_id_x;
            let wg = workgroup_id_x();
            let i = (wg / nb) as usize;
            let j = ((wg % nb) * 256 + workitem_id_x()) as usize;
            let pos = pos0 as usize + i;
            let visible = (pos + 1) / INDEX_KPOOL;
            if visible <= super::KEEP || j >= visible {
                return;
            }
            let key = pools.add(j * INDEX_HEAD_DIM);
            let q = iq.add(i * INDEX_N_HEADS * INDEX_HEAD_DIM);
            let iw = ix.add(i * IX_ROWS + 2 * INDEX_HEAD_DIM);
            let mut sc = 0f32;
            let mut h = 0;
            while h < INDEX_N_HEADS {
                let qh = q.add(h * INDEX_HEAD_DIM);
                let mut d = 0f32;
                let mut c = 0;
                while c < INDEX_HEAD_DIM {
                    d += *qh.add(c) * *key.add(c);
                    c += 1;
                }
                let d = d * scale;
                sc += (*iw.add(h) * wscale) * if d > 0.0 { d } else { 0.0 };
                h += 1;
            }
            *score.add(i * stride as usize + j) = sc;
        }
    }
    /// The indexer's selection on the card: a workgroup a token, `pick`'s result — below `KEEP`
    /// complete pools every position; past it the `KEEP` best pools by score, lower index first on
    /// a tie, in index order, then the incomplete tail — into `sel[i]` and `counts[i]`, what
    /// `mla_attend` reads. The `KEEP`th largest score is found by a radix select over the scores'
    /// order-preserving bits (four passes of 256 bins in LDS); then a pool is taken if its score is
    /// greater, or equal and among the first of the equals by index (a scan over the pools in index
    /// order, chunks of 256). The host's pick and this agree on the set exactly. Grid: tokens;
    /// LDS: SELECT_LDS. `score: [tokens][stride]`.
    #[no_mangle]
    pub extern "gpu-kernel" fn ix_select(score: *const f32, stride: u32, pos0: u32, sel: *mut u32, counts: *mut u32) {
        unsafe {
            use common::gfx11::{lds, workgroup_scan_excl, workgroup_sync};
            use core::arch::amdgpu::workgroup_id_x;
            use core::sync::atomic::{AtomicU32, Ordering};
            let i = workgroup_id_x() as usize;
            let pos = pos0 as usize + i;
            let visible = (pos + 1) / INDEX_KPOOL;
            let tid = workitem_id_x();
            let t = tid as usize;
            let (sel, score) = (sel.add(i * DENSE_BELOW), score.add(i * stride as usize));
            if visible <= super::KEEP {
                // every position
                let mut p = t;
                while p <= pos {
                    *sel.add(p) = p as u32;
                    p += 256;
                }
                if t == 0 {
                    *counts.add(i) = (pos + 1) as u32;
                }
                return;
            }
            let l: *mut u32 = lds();
            let (hist, red) = (l, l.add(256));
            // a score's bits in an order-preserving unsigned key
            let key = |j: usize| -> u32 {
                let b = (*score.add(j)).to_bits();
                if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }
            };
            // the KEEPth largest key: four radix passes from the top byte
            let mut prefix = 0u32;
            let mut mask = 0u32;
            let mut want = super::KEEP as u32; // of the keys matching the prefix, how many the top KEEP still take
            let mut above = 0u32; // keys strictly greater than the final threshold
            let mut shift = 24;
            loop {
                *hist.add(t) = 0;
                workgroup_sync();
                let mut j = t;
                while j < visible {
                    let k = key(j);
                    if k & mask == prefix {
                        AtomicU32::from_ptr(hist.add(((k >> shift) & 255) as usize)).fetch_add(1, Ordering::Relaxed);
                    }
                    j += 256;
                }
                workgroup_sync();
                // every thread walks the bins from the top to the same answer
                let mut cum = 0u32;
                let mut b = 256usize;
                let mut digit = 0u32;
                while b > 0 {
                    b -= 1;
                    let c = *hist.add(b);
                    if cum + c >= want {
                        digit = b as u32;
                        break;
                    }
                    cum += c;
                }
                prefix |= digit << shift;
                mask |= 255 << shift;
                above += cum;
                want -= cum;
                workgroup_sync();
                if shift == 0 {
                    break;
                }
                shift -= 8;
            }
            let thr = prefix;
            let need = super::KEEP as u32 - above; // of the keys equal to the threshold, the first by index
            // the pools in index order, chunks of 256: taken if greater than the threshold, or equal and
            // among the first `need` equals; each taken pool's four positions at its rank
            let mut carry_eq = 0u32;
            let mut carry_sel = 0u32;
            let mut base = 0;
            while base < visible {
                let j = base + t;
                let k = if j < visible { key(j) } else { 0 };
                let gt = j < visible && k > thr;
                let eq = j < visible && k == thr;
                let (eq_before, eq_total) = workgroup_scan_excl(tid, eq as u32, red);
                let take = gt || (eq && carry_eq + eq_before < need);
                let (rank, taken) = workgroup_scan_excl(tid, take as u32, red);
                if take {
                    let r = (carry_sel + rank) as usize * INDEX_KPOOL;
                    let mut q = 0;
                    while q < INDEX_KPOOL {
                        *sel.add(r + q) = (j * INDEX_KPOOL + q) as u32;
                        q += 1;
                    }
                }
                carry_eq += eq_total;
                carry_sel += taken;
                base += 256;
            }
            // the incomplete tail's positions
            let tail0 = visible * INDEX_KPOOL;
            let mut p = tail0 + t;
            while p <= pos {
                *sel.add(super::KEEP * INDEX_KPOOL + p - tail0) = p as u32;
                p += 256;
            }
            if t == 0 {
                *counts.add(i) = (super::KEEP * INDEX_KPOOL + pos + 1 - tail0) as u32;
            }
        }
    }
    const _: () = assert!(V_HEAD_DIM % 8 == 0 && KV_LORA_RANK % 256 == 0 && Q_HEAD <= 256);
}
