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
        let mut acc = 0f32;
        for i in 0..INDEX_KPOOL {
            acc += logit[i] / sum * cache[(j * INDEX_KPOOL + i) * MLA_ROW + KV_LORA_RANK + c];
        }
        out[c] = acc;
    }
}

/// The positions query `pos` attends, ascending, into `sel`; how many.
#[inline(always)]
fn select(pos: usize, iq: &[f32], iw: &[f32], pools: &[f32], pscore: &mut [f32], sel: &mut [usize]) -> usize {
    let visible = (pos + 1) / INDEX_KPOOL;
    let keep = INDEX_TOPK / INDEX_KPOOL;
    let mut n = 0;
    if visible <= keep {
        for p in 0..visible * INDEX_KPOOL {
            sel[n] = p;
            n += 1;
        }
    } else {
        let scale = 1.0 / sqrt(INDEX_HEAD_DIM as f32);
        for j in 0..visible {
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
        // the `keep` best pools by score, lower index first on a tie, in index order: a
        // selection pass per winner (O(keep · visible), fine here), a taken pool marked −∞
        let mut taken = 0;
        let mut picks = [0usize; INDEX_TOPK / INDEX_KPOOL];
        while taken < keep {
            let mut best = usize::MAX;
            for j in 0..visible {
                if pscore[j] != f32::NEG_INFINITY && (best == usize::MAX || pscore[j] > pscore[best]) {
                    best = j;
                }
            }
            picks[taken] = best;
            pscore[best] = f32::NEG_INFINITY;
            taken += 1;
        }
        picks.sort_unstable();
        for &j in &picks {
            for p in j * INDEX_KPOOL..(j + 1) * INDEX_KPOOL {
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
pub fn finish_row(w: &weights::Mla, s: &Scratch, latent: &mut [f32], ikey: &mut [f32]) {
    rmsnorm::<KV_LORA_RANK>(s.kva, &w.kv_a_ln, RMS_NORM_EPS, latent);
    let mut mean = 0f32;
    for &v in s.ik.iter() {
        mean += v;
    }
    let mean = mean / INDEX_HEAD_DIM as f32;
    let mut var = 0f32;
    for &v in s.ik.iter() {
        var += (v - mean) * (v - mean);
    }
    let rs = 1.0 / sqrt(var / INDEX_HEAD_DIM as f32 + 1e-6);
    for c in 0..INDEX_HEAD_DIM {
        ikey[c] = (s.ik[c] - mean) * rs * w.ix_k_norm_w[c] + w.ix_k_norm_b[c];
    }
}

/// The pool key a completed pool gains, the selection, and absorbed MLA over the
/// selected positions, a head per item on its own scratch, into `s.heads`.
#[inline(always)]
pub fn attend(l: usize, w: &weights::Mla, cache: &[f32], pools: &mut [f32], pos: usize, s: &mut Scratch, pool: &dyn Pool, tr: &mut impl Trace) {
    if (pos + 1) % INDEX_KPOOL == 0 {
        let j = (pos + 1) / INDEX_KPOOL - 1;
        let p = &mut pools[j * INDEX_HEAD_DIM..(j + 1) * INDEX_HEAD_DIM];
        pool_key(w, cache, j, p);
    }
    let wscale = 1.0 / sqrt(INDEX_N_HEADS as f32);
    for v in s.iw.iter_mut() {
        *v *= wscale;
    }
    let n = select(pos, s.iq, s.iw, pools, s.pscore, s.sel);
    tr.edge(l, "self_attn.indexer.count", &[n as f32]);

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
    use core::arch::amdgpu::workitem_id_x;

    /// `out[i][row] = W[row] · rmsnorm_ln(qa[i])` for `[m][Q_LORA_RANK]` FP8 rows. Grid: rows_grid(m).
    #[no_mangle]
    pub extern "gpu-kernel" fn mla_qb(qa: *const f32, ln: *const u16, w: *const u8, s: *const f32, out: *mut f32, m: u32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            rows_fmt::<Q_LORA_RANK, true, Fp8>(tid, Fp8 { w, s }, ln, m, n, RMS_NORM_EPS, |i, s| *qa.add(i * Q_LORA_RANK + s * 256 + t), || {}, |row, i, y| *out.add(i * m + row) = y);
        }
    }

    /// The same for bf16 rows: the indexer's query. Grid: rows_grid(m).
    #[no_mangle]
    pub extern "gpu-kernel" fn mla_iq(qa: *const f32, ln: *const u16, w: *const u16, out: *mut f32, m: u32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            rows_fmt::<Q_LORA_RANK, true, Bf16>(tid, Bf16(w), ln, m, n, RMS_NORM_EPS, |i, s| *qa.add(i * Q_LORA_RANK + s * 256 + t), || {}, |row, i, y| *out.add(i * m + row) = y);
        }
    }

    /// `out[i][row] = W[row] · heads[i]` for `[m][O_IN]` FP8 rows: the output projection. Grid: rows_grid(m).
    #[no_mangle]
    pub extern "gpu-kernel" fn mla_o(heads: *const f32, w: *const u8, s: *const f32, out: *mut f32, m: u32, n: u32) {
        unsafe {
            let tid = workitem_id_x();
            let (m, n) = (m as usize, n as usize);
            let t = tid as usize;
            rows_fmt::<O_IN, false, Fp8>(tid, Fp8 { w, s }, core::ptr::null(), m, n, 0.0, |i, s| *heads.add(i * O_IN + s * 256 + t), || {}, |row, i, y| *out.add(i * m + row) = y);
        }
    }
    const _: () = assert!(Q_LORA_RANK % 256 == 0 && O_IN % 256 == 0);
}
