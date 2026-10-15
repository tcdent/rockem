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
use common::gemv::{self, widen};
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

/// `xn` is the normed block input (`[H]`); the cache holds rows `0..pos` and
/// gains row `pos`; `out` is the block output (`[H]`).
#[inline(always)]
pub fn step(l: usize, w: &weights::Mla, xn: &[f32], cache: &mut [f32], pools: &mut [f32], pos: usize, s: &mut Scratch, pool: &dyn Pool, tr: &mut impl Trace, out: &mut [f32]) {
    gemv::fp8_par::<Q_LORA_RANK, H>(w.q_a.w, w.q_a.s, xn, s.qa, pool);
    rmsnorm::<Q_LORA_RANK>(s.qa, &w.q_a_ln, RMS_NORM_EPS, s.qr);
    gemv::fp8_par::<{ MLA_HEADS * Q_HEAD }, Q_LORA_RANK>(w.q_b.w, w.q_b.s, s.qr, s.q, pool);
    // this position's row: the normed latent, the indexer's key (LayerNorm, eps 1e-6) and gate
    gemv::fp8_par::<KV_LORA_RANK, H>(w.kv_a.w, w.kv_a.s, xn, s.kva, pool);
    let row = &mut cache[pos * MLA_ROW..(pos + 1) * MLA_ROW];
    let (latent, rest) = row.split_at_mut(KV_LORA_RANK);
    let (ikey, igate) = rest.split_at_mut(INDEX_HEAD_DIM);
    rmsnorm::<KV_LORA_RANK>(s.kva, &w.kv_a_ln, RMS_NORM_EPS, latent);
    gemv::bf16::<INDEX_HEAD_DIM, H>(w.ix_wk, xn, s.ik);
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
    gemv::bf16::<INDEX_HEAD_DIM, H>(w.ix_gate, xn, igate);
    if (pos + 1) % INDEX_KPOOL == 0 {
        let j = (pos + 1) / INDEX_KPOOL - 1;
        let (c, p) = (&*cache, &mut pools[j * INDEX_HEAD_DIM..(j + 1) * INDEX_HEAD_DIM]);
        pool_key(w, c, j, p);
    }
    gemv::bf16_par::<{ INDEX_N_HEADS * INDEX_HEAD_DIM }, Q_LORA_RANK>(w.ix_wq_b, s.qr, s.iq, pool);
    gemv::bf16::<INDEX_N_HEADS, H>(w.ix_weights, xn, s.iw);
    let wscale = 1.0 / sqrt(INDEX_N_HEADS as f32);
    for v in s.iw.iter_mut() {
        *v *= wscale;
    }
    let n = select(pos, s.iq, s.iw, pools, s.pscore, s.sel);
    tr.edge(l, "self_attn.indexer.count", &[n as f32]);

    // absorbed MLA over the selected positions, a head per item on its own scratch and output
    let (q, sel, cache_r) = (&*s.q, &s.sel[..n], &*cache);
    let (heads_p, hs_p) = (s.heads.as_mut_ptr() as usize, s.hs.as_mut_ptr() as usize);
    pool.run(MLA_HEADS, &move |h| {
        let oh = unsafe { core::slice::from_raw_parts_mut((heads_p as *mut f32).add(h * V_HEAD_DIM), V_HEAD_DIM) };
        let hs = unsafe { core::slice::from_raw_parts_mut((hs_p as *mut f32).add(h * HEAD_SCRATCH), HEAD_SCRATCH) };
        let (q_abs, rest) = hs.split_at_mut(KV_LORA_RANK);
        let (o_lat, att) = rest.split_at_mut(KV_LORA_RANK);
        head(w, h, q, sel, cache_r, q_abs, o_lat, att, oh);
    });
    tr.edge(l, "self_attn.o_proj.in", s.heads);
    gemv::fp8_par::<H, { MLA_HEADS * V_HEAD_DIM }>(w.o.w, w.o.s, s.heads, out, pool);
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
        let qi = qh[i];
        for j in 0..KV_LORA_RANK {
            q_abs[j] += widen(wi[j]) * qi;
        }
    }
    let n = sel.len();
    for (i, &t) in sel.iter().enumerate() {
        let lat = &cache[t * MLA_ROW..t * MLA_ROW + KV_LORA_RANK];
        let mut acc = 0f32;
        for j in 0..KV_LORA_RANK {
            acc += q_abs[j] * lat[j];
        }
        att[i] = acc * scale;
    }
    softmax(&mut att[..n]);
    for j in 0..KV_LORA_RANK {
        o_lat[j] = 0.0;
    }
    for (i, &t) in sel.iter().enumerate() {
        let lat = &cache[t * MLA_ROW..t * MLA_ROW + KV_LORA_RANK];
        let p = att[i];
        for j in 0..KV_LORA_RANK {
            o_lat[j] += p * lat[j];
        }
    }
    gemv::bf16::<V_HEAD_DIM, KV_LORA_RANK>(w_uv, o_lat, oh);
}
