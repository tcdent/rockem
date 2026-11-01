//! NoPE MLA over the indexer's selection, a run of positions: the forward is the inference
//! crate's step (`mla::row`, `project`, `finish_row`, `select_at`, `heads`, `output`) one
//! position at a time, keeping what the backward needs; the backward runs the run's queries
//! and then the run's own latents.
//!
//! The absorbed form the node attends in (`mla::head`), per query and head:
//!
//! ```text
//! q_abs = W_ukᵀ q_h       att_i = softmax_i(⟨q_abs, latent_i⟩ · scale)  over the selected i
//! o_lat = Σ_i att_i latent_i       out_h = W_uv · o_lat
//! ```
//!
//! so its backward accumulates straight into the **latents'** gradient (`[KV_LORA_RANK]` a
//! position — never an expanded K/V): `d o_lat = W_uvᵀ d out_h`; `d att_i = ⟨d o_lat, latent_i⟩`,
//! `d latent_i += att_i · d o_lat`; the softmax; `d q_abs = scale · Σ_i d s_i latent_i`,
//! `d latent_i += scale · d s_i · q_abs`; `d q_h = W_uk · d q_abs`. The query path follows
//! (`q_b`, the norm, `q_a`), and a position's latent, once every query attending it has run,
//! goes back through its norm and `kv_a`. The selection is discrete: the indexer's projections
//! get no gradient. The cache's bf16 rounding of the latent passes the gradient straight through.
//!
//! Runs are processed in reverse: a run's queries contribute to earlier positions' latents
//! (`dlatent` is the caller's, over every position), and its own positions are complete once
//! its queries have run, so `backward` finishes them.

use common::math::sqrt;
use common::pool::Serial;
use common::trace::NoTrace;
use common::gemv;
use glm_5_3_flash::layout::{MLA_ROW_U16};
use glm_5_3_flash::mla::{self, HEAD_SCRATCH};
use glm_5_3_flash::weights;
use glm_5_3_flash::{DENSE_BELOW, H, INDEX_HEAD_DIM, INDEX_KPOOL, INDEX_N_HEADS, KVB_HEAD, KV_LORA_RANK, MLA_HEADS, QK_NOPE_HEAD_DIM, Q_HEAD, Q_LORA_RANK, RMS_NORM_EPS, V_HEAD_DIM};
use train_common::{bf16, fp8, rmsnorm};

/// The inference step's scratch, owned, for contexts up to `max_ctx`.
pub struct Scratch {
    qr: Vec<f32>,
    qa: Vec<f32>,
    q: Vec<f32>,
    kva: Vec<f32>,
    ik: Vec<f32>,
    iq: Vec<f32>,
    iw: Vec<f32>,
    pscore: Vec<f32>,
    pidx: Vec<u32>,
    sel: Vec<usize>,
    heads: Vec<f32>,
    hs: Vec<f32>,
}

impl Scratch {
    pub fn new(max_ctx: usize) -> Self {
        let pools = max_ctx.div_ceil(INDEX_KPOOL) + 1;
        Scratch { qr: vec![0.0; Q_LORA_RANK], qa: vec![0.0; Q_LORA_RANK], q: vec![0.0; MLA_HEADS * Q_HEAD], kva: vec![0.0; KV_LORA_RANK], ik: vec![0.0; INDEX_HEAD_DIM], iq: vec![0.0; INDEX_N_HEADS * INDEX_HEAD_DIM], iw: vec![0.0; INDEX_N_HEADS], pscore: vec![0.0; pools], pidx: vec![0; pools], sel: vec![0; DENSE_BELOW], heads: vec![0.0; MLA_HEADS * V_HEAD_DIM], hs: vec![0.0; MLA_HEADS * HEAD_SCRATCH] }
    }
    fn view(&mut self) -> mla::Scratch<'_> {
        mla::Scratch { qr: &mut self.qr, qa: &mut self.qa, q: &mut self.q, kva: &mut self.kva, ik: &mut self.ik, iq: &mut self.iq, iw: &mut self.iw, pscore: &mut self.pscore, pidx: &mut self.pidx, sel: &mut self.sel, heads: &mut self.heads, hs: &mut self.hs }
    }
}

/// The layer's state as the node keeps it: the packed cache rows and the pool keys.
pub struct State {
    cache: Vec<u32>, // [max_ctx][MLA_ROW_U16] u16, 4-aligned (the gate is f32)
    pub pools: Vec<u16>,
    max_ctx: usize,
}

impl State {
    pub fn new(max_ctx: usize) -> Self {
        State { cache: vec![0; max_ctx * MLA_ROW_U16 / 2], pools: vec![0; (max_ctx.div_ceil(INDEX_KPOOL) + 1) * INDEX_HEAD_DIM], max_ctx }
    }
    pub fn cache(&self) -> &[u16] {
        unsafe { core::slice::from_raw_parts(self.cache.as_ptr() as *const u16, self.max_ctx * MLA_ROW_U16) }
    }
    pub fn cache_mut(&mut self) -> &mut [u16] {
        unsafe { core::slice::from_raw_parts_mut(self.cache.as_mut_ptr() as *mut u16, self.max_ctx * MLA_ROW_U16) }
    }
}

/// What a run's forward keeps, per position.
pub struct Saved {
    pub pos0: usize,
    pub n: usize,
    pub qa: Vec<f32>,  // [n][Q_LORA_RANK]: the query's LoRA residual before its norm
    pub q: Vec<f32>,   // [n][MLA_HEADS·Q_HEAD]
    pub kva: Vec<f32>, // [n][KV_LORA_RANK]: the latent before its norm
    pub sel: Vec<Vec<usize>>,
    pub hs: Vec<f32>, // [n][MLA_HEADS][HEAD_SCRATCH]: q_abs | o_lat | att
}

impl Saved {
    pub fn new(pos0: usize, n: usize) -> Self {
        Saved { pos0, n, qa: vec![0.0; n * Q_LORA_RANK], q: vec![0.0; n * MLA_HEADS * Q_HEAD], kva: vec![0.0; n * KV_LORA_RANK], sel: vec![Vec::new(); n], hs: vec![0.0; n * MLA_HEADS * HEAD_SCRATCH] }
    }
}

/// `n` positions of `xn` (`[n][H]`) at `pos0..`, filling the state's rows; `y: [n][H]` written.
pub fn forward(l: usize, w: &weights::Mla, xn: &[f32], pos0: usize, n: usize, st: &mut State, s: &mut Scratch, saved: &mut Saved, y: &mut [f32]) {
    debug_assert!(saved.pos0 == pos0 && saved.n == n);
    for t in 0..n {
        let pos = pos0 + t;
        let mut ms = s.view();
        {
            let (latent, ikey, igate) = mla::row(st.cache_mut(), pos);
            mla::project(w, &xn[t * H..(t + 1) * H], &mut ms, igate, &Serial);
            mla::finish_row(w, ms.kva, ms.ik, latent, ikey);
        }
        let (cache, pools) = (&st.cache, &mut st.pools);
        let cache: &[u16] = unsafe { core::slice::from_raw_parts(cache.as_ptr() as *const u16, st.max_ctx * MLA_ROW_U16) };
        let count = mla::select_at(l, w, cache, pools, pos, &mut ms, &mut NoTrace);
        mla::heads(l, w, cache, &mut ms, count, &Serial, &mut NoTrace);
        mla::output(w, ms.heads, &mut y[t * H..(t + 1) * H], &Serial);
        saved.qa[t * Q_LORA_RANK..(t + 1) * Q_LORA_RANK].copy_from_slice(ms.qa);
        saved.q[t * MLA_HEADS * Q_HEAD..(t + 1) * MLA_HEADS * Q_HEAD].copy_from_slice(ms.q);
        saved.kva[t * KV_LORA_RANK..(t + 1) * KV_LORA_RANK].copy_from_slice(ms.kva);
        saved.sel[t] = ms.sel[..count].to_vec();
        saved.hs[t * MLA_HEADS * HEAD_SCRATCH..(t + 1) * MLA_HEADS * HEAD_SCRATCH].copy_from_slice(ms.hs);
    }
}

/// The run's queries: given `dy: [n][H]`, accumulate the attended positions' latent gradients
/// (`dlatent: [positions][KV_LORA_RANK]`, `+=`) and the query path's share of `dxn` (`+=`).
pub fn attend_backward(w: &weights::Mla, st: &State, sv: &Saved, dy: &[f32], dlatent: &mut [f32], dxn: &mut [f32]) {
    let cache = st.cache();
    let scale = 1.0 / sqrt(Q_HEAD as f32);
    let mut dheads = vec![0f32; MLA_HEADS * V_HEAD_DIM];
    let mut dq = vec![0f32; MLA_HEADS * Q_HEAD];
    let mut dqr = vec![0f32; Q_LORA_RANK];
    let mut dqa = vec![0f32; Q_LORA_RANK];
    let mut do_lat = vec![0f32; KV_LORA_RANK];
    let mut dq_abs = vec![0f32; KV_LORA_RANK];
    let mut datt = vec![0f32; DENSE_BELOW];
    let mut lat = vec![0f32; KV_LORA_RANK];
    for t in 0..sv.n {
        // d heads = Woᵀ · dy
        dheads.iter_mut().for_each(|v| *v = 0.0);
        fp8::rows_t::<{ MLA_HEADS * V_HEAD_DIM }>(w.o.w, w.o.s, &dy[t * H..(t + 1) * H], &mut dheads, 0, H);
        let sel = &sv.sel[t];
        for h in 0..MLA_HEADS {
            let hs = &sv.hs[(t * MLA_HEADS + h) * HEAD_SCRATCH..(t * MLA_HEADS + h + 1) * HEAD_SCRATCH];
            let (q_abs, att) = (&hs[..KV_LORA_RANK], &hs[2 * KV_LORA_RANK..2 * KV_LORA_RANK + sel.len()]);
            let w_uk = &w.kv_b[h * KVB_HEAD * KV_LORA_RANK..(h * KVB_HEAD + QK_NOPE_HEAD_DIM) * KV_LORA_RANK];
            let w_uv = &w.kv_b[(h * KVB_HEAD + QK_NOPE_HEAD_DIM) * KV_LORA_RANK..(h + 1) * KVB_HEAD * KV_LORA_RANK];
            // d o_lat = W_uvᵀ · d out_h
            do_lat.iter_mut().for_each(|v| *v = 0.0);
            bf16::dense_t::<KV_LORA_RANK>(V_HEAD_DIM, w_uv, &dheads[h * V_HEAD_DIM..(h + 1) * V_HEAD_DIM], &mut do_lat);
            // d att_i = ⟨d o_lat, latent_i⟩; d latent_i += att_i · d o_lat
            let mut dot = 0f32;
            for (i, &p) in sel.iter().enumerate() {
                let li = mla::latent(cache, p);
                let dl = &mut dlatent[p * KV_LORA_RANK..(p + 1) * KV_LORA_RANK];
                let mut d = 0f32;
                for c in 0..KV_LORA_RANK {
                    let lc = gemv::widen(li[c]);
                    d += do_lat[c] * lc;
                    dl[c] += att[i] * do_lat[c];
                }
                datt[i] = d;
                dot += att[i] * d;
            }
            // the softmax; d q_abs = scale · Σ d s_i latent_i; d latent_i += scale · d s_i · q_abs
            dq_abs.iter_mut().for_each(|v| *v = 0.0);
            for (i, &p) in sel.iter().enumerate() {
                let ds = att[i] * (datt[i] - dot) * scale;
                let li = mla::latent(cache, p);
                let dl = &mut dlatent[p * KV_LORA_RANK..(p + 1) * KV_LORA_RANK];
                for c in 0..KV_LORA_RANK {
                    lat[c] = gemv::widen(li[c]);
                    dq_abs[c] += ds * lat[c];
                    dl[c] += ds * q_abs[c];
                }
            }
            // d q_h = W_uk · d q_abs
            gemv::bf16::<QK_NOPE_HEAD_DIM, KV_LORA_RANK>(w_uk, &dq_abs, &mut dq[h * Q_HEAD..(h + 1) * Q_HEAD]);
        }
        // the query path: d qr = q_bᵀ dq; the norm; d xn += q_aᵀ d qa
        dqr.iter_mut().for_each(|v| *v = 0.0);
        fp8::rows_t::<Q_LORA_RANK>(w.q_b.w, w.q_b.s, &dq, &mut dqr, 0, MLA_HEADS * Q_HEAD);
        dqa.iter_mut().for_each(|v| *v = 0.0);
        rmsnorm::backward::<Q_LORA_RANK>(&sv.qa[t * Q_LORA_RANK..(t + 1) * Q_LORA_RANK], w.q_a_ln, RMS_NORM_EPS, &dqr, &mut dqa);
        fp8::rows_t::<H>(w.q_a.w, w.q_a.s, &dqa, &mut dxn[t * H..(t + 1) * H], 0, Q_LORA_RANK);
    }
}

/// The run's own positions, once every query attending them has run: each latent's gradient
/// through its norm and `kv_a` into `dxn` (`+=`).
pub fn latent_backward(w: &weights::Mla, sv: &Saved, dlatent: &[f32], dxn: &mut [f32]) {
    let mut dkva = vec![0f32; KV_LORA_RANK];
    for t in 0..sv.n {
        let p = sv.pos0 + t;
        dkva.iter_mut().for_each(|v| *v = 0.0);
        rmsnorm::backward::<KV_LORA_RANK>(&sv.kva[t * KV_LORA_RANK..(t + 1) * KV_LORA_RANK], w.kv_a_ln, RMS_NORM_EPS, &dlatent[p * KV_LORA_RANK..(p + 1) * KV_LORA_RANK], &mut dkva);
        fp8::rows_t::<H>(w.kv_a.w, w.kv_a.s, &dkva, &mut dxn[t * H..(t + 1) * H], 0, KV_LORA_RANK);
    }
}

/// Both halves for a run whose later runs have already been processed.
pub fn backward(w: &weights::Mla, st: &State, sv: &Saved, dy: &[f32], dlatent: &mut [f32], dxn: &mut [f32]) {
    attend_backward(w, st, sv, dy, dlatent, dxn);
    latent_backward(w, sv, dlatent, dxn);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference;
    use crate::testing::*;
    use common::testing::Rng;

    /// Two runs at the real shape — three positions, then four — against the dense reference over
    /// all seven: the second run's backward gives its `d xn` and every position's latent gradient.
    /// Under `DENSE_BELOW` positions the selection is every position, so no indexer is modelled.
    #[test]
    fn matches_dense_reference() {
        let mut rnd = Rng::new(0x41A0_0001);
        let (n_a, n_b) = (3usize, 4usize);
        let total = n_a + n_b;
        let q_a = fp8_record(Q_LORA_RANK, H, 0.002, &mut rnd);
        let q_b = fp8_record(MLA_HEADS * Q_HEAD, Q_LORA_RANK, 0.002, &mut rnd);
        let kv_a = fp8_record(KV_LORA_RANK, H, 0.002, &mut rnd);
        let o = fp8_record(H, MLA_HEADS * V_HEAD_DIM, 0.001, &mut rnd);
        let kv_b = bf(MLA_HEADS * KVB_HEAD * KV_LORA_RANK, 0.03, &mut rnd);
        let q_a_ln: Vec<f32> = (0..Q_LORA_RANK).map(|_| 1.0 + rnd.unit() * 0.3).collect();
        let kv_a_ln: Vec<f32> = (0..KV_LORA_RANK).map(|_| 1.0 + rnd.unit() * 0.3).collect();
        // the indexer: present for the forward's row and selection; no gradient
        let ix_wq_b = bf(INDEX_N_HEADS * INDEX_HEAD_DIM * Q_LORA_RANK, 0.03, &mut rnd);
        let ix_wk = bf(INDEX_HEAD_DIM * H, 0.03, &mut rnd);
        let ix_k_norm_w: Vec<f32> = vec![1.0; INDEX_HEAD_DIM];
        let ix_k_norm_b: Vec<f32> = vec![0.0; INDEX_HEAD_DIM];
        let ix_weights = bf(INDEX_N_HEADS * H, 0.03, &mut rnd);
        let ix_ape: Vec<f32> = unit(INDEX_KPOOL * INDEX_HEAD_DIM, 0.1, &mut rnd);
        let ix_gate = bf(INDEX_HEAD_DIM * H, 0.03, &mut rnd);
        let w = weights::Mla { q_a: q_a.view(), q_a_ln: &q_a_ln, q_b: q_b.view(), kv_a: kv_a.view(), kv_a_ln: &kv_a_ln, kv_b: &kv_b, o: o.view(), ix_wq_b: &ix_wq_b, ix_wk: &ix_wk, ix_k_norm_w: &ix_k_norm_w, ix_k_norm_b: &ix_k_norm_b, ix_weights: &ix_weights, ix_ape: &ix_ape, ix_gate: &ix_gate };

        let xn = unit(total * H, 1.0, &mut rnd);
        let dy = unit(n_b * H, 1.0, &mut rnd);
        let mut st = State::new(64);
        let mut s = Scratch::new(64);
        let mut y = vec![0f32; total * H];
        let mut saved_a = Saved::new(0, n_a);
        forward(3, &w, &xn[..n_a * H], 0, n_a, &mut st, &mut s, &mut saved_a, &mut y[..n_a * H]);
        let mut saved_b = Saved::new(n_a, n_b);
        forward(3, &w, &xn[n_a * H..], n_a, n_b, &mut st, &mut s, &mut saved_b, &mut y[n_a * H..]);
        let mut dlatent = vec![0f32; total * KV_LORA_RANK];
        let mut dxn = vec![0f32; n_b * H];
        backward(&w, &st, &saved_b, &dy, &mut dlatent, &mut dxn);

        let r = reference::MlaRef { nh: MLA_HEADS, dq: Q_HEAD, dv: V_HEAD_DIM, r: KV_LORA_RANK, qr: Q_LORA_RANK, h: H, rms_eps: RMS_NORM_EPS, q_a: &q_a.dense, q_a_ln: &q_a_ln, q_b: &q_b.dense, kv_a: &kv_a.dense, kv_a_ln: &kv_a_ln, kv_b: &wide(&kv_b), o: &o.dense, round: true };
        let (ry, rsv) = reference::mla_forward(&r, &xn, total);
        close(&y, &ry, 1e-4, "y");
        let mut rdy = vec![0f32; total * H];
        rdy[n_a * H..].copy_from_slice(&dy);
        let (rdx, rdlatent) = reference::mla_backward(&r, &rsv, &rdy);
        close(&dxn, &rdx[n_a * H..], 1e-4, "dxn (run B)");
        close(&dlatent, &rdlatent, 1e-4, "dlatent (every position)");
    }
}
