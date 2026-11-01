//! The model over a run of positions: the layers' forward from the embedding to the logits,
//! keeping what each op's backward needs, and the backward from the logits' gradient down to
//! the first MoE layer's input — below it nothing moves, so the dense layers run forward only.
//!
//! Layer by layer over the run, as prefill does: the attention half's input for every position
//! (hc_pre, the norm), the attention over the run (KDA's recurrence or MLA over the cache),
//! hc_post per position; then the FFN half per position. The backward is the same in reverse,
//! each half: `hc::post_backward` → the block's backward → the norm's → `hc::pre_backward`.
//!
//! Runs chain: the states advance (KDA's window and memory, MLA's cache), and a `Carry` holds
//! the gradient flowing from a run into the one before it (the entering KDA state's, every
//! MLA position's latent) — so a sequence is processed forward run by run, then backward run by
//! run in reverse, with one run's saved intermediates alive at a time.

use crate::expert::Adapter;
use crate::{ffn, hc, head, kda, mla};
use common::gemv::widen;
use common::pool::Serial;
use common::rmsnorm::rmsnorm;
use common::trace::NoTrace;
use glm_5_3_flash::expert::Stream;
use glm_5_3_flash::layout::{KDA_MEMORY, KDA_WINDOW};
use glm_5_3_flash::weights::{Attn, Ffn, Weights};
use glm_5_3_flash::{E, H, HC, HC_COEFS, INTERMEDIATE_SIZE, KV_LORA_RANK, RMS_NORM_EPS, TOP_K};
use train_common::rmsnorm as rms_t;

/// The adapters: one set of `E` per MoE layer (`None` for a dense layer). The same shape holds
/// their gradients.
pub struct Adapters {
    pub layers: Vec<Option<Vec<Adapter>>>,
}

impl Adapters {
    pub fn zeros(w: &Weights, r: usize) -> Self {
        Adapters { layers: w.layers.iter().map(|l| matches!(l.ffn, Ffn::Moe(_)).then(|| (0..E).map(|_| Adapter::zeros(r)).collect())).collect() }
    }
    pub fn rank(&self) -> usize {
        self.layers.iter().flatten().next().map_or(0, |a| a[0].r)
    }
}

/// The layers' states for one sequence.
pub struct States {
    pub kda: Vec<Option<(Vec<f32>, Vec<f32>)>>, // (window, mem)
    pub mla: Vec<Option<mla::State>>,
}

impl States {
    pub fn new(w: &Weights, max_ctx: usize) -> Self {
        States {
            kda: w.layers.iter().map(|l| matches!(l.attn, Attn::Kda(_)).then(|| (vec![0.0; KDA_WINDOW], vec![0.0; KDA_MEMORY]))).collect(),
            mla: w.layers.iter().map(|l| matches!(l.attn, Attn::Mla(_)).then(|| mla::State::new(max_ctx))).collect(),
        }
    }
}

/// The gradient flowing from a run into the run before it.
pub struct Carry {
    pub kda: Vec<Option<(Vec<f32>, Vec<f32>)>>, // (d window, d mem) at the run's entering state
    pub mla: Vec<Option<Vec<f32>>>,             // d latent, [max_ctx][KV_LORA_RANK]
}

impl Carry {
    pub fn zeros(w: &Weights, max_ctx: usize) -> Self {
        Carry {
            kda: w.layers.iter().map(|l| matches!(l.attn, Attn::Kda(_)).then(|| (vec![0.0; KDA_WINDOW], vec![0.0; KDA_MEMORY]))).collect(),
            mla: w.layers.iter().map(|l| matches!(l.attn, Attn::Mla(_)).then(|| vec![0.0; max_ctx * KV_LORA_RANK])).collect(),
        }
    }
}

pub struct Scratch {
    hc_flat: Vec<f32>,
    hc_coef: Vec<f32>,
    dense_gu: Vec<f32>,
    dense_h: Vec<f32>,
    dense_eo: Vec<f32>,
    dense_sh: Vec<f32>,
    dense_sc: Vec<f32>,
    ffn: ffn::Scratch,
    kda: kda::Scratch,
    mla: mla::Scratch,
}

impl Scratch {
    pub fn new(r: usize, max_ctx: usize) -> Self {
        Scratch { hc_flat: vec![0.0; HC * H], hc_coef: vec![0.0; HC_COEFS], dense_gu: vec![0.0; 2 * INTERMEDIATE_SIZE], dense_h: vec![0.0; INTERMEDIATE_SIZE], dense_eo: vec![0.0; TOP_K * H], dense_sh: vec![0.0; H], dense_sc: vec![0.0; E], ffn: ffn::Scratch::new(r), kda: kda::Scratch::new(), mla: mla::Scratch::new(max_ctx) }
    }
}

pub enum AttnSaved {
    Kda(kda::Saved),
    Mla(mla::Saved),
}

/// One layer's intermediates over the run.
pub struct LayerSaved {
    pub s_attn: Vec<f32>, // [n][HC·H]: the streams entering the attention half
    pub hc_attn: Vec<hc::Saved>,
    pub x_attn: Vec<f32>, // [n][H]: the collapsed input before its norm
    pub y_attn: Vec<f32>, // [n][H]: the attention's output
    pub attn: AttnSaved,
    pub s_ffn: Vec<f32>,
    pub hc_ffn: Vec<hc::Saved>,
    pub x_ffn: Vec<f32>,
    pub y_ffn: Vec<f32>,
    pub ffn: Option<Vec<ffn::Saved>>, // the MoE's, per position; a dense layer keeps nothing
}

pub struct RunSaved {
    pub pos0: usize,
    pub n: usize,
    pub layers: Vec<LayerSaved>,
    pub head: Vec<head::Saved>,
}

/// `tokens` at `pos0..`: the logits of every position (`[n][vocab]`, the first `vocab` rows of
/// the head), the states advanced, everything the backward needs in `RunSaved`.
#[allow(clippy::too_many_arguments)]
pub fn forward(w: &Weights, stream: &Stream, ad: &Adapters, tokens: &[u32], pos0: usize, vocab: usize, st: &mut States, s: &mut Scratch, logits: &mut [f32]) -> RunSaved {
    let n = tokens.len();
    let r = ad.rank();
    // the embedding into every stream
    let mut streams = vec![0f32; n * HC * H];
    for (t, &tok) in tokens.iter().enumerate() {
        let row = &w.embed[tok as usize * H..(tok as usize + 1) * H];
        for j in 0..HC {
            for d in 0..H {
                streams[(t * HC + j) * H + d] = widen(row[d]);
            }
        }
    }
    let mut layers = Vec::with_capacity(w.layers.len());
    let mut xn = vec![0f32; n * H];
    let mut y = vec![0f32; n * H];
    for (l, lw) in w.layers.iter().enumerate() {
        // the attention half: every position's input, the attention over the run, every position's mix
        let s_attn = streams.clone();
        let mut hc_attn = Vec::with_capacity(n);
        let mut x_attn = vec![0f32; n * H];
        for t in 0..n {
            let mut hcs = glm_5_3_flash::hc::Scratch { flat: &mut s.hc_flat, coef: &mut s.hc_coef };
            let mix = glm_5_3_flash::hc::pre(&lw.hc_attn, &streams[t * HC * H..(t + 1) * HC * H], &mut hcs, &Serial, &mut x_attn[t * H..(t + 1) * H]);
            hc_attn.push(hc::Saved::from_forward(&lw.hc_attn, &s.hc_coef, mix));
            rmsnorm::<H>(&x_attn[t * H..(t + 1) * H], lw.input_ln, RMS_NORM_EPS, &mut xn[t * H..(t + 1) * H]);
        }
        let attn = match &lw.attn {
            Attn::Kda(a) => {
                let (window, mem) = st.kda[l].as_mut().expect("a KDA layer's state");
                let mut saved = kda::Saved::new(n);
                kda::forward(l, a, &xn, n, window, mem, &mut s.kda, &mut saved, &mut y);
                AttnSaved::Kda(saved)
            }
            Attn::Mla(a) => {
                let state = st.mla[l].as_mut().expect("an MLA layer's state");
                let mut saved = mla::Saved::new(pos0, n);
                mla::forward(l, a, &xn, pos0, n, state, &mut s.mla, &mut saved, &mut y);
                AttnSaved::Mla(saved)
            }
        };
        let y_attn = y.clone();
        for t in 0..n {
            glm_5_3_flash::hc::post(&hc_attn[t].mix, &y[t * H..(t + 1) * H], &mut streams[t * HC * H..(t + 1) * HC * H]);
        }
        // the FFN half, position by position
        let s_ffn = streams.clone();
        let mut hc_ffn = Vec::with_capacity(n);
        let mut x_ffn = vec![0f32; n * H];
        let mut ffn_saved: Option<Vec<ffn::Saved>> = matches!(lw.ffn, Ffn::Moe(_)).then(Vec::new);
        for t in 0..n {
            let mut hcs = glm_5_3_flash::hc::Scratch { flat: &mut s.hc_flat, coef: &mut s.hc_coef };
            let mix = glm_5_3_flash::hc::pre(&lw.hc_ffn, &streams[t * HC * H..(t + 1) * HC * H], &mut hcs, &Serial, &mut x_ffn[t * H..(t + 1) * H]);
            hc_ffn.push(hc::Saved::from_forward(&lw.hc_ffn, &s.hc_coef, mix));
            rmsnorm::<H>(&x_ffn[t * H..(t + 1) * H], lw.post_ln, RMS_NORM_EPS, &mut xn[t * H..(t + 1) * H]);
            let yt = &mut y[t * H..(t + 1) * H];
            match &lw.ffn {
                Ffn::Dense(d) => {
                    let mut fs = glm_5_3_flash::ffn::Scratch { gu: &mut s.dense_gu, h: &mut s.dense_h, expert_out: &mut s.dense_eo, shared: &mut s.dense_sh, score: &mut s.dense_sc };
                    glm_5_3_flash::ffn::dense(l, d, &xn[t * H..(t + 1) * H], &mut fs, &Serial, &mut NoTrace, yt);
                }
                Ffn::Moe(m) => {
                    let mut saved = ffn::Saved::new(r);
                    ffn::forward(l, m, stream, ad.layers[l].as_ref().expect("a MoE layer's adapters"), &xn[t * H..(t + 1) * H], &mut s.ffn, &mut saved, yt);
                    ffn_saved.as_mut().unwrap().push(saved);
                }
            }
            glm_5_3_flash::hc::post(&hc_ffn[t].mix, yt, &mut streams[t * HC * H..(t + 1) * HC * H]);
        }
        layers.push(LayerSaved { s_attn, hc_attn, x_attn, y_attn, attn, s_ffn, hc_ffn, x_ffn, y_ffn: y.clone(), ffn: ffn_saved });
    }
    let mut hs = Vec::with_capacity(n);
    for t in 0..n {
        let mut saved = head::Saved::new();
        head::forward(w.norm, w.lm_head, vocab, &streams[t * HC * H..(t + 1) * HC * H], &mut saved, &mut logits[t * vocab..(t + 1) * vocab]);
        hs.push(saved);
    }
    RunSaved { pos0, n, layers, head: hs }
}

/// The first layer whose FFN is a MoE: the backward stops at its input.
fn first_moe(w: &Weights) -> usize {
    w.layers.iter().position(|l| matches!(l.ffn, Ffn::Moe(_))).unwrap_or(w.layers.len())
}

/// Given `dlogits: [n][vocab]`, accumulate the adapters' gradients and the carry; the run after
/// this one (if any) must have been processed first, its carry in `carry`.
#[allow(clippy::too_many_arguments)]
pub fn backward(w: &Weights, stream: &Stream, ad: &Adapters, sv: &RunSaved, st: &States, vocab: usize, dlogits: &[f32], s: &mut Scratch, grads: &mut Adapters, carry: &mut Carry) {
    let n = sv.n;
    let mut dstreams = vec![0f32; n * HC * H];
    for t in 0..n {
        head::backward(w.norm, w.lm_head, vocab, &sv.head[t], &dlogits[t * vocab..(t + 1) * vocab], &mut dstreams[t * HC * H..(t + 1) * HC * H]);
    }
    let stop = first_moe(w);
    let mut dy = vec![0f32; n * H];
    let mut dxn = vec![0f32; n * H];
    let mut dx = vec![0f32; H];
    let mut ds = vec![0f32; n * HC * H];
    for l in (stop..w.layers.len()).rev() {
        let lw = &w.layers[l];
        let ls = &sv.layers[l];
        // the FFN half
        ds.iter_mut().for_each(|v| *v = 0.0);
        for t in 0..n {
            let (sl, dsl) = (t * HC * H..(t + 1) * HC * H, t * H..(t + 1) * H);
            let mut dyt = vec![0f32; H];
            let dm = hc::post_backward(&ls.hc_ffn[t].mix, &ls.y_ffn[dsl.clone()], &ls.s_ffn[sl.clone()], &dstreams[sl.clone()], &mut dyt, &mut ds[sl.clone()]);
            dx.iter_mut().for_each(|v| *v = 0.0);
            match &lw.ffn {
                Ffn::Moe(m) => {
                    let mut dxnt = vec![0f32; H];
                    ffn::backward(l, m, stream, ad.layers[l].as_ref().unwrap(), &ls.ffn.as_ref().unwrap()[t], &dyt, &mut s.ffn, grads.layers[l].as_mut().unwrap(), &mut dxnt);
                    rms_t::backward::<H>(&ls.x_ffn[dsl.clone()], lw.post_ln, RMS_NORM_EPS, &dxnt, &mut dx);
                }
                Ffn::Dense(_) => unreachable!("a dense layer is below the first MoE"),
            }
            hc::pre_backward(&lw.hc_ffn, &ls.s_ffn[sl.clone()], &ls.hc_ffn[t], &dx, &dm, &mut ds[sl.clone()]);
        }
        dstreams.copy_from_slice(&ds);
        // the attention half: every position's post, the attention over the run, every position's pre
        ds.iter_mut().for_each(|v| *v = 0.0);
        dy.iter_mut().for_each(|v| *v = 0.0);
        let mut dms = Vec::with_capacity(n);
        for t in 0..n {
            let (sl, dsl) = (t * HC * H..(t + 1) * HC * H, t * H..(t + 1) * H);
            dms.push(hc::post_backward(&ls.hc_attn[t].mix, &ls.y_attn[dsl.clone()], &ls.s_attn[sl.clone()], &dstreams[sl.clone()], &mut dy[dsl], &mut ds[sl]));
        }
        dxn.iter_mut().for_each(|v| *v = 0.0);
        match (&lw.attn, &ls.attn) {
            (Attn::Kda(a), AttnSaved::Kda(ks)) => {
                let (dwin, dmem) = carry.kda[l].as_mut().unwrap();
                // the carry enters as the gradient at the run's exit state and leaves as the entering state's
                let (dwin_out, dmem_out) = (dwin.clone(), dmem.clone());
                dwin.iter_mut().for_each(|v| *v = 0.0);
                dmem.iter_mut().for_each(|v| *v = 0.0);
                kda::backward(a, ks, &dy, &dwin_out, &dmem_out, &mut dxn, dwin, dmem);
            }
            (Attn::Mla(a), AttnSaved::Mla(ms)) => {
                let dlatent = carry.mla[l].as_mut().unwrap();
                mla::backward(a, st.mla[l].as_ref().unwrap(), ms, &dy, dlatent, &mut dxn);
            }
            _ => unreachable!("a layer's attention and its saved state disagree"),
        }
        for t in 0..n {
            let (sl, dsl) = (t * HC * H..(t + 1) * HC * H, t * H..(t + 1) * H);
            dx.iter_mut().for_each(|v| *v = 0.0);
            rms_t::backward::<H>(&ls.x_attn[dsl.clone()], lw.input_ln, RMS_NORM_EPS, &dxn[dsl], &mut dx);
            hc::pre_backward(&lw.hc_attn, &ls.s_attn[sl.clone()], &ls.hc_attn[t], &dx, &dms[t], &mut ds[sl]);
        }
        dstreams.copy_from_slice(&ds);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert::tests::random_adapter;
    use crate::loss::{forward_kl_topk, Teacher};
    use crate::testing::*;
    use common::q4::{self, GROUP};
    use common::testing::Rng;
    use glm_5_3_flash::expert::{pack_items, ITEMS};
    use glm_5_3_flash::layout::{self, DOWN_ITEMS, GU_ITEMS};
    use glm_5_3_flash::weights::{DenseFfn, Hc, Kda, Layer, Mla, Moe};
    use glm_5_3_flash::*;

    /// Random weights in the bundle's formats for one of each layer kind, held so the `Weights`
    /// can borrow them; the dense layers share one KDA and one FFN so the test stays in memory.
    struct Bufs {
        embed: Vec<u16>,
        norm: Vec<f32>,
        lm_head: Vec<u16>,
        ln: Vec<f32>,
        hc_fn: Vec<u16>,
        hc_base: Vec<f32>,
        hc_scale: [f32; 3],
        kda: (Vec<u16>, Vec<u16>, Vec<u16>, [Vec<u16>; 3], Vec<u16>, Vec<u16>, Vec<f32>, Vec<f32>, Vec<u16>, Vec<u16>, Vec<u16>, Vec<f32>, Vec<u16>),
        mla: (Fp8Buf, Vec<f32>, Fp8Buf, Fp8Buf, Vec<f32>, Vec<u16>, Fp8Buf, Vec<u16>, Vec<u16>, Vec<f32>, Vec<f32>, Vec<u16>, Vec<f32>, Vec<u16>),
        dense: (Fp8Buf, Fp8Buf, Fp8Buf),
        router: Vec<u16>,
        bias: Vec<f32>,
        shared: (Fp8Buf, Fp8Buf, Fp8Buf),
    }

    impl Bufs {
        fn new(vocab: usize, rnd: &mut Rng) -> Self {
            let fp = |m, k, sc, rnd: &mut Rng| fp8_record(m, k, sc, rnd);
            Bufs {
                embed: bf(vocab * H, 1.0, rnd),
                norm: (0..H).map(|_| 1.0 + rnd.unit() * 0.2).collect(),
                lm_head: bf(vocab * H, 0.05, rnd),
                ln: (0..H).map(|_| 1.0 + rnd.unit() * 0.2).collect(),
                hc_fn: bf(HC_COEFS * HC * H, 0.02, rnd),
                hc_base: unit(HC_COEFS, 1.0, rnd),
                hc_scale: [0.7, 1.3, 0.9],
                kda: (bf(KDA_QKV * H, 0.03, rnd), bf(KDA_QKV * H, 0.03, rnd), bf(KDA_QKV * H, 0.03, rnd), [bf(KDA_QKV * CONV_TAPS, 0.5, rnd), bf(KDA_QKV * CONV_TAPS, 0.5, rnd), bf(KDA_QKV * CONV_TAPS, 0.5, rnd)], bf(KDA_DIM * H, 0.03, rnd), bf(KDA_QKV * KDA_DIM, 0.1, rnd), unit(KDA_QKV, 1.0, rnd), unit(KDA_HEADS, 0.5, rnd), bf(KDA_HEADS * H, 0.03, rnd), bf(KDA_DIM * H, 0.03, rnd), bf(KDA_QKV * KDA_DIM, 0.1, rnd), (0..KDA_DIM).map(|_| 1.0 + rnd.unit() * 0.3).collect(), bf(H * KDA_QKV, 0.03, rnd)),
                mla: (fp(Q_LORA_RANK, H, 0.002, rnd), (0..Q_LORA_RANK).map(|_| 1.0 + rnd.unit() * 0.3).collect(), fp(MLA_HEADS * Q_HEAD, Q_LORA_RANK, 0.002, rnd), fp(KV_LORA_RANK, H, 0.002, rnd), (0..KV_LORA_RANK).map(|_| 1.0 + rnd.unit() * 0.3).collect(), bf(MLA_HEADS * KVB_HEAD * KV_LORA_RANK, 0.03, rnd), fp(H, MLA_HEADS * V_HEAD_DIM, 0.001, rnd), bf(INDEX_N_HEADS * INDEX_HEAD_DIM * Q_LORA_RANK, 0.03, rnd), bf(INDEX_HEAD_DIM * H, 0.03, rnd), vec![1.0; INDEX_HEAD_DIM], vec![0.0; INDEX_HEAD_DIM], bf(INDEX_N_HEADS * H, 0.03, rnd), unit(INDEX_KPOOL * INDEX_HEAD_DIM, 0.1, rnd), bf(INDEX_HEAD_DIM * H, 0.03, rnd)),
                dense: (fp(INTERMEDIATE_SIZE, H, 0.002, rnd), fp(INTERMEDIATE_SIZE, H, 0.002, rnd), fp(H, INTERMEDIATE_SIZE, 0.001, rnd)),
                router: bf(E * H, 0.05, rnd),
                // the picks are always experts 0..TOP_K: those are the ones packed
                bias: (0..E).map(|e| if e < TOP_K { 0.0 } else { -100.0 }).collect(),
                shared: (fp(D_FF, H, 0.002, rnd), fp(D_FF, H, 0.002, rnd), fp(H, D_FF, 0.001, rnd)),
            }
        }
        fn hc(&self) -> Hc<'_> {
            Hc { fn_: &self.hc_fn, base: &self.hc_base, scale: &self.hc_scale }
        }
        fn kda(&self) -> Attn<'_> {
            let k = &self.kda;
            Attn::Kda(Kda { q: &k.0, k: &k.1, v: &k.2, conv: [&k.3[0], &k.3[1], &k.3[2]], f_a: &k.4, f_b: &k.5, dt_bias: &k.6, a_log: &k.7, b: &k.8, g_a: &k.9, g_b: &k.10, o_norm: &k.11, o: &k.12 })
        }
        fn mla(&self) -> Attn<'_> {
            let m = &self.mla;
            Attn::Mla(Mla { q_a: m.0.view(), q_a_ln: &m.1, q_b: m.2.view(), kv_a: m.3.view(), kv_a_ln: &m.4, kv_b: &m.5, o: m.6.view(), ix_wq_b: &m.7, ix_wk: &m.8, ix_k_norm_w: &m.9, ix_k_norm_b: &m.10, ix_weights: &m.11, ix_ape: &m.12, ix_gate: &m.13 })
        }
        fn dense(&self) -> Ffn<'_> {
            Ffn::Dense(DenseFfn { gate: self.dense.0.view(), up: self.dense.1.view(), down: self.dense.2.view() })
        }
        fn moe(&self) -> Ffn<'_> {
            Ffn::Moe(Moe { router: &self.router, bias: &self.bias, shared_gate: self.shared.0.view(), shared_up: self.shared.1.view(), shared_down: self.shared.2.view() })
        }
        /// Layers 0..FIRST_K_DENSE_REPLACE dense over KDA (one weight set), then an MLA MoE layer and a KDA MoE layer.
        fn weights(&self) -> Weights<'_> {
            let mut layers = Vec::new();
            for _ in 0..FIRST_K_DENSE_REPLACE {
                layers.push(Layer { input_ln: &self.ln, post_ln: &self.ln, hc_attn: self.hc(), hc_ffn: self.hc(), attn: self.kda(), ffn: self.dense() });
            }
            layers.push(Layer { input_ln: &self.ln, post_ln: &self.ln, hc_attn: self.hc(), hc_ffn: self.hc(), attn: self.mla(), ffn: self.moe() });
            layers.push(Layer { input_ln: &self.ln, post_ln: &self.ln, hc_attn: self.hc(), hc_ffn: self.hc(), attn: self.kda(), ffn: self.moe() });
            Weights { embed: &self.embed, norm: &self.norm, lm_head: &self.lm_head, layers, mtp: None }
        }
    }

    /// A stream holding experts `0..TOP_K` of the two MoE layers in one block: layer 3's in its
    /// own slots, layer 4's in slots `TOP_K..2·TOP_K` of the same block, the table pointing there.
    struct Experts {
        block: Vec<u32>,
        addr: Vec<u64>,
    }
    impl Experts {
        fn new(rnd: &mut Rng) -> Self {
            let mut block = vec![0u32; layout::XBLOCK_BYTES / 4];
            let bytes: &mut [u8] = unsafe { core::slice::from_raw_parts_mut(block.as_mut_ptr() as *mut u8, layout::XBLOCK_BYTES) };
            let mut addr = vec![0u64; layout::MOE_LAYERS * E * ITEMS];
            for (li, l) in [FIRST_K_DENSE_REPLACE, FIRST_K_DENSE_REPLACE + 1].into_iter().enumerate() {
                for e in 0..TOP_K {
                    let slot = li * TOP_K + e;
                    let mk = |m: usize, k: usize, rnd: &mut Rng| {
                        let w: Vec<i8> = (0..m * k).map(|_| rnd.i4()).collect();
                        let s: Vec<f32> = (0..m * (k / GROUP)).map(|_| 0.002 + (rnd.next() % 100) as f32 * 2e-5).collect();
                        q4::pack(m, k, &w, &s)
                    };
                    let (guw, gus) = mk(2 * D_FF, H, rnd);
                    let (dw, ds) = mk(H, D_FF, rnd);
                    // the items at the block positions of expert `slot` of the first MoE layer
                    for (c, (off, it)) in pack_items(FIRST_K_DENSE_REPLACE, slot, &guw, &gus, &dw, &ds).into_iter().enumerate() {
                        bytes[off..off + it.len()].copy_from_slice(&it);
                        addr[glm_5_3_flash::expert::item_index(l, e, c)] = off as u64;
                    }
                }
            }
            let _ = (GU_ITEMS, DOWN_ITEMS);
            Experts { block, addr }
        }
        fn stream(&self) -> Stream<'_> {
            let bytes: &[u8] = unsafe { core::slice::from_raw_parts(self.block.as_ptr() as *const u8, layout::XBLOCK_BYTES) };
            Stream::placed(bytes, self.addr.clone(), vec![0; layout::MOE_LAYERS * E * ITEMS], Vec::new())
        }
    }

    /// The whole run's gradient against a central difference along its own direction, one MoE
    /// layer's adapters at a time, through every op from the loss to the first MoE layer: the
    /// wiring, end to end. The lower layer's gradient passes the upper layer's expert-input
    /// quantizer, so its tolerance allows a flip.
    #[test]
    fn run_gradient_matches_finite_differences() {
        let mut rnd = Rng::new(0x30DE_1001);
        let vocab = 4096;
        let bufs = Bufs::new(vocab, &mut rnd);
        let w = bufs.weights();
        let experts = Experts::new(&mut rnd);
        let stream = experts.stream();
        let r = 4;
        let mut ad = Adapters::zeros(&w, r);
        for l in [FIRST_K_DENSE_REPLACE, FIRST_K_DENSE_REPLACE + 1] {
            for e in 0..TOP_K {
                ad.layers[l].as_mut().unwrap()[e] = random_adapter(r, 0.05, &mut rnd);
            }
        }
        let n = 3;
        let tokens: Vec<u32> = (0..n).map(|_| (rnd.next() % vocab as u64) as u32).collect();
        let teachers: Vec<(Vec<u32>, Vec<f32>)> = (0..n).map(|_| ((0..8).map(|_| (rnd.next() % vocab as u64) as u32).collect(), unit(8, 2.0, &mut rnd))).collect();
        let max_ctx = 16;

        let run = |ad: &Adapters, grads: Option<(&mut Adapters, &mut Carry)>| -> f64 {
            let mut st = States::new(&w, max_ctx);
            let mut s = Scratch::new(r, max_ctx);
            let mut logits = vec![0f32; n * vocab];
            let sv = forward(&w, &stream, ad, &tokens, 0, vocab, &mut st, &mut s, &mut logits);
            let mut dlogits = vec![0f32; n * vocab];
            let mut loss = 0f64;
            for t in 0..n {
                loss += forward_kl_topk(&logits[t * vocab..(t + 1) * vocab], &Teacher { ids: &teachers[t].0, logp: &teachers[t].1 }, &mut dlogits[t * vocab..(t + 1) * vocab]) as f64;
            }
            if let Some((g, c)) = grads {
                backward(&w, &stream, ad, &sv, &st, vocab, &dlogits, &mut s, g, c);
            }
            loss
        };
        let mut grads = Adapters::zeros(&w, r);
        let mut carry = Carry::zeros(&w, max_ctx);
        let l0 = run(&ad, Some((&mut grads, &mut carry)));
        assert!(l0.is_finite() && l0 > 0.0);

        // a directional derivative along the gradient restricted to one layer's adapters, normalized:
        // d loss / d ε at θ + ε·v is ‖g_layer‖ if g is the gradient. Summed over the layer's ~1M
        // parameters the signal is ~1e-3, well above the f32 forward's ~1e-7 noise on the loss;
        // a single parameter's derivative (~1e-6) is not.
        // Measured (2026-11-01) along the gradient restricted to the `down` adapters' B — the one
        // direction no quantizer sits on — the central difference is within the f32 forward's
        // noise of ‖g‖: layer 4 0.9 % at ε 0.05 (2.2 % at ε 0.01, the noise growing as 1/ε),
        // layer 3 0.2 %. Along every matrix the step crosses the experts' activation quantizer
        // and the difference shrinks with a wider interval — 9.4 % at ε 0.01, 3.5 % at 0.05,
        // 0.6 % at 0.2 — the mark of flips averaged out, not of a gradient error.
        for (l, all, eps) in [(FIRST_K_DENSE_REPLACE + 1, false, 0.05f32), (FIRST_K_DENSE_REPLACE, false, 0.05), (FIRST_K_DENSE_REPLACE + 1, true, 0.2)] {
            let g = grads.layers[l].as_ref().unwrap();
            fn pick<'a>(a: &'a Adapter, all: bool) -> Vec<&'a Vec<f32>> {
                if all { vec![&a.ga, &a.gb, &a.ua, &a.ub, &a.da, &a.db] } else { vec![&a.db] }
            }
            let norm: f32 = g.iter().map(|a| pick(a, all).into_iter().flatten().map(|v| v * v).sum::<f32>()).sum::<f32>().sqrt();
            assert!(norm > 1e-5, "layer {l}: the gradient is {norm}");
            let step = |ad: &mut Adapters, sign: f32| {
                let a = ad.layers[l].as_mut().unwrap();
                for e in 0..E {
                    let (x, gg) = (&mut a[e], &g[e]);
                    let pairs: Vec<(&mut Vec<f32>, &Vec<f32>)> = if all { vec![(&mut x.ga, &gg.ga), (&mut x.gb, &gg.gb), (&mut x.ua, &gg.ua), (&mut x.ub, &gg.ub), (&mut x.da, &gg.da), (&mut x.db, &gg.db)] } else { vec![(&mut x.db, &gg.db)] };
                    for (dst, src) in pairs {
                        for (d, s) in dst.iter_mut().zip(src) {
                            *d += sign * eps * s / norm;
                        }
                    }
                }
            };
            step(&mut ad, 1.0);
            let p = run(&ad, None);
            step(&mut ad, -2.0);
            let m = run(&ad, None);
            step(&mut ad, 1.0);
            let num = ((p - m) / (2.0 * eps as f64)) as f32;
            assert!((num - norm).abs() <= 2e-2 * norm, "layer {l} (all matrices: {all}, ε {eps}): directional derivative {num} against ‖g‖ {norm}");
        }
    }
}
