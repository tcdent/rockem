//! Construction: weights as slices into the mmap'd checkpoint and bundle,
//! the YaRN tables, the state allocated once, and the decode step that ties
//! the fused kernels together. `std` only; never on the token path except
//! `State::decode`, which allocates nothing.

use crate::*;
use common::gemv::widen;
use common::trace::{Trace, NONE};

pub struct Layer<'a> {
    pub input_ln: Vec<f32>,
    pub post_ln: Vec<f32>,
    pub q_proj: &'a [u16],
    pub kv_a: &'a [u16],
    pub kv_a_ln: Vec<f32>,
    pub kv_b: Vec<f32>,
    pub o_proj: &'a [u16],
    pub ffn: Ffn<'a>,
}
pub enum Ffn<'a> {
    Dense { gate: &'a [u16], up: &'a [u16], down: &'a [u16] },
    Moe { router: &'a [u16], shared_gate: &'a [u16], shared_up: &'a [u16], shared_down: &'a [u16] },
}
pub struct Weights<'a> {
    pub embed: &'a [u16], // [VOCAB_SIZE][H]
    pub layers: Vec<Layer<'a>>,
    pub norm: Vec<f32>,
    pub lm_head: &'a [u16], // [VOCAB_SIZE][H]
}

impl<'a> Weights<'a> {
    /// `get(name)` returns a checkpoint tensor by HF key as row-major bf16.
    /// The routed experts are not here: they belong to a `Routed` realization.
    pub fn new(mut get: impl FnMut(&str) -> &'a [u16]) -> Self {
        let widen_all = |w: &[u16]| w.iter().map(|&v| widen(v)).collect::<Vec<f32>>();
        let mut layers = Vec::with_capacity(NUM_HIDDEN_LAYERS);
        for l in 0..NUM_HIDDEN_LAYERS {
            let p = format!("model.layers.{l}.");
            let mut t = |s: &str| get(&format!("{p}{s}"));
            let ffn = if l < FIRST_K_DENSE_REPLACE {
                Ffn::Dense { gate: t("mlp.gate_proj.weight"), up: t("mlp.up_proj.weight"), down: t("mlp.down_proj.weight") }
            } else {
                Ffn::Moe {
                    router: t("mlp.gate.weight"),
                    shared_gate: t("mlp.shared_experts.gate_proj.weight"),
                    shared_up: t("mlp.shared_experts.up_proj.weight"),
                    shared_down: t("mlp.shared_experts.down_proj.weight"),
                }
            };
            layers.push(Layer {
                input_ln: widen_all(t("input_layernorm.weight")),
                post_ln: widen_all(t("post_attention_layernorm.weight")),
                q_proj: t("self_attn.q_proj.weight"),
                kv_a: t("self_attn.kv_a_proj_with_mqa.weight"),
                kv_a_ln: widen_all(t("self_attn.kv_a_layernorm.weight")),
                kv_b: widen_all(t("self_attn.kv_b_proj.weight")),
                o_proj: t("self_attn.o_proj.weight"),
                ffn,
            });
        }
        let w = Weights { embed: get("model.embed_tokens.weight"), layers, norm: widen_all(get("model.norm.weight")), lm_head: get("lm_head.weight") };
        w.check();
        w
    }

    /// The same weights from the bundle's dense file (`layout::dense`) —
    /// what the node loads; no checkpoint in sight.
    pub fn from_dense(dense: &'a [u16]) -> Self {
        use layout::dense as d;
        assert_eq!(dense.len(), d::TOTAL, "dense file size");
        let widen_all = |w: &[u16]| w.iter().map(|&v| widen(v)).collect::<Vec<f32>>();
        let at = |off: usize, len: usize| &dense[off..off + len];
        let mut layers = Vec::with_capacity(NUM_HIDDEN_LAYERS);
        for l in 0..NUM_HIDDEN_LAYERS {
            let b = d::layer(l);
            let ffn = if l < FIRST_K_DENSE_REPLACE {
                Ffn::Dense {
                    gate: at(b + d::DENSE_GATE, INTERMEDIATE_SIZE * H),
                    up: at(b + d::DENSE_UP, INTERMEDIATE_SIZE * H),
                    down: at(b + d::DENSE_DOWN, H * INTERMEDIATE_SIZE),
                }
            } else {
                Ffn::Moe {
                    router: at(b + d::ROUTER, N_ROUTED_EXPERTS * H),
                    shared_gate: at(b + d::SHARED_GATE, SHARED_FF * H),
                    shared_up: at(b + d::SHARED_UP, SHARED_FF * H),
                    shared_down: at(b + d::SHARED_DOWN, H * SHARED_FF),
                }
            };
            layers.push(Layer {
                input_ln: widen_all(at(b + d::INPUT_LN, H)),
                post_ln: widen_all(at(b + d::POST_LN, H)),
                q_proj: at(b + d::Q_PROJ, Q_DIM * H),
                kv_a: at(b + d::KV_A, KVA_DIM * H),
                kv_a_ln: widen_all(at(b + d::KV_A_LN, KV_LORA_RANK)),
                kv_b: widen_all(at(b + d::KV_B, KVB_DIM * KV_LORA_RANK)),
                o_proj: at(b + d::O_PROJ, H * O_IN_DIM),
                ffn,
            });
        }
        let w = Weights {
            embed: at(d::EMBED, VOCAB_SIZE * H),
            layers,
            norm: widen_all(at(d::NORM, H)),
            lm_head: at(d::LM_HEAD, VOCAB_SIZE * H),
        };
        w.check();
        w
    }

    fn check(&self) {
        assert_eq!(self.embed.len(), VOCAB_SIZE * H);
        assert_eq!(self.lm_head.len(), VOCAB_SIZE * H);
        assert_eq!(self.norm.len(), H);
        for l in &self.layers {
            assert_eq!(l.q_proj.len(), Q_DIM * H);
            assert_eq!(l.kv_a.len(), KVA_DIM * H);
            assert_eq!(l.kv_a_ln.len(), KV_LORA_RANK);
            assert_eq!(l.kv_b.len(), KVB_DIM * KV_LORA_RANK);
            assert_eq!(l.o_proj.len(), H * O_IN_DIM);
            match &l.ffn {
                Ffn::Dense { gate, down, .. } => {
                    assert_eq!(gate.len(), INTERMEDIATE_SIZE * H);
                    assert_eq!(down.len(), H * INTERMEDIATE_SIZE);
                }
                Ffn::Moe { router, shared_down, .. } => {
                    assert_eq!(router.len(), N_ROUTED_EXPERTS * H);
                    assert_eq!(shared_down.len(), H * SHARED_FF);
                }
            }
        }
    }
}

/// The routed experts on this CPU, sequentially, from the bundle's expert
/// stream in this process's memory: the reference placement.
pub struct CpuExperts<'a> {
    pub stream: &'a [u8],
    scratch: expert::ScratchBuf,
}

impl<'a> CpuExperts<'a> {
    pub fn new(stream: &'a [u8]) -> Self {
        assert_eq!(stream.len(), layout::STREAM, "expert stream size");
        CpuExperts { stream, scratch: expert::ScratchBuf::new() }
    }
}

impl ffn::Routed for CpuExperts<'_> {
    fn run(&mut self, l: usize, idx: &[usize; TOP_K], x: &[f32], out: &mut [f32]) {
        for k in 0..TOP_K {
            let ew = expert::Weights::from_record(&self.stream[layout::record(l, idx[k])]);
            expert::forward(&ew, x, &mut self.scratch.scratch(), &mut out[k * H..(k + 1) * H]);
        }
    }
}

/// A boxed realization is one too — how a node picks placement at construction.
impl<R: ffn::Routed + ?Sized> ffn::Routed for Box<R> {
    fn run(&mut self, l: usize, idx: &[usize; TOP_K], x: &[f32], out: &mut [f32]) {
        (**self).run(l, idx, x, out)
    }
}
/// A tier's whole model: attention, the FFN half and the head, chosen at
/// construction. `decode` drives one through a token. A tier that keeps the
/// residual to itself between layers (the card) hands the trace what it
/// needs after the token through `token_trace`; the CPU tier traces as it goes.
pub trait Model: attn::Attention + ffn::FeedForward + ffn::Head {
    /// Token `i` of the last step: every layer's residual output
    /// (`[NUM_HIDDEN_LAYERS][H]`), routing (`[NUM_HIDDEN_LAYERS][TOP_K]`, 0
    /// for dense layers) and its logits, for a tier that could not trace them
    /// during the step.
    fn token_trace(&self, _i: usize, _layers: &mut [f32], _topk: &mut [u32], _logits: &mut [f32]) {}
    /// The token drawn from token `i`'s `logits` (its row) under the recipe
    /// with the uniform `u` (`common::sample::draw`); temperature 0 is the
    /// argmax. A tier that holds the logits itself draws there, from row `i`.
    fn draw(&mut self, _i: usize, recipe: &Sampling, u: f64, logits: &[f32]) -> usize {
        common::sample::draw(logits, recipe, u)
    }
    /// ln p(id) under token `i`'s `logits` (`common::softmax::log_prob`): the
    /// model's own distribution, no recipe. A tier that holds the logits
    /// scores there.
    fn score(&mut self, _i: usize, id: usize, logits: &[f32]) -> f32 {
        common::softmax::log_prob(logits, id)
    }
}
impl<M: Model + ?Sized> Model for Box<M> {
    fn token_trace(&self, i: usize, layers: &mut [f32], topk: &mut [u32], logits: &mut [f32]) {
        (**self).token_trace(i, layers, topk, logits)
    }
    fn draw(&mut self, i: usize, recipe: &Sampling, u: f64, logits: &[f32]) -> usize {
        (**self).draw(i, recipe, u, logits)
    }
    fn score(&mut self, i: usize, id: usize, logits: &[f32]) -> f32 {
        (**self).score(i, id, logits)
    }
}

/// The model on this CPU: the reference placement, every op in this process.
pub struct CpuModel<R: ffn::Routed> {
    pub attention: CpuAttention,
    pub ffn: CpuFfn<R>,
}
impl<R: ffn::Routed> attn::Attention for CpuModel<R> {
    fn step(&mut self, l: usize, w: &attn::Weights, h: &mut [f32], pos: usize, n: usize, cos: &[f32], sin: &[f32], scale: f32, tr: &mut dyn Trace) {
        self.attention.step(l, w, h, pos, n, cos, sin, scale, tr)
    }
    fn seed(&mut self, rows: &[f32]) {
        self.attention.seed(rows)
    }
    fn rows(&self, i: usize, out: &mut [f32]) {
        self.attention.rows(i, out)
    }
}
impl<R: ffn::Routed> ffn::FeedForward for CpuModel<R> {
    fn dense(&mut self, l: usize, w: &ffn::Dense, h: &mut [f32], n: usize, tr: &mut dyn Trace) {
        self.ffn.dense(l, w, h, n, tr)
    }
    fn moe(&mut self, l: usize, w: &ffn::Moe, h: &mut [f32], n: usize, tr: &mut dyn Trace) {
        self.ffn.moe(l, w, h, n, tr)
    }
}
impl<R: ffn::Routed> ffn::Head for CpuModel<R> {
    fn logits(&mut self, norm: &[f32], lm_head: &[u16], h: &[f32], n: usize, logits: &mut [f32]) {
        self.ffn.logits(norm, lm_head, h, n, logits)
    }
}
impl<R: ffn::Routed> Model for CpuModel<R> {}

impl<A: attn::Attention + ?Sized> attn::Attention for Box<A> {
    fn step(&mut self, l: usize, w: &attn::Weights, h: &mut [f32], pos: usize, n: usize, cos: &[f32], sin: &[f32], scale: f32, tr: &mut dyn Trace) {
        (**self).step(l, w, h, pos, n, cos, sin, scale, tr)
    }
    fn seed(&mut self, rows: &[f32]) {
        (**self).seed(rows)
    }
    fn rows(&self, i: usize, out: &mut [f32]) {
        (**self).rows(i, out)
    }
}

/// The routed experts from bf16 tensors looked up by (layer, expert, name)
/// — `"gate_proj"`, `"up_proj"`, `"down_proj"` — through the plain bf16
/// GEMV: the recipe-free realization. The bundle builder uses it to compute
/// the sink state; bring-up uses it to price recipes.
pub struct Bf16Experts<'a, F: FnMut(usize, usize, &str) -> &'a [u16]> {
    pub get: F,
    gu: Vec<f32>,
    h: Vec<f32>,
}
impl<'a, F: FnMut(usize, usize, &str) -> &'a [u16]> Bf16Experts<'a, F> {
    pub fn new(get: F) -> Self {
        Bf16Experts { get, gu: vec![0.0; 2 * D_FF], h: vec![0.0; D_FF] }
    }
}
impl<'a, F: FnMut(usize, usize, &str) -> &'a [u16]> ffn::Routed for Bf16Experts<'a, F> {
    fn run(&mut self, l: usize, idx: &[usize; TOP_K], x: &[f32], out: &mut [f32]) {
        for k in 0..TOP_K {
            common::gemv::bf16::<D_FF, H>((self.get)(l, idx[k], "gate_proj"), x, &mut self.gu[..D_FF]);
            common::gemv::bf16::<D_FF, H>((self.get)(l, idx[k], "up_proj"), x, &mut self.gu[D_FF..]);
            common::swiglu::swiglu::<D_FF>(&self.gu, &mut self.h);
            common::gemv::bf16::<H, D_FF>((self.get)(l, idx[k], "down_proj"), &self.h, &mut out[k * H..(k + 1) * H]);
        }
    }
}

impl<F: ffn::FeedForward + ?Sized> ffn::FeedForward for Box<F> {
    fn dense(&mut self, l: usize, w: &ffn::Dense, h: &mut [f32], n: usize, tr: &mut dyn Trace) {
        (**self).dense(l, w, h, n, tr)
    }
    fn moe(&mut self, l: usize, w: &ffn::Moe, h: &mut [f32], n: usize, tr: &mut dyn Trace) {
        (**self).moe(l, w, h, n, tr)
    }
}
impl<D: ffn::Head + ?Sized> ffn::Head for Box<D> {
    fn logits(&mut self, norm: &[f32], lm_head: &[u16], h: &[f32], n: usize, logits: &mut [f32]) {
        (**self).logits(norm, lm_head, h, n, logits)
    }
}

/// The FFN half and the head on this CPU, with the routed experts on
/// whatever `Routed` it was given. The reference placement.
pub struct CpuFfn<R: ffn::Routed> {
    pub routed: R,
    hn: Vec<f32>,
    gu: Vec<f32>,
    h: Vec<f32>,
    out: Vec<f32>,
    expert_out: Vec<f32>,
    router: Vec<f32>,
}

impl<R: ffn::Routed> CpuFfn<R> {
    pub fn new(routed: R) -> Self {
        CpuFfn {
            routed,
            hn: vec![0.0; H],
            gu: vec![0.0; 2 * INTERMEDIATE_SIZE],
            h: vec![0.0; INTERMEDIATE_SIZE],
            out: vec![0.0; H],
            expert_out: vec![0.0; TOP_K * H],
            router: vec![0.0; N_ROUTED_EXPERTS],
        }
    }
    fn scratch(&mut self) -> ffn::Scratch<'_> {
        ffn::Scratch { hn: &mut self.hn, gu: &mut self.gu, h: &mut self.h, out: &mut self.out, expert_out: &mut self.expert_out, router: &mut self.router }
    }
}

impl<R: ffn::Routed> ffn::FeedForward for CpuFfn<R> {
    /// The CPU steps its tokens one after another: `n` tokens are `n` calls of the one-token op.
    fn dense(&mut self, l: usize, w: &ffn::Dense, h: &mut [f32], n: usize, tr: &mut dyn Trace) {
        let mut tr = tr;
        for i in 0..n {
            ffn::dense(l, w, &mut h[i * H..(i + 1) * H], &mut self.scratch(), &mut tr);
        }
    }
    fn moe(&mut self, l: usize, w: &ffn::Moe, h: &mut [f32], n: usize, tr: &mut dyn Trace) {
        let mut tr = tr;
        // the scratch and the routed realization are both fields; split the borrow by hand
        let s = self;
        for i in 0..n {
            let mut sc = ffn::Scratch { hn: &mut s.hn, gu: &mut s.gu, h: &mut s.h, out: &mut s.out, expert_out: &mut s.expert_out, router: &mut s.router };
            ffn::moe(l, w, &mut h[i * H..(i + 1) * H], &mut sc, &mut s.routed, &mut tr);
        }
    }
}
impl<R: ffn::Routed> ffn::Head for CpuFfn<R> {
    fn logits(&mut self, norm: &[f32], lm_head: &[u16], h: &[f32], n: usize, logits: &mut [f32]) {
        for i in 0..n {
            common::rmsnorm::rmsnorm::<H>(&h[i * H..(i + 1) * H], norm, RMS_NORM_EPS, &mut self.hn);
            common::gemv::bf16::<VOCAB_SIZE, H>(lm_head, &self.hn, &mut logits[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE]);
        }
    }
}

/// Attention on this CPU: `attn::step` with the cache and scratch in this
/// process's memory. The reference placement.
pub struct CpuAttention {
    pub cache: Vec<Vec<f32>>, // per layer: [max_ctx][KV_ROW]
    last_pos: usize,          // position of token 0 of the last step, for `rows`
    hn: Vec<f32>,
    q: Vec<f32>,
    kva: Vec<f32>,
    heads: Vec<f32>,
    o: Vec<f32>,
    q_abs: Vec<f32>,
    scores: Vec<f32>,
    o_lat: Vec<f32>,
}

impl CpuAttention {
    pub fn new(max_ctx: usize) -> Self {
        CpuAttention {
            cache: (0..NUM_HIDDEN_LAYERS).map(|_| vec![0f32; max_ctx * KV_ROW]).collect(),
            last_pos: 0,
            hn: vec![0.0; H],
            q: vec![0.0; Q_DIM],
            kva: vec![0.0; KVA_DIM],
            heads: vec![0.0; O_IN_DIM],
            o: vec![0.0; H],
            q_abs: vec![0.0; KV_LORA_RANK],
            scores: vec![0.0; max_ctx],
            o_lat: vec![0.0; KV_LORA_RANK],
        }
    }
}

impl attn::Attention for CpuAttention {
    /// One token after another: `n` tokens are `n` calls of the one-token op.
    fn step(&mut self, l: usize, w: &attn::Weights, h: &mut [f32], pos: usize, n: usize, cos: &[f32], sin: &[f32], scale: f32, tr: &mut dyn Trace) {
        const D: usize = QK_ROPE_HEAD_DIM;
        let s = self;
        s.last_pos = pos;
        let mut tr = tr;
        for i in 0..n {
            let mut sc = attn::Scratch { hn: &mut s.hn, q: &mut s.q, kva: &mut s.kva, heads: &mut s.heads, o: &mut s.o, q_abs: &mut s.q_abs, scores: &mut s.scores, o_lat: &mut s.o_lat };
            attn::step(l, w, &mut h[i * H..(i + 1) * H], &mut s.cache[l], pos + i, &cos[i * D..(i + 1) * D], &sin[i * D..(i + 1) * D], scale, &mut sc, &mut tr);
        }
    }
    fn seed(&mut self, rows: &[f32]) {
        assert_eq!(rows.len() % layout::POSITION, 0);
        for (p, position) in rows.chunks_exact(layout::POSITION).enumerate() {
            for (l, c) in self.cache.iter_mut().enumerate() {
                c[p * KV_ROW..(p + 1) * KV_ROW].copy_from_slice(&position[l * KV_ROW..(l + 1) * KV_ROW]);
            }
        }
    }
    fn rows(&self, i: usize, out: &mut [f32]) {
        let pos = self.last_pos + i;
        for (l, c) in self.cache.iter().enumerate() {
            out[l * KV_ROW..(l + 1) * KV_ROW].copy_from_slice(&c[pos * KV_ROW..(pos + 1) * KV_ROW]);
        }
    }
}

/// YaRN rotary tables and the attention scale, built exactly as the oracle's
/// `DeepseekV2YarnRotaryEmbedding._set_cos_sin_cache` and
/// `DeepseekV2Attention.__init__` build them.
pub struct Rope {
    pub cos: Vec<f32>, // [max_ctx][QK_ROPE_HEAD_DIM]
    pub sin: Vec<f32>,
    pub softmax_scale: f32,
    pub inv_freq: Vec<f32>, // [QK_ROPE_HEAD_DIM / 2], as the oracle's
}

impl Rope {
    pub fn new(max_ctx: usize) -> Self {
        const D: usize = QK_ROPE_HEAD_DIM;
        let factor = ROPE_FACTOR as f64;
        // the mscales are the oracle's formulas; the inverse frequencies are its values
        // (ROPE_INV_FREQ): f32 pow does not round like f64 pow rounded, and at long positions it shows
        let mscale = |m: f64| if factor <= 1.0 { 1.0 } else { 0.1 * m * factor.ln() + 1.0 };
        let cs_mscale = (mscale(ROPE_MSCALE as f64) / mscale(ROPE_MSCALE_ALL_DIM as f64)) as f32;
        let inv_freq: Vec<f32> = ROPE_INV_FREQ.to_vec();
        let mut cos = vec![0f32; max_ctx * D];
        let mut sin = vec![0f32; max_ctx * D];
        for p in 0..max_ctx {
            for j in 0..D / 2 {
                let f = p as f32 * inv_freq[j];
                let (c, s) = (f.cos() * cs_mscale, f.sin() * cs_mscale);
                cos[p * D + j] = c;
                cos[p * D + j + D / 2] = c;
                sin[p * D + j] = s;
                sin[p * D + j + D / 2] = s;
            }
        }
        let m = mscale(ROPE_MSCALE_ALL_DIM as f64);
        Rope { cos, sin, softmax_scale: ((Q_HEAD_DIM as f64).powf(-0.5) * m * m) as f32, inv_freq }
    }
}

/// Everything a step touches that no realization owns, allocated once:
/// `h` is `[TOKENS][H]`, `logits` `[TOKENS][VOCAB_SIZE]`.
pub struct State {
    pub max_ctx: usize,
    pub rope: Rope,
    pub h: Vec<f32>,
    pub hn: Vec<f32>,
    pub logits: Vec<f32>,
}

impl State {
    pub fn new(max_ctx: usize) -> Self {
        #[cfg(target_arch = "x86_64")]
        assert!(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"), "the shipped realizations need AVX2+FMA");
        State {
            max_ctx,
            rope: Rope::new(max_ctx),
            h: vec![0.0; TOKENS * H],
            hn: vec![0.0; H],
            logits: vec![0.0; TOKENS * VOCAB_SIZE],
        }
    }

    /// `toks` (at most TOKENS of them) at positions `pos..pos + n` — the
    /// cache holds `0..pos` — through `model`, leaving every token's logits
    /// in `logits[i]`. The trace sees the first token's edges. Allocation-free.
    pub fn run(&mut self, w: &Weights, toks: &[u32], pos: usize, model: &mut impl Model, tr: &mut impl Trace) {
        const D: usize = QK_ROPE_HEAD_DIM;
        let n = toks.len();
        assert!(n >= 1 && n <= TOKENS && pos + n <= self.max_ctx);
        let s = self;
        for (i, &tok) in toks.iter().enumerate() {
            let row = &w.embed[tok as usize * H..(tok as usize + 1) * H];
            for j in 0..H {
                s.h[i * H + j] = widen(row[j]);
            }
        }
        tr.edge(NONE, "model.embed_tokens.out", &s.h[..H]);
        let cos = &s.rope.cos[pos * D..(pos + n) * D];
        let sin = &s.rope.sin[pos * D..(pos + n) * D];

        for (l, lw) in w.layers.iter().enumerate() {
            let aw = attn::Weights { input_ln: &lw.input_ln, q_proj: lw.q_proj, kv_a: lw.kv_a, kv_a_ln: &lw.kv_a_ln, kv_b: &lw.kv_b, o_proj: lw.o_proj };
            model.step(l, &aw, &mut s.h[..n * H], pos, n, cos, sin, s.rope.softmax_scale, tr);

            match &lw.ffn {
                Ffn::Dense { gate, up, down } => model.dense(l, &ffn::Dense { post_ln: &lw.post_ln, gate, up, down }, &mut s.h[..n * H], n, tr),
                Ffn::Moe { router, shared_gate, shared_up, shared_down } => {
                    model.moe(l, &ffn::Moe { post_ln: &lw.post_ln, router, shared_gate, shared_up, shared_down }, &mut s.h[..n * H], n, tr)
                }
            }
            tr.edge(l, "out", &s.h[..H]);
        }

        common::rmsnorm::rmsnorm::<H>(&s.h[..H], &w.norm, RMS_NORM_EPS, &mut s.hn);
        tr.edge(NONE, "model.norm.out", &s.hn);
        model.logits(&w.norm, w.lm_head, &s.h[..n * H], n, &mut s.logits[..n * VOCAB_SIZE]);
        tr.edge(NONE, "lm_head.out", &s.logits[..VOCAB_SIZE]);
    }

    /// One token at `pos` through `model`, then the next token drawn under
    /// `recipe` with the uniform `u` (`Sampling::GREEDY` for the argmax).
    pub fn decode(&mut self, w: &Weights, tok: u32, pos: usize, model: &mut impl Model, recipe: &Sampling, u: f64, tr: &mut impl Trace) -> usize {
        self.run(w, &[tok], pos, model, tr);
        model.draw(0, recipe, u, &self.logits[..VOCAB_SIZE])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The YaRN tables against the oracle's own at positions up to the
    /// checkpoint's maximum (`bringup/rope_tables.py --out goldens/v2lite`
    /// writes them). The products are f32 as torch's are, so the angles round
    /// alike; the cosines then agree to a few ulp.
    #[test]
    fn rope_matches_the_oracle_at_long_positions() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bringup/goldens/v2lite/rope_long.json");
        let Ok(text) = std::fs::read_to_string(path) else {
            eprintln!("no {path}: run bringup/rope_tables.py --out goldens/v2lite");
            return;
        };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let positions: Vec<usize> = v["positions"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let max = positions.iter().max().unwrap() + 1;
        let rope = Rope::new(max);
        if let Ok(t) = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../bringup/goldens/v2lite/rope.json")) {
            let r: serde_json::Value = serde_json::from_str(&t).unwrap();
            let theirs: Vec<f32> = r["inv_freq"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            for (j, (a, b)) in rope.inv_freq.iter().zip(&theirs).enumerate() {
                if a != b {
                    eprintln!("inv_freq[{j}]: ours {a:e} ({:#x}) oracle {b:e} ({:#x})", a.to_bits(), b.to_bits());
                }
            }
        }
        const D: usize = QK_ROPE_HEAD_DIM;
        let mut worst = 0f32;
        for (k, &p) in positions.iter().enumerate() {
            for j in 0..D {
                let c = v["cos"][k][j].as_f64().unwrap() as f32;
                let s = v["sin"][k][j].as_f64().unwrap() as f32;
                worst = worst.max((rope.cos[p * D + j] - c).abs()).max((rope.sin[p * D + j] - s).abs());
            }
        }
        assert!(worst < 2e-6, "rope tables differ from the oracle by {worst}");
    }
}
