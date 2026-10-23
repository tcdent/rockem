//! The decoder on the host: Mistral Small 3.1's 40 layers over the q4 stream, a step of up to `STEP`
//! tokens at a time. Every matrix is streamed as items over the pool (`q4::gemm` over the step's tokens
//! an item, so an item's rows are read once a step), the activations INT8 per token (`q4::QAct`), the
//! residual, norms, rotary, attention and SiLU in f32. The K/V cache is f32, `[layer][position][KV_DIM]`
//! each; a position's rows for the node's context store are its K then V of every layer
//! (`POSITION` f32). An audio placeholder's input is the row given for it instead of its embedding.

use crate::layout::{self, Which};
use crate::weights::{widen, Dense, Stream};
use crate::*;
use common::math::{exp, sqrt};
use common::pool::Pool;
use common::q4::{self, Act, QAct, GROUP};
use common::rmsnorm::rmsnorm;
use common::rope::rotate_half;
use common::swiglu::silu;

/// Tokens a step takes at most.
pub const STEP: usize = 32;
/// f32s of a position's rows: K and V of every layer.
pub const POSITION: usize = NUM_HIDDEN_LAYERS * 2 * KV_DIM;
/// The model's own context limit (the checkpoint's `max_position_embeddings`).
pub const MAX_CTX: usize = 32768;
const _: () = assert!(STEP <= q4::GEMM_TOKENS);

/// A pointer the pool's items write disjoint parts of.
#[derive(Clone, Copy)]
struct Out(*mut f32);
unsafe impl Sync for Out {}
unsafe impl Send for Out {}
impl Out {
    /// The pointer `i` floats in (a method, so a closure captures the whole `Out`, not its raw field).
    #[inline(always)]
    fn at(self, i: usize) -> *mut f32 {
        unsafe { self.0.add(i) }
    }
}

/// Per-token INT8 activations of width `K`, as a GEMM's `act(t)`.
struct Acts {
    q: Vec<i8>,
    scale: Vec<f32>,
    gsum: Vec<i32>,
    k: usize,
}
impl Acts {
    fn new(k: usize) -> Self {
        Acts { q: vec![0; STEP * k], scale: vec![0.0; STEP * k / GROUP], gsum: vec![0; STEP * k / GROUP], k }
    }
    fn quantize(&mut self, t: usize, x: &[f32]) {
        let (k, g) = (self.k, self.k / GROUP);
        let mut qa = QAct { q: &mut self.q[t * k..(t + 1) * k], scale: &mut self.scale[t * g..(t + 1) * g], gsum: &mut self.gsum[t * g..(t + 1) * g] };
        qa.quantize(&x[..k]);
    }
    fn act(&self, t: usize) -> Act<'_> {
        let (k, g) = (self.k, self.k / GROUP);
        Act { q: &self.q[t * k..(t + 1) * k], scale: &self.scale[t * g..(t + 1) * g], gsum: &self.gsum[t * g..(t + 1) * g] }
    }
}

pub struct Decoder<'a> {
    dense: Dense<'a>,
    stream: Stream<'a>,
    pool: &'a dyn Pool,
    /// NUMA nodes the stream's runs are spread over (run `r` on node `r mod nodes`).
    nodes: usize,
    max_ctx: usize,
    norms: Vec<Vec<f32>>, // [l][2][H] widened
    final_norm: Vec<f32>,
    cos: Vec<f32>, // [max_ctx][HEAD_DIM]
    sin: Vec<f32>,
    k_cache: Vec<f32>, // [layers][max_ctx][KV_DIM]
    v_cache: Vec<f32>,
    // the step's scratch
    x: Vec<f32>,    // [STEP][H]
    xn: Vec<f32>,   // [STEP][H]
    qkv: Vec<f32>,  // [STEP][QKV rows]
    attn: Vec<f32>, // [STEP][Q_DIM]
    gu: Vec<f32>,   // [2][STEP][D_FF]: the gate rows, then the up rows
    hid: Vec<f32>,  // [STEP][D_FF]
    act_h: Acts,
    act_q: Acts,
    act_ff: Acts,
    pub logits: Vec<f32>, // [STEP][VOCAB_SIZE]
    /// Dev: every layer's residual output of the last step, `[layers][n][H]`, when tracing.
    pub trace: Option<Vec<f32>>,
}

impl<'a> Decoder<'a> {
    pub fn new(dense: Dense<'a>, stream: Stream<'a>, pool: &'a dyn Pool, nodes: usize, max_ctx: usize) -> Self {
        let max_ctx = max_ctx.min(MAX_CTX);
        let norms = (0..NUM_HIDDEN_LAYERS).map(|l| [widen(dense.layer_norm(l, 0)), widen(dense.layer_norm(l, 1))].concat()).collect();
        // rotary: inv_freq_i = θ^(−2i/d), the position's angles repeated over both halves (rotate_half)
        let half = HEAD_DIM / 2;
        let inv: Vec<f64> = (0..half).map(|i| 1.0 / (ROPE_THETA as f64).powf(2.0 * i as f64 / HEAD_DIM as f64)).collect();
        let (mut cos, mut sin) = (vec![0f32; max_ctx * HEAD_DIM], vec![0f32; max_ctx * HEAD_DIM]);
        for p in 0..max_ctx {
            for i in 0..half {
                let a = p as f64 * inv[i];
                let (s, c) = a.sin_cos();
                cos[p * HEAD_DIM + i] = c as f32;
                cos[p * HEAD_DIM + half + i] = c as f32;
                sin[p * HEAD_DIM + i] = s as f32;
                sin[p * HEAD_DIM + half + i] = s as f32;
            }
        }
        Decoder {
            dense,
            stream,
            pool,
            nodes: nodes.max(1),
            max_ctx,
            norms,
            final_norm: widen(dense.final_norm()),
            cos,
            sin,
            k_cache: vec![0.0; NUM_HIDDEN_LAYERS * max_ctx * KV_DIM],
            v_cache: vec![0.0; NUM_HIDDEN_LAYERS * max_ctx * KV_DIM],
            x: vec![0.0; STEP * H],
            xn: vec![0.0; STEP * H],
            qkv: vec![0.0; STEP * layout::QKV.rows],
            attn: vec![0.0; STEP * Q_DIM],
            gu: vec![0.0; 2 * STEP * D_FF],
            hid: vec![0.0; STEP * D_FF],
            act_h: Acts::new(H),
            act_q: Acts::new(Q_DIM),
            act_ff: Acts::new(D_FF),
            logits: vec![0.0; STEP * VOCAB_SIZE],
            trace: None,
        }
    }
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// Matrix `which` of layer `l` over the step's `n` tokens: every item on the pool, on the node that holds
    /// it; `out(t)` is token `t`'s output row.
    fn gemv(&self, l: usize, which: Which, n: usize, act: &Acts, out: Out, stride: usize) {
        let m = layout::mat(which);
        let (stream, nodes) = (self.stream, self.nodes);
        let per_run = m.per_run();
        let node = |c: usize| (c / per_run) % nodes;
        let ccd = |c: usize| c;
        match which {
            Which::Gu => {
                let half = m.chunk / 2;
                let gate = Out(out.0);
                let up = Out(out.at(STEP * D_FF));
                self.pool.run_on(m.items(), &node, &ccd, &move |c| {
                    let (gw, uw, gs, us) = stream.gu(l, c);
                    q4::gemm::<H>(half, gw, gs, n, |t| act.act(t), |t| gate.at(t * stride + c * half));
                    q4::gemm::<H>(half, uw, us, n, |t| act.act(t), |t| up.at(t * stride + c * half));
                });
            }
            Which::Down => self.pool.run_on(m.items(), &node, &ccd, &move |c| {
                let it = stream.item(l, which, c);
                q4::gemm::<D_FF>(m.chunk, it.w, it.s, n, |t| act.act(t), |t| out.at(t * stride + c * m.chunk));
            }),
            Which::O => self.pool.run_on(m.items(), &node, &ccd, &move |c| {
                let it = stream.item(l, which, c);
                q4::gemm::<Q_DIM>(m.chunk, it.w, it.s, n, |t| act.act(t), |t| out.at(t * stride + c * m.chunk));
            }),
            Which::Qkv => self.pool.run_on(m.items(), &node, &ccd, &move |c| {
                let it = stream.item(l, which, c);
                q4::gemm::<H>(m.chunk, it.w, it.s, n, |t| act.act(t), |t| out.at(t * stride + c * m.chunk));
            }),
        }
    }

    /// `toks` at positions `pos..pos + n`: each token's embedding, or `row(t)`'s row for a placeholder (an
    /// audio row); afterwards `logits` holds every token's logits (`all`) or the last token's at row `n - 1`.
    pub fn step(&mut self, toks: &[u32], pos: usize, row: impl Fn(usize) -> Option<&'a [f32]>, all: bool) {
        let n = toks.len();
        assert!(n > 0 && n <= STEP && pos + n <= self.max_ctx);
        let emb = self.dense.embed_tokens();
        for t in 0..n {
            let x = &mut self.x[t * H..(t + 1) * H];
            match row(t) {
                Some(r) => x.copy_from_slice(&r[..H]),
                None => {
                    let id = toks[t] as usize;
                    for (o, &v) in x.iter_mut().zip(&emb[id * H..(id + 1) * H]) {
                        *o = f32::from_bits((v as u32) << 16);
                    }
                }
            }
        }
        if let Some(tr) = &mut self.trace {
            tr.clear();
        }
        for l in 0..NUM_HIDDEN_LAYERS {
            self.layer(l, n, pos);
            if let Some(tr) = &mut self.trace {
                tr.extend_from_slice(&self.x[..n * H]);
            }
        }
        // the head: the normed residual of the tokens wanted, the lm_head's items over the pool
        let first = if all { 0 } else { n - 1 };
        for t in first..n {
            rmsnorm::<H>(&self.x[t * H..(t + 1) * H], &self.final_norm, RMS_NORM_EPS, &mut self.xn[t * H..(t + 1) * H]);
            self.act_h.quantize(t, &self.xn[t * H..(t + 1) * H]);
        }
        let m = layout::LM_HEAD;
        let (stream, nodes, per_run) = (self.stream, self.nodes, m.per_run());
        let out = Out(self.logits.as_mut_ptr());
        let act = &self.act_h;
        self.pool.run_on(m.items(), &|c| (c / per_run) % nodes, &|c| c, &move |c| {
            let it = stream.head_item(c);
            q4::gemm::<H>(m.chunk, it.w, it.s, n - first, |t| act.act(first + t), |t| out.at((first + t) * VOCAB_SIZE + c * m.chunk));
        });
    }

    fn layer(&mut self, l: usize, n: usize, pos: usize) {
        let (ln0, ln1) = self.norms[l].split_at(H);
        // attention: the normed input quantized a token, q|k|v an item over the pool
        for t in 0..n {
            rmsnorm::<H>(&self.x[t * H..(t + 1) * H], ln0, RMS_NORM_EPS, &mut self.xn[t * H..(t + 1) * H]);
            self.act_h.quantize(t, &self.xn[t * H..(t + 1) * H]);
        }
        let stride = layout::QKV.rows;
        let out = Out(self.qkv.as_mut_ptr());
        self.gemv(l, Which::Qkv, n, &self.act_h, out, stride);
        // rotary on q and k, then k and v into the cache
        for t in 0..n {
            let p = pos + t;
            let (cos, sin) = (&self.cos[p * HEAD_DIM..(p + 1) * HEAD_DIM], &self.sin[p * HEAD_DIM..(p + 1) * HEAD_DIM]);
            let row = &mut self.qkv[t * stride..(t + 1) * stride];
            for h in 0..HEADS {
                rotate_half::<HEAD_DIM>(&mut row[h * HEAD_DIM..(h + 1) * HEAD_DIM], cos, sin);
            }
            for h in 0..KV_HEADS {
                rotate_half::<HEAD_DIM>(&mut row[Q_DIM + h * HEAD_DIM..Q_DIM + (h + 1) * HEAD_DIM], cos, sin);
            }
            let at = (l * self.max_ctx + p) * KV_DIM;
            self.k_cache[at..at + KV_DIM].copy_from_slice(&row[Q_DIM..Q_DIM + KV_DIM]);
            self.v_cache[at..at + KV_DIM].copy_from_slice(&row[Q_DIM + KV_DIM..Q_DIM + 2 * KV_DIM]);
        }
        // causal attention, a (token, head) an item: scores over the positions up to the token's, softmax, the values
        {
            let (qkv, k_cache, v_cache) = (&self.qkv, &self.k_cache, &self.v_cache);
            let (max_ctx, out) = (self.max_ctx, Out(self.attn.as_mut_ptr()));
            let scale = 1.0 / sqrt(HEAD_DIM as f32);
            self.pool.run(n * HEADS, &move |i| {
                let (t, h) = (i / HEADS, i % HEADS);
                let kvh = h / KV_GROUP;
                let q = &qkv[t * stride + h * HEAD_DIM..t * stride + (h + 1) * HEAD_DIM];
                let len = pos + t + 1;
                let mut s = vec![0f32; len];
                let mut mx = f32::NEG_INFINITY;
                for p in 0..len {
                    let k = &k_cache[(l * max_ctx + p) * KV_DIM + kvh * HEAD_DIM..][..HEAD_DIM];
                    let d = common::linalg::dot(q, k) * scale;
                    s[p] = d;
                    if d > mx {
                        mx = d;
                    }
                }
                let mut sum = 0f32;
                for v in s.iter_mut() {
                    *v = exp(*v - mx);
                    sum += *v;
                }
                let inv = 1.0 / sum;
                let o = unsafe { core::slice::from_raw_parts_mut(out.at(t * Q_DIM + h * HEAD_DIM), HEAD_DIM) };
                o.fill(0.0);
                for p in 0..len {
                    let v = &v_cache[(l * max_ctx + p) * KV_DIM + kvh * HEAD_DIM..][..HEAD_DIM];
                    common::linalg::axpy(o, s[p] * inv, v);
                }
            });
        }
        // o projection over the attention output, into the residual
        for t in 0..n {
            self.act_q.quantize(t, &self.attn[t * Q_DIM..(t + 1) * Q_DIM]);
        }
        let out = Out(self.xn.as_mut_ptr());
        self.gemv(l, Which::O, n, &self.act_q, out, H);
        for t in 0..n * H {
            self.x[t] += self.xn[t];
        }
        // the FFN: gate|up an item over the pool, SiLU(gate)·up a token, down an item, into the residual
        for t in 0..n {
            rmsnorm::<H>(&self.x[t * H..(t + 1) * H], ln1, RMS_NORM_EPS, &mut self.xn[t * H..(t + 1) * H]);
            self.act_h.quantize(t, &self.xn[t * H..(t + 1) * H]);
        }
        // the gate rows into `gu`'s first half, the up rows into its second (`STEP · D_FF` floats on)
        let out = Out(self.gu.as_mut_ptr());
        self.gemv(l, Which::Gu, n, &self.act_h, out, D_FF);
        let (gate, up) = self.gu.split_at(STEP * D_FF);
        for t in 0..n {
            for i in 0..D_FF {
                self.hid[t * D_FF + i] = silu(gate[t * D_FF + i]) * up[t * D_FF + i];
            }
            self.act_ff.quantize(t, &self.hid[t * D_FF..(t + 1) * D_FF]);
        }
        let out = Out(self.xn.as_mut_ptr());
        self.gemv(l, Which::Down, n, &self.act_ff, out, H);
        for t in 0..n * H {
            self.x[t] += self.xn[t];
        }
    }

    /// Position `pos`'s rows for the store: K then V of every layer.
    pub fn rows(&self, pos: usize, out: &mut [f32]) {
        for l in 0..NUM_HIDDEN_LAYERS {
            let at = (l * self.max_ctx + pos) * KV_DIM;
            out[l * 2 * KV_DIM..l * 2 * KV_DIM + KV_DIM].copy_from_slice(&self.k_cache[at..at + KV_DIM]);
            out[l * 2 * KV_DIM + KV_DIM..(l + 1) * 2 * KV_DIM].copy_from_slice(&self.v_cache[at..at + KV_DIM]);
        }
    }
    /// Install `rows` (`[n][POSITION]`) as positions `0..n`.
    pub fn seed(&mut self, rows: &[f32]) {
        let n = rows.len() / POSITION;
        for p in 0..n {
            for l in 0..NUM_HIDDEN_LAYERS {
                let at = (l * self.max_ctx + p) * KV_DIM;
                let r = &rows[p * POSITION + l * 2 * KV_DIM..p * POSITION + (l + 1) * 2 * KV_DIM];
                self.k_cache[at..at + KV_DIM].copy_from_slice(&r[..KV_DIM]);
                self.v_cache[at..at + KV_DIM].copy_from_slice(&r[KV_DIM..]);
            }
        }
    }
}
