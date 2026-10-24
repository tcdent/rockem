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
    /// The slices `quantize(t)` writes, as pointers an item of the pool takes (the tokens' slices are disjoint).
    fn writer(&mut self) -> ActsOut {
        ActsOut { q: self.q.as_mut_ptr() as usize, scale: self.scale.as_mut_ptr() as usize, gsum: self.gsum.as_mut_ptr() as usize, k: self.k }
    }
}
#[derive(Clone, Copy)]
struct ActsOut {
    q: usize,
    scale: usize,
    gsum: usize,
    k: usize,
}
impl ActsOut {
    /// Token `t`'s INT8 activation from `x` (`quantize`'s bits).
    unsafe fn quantize(self, t: usize, x: &[f32]) {
        let (k, g) = (self.k, self.k / GROUP);
        let mut qa = QAct {
            q: core::slice::from_raw_parts_mut((self.q as *mut i8).add(t * k), k),
            scale: core::slice::from_raw_parts_mut((self.scale as *mut f32).add(t * g), g),
            gsum: core::slice::from_raw_parts_mut((self.gsum as *mut i32).add(t * g), g),
        };
        qa.quantize(&x[..k]);
    }
}

/// The step's per-token work over the pool, a token an item; one token runs inline.
fn tokens(pool: &dyn Pool, n: usize, f: &(dyn Fn(usize) + Sync)) {
    if n == 1 {
        f(0)
    } else {
        pool.run(n, f)
    }
}

/// Where a step's time goes (`phases`, seconds), in order.
pub const PHASES: [&str; 10] = ["embed", "norm", "qkv", "rope", "attn", "o", "gu", "silu", "down", "head"];
fn phase(phases: &mut [f64; 10], i: usize, t: &mut std::time::Instant) {
    let now = std::time::Instant::now();
    phases[i] += (now - *t).as_secs_f64();
    *t = now;
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
    /// Seconds in each of `PHASES` over the last `step`.
    pub phases: [f64; 10],
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
            phases: [0.0; 10],
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
            // down's 20 items are 256 rows each: split four ways at runtime so 80 cores stream it, a sub-item's rows
            // (whole tiles) and their scales contiguous within the item
            Which::Down => {
                const SPLIT: usize = 4;
                let rows = m.chunk / SPLIT;
                self.pool.run_on(m.items() * SPLIT, &|i| node(i / SPLIT), &|i| i, &move |i| {
                    let (c, s) = (i / SPLIT, i % SPLIT);
                    let it = stream.item(l, which, c);
                    q4::gemm::<D_FF>(rows, &it.w[s * rows * D_FF / 2..], &it.s[s * rows * (D_FF / GROUP)..], n, |t| act.act(t), |t| out.at(t * stride + c * m.chunk + s * rows));
                })
            }
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
        self.phases = [0.0; 10];
        let mut tm = std::time::Instant::now();
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
        phase(&mut self.phases, 0, &mut tm);
        for l in 0..NUM_HIDDEN_LAYERS {
            self.layer(l, n, pos, &mut tm);
            if let Some(tr) = &mut self.trace {
                tr.extend_from_slice(&self.x[..n * H]);
            }
        }
        // the head: the normed residual of the tokens wanted, the lm_head's items over the pool
        let first = if all { 0 } else { n - 1 };
        {
            let (x, fnorm, xn, act) = (&self.x, &self.final_norm, Out(self.xn.as_mut_ptr()), self.act_h.writer());
            tokens(self.pool, n - first, &move |i| unsafe {
                let t = first + i;
                let row = core::slice::from_raw_parts_mut(xn.at(t * H), H);
                rmsnorm::<H>(&x[t * H..(t + 1) * H], fnorm, RMS_NORM_EPS, row);
                act.quantize(t, row);
            });
        }
        let m = layout::LM_HEAD;
        let (stream, nodes, per_run) = (self.stream, self.nodes, m.per_run());
        let out = Out(self.logits.as_mut_ptr());
        let act = &self.act_h;
        self.pool.run_on(m.items(), &|c| (c / per_run) % nodes, &|c| c, &move |c| {
            let it = stream.head_item(c);
            q4::gemm::<H>(m.chunk, it.w, it.s, n - first, |t| act.act(first + t), |t| out.at((first + t) * VOCAB_SIZE + c * m.chunk));
        });
        phase(&mut self.phases, 9, &mut tm);
    }

    /// The step's tokens normed by `norm` and quantized into `act_h`, a token an item.
    fn norm_tokens(&mut self, n: usize, norm: &[f32]) {
        let (x, xn, act) = (&self.x, Out(self.xn.as_mut_ptr()), self.act_h.writer());
        tokens(self.pool, n, &move |t| unsafe {
            let row = core::slice::from_raw_parts_mut(xn.at(t * H), H);
            rmsnorm::<H>(&x[t * H..(t + 1) * H], norm, RMS_NORM_EPS, row);
            act.quantize(t, row);
        });
    }

    fn layer(&mut self, l: usize, n: usize, pos: usize, tm: &mut std::time::Instant) {
        let norms = std::mem::take(&mut self.norms[l]);
        let (ln0, ln1) = norms.split_at(H);
        // attention: the normed input quantized a token, q|k|v an item over the pool
        self.norm_tokens(n, ln0);
        phase(&mut self.phases, 1, tm);
        let stride = layout::QKV.rows;
        let out = Out(self.qkv.as_mut_ptr());
        self.gemv(l, Which::Qkv, n, &self.act_h, out, stride);
        phase(&mut self.phases, 2, tm);
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
        phase(&mut self.phases, 3, tm);
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
        phase(&mut self.phases, 4, tm);
        // o projection over the attention output, into the residual
        {
            let (attn, act) = (&self.attn, self.act_q.writer());
            tokens(self.pool, n, &move |t| unsafe { act.quantize(t, &attn[t * Q_DIM..(t + 1) * Q_DIM]) });
        }
        let out = Out(self.xn.as_mut_ptr());
        self.gemv(l, Which::O, n, &self.act_q, out, H);
        for t in 0..n * H {
            self.x[t] += self.xn[t];
        }
        phase(&mut self.phases, 5, tm);
        // the FFN: gate|up an item over the pool, SiLU(gate)·up a token (an item), down an item, into the residual
        self.norm_tokens(n, ln1);
        phase(&mut self.phases, 1, tm);
        // the gate rows into `gu`'s first half, the up rows into its second (`STEP · D_FF` floats on)
        let out = Out(self.gu.as_mut_ptr());
        self.gemv(l, Which::Gu, n, &self.act_h, out, D_FF);
        phase(&mut self.phases, 6, tm);
        {
            let (gate, up) = self.gu.split_at(STEP * D_FF);
            let (hid, act) = (Out(self.hid.as_mut_ptr()), self.act_ff.writer());
            tokens(self.pool, n, &move |t| unsafe {
                let row = core::slice::from_raw_parts_mut(hid.at(t * D_FF), D_FF);
                for i in 0..D_FF {
                    row[i] = silu(gate[t * D_FF + i]) * up[t * D_FF + i];
                }
                act.quantize(t, row);
            });
        }
        phase(&mut self.phases, 7, tm);
        let out = Out(self.xn.as_mut_ptr());
        self.gemv(l, Which::Down, n, &self.act_ff, out, H);
        for t in 0..n * H {
            self.x[t] += self.xn[t];
        }
        phase(&mut self.phases, 8, tm);
        self.norms[l] = norms;
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
