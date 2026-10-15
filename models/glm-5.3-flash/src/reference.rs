//! The f32 transcription of the reference forward, layer-major: a layer runs
//! over every position of the passage before the next layer starts, which is
//! the oracle's order, so one layer's weights are resident at a time. The
//! projections are matrix products over the positions; the recurrences (the
//! short convolution, the delta rule) step through them in order, which is the
//! decode path's arithmetic — the oracle's prefill uses the chunked form of the
//! same recurrence, so the two differ by rounding only.
//!
//! Weights are f32, widened from the checkpoint by the caller. This module is
//! the reference the shipped kernels are diffed against, not one of them.

use crate::*;
use common::math::{exp, sqrt};
use common::swiglu::silu;

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, ra) = a.split_at(a.len() / 8 * 8);
    let (cb, rb) = b.split_at(ca.len());
    for (x, y) in ca.chunks_exact(8).zip(cb.chunks_exact(8)) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// `matmul` on the calling thread, for work that is already spread across threads.
pub fn matmul_serial(w: &[f32], m: usize, k: usize, x: &[f32], n: usize) -> Vec<f32> {
    let mut y = vec![0f32; n * m];
    for i in 0..m {
        let row = &w[i * k..(i + 1) * k];
        for s in 0..n {
            y[s * m + i] = dot(row, &x[s * k..(s + 1) * k]);
        }
    }
    y
}

/// `y[s][m] = Σ_k w[m][k] · x[s][k]` for `w: [m][k]`, `x: [n][k]`; the rows of
/// `w` are split across the machine's threads, each read once for all `n`.
pub fn matmul(w: &[f32], m: usize, k: usize, x: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(w.len(), m * k);
    assert_eq!(x.len(), n * k);
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).min(m);
    let per = m.div_ceil(threads);
    let parts: Vec<Vec<f32>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                sc.spawn(move || {
                    let (m0, m1) = (t * per, ((t + 1) * per).min(m));
                    let mut y = vec![0f32; n * (m1.saturating_sub(m0))];
                    for i in m0..m1 {
                        let row = &w[i * k..(i + 1) * k];
                        for s in 0..n {
                            y[s * (m1 - m0) + i - m0] = dot(row, &x[s * k..(s + 1) * k]);
                        }
                    }
                    y
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut y = vec![0f32; n * m];
    for (t, p) in parts.iter().enumerate() {
        let (m0, m1) = (t * per, ((t + 1) * per).min(m));
        if m1 <= m0 {
            continue;
        }
        for s in 0..n {
            y[s * m + m0..s * m + m1].copy_from_slice(&p[s * (m1 - m0)..(s + 1) * (m1 - m0)]);
        }
    }
    y
}

/// `w ⊙ x · rsqrt(mean(x²) + eps)` per row of `n` (the reference's RMSNorm;
/// `w` empty for the unweighted one mHC applies to the flattened streams).
pub fn rmsnorm(x: &[f32], n: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for (xr, yr) in x.chunks_exact(n).zip(y.chunks_exact_mut(n)) {
        let ss: f32 = xr.iter().map(|v| v * v).sum();
        let r = 1.0 / sqrt(ss / n as f32 + eps);
        for i in 0..n {
            yr[i] = if w.is_empty() { xr[i] * r } else { w[i] * (xr[i] * r) };
        }
    }
    y
}

/// One hyper-connection (`Glm5NextTextHyperConnection`): from the four streams,
/// the block's input and the weights that put its output back.
pub struct Hc {
    /// `[HC_COEFS][HC·H]`
    pub fn_: Vec<f32>,
    pub base: Vec<f32>,
    pub scale: [f32; 3],
}

#[derive(Clone, Copy)]
pub struct Mix {
    pub post: [f32; HC],
    /// `comb[i][j]`: how much of input stream `i` goes to output stream `j`.
    pub comb: [[f32; HC]; HC],
}

impl Hc {
    /// `streams: [HC][H]` of one position → the collapsed block input `[H]` and the mix.
    pub fn pre(&self, streams: &[f32], collapsed: &mut [f32]) -> Mix {
        let flat = rmsnorm(streams, HC * H, &[], RMS_NORM_EPS);
        let mut c = [0f32; HC_COEFS];
        for (i, ci) in c.iter_mut().enumerate() {
            *ci = dot(&self.fn_[i * HC * H..(i + 1) * HC * H], &flat);
        }
        let mut pre = [0f32; HC];
        let mut mix = Mix { post: [0.0; HC], comb: [[0.0; HC]; HC] };
        for i in 0..HC {
            pre[i] = sigmoid(c[i] * self.scale[0] + self.base[i]) + HC_EPS;
            mix.post[i] = 2.0 * sigmoid(c[HC + i] * self.scale[1] + self.base[HC + i]);
        }
        // comb: a softmax per row, then Sinkhorn-Knopp toward doubly stochastic
        // (columns first, then rows and columns `HC_SINKHORN_ITERS - 1` times)
        let cb = &mut mix.comb;
        for i in 0..HC {
            let row: [f32; HC] = core::array::from_fn(|j| c[2 * HC + i * HC + j] * self.scale[2] + self.base[2 * HC + i * HC + j]);
            let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let e: [f32; HC] = core::array::from_fn(|j| exp(row[j] - mx));
            let s: f32 = e.iter().sum();
            for j in 0..HC {
                cb[i][j] = e[j] / s + HC_EPS;
            }
        }
        let cols = |cb: &mut [[f32; HC]; HC]| {
            for j in 0..HC {
                let s: f32 = (0..HC).map(|i| cb[i][j]).sum::<f32>() + HC_EPS;
                for row in cb.iter_mut() {
                    row[j] /= s;
                }
            }
        };
        cols(cb);
        for _ in 0..HC_SINKHORN_ITERS - 1 {
            for row in cb.iter_mut() {
                let s: f32 = row.iter().sum::<f32>() + HC_EPS;
                for v in row.iter_mut() {
                    *v /= s;
                }
            }
            cols(cb);
        }
        for d in 0..H {
            collapsed[d] = (0..HC).map(|i| pre[i] * streams[i * H + d]).sum();
        }
        mix
    }
}

impl Mix {
    /// `out_j = post_j · y + Σ_i comb_ij · residual_i`, all `[HC][H]` but `y: [H]`.
    pub fn post(&self, y: &[f32], residual: &[f32], out: &mut [f32]) {
        for j in 0..HC {
            for d in 0..H {
                let mut acc = 0f32;
                for i in 0..HC {
                    acc += self.comb[i][j] * residual[i * H + d];
                }
                out[j * H + d] = self.post[j] * y[d] + acc;
            }
        }
    }
}

/// Kimi delta attention (`Glm5NextTextLinearAttention`). Shapes are `[out][in]`.
pub struct Kda {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// `[CONV_DIM][CONV_TAPS]`, q's channels then k's then v's
    pub conv: Vec<f32>,
    pub f_a: Vec<f32>,
    pub f_b: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub a_log: Vec<f32>,
    pub b: Vec<f32>,
    pub g_a: Vec<f32>,
    pub g_b: Vec<f32>,
    pub o_norm: Vec<f32>,
    pub o: Vec<f32>,
}

/// A sequence's KDA state for one layer: the convolution's last inputs and the
/// per-head `[k][v]` memory.
pub struct KdaState {
    pub window: Vec<f32>,
    pub s: Vec<f32>,
}

impl Default for KdaState {
    fn default() -> Self {
        KdaState { window: vec![0.0; (CONV_TAPS - 1) * CONV_DIM], s: vec![0.0; KDA_HEADS * KDA_DIM * KDA_DIM] }
    }
}

fn l2norm(x: &mut [f32]) {
    let r = 1.0 / sqrt(x.iter().map(|v| v * v).sum::<f32>() + 1e-6);
    for v in x.iter_mut() {
        *v *= r;
    }
}

impl Kda {
    /// `x: [n][H]`, the normed block input at each position, in order → `[n][H]`.
    pub fn forward(&self, x: &[f32], n: usize, st: &mut KdaState) -> Vec<f32> {
        let (q, k, v) = (matmul(&self.q, KDA_QKV, H, x, n), matmul(&self.k, KDA_QKV, H, x, n), matmul(&self.v, KDA_QKV, H, x, n));
        let f = matmul(&self.f_b, KDA_QKV, KDA_DIM, &matmul(&self.f_a, KDA_DIM, H, x, n), n);
        let beta = matmul(&self.b, KDA_HEADS, H, x, n);
        let gate = matmul(&self.g_b, KDA_QKV, KDA_DIM, &matmul(&self.g_a, KDA_DIM, H, x, n), n);
        let scale = 1.0 / sqrt(KDA_DIM as f32);
        let mut normed = vec![0f32; n * KDA_QKV];
        let mut mixed = vec![0f32; CONV_DIM];
        let mut conv = vec![0f32; CONV_DIM];
        for t in 0..n {
            mixed[..KDA_QKV].copy_from_slice(&q[t * KDA_QKV..(t + 1) * KDA_QKV]);
            mixed[KDA_QKV..2 * KDA_QKV].copy_from_slice(&k[t * KDA_QKV..(t + 1) * KDA_QKV]);
            mixed[2 * KDA_QKV..].copy_from_slice(&v[t * KDA_QKV..(t + 1) * KDA_QKV]);
            // causal depthwise conv: tap j sees the input j - (TAPS-1) steps back; window holds the last TAPS-1
            for c in 0..CONV_DIM {
                let w = &self.conv[c * CONV_TAPS..(c + 1) * CONV_TAPS];
                let mut acc = 0f32;
                for j in 0..CONV_TAPS - 1 {
                    acc += w[j] * st.window[j * CONV_DIM + c];
                }
                acc += w[CONV_TAPS - 1] * mixed[c];
                conv[c] = silu(acc);
            }
            st.window.copy_within(CONV_DIM.., 0);
            st.window[(CONV_TAPS - 2) * CONV_DIM..].copy_from_slice(&mixed);

            let ft = &f[t * KDA_QKV..(t + 1) * KDA_QKV];
            let gt = &gate[t * KDA_QKV..(t + 1) * KDA_QKV];
            for h in 0..KDA_HEADS {
                let r = h * KDA_DIM..(h + 1) * KDA_DIM;
                let mut qh: Vec<f32> = conv[r.clone()].to_vec();
                let mut kh: Vec<f32> = conv[KDA_QKV + h * KDA_DIM..KDA_QKV + (h + 1) * KDA_DIM].to_vec();
                let vh = &conv[2 * KDA_QKV + h * KDA_DIM..2 * KDA_QKV + (h + 1) * KDA_DIM];
                l2norm(&mut qh);
                l2norm(&mut kh);
                for v in qh.iter_mut() {
                    *v *= scale;
                }
                let decay = exp(self.a_log[h]);
                let b = sigmoid(beta[t * KDA_HEADS + h]);
                let s = &mut st.s[h * KDA_DIM * KDA_DIM..(h + 1) * KDA_DIM * KDA_DIM];
                // the forget gate, bounded below: log-decay in (LOWER, 0) per key channel
                for kk in 0..KDA_DIM {
                    let g = LINEAR_GATE_LOWER_BOUND * sigmoid(decay * (ft[h * KDA_DIM + kk] + self.dt_bias[h * KDA_DIM + kk]));
                    let e = exp(g);
                    for vv in 0..KDA_DIM {
                        s[kk * KDA_DIM + vv] *= e;
                    }
                }
                let mut delta = [0f32; KDA_DIM];
                for vv in 0..KDA_DIM {
                    let mem: f32 = (0..KDA_DIM).map(|kk| s[kk * KDA_DIM + vv] * kh[kk]).sum();
                    delta[vv] = (vh[vv] - mem) * b;
                }
                for kk in 0..KDA_DIM {
                    for vv in 0..KDA_DIM {
                        s[kk * KDA_DIM + vv] += kh[kk] * delta[vv];
                    }
                }
                let mut o = [0f32; KDA_DIM];
                for vv in 0..KDA_DIM {
                    o[vv] = (0..KDA_DIM).map(|kk| s[kk * KDA_DIM + vv] * qh[kk]).sum();
                }
                // gated RMSNorm per head
                let on = rmsnorm(&o, KDA_DIM, &self.o_norm, RMS_NORM_EPS);
                for vv in 0..KDA_DIM {
                    normed[t * KDA_QKV + h * KDA_DIM + vv] = on[vv] * sigmoid(gt[h * KDA_DIM + vv]);
                }
            }
        }
        matmul(&self.o, H, KDA_QKV, &normed, n)
    }
}

/// The dense SwiGLU (`Glm5NextTextMLP`), clamped: gate above at the limit, up both ways.
pub struct Mlp {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
    pub ff: usize,
}

impl Mlp {
    pub fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        let g = matmul(&self.gate, self.ff, H, x, n);
        let u = matmul(&self.up, self.ff, H, x, n);
        let a: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| silu(g.min(SWIGLU_LIMIT)) * u.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)).collect(); // the same clamp as `clamped_swiglu`
        matmul(&self.down, H, self.ff, &a, n)
    }
}

/// NoPE MLA (`Glm5NextTextAttention`) over the keys the indexer selects. Below
/// `DENSE_BELOW` positions that is every earlier key, which is what this runs;
/// the indexer is its own piece.
pub struct Mla {
    pub q_a: Vec<f32>,
    pub q_a_ln: Vec<f32>,
    pub q_b: Vec<f32>,
    pub kv_a: Vec<f32>,
    pub kv_a_ln: Vec<f32>,
    /// `[MLA_HEADS][KVB_HEAD][KV_LORA_RANK]`: per head the key rows, then the value rows
    pub kv_b: Vec<f32>,
    pub o: Vec<f32>,
}

impl Mla {
    /// `x: [n][H]` at the positions after the `cache`'s; the cache (normed latents, one
    /// `KV_LORA_RANK` row a position) grows by `n`.
    pub fn forward(&self, x: &[f32], n: usize, cache: &mut Vec<f32>) -> Vec<f32> {
        let qr = rmsnorm(&matmul(&self.q_a, Q_LORA_RANK, H, x, n), Q_LORA_RANK, &self.q_a_ln, RMS_NORM_EPS);
        let q = matmul(&self.q_b, MLA_HEADS * Q_HEAD, Q_LORA_RANK, &qr, n);
        let past = cache.len() / KV_LORA_RANK;
        cache.extend(rmsnorm(&matmul(&self.kv_a, KV_LORA_RANK, H, x, n), KV_LORA_RANK, &self.kv_a_ln, RMS_NORM_EPS));
        let total = past + n;
        assert!(total <= DENSE_BELOW, "{total} positions: past {DENSE_BELOW} the indexer selects, and it is not transcribed yet");
        let kv = matmul(&self.kv_b, MLA_HEADS * KVB_HEAD, KV_LORA_RANK, cache, total);
        let scale = 1.0 / sqrt(Q_HEAD as f32);
        let mut out = vec![0f32; n * MLA_HEADS * V_HEAD_DIM];
        let mut w = vec![0f32; total];
        for t in 0..n {
            let p = past + t;
            for h in 0..MLA_HEADS {
                let qh = &q[(t * MLA_HEADS + h) * Q_HEAD..(t * MLA_HEADS + h + 1) * Q_HEAD];
                let key = |j: usize| &kv[(j * MLA_HEADS + h) * KVB_HEAD..(j * MLA_HEADS + h) * KVB_HEAD + QK_NOPE_HEAD_DIM];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..=p {
                    w[j] = dot(qh, key(j)) * scale;
                    mx = mx.max(w[j]);
                }
                let mut sum = 0f32;
                for wj in w[..=p].iter_mut() {
                    *wj = exp(*wj - mx);
                    sum += *wj;
                }
                let o = &mut out[(t * MLA_HEADS + h) * V_HEAD_DIM..(t * MLA_HEADS + h + 1) * V_HEAD_DIM];
                for j in 0..=p {
                    let a = w[j] / sum;
                    let v = &kv[(j * MLA_HEADS + h) * KVB_HEAD + QK_NOPE_HEAD_DIM..(j * MLA_HEADS + h + 1) * KVB_HEAD];
                    for (od, vd) in o.iter_mut().zip(v) {
                        *od += a * vd;
                    }
                }
            }
        }
        matmul(&self.o, H, MLA_HEADS * V_HEAD_DIM, &out, n)
    }
}

/// One routed expert, widened: `gate_up: [2·D_FF][H]` (gate rows then up rows), `down: [H][D_FF]`.
pub struct Expert {
    pub gate_up: Vec<f32>,
    pub down: Vec<f32>,
}

/// The MoE (`Glm5NextTextMoE`): a sigmoid router with a selection bias, the top
/// `TOP_K` of `E` weighted by their own sigmoid renormalized and scaled, plus
/// the shared expert. The experts come from the caller one at a time, so only
/// those a passage touches are ever widened.
pub struct Moe {
    pub router: Vec<f32>,
    pub bias: Vec<f32>,
    pub shared: Mlp,
}

/// The routing of `n` positions: `[n][TOP_K]` experts (ascending) and their weights.
pub struct Routing {
    pub idx: Vec<usize>,
    pub weight: Vec<f32>,
}

fn clamped_swiglu(gu: &[f32], ff: usize, out: &mut [f32]) {
    for i in 0..ff {
        out[i] = silu(gu[i].min(SWIGLU_LIMIT)) * gu[ff + i].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
    }
}

impl Moe {
    pub fn route(&self, x: &[f32], n: usize) -> Routing {
        let logits = matmul(&self.router, E, H, x, n);
        let mut r = Routing { idx: vec![0; n * TOP_K], weight: vec![0.0; n * TOP_K] };
        for t in 0..n {
            let s: Vec<f32> = logits[t * E..(t + 1) * E].iter().map(|&l| sigmoid(l)).collect();
            let mut order: Vec<usize> = (0..E).collect();
            order.sort_by(|&a, &b| (s[b] + self.bias[b]).total_cmp(&(s[a] + self.bias[a])));
            let mut pick = order[..TOP_K].to_vec();
            pick.sort();
            let sum: f32 = pick.iter().map(|&e| s[e]).sum::<f32>() + 1e-20;
            for (k, &e) in pick.iter().enumerate() {
                r.idx[t * TOP_K + k] = e;
                r.weight[t * TOP_K + k] = s[e] / sum * ROUTED_SCALING_FACTOR;
            }
        }
        r
    }

    /// → (routed, shared), each `[n][H]`; the routed sum adds a position's experts in
    /// ascending order, the reference's `index_add_` order. The touched experts are
    /// widened and run in parallel, one expert a thread at a time: each sees only the
    /// few positions routed to it, so the parallelism is across experts, not rows.
    pub fn forward(&self, x: &[f32], n: usize, r: &Routing, expert: &(dyn Fn(usize) -> Expert + Sync)) -> (Vec<f32>, Vec<f32>) {
        let mut touched: Vec<usize> = r.idx.clone();
        touched.sort();
        touched.dedup();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).min(touched.len().max(1));
        // per worker: (slot, weighted output) pairs
        let results: Vec<Vec<(usize, Vec<f32>)>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|_| {
                    sc.spawn(|| {
                        let mut mine = Vec::new();
                        let mut act = vec![0f32; D_FF];
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(&e) = touched.get(i) else { break };
                            let slots: Vec<usize> = (0..n * TOP_K).filter(|&i| r.idx[i] == e).collect();
                            let xs: Vec<f32> = slots.iter().flat_map(|&i| x[(i / TOP_K) * H..(i / TOP_K + 1) * H].iter().copied()).collect();
                            let w = expert(e);
                            let gu = matmul_serial(&w.gate_up, 2 * D_FF, H, &xs, slots.len());
                            let mut a = vec![0f32; slots.len() * D_FF];
                            for (m, row) in gu.chunks_exact(2 * D_FF).enumerate() {
                                clamped_swiglu(row, D_FF, &mut act);
                                a[m * D_FF..(m + 1) * D_FF].copy_from_slice(&act);
                            }
                            let y = matmul_serial(&w.down, H, D_FF, &a, slots.len());
                            for (m, &slot) in slots.iter().enumerate() {
                                mine.push((slot, y[m * H..(m + 1) * H].iter().map(|v| v * r.weight[slot]).collect()));
                            }
                        }
                        mine
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut each = vec![0f32; n * TOP_K * H];
        for (slot, y) in results.into_iter().flatten() {
            each[slot * H..(slot + 1) * H].copy_from_slice(&y);
        }
        let mut routed = vec![0f32; n * H];
        for t in 0..n {
            for k in 0..TOP_K {
                for d in 0..H {
                    routed[t * H + d] += each[(t * TOP_K + k) * H + d];
                }
            }
        }
        (routed, self.shared.forward(x, n))
    }
}

pub enum Attn {
    Kda(Kda),
    Mla(Mla),
}

pub enum Ffn {
    Dense(Mlp),
    Moe(Moe),
}

pub struct Layer {
    pub attn_hc: Hc,
    pub ffn_hc: Hc,
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub attn: Attn,
    pub ffn: Ffn,
}

/// What a layer carries from one position to the next.
pub enum LayerState {
    Kda(KdaState),
    /// the normed latents, `[positions][KV_LORA_RANK]`
    Mla(Vec<f32>),
}

impl Layer {
    pub fn state(&self) -> LayerState {
        match self.attn {
            Attn::Kda(_) => LayerState::Kda(KdaState::default()),
            Attn::Mla(_) => LayerState::Mla(Vec::new()),
        }
    }

    /// `streams: [n][HC][H]` in place; `edge(name, [n][H])` sees the oracle's edges;
    /// `expert(e)` widens routed expert `e` on demand, from any thread. Returns the routing of a MoE layer.
    pub fn forward(&self, streams: &mut [f32], n: usize, st: &mut LayerState, edge: &mut dyn FnMut(&str, &[f32]), expert: &(dyn Fn(usize) -> Expert + Sync)) -> Option<Routing> {
        let mut x = vec![0f32; n * H];
        let mut mixes = Vec::with_capacity(n);
        for t in 0..n {
            mixes.push(self.attn_hc.pre(&streams[t * HC * H..(t + 1) * HC * H], &mut x[t * H..(t + 1) * H]));
        }
        let x = rmsnorm(&x, H, &self.input_layernorm, RMS_NORM_EPS);
        edge("attn_in", &x);
        let y = match (&self.attn, st) {
            (Attn::Kda(a), LayerState::Kda(s)) => a.forward(&x, n, s),
            (Attn::Mla(a), LayerState::Mla(c)) => a.forward(&x, n, c),
            _ => unreachable!("state of another attention"),
        };
        edge("attn_out", &y);
        let mut out = vec![0f32; HC * H];
        for t in 0..n {
            mixes[t].post(&y[t * H..(t + 1) * H], &streams[t * HC * H..(t + 1) * HC * H], &mut out);
            streams[t * HC * H..(t + 1) * HC * H].copy_from_slice(&out);
        }

        let mut x = vec![0f32; n * H];
        mixes.clear();
        for t in 0..n {
            mixes.push(self.ffn_hc.pre(&streams[t * HC * H..(t + 1) * HC * H], &mut x[t * H..(t + 1) * H]));
        }
        let x = rmsnorm(&x, H, &self.post_attention_layernorm, RMS_NORM_EPS);
        edge("ffn_in", &x);
        let (y, routing) = match &self.ffn {
            Ffn::Dense(m) => (m.forward(&x, n), None),
            Ffn::Moe(m) => {
                let r = m.route(&x, n);
                let (routed, shared) = m.forward(&x, n, &r, expert);
                edge("routed", &routed);
                edge("shared", &shared);
                (routed.iter().zip(&shared).map(|(a, b)| a + b).collect(), Some(r))
            }
        };
        edge("ffn_out", &y);
        for t in 0..n {
            mixes[t].post(&y[t * H..(t + 1) * H], &streams[t * HC * H..(t + 1) * HC * H], &mut out);
            streams[t * HC * H..(t + 1) * HC * H].copy_from_slice(&out);
        }
        let mean: Vec<f32> = (0..n * H).map(|i| (0..HC).map(|j| streams[(i / H) * HC * H + j * H + i % H]).sum::<f32>() / HC as f32).collect();
        edge("out", &mean);
        routing
    }
}
