//! The audio encoder and the projector on the host, f32 over the checkpoint's bf16 values (packed once at
//! construction): a request's log-mel windows → one row `[H]` per 100 ms for the decoder. The reference's
//! `GraniteSpeechCTCEncoder` and `GraniteSpeechEncoderProjector`, as `modeling_granite_speech.py` has them:
//!
//! - the frames: 80 log-mels at 10 ms, two stacked into `[T][160]` at 20 ms; `input_linear` → `[T][1024]`;
//! - 16 Conformer blocks: `h += ½·FF₁(h)`, `h += Attn(h)`, `h += Conv(h)`, `h += ½·FF₂(h)`, `h = LN_post(h)`. FF is
//!   `down(SiLU(up(LN(h))))` (the ½ folded into `down`); the attention is over blocks of 200 frames (4 s), each
//!   block on its own, with Shaw relative positions: `logit(c, r) = s·q_c·k_r + s·q_c·e_{c−r}` (`s = 1/√128` folded
//!   into q, `e` a table over distances clipped to ±200); the convolution module is `LN → up (1024 → 4096) →
//!   GLU → depthwise k15 → BatchNorm → SiLU → down (2048 → 1024)` (the batch norm folded into the depthwise
//!   taps and a bias);
//! - after block 8 the self-conditioned CTC: `h += out_mid(softmax(out(h)))` over the 348 CTC characters;
//! - the projector: every 15 frames a window, 3 learned queries through a 2-layer BLIP-2 Q-Former (post-LN at
//!   1e-12: self-attention over the 3, cross-attention to the window's 15 frames, a GELU FFN), then a linear to
//!   `[H]`: 3 rows a window, 10 a second.
//!
//! Every product is `common::sgemm` over the pool (items by column block to one core complex: `run_contiguous`);
//! the per-row work — norms, GLU, the depthwise taps, softmaxes — is rows over the pool. The wire's window is
//! `ROWS_PER_WINDOW` rows, but the encoder runs over all of a request's windows at once (its blocks and
//! convolutions span them): `encode` takes the request's whole log-mel.

use crate::weights::{bf, widen, Dense};
use crate::*;
use common::pool::Pool;
use common::sgemm::{self, Epilogue, Packed};
use std::cell::RefCell;

const LN_EPS: f32 = 1e-5;
const BN_EPS: f32 = 1e-5;
/// Rows a per-row item takes.
const ROWS: usize = sgemm::MB;
/// The attention block's keys padded to whole panels.
const KP: usize = ENC_CONTEXT.div_ceil(sgemm::NR) * sgemm::NR;
/// Relative distances a block spans (`−(C−1) ..= C−1`), padded to whole panels; row `m` of the packed table is
/// distance `m − (C − 1)`.
const REL: usize = 2 * ENC_CONTEXT - 1;
const RELP: usize = REL.div_ceil(sgemm::NR) * sgemm::NR;
/// The CTC characters padded to whole panels.
const CTC: usize = ENC_OUTPUT_DIM.div_ceil(sgemm::NR) * sgemm::NR;
const PAD: usize = ENC_CONV_KERNEL / 2;
const PROJ_QKV: usize = 3 * PROJ_H;
pub const PHASES: [&str; 7] = ["input", "ff", "attention", "conv", "norm", "ctc", "projector"];

#[derive(Clone, Copy)]
struct Out(*mut f32);
unsafe impl Sync for Out {}
unsafe impl Send for Out {}
impl Out {
    #[inline(always)]
    fn at(self, i: usize) -> *mut f32 {
        unsafe { self.0.add(i) }
    }
    #[inline(always)]
    unsafe fn row(self, i: usize, n: usize) -> &'static mut [f32] {
        core::slice::from_raw_parts_mut(self.0.add(i), n)
    }
}

/// `n` items over the pool in contiguous ranges, a core complex each.
fn run_contiguous(pool: &dyn Pool, n: usize, f: &(dyn Fn(usize) + Sync)) {
    let cx = pool.complexes();
    let of = |i: usize| cx[i * cx.len() / n];
    pool.run_on(n, &|i| of(i).0, &|i| of(i).1, f);
}

/// `out[t][m] = Σ_k x[t][k] · w[m][k] (+ b[m])` over `n` rows of `x` (row stride `ldx`), through the epilogue.
fn gemm(pool: &dyn Pool, w: &Packed, x: &[f32], ldx: usize, n: usize, out: *mut f32, ldo: usize, ep: Epilogue) {
    debug_assert!(x.len() >= (n - 1) * ldx + w.k);
    let (xp, o) = (x.as_ptr() as usize, Out(out));
    run_contiguous(pool, sgemm::items(w.m, n), &move |i| unsafe { sgemm::item(w, i, xp as *const f32, ldx, n, o.at(0), ldo, ep) });
}

/// `f(t)` for every row `t < n`, `ROWS` an item.
fn each_row(pool: &dyn Pool, n: usize, f: &(dyn Fn(usize) + Sync)) {
    pool.run(n.div_ceil(ROWS), &|i| {
        for t in i * ROWS..((i + 1) * ROWS).min(n) {
            f(t)
        }
    });
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / common::math::sqrt(var + eps);
    for i in 0..x.len() {
        out[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

/// `out[t] = LN(x[t])` over `n` rows of `width`.
fn norm_rows(pool: &dyn Pool, x: &[f32], width: usize, w: &[f32], b: &[f32], eps: f32, n: usize, out: *mut f32) {
    let o = Out(out);
    each_row(pool, n, &move |t| layer_norm(&x[t * width..(t + 1) * width], w, b, eps, unsafe { o.row(t * width, width) }));
}

struct Ln {
    w: Vec<f32>,
    b: Vec<f32>,
}
struct Lin {
    w: Vec<f32>, // packed
    b: Vec<f32>,
    m: usize,
    k: usize,
}
impl Lin {
    fn packed(&self) -> Packed<'_> {
        Packed { w: &self.w, m: self.m, k: self.k, bias: Some(&self.b) }
    }
}
struct Ff {
    ln: Ln,
    up: Lin,
    down: Lin, // × ½
}
struct Block {
    ff1: Ff,
    attn_ln: Ln,
    wq: Vec<f32>, // packed, × 1/√d
    wkv: Vec<f32>,
    wo: Lin,
    rel: Vec<f32>, // the distance table packed: `[RELP / NR][ENC_HEAD_DIM][NR]`, row m = distance m − (C−1)
    conv_ln: Ln,
    up_conv: Lin,
    depth: Vec<f32>, // `[K][ENC_CONV]` folded with the batch norm
    depth_b: Vec<f32>,
    down_conv: Lin,
    ff2: Ff,
    post_ln: Ln,
}
struct QLayer {
    qkv: Lin, // self-attention q|k|v fused, q × 1/8
    o: Lin,
    ln1: Ln,
    cq: Lin, // cross-attention query × 1/8
    ckv: Lin, // cross-attention k|v over the encoder frames
    co: Lin,
    ln2: Ln,
    up: Lin,
    down: Lin,
    ln3: Ln,
}

pub struct Encoder<'a> {
    pool: &'a dyn Pool,
    input: Lin,
    blocks: Vec<Block>,
    ctc_out: Lin, // padded to CTC rows
    ctc_mid: Lin,
    q0: Vec<f32>, // the queries after the Q-Former's input norm, `[QUERIES][PROJ_H]`
    qlayers: Vec<QLayer>,
    linear: Lin,
    // scratch, sized to the request
    xin: Vec<f32>,  // [T][160]
    h: Vec<f32>,    // [T][ENC_H] the residual
    hn: Vec<f32>,   // [T][ENC_H]
    ff: Vec<f32>,   // [T][ENC_FF]
    q: Vec<f32>,    // [T][ENC_H]
    kv: Vec<f32>,   // [T][2·ENC_H]
    a: Vec<f32>,    // [T][ENC_H]
    kp: Vec<f32>,   // [blocks][heads][KP/NR][ENC_HEAD_DIM][NR]
    vp: Vec<f32>,   // [blocks][heads][ENC_HEAD_DIM/NR][ENC_CONTEXT][NR]
    glu: Vec<f32>,  // [T][ENC_CONV]
    conv: Vec<f32>, // [T][ENC_CONV]
    ctc: Vec<f32>,  // [T][CTC]
    qh: Vec<f32>,   // [3W][PROJ_H]
    qs: Vec<f32>,   // [3W][3·PROJ_H]
    kvc: Vec<f32>,  // [T][2·PROJ_H]
    qc: Vec<f32>,   // [3W][PROJ_H]
    ctx: Vec<f32>,  // [3W][PROJ_H]
    pff: Vec<f32>,  // [3W][PROJ_INTERMEDIATE_SIZE]
    tmp: Vec<f32>,  // [3W][PROJ_H]
    /// Seconds in each of `PHASES` over the last `encode`.
    pub phases: [f64; 7],
    /// Dev: the encoder's output `[T][ENC_H]` of the last request (before the projector), when tracing.
    pub trace: Option<Vec<f32>>,
    /// Dev: every block's output (before the self-conditioned CTC after block 8, as the reference's hooks see it), when tracing.
    pub trace_blocks: Option<Vec<Vec<f32>>>,
}

const KP_HEAD: usize = KP * ENC_HEAD_DIM;
const VP_HEAD: usize = ENC_HEAD_DIM * ENC_CONTEXT;
thread_local! {
    /// An attention item's scores (`ENC_CONTEXT` rows of `KP`) and its distance terms (`ENC_CONTEXT` rows of `RELP`).
    static SCRATCH: RefCell<Vec<f32>> = RefCell::new(vec![0f32; ENC_CONTEXT * (KP + RELP)]);
}

fn phase(phases: &mut [f64; 7], i: usize, t: &mut std::time::Instant) {
    let now = std::time::Instant::now();
    phases[i] += (now - *t).as_secs_f64();
    *t = now;
}

impl<'a> Encoder<'a> {
    pub fn new(dense: Dense<'a>, pool: &'a dyn Pool) -> Self {
        let ln = |w: &[u16], b: &[u16]| Ln { w: widen(w), b: widen(b) };
        let lin = |w: &[u16], b: &[u16], m: usize, k: usize| Lin { w: sgemm::pack_bf16(w, m, k), b: widen(b), m, k };
        let lin_scaled = |w: &[u16], b: &[u16], m: usize, k: usize, s: f32| Lin { w: sgemm::pack_with(m, m, k, |j, kk| bf(w[j * k + kk]) * s), b: widen(b).iter().map(|v| v * s).collect(), m, k };
        let blocks = (0..ENC_NUM_HIDDEN_LAYERS)
            .map(|l| {
                let w = |i: usize| dense.enc_layer(l, i);
                let ff = |o: usize| Ff { ln: ln(w(o), w(o + 1)), up: lin(w(o + 2), w(o + 3), ENC_FF, ENC_H), down: lin_scaled(w(o + 4), w(o + 5), ENC_H, ENC_FF, 0.5) };
                let scale = 1.0 / common::math::sqrt(ENC_HEAD_DIM as f32);
                let wq = sgemm::pack_with(ENC_H, ENC_H, ENC_H, |j, kk| bf(w(8)[j * ENC_H + kk]) * scale);
                // the distance table: row m is the embedding of distance m − (C−1), i.e. table row (m − (C−1)) + MAX_POS_EMB
                let e = w(12);
                let rel = sgemm::pack_with(RELP, REL, ENC_HEAD_DIM, |m, d| bf(e[(m + ENC_MAX_POS_EMB + 1 - ENC_CONTEXT) * ENC_HEAD_DIM + d]));
                // the batch norm folded into the depthwise taps: w' = w · γ/σ, b' = β − μ · γ/σ
                let (dw, g, be, mu, var) = (w(17), w(18), w(19), w(20), w(21));
                let mut depth = vec![0f32; ENC_CONV_KERNEL * ENC_CONV];
                let mut depth_b = vec![0f32; ENC_CONV];
                for ch in 0..ENC_CONV {
                    let s = bf(g[ch]) / common::math::sqrt(bf(var[ch]) + BN_EPS);
                    depth_b[ch] = bf(be[ch]) - bf(mu[ch]) * s;
                    for k in 0..ENC_CONV_KERNEL {
                        depth[k * ENC_CONV + ch] = bf(dw[ch * ENC_CONV_KERNEL + k]) * s;
                    }
                }
                Block {
                    ff1: ff(0),
                    attn_ln: ln(w(6), w(7)),
                    wq,
                    wkv: sgemm::pack_bf16(w(9), 2 * ENC_H, ENC_H),
                    wo: lin(w(10), w(11), ENC_H, ENC_H),
                    rel,
                    conv_ln: ln(w(13), w(14)),
                    up_conv: lin(w(15), w(16), 2 * ENC_CONV, ENC_H),
                    depth,
                    depth_b,
                    down_conv: lin(w(22), w(23), ENC_H, ENC_CONV),
                    ff2: ff(24),
                    post_ln: ln(w(30), w(31)),
                }
            })
            .collect();
        let e = |i: usize| dense.enc_head(i);
        let (ow, ob) = (e(2), e(3));
        let ctc_out = Lin { w: sgemm::pack_with(CTC, ENC_OUTPUT_DIM, ENC_H, |j, kk| bf(ow[j * ENC_H + kk])), b: (0..CTC).map(|j| if j < ENC_OUTPUT_DIM { bf(ob[j]) } else { 0.0 }).collect(), m: CTC, k: ENC_H };
        let p = |i: usize| dense.proj_head(i);
        let mut q0 = vec![0f32; QUERIES * PROJ_H];
        let (qw, lw, lb) = (widen(p(0)), widen(p(1)), widen(p(2)));
        for i in 0..QUERIES {
            layer_norm(&qw[i * PROJ_H..(i + 1) * PROJ_H], &lw, &lb, PROJ_LN_EPS, &mut q0[i * PROJ_H..(i + 1) * PROJ_H]);
        }
        let qlayers = (0..PROJ_NUM_HIDDEN_LAYERS)
            .map(|l| {
                let w = |i: usize| dense.proj_layer(l, i);
                let s = 1.0 / common::math::sqrt(PROJ_HEAD_DIM as f32);
                let (q, k, v) = (w(0), w(2), w(4));
                let qkv = Lin {
                    w: sgemm::pack_with(PROJ_QKV, PROJ_QKV, PROJ_H, |j, kk| match j / PROJ_H {
                        0 => bf(q[j * PROJ_H + kk]) * s,
                        1 => bf(k[(j - PROJ_H) * PROJ_H + kk]),
                        _ => bf(v[(j - 2 * PROJ_H) * PROJ_H + kk]),
                    }),
                    b: [widen(w(1)).iter().map(|v| v * s).collect::<Vec<f32>>(), widen(w(3)), widen(w(5))].concat(),
                    m: PROJ_QKV,
                    k: PROJ_H,
                };
                let (ck, cv) = (w(12), w(14));
                let ckv = Lin {
                    w: sgemm::pack_with(2 * PROJ_H, 2 * PROJ_H, ENC_H, |j, kk| if j < PROJ_H { bf(ck[j * ENC_H + kk]) } else { bf(cv[(j - PROJ_H) * ENC_H + kk]) }),
                    b: [widen(w(13)), widen(w(15))].concat(),
                    m: 2 * PROJ_H,
                    k: ENC_H,
                };
                QLayer {
                    qkv,
                    o: lin(w(6), w(7), PROJ_H, PROJ_H),
                    ln1: ln(w(8), w(9)),
                    cq: lin_scaled(w(10), w(11), PROJ_H, PROJ_H, s),
                    ckv,
                    co: lin(w(16), w(17), PROJ_H, PROJ_H),
                    ln2: ln(w(18), w(19)),
                    up: lin(w(20), w(21), PROJ_INTERMEDIATE_SIZE, PROJ_H),
                    down: lin(w(22), w(23), PROJ_H, PROJ_INTERMEDIATE_SIZE),
                    ln3: ln(w(24), w(25)),
                }
            })
            .collect();
        Encoder {
            pool,
            input: lin(e(0), e(1), ENC_H, ENC_INPUT_DIM),
            blocks,
            ctc_out,
            ctc_mid: lin(e(4), e(5), ENC_H, ENC_OUTPUT_DIM),
            q0,
            qlayers,
            linear: lin(p(3), p(4), H, PROJ_H),
            xin: Vec::new(),
            h: Vec::new(),
            hn: Vec::new(),
            ff: Vec::new(),
            q: Vec::new(),
            kv: Vec::new(),
            a: Vec::new(),
            kp: Vec::new(),
            vp: Vec::new(),
            glu: Vec::new(),
            conv: Vec::new(),
            ctc: Vec::new(),
            qh: Vec::new(),
            qs: Vec::new(),
            kvc: Vec::new(),
            qc: Vec::new(),
            ctx: Vec::new(),
            pff: Vec::new(),
            tmp: Vec::new(),
            phases: [0.0; 7],
            trace: None,
            trace_blocks: None,
        }
    }

    fn reserve(&mut self, t: usize) {
        let blocks = t.div_ceil(ENC_CONTEXT);
        let w3 = t.div_ceil(WINDOW_SIZE) * QUERIES;
        let fit = |v: &mut Vec<f32>, n: usize| {
            if v.len() < n {
                v.resize(n, 0.0)
            }
        };
        fit(&mut self.xin, t * ENC_INPUT_DIM);
        fit(&mut self.h, t * ENC_H);
        fit(&mut self.hn, t * ENC_H);
        fit(&mut self.ff, t * ENC_FF);
        fit(&mut self.q, t * ENC_H);
        fit(&mut self.kv, t * 2 * ENC_H);
        fit(&mut self.a, t * ENC_H);
        fit(&mut self.kp, blocks * ENC_HEADS * KP_HEAD);
        fit(&mut self.vp, blocks * ENC_HEADS * VP_HEAD);
        fit(&mut self.glu, t * ENC_CONV);
        fit(&mut self.conv, t * ENC_CONV);
        fit(&mut self.ctc, t * CTC);
        fit(&mut self.qh, w3 * PROJ_H);
        fit(&mut self.qs, w3 * PROJ_QKV);
        fit(&mut self.kvc, t * 2 * PROJ_H);
        fit(&mut self.qc, w3 * PROJ_H);
        fit(&mut self.ctx, w3 * PROJ_H);
        fit(&mut self.pff, w3 * PROJ_INTERMEDIATE_SIZE);
        fit(&mut self.tmp, w3 * PROJ_H);
    }

    /// A request's log-mel, `[NUM_MEL_BINS][frames]` bin-major (its windows concatenated; `frames` even) → one row `[H]`
    /// per 15 encoder frames, the last projector window padded with zero hidden rows as the reference pads it:
    /// `ceil(frames / 2 / 15) · 3` rows, returned.
    pub fn encode(&mut self, mel: &[f32], frames: usize, rows: &mut [f32]) -> usize {
        assert!(frames >= 2 && frames % 2 == 0 && mel.len() == NUM_MEL_BINS * frames);
        let t = frames / 2;
        let windows = t.div_ceil(WINDOW_SIZE);
        let tp = windows * WINDOW_SIZE;
        let n_rows = windows * QUERIES;
        assert!(rows.len() >= n_rows * H);
        self.phases = [0.0; 7];
        let mut tm = std::time::Instant::now();
        self.reserve(tp);
        let pool = self.pool;
        // two 10 ms frames an encoder frame: frame 2f's bins, then frame 2f+1's
        {
            let o = Out(self.xin.as_mut_ptr());
            each_row(pool, t, &move |f| {
                let row = unsafe { o.row(f * ENC_INPUT_DIM, ENC_INPUT_DIM) };
                for bin in 0..NUM_MEL_BINS {
                    row[bin] = mel[bin * frames + 2 * f];
                    row[NUM_MEL_BINS + bin] = mel[bin * frames + 2 * f + 1];
                }
            });
        }
        gemm(pool, &self.input.packed(), &self.xin, ENC_INPUT_DIM, t, self.h.as_mut_ptr(), ENC_H, sgemm::STORE);
        phase(&mut self.phases, 0, &mut tm);
        for l in 0..ENC_NUM_HIDDEN_LAYERS {
            self.block(l, t, &mut tm);
            if let Some(tr) = &mut self.trace_blocks {
                if tr.len() <= l {
                    tr.push(Vec::new());
                }
                tr[l].clear();
                tr[l].extend_from_slice(&self.h[..t * ENC_H]);
            }
            if l + 1 == ENC_NUM_HIDDEN_LAYERS / 2 {
                self.ctc(t, &mut tm);
                // dev: the residual after the CTC term too, as entry ENC_NUM_HIDDEN_LAYERS of the block traces
                if let Some(tr) = &mut self.trace_blocks {
                    while tr.len() <= ENC_NUM_HIDDEN_LAYERS {
                        tr.push(Vec::new());
                    }
                    tr[ENC_NUM_HIDDEN_LAYERS].clear();
                    tr[ENC_NUM_HIDDEN_LAYERS].extend_from_slice(&self.h[..t * ENC_H]);
                }
            }
        }
        if let Some(tr) = &mut self.trace {
            tr.clear();
            tr.extend_from_slice(&self.h[..t * ENC_H]);
        }
        // the projector's last window padded with zero hidden rows
        self.h[t * ENC_H..tp * ENC_H].fill(0.0);
        self.projector(t, tp, rows, &mut tm);
        n_rows
    }

    fn ff(&mut self, f: &Ff, t: usize) {
        let pool = self.pool;
        norm_rows(pool, &self.h, ENC_H, &f.ln.w, &f.ln.b, LN_EPS, t, self.hn.as_mut_ptr());
        gemm(pool, &f.up.packed(), &self.hn, ENC_H, t, self.ff.as_mut_ptr(), ENC_FF, sgemm::SILU);
        gemm(pool, &f.down.packed(), &self.ff, ENC_FF, t, self.h.as_mut_ptr(), ENC_H, sgemm::ADD);
    }

    fn block(&mut self, l: usize, t: usize, tm: &mut std::time::Instant) {
        let pool = self.pool;
        let b = std::mem::replace(&mut self.blocks[l], Block::empty());
        self.ff(&b.ff1, t);
        phase(&mut self.phases, 1, tm);
        // attention: q (scaled) and k|v projections, then a (block, head) an item
        norm_rows(pool, &self.h, ENC_H, &b.attn_ln.w, &b.attn_ln.b, LN_EPS, t, self.hn.as_mut_ptr());
        gemm(pool, &Packed { w: &b.wq, m: ENC_H, k: ENC_H, bias: None }, &self.hn, ENC_H, t, self.q.as_mut_ptr(), ENC_H, sgemm::STORE);
        gemm(pool, &Packed { w: &b.wkv, m: 2 * ENC_H, k: ENC_H, bias: None }, &self.hn, ENC_H, t, self.kv.as_mut_ptr(), 2 * ENC_H, sgemm::STORE);
        let blocks = t.div_ceil(ENC_CONTEXT);
        {
            // the block's keys as panels of positions (k = the head dim), its values transposed as panels of the head
            // dim (k = the block's positions); the padding keys zero
            let (kv, kp, vp) = (&self.kv, Out(self.kp.as_mut_ptr()), Out(self.vp.as_mut_ptr()));
            pool.run(blocks * ENC_HEADS, &move |i| {
                let (blk, hd) = (i / ENC_HEADS, i % ENC_HEADS);
                let (p0, p1) = (blk * ENC_CONTEXT, ((blk + 1) * ENC_CONTEXT).min(t));
                let nb = p1 - p0;
                let kb = (blk * ENC_HEADS + hd) * KP_HEAD;
                let vb = (blk * ENC_HEADS + hd) * VP_HEAD;
                unsafe { kp.row(kb, KP_HEAD) }.fill(0.0);
                for p in 0..nb {
                    let k = &kv[(p0 + p) * 2 * ENC_H + hd * ENC_HEAD_DIM..][..ENC_HEAD_DIM];
                    let v = &kv[(p0 + p) * 2 * ENC_H + ENC_H + hd * ENC_HEAD_DIM..][..ENC_HEAD_DIM];
                    let ki = kb + (p / sgemm::NR) * ENC_HEAD_DIM * sgemm::NR + p % sgemm::NR;
                    for d in 0..ENC_HEAD_DIM {
                        unsafe {
                            *kp.at(ki + d * sgemm::NR) = k[d];
                            *vp.at(vb + (d / sgemm::NR) * nb * sgemm::NR + p * sgemm::NR + d % sgemm::NR) = v[d];
                        }
                    }
                }
            });
        }
        {
            let (q, kp, vp, rel) = (&self.q, &self.kp, &self.vp, &b.rel);
            let o = Out(self.a.as_mut_ptr());
            run_contiguous(pool, blocks * ENC_HEADS, &move |i| {
                let (blk, hd) = (i / ENC_HEADS, i % ENC_HEADS);
                let (p0, p1) = (blk * ENC_CONTEXT, ((blk + 1) * ENC_CONTEXT).min(t));
                let nb = p1 - p0;
                SCRATCH.with(|s| {
                    let mut s = s.borrow_mut();
                    let (sc, pe) = s.split_at_mut(ENC_CONTEXT * KP);
                    let qp = q[p0 * ENC_H + hd * ENC_HEAD_DIM..].as_ptr();
                    unsafe {
                        // scores c·r over the block's keys, and the distance terms c·e_m
                        sgemm::block(kp[(blk * ENC_HEADS + hd) * KP_HEAD..].as_ptr(), ENC_HEAD_DIM, 0..KP / sgemm::NR, qp, ENC_H, nb, sc.as_mut_ptr(), KP);
                        sgemm::block(rel.as_ptr(), ENC_HEAD_DIM, 0..RELP / sgemm::NR, qp, ENC_H, nb, pe.as_mut_ptr(), RELP);
                    }
                    for c in 0..nb {
                        let row = &mut sc[c * KP..(c + 1) * KP];
                        let dist = &pe[c * RELP..(c + 1) * RELP];
                        for r in 0..nb {
                            row[r] += dist[c + ENC_CONTEXT - 1 - r];
                        }
                        sgemm::softmax_row(row, nb);
                    }
                    unsafe {
                        sgemm::block(vp[(blk * ENC_HEADS + hd) * VP_HEAD..].as_ptr(), nb, 0..ENC_HEAD_DIM / sgemm::NR, sc.as_ptr(), KP, nb, o.at(p0 * ENC_H + hd * ENC_HEAD_DIM), ENC_H);
                    }
                });
            });
        }
        gemm(pool, &b.wo.packed(), &self.a, ENC_H, t, self.h.as_mut_ptr(), ENC_H, sgemm::ADD);
        phase(&mut self.phases, 2, tm);
        // the convolution module
        norm_rows(pool, &self.h, ENC_H, &b.conv_ln.w, &b.conv_ln.b, LN_EPS, t, self.hn.as_mut_ptr());
        gemm(pool, &b.up_conv.packed(), &self.hn, ENC_H, t, self.ff.as_mut_ptr(), 2 * ENC_CONV, sgemm::STORE);
        {
            let (ff, g) = (&self.ff, Out(self.glu.as_mut_ptr()));
            each_row(pool, t, &move |f| sgemm::glu(&ff[f * 2 * ENC_CONV..(f + 1) * 2 * ENC_CONV], unsafe { g.row(f * ENC_CONV, ENC_CONV) }));
        }
        {
            // depthwise over time: out[f] = b' + Σ_k w'[k] · glu[f + k − PAD], zero outside; then SiLU
            let (g, depth, depth_b, o) = (&self.glu, &b.depth, &b.depth_b, Out(self.conv.as_mut_ptr()));
            each_row(pool, t, &move |f| {
                let out = unsafe { o.row(f * ENC_CONV, ENC_CONV) };
                out.copy_from_slice(depth_b);
                for k in 0..ENC_CONV_KERNEL {
                    let src = f as isize + k as isize - PAD as isize;
                    if src >= 0 && (src as usize) < t {
                        sgemm::mul_add(out, &depth[k * ENC_CONV..(k + 1) * ENC_CONV], &g[src as usize * ENC_CONV..(src as usize + 1) * ENC_CONV]);
                    }
                }
                sgemm::silu(out);
            });
        }
        gemm(pool, &b.down_conv.packed(), &self.conv, ENC_CONV, t, self.h.as_mut_ptr(), ENC_H, sgemm::ADD);
        phase(&mut self.phases, 3, tm);
        self.ff(&b.ff2, t);
        phase(&mut self.phases, 1, tm);
        // the post norm, in place a row
        {
            let (h, w, bb) = (Out(self.h.as_mut_ptr()), &b.post_ln.w, &b.post_ln.b);
            each_row(pool, t, &move |f| {
                let row = unsafe { h.row(f * ENC_H, ENC_H) };
                let x: Vec<f32> = row.to_vec();
                layer_norm(&x, w, bb, LN_EPS, row);
            });
        }
        phase(&mut self.phases, 4, tm);
        self.blocks[l] = b;
    }

    /// The self-conditioned CTC: `h += out_mid(softmax(out(h)))`.
    fn ctc(&mut self, t: usize, tm: &mut std::time::Instant) {
        let pool = self.pool;
        gemm(pool, &self.ctc_out.packed(), &self.h, ENC_H, t, self.ctc.as_mut_ptr(), CTC, sgemm::STORE);
        {
            let c = Out(self.ctc.as_mut_ptr());
            each_row(pool, t, &move |f| sgemm::softmax_row(unsafe { c.row(f * CTC, CTC) }, ENC_OUTPUT_DIM));
        }
        gemm(pool, &self.ctc_mid.packed(), &self.ctc, CTC, t, self.h.as_mut_ptr(), ENC_H, sgemm::ADD);
        phase(&mut self.phases, 5, tm);
    }

    /// The Q-Former over every window of 15 frames, then the linear: 3 rows a window. Post-LN throughout: a
    /// sublayer's output is added to its input (`tmp = qh; tmp += f(qh)`) and normed back into `qh`.
    fn projector(&mut self, _t: usize, tp: usize, rows_out: &mut [f32], tm: &mut std::time::Instant) {
        let pool = self.pool;
        let windows = tp / WINDOW_SIZE;
        let w3 = windows * QUERIES;
        for w in 0..windows {
            self.qh[w * QUERIES * PROJ_H..(w + 1) * QUERIES * PROJ_H].copy_from_slice(&self.q0);
        }
        for l in 0..PROJ_NUM_HIDDEN_LAYERS {
            let ql = &self.qlayers[l];
            // self-attention over a window's queries (q scaled)
            gemm(pool, &ql.qkv.packed(), &self.qh, PROJ_H, w3, self.qs.as_mut_ptr(), PROJ_QKV, sgemm::STORE);
            {
                let (qs, o) = (&self.qs, Out(self.ctx.as_mut_ptr()));
                pool.run(windows, &move |w| {
                    for i in 0..QUERIES {
                        for hd in 0..PROJ_NUM_ATTENTION_HEADS {
                            let at = |j: usize, part: usize| &qs[(w * QUERIES + j) * PROJ_QKV + part * PROJ_H + hd * PROJ_HEAD_DIM..][..PROJ_HEAD_DIM];
                            let mut s = [0f32; QUERIES];
                            for j in 0..QUERIES {
                                s[j] = common::linalg::dot(at(i, 0), at(j, 1));
                            }
                            let out = unsafe { o.row((w * QUERIES + i) * PROJ_H + hd * PROJ_HEAD_DIM, PROJ_HEAD_DIM) };
                            softmax_small(&mut s);
                            out.fill(0.0);
                            for j in 0..QUERIES {
                                common::linalg::axpy(out, s[j], at(j, 2));
                            }
                        }
                    }
                });
            }
            self.tmp[..w3 * PROJ_H].copy_from_slice(&self.qh[..w3 * PROJ_H]);
            gemm(pool, &ql.o.packed(), &self.ctx, PROJ_H, w3, self.tmp.as_mut_ptr(), PROJ_H, sgemm::ADD);
            norm_rows(pool, &self.tmp, PROJ_H, &ql.ln1.w, &ql.ln1.b, PROJ_LN_EPS, w3, self.qh.as_mut_ptr());
            // cross-attention: the queries (scaled) against the window's 15 encoder frames
            gemm(pool, &ql.cq.packed(), &self.qh, PROJ_H, w3, self.qc.as_mut_ptr(), PROJ_H, sgemm::STORE);
            gemm(pool, &ql.ckv.packed(), &self.h, ENC_H, tp, self.kvc.as_mut_ptr(), 2 * PROJ_H, sgemm::STORE);
            {
                let (qc, kvc, o) = (&self.qc, &self.kvc, Out(self.ctx.as_mut_ptr()));
                pool.run(windows, &move |w| {
                    for i in 0..QUERIES {
                        for hd in 0..PROJ_NUM_ATTENTION_HEADS {
                            let q = &qc[(w * QUERIES + i) * PROJ_H + hd * PROJ_HEAD_DIM..][..PROJ_HEAD_DIM];
                            let k = |f: usize| &kvc[(w * WINDOW_SIZE + f) * 2 * PROJ_H + hd * PROJ_HEAD_DIM..][..PROJ_HEAD_DIM];
                            let v = |f: usize| &kvc[(w * WINDOW_SIZE + f) * 2 * PROJ_H + PROJ_H + hd * PROJ_HEAD_DIM..][..PROJ_HEAD_DIM];
                            let mut s = [0f32; WINDOW_SIZE];
                            for f in 0..WINDOW_SIZE {
                                s[f] = common::linalg::dot(q, k(f));
                            }
                            softmax_small(&mut s);
                            let out = unsafe { o.row((w * QUERIES + i) * PROJ_H + hd * PROJ_HEAD_DIM, PROJ_HEAD_DIM) };
                            out.fill(0.0);
                            for f in 0..WINDOW_SIZE {
                                common::linalg::axpy(out, s[f], v(f));
                            }
                        }
                    }
                });
            }
            self.tmp[..w3 * PROJ_H].copy_from_slice(&self.qh[..w3 * PROJ_H]);
            gemm(pool, &ql.co.packed(), &self.ctx, PROJ_H, w3, self.tmp.as_mut_ptr(), PROJ_H, sgemm::ADD);
            norm_rows(pool, &self.tmp, PROJ_H, &ql.ln2.w, &ql.ln2.b, PROJ_LN_EPS, w3, self.qh.as_mut_ptr());
            // the GELU feed-forward
            gemm(pool, &ql.up.packed(), &self.qh, PROJ_H, w3, self.pff.as_mut_ptr(), PROJ_INTERMEDIATE_SIZE, sgemm::GELU);
            self.tmp[..w3 * PROJ_H].copy_from_slice(&self.qh[..w3 * PROJ_H]);
            gemm(pool, &ql.down.packed(), &self.pff, PROJ_INTERMEDIATE_SIZE, w3, self.tmp.as_mut_ptr(), PROJ_H, sgemm::ADD);
            norm_rows(pool, &self.tmp, PROJ_H, &ql.ln3.w, &ql.ln3.b, PROJ_LN_EPS, w3, self.qh.as_mut_ptr());
        }
        gemm(pool, &self.linear.packed(), &self.qh, PROJ_H, w3, rows_out.as_mut_ptr(), H, sgemm::STORE);
        phase(&mut self.phases, 6, tm);
    }
}

/// Softmax of a few scores in place.
fn softmax_small(s: &mut [f32]) {
    let mx = s.iter().cloned().fold(f32::MIN, f32::max);
    let mut sum = 0.0;
    for v in s.iter_mut() {
        *v = common::math::exp(*v - mx);
        sum += *v;
    }
    for v in s.iter_mut() {
        *v /= sum;
    }
}

impl Block {
    fn empty() -> Self {
        let ln = || Ln { w: Vec::new(), b: Vec::new() };
        let lin = || Lin { w: Vec::new(), b: Vec::new(), m: 0, k: 0 };
        let ff = || Ff { ln: ln(), up: lin(), down: lin() };
        Block { ff1: ff(), attn_ln: ln(), wq: Vec::new(), wkv: Vec::new(), wo: lin(), rel: Vec::new(), conv_ln: ln(), up_conv: lin(), depth: Vec::new(), depth_b: Vec::new(), down_conv: lin(), ff2: ff(), post_ln: ln() }
    }
}
