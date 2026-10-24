//! The audio encoder and the projector on the host, f32 over the checkpoint's bf16 values (packed once at
//! construction, 2.7 GB): one 30 s window of log-mel `[NUM_MEL_BINS][MEL_FRAMES]` → the window's
//! `ROWS_PER_WINDOW` rows `[H]` for the decoder. Whisper-large-v3's shape: two convolutions (the second
//! halving the frames) with GELU, sinusoidal positions, 32 pre-norm layers of bidirectional attention
//! (20 heads of 64, the query scaled by 1/8, k without bias) and a GELU FFN, a final norm; then four
//! consecutive positions concatenated and the projector's linear → GELU → linear (no biases).
//!
//! Every product is `common::sgemm` over the pool: the projections (q|k|v fused, the 1/8 folded into q's rows
//! and bias — a power of two, so the bits are the reference's), the convolutions as GEMMs over the unrolled
//! input, the FFN with the GELU and the residual add in the epilogue; the attention a (head, block of
//! positions) an item — the scores against the head's keys packed as panels, a softmax a row, the values
//! against the head's values packed transposed. The norms and the packing are rows over the pool too: nothing
//! of a window runs on one thread.

use crate::weights::Dense;
use crate::*;
use common::pool::Pool;
use common::sgemm::{self, Epilogue, Packed};
use std::cell::RefCell;

const T: usize = ENC_POSITIONS;
const LN_EPS: f32 = 1e-5;
/// Positions a norm / pack / attention item takes.
const ROWS: usize = sgemm::MB;
/// The scores' row: the positions, padded to whole panels (the padding's keys are zero, its weights zeroed).
const KP: usize = T.div_ceil(sgemm::NR) * sgemm::NR;
const QKV_ROWS: usize = 3 * ENC_H;
const CONV1_K: usize = NUM_MEL_BINS * 3;
const CONV2_K: usize = ENC_H * 3;
pub const PHASES: [&str; 7] = ["conv", "norm+pack", "qkv", "attention", "o", "ffn", "projector"];

#[derive(Clone, Copy)]
struct Out(*mut f32);
unsafe impl Sync for Out {}
unsafe impl Send for Out {}
impl Out {
    #[inline(always)]
    fn at(self, i: usize) -> *mut f32 {
        unsafe { self.0.add(i) }
    }
}

/// exact GELU, `x · Φ(x)`, with erf to 1.5e-7 (Abramowitz–Stegun 7.1.26); `sgemm::gelu8` is this a lane.
#[inline]
pub fn gelu(x: f32) -> f32 {
    let z = x * core::f32::consts::FRAC_1_SQRT_2;
    let s = if z < 0.0 { -1.0 } else { 1.0 };
    let a = if z < 0.0 { -z } else { z };
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * common::math::exp(-a * a);
    0.5 * x * (1.0 + s * y)
}

/// `n` items over the pool in contiguous ranges, a core complex each: items that share an operand — a GEMM's
/// column block, an attention head's keys and values — land on cores that share an L3, and it is read into it once.
fn run_contiguous(pool: &dyn Pool, n: usize, f: &(dyn Fn(usize) + Sync)) {
    let cx = pool.complexes();
    let of = |i: usize| cx[i * cx.len() / n];
    pool.run_on(n, &|i| of(i).0, &|i| of(i).1, f);
}

/// `out[t][m] = Σ_k x[t][k] · w[m][k] (+ b[m])` over `n` positions, through the epilogue.
fn gemm(pool: &dyn Pool, w: &Packed, x: &[f32], n: usize, out: *mut f32, ep: Epilogue) {
    debug_assert!(x.len() >= n * w.k);
    let (xp, o) = (x.as_ptr() as usize, Out(out));
    run_contiguous(pool, sgemm::items(w.m, n), &move |i| unsafe { sgemm::item(w, i, xp as *const f32, w.k, n, o.at(0), w.m, ep) });
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / common::math::sqrt(var + LN_EPS);
    for i in 0..x.len() {
        out[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

/// `out[t] = LN(x[t])` over `T` rows, `ROWS` an item.
fn norm_rows(pool: &dyn Pool, x: &[f32], w: &[f32], b: &[f32], out: &mut [f32]) {
    let o = Out(out.as_mut_ptr());
    pool.run(T.div_ceil(ROWS), &move |i| {
        for t in i * ROWS..((i + 1) * ROWS).min(T) {
            let row = unsafe { core::slice::from_raw_parts_mut(o.at(t * ENC_H), ENC_H) };
            layer_norm(&x[t * ENC_H..(t + 1) * ENC_H], w, b, row);
        }
    });
}

struct Layer {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    qkv: Vec<f32>,   // packed [3·ENC_H][ENC_H]: q's rows (× 1/8), k's, v's
    qkv_b: Vec<f32>, // [3·ENC_H]: q's bias (× 1/8), zeros for k, v's
    o: Vec<f32>,
    o_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    fc1: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2: Vec<f32>,
    fc2_b: Vec<f32>,
}

pub struct Encoder<'a> {
    pool: &'a dyn Pool,
    conv1: Vec<f32>, // packed [ENC_H][NUM_MEL_BINS·3]
    conv1_b: Vec<f32>,
    conv2: Vec<f32>, // packed [ENC_H][ENC_H·3]
    conv2_b: Vec<f32>,
    positions: &'a [f32], // [T][ENC_H]
    layers: Vec<Layer>,
    ln_w: Vec<f32>,
    ln_b: Vec<f32>,
    proj1: Vec<f32>, // packed [H][ENC_INTERMEDIATE_SIZE]
    proj2: Vec<f32>, // packed [H][H]
    // scratch
    h: Vec<f32>,    // [T][ENC_H] the residual
    hn: Vec<f32>,   // [T][ENC_H] its norm
    qkv: Vec<f32>,  // [T][3·ENC_H]
    kp: Vec<f32>,   // [ENC_HEADS] × the head's keys as panels [KP/NR][ENC_HEAD_DIM][NR]
    vp: Vec<f32>,   // [ENC_HEADS] × the head's values transposed as panels [ENC_HEAD_DIM/NR][T][NR]
    a: Vec<f32>,    // [T][ENC_H] attention output
    ff: Vec<f32>,   // [T][ENC_INTERMEDIATE_SIZE]
    c1in: Vec<f32>, // [MEL_FRAMES][NUM_MEL_BINS·3] conv1's unrolled input
    c1: Vec<f32>,   // [MEL_FRAMES][ENC_H] after conv1 (frame-major)
    c2in: Vec<f32>, // [T][ENC_H·3] conv2's unrolled input
    /// Seconds in each of `PHASES` over the last `encode`.
    pub phases: [f64; 7],
    /// Dev: the encoder's output `[T][ENC_H]` of the last window (before the projector), when tracing.
    pub trace: Option<Vec<f32>>,
}

fn phase(phases: &mut [f64; 7], i: usize, t: &mut std::time::Instant) {
    let now = std::time::Instant::now();
    phases[i] += (now - *t).as_secs_f64();
    *t = now;
}

const KP_HEAD: usize = KP * ENC_HEAD_DIM;
const VP_HEAD: usize = ENC_HEAD_DIM * T;

thread_local! {
    /// An attention item's scores, `ROWS` rows of `KP`.
    static SCORES: RefCell<Vec<f32>> = RefCell::new(vec![0f32; ROWS * KP]);
}

impl<'a> Encoder<'a> {
    pub fn new(dense: Dense<'a>, pool: &'a dyn Pool) -> Self {
        let bf = |v: u16| f32::from_bits((v as u32) << 16);
        let widen = |w: &[u16]| w.iter().map(|&v| bf(v)).collect::<Vec<f32>>();
        let e = |i: usize| dense.enc_head(i);
        let scale = 1.0 / common::math::sqrt(ENC_HEAD_DIM as f32);
        let layers = (0..ENC_NUM_HIDDEN_LAYERS)
            .map(|l| {
                let w = |i: usize| dense.enc_layer(l, i);
                let (q, k, v) = (w(2), w(4), w(5));
                let qkv = sgemm::pack_with(QKV_ROWS, QKV_ROWS, ENC_H, |j, kk| match j / ENC_H {
                    0 => bf(q[j * ENC_H + kk]) * scale,
                    1 => bf(k[(j - ENC_H) * ENC_H + kk]),
                    _ => bf(v[(j - 2 * ENC_H) * ENC_H + kk]),
                });
                let mut qkv_b = vec![0f32; QKV_ROWS];
                for (i, &b) in w(3).iter().enumerate() {
                    qkv_b[i] = bf(b) * scale;
                }
                for (i, &b) in w(6).iter().enumerate() {
                    qkv_b[2 * ENC_H + i] = bf(b);
                }
                Layer {
                    ln1_w: widen(w(0)),
                    ln1_b: widen(w(1)),
                    qkv,
                    qkv_b,
                    o: sgemm::pack_bf16(w(7), ENC_H, ENC_H),
                    o_b: widen(w(8)),
                    ln2_w: widen(w(9)),
                    ln2_b: widen(w(10)),
                    fc1: sgemm::pack_bf16(w(11), ENC_INTERMEDIATE_SIZE, ENC_H),
                    fc1_b: widen(w(12)),
                    fc2: sgemm::pack_bf16(w(13), ENC_H, ENC_INTERMEDIATE_SIZE),
                    fc2_b: widen(w(14)),
                }
            })
            .collect();
        Encoder {
            pool,
            conv1: sgemm::pack_bf16(e(0), ENC_H, CONV1_K),
            conv1_b: widen(e(1)),
            conv2: sgemm::pack_bf16(e(2), ENC_H, CONV2_K),
            conv2_b: widen(e(3)),
            positions: dense.enc_positions(),
            layers,
            ln_w: widen(e(5)),
            ln_b: widen(e(6)),
            proj1: sgemm::pack_bf16(dense.projector(0), H, ENC_INTERMEDIATE_SIZE),
            proj2: sgemm::pack_bf16(dense.projector(1), H, H),
            h: vec![0.0; T * ENC_H],
            hn: vec![0.0; T * ENC_H],
            qkv: vec![0.0; T * QKV_ROWS],
            kp: vec![0.0; ENC_HEADS * KP_HEAD],
            vp: vec![0.0; ENC_HEADS * VP_HEAD],
            a: vec![0.0; T * ENC_H],
            ff: vec![0.0; T * ENC_INTERMEDIATE_SIZE],
            c1in: vec![0.0; MEL_FRAMES * CONV1_K],
            c1: vec![0.0; MEL_FRAMES * ENC_H],
            c2in: vec![0.0; T * CONV2_K],
            phases: [0.0; 7],
            trace: None,
        }
    }


    /// One window `[NUM_MEL_BINS][MEL_FRAMES]` → its `ROWS_PER_WINDOW` rows `[H]` into `rows`.
    pub fn encode(&mut self, mel: &[f32], rows: &mut [f32]) {
        assert_eq!(mel.len(), NUM_MEL_BINS * MEL_FRAMES);
        assert!(rows.len() >= ROWS_PER_WINDOW * H);
        self.phases = [0.0; 7];
        let mut t = std::time::Instant::now();
        let pool = self.pool;
        // conv1 (k3, pad 1, stride 1): frame f's input is bins × taps {f−1, f, f+1} as `[bin][tap]`, the weight's order
        {
            let o = Out(self.c1in.as_mut_ptr());
            pool.run(MEL_FRAMES.div_ceil(ROWS), &move |c| {
                for f in c * ROWS..((c + 1) * ROWS).min(MEL_FRAMES) {
                    for bin in 0..NUM_MEL_BINS {
                        for tap in 0..3 {
                            let ff = f as isize + tap as isize - 1;
                            unsafe { *o.at(f * CONV1_K + bin * 3 + tap) = if ff < 0 || ff >= MEL_FRAMES as isize { 0.0 } else { mel[bin * MEL_FRAMES + ff as usize] } };
                        }
                    }
                }
            });
            let w = Packed { w: &self.conv1, m: ENC_H, k: CONV1_K, bias: Some(&self.conv1_b) };
            gemm(pool, &w, &self.c1in, MEL_FRAMES, self.c1.as_mut_ptr(), sgemm::GELU);
        }
        // conv2 (k3, pad 1, stride 2): position p takes frames {2p−1, 2p, 2p+1}, channel-major `[ch][tap]`; GELU, then the positions
        {
            let (c1, o) = (&self.c1, Out(self.c2in.as_mut_ptr()));
            pool.run(T.div_ceil(ROWS), &move |c| {
                for p in c * ROWS..((c + 1) * ROWS).min(T) {
                    for tap in 0..3 {
                        let f = 2 * p as isize + tap as isize - 1;
                        for ch in 0..ENC_H {
                            unsafe { *o.at(p * CONV2_K + ch * 3 + tap) = if f < 0 || f >= MEL_FRAMES as isize { 0.0 } else { c1[f as usize * ENC_H + ch] } };
                        }
                    }
                }
            });
            self.h.copy_from_slice(self.positions);
            let w = Packed { w: &self.conv2, m: ENC_H, k: CONV2_K, bias: Some(&self.conv2_b) };
            gemm(pool, &w, &self.c2in, T, self.h.as_mut_ptr(), sgemm::GELU_ADD);
        }
        phase(&mut self.phases, 0, &mut t);
        for l in 0..ENC_NUM_HIDDEN_LAYERS {
            self.layer(l, &mut t);
        }
        norm_rows(pool, &self.h, &self.ln_w, &self.ln_b, &mut self.hn);
        phase(&mut self.phases, 1, &mut t);
        if let Some(tr) = &mut self.trace {
            tr.clear();
            tr.extend_from_slice(&self.hn);
        }
        // the projector over four consecutive positions concatenated: `hn` is already `[T/4][4·ENC_H]` by layout
        let w = Packed { w: &self.proj1, m: H, k: ENC_INTERMEDIATE_SIZE, bias: None };
        gemm(pool, &w, &self.hn, ROWS_PER_WINDOW, self.ff.as_mut_ptr(), sgemm::GELU);
        let w = Packed { w: &self.proj2, m: H, k: H, bias: None };
        gemm(pool, &w, &self.ff[..ROWS_PER_WINDOW * H], ROWS_PER_WINDOW, rows.as_mut_ptr(), sgemm::STORE);
        phase(&mut self.phases, 6, &mut t);
    }

    fn layer(&mut self, l: usize, t: &mut std::time::Instant) {
        let pool = self.pool;
        let ly = &self.layers[l];
        norm_rows(pool, &self.h, &ly.ln1_w, &ly.ln1_b, &mut self.hn);
        phase(&mut self.phases, 1, t);
        let w = Packed { w: &ly.qkv, m: QKV_ROWS, k: ENC_H, bias: Some(&ly.qkv_b) };
        gemm(pool, &w, &self.hn, T, self.qkv.as_mut_ptr(), sgemm::STORE);
        phase(&mut self.phases, 2, t);
        // a head's keys as panels of positions (k = the head dim), its values transposed as panels of the head dim (k =
        // the positions): an item a head and a block of positions
        {
            let (qkv, kp, vp) = (&self.qkv, Out(self.kp.as_mut_ptr()), Out(self.vp.as_mut_ptr()));
            let blocks = T.div_ceil(ROWS);
            pool.run(ENC_HEADS * blocks, &move |i| {
                let (hd, b) = (i / blocks, i % blocks);
                for p in b * ROWS..((b + 1) * ROWS).min(T) {
                    let k = &qkv[p * QKV_ROWS + ENC_H + hd * ENC_HEAD_DIM..][..ENC_HEAD_DIM];
                    let v = &qkv[p * QKV_ROWS + 2 * ENC_H + hd * ENC_HEAD_DIM..][..ENC_HEAD_DIM];
                    let kb = hd * KP_HEAD + (p / sgemm::NR) * ENC_HEAD_DIM * sgemm::NR + p % sgemm::NR;
                    let vb = hd * VP_HEAD + p * sgemm::NR;
                    for j in 0..ENC_HEAD_DIM {
                        unsafe {
                            *kp.at(kb + j * sgemm::NR) = k[j];
                            *vp.at(vb + (j / sgemm::NR) * T * sgemm::NR + j % sgemm::NR) = v[j];
                        }
                    }
                }
            });
        }
        phase(&mut self.phases, 1, t);
        // bidirectional attention: scores = q · Kᵀ (q already scaled), softmax a row, · V
        {
            let (qkv, kp, vp) = (&self.qkv, &self.kp, &self.vp);
            let o = Out(self.a.as_mut_ptr());
            let blocks = T.div_ceil(ROWS);
            run_contiguous(pool, ENC_HEADS * blocks, &move |i| {
                let (hd, b) = (i / blocks, i % blocks);
                let (t0, t1) = (b * ROWS, ((b + 1) * ROWS).min(T));
                let n = t1 - t0;
                SCORES.with(|s| {
                    let mut s = s.borrow_mut();
                    unsafe {
                        sgemm::block(kp[hd * KP_HEAD..].as_ptr(), ENC_HEAD_DIM, 0..KP / sgemm::NR, qkv[t0 * QKV_ROWS + hd * ENC_HEAD_DIM..].as_ptr(), QKV_ROWS, n, s.as_mut_ptr(), KP);
                    }
                    for r in 0..n {
                        sgemm::softmax_row(&mut s[r * KP..(r + 1) * KP], T);
                    }
                    unsafe {
                        sgemm::block(vp[hd * VP_HEAD..].as_ptr(), T, 0..ENC_HEAD_DIM / sgemm::NR, s.as_ptr(), KP, n, o.at(t0 * ENC_H + hd * ENC_HEAD_DIM), ENC_H);
                    }
                });
            });
        }
        phase(&mut self.phases, 3, t);
        // o projection into the residual
        let w = Packed { w: &ly.o, m: ENC_H, k: ENC_H, bias: Some(&ly.o_b) };
        gemm(pool, &w, &self.a, T, self.h.as_mut_ptr(), sgemm::ADD);
        phase(&mut self.phases, 4, t);
        norm_rows(pool, &self.h, &ly.ln2_w, &ly.ln2_b, &mut self.hn);
        phase(&mut self.phases, 1, t);
        let w = Packed { w: &ly.fc1, m: ENC_INTERMEDIATE_SIZE, k: ENC_H, bias: Some(&ly.fc1_b) };
        gemm(pool, &w, &self.hn, T, self.ff.as_mut_ptr(), sgemm::GELU);
        let w = Packed { w: &ly.fc2, m: ENC_H, k: ENC_INTERMEDIATE_SIZE, bias: Some(&ly.fc2_b) };
        gemm(pool, &w, &self.ff, T, self.h.as_mut_ptr(), sgemm::ADD);
        phase(&mut self.phases, 5, t);
    }
}
