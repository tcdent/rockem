//! The audio encoder and the projector on the host, f32 over the checkpoint's bf16 values (widened once
//! at construction, 2.7 GB): one 30 s window of log-mel `[NUM_MEL_BINS][MEL_FRAMES]` → the window's
//! `ROWS_PER_WINDOW` rows `[H]` for the decoder. Whisper-large-v3's shape: two convolutions (the second
//! halving the frames) with GELU, sinusoidal positions, 32 pre-norm layers of bidirectional attention
//! (20 heads of 64, the query scaled by 1/8, k without bias) and a GELU FFN, a final norm; then four
//! consecutive positions concatenated and the projector's linear → GELU → linear (no biases). The matrix
//! products are a tiled f32 GEMM over the pool: a tile of output rows against every position, the rows
//! resident in cache while the positions stream.

use crate::weights::{widen, Dense};
use crate::*;
use common::math::{exp, sqrt};
use common::pool::Pool;

const T: usize = ENC_POSITIONS;
const LN_EPS: f32 = 1e-5;
/// Output rows a GEMM item takes.
const ROWS: usize = 16;
/// Positions a GEMM item runs at once (their inputs stay in L2 beside the item's rows).
const COLS: usize = 64;

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

/// exact GELU, `x · Φ(x)`, with erf to 1.5e-7 (Abramowitz–Stegun 7.1.26).
#[inline]
pub fn gelu(x: f32) -> f32 {
    let z = x * core::f32::consts::FRAC_1_SQRT_2;
    let s = if z < 0.0 { -1.0 } else { 1.0 };
    let a = if z < 0.0 { -z } else { z };
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    0.5 * x * (1.0 + s * y)
}

/// `out[t][m] = Σ_k x[t][k] · w[m][k] + b[m]` over `n` positions: items of `ROWS` rows × `COLS` positions on the pool.
fn gemm(pool: &dyn Pool, w: &[f32], b: Option<&[f32]>, m: usize, k: usize, x: &[f32], n: usize, out: &mut [f32]) {
    debug_assert!(w.len() == m * k && x.len() >= n * k && out.len() >= n * m && m % ROWS == 0);
    let (rt, ct) = (m / ROWS, n.div_ceil(COLS));
    let o = Out(out.as_mut_ptr());
    pool.run(rt * ct, &move |i| {
        let (r, c) = (i % rt, i / rt);
        let (m0, t0, t1) = (r * ROWS, c * COLS, (c * COLS + COLS).min(n));
        for t in t0..t1 {
            let xt = &x[t * k..(t + 1) * k];
            for j in 0..ROWS {
                let acc = common::linalg::dot(xt, &w[(m0 + j) * k..(m0 + j + 1) * k]) + b.map_or(0.0, |b| b[m0 + j]);
                unsafe { *o.at(t * m + m0 + j) = acc };
            }
        }
    });
}

fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / sqrt(var + LN_EPS);
    for (i, v) in x.iter_mut().enumerate() {
        *v = (*v - mean) * inv * w[i] + b[i];
    }
}

struct Layer {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    o_w: Vec<f32>,
    o_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
}

pub struct Encoder<'a> {
    pool: &'a dyn Pool,
    conv1_w: Vec<f32>, // [ENC_H][NUM_MEL_BINS·3]
    conv1_b: Vec<f32>,
    conv2_w: Vec<f32>, // [ENC_H][ENC_H·3]
    conv2_b: Vec<f32>,
    positions: &'a [f32], // [T][ENC_H]
    layers: Vec<Layer>,
    ln_w: Vec<f32>,
    ln_b: Vec<f32>,
    proj1: Vec<f32>, // [H][ENC_INTERMEDIATE_SIZE]
    proj2: Vec<f32>, // [H][H]
    // scratch
    h: Vec<f32>,   // [T][ENC_H]
    hn: Vec<f32>,  // [T][ENC_H]
    q: Vec<f32>,   // [T][ENC_H]
    k: Vec<f32>,
    v: Vec<f32>,
    a: Vec<f32>,   // [T][ENC_H] attention output
    ff: Vec<f32>,  // [T][ENC_INTERMEDIATE_SIZE]
    c1: Vec<f32>,  // [MEL_FRAMES][ENC_H] after conv1 (frame-major)
    /// Dev: the encoder's output `[T][ENC_H]` of the last window (before the projector), when tracing.
    pub trace: Option<Vec<f32>>,
}

impl<'a> Encoder<'a> {
    pub fn new(dense: Dense<'a>, pool: &'a dyn Pool) -> Self {
        let e = |i: usize| widen(dense.enc_head(i));
        let layers = (0..ENC_NUM_HIDDEN_LAYERS)
            .map(|l| {
                let w = |i: usize| widen(dense.enc_layer(l, i));
                Layer { ln1_w: w(0), ln1_b: w(1), q_w: w(2), q_b: w(3), k_w: w(4), v_w: w(5), v_b: w(6), o_w: w(7), o_b: w(8), ln2_w: w(9), ln2_b: w(10), fc1_w: w(11), fc1_b: w(12), fc2_w: w(13), fc2_b: w(14) }
            })
            .collect();
        Encoder {
            pool,
            conv1_w: e(0),
            conv1_b: e(1),
            conv2_w: e(2),
            conv2_b: e(3),
            positions: dense.enc_positions(),
            layers,
            ln_w: e(5),
            ln_b: e(6),
            proj1: widen(dense.projector(0)),
            proj2: widen(dense.projector(1)),
            h: vec![0.0; T * ENC_H],
            hn: vec![0.0; T * ENC_H],
            q: vec![0.0; T * ENC_H],
            k: vec![0.0; T * ENC_H],
            v: vec![0.0; T * ENC_H],
            a: vec![0.0; T * ENC_H],
            ff: vec![0.0; T * ENC_INTERMEDIATE_SIZE],
            c1: vec![0.0; MEL_FRAMES * ENC_H],
            trace: None,
        }
    }

    /// One window `[NUM_MEL_BINS][MEL_FRAMES]` → its `ROWS_PER_WINDOW` rows `[H]` into `rows`.
    pub fn encode(&mut self, mel: &[f32], rows: &mut [f32]) {
        assert_eq!(mel.len(), NUM_MEL_BINS * MEL_FRAMES);
        assert!(rows.len() >= ROWS_PER_WINDOW * H);
        // conv1 (k3, pad 1, stride 1): frame f's input is bins × taps {f−1, f, f+1} as `[bin][tap]`, the weight's order
        {
            let (w, b, pool) = (&self.conv1_w, &self.conv1_b, self.pool);
            let o = Out(self.c1.as_mut_ptr());
            pool.run(MEL_FRAMES.div_ceil(COLS), &move |c| {
                let mut xin = [0f32; NUM_MEL_BINS * 3];
                for f in c * COLS..((c + 1) * COLS).min(MEL_FRAMES) {
                    for bin in 0..NUM_MEL_BINS {
                        for tap in 0..3 {
                            let ff = f as isize + tap as isize - 1;
                            xin[bin * 3 + tap] = if ff < 0 || ff >= MEL_FRAMES as isize { 0.0 } else { mel[bin * MEL_FRAMES + ff as usize] };
                        }
                    }
                    for m in 0..ENC_H {
                        let acc = common::linalg::dot(&xin, &w[m * NUM_MEL_BINS * 3..(m + 1) * NUM_MEL_BINS * 3]) + b[m];
                        unsafe { *o.at(f * ENC_H + m) = gelu(acc) };
                    }
                }
            });
        }
        // conv2 (k3, pad 1, stride 2): position p takes frames {2p−1, 2p, 2p+1}, channel-major `[ch][tap]`; then GELU, then the positions
        {
            let (w, b, c1, positions, pool) = (&self.conv2_w, &self.conv2_b, &self.c1, self.positions, self.pool);
            let o = Out(self.h.as_mut_ptr());
            pool.run(T.div_ceil(COLS), &move |c| {
                let mut xin = vec![0f32; ENC_H * 3];
                for p in c * COLS..((c + 1) * COLS).min(T) {
                    for tap in 0..3 {
                        let f = 2 * p as isize + tap as isize - 1;
                        for ch in 0..ENC_H {
                            xin[ch * 3 + tap] = if f < 0 || f >= MEL_FRAMES as isize { 0.0 } else { c1[f as usize * ENC_H + ch] };
                        }
                    }
                    for m in 0..ENC_H {
                        let acc = common::linalg::dot(&xin, &w[m * ENC_H * 3..(m + 1) * ENC_H * 3]) + b[m];
                        unsafe { *o.at(p * ENC_H + m) = gelu(acc) + positions[p * ENC_H + m] };
                    }
                }
            });
        }
        for l in 0..ENC_NUM_HIDDEN_LAYERS {
            self.layer(l);
        }
        for t in 0..T {
            layer_norm(&mut self.h[t * ENC_H..(t + 1) * ENC_H], &self.ln_w, &self.ln_b);
        }
        if let Some(tr) = &mut self.trace {
            tr.clear();
            tr.extend_from_slice(&self.h);
        }
        // the projector over four consecutive positions concatenated: `h` is already `[T/4][4·ENC_H]` by layout
        gemm(self.pool, &self.proj1, None, H, ENC_INTERMEDIATE_SIZE, &self.h, ROWS_PER_WINDOW, &mut self.ff[..ROWS_PER_WINDOW * H]);
        for v in self.ff[..ROWS_PER_WINDOW * H].iter_mut() {
            *v = gelu(*v);
        }
        gemm(self.pool, &self.proj2, None, H, H, &self.ff[..ROWS_PER_WINDOW * H], ROWS_PER_WINDOW, &mut rows[..ROWS_PER_WINDOW * H]);
    }

    fn layer(&mut self, l: usize) {
        let ly = &self.layers[l];
        for t in 0..T {
            self.hn[t * ENC_H..(t + 1) * ENC_H].copy_from_slice(&self.h[t * ENC_H..(t + 1) * ENC_H]);
            layer_norm(&mut self.hn[t * ENC_H..(t + 1) * ENC_H], &ly.ln1_w, &ly.ln1_b);
        }
        gemm(self.pool, &ly.q_w, Some(&ly.q_b), ENC_H, ENC_H, &self.hn, T, &mut self.q);
        gemm(self.pool, &ly.k_w, None, ENC_H, ENC_H, &self.hn, T, &mut self.k);
        gemm(self.pool, &ly.v_w, Some(&ly.v_b), ENC_H, ENC_H, &self.hn, T, &mut self.v);
        // bidirectional attention, a (position, head) an item; the query scaled by 1/√d (the reference scales q before the dot)
        {
            let (q, k, v) = (&self.q, &self.k, &self.v);
            let o = Out(self.a.as_mut_ptr());
            let scale = 1.0 / sqrt(ENC_HEAD_DIM as f32);
            self.pool.run(T * ENC_HEADS, &move |i| {
                let (t, hd) = (i / ENC_HEADS, i % ENC_HEADS);
                let qv: [f32; ENC_HEAD_DIM] = core::array::from_fn(|j| q[t * ENC_H + hd * ENC_HEAD_DIM + j] * scale);
                let mut s = [0f32; T];
                let mut mx = f32::NEG_INFINITY;
                for p in 0..T {
                    let d = common::linalg::dot(&qv, &k[p * ENC_H + hd * ENC_HEAD_DIM..p * ENC_H + (hd + 1) * ENC_HEAD_DIM]);
                    s[p] = d;
                    if d > mx {
                        mx = d;
                    }
                }
                let mut sum = 0f32;
                for x in s.iter_mut() {
                    *x = exp(*x - mx);
                    sum += *x;
                }
                let inv = 1.0 / sum;
                let out = unsafe { core::slice::from_raw_parts_mut(o.at(t * ENC_H + hd * ENC_HEAD_DIM), ENC_HEAD_DIM) };
                out.fill(0.0);
                for p in 0..T {
                    common::linalg::axpy(out, s[p] * inv, &v[p * ENC_H + hd * ENC_HEAD_DIM..p * ENC_H + (hd + 1) * ENC_HEAD_DIM]);
                }
            });
        }
        gemm(self.pool, &ly.o_w, Some(&ly.o_b), ENC_H, ENC_H, &self.a, T, &mut self.hn);
        for i in 0..T * ENC_H {
            self.h[i] += self.hn[i];
        }
        for t in 0..T {
            self.hn[t * ENC_H..(t + 1) * ENC_H].copy_from_slice(&self.h[t * ENC_H..(t + 1) * ENC_H]);
            layer_norm(&mut self.hn[t * ENC_H..(t + 1) * ENC_H], &ly.ln2_w, &ly.ln2_b);
        }
        gemm(self.pool, &ly.fc1_w, Some(&ly.fc1_b), ENC_INTERMEDIATE_SIZE, ENC_H, &self.hn, T, &mut self.ff);
        for v in self.ff.iter_mut() {
            *v = gelu(*v);
        }
        gemm(self.pool, &ly.fc2_w, Some(&ly.fc2_b), ENC_H, ENC_INTERMEDIATE_SIZE, &self.ff, T, &mut self.hn);
        for i in 0..T * ENC_H {
            self.h[i] += self.hn[i];
        }
    }
}
