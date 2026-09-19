//! Softmax in place, greedy top-k, argmax, and the log-probability of one id.

use super::math::{exp, ln};

/// ln softmax(x)[id]: `x[id] − max − ln Σ exp(x − max)`, the sum in f64 so a
/// 100k-wide tail is not lost. The card's `score` kernel is this op's twin
/// (256 f32 partials joined in f64, so the two agree to ~1e-6, not to the bit).
#[inline(always)]
pub fn log_prob(x: &[f32], id: usize) -> f32 {
    let mut m = f32::NEG_INFINITY;
    for &v in x {
        if v > m {
            m = v;
        }
    }
    let mut s = 0f64;
    for &v in x {
        s += exp(v - m) as f64;
    }
    x[id] - m - ln(s as f32)
}

#[inline(always)]
pub fn softmax(x: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > m {
            m = v;
        }
    }
    let mut s = 0f32;
    for v in x.iter_mut() {
        *v = exp(*v - m);
        s += *v;
    }
    let inv = 1.0 / s;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// The K largest by repeated argmax; ties resolve to the lowest index.
#[inline(always)]
pub fn topk<const K: usize>(x: &[f32], idx: &mut [usize; K], val: &mut [f32; K]) {
    let mut taken = [false; 256];
    assert!(x.len() <= 256);
    for k in 0..K {
        let mut best = usize::MAX;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in x.iter().enumerate() {
            if !taken[i] && v > bv {
                bv = v;
                best = i;
            }
        }
        taken[best] = true;
        idx[k] = best;
        val[k] = bv;
    }
}

#[inline(always)]
pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best
}
