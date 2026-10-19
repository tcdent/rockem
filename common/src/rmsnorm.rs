//! `y = w ⊙ x · rsqrt(mean(x²) + eps)`. A prologue step, inlined into
//! whatever projection follows it.

use super::math::rsqrt;

#[inline(always)]
pub fn rmsnorm<const N: usize>(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let mut ss = 0f32;
    for &v in &x[..N] {
        ss += v * v;
    }
    let r = rsqrt(ss / N as f32 + eps);
    for i in 0..N {
        out[i] = w[i] * (x[i] * r);
    }
}

/// `rmsnorm` with the squares summed in eight strided accumulators (lane `j` takes `x[8i + j]`), the eight
/// added in order at the end: the same expression on a CPU and on a card's eight lanes, so a norm both tiers
/// compute agrees bit for bit at the speed of eight chains (one chain of 4096 was 30 µs a token on gfx1100,
/// 2026-10-19). The routed experts' input norm uses it (`ffn_act`); the other norms keep `rmsnorm`.
#[inline(always)]
pub fn rmsnorm8<const N: usize>(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let r = rsqrt(sumsq8::<N>(x) / N as f32 + eps);
    for i in 0..N {
        out[i] = w[i] * (x[i] * r);
    }
}
/// The sum of squares of `x[..N]` in eight strided accumulators, added in order.
#[inline(always)]
pub fn sumsq8<const N: usize>(x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let mut i = 0;
    while i < N {
        for j in 0..8 {
            acc[j] += x[i + j] * x[i + j];
        }
        i += 8;
    }
    let mut ss = 0f32;
    for a in acc {
        ss += a;
    }
    ss
}
