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

/// `rmsnorm` with the squares summed as a tree a card's workgroup computes in parallel (`sumsq_tree`):
/// the same expression on the CPU and on the card, so a norm both tiers compute agrees bit for bit. The
/// routed experts' input norm uses it (the FFN card's `hc_pre`, queue 001); the other norms keep `rmsnorm`.
#[inline(always)]
pub fn rmsnorm_tree<const N: usize>(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let r = rsqrt(sumsq_tree::<N>(x) / N as f32 + eps);
    for i in 0..N {
        out[i] = w[i] * (x[i] * r);
    }
}
/// The sum of squares of `x[..N]` (N = 256 · 16): p[j] = Σ_i x[j + 256 i]² in order, q[l] = Σ_k p[8 l + k]
/// in order, the result Σ_l q[l] in order — a workgroup's 256 threads, wave 0's 32 lanes, lane 0.
#[inline(always)]
pub fn sumsq_tree<const N: usize>(x: &[f32]) -> f32 {
    assert!(N == 256 * 16);
    let mut p = [0f32; 256];
    for (j, pj) in p.iter_mut().enumerate() {
        for i in 0..16 {
            let v = x[j + 256 * i];
            *pj += v * v;
        }
    }
    let mut ss = 0f32;
    for l in 0..32 {
        let mut q = 0f32;
        for k in 0..8 {
            q += p[8 * l + k];
        }
        ss += q;
    }
    ss
}
