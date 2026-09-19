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
