//! The clamped SwiGLU's backward: `h = silu(min(g, L)) ⊙ clamp(u, ±L)`; given `dh`
//! and the pre-activations, `dg` and `du`. The clamps pass the gradient on the closed
//! interval and stop it outside, as torch's `clamp` does.

use common::swiglu::silu;

/// d silu(x) / dx = σ(x) · (1 + x · (1 − σ(x))).
#[inline(always)]
pub fn dsilu(x: f32) -> f32 {
    let s = 1.0 / (1.0 + common::math::exp(-x));
    s * (1.0 + x * (1.0 - s))
}

/// `dg[i]`, `du[i]` from `dh[i]` for `i < n`; `dg` may alias `dh` (it is written after `dh[i]` is read).
#[inline(always)]
pub fn clamped_backward(n: usize, g: &[f32], u: &[f32], limit: f32, dh: &[f32], dg: &mut [f32], du: &mut [f32]) {
    for i in 0..n {
        let (gi, ui) = (g[i], u[i]);
        let gc = if gi > limit { limit } else { gi };
        let uc = if ui > limit { limit } else if ui < -limit { -limit } else { ui };
        let d = dh[i];
        du[i] = if ui > limit || ui < -limit { 0.0 } else { d * silu(gc) };
        dg[i] = if gi > limit { 0.0 } else { d * uc * dsilu(gc) };
    }
}
