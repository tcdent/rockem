//! `h = silu(gate) ⊙ up` over the two halves of one `2·N` buffer — the
//! epilogue of a fused gate/up projection.

use super::math::exp;

#[inline(always)]
pub fn silu(g: f32) -> f32 {
    g / (1.0 + exp(-g))
}

#[inline(always)]
pub fn swiglu<const N: usize>(gate_up: &[f32], out: &mut [f32]) {
    let (gate, up) = gate_up[..2 * N].split_at(N);
    for i in 0..N {
        out[i] = silu(gate[i]) * up[i];
    }
}

/// The same with the clamp some models train with: the gate capped above at
/// `limit`, the up projection clamped to `[-limit, limit]` (GLM-5.3-Flash's `swiglu_limit`).
#[inline(always)]
pub fn swiglu_clamped<const N: usize>(gate_up: &[f32], limit: f32, out: &mut [f32]) {
    let (gate, up) = gate_up[..2 * N].split_at(N);
    for i in 0..N {
        let g = if gate[i] > limit { limit } else { gate[i] };
        let u = if up[i] > limit { limit } else if up[i] < -limit { -limit } else { up[i] };
        out[i] = silu(g) * u;
    }
}
