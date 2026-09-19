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
