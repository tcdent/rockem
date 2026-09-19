//! Rotary steps on one `D`-wide vector, in place. `cos`/`sin` are the
//! position's row of a table built at construction; scaling schemes (YaRN)
//! live in the table, never here.

/// `rotate_half` convention: `out[j] = x[j]·cos[j] − x[j+D/2]·sin[j]`,
/// `out[j+D/2] = x[j+D/2]·cos[j+D/2] + x[j]·sin[j+D/2]`.
#[inline(always)]
pub fn rotate_half<const D: usize>(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    const { assert!(D % 2 == 0) };
    let h = D / 2;
    for j in 0..h {
        let (a, b) = (x[j], x[j + h]);
        x[j] = a * cos[j] - b * sin[j];
        x[j + h] = b * cos[j + h] + a * sin[j + h];
    }
}

/// DeepSeek's variant: the vector is stored as interleaved pairs and is
/// first de-interleaved (evens, then odds) and left that way — the oracle
/// never permutes back, and q and k both go through it so their dot product
/// is unaffected. (`modeling_deepseek.py::apply_rotary_pos_emb`.)
#[inline(always)]
pub fn deinterleave_rotate_half<const D: usize>(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    const { assert!(D % 2 == 0 && D <= 256) };
    let h = D / 2;
    let mut t = [0f32; 256];
    for j in 0..h {
        t[j] = x[2 * j];
        t[j + h] = x[2 * j + 1];
    }
    x[..D].copy_from_slice(&t[..D]);
    rotate_half::<D>(x, cos, sin);
}
