//! Test-only helpers shared by the layer tests: random weights in the bundle's formats with
//! their dense f32 twins, and the comparison every kernel-vs-reference test makes.

use common::gemv::{e4m3, widen, FP8_BLOCK};
use common::testing::Rng;

/// Random FP8 rows (never the NaN codes) with block scales, and the dense twin.
pub struct Fp8Buf {
    pub w: Vec<u8>,
    pub s: Vec<f32>,
    pub dense: Vec<f32>,
}
pub fn fp8_record(m: usize, k: usize, scale: f32, rnd: &mut Rng) -> Fp8Buf {
    let w: Vec<u8> = (0..m * k).map(|_| { let b = (rnd.next() % 256) as u8; if b & 0x7f == 0x7f { b ^ 1 } else { b } }).collect();
    let kb = k / FP8_BLOCK;
    let s: Vec<f32> = (0..m.div_ceil(FP8_BLOCK) * kb).map(|_| scale * (0.5 + (rnd.next() % 100) as f32 * 0.01)).collect();
    let dense: Vec<f32> = (0..m * k).map(|i| e4m3(w[i]) * s[(i / k / FP8_BLOCK) * kb + (i % k) / FP8_BLOCK]).collect();
    Fp8Buf { w, s, dense }
}
impl Fp8Buf {
    pub fn view(&self) -> glm_5_3_flash::weights::Fp8<'_> {
        glm_5_3_flash::weights::Fp8 { w: &self.w, s: &self.s }
    }
}

/// Random bf16 values in `±scale`.
pub fn bf(m: usize, scale: f32, rnd: &mut Rng) -> Vec<u16> {
    (0..m).map(|_| ((rnd.unit() * scale).to_bits() >> 16) as u16).collect()
}
pub fn wide(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| widen(b)).collect()
}
pub fn unit(m: usize, scale: f32, rnd: &mut Rng) -> Vec<f32> {
    (0..m).map(|_| rnd.unit() * scale).collect()
}

/// `a` within `tol` of `b`, relative to `b`'s largest magnitude.
pub fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    assert!(worst <= tol * scale, "{what}: worst |Δ| {worst} against a scale of {scale}");
}
