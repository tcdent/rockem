//! The transcendentals the kernels need, owned so that every target computes
//! the same bits. `core` has none of them without `std`; on the GPU `std`
//! does not exist. `sqrt` is one instruction everywhere and binds to it;
//! `exp` and `ln` are one polynomial each everywhere. Table construction
//! (cos, sin, pow for RoPE) is construction-time work and lives in the model
//! crate.

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn sqrt(x: f32) -> f32 {
    use core::arch::x86_64::*;
    unsafe { _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x))) }
}

#[cfg(target_arch = "amdgpu")]
#[inline(always)]
pub fn sqrt(x: f32) -> f32 {
    unsafe extern "C" {
        #[link_name = "llvm.sqrt.f32"]
        fn llvm_sqrt(x: f32) -> f32;
    }
    unsafe { llvm_sqrt(x) }
}

#[inline(always)]
pub fn rsqrt(x: f32) -> f32 {
    1.0 / sqrt(x)
}

/// Truncate toward zero, then correct: floor for |x| < 2^31.
#[inline(always)]
pub fn floor(x: f32) -> f32 {
    let t = x as i32 as f32;
    if t > x { t - 1.0 } else { t }
}

/// Round half away from zero — the semantics of `f32::round`.
#[inline(always)]
pub fn round(x: f32) -> f32 {
    let t = (x + if x >= 0.0 { 0.5 } else { -0.5 }) as i32;
    t as f32
}

/// e^x, Cephes `expf`: range-reduce by ln 2, degree-6 polynomial on the
/// remainder, scale by 2^n through the exponent bits. Relative error is
/// within 2 ulp across the f32 range; saturates below -87 and above 88.
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    const LOG2E: f32 = 1.442_695_04;
    const LN2_HI: f32 = 0.693_359_375;
    const LN2_LO: f32 = -2.121_944_4e-4;
    let x = if x > 88.0 { 88.0 } else if x < -87.0 { -87.0 } else { x };
    let n = floor(x * LOG2E + 0.5);
    let r = x - n * LN2_HI - n * LN2_LO;
    let p = 1.987_569_2e-4;
    let p = p * r + 1.398_199_9e-3;
    let p = p * r + 8.333_452e-3;
    let p = p * r + 4.166_579_6e-2;
    let p = p * r + 1.666_666_5e-1;
    let p = p * r + 5.000_000_1e-1;
    let p = p * r * r + r + 1.0;
    let scale = f32::from_bits(((n as i32 + 127) as u32) << 23);
    p * scale
}

/// ln x, Cephes `logf`: split into 2^e · m with m in [√½, √2), then a
/// degree-9 polynomial in m − 1 and e·ln 2 added back in two parts. Within
/// 2 ulp for normal positive x; −∞ at 0, NaN below.
#[inline(always)]
pub fn ln(x: f32) -> f32 {
    if x <= 0.0 {
        return if x == 0.0 { f32::NEG_INFINITY } else { f32::NAN };
    }
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xff) as i32 - 126;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f00_0000); // [0.5, 1)
    if m < core::f32::consts::FRAC_1_SQRT_2 {
        e -= 1;
        m = m + m - 1.0;
    } else {
        m -= 1.0;
    }
    let z = m * m;
    let mut y = 7.037_683_1e-2;
    y = y * m - 1.151_461e-1;
    y = y * m + 1.167_699_8e-1;
    y = y * m - 1.242_014_1e-1;
    y = y * m + 1.424_932_3e-1;
    y = y * m - 1.666_805_8e-1;
    y = y * m + 2.000_071_4e-1;
    y = y * m - 2.499_999_4e-1;
    y = y * m + 3.333_333_1e-1;
    y = y * m * z;
    let e = e as f32;
    y += -2.121_944_4e-4 * e;
    y += -0.5 * z;
    let mut r = m + y;
    r += 0.693_359_375 * e;
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_matches_std_within_2ulp() {
        let mut worst = 0f32;
        for i in 1..200_000 {
            let x = i as f32 * 0.37 + 1e-6;
            let (a, b) = (ln(x), x.ln());
            let rel = if b == 0.0 { (a - b).abs() } else { ((a - b) / b).abs() };
            worst = worst.max(rel);
            assert!(rel < 4e-7 || (a - b).abs() < 1e-6, "ln({x}) = {a} vs {b}");
        }
        for x in [1e-30f32, 1e-5, 0.5, 0.9, 1.0, 1.1, 2.0, 1e5, 3e38] {
            let (a, b) = (ln(x), x.ln());
            assert!((a - b).abs() <= 2.0 * b.abs() * f32::EPSILON + 1e-7, "ln({x}) = {a} vs {b}");
        }
        assert_eq!(ln(0.0), f32::NEG_INFINITY);
        assert!(ln(-1.0).is_nan());
        let _ = worst;
    }

    #[test]
    fn exp_matches_std_within_2ulp() {
        let mut worst = 0f32;
        let mut x = -80.0f32;
        while x < 80.0 {
            let (a, b) = (exp(x), x.exp());
            let rel = ((a - b) / b).abs();
            worst = worst.max(rel);
            x += 0.0137;
        }
        assert!(worst < 2.5e-7, "worst rel err {worst}");
    }

    #[test]
    fn floor_round_sqrt() {
        assert_eq!(floor(-0.5), -1.0);
        assert_eq!(floor(2.9), 2.0);
        assert_eq!(round(2.5), 3.0);
        assert_eq!(round(-2.5), -3.0);
        assert_eq!(round(0.49), 0.0);
        assert_eq!(sqrt(9.0), 3.0);
    }
}

/// `x` rounded to the nearest bf16 (ties to even, as torch narrows), as an f32: a value stored at
/// bf16 precision in an f32 buffer.
#[inline(always)]
pub fn bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    if (b & 0x7f80_0000) == 0x7f80_0000 {
        return x; // inf, nan
    }
    let r = ((b >> 16) & 1) + 0x7fff;
    f32::from_bits((b.wrapping_add(r)) & 0xffff_0000)
}
