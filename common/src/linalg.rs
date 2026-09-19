//! Dense f32 linear algebra for the offline half: what GPTQ needs to turn a
//! calibration Hessian into the factor it sweeps with. Row-major `[k][k]`,
//! single-threaded; callers parallelize across matrices. Every inner loop
//! is a `dot` or an `axpy` over contiguous row segments, so the three
//! triangular routines are loop nests over those two.
//!
//! `dot`/`axpy` ship the AVX2 realization on x86_64; `*_scalar` are the
//! twins. They are not bit-identical (FMA vs mul+add) — this is the offline
//! half, and the tests compare with a tolerance.

use crate::math::sqrt;
use std::vec::Vec;

pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

pub fn axpy_scalar(y: &mut [f32], a: f32, x: &[f32]) {
    for (y, x) in y.iter_mut().zip(x) {
        *y += a * x;
    }
}

#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avx2::dot(a.as_ptr(), b.as_ptr(), a.len())
    }
    #[cfg(not(target_arch = "x86_64"))]
    dot_scalar(a, b)
}

/// `y += a · x`.
#[inline(always)]
pub fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avx2::axpy(y.as_mut_ptr(), a, x.as_ptr(), y.len())
    }
    #[cfg(not(target_arch = "x86_64"))]
    axpy_scalar(y, a, x)
}

/// In-place lower Cholesky of a symmetric positive-definite `[k][k]`: on
/// return the lower triangle holds L with A = L·Lᵀ; the upper triangle is
/// left as it was.
pub fn cholesky(k: usize, a: &mut [f32]) {
    for i in 0..k {
        for j in 0..=i {
            let s = a[i * k + j] - dot(&a[i * k..i * k + j], &a[j * k..j * k + j]);
            a[i * k + j] = if i == j {
                assert!(s > 0.0, "not positive definite at {i}");
                sqrt(s)
            } else {
                s / a[j * k + j]
            };
        }
    }
}

/// The upper factor U of A⁻¹ (A⁻¹ = Uᵀ·U) for a symmetric positive-definite
/// `[k][k]` — what GPTQ sweeps with. `a` is destroyed. Three triangular
/// passes: A = L·Lᵀ; M = L⁻¹ kept transposed so its rows are contiguous;
/// A⁻¹ = Mᵀ·M as row dots; then the lower Cholesky of A⁻¹, transposed.
pub fn chol_of_inverse(k: usize, a: &mut [f32]) -> Vec<f32> {
    cholesky(k, a);
    // mt[j][i] = (L⁻¹)[i][j]: column j of L⁻¹ solves L·m = e_j by forward substitution.
    let mut mt = std::vec![0f32; k * k];
    for j in 0..k {
        mt[j * k + j] = 1.0 / a[j * k + j];
        for i in j + 1..k {
            let s = dot(&a[i * k + j..i * k + i], &mt[j * k + j..j * k + i]);
            mt[j * k + i] = -s / a[i * k + i];
        }
    }
    // A⁻¹ = L⁻ᵀ·L⁻¹ → A⁻¹[i][j] = Σ_p mt[i][p]·mt[j][p] over p ≥ max(i, j).
    for i in 0..k {
        for j in 0..=i {
            let v = dot(&mt[i * k + i..(i + 1) * k], &mt[j * k + i..(j + 1) * k]);
            a[i * k + j] = v;
            a[j * k + i] = v;
        }
    }
    cholesky(k, a);
    let mut u = std::vec![0f32; k * k];
    for i in 0..k {
        for j in 0..=i {
            u[j * k + i] = a[i * k + j];
        }
    }
    u
}

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use crate::gemv::avx2::hsum;
    use core::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: *const f32, b: *const f32, n: usize) -> f32 {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 32 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i + 8)), _mm256_loadu_ps(b.add(i + 8)), a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i + 16)), _mm256_loadu_ps(b.add(i + 16)), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i + 24)), _mm256_loadu_ps(b.add(i + 24)), a3);
            i += 32;
        }
        while i + 8 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), a0);
            i += 8;
        }
        let mut s = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        while i < n {
            s += *a.add(i) * *b.add(i);
            i += 1;
        }
        s
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy(y: *mut f32, a: f32, x: *const f32, n: usize) {
        let av = _mm256_set1_ps(a);
        let mut i = 0;
        while i + 8 <= n {
            _mm256_storeu_ps(y.add(i), _mm256_fmadd_ps(av, _mm256_loadu_ps(x.add(i)), _mm256_loadu_ps(y.add(i))));
            i += 8;
        }
        while i < n {
            *y.add(i) += a * *x.add(i);
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Rng;

    #[test]
    fn dot_axpy_match_scalar() {
        let mut rng = Rng::new(3);
        for n in [0, 1, 7, 8, 33, 257] {
            let a: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
            let b: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
            assert!((dot(&a, &b) - dot_scalar(&a, &b)).abs() < 1e-4 * (n as f32 + 1.0));
            let (mut y1, mut y2) = (a.clone(), a.clone());
            axpy(&mut y1, 0.37, &b);
            axpy_scalar(&mut y2, 0.37, &b);
            for i in 0..n {
                assert!((y1[i] - y2[i]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn chol_of_inverse_inverts() {
        // A = XᵀX + I for random X: SPD. Check UᵀU·A ≈ I.
        let (k, n) = (48, 64);
        let mut rng = Rng::new(7);
        let x: Vec<f32> = (0..n * k).map(|_| rng.unit()).collect();
        let mut a = std::vec![0f32; k * k];
        for i in 0..k {
            for j in 0..k {
                a[i * k + j] = (0..n).map(|p| x[p * k + i] * x[p * k + j]).sum::<f32>() + if i == j { 1.0 } else { 0.0 };
            }
        }
        let a0 = a.clone();
        let u = chol_of_inverse(k, &mut a);
        // (UᵀU)·A ≈ I
        let mut ainv = std::vec![0f32; k * k];
        for i in 0..k {
            for j in 0..k {
                ainv[i * k + j] = (0..k).map(|p| u[p * k + i] * u[p * k + j]).sum();
            }
        }
        for i in 0..k {
            for j in 0..k {
                let v: f32 = (0..k).map(|p| ainv[i * k + p] * a0[p * k + j]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 2e-3, "({i},{j}) = {v}");
            }
        }
        // U is upper
        for i in 0..k {
            for j in 0..i {
                assert_eq!(u[i * k + j], 0.0);
            }
        }
    }
}
