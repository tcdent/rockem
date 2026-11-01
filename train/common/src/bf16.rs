//! The transposed product over bf16 rows: `dx[0..K) += Σ_r dy_r · W[row_r]` for the rows a
//! caller names — the router's picked rows — or for every row of a `[M][K]` matrix (a dense
//! projection's backward). Row-major bf16 (`common::gemv::bf16`'s format); the widen is a
//! shift; one fused multiply-add a lane on both sides, so the AVX2 realization matches the
//! scalar bit for bit.

use common::gemv::widen;

/// The definition of one row's share: `dx += c · W[row]`.
pub fn row_t_scalar(k: usize, w: &[u16], row: usize, c: f32, dx: &mut [f32]) {
    let r = &w[row * k..(row + 1) * k];
    for kk in 0..k {
        dx[kk] = widen(r[kk]).mul_add(c, dx[kk]);
    }
}

/// For each `(row, c)` in `rows`, `dx += c · W[row]`.
pub fn rows_t_scalar(k: usize, w: &[u16], rows: &[(usize, f32)], dx: &mut [f32]) {
    for &(row, c) in rows {
        row_t_scalar(k, w, row, c, dx);
    }
}

/// Every row: `dx += Wᵀ · dy` over `[m][k]`.
pub fn dense_t_scalar(k: usize, m: usize, w: &[u16], dy: &[f32], dx: &mut [f32]) {
    for row in 0..m {
        row_t_scalar(k, w, row, dy[row], dx);
    }
}

/// The realization this target ships, named rows.
#[inline(always)]
pub fn rows_t<const K: usize>(w: &[u16], rows: &[(usize, f32)], dx: &mut [f32]) {
    debug_assert!(dx.len() >= K);
    #[cfg(target_arch = "x86_64")]
    unsafe {
        for &(row, c) in rows {
            avx2::row_t::<K>(w.as_ptr(), row, c, dx.as_mut_ptr())
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    rows_t_scalar(K, w, rows, dx)
}

/// The realization this target ships, every row of `[m][K]`.
#[inline(always)]
pub fn dense_t<const K: usize>(m: usize, w: &[u16], dy: &[f32], dx: &mut [f32]) {
    debug_assert!(w.len() >= m * K && dy.len() >= m && dx.len() >= K);
    #[cfg(target_arch = "x86_64")]
    unsafe {
        for row in 0..m {
            avx2::row_t::<K>(w.as_ptr(), row, dy[row], dx.as_mut_ptr())
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    dense_t_scalar(K, m, w, dy, dx)
}

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn widen8(p: __m128i) -> __m256 {
        _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(p)))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn row_t<const K: usize>(w: *const u16, row: usize, c: f32, dx: *mut f32) {
        const { assert!(K % 16 == 0) };
        let r = w.add(row * K);
        let cv = _mm256_set1_ps(c);
        let mut k = 0;
        while k < K {
            let p = _mm256_loadu_si256(r.add(k) as *const __m256i);
            let d0 = dx.add(k);
            let d1 = dx.add(k + 8);
            _mm256_storeu_ps(d0, _mm256_fmadd_ps(widen8(_mm256_castsi256_si128(p)), cv, _mm256_loadu_ps(d0)));
            _mm256_storeu_ps(d1, _mm256_fmadd_ps(widen8(_mm256_extracti128_si256::<1>(p)), cv, _mm256_loadu_ps(d1)));
            k += 16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::testing::Rng;

    pub fn bf16_of(x: f32) -> u16 {
        (x.to_bits() >> 16) as u16
    }

    #[test]
    fn avx2_matches_scalar_and_the_dense_product() {
        let mut rnd = Rng::new(0xBF16_0001);
        let (m, k) = (40usize, 256usize);
        let w: Vec<u16> = (0..m * k).map(|_| bf16_of(rnd.unit())).collect();
        let rows: Vec<(usize, f32)> = [3usize, 7, 7, 39, 0].iter().map(|&r| (r, rnd.unit())).collect();
        let start: Vec<f32> = (0..k).map(|_| rnd.unit()).collect();
        let (mut a, mut b) = (start.clone(), start.clone());
        rows_t_scalar(k, &w, &rows, &mut a);
        rows_t::<256>(&w, &rows, &mut b);
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
        for kk in 0..k {
            let want: f32 = start[kk] + rows.iter().map(|&(r, c)| widen(w[r * k + kk]) * c).sum::<f32>();
            assert!((a[kk] - want).abs() <= 1e-5 * (1.0 + want.abs()));
        }
        // every row
        let dy: Vec<f32> = (0..m).map(|_| rnd.unit()).collect();
        let (mut a, mut b) = (start.clone(), start.clone());
        dense_t_scalar(k, m, &w, &dy, &mut a);
        dense_t::<256>(m, &w, &dy, &mut b);
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
        for kk in 0..k {
            let want: f32 = start[kk] + (0..m).map(|r| widen(w[r * k + kk]) * dy[r]).sum::<f32>();
            assert!((a[kk] - want).abs() <= 1e-4 * (1.0 + want.abs()));
        }
    }
}
