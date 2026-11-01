//! The transposed product over FP8 rows: `dx[0..K) += Wᵀ · dy` for a row-major `[M][K]`
//! e4m3 matrix with one f32 scale per 128×128 block (`common::gemv::fp8`'s format). Rows
//! are streamed in order, each row's coefficient `dy[m] · scale[block]` multiplying its
//! exactly-widened bytes into the `K` lanes of `dx`, which stay in L1.
//!
//! One fused multiply-add a lane, `dx[k] = fma(w, c, dx[k])`, on both sides: the scalar
//! definition has `std`'s `mul_add`, so the AVX2 realization matches it bit for bit —
//! unlike the forward pair, whose `no_std` scalar twin rounds twice.

use common::gemv::{e4m3, FP8_BLOCK};

/// The definition, over rows `m0..m1`.
pub fn rows_t_scalar(k: usize, w: &[u8], scales: &[f32], dy: &[f32], dx: &mut [f32], m0: usize, m1: usize) {
    debug_assert!(k % FP8_BLOCK == 0);
    let kb = k / FP8_BLOCK;
    for m in m0..m1 {
        let row = &w[m * k..(m + 1) * k];
        let srow = &scales[(m / FP8_BLOCK) * kb..(m / FP8_BLOCK + 1) * kb];
        for g in 0..kb {
            let c = dy[m] * srow[g];
            for kk in g * FP8_BLOCK..(g + 1) * FP8_BLOCK {
                dx[kk] = e4m3(row[kk]).mul_add(c, dx[kk]);
            }
        }
    }
}

/// The realization this target ships, over rows `m0..m1` so a pool can split the rows.
#[inline(always)]
pub fn rows_t<const K: usize>(w: &[u8], scales: &[f32], dy: &[f32], dx: &mut [f32], m0: usize, m1: usize) {
    debug_assert!(w.len() >= m1 * K && dy.len() >= m1 && dx.len() >= K);
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avx2::rows_t::<K>(w.as_ptr(), scales.as_ptr(), dy.as_ptr(), dx.as_mut_ptr(), m0, m1)
    }
    #[cfg(not(target_arch = "x86_64"))]
    rows_t_scalar(K, w, scales, dy, dx, m0, m1)
}

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use super::FP8_BLOCK;
    use core::arch::x86_64::*;

    /// Eight e4m3 bytes → eight f32, exactly (`common::gemv::e4m3`; the forward's lanes, kept private there).
    #[inline(always)]
    unsafe fn e4m3x8(p: *const u8) -> __m256 {
        let b = _mm256_cvtepu8_epi32(_mm_loadl_epi64(p as *const __m128i));
        let sign = _mm256_slli_epi32::<24>(_mm256_and_si256(b, _mm256_set1_epi32(0x80)));
        let e = _mm256_and_si256(_mm256_srli_epi32::<3>(b), _mm256_set1_epi32(0xf));
        let m = _mm256_and_si256(b, _mm256_set1_epi32(7));
        let normal = _mm256_or_si256(sign, _mm256_or_si256(_mm256_slli_epi32::<23>(_mm256_add_epi32(e, _mm256_set1_epi32(120))), _mm256_slli_epi32::<20>(m)));
        let sub = _mm256_or_si256(_mm256_castps_si256(_mm256_mul_ps(_mm256_cvtepi32_ps(m), _mm256_set1_ps(1.0 / 512.0))), sign);
        let is_sub = _mm256_cmpeq_epi32(e, _mm256_setzero_si256());
        _mm256_castsi256_ps(_mm256_blendv_epi8(normal, sub, is_sub))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn rows_t<const K: usize>(w: *const u8, scales: *const f32, dy: *const f32, dx: *mut f32, m0: usize, m1: usize) {
        const { assert!(K % FP8_BLOCK == 0) };
        let kb = K / FP8_BLOCK;
        for m in m0..m1 {
            let row = w.add(m * K);
            let srow = scales.add((m / FP8_BLOCK) * kb);
            let d = *dy.add(m);
            for g in 0..kb {
                let c = _mm256_set1_ps(d * *srow.add(g));
                let mut k = g * FP8_BLOCK;
                let end = k + FP8_BLOCK;
                while k < end {
                    let p = dx.add(k);
                    _mm256_storeu_ps(p, _mm256_fmadd_ps(e4m3x8(row.add(k)), c, _mm256_loadu_ps(p)));
                    k += 8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::testing::Rng;

    /// Random e4m3 bytes (never the NaN codes) and block scales, with the dense f32 twin.
    pub fn record(m: usize, k: usize, rnd: &mut Rng) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let w: Vec<u8> = (0..m * k).map(|_| { let b = (rnd.next() % 256) as u8; if b & 0x7f == 0x7f { b ^ 1 } else { b } }).collect();
        let kb = k / FP8_BLOCK;
        let s: Vec<f32> = (0..m.div_ceil(FP8_BLOCK) * kb).map(|_| 0.001 + (rnd.next() % 100) as f32 * 1e-4).collect();
        let dense: Vec<f32> = (0..m * k).map(|i| e4m3(w[i]) * s[(i / k / FP8_BLOCK) * kb + (i % k) / FP8_BLOCK]).collect();
        (w, s, dense)
    }

    #[test]
    fn avx2_matches_scalar_bit_for_bit() {
        let mut rnd = Rng::new(0xF8F8_0001);
        let (m, k) = (300usize, 512usize);
        let (w, s, _) = record(m, k, &mut rnd);
        let dy: Vec<f32> = (0..m).map(|_| rnd.unit()).collect();
        let start: Vec<f32> = (0..k).map(|_| rnd.unit()).collect();
        let (mut a, mut b) = (start.clone(), start);
        rows_t_scalar(k, &w, &s, &dy, &mut a, 3, m);
        rows_t::<512>(&w, &s, &dy, &mut b, 3, m);
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn scalar_is_the_transposed_product() {
        let mut rnd = Rng::new(0xF8F8_0002);
        let (m, k) = (256usize, 256usize);
        let (w, s, dense) = record(m, k, &mut rnd);
        let dy: Vec<f32> = (0..m).map(|_| rnd.unit()).collect();
        let mut dx = vec![0f32; k];
        rows_t_scalar(k, &w, &s, &dy, &mut dx, 0, m);
        for kk in 0..k {
            let want: f32 = (0..m).map(|r| dense[r * k + kk] * dy[r]).sum();
            assert!((dx[kk] - want).abs() <= 1e-4 * (1.0 + want.abs()), "k {kk}: {} vs {want}", dx[kk]);
        }
    }
}
