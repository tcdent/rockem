//! The transposed product over `q4g128` bytes: `dx[0..K) += Wᵀ[K × m] · dy[0..m)`,
//! the gradient at the input of a frozen INT4 matrix given the gradient at
//! its output. The forward (`common::q4::gemv`) walks the tensor tile by tile
//! (four rows), group by group (128 columns), and so does this: the bytes are
//! streamed once, in the forward's order, and the 128 lanes of `dx` a group
//! touches sit in L1 for the four rows of the tile.
//!
//! The gradient is f32, not an INT8 activation, so there is no integer dot:
//! the nibble becomes an f32 and multiplies the row's coefficient
//! `c_r = dy[row] · scale[row, g]`. The +8 bias of the stored nibble is one
//! scalar a (tile, group) — `8 · Σ_r c_r` — subtracted once a lane, as the
//! forward subtracts `8 · Σa` once a row.
//!
//! One fixed expression a lane, so every realization gives the same bits:
//!
//! ```text
//! v = −8·(c0 + c1 + c2 + c3)        then, rows in order 0..3:  v = v + n_r · c_r
//! dx[k] = dx[k] + v
//! ```
//! with the products and sums separate (no fused multiply-add on either side).

use common::q4::{GROUP, TILE, WBIAS};

/// The definition. Every fast realization must match it bit for bit.
pub fn gemv_t_scalar(k: usize, m: usize, w: &[u8], ws: &[f32], dy: &[f32], dx: &mut [f32]) {
    debug_assert!(k % GROUP == 0 && m % TILE == 0);
    let ng = k / GROUP;
    for t in 0..m / TILE {
        for g in 0..ng {
            let mut c = [0f32; TILE];
            for r in 0..TILE {
                c[r] = dy[t * TILE + r] * ws[t * ng * TILE + g * TILE + r];
            }
            let bias = -(WBIAS as f32) * (((c[0] + c[1]) + c[2]) + c[3]);
            let base = t * k * 2 + g * (GROUP / 2) * TILE;
            for ch in 0..2 {
                for b in 0..32 {
                    let (mut lo, mut hi) = (bias, bias);
                    for r in 0..TILE {
                        let byte = w[base + r * (GROUP / 2) + ch * 32 + b];
                        lo = lo + (byte & 0x0F) as f32 * c[r];
                        hi = hi + (byte >> 4) as f32 * c[r];
                    }
                    let k0 = g * GROUP + ch * 64 + b;
                    dx[k0] = dx[k0] + lo;
                    dx[k0 + 32] = dx[k0 + 32] + hi;
                }
            }
        }
    }
}

/// The realization this target ships.
#[inline(always)]
pub fn gemv_t<const K: usize>(m: usize, w: &[u8], ws: &[f32], dy: &[f32], dx: &mut [f32]) {
    debug_assert!(w.len() >= m * K / 2 && dy.len() >= m && dx.len() >= K);
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avx2::gemv_t::<K>(m, w.as_ptr(), ws.as_ptr(), dy.as_ptr(), dx.as_mut_ptr())
    }
    #[cfg(not(target_arch = "x86_64"))]
    gemv_t_scalar(K, m, w, ws, dy, dx)
}

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    //! A tile's group: one 32-byte load a row a half-group, its nibbles widened to
    //! four f32 vectors each way (low nibbles are lanes 0..31 of the half, high
    //! nibbles lanes 32..63), eight running vectors for the half's 64 lanes of `dx`.
    //! Eight accumulators, the 0x0F mask, a coefficient and the widened nibbles
    //! fit the sixteen `ymm`.
    use super::{GROUP, TILE, WBIAS};
    use core::arch::x86_64::*;

    /// Eight bytes at `p` (one `__m128i`'s low half) → eight f32.
    #[inline(always)]
    unsafe fn widen8(p: __m128i) -> __m256 {
        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(p))
    }

    /// `acc[i] = acc[i] + n[i] · c` over the 32 nibbles of `n`, as 4 × 8 lanes.
    #[inline(always)]
    unsafe fn step32(acc: &mut [__m256; 4], n: __m256i, c: __m256) {
        let lo = _mm256_castsi256_si128(n);
        let hi = _mm256_extracti128_si256::<1>(n);
        acc[0] = _mm256_add_ps(acc[0], _mm256_mul_ps(widen8(lo), c));
        acc[1] = _mm256_add_ps(acc[1], _mm256_mul_ps(widen8(_mm_srli_si128::<8>(lo)), c));
        acc[2] = _mm256_add_ps(acc[2], _mm256_mul_ps(widen8(hi), c));
        acc[3] = _mm256_add_ps(acc[3], _mm256_mul_ps(widen8(_mm_srli_si128::<8>(hi)), c));
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn gemv_t<const K: usize>(m: usize, w: *const u8, ws: *const f32, dy: *const f32, dx: *mut f32) {
        const { assert!(K % GROUP == 0) };
        let ng = K / GROUP;
        let mask = _mm256_set1_epi8(0x0F);
        for t in 0..m / TILE {
            let wt = w.add(t * K * 2);
            let st = ws.add(t * ng * TILE);
            let dyt = dy.add(t * TILE);
            for g in 0..ng {
                let mut c = [0f32; TILE];
                for r in 0..TILE {
                    c[r] = *dyt.add(r) * *st.add(g * TILE + r);
                }
                let bias = _mm256_set1_ps(-(WBIAS as f32) * (((c[0] + c[1]) + c[2]) + c[3]));
                let wg = wt.add(g * (GROUP / 2) * TILE);
                let dxg = dx.add(g * GROUP);
                for ch in 0..2 {
                    // lanes ch·64 .. ch·64+31 from the low nibbles, +32 from the high
                    let mut lo = [bias; 4];
                    let mut hi = [bias; 4];
                    for r in 0..TILE {
                        let p = _mm256_loadu_si256(wg.add(r * (GROUP / 2) + ch * 32) as *const __m256i);
                        let cr = _mm256_set1_ps(c[r]);
                        step32(&mut lo, _mm256_and_si256(p, mask), cr);
                        step32(&mut hi, _mm256_and_si256(_mm256_srli_epi16::<4>(p), mask), cr);
                    }
                    let d = dxg.add(ch * 64);
                    for i in 0..4 {
                        let p = d.add(i * 8);
                        _mm256_storeu_ps(p, _mm256_add_ps(_mm256_loadu_ps(p), lo[i]));
                        let p = d.add(32 + i * 8);
                        _mm256_storeu_ps(p, _mm256_add_ps(_mm256_loadu_ps(p), hi[i]));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::q4;
    use common::testing::Rng;

    fn record(m: usize, k: usize, rnd: &mut Rng) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let w: Vec<i8> = (0..m * k).map(|_| rnd.i4()).collect();
        let s: Vec<f32> = (0..m * (k / GROUP)).map(|_| 0.01 + (rnd.next() % 100) as f32 * 1e-4).collect();
        let (pw, ps) = q4::pack(m, k, &w, &s);
        let dense: Vec<f32> = (0..m * k).map(|i| w[i] as f32 * s[(i / k) * (k / GROUP) + (i % k) / GROUP]).collect();
        (pw, ps, dense)
    }

    /// The AVX2 realization against the definition, bit for bit, on both expert shapes.
    #[test]
    fn avx2_matches_scalar_bit_for_bit() {
        let mut rnd = Rng::new(0x5EED_0001);
        for &(m, k) in &[(32usize, 4096usize), (128, 2048), (8, 256)] {
            let (pw, ps, _) = record(m, k, &mut rnd);
            let dy: Vec<f32> = (0..m).map(|_| rnd.unit()).collect();
            let start: Vec<f32> = (0..k).map(|_| rnd.unit()).collect();
            let mut a = start.clone();
            let mut b = start.clone();
            gemv_t_scalar(k, m, &pw, &ps, &dy, &mut a);
            match k {
                4096 => gemv_t::<4096>(m, &pw, &ps, &dy, &mut b),
                2048 => gemv_t::<2048>(m, &pw, &ps, &dy, &mut b),
                _ => gemv_t::<256>(m, &pw, &ps, &dy, &mut b),
            }
            let (ab, bb): (Vec<u32>, Vec<u32>) = (a.iter().map(|v| v.to_bits()).collect(), b.iter().map(|v| v.to_bits()).collect());
            assert_eq!(ab, bb, "m {m} k {k}");
        }
    }

    /// The definition against the dense product `Wᵀ·dy` on the dequantized matrix.
    #[test]
    fn scalar_is_the_transposed_product() {
        let mut rnd = Rng::new(0x5EED_0002);
        let (m, k) = (64usize, 512usize);
        let (pw, ps, dense) = record(m, k, &mut rnd);
        let dy: Vec<f32> = (0..m).map(|_| rnd.unit()).collect();
        let mut dx = vec![0f32; k];
        gemv_t_scalar(k, m, &pw, &ps, &dy, &mut dx);
        for kk in 0..k {
            let want: f32 = (0..m).map(|r| dense[r * k + kk] * dy[r]).sum();
            assert!((dx[kk] - want).abs() <= 1e-4 * (1.0 + want.abs()), "k {kk}: {} vs {want}", dx[kk]);
        }
    }
}
