//! `q4g128-a8`: INT4 symmetric weights with one f32 scale per 128 columns,
//! stored biased by +8 and tiled four rows at a time, against INT8
//! activations quantized per group with the same granularity. The format,
//! the activation quantizer, and the GEMV inner loop on it. Consumed by
//! every model's CPU expert kernel; the layout is chosen for the AVX2 load
//! pattern and the GPU realization will read the same bytes.
//!
//! A weight tensor of M rows × K columns is M/4 **tiles**. A tile is four
//! output rows walked group-major: for each group of 128 k-values, the four
//! rows' 64 bytes each, back to back — 256 bytes per group. One sequential
//! stream feeds four independent accumulator chains.
//!
//! ```text
//! tile t, group g, row r  ->  byte offset  t·(K·2) + g·256 + r·64
//! ```
//!
//! Inside a row's 64 bytes for one group: two 32-byte chunks. Chunk c covers
//! k = c·64 .. c·64+63. Byte b holds w[b] in its low nibble and w[b+32] in
//! its high nibble, so one `vpand` yields lanes 0..31 in k order and a
//! `vpsrlw`+`vpand` yields lanes 32..63 — matching two straight 32-byte loads
//! of the activation vector. No shuffle.
//!
//! **Nibble = w + 8.** `vpmaddubsw` wants one unsigned operand; nibbles come
//! out of `vpand` unsigned for free. The bias adds `8·Σa` per group, which
//! depends only on the activations, so it is one scalar per group per token
//! shared by every row of every expert. Nothing per-row is stored to correct
//! it and no sign-extension instruction ever runs.
//!
//! Scales mirror the tiling: tile t, group g, four f32 at `t·(K/128·4) + g·4`.

use super::math::round;

pub const GROUP: usize = 128;
pub const TILE: usize = 4;
pub const WBIAS: i32 = 8;
/// Tokens `gemm` takes at once: its i32 partials and f32 sums a token (32 B) stay in L1; a caller
/// with more tokens loops blocks of this many.
pub const GEMM_TOKENS: usize = 64;

/// Byte offset of (row, k)'s byte within the packed tensor, and whether k is
/// in the high nibble.
#[inline(always)]
pub fn byte_index(k_total: usize, row: usize, k: usize) -> (usize, bool) {
    let (t, r) = (row / TILE, row % TILE);
    let g = k / GROUP;
    let kk = k % GROUP;
    let c = kk / 64;
    let b = kk % 64;
    let base = t * k_total * 2 + g * (GROUP / 2) * TILE + r * (GROUP / 2);
    (base + c * 32 + (b % 32), b >= 32)
}

#[inline(always)]
pub fn scale_index(k_total: usize, row: usize, g: usize) -> usize {
    let (t, r) = (row / TILE, row % TILE);
    t * (k_total / GROUP) * TILE + g * TILE + r
}

pub fn unpack(k_total: usize, w: &[u8], row: usize, k: usize) -> i8 {
    let (i, hi) = byte_index(k_total, row, k);
    let n = if hi { w[i] >> 4 } else { w[i] & 0x0F };
    (n as i32 - WBIAS) as i8
}

/// Pack a row-major `[m][k]` matrix of INT4 values in [-8, 7] with per
/// (row, group) scales. The offline half; the bundle builder calls this.
#[cfg(any(test, feature = "std"))]
pub fn pack(m: usize, k: usize, w: &[i8], scales: &[f32]) -> (std::vec::Vec<u8>, std::vec::Vec<f32>) {
    assert!(m % TILE == 0 && k % GROUP == 0);
    let ng = k / GROUP;
    let mut out = std::vec![0u8; m * k / 2];
    let mut sc = std::vec![0f32; m * ng];
    for row in 0..m {
        for kk in 0..k {
            let v = (w[row * k + kk] as i32 + WBIAS) as u8;
            assert!(v < 16);
            let (i, hi) = byte_index(k, row, kk);
            out[i] |= if hi { v << 4 } else { v };
        }
        for g in 0..ng {
            sc[scale_index(k, row, g)] = scales[row * ng + g];
        }
    }
    (out, sc)
}

/// GPTQ (Frantar et al.): the same format as `pack` expects, integers chosen
/// with error feedback instead of nearest. `w` is row-major `[m][k]` f32 and
/// is destroyed; `u` is `[k][k]`, the upper factor of the calibration
/// Hessian's inverse (`linalg::chol_of_inverse`). Column by column: round,
/// push the rounding error onto the unrounded columns of the group through
/// `u`'s row, and after each group push the group's errors onto the rest of
/// the row in one pass. Scales are absmax/7 of the error-fed weights at the
/// group's first column, as in the reference. `u = I` reduces this to RTN.
#[cfg(any(test, feature = "std"))]
pub fn gptq(m: usize, k: usize, w: &mut [f32], u: &[f32]) -> (std::vec::Vec<i8>, std::vec::Vec<f32>) {
    use crate::linalg::axpy;
    assert!(k % GROUP == 0 && w.len() == m * k && u.len() == k * k);
    let ng = k / GROUP;
    let mut q = std::vec![0i8; m * k];
    let mut s = std::vec![0f32; m * ng];
    let mut err = std::vec![0f32; m * GROUP];
    for g in 0..ng {
        let (i1, i2) = (g * GROUP, (g + 1) * GROUP);
        // Rows are independent within a group (only columns interact), so each
        // row's 128-column segment is swept to completion while it sits in L1.
        for r in 0..m {
            let row = &mut w[r * k + i1..r * k + i2];
            let amax = row.iter().fold(0f32, |a, v| a.max(v.abs()));
            let sc = amax / 7.0;
            let sc = if sc > 0.0 { sc } else { 1.0 };
            s[r * ng + g] = sc;
            let inv = 1.0 / sc;
            for i in 0..GROUP {
                let wv = row[i];
                let qi = round(wv * inv).clamp(-8.0, 7.0);
                q[r * k + i1 + i] = qi as i8;
                let e = (wv - qi * sc) / u[(i1 + i) * k + i1 + i];
                axpy(&mut row[i..], -e, &u[(i1 + i) * k + i1 + i..(i1 + i) * k + i2]);
                err[r * GROUP + i] = e;
            }
        }
        // w[:, i2..] -= err · u[i1..i2][i2..], tiled so a slab of factor rows is
        // read from L2 for a tile of weight rows rather than from memory for every row.
        if i2 < k {
            const RT: usize = 16;
            const BT: usize = 32;
            for b0 in (0..GROUP).step_by(BT) {
                for r0 in (0..m).step_by(RT) {
                    for r in r0..(r0 + RT).min(m) {
                        let wrow = &mut w[r * k + i2..(r + 1) * k];
                        for b in b0..b0 + BT {
                            axpy(wrow, -err[r * GROUP + b], &u[(i1 + b) * k + i2..(i1 + b + 1) * k]);
                        }
                    }
                }
            }
        }
    }
    (q, s)
}

/// Activations: f32 → i8 per group of 128 (symmetric, absmax/127), plus the
/// two per-group scalars the kernel needs — the scale and `Σq` for the bias
/// term. Once per token per expert input; scalar is fine.
pub struct QAct<'a> {
    pub q: &'a mut [i8],      // K
    pub scale: &'a mut [f32], // K / GROUP
    pub gsum: &'a mut [i32],  // K / GROUP
}
/// A quantized activation as the GEMVs read it: shared freely, by many rows on many cores.
#[derive(Clone, Copy)]
pub struct Act<'a> {
    pub q: &'a [i8],
    pub scale: &'a [f32],
    pub gsum: &'a [i32],
}
impl QAct<'_> {
    #[inline(always)]
    pub fn act(&self) -> Act<'_> {
        Act { q: self.q, scale: self.scale, gsum: self.gsum }
    }
}

impl QAct<'_> {
    #[inline(always)]
    pub fn quantize(&mut self, x: &[f32]) {
        #[cfg(target_arch = "x86_64")]
        if GROUP == 128 {
            unsafe { quantize_avx2(x, self.q, self.scale, self.gsum) };
            return;
        }
        self.quantize_scalar(x)
    }
    /// The definition: per group, the absmax scale, round half away from zero, the group's sum.
    pub fn quantize_scalar(&mut self, x: &[f32]) {
        for (g, chunk) in x.chunks_exact(GROUP).enumerate() {
            let mut amax = 0f32;
            for &v in chunk {
                let a = if v < 0.0 { -v } else { v };
                if a > amax {
                    amax = a;
                }
            }
            let s = amax / 127.0;
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            let mut sum = 0i32;
            for (i, &v) in chunk.iter().enumerate() {
                let r = round(v * inv);
                let r = if r > 127.0 { 127.0 } else if r < -127.0 { -127.0 } else { r };
                let q = r as i8;
                self.q[g * GROUP + i] = q;
                sum += q as i32;
            }
            self.scale[g] = s;
            self.gsum[g] = sum;
        }
    }
}

/// `QAct::quantize_scalar` on AVX2, the same arithmetic in the same order per element (absmax,
/// `amax / 127`, `x · (1 / s)`, `(r ± 0.5) as i32` truncated, clamped to ±127) — bit for bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_avx2(x: &[f32], q: &mut [i8], scale: &mut [f32], gsum: &mut [i32]) {
    use core::arch::x86_64::*;
    debug_assert!(GROUP == 128);
    let sign = _mm256_set1_ps(-0.0);
    let half = _mm256_set1_ps(0.5);
    let hi = _mm256_set1_epi32(127);
    let lo = _mm256_set1_epi32(-127);
    for (g, chunk) in x.chunks_exact(GROUP).enumerate() {
        let p = chunk.as_ptr();
        let mut m = _mm256_setzero_ps();
        for i in (0..GROUP).step_by(8) {
            m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(p.add(i))));
        }
        // the lanes' max, then the scalar's exact `amax / 127` and `1 / s`
        let m128 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        let m32 = _mm_max_ss(m64, _mm_shuffle_ps(m64, m64, 1));
        let amax = _mm_cvtss_f32(m32);
        let s = amax / 127.0;
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        let vinv = _mm256_set1_ps(inv);
        let mut sum = _mm256_setzero_si256();
        for i in (0..GROUP).step_by(8) {
            let v = _mm256_mul_ps(_mm256_loadu_ps(p.add(i)), vinv);
            // (v ± 0.5) as i32: the half with v's sign, then truncation
            let r = _mm256_add_ps(v, _mm256_or_ps(half, _mm256_and_ps(v, sign)));
            let t = _mm256_cvttps_epi32(r);
            let c = _mm256_min_epi32(hi, _mm256_max_epi32(lo, t));
            sum = _mm256_add_epi32(sum, c);
            // eight i32 → eight i8
            let packed = _mm256_packs_epi16(_mm256_packs_epi32(c, c), _mm256_setzero_si256());
            let lo64 = _mm256_castsi256_si128(packed);
            let hi64 = _mm256_extracti128_si256(packed, 1);
            let b0 = _mm_cvtsi128_si32(lo64) as u32;
            let b1 = _mm_cvtsi128_si32(hi64) as u32;
            let bytes = (b0 as u64) | ((b1 as u64) << 32);
            core::ptr::copy_nonoverlapping(&bytes as *const u64 as *const i8, q.as_mut_ptr().add(g * GROUP + i), 8);
        }
        let s128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        let s64 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0b01_00_11_10));
        let s32 = _mm_add_epi32(s64, _mm_shuffle_epi32(s64, 0b00_00_00_01));
        scale[g] = s;
        gsum[g] = _mm_cvtsi128_si32(s32);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod quantize_tests {
    use super::*;
    #[test]
    fn avx2_matches_scalar_bit_for_bit() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for trial in 0..2000 {
            let k = 4096;
            let mut x = std::vec![0f32; k];
            for v in x.iter_mut() {
                let r = (next() % 2_000_001) as f32 / 1000.0 - 1000.0;
                *v = match trial % 4 {
                    0 => r,
                    1 => r * 1e-3,
                    2 => (r as i32 as f32) * 0.5, // many exact halves after scaling
                    _ => if next() % 7 == 0 { 0.0 } else { r * 1e4 },
                };
            }
            if trial % 11 == 0 {
                for v in x[..GROUP].iter_mut() {
                    *v = 0.0;
                }
            }
            let (mut q1, mut s1, mut g1) = (std::vec![0i8; k], std::vec![0f32; k / GROUP], std::vec![0i32; k / GROUP]);
            let (mut q2, mut s2, mut g2) = (std::vec![0i8; k], std::vec![0f32; k / GROUP], std::vec![0i32; k / GROUP]);
            QAct { q: &mut q1, scale: &mut s1, gsum: &mut g1 }.quantize_scalar(&x);
            unsafe { quantize_avx2(&x, &mut q2, &mut s2, &mut g2) };
            assert_eq!(q1, q2, "trial {trial}");
            assert_eq!(s1.iter().map(|v| v.to_bits()).collect::<std::vec::Vec<_>>(), s2.iter().map(|v| v.to_bits()).collect::<std::vec::Vec<_>>(), "trial {trial}");
            assert_eq!(g1, g2, "trial {trial}");
        }
    }
}

/// y[0..m) = W[m × K] · a[K], scalar. Every fast realization must match
/// this bit for bit: per-group i32 accumulation (exact in any order), one
/// f32 flush per group in one fixed expression, groups summed in order.
pub fn gemv_scalar(k: usize, m: usize, w: &[u8], ws: &[f32], act: &Act, out: &mut [f32]) {
    let ng = k / GROUP;
    for row in 0..m {
        let (t, r) = (row / TILE, row % TILE);
        let mut y = 0f32;
        for g in 0..ng {
            let base = t * k * 2 + g * (GROUP / 2) * TILE + r * (GROUP / 2);
            let mut acc = 0i32;
            for c in 0..2 {
                for b in 0..32 {
                    let byte = w[base + c * 32 + b];
                    let k0 = g * GROUP + c * 64 + b;
                    acc += ((byte & 0x0F) as i32 - WBIAS) * act.q[k0] as i32;
                    acc += ((byte >> 4) as i32 - WBIAS) * act.q[k0 + 32] as i32;
                }
            }
            // the flush, in exactly the form every fast realization uses: cvt(Σ) · (w_scale · a_scale)
            y += acc as f32 * (ws[t * ng * TILE + g * TILE + r] * act.scale[g]);
        }
        out[row] = y;
    }
}

/// `gemv` for `n` activations at once: `out(t)[0..m) = W · act(t)`. The weights are read once a
/// tile and group and every token's dot runs on them, so the weight stream stays sequential while
/// the dots run (a token at a time, the first token streams a part from DRAM and the rest re-read
/// it from L2, and the two times add). The same integer sum a group (exact in any order), the same
/// flush, the groups in order: `gemv`'s bits for every token.
#[inline(always)]
pub fn gemm<'a, const K: usize>(m: usize, w: &[u8], ws: &[f32], n: usize, act: impl Fn(usize) -> Act<'a>, out: impl Fn(usize) -> *mut f32) {
    debug_assert!(w.len() >= m * K / 2);
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avx2::gemm::<K>(m, w.as_ptr(), ws.as_ptr(), n, act, out)
    }
    #[cfg(not(target_arch = "x86_64"))]
    for t in 0..n {
        gemv_scalar(K, m, w, ws, &act(t), unsafe { core::slice::from_raw_parts_mut(out(t), m) });
    }
}

/// The realization this target ships.
#[inline(always)]
pub fn gemv<const K: usize>(m: usize, w: &[u8], ws: &[f32], act: &Act, out: &mut [f32]) {
    debug_assert!(w.len() >= m * K / 2 && out.len() >= m);
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avx2::gemv::<K>(m, w.as_ptr(), ws.as_ptr(), act, out.as_mut_ptr())
    }
    #[cfg(not(target_arch = "x86_64"))]
    gemv_scalar(K, m, w, ws, act, out)
}

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    //! Zen 3, no VNNI. 64 MACs per nine instructions; four rows per tile
    //! because four accumulators, four activation vectors, two constants and
    //! temporaries is what fits in sixteen `ymm` without spilling — a
    //! hardware constant, not a tuning knob.
    use super::{Act, GROUP, TILE, WBIAS};
    use core::arch::x86_64::*;

    /// 64 MACs: one 32-byte load of packed weights against two 32-byte
    /// vectors of activations, accumulated into 8 i32 lanes. Range: nibbles
    /// in [0, 15], activations in [-127, 127]; a `maddubs` pair sum is within
    /// ±3810 and the sum of two within ±7620, inside i16. Widened to i32
    /// before anything else. No saturation path is relied on.
    #[inline(always)]
    unsafe fn dot64(w: *const u8, a_lo: __m256i, a_hi: __m256i, mask: __m256i, ones: __m256i, acc: __m256i) -> __m256i {
        let p = _mm256_loadu_si256(w as *const __m256i);
        let lo = _mm256_and_si256(p, mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(p), mask);
        let s = _mm256_add_epi16(_mm256_maddubs_epi16(lo, a_lo), _mm256_maddubs_epi16(hi, a_hi));
        _mm256_add_epi32(acc, _mm256_madd_epi16(s, ones))
    }

    /// [Σa, Σb, Σc, Σd] as four i32 lanes, six instructions.
    #[inline(always)]
    unsafe fn hsum4(a: __m256i, b: __m256i, c: __m256i, d: __m256i) -> __m128i {
        let ab = _mm256_hadd_epi32(a, b);
        let cd = _mm256_hadd_epi32(c, d);
        let x = _mm256_hadd_epi32(ab, cd);
        _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256::<1>(x))
    }

    /// `K` is const so the group count and strides are literals; `m` is a
    /// loop bound because one symbol serves gate_up (m = 2·D_FF) and down
    /// (m = D_MODEL).
    #[target_feature(enable = "avx2")]
    pub unsafe fn gemv<const K: usize>(m: usize, w: *const u8, ws: *const f32, act: &Act, out: *mut f32) {
        const { assert!(K % GROUP == 0) };
        let ngroups = K / GROUP;
        let mask = _mm256_set1_epi8(0x0F);
        let ones = _mm256_set1_epi16(1);
        let a = act.q.as_ptr();

        for t in 0..m / TILE {
            let wt = w.add(t * K * 2);
            let st = ws.add(t * ngroups * TILE);
            let mut y = _mm_setzero_ps();

            for g in 0..ngroups {
                let ag = a.add(g * GROUP);
                let a0 = _mm256_loadu_si256(ag as *const __m256i);
                let a1 = _mm256_loadu_si256(ag.add(32) as *const __m256i);
                let a2 = _mm256_loadu_si256(ag.add(64) as *const __m256i);
                let a3 = _mm256_loadu_si256(ag.add(96) as *const __m256i);

                let wg = wt.add(g * (GROUP / 2) * TILE);
                let mut c0 = _mm256_setzero_si256();
                let mut c1 = _mm256_setzero_si256();
                let mut c2 = _mm256_setzero_si256();
                let mut c3 = _mm256_setzero_si256();
                c0 = dot64(wg, a0, a1, mask, ones, c0);
                c0 = dot64(wg.add(32), a2, a3, mask, ones, c0);
                c1 = dot64(wg.add(64), a0, a1, mask, ones, c1);
                c1 = dot64(wg.add(96), a2, a3, mask, ones, c1);
                c2 = dot64(wg.add(128), a0, a1, mask, ones, c2);
                c2 = dot64(wg.add(160), a2, a3, mask, ones, c2);
                c3 = dot64(wg.add(192), a0, a1, mask, ones, c3);
                c3 = dot64(wg.add(224), a2, a3, mask, ones, c3);

                // Per-group flush: y += w_scale · a_scale · (Σ ŵ·a − 8·Σa)
                let sums = hsum4(c0, c1, c2, c3);
                let bias = _mm_set1_epi32(WBIAS * *act.gsum.get_unchecked(g));
                let s = _mm_cvtepi32_ps(_mm_sub_epi32(sums, bias));
                let sc = _mm_mul_ps(_mm_loadu_ps(st.add(g * TILE)), _mm_set1_ps(*act.scale.get_unchecked(g)));
                y = _mm_add_ps(y, _mm_mul_ps(s, sc));
            }
            _mm_storeu_ps(out.add(t * TILE), y);
        }
    }

    use super::GEMM_TOKENS as TOKENS;

    /// `gemv` over `n ≤ TOKENS` activations with the weights read, unpacked and rearranged once a
    /// pair of tiles and group, every token's dots run on the rearranged bytes. The rearrangement
    /// (ik_llama.cpp's `q4_0_r8` shape) puts the eight rows in the lanes: a 32-byte block holds
    /// one quad of k for each row, so `maddubs` against the token's quad broadcast to every lane
    /// gives per-row partials and the group's eight row sums arrive in one i32 vector — no
    /// horizontal sums. Per group a pair of tiles is four sub-blocks of 32 k (a chunk's low or
    /// high nibbles), each an 8×8 dword transpose of the unpacked rows (24 shuffles, amortized
    /// over the tokens), kept in a 1 KB scratch. Tokens go four at a time through a sub-block so
    /// its eight weight vectors are loaded once for four tokens (8 weights + 4 sums in registers).
    /// i16 sums over a sub-block stay exact (8 pair-products of at most 3810), widened to i32 a
    /// sub-block; the flush is `gemv`'s expression lane-wise, groups in order: `gemv`'s bits.
    #[target_feature(enable = "avx2")]
    pub unsafe fn gemm<'a, const K: usize>(m: usize, w: *const u8, ws: *const f32, n: usize, act: impl Fn(usize) -> Act<'a>, out: impl Fn(usize) -> *mut f32) {
        const { assert!(K % GROUP == 0) };
        assert!(n <= TOKENS && m % (2 * TILE) == 0, "gemm: {n} tokens, {m} rows (pairs of tiles)");
        if n == 1 {
            // the rearrangement pays from a few tokens on (one core, 7551P: 0.73× at one token, 1.2× at 16)
            return gemv::<K>(m, w, ws, &act(0), out(0));
        }
        let ngroups = K / GROUP;
        let mask = _mm256_set1_epi8(0x0F);
        // the rearranged group: sub-block (c, h) of 32 k, quad j of it, eight rows in the lanes
        let mut blk = [_mm256_setzero_si256(); 32];
        let mut y = [_mm256_setzero_ps(); TOKENS];
        let mut tp = 0;
        while tp < m / TILE {
            for i in 0..n {
                y[i] = _mm256_setzero_ps();
            }
            let (w0, w1) = (w.add(tp * K * 2), w.add((tp + 1) * K * 2));
            let (s0, s1) = (ws.add(tp * ngroups * TILE), ws.add((tp + 1) * ngroups * TILE));
            for g in 0..ngroups {
                let (g0, g1) = (w0.add(g * (GROUP / 2) * TILE), w1.add(g * (GROUP / 2) * TILE));
                for c in 0..2 {
                    // the eight rows' packed chunk c: low nibbles are k c·64.., high k c·64+32..
                    let mut lo = [_mm256_setzero_si256(); 8];
                    let mut hi = [_mm256_setzero_si256(); 8];
                    for r in 0..8 {
                        let base = if r < 4 { g0 } else { g1 };
                        let p = _mm256_loadu_si256(base.add((r % 4) * (GROUP / 2) + c * 32) as *const __m256i);
                        lo[r] = _mm256_and_si256(p, mask);
                        hi[r] = _mm256_and_si256(_mm256_srli_epi16::<4>(p), mask);
                    }
                    transpose8(&lo, &mut blk[(c * 2) * 8..(c * 2 + 1) * 8]);
                    transpose8(&hi, &mut blk[(c * 2 + 1) * 8..(c * 2 + 2) * 8]);
                }
                // the eight rows' scales for the group
                let sc_w = _mm256_set_m128(_mm_loadu_ps(s1.add(g * TILE)), _mm_loadu_ps(s0.add(g * TILE)));
                let mut i = 0;
                while i < n {
                    let nt = (n - i).min(4);
                    let acc = match nt {
                        4 => dots::<4>(&blk, g, &act, i),
                        3 => dots::<3>(&blk, g, &act, i),
                        2 => dots::<2>(&blk, g, &act, i),
                        _ => dots::<1>(&blk, g, &act, i),
                    };
                    for t in 0..nt {
                        let a = act(i + t);
                        let bias = _mm256_set1_epi32(WBIAS * *a.gsum.get_unchecked(g));
                        let sum = _mm256_cvtepi32_ps(_mm256_sub_epi32(acc[t], bias));
                        let sc = _mm256_mul_ps(sc_w, _mm256_set1_ps(*a.scale.get_unchecked(g)));
                        y[i + t] = _mm256_add_ps(y[i + t], _mm256_mul_ps(sum, sc));
                    }
                    i += nt;
                }
            }
            for i in 0..n {
                _mm256_storeu_ps(out(i).add(tp * TILE), y[i]);
            }
            tp += 2;
        }
    }

    /// `NT ≤ 4` tokens' integer dots over a rearranged group: for each sub-block its eight weight
    /// vectors once, each token's quads broadcast against them, i16 sums widened a sub-block.
    #[inline(always)]
    unsafe fn dots<'a, const NT: usize>(blk: &[__m256i; 32], g: usize, act: &impl Fn(usize) -> Act<'a>, i0: usize) -> [__m256i; 4] {
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_si256(); 4];
        let mut aq = [core::ptr::null::<i32>(); NT];
        for t in 0..NT {
            aq[t] = act(i0 + t).q.as_ptr().add(g * GROUP) as *const i32;
        }
        for sb in 0..4 {
            let wv = &blk[sb * 8..(sb + 1) * 8];
            let mut s16 = [_mm256_setzero_si256(); NT];
            for j in 0..8 {
                for t in 0..NT {
                    let a4 = _mm256_set1_epi32(*aq[t].add(sb * 8 + j));
                    s16[t] = _mm256_add_epi16(s16[t], _mm256_maddubs_epi16(wv[j], a4));
                }
            }
            for t in 0..NT {
                acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(s16[t], ones));
            }
        }
        acc
    }

    /// Eight rows of eight dwords into eight dwords of eight rows.
    #[inline(always)]
    unsafe fn transpose8(r: &[__m256i; 8], o: &mut [__m256i]) {
        let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
        let t1 = _mm256_unpackhi_epi32(r[0], r[1]);
        let t2 = _mm256_unpacklo_epi32(r[2], r[3]);
        let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
        let t4 = _mm256_unpacklo_epi32(r[4], r[5]);
        let t5 = _mm256_unpackhi_epi32(r[4], r[5]);
        let t6 = _mm256_unpacklo_epi32(r[6], r[7]);
        let t7 = _mm256_unpackhi_epi32(r[6], r[7]);
        let u0 = _mm256_unpacklo_epi64(t0, t2);
        let u1 = _mm256_unpackhi_epi64(t0, t2);
        let u2 = _mm256_unpacklo_epi64(t1, t3);
        let u3 = _mm256_unpackhi_epi64(t1, t3);
        let u4 = _mm256_unpacklo_epi64(t4, t6);
        let u5 = _mm256_unpackhi_epi64(t4, t6);
        let u6 = _mm256_unpacklo_epi64(t5, t7);
        let u7 = _mm256_unpackhi_epi64(t5, t7);
        o[0] = _mm256_permute2x128_si256::<0x20>(u0, u4);
        o[1] = _mm256_permute2x128_si256::<0x20>(u1, u5);
        o[2] = _mm256_permute2x128_si256::<0x20>(u2, u6);
        o[3] = _mm256_permute2x128_si256::<0x20>(u3, u7);
        o[4] = _mm256_permute2x128_si256::<0x31>(u0, u4);
        o[5] = _mm256_permute2x128_si256::<0x31>(u1, u5);
        o[6] = _mm256_permute2x128_si256::<0x31>(u2, u6);
        o[7] = _mm256_permute2x128_si256::<0x31>(u3, u7);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn gptq_with_identity_is_rtn() {
        use crate::testing::Rng;
        let (m, k) = (8, 2 * GROUP);
        let mut rng = Rng::new(11);
        let w0: std::vec::Vec<f32> = (0..m * k).map(|_| rng.unit()).collect();
        let mut u = std::vec![0f32; k * k];
        for i in 0..k {
            u[i * k + i] = 1.0;
        }
        let mut w = w0.clone();
        let (q, s) = gptq(m, k, &mut w, &u);
        for r in 0..m {
            for g in 0..k / GROUP {
                let seg = &w0[r * k + g * GROUP..r * k + (g + 1) * GROUP];
                let sc = seg.iter().fold(0f32, |a, v| a.max(v.abs())) / 7.0;
                assert_eq!(s[r * (k / GROUP) + g], sc);
                for (i, v) in seg.iter().enumerate() {
                    assert_eq!(q[r * k + g * GROUP + i], round(v * (1.0 / sc)).clamp(-8.0, 7.0) as i8);
                }
            }
        }
    }

    use super::*;
    use std::vec;
    use std::vec::Vec;

    use crate::testing::Rng;

    #[test]
    fn pack_roundtrip() {
        const K: usize = 256;
        const M: usize = 8;
        let mut rng = Rng::new(7);
        let w: Vec<i8> = (0..M * K).map(|_| rng.i4()).collect();
        let s: Vec<f32> = (0..M * K / GROUP).map(|i| i as f32).collect();
        let (pw, ps) = pack(M, K, &w, &s);
        for row in 0..M {
            for k in 0..K {
                assert_eq!(unpack(K, &pw, row, k), w[row * K + k]);
            }
            for g in 0..K / GROUP {
                assert_eq!(ps[scale_index(K, row, g)], s[row * (K / GROUP) + g]);
            }
        }
    }

    #[test]
    fn gemv_realizations_agree() {
        const K: usize = 1024;
        const M: usize = 64;
        let mut rng = Rng::new(0x9E3779B97F4A7C15);
        let w: Vec<i8> = (0..M * K).map(|_| rng.i4()).collect();
        let sc: Vec<f32> = (0..M * (K / GROUP)).map(|_| 0.01 + (rng.next() % 100) as f32 * 1e-4).collect();
        let x: Vec<f32> = (0..K).map(|_| rng.unit()).collect();
        let (pw, ps) = pack(M, K, &w, &sc);
        let (mut q, mut s, mut g) = (vec![0i8; K], vec![0f32; K / GROUP], vec![0i32; K / GROUP]);
        let mut act = QAct { q: &mut q, scale: &mut s, gsum: &mut g };
        act.quantize(&x);
        let mut a = vec![0f32; M];
        let mut b = vec![0f32; M];
        gemv_scalar(K, M, &pw, &ps, &act.act(), &mut a);
        gemv::<K>(M, &pw, &ps, &act.act(), &mut b);
        assert_eq!(a, b, "the AVX2 realization must be bit-identical to the scalar one");
    }

    #[test]
    fn gemm_is_gemv_for_every_token() {
        const K: usize = 1024;
        const M: usize = 64;
        const N: usize = 37;
        let mut rng = Rng::new(0x1234_5678_9ABC_DEF1);
        let w: Vec<i8> = (0..M * K).map(|_| rng.i4()).collect();
        let sc: Vec<f32> = (0..M * (K / GROUP)).map(|_| 0.01 + (rng.next() % 100) as f32 * 1e-4).collect();
        let (pw, ps) = pack(M, K, &w, &sc);
        let xs: Vec<Vec<f32>> = (0..N).map(|_| (0..K).map(|_| rng.unit()).collect()).collect();
        let (mut q, mut s, mut g) = (vec![0i8; N * K], vec![0f32; N * K / GROUP], vec![0i32; N * K / GROUP]);
        for i in 0..N {
            QAct { q: &mut q[i * K..(i + 1) * K], scale: &mut s[i * (K / GROUP)..(i + 1) * (K / GROUP)], gsum: &mut g[i * (K / GROUP)..(i + 1) * (K / GROUP)] }.quantize(&xs[i]);
        }
        let act = |i: usize| Act { q: &q[i * K..(i + 1) * K], scale: &s[i * (K / GROUP)..(i + 1) * (K / GROUP)], gsum: &g[i * (K / GROUP)..(i + 1) * (K / GROUP)] };
        let mut a = vec![0f32; N * M];
        let mut b = vec![0f32; N * M];
        for i in 0..N {
            gemv::<K>(M, &pw, &ps, &act(i), &mut a[i * M..(i + 1) * M]);
        }
        let bp = b.as_mut_ptr() as usize;
        gemm::<K>(M, &pw, &ps, N, act, |i| unsafe { (bp as *mut f32).add(i * M) });
        assert_eq!(a, b, "gemm must give gemv's bits for every token");
    }

    /// `cargo test --release -p common gemm_rate -- --ignored --nocapture`: one core, the expert
    /// shapes, gemv a token at a time against gemm over the tokens.
    #[test]
    #[ignore]
    fn gemm_rate() {
        // two weight shapes: 32 rows (64 KB, cache-resident: the kernel's own rate) and 16,384 rows
        // (32 MB of nibbles, past one core's share of L3: the rate when the rows stream from DRAM)
        gemm_rate_rows::<32>(2000);
        gemm_rate_rows::<16384>(8);
        gemm_rate_slabs();
    }
    /// The prefill expert phase's own shape: many 32-row slabs (an item's gate or up rows, 64 KB each) from a
    /// pool far past the caches, each applied to `n` tokens and then the next — the rate when every item
    /// starts a fresh stream, against the long slab above.
    fn gemm_rate_slabs() {
        const K: usize = 4096;
        const M: usize = 32;
        const SLABS: usize = 2048; // 128 MB of nibbles
        let mut rng = Rng::new(5);
        let w: Vec<i8> = (0..M * K).map(|_| rng.i4()).collect();
        let sc: Vec<f32> = (0..M * (K / GROUP)).map(|_| 0.01).collect();
        let (pw1, ps1) = pack(M, K, &w, &sc);
        let pw: Vec<u8> = (0..SLABS).flat_map(|_| pw1.iter().copied()).collect();
        let ps: Vec<f32> = (0..SLABS).flat_map(|_| ps1.iter().copied()).collect();
        std::println!("{SLABS} slabs of {M} rows x {K} ({:.1} MB), each applied to n tokens then the next:", pw.len() as f64 / 1e6);
        // one activation set reused by every slab, then a prefill step's shape: each slab's n tokens drawn from a
        // pool of 1024 quantized activations (4 MB: L3, not L1), every slab a different set
        const POOL: usize = 1024;
        let (mut q, mut s, mut g) = (vec![0i8; POOL * K], vec![0f32; POOL * K / GROUP], vec![0i32; POOL * K / GROUP]);
        for i in 0..POOL {
            let x: Vec<f32> = (0..K).map(|_| rng.unit()).collect();
            QAct { q: &mut q[i * K..(i + 1) * K], scale: &mut s[i * (K / GROUP)..(i + 1) * (K / GROUP)], gsum: &mut g[i * (K / GROUP)..(i + 1) * (K / GROUP)] }.quantize(&x);
        }
        let act_at = |t: usize| Act { q: &q[t * K..(t + 1) * K], scale: &s[t * (K / GROUP)..(t + 1) * (K / GROUP)], gsum: &g[t * (K / GROUP)..(t + 1) * (K / GROUP)] };
        for &(n, spread) in &[(16usize, false), (64, false), (16, true), (32, true), (64, true)] {
            let mut out = vec![0f32; n * M];
            let op = out.as_mut_ptr() as usize;
            let t0 = std::time::Instant::now();
            for sl in 0..SLABS {
                let base = if spread { (sl * 37 * n) % (POOL - n) } else { 0 };
                let act = |i: usize| act_at(base + i);
                gemm::<K>(M, &pw[sl * pw1.len()..(sl + 1) * pw1.len()], &ps[sl * ps1.len()..(sl + 1) * ps1.len()], n, act, |i| unsafe { (op as *mut f32).add(i * M) });
            }
            let b = t0.elapsed().as_secs_f64() / (SLABS * n) as f64;
            std::println!("n {n:>2}{}: gemm {:.2} µs a token a slab ({:.1} GMAC/s, {:.1} GB/s of weights)", if spread { " spread" } else { "       " }, b * 1e6, (M * K) as f64 / b / 1e9, pw1.len() as f64 / (b * n as f64) / 1e9);
        }
    }
    fn gemm_rate_rows<const M: usize>(reps_at_one: usize) {
        const K: usize = 4096;
        let mut rng = Rng::new(3);
        let w: Vec<i8> = (0..M * K).map(|_| rng.i4()).collect();
        let sc: Vec<f32> = (0..M * (K / GROUP)).map(|_| 0.01).collect();
        let (pw, ps) = pack(M, K, &w, &sc);
        std::println!("{M} rows x {K} ({:.1} MB packed):", (pw.len()) as f64 / 1e6);
        for &n in &[1usize, 4, 16, 64] {
            let (mut q, mut s, mut g) = (vec![0i8; n * K], vec![0f32; n * K / GROUP], vec![0i32; n * K / GROUP]);
            for i in 0..n {
                let x: Vec<f32> = (0..K).map(|_| rng.unit()).collect();
                QAct { q: &mut q[i * K..(i + 1) * K], scale: &mut s[i * (K / GROUP)..(i + 1) * (K / GROUP)], gsum: &mut g[i * (K / GROUP)..(i + 1) * (K / GROUP)] }.quantize(&x);
            }
            let act = |i: usize| Act { q: &q[i * K..(i + 1) * K], scale: &s[i * (K / GROUP)..(i + 1) * (K / GROUP)], gsum: &g[i * (K / GROUP)..(i + 1) * (K / GROUP)] };
            let mut out = vec![0f32; n * M];
            let reps = (reps_at_one / n).max(1);
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                for i in 0..n {
                    gemv::<K>(M, &pw, &ps, &act(i), &mut out[i * M..(i + 1) * M]);
                }
            }
            let a = t0.elapsed().as_secs_f64() / (reps * n) as f64;
            let op = out.as_mut_ptr() as usize;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                gemm::<K>(M, &pw, &ps, n, act, |i| unsafe { (op as *mut f32).add(i * M) });
            }
            let b = t0.elapsed().as_secs_f64() / (reps * n) as f64;
            std::println!("n {n:>2}: gemv {:.2} µs a token ({:.1} GMAC/s), gemm {:.2} µs ({:.1} GMAC/s), {:.2}×", a * 1e6, (M * K) as f64 / a / 1e9, b * 1e6, (M * K) as f64 / b / 1e9, a / b);
        }
    }
}

#[cfg(target_arch = "amdgpu")]
pub mod gfx11 {
    //! gfx11 (wave32, `v_dot4_i32_iu8`). One wave computes one 4-row tile. Lane `l` owns row
    //! `l / 8` and the groups `l % 8, l % 8 + 8, …`: for each it reads that
    //! (row, group) cell — 64 contiguous bytes, one cache line — and produces
    //! the group's flushed f32 term on its own, with no cross-lane traffic.
    //! The terms are then summed in group order by walking the row's lanes
    //! with `ds_bpermute`, once per tile. Same bytes as the AVX2 realization,
    //! the same integer sum per group, the same flush expression, the same
    //! order of f32 additions: same bits, whichever tier ran it.
    use super::{GROUP, TILE, WBIAS};
    use crate::gfx11::{lds, workgroup_sync};
    use crate::math::round;
    use core::arch::amdgpu::ds_bpermute;

    /// Bytes of launch-sized LDS one workgroup needs to stage a `K`-wide
    /// quantized activation: the i8 vector, then its per-group scales and Σq.
    pub const fn stage_bytes(k: usize) -> usize {
        k + 8 * (k / GROUP)
    }

    /// Copy one expert's quantized activation into LDS, cooperatively across
    /// the workgroup (`tid` in 0..256, 8 bytes each), and barrier. Returns the
    /// LDS pointers in the same shape `gemv_tile` takes, so a cell's 128
    /// activation reads become `ds_load`s instead of 16 global loads per word.
    #[inline(always)]
    pub unsafe fn stage_act<const K: usize>(tid: u32, q: *const i8, ascale: *const f32, gsum: *const i32) -> (*const i8, *const f32, *const i32) {
        const { assert!(K % 8 == 0 && K / 8 <= 256 && K % GROUP == 0) };
        let lds: *mut u8 = lds::<u8>();
        let t = tid as usize;
        if t < K / 8 {
            let v = core::ptr::read_unaligned(q.add(t * 8) as *const u64);
            core::ptr::write_unaligned(lds.add(t * 8) as *mut u64, v);
        }
        if t < K / GROUP {
            *(lds.add(K) as *mut f32).add(t) = *ascale.add(t);
            *(lds.add(K + 4 * (K / GROUP)) as *mut i32).add(t) = *gsum.add(t);
        }
        workgroup_sync();
        (lds as *const i8, lds.add(K) as *const f32, lds.add(K + 4 * (K / GROUP)) as *const i32)
    }

    /// Quantize a `K`-wide f32 vector into LDS as `QAct::quantize` does —
    /// per group of 128: absmax/127 scale, round-half-away, Σq — with each
    /// wave of the workgroup taking groups `wave, wave + 8, …` and each lane
    /// four elements, supplied by `load(i)` so a producer's epilogue (SwiGLU)
    /// can be folded in. Max and integer sum are order-free, and `round` is
    /// the shared one, so the bytes match the CPU's. Returns LDS pointers in
    /// the shape `gemv_tile` takes.
    #[inline(always)]
    pub unsafe fn quantize_to_lds<const K: usize>(tid: u32, load: impl Fn(usize) -> [f32; 4]) -> (*const i8, *const f32, *const i32) {
        const { assert!(K % GROUP == 0 && GROUP == 128) };
        let lds: *mut u8 = lds::<u8>();
        let scales = lds.add(K) as *mut f32;
        let sums = lds.add(K + 4 * (K / GROUP)) as *mut i32;
        let (wave, lane) = ((tid / 32) as usize, (tid % 32) as usize);
        let mut g = wave;
        while g < K / GROUP {
            let ([packed], [s], [sum]) = quantize_groups(lane as u32, [load(g * GROUP + lane * 4)]);
            *(lds.add(g * GROUP + lane * 4) as *mut u32) = packed;
            if lane == 0 {
                *scales.add(g) = s;
                *sums.add(g) = sum;
            }
            g += 8;
        }
        workgroup_sync();
        (lds as *const i8, scales as *const f32, sums as *const i32)
    }

    /// `N` groups quantized as `QAct::quantize`, a wave a group (lane l holding the group's values 4l..4l + 3):
    /// the scale amax / 127, each value rounded by the shared `round` and clamped to ±127, the bytes' sum. The `N`
    /// groups' butterflies run in lockstep, so the wave waits once a step for all of them. Returns each group's
    /// lane bytes packed (value 4l + i in byte i) and, on every lane, its scale and sum.
    #[inline(always)]
    pub unsafe fn quantize_groups<const N: usize>(lane: u32, v: [[f32; 4]; N]) -> ([u32; N], [f32; N], [i32; N]) {
        let mut amax = [0u32; N];
        let mut g = 0;
        while g < N {
            let mut m = 0f32;
            let mut i = 0;
            while i < 4 {
                let a = if v[g][i] < 0.0 { -v[g][i] } else { v[g][i] };
                if a > m {
                    m = a;
                }
                i += 1;
            }
            amax[g] = m.to_bits();
            g += 1;
        }
        let amax = crate::gfx11::wave_max_u32(lane, amax);
        let (mut packed, mut scale, mut sum) = ([0u32; N], [0f32; N], [0i32; N]);
        let mut g = 0;
        while g < N {
            let s = f32::from_bits(amax[g]) / 127.0;
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            let mut i = 0;
            while i < 4 {
                let r = round(v[g][i] * inv);
                let r = if r > 127.0 { 127.0 } else if r < -127.0 { -127.0 } else { r };
                let q = r as i8;
                packed[g] |= ((q as u8) as u32) << (8 * i);
                sum[g] += q as i32;
                i += 1;
            }
            scale[g] = s;
            g += 1;
        }
        (packed, scale, crate::gfx11::wave_sum_i32(lane, sum))
    }

    unsafe extern "C" {
        /// `v_dot4_i32_iu8`: four byte products with per-operand signedness,
        /// accumulated into i32. The biased nibbles are the unsigned operand,
        /// the activations the signed one — the same pairing `vpmaddubsw`
        /// makes on the CPU.
        #[link_name = "llvm.amdgcn.sudot4"]
        fn sudot4(a_signed: bool, a: u32, b_signed: bool, b: u32, c: i32, clamp: bool) -> i32;
    }

    /// 64 MACs on one 8-byte word of a row's 64-byte group cell: word `i`
    /// holds k = c·64 + p + j (low nibbles) and +32 (high), c = i / 4,
    /// p = (i % 4)·8. Four `sudot4`s: the low nibbles of bytes 0..3 and 4..7
    /// against q[k..k+4] and q[k+4..k+8], then the high nibbles against the
    /// +32 activations.
    #[inline(always)]
    unsafe fn word_dot(packed: u64, qg: *const i8, i: usize) -> i32 {
        let k_lo = (i / 4) * 64 + (i % 4) * 8;
        let qlo = core::ptr::read_unaligned(qg.add(k_lo) as *const u64);
        let qhi = core::ptr::read_unaligned(qg.add(k_lo + 32) as *const u64);
        const M: u32 = 0x0F0F_0F0F;
        let (w0, w1) = (packed as u32, (packed >> 32) as u32);
        let (q0, q1) = (qlo as u32, (qlo >> 32) as u32);
        let (q2, q3) = (qhi as u32, (qhi >> 32) as u32);
        let acc = sudot4(false, w0 & M, true, q0, 0, false);
        let acc = sudot4(false, w1 & M, true, q1, acc, false);
        let acc = sudot4(false, (w0 >> 4) & M, true, q2, acc, false);
        sudot4(false, (w1 >> 4) & M, true, q3, acc, false)
    }

    /// The most tokens one launch carries, and a wave's pass of them (`gemv::gfx11`).
    pub const TOKENS: usize = crate::gemv::gfx11::TOKENS;
    pub const PASS: usize = crate::gemv::gfx11::PASS;
    /// Bytes of one staged step (two groups) of one token's quantized
    /// activation: 256 bytes of i8, two scales, two Σq.
    pub const STEP_BYTES: usize = 2 * GROUP + 16;

    /// One step of `gemv_tile` for `n` tokens: the lane's 16 bytes of the
    /// tile's step once, then for each token the same word dots, quad sum and
    /// flush, its two group terms added to `y[i]` in group order. Steps in
    /// order give, per token, exactly `gemv_tile`'s bits. `slab` holds the
    /// step's activations as `stage_step` lays them out.
    #[inline(always)]
    pub unsafe fn gemv_tile_step<const K: usize, const NP: usize>(lane: u32, w_tile: *const u8, ws_tile: *const f32, st: usize, slab: *const u8, y: &mut [f32; PASS]) {
        const { assert!(K % GROUP == 0 && K / GROUP <= 32 && NP <= PASS) };
        const STEP: usize = 512;
        let ng = K / GROUP;
        let l = lane as usize;
        let r = (l % 16) / 4;
        let col = (l % 4) * 16;
        let gpar = l / 16;
        let i0 = col / 8;
        let g = 2 * st + gpar;
        let (mut w0, mut w1) = (0u64, 0u64);
        if g < ng {
            let p = w_tile.add(st * STEP + l * 16);
            w0 = core::ptr::read_unaligned(p as *const u64);
            w1 = core::ptr::read_unaligned(p.add(8) as *const u64);
        }
        // unrolled over NP with static indices and nothing predicated: `y` stays in registers
        let mut i = 0;
        while i < NP {
            let tok = slab.add(i * STEP_BYTES);
            let mut part = 0i32;
            if g < ng {
                let qg = (tok as *const i8).add(gpar * GROUP);
                part = word_dot(w0, qg, i0) + word_dot(w1, qg, i0 + 1);
            }
            let v = part as u32;
            let v = v.wrapping_add(ds_bpermute(((lane ^ 1) * 4) as u32, v));
            let v = v.wrapping_add(ds_bpermute(((lane ^ 2) * 4) as u32, v));
            let sum = v as i32;
            let mut term = 0f32;
            if g < ng {
                let sc = *ws_tile.add(g * TILE + r) * *(tok.add(2 * GROUP) as *const f32).add(gpar);
                let sv = (sum - WBIAS * *(tok.add(2 * GROUP + 8) as *const i32).add(gpar)) as f32;
                term = sv * sc;
            }
            let even = f32::from_bits(ds_bpermute(((4 * r) * 4) as u32, term.to_bits()));
            let odd = f32::from_bits(ds_bpermute(((16 + 4 * r) * 4) as u32, term.to_bits()));
            y[i] += even;
            if 2 * st + 1 < ng {
                y[i] += odd;
            }
            i += 1;
        }
    }

    /// One wave, one 4-row tile. The tile's bytes are read in 512-byte
    /// wave-contiguous steps (lane `l` takes bytes `l·16..l·16+16` of each
    /// step), so every load instruction is one coalesced request. A step
    /// covers two groups; a lane's 16 bytes are a quarter of one (row, group)
    /// cell, and the four lanes of a quad share the cell — two cross-lane
    /// adds complete the group's exact i32 sum. Each group's flush term is
    /// then gathered in group order, once per tile.
    #[inline(always)]
    pub unsafe fn gemv_tile<const K: usize>(lane: u32, w_tile: *const u8, ws_tile: *const f32, q: *const i8, ascale: *const f32, gsum: *const i32, out: *mut f32) {
        const { assert!(K % GROUP == 0 && K / GROUP <= 32) };
        const STEP: usize = 512; // bytes per wave step = 2 groups × 4 rows × 64
        const NSTEP_MAX: usize = 16; // K/GROUP ≤ 32 groups → ≤ 16 steps
        let ng = K / GROUP;
        let nstep = (ng + 1) / 2;
        let l = lane as usize;
        let r = (l % 16) / 4; // this lane's row, fixed across steps
        let col = (l % 4) * 16; // this lane's 16 bytes within the row's 64
        let gpar = l / 16; // group parity within a step
        let i0 = col / 8; // the two 8-byte words this lane holds are words i0, i0+1 of the cell

        // per step: the lane's quarter-cell partial, the quad sum, the flush term —
        // then the step's two groups are added to the row's running sum in group
        // order (group g's term sits in lane 16·(g%2) + 4·r), so nothing is kept
        // across steps but `y`.
        let mut y = 0f32;
        let mut st = 0;
        while st < NSTEP_MAX {
            if st < nstep {
                let g = 2 * st + gpar;
                let mut part = 0i32;
                if g < ng {
                    let p = w_tile.add(st * STEP + l * 16);
                    let w0 = core::ptr::read_unaligned(p as *const u64);
                    let w1 = core::ptr::read_unaligned(p.add(8) as *const u64);
                    let qg = q.add(g * GROUP);
                    part = word_dot(w0, qg, i0) + word_dot(w1, qg, i0 + 1);
                }
                // quad sum: lanes 4k..4k+3 hold the same (row, group)
                let v = part as u32;
                let v = v.wrapping_add(ds_bpermute(((lane ^ 1) * 4) as u32, v));
                let v = v.wrapping_add(ds_bpermute(((lane ^ 2) * 4) as u32, v));
                let sum = v as i32;
                let mut term = 0f32;
                if g < ng {
                    // the flush, the CPU's expression exactly: cvt(Σ − 8·Σa) · (w_scale · a_scale)
                    let sc = *ws_tile.add(g * TILE + r) * *ascale.add(g);
                    let sv = (sum - WBIAS * *gsum.add(g)) as f32;
                    term = sv * sc;
                }
                // y = ((t0 + t1) + t2) + … : this step's even group, then its odd group
                let even = f32::from_bits(ds_bpermute(((4 * r) * 4) as u32, term.to_bits()));
                let odd = f32::from_bits(ds_bpermute(((16 + 4 * r) * 4) as u32, term.to_bits()));
                y += even;
                if 2 * st + 1 < ng {
                    y += odd;
                }
            }
            st += 1;
        }
        if l < 16 && l % 4 == 0 {
            *out.add(r) = y;
        }
    }
}
