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
    use core::arch::amdgpu::{ds_bpermute, wave_reduce_add, wave_reduce_umax};

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
            let v = load(g * GROUP + lane * 4);
            let mut amax = 0f32;
            let mut i = 0;
            while i < 4 {
                let a = if v[i] < 0.0 { -v[i] } else { v[i] };
                if a > amax {
                    amax = a;
                }
                i += 1;
            }
            let amax = f32::from_bits(wave_reduce_umax::<0>(amax.to_bits()));
            let s = amax / 127.0;
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            let mut packed = 0u32;
            let mut sum = 0i32;
            let mut i = 0;
            while i < 4 {
                let r = round(v[i] * inv);
                let r = if r > 127.0 { 127.0 } else if r < -127.0 { -127.0 } else { r };
                let q = r as i8;
                packed |= ((q as u8) as u32) << (8 * i);
                sum += q as i32;
                i += 1;
            }
            let sum = wave_reduce_add::<0>(sum as u32) as i32;
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
