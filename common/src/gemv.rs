//! `y[m] = Σ_k W[m][k] · x[k]` with bf16 weights, row-major `[M][K]`, f32
//! activations. The inner loop of every dense projection; the unpack is a
//! shift. `bf16` picks the fast realization for the target; `bf16_scalar`
//! is always there to diff against.
//!
//! `fp8` is the same over FP8 e4m3 weights with one f32 scale per 128×128
//! block, the format GLM-5.3-Flash's checkpoint stores its dense tensors in
//! and the bundle ships them in. Every e4m3 value is exact in f32, so the
//! unpack is bit arithmetic; a block's products are summed in eight lanes,
//! then the lanes scaled once. The AVX2 realization fuses each multiply-add
//! and the scalar twin (no `mul_add` without std) rounds twice, so the two
//! agree to a few ulp, not bit for bit — as the bf16 pair do.

#[inline(always)]
pub fn widen(w: u16) -> f32 {
    f32::from_bits((w as u32) << 16)
}

#[inline(always)]
pub fn bf16_scalar<const M: usize, const K: usize>(w: &[u16], x: &[f32], out: &mut [f32]) {
    bf16_scalar_rows::<K>(w, x, out, 0, M)
}
/// Rows `m0..m1` only, into `out[m0..m1]`.
#[inline(always)]
pub fn bf16_scalar_rows<const K: usize>(w: &[u16], x: &[f32], out: &mut [f32], m0: usize, m1: usize) {
    for m in m0..m1 {
        let row = &w[m * K..(m + 1) * K];
        let mut acc = 0f32;
        for k in 0..K {
            acc += widen(row[k]) * x[k];
        }
        out[m] = acc;
    }
}

/// The realization this target ships. AVX2+FMA presence is asserted once at
/// construction by the harness, never probed here.
#[inline(always)]
pub fn bf16<const M: usize, const K: usize>(w: &[u16], x: &[f32], out: &mut [f32]) {
    debug_assert!(w.len() >= M * K && x.len() >= K && out.len() >= M);
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avx2::bf16_rows::<K>(w.as_ptr(), x.as_ptr(), out.as_mut_ptr(), 0, M)
    }
    #[cfg(not(target_arch = "x86_64"))]
    bf16_scalar::<M, K>(w, x, out)
}

/// `bf16` with the rows split over `pool` in `pool::ROWS`-row items: each item
/// writes its own rows of `out`, so the result is the serial one exactly.
#[inline(always)]
pub fn bf16_par<const M: usize, const K: usize>(w: &[u16], x: &[f32], out: &mut [f32], pool: &dyn super::pool::Pool) {
    bf16_par_r::<M, K, { super::pool::ROWS }>(w, x, out, pool)
}

/// `bf16_par` with `R` rows an item: for a few very wide rows (the
/// hyper-connections' 24 rows of 40 KB), one row an item puts a core on each.
#[inline(always)]
pub fn bf16_par_r<const M: usize, const K: usize, const R: usize>(w: &[u16], x: &[f32], out: &mut [f32], pool: &dyn super::pool::Pool) {
    debug_assert!(w.len() >= M * K && x.len() >= K && out.len() >= M);
    let (wp, xp, op) = (w.as_ptr() as usize, x.as_ptr() as usize, out.as_mut_ptr() as usize);
    pool.run_at(M.div_ceil(R), &move |i| (wp + i * R * K * 2) as *const u8, &move |i| {
        let (m0, m1) = (i * R, ((i + 1) * R).min(M));
        #[cfg(target_arch = "x86_64")]
        unsafe {
            avx2::bf16_rows::<K>(wp as *const u16, xp as *const f32, op as *mut f32, m0, m1)
        }
        #[cfg(not(target_arch = "x86_64"))]
        unsafe {
            bf16_scalar_rows::<K>(core::slice::from_raw_parts(wp as *const u16, M * K), core::slice::from_raw_parts(xp as *const f32, K), core::slice::from_raw_parts_mut(op as *mut f32, M), m0, m1)
        }
    });
}

/// FP8 e4m3fn → f32, exact: 1 sign, 4 exponent bits biased by 7, 3 mantissa
/// bits; no infinities (0x7f and 0xff are NaN, which weights never hold).
#[inline(always)]
pub fn e4m3(b: u8) -> f32 {
    let (s, e, m) = ((b as u32) >> 7, ((b as u32) >> 3) & 0xf, (b as u32) & 7);
    if e == 0 {
        // subnormal: m · 2⁻⁹
        let v = m as f32 * (1.0 / 512.0);
        return if s == 1 { -v } else { v };
    }
    f32::from_bits((s << 31) | ((e + 120) << 23) | (m << 20))
}

/// `e4m3` without a branch, for the card: a subnormal is built one binade up
/// (`2⁻⁶ · (1 + m/8)`) and its leading `2⁻⁶` taken back, which is `m · 2⁻⁹`
/// exactly; the sign goes on last. Bit-identical to `e4m3`.
#[inline(always)]
pub fn e4m3_sel(b: u8) -> f32 {
    let (s, e, m) = ((b as u32) >> 7, ((b as u32) >> 3) & 0xf, (b as u32) & 7);
    let z = (e == 0) as u32;
    let v = f32::from_bits(((e + 120 + z) << 23) | (m << 20)) - z as f32 * 0.015625;
    f32::from_bits(v.to_bits() | (s << 31))
}

/// Side of the FP8 scale blocks.
pub const FP8_BLOCK: usize = 128;

/// `scales[(m / 128) · (K / 128) + k / 128]`: the checkpoint's `<name>_scale_inv`.
#[inline(always)]
pub fn fp8_scalar<const M: usize, const K: usize>(w: &[u8], scales: &[f32], x: &[f32], out: &mut [f32]) {
    fp8_scalar_rows::<K>(w, scales, x, out, 0, M)
}
#[inline(always)]
pub fn fp8_scalar_rows<const K: usize>(w: &[u8], scales: &[f32], x: &[f32], out: &mut [f32], m0: usize, m1: usize) {
    const { assert!(K % FP8_BLOCK == 0) };
    let kb = K / FP8_BLOCK;
    for m in m0..m1 {
        let row = &w[m * K..(m + 1) * K];
        let mut lanes = [0f32; 8];
        for g in 0..kb {
            let mut acc = [0f32; 8];
            for i in 0..FP8_BLOCK / 8 {
                for j in 0..8 {
                    let k = g * FP8_BLOCK + i * 8 + j;
                    acc[j] += e4m3(row[k]) * x[k];
                }
            }
            let sc = scales[(m / FP8_BLOCK) * kb + g];
            for j in 0..8 {
                lanes[j] += acc[j] * sc;
            }
        }
        out[m] = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6])) + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
    }
}

/// The realization this target ships.
#[inline(always)]
pub fn fp8<const M: usize, const K: usize>(w: &[u8], scales: &[f32], x: &[f32], out: &mut [f32]) {
    debug_assert!(w.len() >= M * K && scales.len() >= M.div_ceil(FP8_BLOCK) * (K / FP8_BLOCK) && x.len() >= K && out.len() >= M);
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avx2::fp8_rows::<K>(w.as_ptr(), scales.as_ptr(), x.as_ptr(), out.as_mut_ptr(), 0, M)
    }
    #[cfg(not(target_arch = "x86_64"))]
    fp8_scalar::<M, K>(w, scales, x, out)
}

/// `fp8` with the rows split over `pool`, as `bf16_par`.
#[inline(always)]
pub fn fp8_par<const M: usize, const K: usize>(w: &[u8], scales: &[f32], x: &[f32], out: &mut [f32], pool: &dyn super::pool::Pool) {
    debug_assert!(w.len() >= M * K && x.len() >= K && out.len() >= M);
    let (wp, sp, xp, op) = (w.as_ptr() as usize, scales.as_ptr() as usize, x.as_ptr() as usize, out.as_mut_ptr() as usize);
    pool.run_at(M.div_ceil(super::pool::ROWS), &move |i| (wp + i * super::pool::ROWS * K) as *const u8, &move |i| {
        let (m0, m1) = (i * super::pool::ROWS, ((i + 1) * super::pool::ROWS).min(M));
        #[cfg(target_arch = "x86_64")]
        unsafe {
            avx2::fp8_rows::<K>(wp as *const u8, sp as *const f32, xp as *const f32, op as *mut f32, m0, m1)
        }
        #[cfg(not(target_arch = "x86_64"))]
        unsafe {
            fp8_scalar_rows::<K>(core::slice::from_raw_parts(wp as *const u8, M * K), core::slice::from_raw_parts(sp as *const f32, M.div_ceil(FP8_BLOCK) * (K / FP8_BLOCK)), core::slice::from_raw_parts(xp as *const f32, K), core::slice::from_raw_parts_mut(op as *mut f32, M), m0, m1)
        }
    });
}

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    //! Sixteen bf16 per load, widened by zero-extend and a 16-bit shift,
    //! four FMA chains per row. Bandwidth-bound.
    use core::arch::x86_64::*;

    #[inline(always)]
    pub unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
        _mm_cvtss_f32(s)
    }

    #[inline(always)]
    unsafe fn widen8(p: __m128i) -> __m256 {
        _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(p)))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn bf16<const M: usize, const K: usize>(w: *const u16, x: *const f32, out: *mut f32) {
        bf16_rows::<K>(w, x, out, 0, M)
    }
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn bf16_rows<const K: usize>(w: *const u16, x: *const f32, out: *mut f32, m0: usize, m1: usize) {
        const { assert!(K % 32 == 0) };
        for m in m0..m1 {
            let row = w.add(m * K);
            let mut a0 = _mm256_setzero_ps();
            let mut a1 = _mm256_setzero_ps();
            let mut a2 = _mm256_setzero_ps();
            let mut a3 = _mm256_setzero_ps();
            let mut k = 0;
            while k < K {
                let p = _mm256_loadu_si256(row.add(k) as *const __m256i);
                let q = _mm256_loadu_si256(row.add(k + 16) as *const __m256i);
                a0 = _mm256_fmadd_ps(widen8(_mm256_castsi256_si128(p)), _mm256_loadu_ps(x.add(k)), a0);
                a1 = _mm256_fmadd_ps(widen8(_mm256_extracti128_si256::<1>(p)), _mm256_loadu_ps(x.add(k + 8)), a1);
                a2 = _mm256_fmadd_ps(widen8(_mm256_castsi256_si128(q)), _mm256_loadu_ps(x.add(k + 16)), a2);
                a3 = _mm256_fmadd_ps(widen8(_mm256_extracti128_si256::<1>(q)), _mm256_loadu_ps(x.add(k + 24)), a3);
                k += 32;
            }
            *out.add(m) = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        }
    }

    /// Eight e4m3 bytes → eight f32, exactly (`super::e4m3`).
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
    pub unsafe fn fp8<const M: usize, const K: usize>(w: *const u8, scales: *const f32, x: *const f32, out: *mut f32) {
        fp8_rows::<K>(w, scales, x, out, 0, M)
    }
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn fp8_rows<const K: usize>(w: *const u8, scales: *const f32, x: *const f32, out: *mut f32, m0: usize, m1: usize) {
        const { assert!(K % super::FP8_BLOCK == 0) };
        let kb = K / super::FP8_BLOCK;
        for m in m0..m1 {
            let row = w.add(m * K);
            let srow = scales.add((m / super::FP8_BLOCK) * kb);
            let mut lanes = _mm256_setzero_ps();
            for g in 0..kb {
                let mut acc = _mm256_setzero_ps();
                let mut k = g * super::FP8_BLOCK;
                let end = k + super::FP8_BLOCK;
                while k < end {
                    acc = _mm256_fmadd_ps(e4m3x8(row.add(k)), _mm256_loadu_ps(x.add(k)), acc);
                    k += 8;
                }
                lanes = _mm256_fmadd_ps(acc, _mm256_set1_ps(*srow.add(g)), lanes);
            }
            *out.add(m) = hsum(lanes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    #[test]
    fn e4m3_sel_matches() {
        for b in 0..=255u8 {
            if b & 0x7f == 0x7f {
                continue; // NaN, never a weight
            }
            assert_eq!(e4m3(b).to_bits(), e4m3_sel(b).to_bits(), "{b:#x}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_matches_scalar() {
        const M: usize = 37;
        const K: usize = 2048;
        let mut rng = crate::testing::Rng::new(0x1234_5678_9abc_def1);
        let w: Vec<u16> = (0..M * K).map(|_| (rng.unit().to_bits() >> 16) as u16).collect();
        let x: Vec<f32> = (0..K).map(|_| rng.unit()).collect();
        let mut a = std::vec![0f32; M];
        let mut b = std::vec![0f32; M];
        bf16_scalar::<M, K>(&w, &x, &mut a);
        bf16::<M, K>(&w, &x, &mut b);
        for i in 0..M {
            assert!((a[i] - b[i]).abs() <= 1e-4 * a[i].abs().max(1.0), "{i}: {} vs {}", a[i], b[i]);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn par_matches_serial_bitwise() {
        const M: usize = 37 + 128;
        const K: usize = 384;
        let mut rng = crate::testing::Rng::new(0x9a11_e1e1_0000_0001);
        let wb: Vec<u16> = (0..M * K).map(|_| (rng.unit().to_bits() >> 16) as u16).collect();
        let w8: Vec<u8> = (0..M * K).map(|_| (rng.next() & 0x7e) as u8).collect();
        let scales: Vec<f32> = (0..M.div_ceil(FP8_BLOCK) * (K / FP8_BLOCK)).map(|_| rng.unit() * 1e-3).collect();
        let x: Vec<f32> = (0..K).map(|_| rng.unit()).collect();
        let (mut a, mut b) = (std::vec![0f32; M], std::vec![0f32; M]);
        bf16::<M, K>(&wb, &x, &mut a);
        bf16_par::<M, K>(&wb, &x, &mut b, &crate::pool::Serial);
        assert_eq!(a, b);
        fp8::<M, K>(&w8, &scales, &x, &mut a);
        fp8_par::<M, K>(&w8, &scales, &x, &mut b, &crate::pool::Serial);
        assert_eq!(a, b);
    }

    #[test]
    fn e4m3_values() {
        assert_eq!(e4m3(0x00), 0.0);
        assert_eq!(e4m3(0x38), 1.0); // e=7, m=0
        assert_eq!(e4m3(0x7e), 448.0); // the format's maximum: e=15, m=6
        assert_eq!(e4m3(0x01), 1.0 / 512.0); // smallest subnormal
        assert_eq!(e4m3(0x07), 7.0 / 512.0);
        assert_eq!(e4m3(0x08), 1.0 / 64.0); // smallest normal: 2⁻⁶
        assert_eq!(e4m3(0xb8), -1.0);
        assert_eq!(e4m3(0x81), -1.0 / 512.0);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn fp8_avx2_matches_scalar() {
        const M: usize = 37 + 128;
        const K: usize = 384;
        let mut rng = crate::testing::Rng::new(0x5eed_f8f8_0000_0001);
        let w: Vec<u8> = (0..M * K).map(|_| (rng.next() & 0xff) as u8).map(|b| if b & 0x7f == 0x7f { b & 0x7e } else { b }).collect();
        let scales: Vec<f32> = (0..M.div_ceil(FP8_BLOCK) * (K / FP8_BLOCK)).map(|_| rng.unit() * 0.01).collect();
        let x: Vec<f32> = (0..K).map(|_| rng.unit()).collect();
        let mut a = std::vec![0f32; M];
        let mut b = std::vec![0f32; M];
        fp8_scalar::<M, K>(&w, &scales, &x, &mut a);
        fp8::<M, K>(&w, &scales, &x, &mut b);
        for i in 0..M {
            assert!((a[i] - b[i]).abs() <= 1e-4 * a[i].abs().max(1.0), "row {i}: {} vs {}", a[i], b[i]);
        }
    }
}

/// Launch geometry of the gfx11 rows kernels, visible to hosts that launch them.
/// The most tokens one launch carries: sized by LDS, a step of 256 f32 per
/// token staged for the workgroup's rows is `TOKENS · 1 KB`, and the slab
/// holds `TOKENS` token-steps however they split between tokens and steps.
pub const TOKENS: usize = 32;
/// Tokens a wave accumulates at once: its weights are read once per pass of
/// PASS tokens (again from L2 for the next pass), and PASS accumulators keep
/// the registers few enough for the memory-bound kernels' occupancy (32
/// accumulators measured 30 % slower on one token, 2026-09-19).
pub const PASS: usize = 8;
/// The width a pass of a few tokens (two to FEW: a speculative verify step's) runs at, with
/// four-step rounds and the kernel's R rows a wave: at PASS with two-step rounds a two-token
/// step's rows kernels ran ~2× one token's (rows_h 199 µs a launch against ~90, timeline
/// 2026-10-17); one or two rows a wave for this width measured worse for the FP8 kernels
/// (ffn_down_shared 1277 / 1208 µs a step against 994 at four), the same for bf16.
pub const FEW: usize = 4;
/// Steps per round at most: a round is unrolled with every load hoisted, so
/// its depth is the registers it takes — forty-three steps of a 10,944-wide
/// row put 190 registers and scratch in a kernel that needs 23 (2026-09-19).
/// Eight is the old row primitive's depth; a compiler fence between steps did
/// not hold the hoisting back.
pub const FLIGHT: usize = 8;
/// Floats per staged token-step (`gfx11::STEP`): 32 lanes at a stride of 10.
pub const STEP: usize = 320;
/// Rows a wave takes in a pass of many tokens: each staged activation read from LDS feeds this
/// many FMAs (a row a wave read one per FMA: the rows kernels ran LDS-bound at ~0.5 TFLOPS over
/// 32 tokens, 830 µs a launch against 125 for one token, timeline 2026-10-17). One token keeps a
/// row a wave: its kernels stream weights and want the waves.
pub const ROWS: usize = 4;
/// Rows a workgroup takes for a launch over `n` tokens: eight waves, `r` rows each past one token
/// (`ROWS`, or a kernel's own fewer where four spill: the FP8 gate/up at 192 registers).
pub const fn rows_per_wg(n: usize, r: usize) -> usize {
    8 * if n <= 1 { 1 } else { r }
}

/// What the host needs of the card's rows kernels: the FP8 decode table's size (floats of LDS).
pub mod gfx11_lds {
    pub const FP8_LDS: usize = 256;
}

#[cfg(target_arch = "amdgpu")]
pub mod gfx11 {
    //! The bf16 GEMV row on gfx11: a wave reads one row in 512-byte
    //! contiguous steps (lane `l` takes elements `l·8..l·8+8` of each
    //! 256-element step), dots them against an f32 vector — usually staged in
    //! LDS by the workgroup — and reduces across the wave.
    use crate::gfx11::wave_sum;

    /// `y = Σ_k W[k] · x[k]` for one `K`-wide bf16 row (`K % 8 == 0`; a
    /// last partial step is masked per lane); every lane returns the sum.
    #[inline(always)]
    /// Eight consecutive bf16 at `p` (16-byte aligned) widened, in one wide load: a u16 loop the
    /// compiler would read a halfword at a time (2026-10-18: the attend's and scorer's rows).
    #[inline(always)]
    pub unsafe fn widen8_at(p: *const u16) -> [f32; 8] {
        let q = p as *const u64;
        let (a, b) = (*q, *q.add(1));
        [
            super::widen(a as u16),
            super::widen((a >> 16) as u16),
            super::widen((a >> 32) as u16),
            super::widen((a >> 48) as u16),
            super::widen(b as u16),
            super::widen((b >> 16) as u16),
            super::widen((b >> 32) as u16),
            super::widen((b >> 48) as u16),
        ]
    }
    pub unsafe fn bf16_row<const K: usize>(lane: u32, w: *const u16, x: *const f32) -> f32 {
        const { assert!(K % 8 == 0) };
        let l = lane as usize;
        let mut acc = 0f32;
        let mut s = 0;
        while s < K.div_ceil(256) {
            let base = s * 256 + l * 8;
            if base < K {
                let p = w.add(base) as *const u64;
                let (a, b) = (core::ptr::read_unaligned(p), core::ptr::read_unaligned(p.add(1)));
                let mut j = 0;
                while j < 4 {
                    acc += super::widen((a >> (16 * j)) as u16) * *x.add(base + j);
                    acc += super::widen((b >> (16 * j)) as u16) * *x.add(base + 4 + j);
                    j += 1;
                }
            }
            s += 1;
        }
        wave_sum(lane, acc)
    }


    /// Steps staged per round for a pass at width `np` over `steps` steps:
    /// FLIGHT at most, TOKENS / np at most (the slab), the whole matrix if
    /// shorter. A constant of the pass, so the step loops have constant trip
    /// counts and unroll with their loads hoisted — a runtime bound measured
    /// as every weight load waited on in turn, twice the time (2026-09-19).
    #[inline(always)]
    pub const fn round(np: usize, steps: usize) -> usize {
        // a pass of PASS tokens: two steps a round — a round's LDS reads unroll and hoist
        // per token per step, and four steps of eight tokens spilled (2026-09-19); a pass of
        // FEW tokens (a verify step's) takes four, the slab the same 16 token-steps
        let r = if np == 1 { FLIGHT } else if np <= FEW { 4 } else { 2 };
        if r < steps { r } else { steps }
    }

    /// A staged step in LDS: lane `l`'s eight elements of the step at
    /// `l · STRIDE`, ten floats apart, so a wave's reads of them are
    /// conflict-free across the 32 banks (at eight apart, lanes 0, 4, 8, …
    /// share banks: an 8-way conflict that cost the multi-token rows kernels
    /// 20× their weight-streaming time, 2026-09-19). `STEP` floats per token-step.
    pub const STRIDE: usize = 10;
    pub const STEP: usize = super::STEP;
    const _: () = assert!(STEP == 32 * STRIDE);
    /// Where element `t` of a step (0..256) sits in a staged token-step.
    #[inline(always)]
    pub const fn slot(t: usize) -> usize {
        (t / 8) * STRIDE + t % 8
    }

    /// Stage step `s` (elements `s·256..s·256+256`) of `n` activation vectors
    /// `x[i]` (each `K` wide, stride `K`) into LDS at `slab` as `[n][STEP]`,
    /// cooperatively over the 256 threads; the caller barriers after its
    /// round. Elements past `K` are not read (the row primitive masks them).
    #[inline(always)]
    pub unsafe fn stage_step<const K: usize>(tid: u32, x: *const f32, n: usize, s: usize, slab: *mut f32) {
        let t = tid as usize;
        let base = s * 256;
        if base + t < K {
            let mut i = 0;
            while i < n {
                *slab.add(i * STEP + slot(t)) = *x.add(i * K + base + t);
                i += 1;
            }
        }
    }

    /// One step of `bf16_row` for `NP` tokens: the lane's 8 weights of step
    /// `s` once, then for each token the same 8 products in the same order
    /// into `acc[i]`, from the step's activations staged at `slab`
    /// (`[NP][STEP]`). Summing the steps in order and `wave_sum`ing at the
    /// end gives, per token, exactly `bf16_row`'s bits: the one-token path is
    /// this with NP = 1. `NP` is a constant so the token loop unrolls with
    /// nothing predicated: a runtime count measured as every block executed
    /// under masks, twice the time on one token (2026-09-19). `w` is the row.
    #[inline(always)]
    pub unsafe fn bf16_row_step<const K: usize, const NP: usize, const R: usize>(lane: u32, w: &[*const u16; R], s: usize, slab: *const f32, acc: &mut [[f32; PASS]; R]) {
        const { assert!(K % 8 == 0 && NP <= PASS) };
        let l = lane as usize;
        let base = s * 256 + l * 8;
        if base < K {
            // each row's eight weights widened once, not once per token
            let mut wa = [[0f32; 4]; R];
            let mut wb = [[0f32; 4]; R];
            let mut r = 0;
            while r < R {
                let p = w[r].add(base) as *const u64;
                let (a, b) = (core::ptr::read_unaligned(p), core::ptr::read_unaligned(p.add(1)));
                let mut j = 0;
                while j < 4 {
                    wa[r][j] = super::widen((a >> (16 * j)) as u16);
                    wb[r][j] = super::widen((b >> (16 * j)) as u16);
                    j += 1;
                }
                r += 1;
            }
            // a staged element read once for the R rows; each (row, token)'s products in the one-row order
            let mut i = 0;
            while i < NP {
                let x = slab.add(i * STEP + l * STRIDE);
                let mut j = 0;
                while j < 4 {
                    let (xa, xb) = (*x.add(j), *x.add(4 + j));
                    let mut r = 0;
                    while r < R {
                        acc[r][i] += wa[r][j] * xa;
                        acc[r][i] += wb[r][j] * xb;
                        r += 1;
                    }
                    j += 1;
                }
                i += 1;
            }
        }
    }

    /// A rows kernel's weight format: a row of a `[m][K]` matrix, and a lane's
    /// eight weights of one of its steps against each token's staged activations.
    pub trait Rows<const K: usize>: Copy {
        type Row: Copy;
        /// Floats of LDS the format keeps past the skeleton's (a decode table), filled by `setup`.
        const LDS: usize;
        /// Once per workgroup, before the first round (the round's barrier publishes it).
        unsafe fn setup(tid: u32, lds: *mut f32);
        unsafe fn row(self, row: usize) -> Self::Row;
        /// The wave's `R` rows' step `s` against the pass's `NP` staged tokens.
        unsafe fn step<const NP: usize, const R: usize>(lane: u32, r: &[Self::Row; R], s: usize, slab: *const f32, lds: *const f32, acc: &mut [[f32; PASS]; R]);
    }
    /// bf16 rows: `bf16_row_step`.
    #[derive(Clone, Copy)]
    pub struct Bf16(pub *const u16);
    impl<const K: usize> Rows<K> for Bf16 {
        type Row = *const u16;
        const LDS: usize = 0;
        #[inline(always)]
        unsafe fn setup(_: u32, _: *mut f32) {}
        #[inline(always)]
        unsafe fn row(self, row: usize) -> *const u16 {
            self.0.add(row * K)
        }
        #[inline(always)]
        unsafe fn step<const NP: usize, const R: usize>(lane: u32, r: &[*const u16; R], s: usize, slab: *const f32, _: *const f32, acc: &mut [[f32; PASS]; R]) {
            bf16_row_step::<K, NP, R>(lane, r, s, slab, acc)
        }
    }
    /// FP8 e4m3 rows with an f32 scale per `FP8_BLOCK`-square block (`super::fp8`'s
    /// layout): a row is its bytes and its block row of scales. The decode is a
    /// an f16 bit transplant in registers (`fp8_row_step`; the 256-entry LDS table of 2026-10-15
    /// is set up still and unused — removed with its LDS when the next kernel change lands).
    #[derive(Clone, Copy)]
    pub struct Fp8 {
        pub w: *const u8,
        pub s: *const f32,
    }
    pub use super::gfx11_lds::FP8_LDS;
    impl<const K: usize> Rows<K> for Fp8 {
        type Row = (*const u8, *const f32);
        const LDS: usize = FP8_LDS;
        #[inline(always)]
        unsafe fn setup(tid: u32, lds: *mut f32) {
            *lds.add(tid as usize) = super::e4m3(tid as u8);
        }
        #[inline(always)]
        unsafe fn row(self, row: usize) -> (*const u8, *const f32) {
            const { assert!(K % super::FP8_BLOCK == 0) };
            (self.w.add(row * K), self.s.add(row / super::FP8_BLOCK * (K / super::FP8_BLOCK)))
        }
        #[inline(always)]
        unsafe fn step<const NP: usize, const R: usize>(lane: u32, r: &[(*const u8, *const f32); R], s: usize, slab: *const f32, lds: *const f32, acc: &mut [[f32; PASS]; R]) {
            fp8_row_step::<K, NP, R>(lane, r, s, slab, lds, acc)
        }
    }

    unsafe extern "C" {
        #[link_name = "llvm.convert.from.fp16.f32"]
        fn f16_to_f32(h: i16) -> f32;
    }
    /// `bf16_row_step` for FP8 rows: the lane's eight bytes of step `s` decoded
    /// through the table at `lut` and scaled by their block's scale once, then
    /// the same products in the same order per token. A lane's eight elements
    /// lie in one block (`FP8_BLOCK` is a multiple of 8). The block scale is
    /// folded into the weights here, where the CPU's `fp8` applies it to each
    /// block's sum: the two differ by rounding, not by value.
    #[inline(always)]
    pub unsafe fn fp8_row_step<const K: usize, const NP: usize, const R: usize>(lane: u32, w: &[(*const u8, *const f32); R], s: usize, slab: *const f32, lut: *const f32, acc: &mut [[f32; PASS]; R]) {
        const { assert!(K % 8 == 0 && NP <= PASS && super::FP8_BLOCK % 8 == 0) };
        let l = lane as usize;
        let base = s * 256 + l * 8;
        if base < K {
            let mut q = [0u64; R];
            let mut scale = [0f32; R];
            let mut r = 0;
            while r < R {
                q[r] = core::ptr::read_unaligned(w[r].0.add(base) as *const u64);
                scale[r] = *w[r].1.add(base / super::FP8_BLOCK);
                r += 1;
            }
            // four weights at a time: eight decoded at once spilled a register (2026-10-15).
            // The decode transplants each byte's bits into an f16 (sign to bit 15, the seven
            // magnitude bits one place down, Marlin's form): the f16 is the e4m3 value × 2⁻⁸ exactly,
            // normals and subnormals alike (both formats are linear below their first binade and the
            // mantissa lands top-aligned), so the block scale carries the 2⁸ and `w · scale` rounds as
            // it did through the table (`e4m3`) — bit-identical except at the NaN codes 0x7f/0xff,
            // which weights never hold. The table (`lut`, an LDS load a weight) ran `ffn_gu` at one
            // token in 59.9 µs, this in 48.8 (2026-10-18, bench-fp8-rows); sixteen bytes a lane over a
            // step pair measured 42.9 at one token but slower at two to thirty-two, and is not kept.
            let _ = lut;
            let mut h = 0;
            while h < 2 {
                let mut wv = [[0f32; 4]; R];
                let mut r = 0;
                while r < R {
                    let word = (q[r] >> (32 * h)) as u32;
                    let odd = (word & 0x8000_8000) | ((word & 0x7f00_7f00) >> 1); // bytes 1 and 3
                    let sh = word << 8;
                    let even = (sh & 0x8000_8000) | ((sh & 0x7f00_7f00) >> 1); // bytes 0 and 2
                    let sc = scale[r] * 256.0;
                    wv[r][0] = f16_to_f32((even & 0xffff) as i16) * sc;
                    wv[r][1] = f16_to_f32((odd & 0xffff) as i16) * sc;
                    wv[r][2] = f16_to_f32((even >> 16) as i16) * sc;
                    wv[r][3] = f16_to_f32((odd >> 16) as i16) * sc;
                    r += 1;
                }
                let mut i = 0;
                while i < NP {
                    let x = slab.add(i * STEP + l * STRIDE + 4 * h);
                    let mut j = 0;
                    while j < 4 {
                        let xv = *x.add(j);
                        let mut r = 0;
                        while r < R {
                            acc[r][i] += wv[r][j] * xv;
                            r += 1;
                        }
                        j += 1;
                    }
                    i += 1;
                }
                h += 1;
            }
        }
    }

    /// The rows kernel skeleton: the workgroup's rows (`rows_per_wg(n)`: a wave each for one
    /// token, `ROWS` a wave past it) of a `[m][K]` matrix over `n` activations, a pass of PASS tokens at a
    /// time, each pass staged a round of steps at a time from `stage(token,
    /// s)` — each thread returns element `s · 256 + thread` of that token's
    /// activation (anything past `K`; the row primitive masks it) and the
    /// skeleton owns the slab; `emit(row, token, y)` receives every (row,
    /// token) sum from lane 0. Every rows kernel is this with its own stage
    /// and emit. `ready` runs in every thread after the rows are summed and
    /// before the emits: a kernel whose emit needs data that may not have
    /// landed yet waits there, with the rows already behind it.
    ///
    /// With `NORM` the activations are rms-normed on the way, `ln` the norm's
    /// weight (read only then): the staged element is `ln[j] · x[j]`, the
    /// sum of squares of the raw elements rides along on the staging each
    /// thread already does, and the row sum is scaled once at the emit, `y ·
    /// rsqrt(mean(x²) + eps)`. That is `W · (ln ⊙ x) · r` where the CPU
    /// computes `W · (ln ⊙ (x · r))`: the same value to a rounding per
    /// element, on top of the summation order that already parts the card's
    /// stream from the CPU's (2026-09-19, tcdent). A norm launch per rows
    /// launch was 28 launches a token; norming in a prologue of every
    /// workgroup, bit for bit, cost more than the launches (each workgroup
    /// re-read its activations and waited on the reads before its first
    /// weights: +12 µs a launch).
    #[inline(always)]
    pub unsafe fn rows_over<const K: usize, const NORM: bool, const R: usize>(tid: u32, w: *const u16, ln: *const u16, m: usize, n: usize, eps: f32, stage: impl Fn(usize, usize) -> f32, ready: impl Fn(), emit: impl Fn(usize, usize, f32)) {
        rows_fmt::<K, NORM, Bf16, R>(tid, Bf16(w), ln, m, n, eps, stage, ready, emit)
    }
    /// `rows_over` for any weight format `F` (`Bf16`, `Fp8`); `R` rows a wave past one token (the
    /// launch's grid is `rows_grid(m, n, R)`).
    #[inline(always)]
    pub unsafe fn rows_fmt<const K: usize, const NORM: bool, F: Rows<K>, const R: usize>(tid: u32, w: F, ln: *const u16, m: usize, n: usize, eps: f32, stage: impl Fn(usize, usize) -> f32, ready: impl Fn(), emit: impl Fn(usize, usize, f32)) {
        let mut p0 = 0;
        while p0 < n {
            let np = if n - p0 < PASS { n - p0 } else { PASS };
            // the pass runs at a constant width so its token loop has nothing predicated
            match width(np) {
                1 => rows_pass::<K, 1, NORM, F, 1>(tid, w, ln, m, p0, np, eps, &stage, &ready, &emit),
                FEW => rows_pass::<K, FEW, NORM, F, R>(tid, w, ln, m, p0, np, eps, &stage, &ready, &emit),
                _ => rows_pass::<K, PASS, NORM, F, R>(tid, w, ln, m, p0, np, eps, &stage, &ready, &emit),
            }
            p0 += PASS;
        }
    }

    /// LDS a pass at width `NP` over `[m][K]` uses for its slab, in floats.
    pub const fn slab_floats(k: usize, np: usize) -> usize {
        let steps = k.div_ceil(256);
        let slots = if np <= 1 { if steps < FLIGHT { steps } else { FLIGHT } } else { 2 * PASS };
        slots * STEP
    }

    /// Step `s` of the pass's NP tokens from `p0` staged at `slab`
    /// (`[NP][STEP]`), this thread's element of each at `slot(t)`. All NP,
    /// not the `np` real ones: the trip count is a constant, so the loads
    /// issue together and `ss[i]` stays in registers (a runtime count put it
    /// in scratch, a predicate cost 10 % at 32 tokens, 2026-09-19). The
    /// tokens past `np` read stale rows of the TOKENS-deep buffers (`p0 + NP
    /// ≤ TOKENS`) and are never emitted. With NORM the norm weight is read
    /// once for the step.
    #[inline(always)]
    unsafe fn stage_round<const K: usize, const NP: usize, const NORM: bool>(t: usize, ln: *const u16, p0: usize, s: usize, slab: *mut f32, stage: &impl Fn(usize, usize) -> f32, ss: &mut [f32; PASS]) {
        const { assert!(TOKENS % PASS == 0) };
        let dst = slab.add(slot(t));
        let j = s * 256 + t;
        let wl = if NORM && j < K { super::widen(*ln.add(j)) } else { 1.0 };
        let mut i = 0;
        while i < NP {
            let v = stage(p0 + i, s);
            if NORM {
                ss[i] += v * v;
                *dst.add(i * STEP) = wl * v;
            } else {
                *dst.add(i * STEP) = v;
            }
            i += 1;
        }
    }

    /// One pass of `np` tokens (from `p0`) at width `NP`, a wave `R` rows (the workgroup's rows
    /// `wg · 8R..`): rounds of staged steps, the row sums, the emits for the `np` real tokens.
    #[inline(always)]
    unsafe fn rows_pass<const K: usize, const NP: usize, const NORM: bool, F: Rows<K>, const R: usize>(tid: u32, w: F, ln: *const u16, m: usize, p0: usize, np: usize, eps: f32, stage: &impl Fn(usize, usize) -> f32, ready: &impl Fn(), emit: &impl Fn(usize, usize, f32)) {
        const { assert!((slab_floats(K, NP) + 8 * PASS + F::LDS) * 4 <= 60 * 1024) };
        let (lane, wave) = (tid % 32, (tid / 32) as usize);
        let slab: *mut f32 = crate::gfx11::lds();
        let red = slab.add(slab_floats(K, NP)); // 8 wave partials per token, the norm's
        let fmt = red.add(8 * PASS); // the format's own (a decode table)
        F::setup(tid, fmt);
        let row = (core::arch::amdgpu::workgroup_id_x() as usize * 8 + wave) * R;
        let steps = K.div_ceil(256);
        let r = round(NP, steps);
        let mut acc = [[0f32; PASS]; R];
        let mut ss = [0f32; PASS];
        let live = row < m;
        // a row past the matrix computes the last row again and is never emitted
        let mut wr = [w.row(if row < m { row } else { m - 1 }); R];
        let mut k = 1;
        while k < R {
            wr[k] = w.row(if row + k < m { row + k } else { m - 1 });
            k += 1;
        }
        // full rounds carry no guards, so their loads hoist; one guarded round takes the tail
        let mut s0 = 0;
        while s0 + r <= steps {
            let mut k = 0;
            while k < r {
                stage_round::<K, NP, NORM>(tid as usize, ln, p0, s0 + k, slab.add(k * NP * STEP), stage, &mut ss);
                k += 1;
            }
            crate::gfx11::workgroup_sync();
            if live {
                let mut k = 0;
                while k < r {
                    F::step::<NP, R>(lane, &wr, s0 + k, slab.add(k * NP * STEP), fmt, &mut acc);
                    k += 1;
                }
            }
            crate::gfx11::workgroup_sync();
            s0 += r;
        }
        // the tail, a runtime trip count so it stays a loop and not another unrolled round
        let tail = steps - s0;
        if tail > 0 {
            let mut k = 0;
            while k < tail {
                stage_round::<K, NP, NORM>(tid as usize, ln, p0, s0 + k, slab.add(k * NP * STEP), stage, &mut ss);
                k += 1;
            }
            crate::gfx11::workgroup_sync();
            if live {
                let mut k = 0;
                while k < tail {
                    F::step::<NP, R>(lane, &wr, s0 + k, slab.add(k * NP * STEP), fmt, &mut acc);
                    k += 1;
                }
            }
            crate::gfx11::workgroup_sync();
        }
        // the norm's scales: each thread's sum of squares over what it staged, the wave's,
        // the eight waves' in order (`workgroup_sum`'s order); one barrier
        let mut rr = [1f32; PASS];
        if NORM {
            let mut i = 0;
            while i < NP {
                let s = crate::gfx11::wave_sum(lane, ss[i]);
                if lane == 0 {
                    *red.add(i * 8 + wave) = s;
                }
                i += 1;
            }
            crate::gfx11::workgroup_sync();
            let mut i = 0;
            while i < NP {
                let mut total = 0f32;
                let mut w = 0;
                while w < 8 {
                    total += *red.add(i * 8 + w);
                    w += 1;
                }
                rr[i] = crate::math::rsqrt(total / K as f32 + eps);
                i += 1;
            }
        }
        ready();
        let mut k = 0;
        while k < R {
            let mut i = 0;
            while i < NP {
                let y = crate::gfx11::wave_sum(lane, acc[k][i]);
                if lane == 0 && row + k < m && i < np {
                    emit(row + k, p0 + i, if NORM { y * rr[i] } else { y });
                }
                i += 1;
            }
            k += 1;
        }
    }

    pub use super::{FEW, FLIGHT, PASS, TOKENS};

    /// The pass width a pass of `np` tokens runs at: one token runs at 1
    /// (the decode path, its registers and code its own), a few (a verify step's) at FEW,
    /// any more at PASS, the tokens past `np` computed on stale slab entries and never
    /// written. Four widths measured as one kernel carrying four copies:
    /// 192 registers and scratch (2026-09-19).
    #[inline(always)]
    pub const fn width(np: usize) -> usize {
        if np <= 1 {
            1
        } else if np <= FEW {
            FEW
        } else {
            PASS
        }
    }

    /// The matrix path for a step of many tokens: `out[i][row] = W[row] · x[i]` over `x: [n][K]` f32 and
    /// `W: [m][K]` bf16 on `v_wmma_f32_16x16x16_bf16`, the activations rounded to bf16 as they are staged
    /// (the reference runs its activations in bf16), f32 accumulation. A workgroup takes WMMA_ROWS rows ×
    /// WMMA_TOKENS tokens (a wave 16 rows, eight 16-token tiles in registers); a round stages WMMA_K of the
    /// activations as bf16 in LDS and each wave runs its 4 k-steps against them, B (the weights' 16 rows
    /// × 16 k, one 32-byte load a lane) reused across the eight tiles. With NORM the pre-norm scale is
    /// folded in: the norm weight on the staged values, each token's 1/rms on its outputs (the scalar
    /// `rows_pass` does the same). Numerically a different sum (bf16 operands, 16-wide accumulation
    /// order): a prefill kernel, gated numerically against the token-at-a-time path, never a decode one.
    /// Grid: `wmma_grid(m, n)` workgroups of 256; LDS `WMMA_LDS` bytes.
    pub const WMMA_TOKENS: usize = 128;
    pub const WMMA_ROWS: usize = 128;
    pub const WMMA_K: usize = 64;
    /// A staged token row's stride in halfwords: the round's k plus a pad, so the sixteen rows an A fragment
    /// is read from fall in different LDS banks (at exactly 128 bytes a row every lane hit the same banks).
    pub const WMMA_XS: usize = WMMA_K + 8;
    pub const WMMA_LDS: usize = WMMA_TOKENS * WMMA_XS * 2 + WMMA_TOKENS * 4;
    pub const fn wmma_grid(m: usize, n: usize) -> u32 {
        (m.div_ceil(WMMA_ROWS) * n.div_ceil(WMMA_TOKENS)) as u32
    }
    #[repr(simd)]
    #[derive(Clone, Copy)]
    #[allow(non_camel_case_types)]
    pub struct v16i16(pub [i16; 16]);
    #[repr(simd)]
    #[derive(Clone, Copy)]
    #[allow(non_camel_case_types)]
    pub struct v8f32(pub [f32; 8]);
    unsafe extern "C" {
        #[link_name = "llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16"]
        fn wmma_f32_bf16(a: v16i16, b: v16i16, c: v8f32) -> v8f32;
    }
    /// `x` rounded to bf16 (ties to even), as the 16 bits.
    #[inline(always)]
    fn to_bf16(x: f32) -> u16 {
        (crate::math::bf16_round(x).to_bits() >> 16) as u16
    }
    /// The weights' side of a WMMA step: 16 bf16 of row `row`, `k..k + 16`, as the B fragment.
    pub trait WmmaW<const K: usize>: Copy {
        unsafe fn frag(self, row: usize, k: usize) -> v16i16;
    }
    #[derive(Clone, Copy)]
    pub struct WBf16(pub *const u16);
    impl<const K: usize> WmmaW<K> for WBf16 {
        #[inline(always)]
        unsafe fn frag(self, row: usize, k: usize) -> v16i16 {
            core::ptr::read_unaligned(self.0.add(row * K + k) as *const v16i16)
        }
    }
    /// FP8 e4m3 rows with an f32 scale a `FP8_BLOCK` square block: each weight times its block's scale,
    /// rounded to bf16 — the reference's dequantization of these checkpoints (fp8 × scale → bf16).
    #[derive(Clone, Copy)]
    pub struct WFp8 {
        pub w: *const u8,
        pub s: *const f32,
    }
    impl<const K: usize> WmmaW<K> for WFp8 {
        #[inline(always)]
        unsafe fn frag(self, row: usize, k: usize) -> v16i16 {
            const { assert!(K % super::FP8_BLOCK == 0) };
            let b: [u8; 16] = core::ptr::read_unaligned(self.w.add(row * K + k) as *const [u8; 16]);
            let sc = *self.s.add(row / super::FP8_BLOCK * (K / super::FP8_BLOCK) + k / super::FP8_BLOCK);
            let mut v = [0i16; 16];
            let mut i = 0;
            while i < 16 {
                v[i] = to_bf16(super::e4m3_sel(b[i]) * sc) as i16;
                i += 1;
            }
            v16i16(v)
        }
    }
    /// `rows_wmma` over bf16 weights and an activation matrix in memory.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rows_wmma<const K: usize, const NORM: bool>(tid: u32, w: *const u16, ln: *const u16, m: usize, n: usize, eps: f32, x: *const f32, out: *mut f32) {
        rows_wmma_with::<K, NORM, WBf16>(tid, WBf16(w), ln, m, n, eps, |i, k| *x.add(i * K + k), out)
    }
    /// The matrix path over any weights (`WmmaW`) and any activation source: `load(token, k)` is the
    /// activation's element (the down projections' SwiGLU of the gate|up output, say).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rows_wmma_with<const K: usize, const NORM: bool, W: WmmaW<K>>(tid: u32, wt: W, ln: *const u16, m: usize, n: usize, eps: f32, load: impl Fn(usize, usize) -> f32, out: *mut f32) {
        use core::arch::amdgpu::workgroup_id_x;
        const { assert!(K % WMMA_K == 0 && WMMA_K % 16 == 0 && WMMA_ROWS == 8 * 16 && WMMA_TOKENS == 8 * 16) };
        const STEPS: usize = WMMA_K / 16;
        const TILES: usize = WMMA_TOKENS / 16;
        let t = tid as usize;
        let (wave, lane) = (t / 32, t % 32);
        let row_blocks = m.div_ceil(WMMA_ROWS);
        let wg = workgroup_id_x() as usize;
        let (rb, tb) = (wg % row_blocks, wg / row_blocks);
        let (row0, tok0) = (rb * WMMA_ROWS + wave * 16, tb * WMMA_TOKENS);
        let xs: *mut u16 = crate::gfx11::lds();
        let rr = xs.add(WMMA_TOKENS * WMMA_XS) as *mut f32;
        // each token's 1/rms (NORM): a wave takes 16 tokens, its lanes 32 consecutive floats a load
        if NORM {
            let mut tt = 0;
            while tt < 16 {
                let ltok = wave * 16 + tt;
                let tok = tok0 + ltok;
                let mut ss = 0f32;
                if tok < n {
                    let mut k = lane;
                    while k < K {
                        let v = load(tok, k);
                        ss += v * v;
                        k += 32;
                    }
                }
                let total = crate::gfx11::wave_sum(lane as u32, ss);
                if lane == 0 {
                    *rr.add(ltok) = crate::math::rsqrt(total / K as f32 + eps);
                }
                tt += 1;
            }
        }
        // the wave's 16 rows of W, clamped into the matrix (a row past it is computed and never written)
        let myrow = row0 + lane % 16;
        let wrow = if myrow < m { myrow } else { m - 1 };
        let mut acc = [v8f32([0.0; 8]); TILES];
        let mut k0 = 0;
        while k0 < K {
            // stage: a lane keeps one k of the round (a wave reads 32 consecutive floats of a token a load),
            // the norm weight folded in, bf16; tokens past n stage zeros
            {
                let kk = t % WMMA_K;
                let wl = if NORM { super::widen(*ln.add(k0 + kk)) } else { 1.0 };
                let mut tok = t / WMMA_K;
                while tok < WMMA_TOKENS {
                    let gtok = tok0 + tok;
                    let v = if gtok < n { load(gtok, k0 + kk) * wl } else { 0.0 };
                    *xs.add(tok * WMMA_XS + kk) = to_bf16(v);
                    tok += 256 / WMMA_K;
                }
            }
            crate::gfx11::workgroup_sync();
            let mut s = 0;
            while s < STEPS {
                let b = wt.frag(wrow, k0 + s * 16);
                let mut tt = 0;
                while tt < TILES {
                    let a = core::ptr::read_unaligned(xs.add((tt * 16 + lane % 16) * WMMA_XS + s * 16) as *const v16i16);
                    acc[tt] = wmma_f32_bf16(a, b, acc[tt]);
                    tt += 1;
                }
                s += 1;
            }
            crate::gfx11::workgroup_sync();
            k0 += WMMA_K;
        }
        // D's register r of lane l is (token 2r + l/16 of the tile, row l % 16 of the wave's)
        if myrow < m {
            let mut tt = 0;
            while tt < TILES {
                let d: [f32; 8] = core::mem::transmute(acc[tt]);
                let mut r = 0;
                while r < 8 {
                    let ltok = tt * 16 + 2 * r + lane / 16;
                    let tok = tok0 + ltok;
                    if tok < n {
                        let y = if NORM { d[r] * *rr.add(ltok) } else { d[r] };
                        *out.add(tok * m + myrow) = y;
                    }
                    r += 1;
                }
                tt += 1;
            }
        }
    }
}
