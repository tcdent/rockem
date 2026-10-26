//! The decode `gemv` with a software prefetch hint, shared by the cache benches (bench-order, bench-window).
/// The production one-token kernel (`common::q4::avx2::gemv`, copied: its helpers are private) with a software prefetch
/// `dist` bytes ahead of its weight reads: `hint` 0 none (the production kernel's behaviour), 1 `prefetchnta` (the line
/// marked non-temporal: kept out of the cache levels it would otherwise displace), 2 `prefetcht0` (the control: a
/// prefetch with ordinary allocation).
#[target_feature(enable = "avx2")]
pub unsafe fn gemv_hint<const K: usize>(m: usize, w: *const u8, ws: *const f32, act: &common::q4::Act, out: *mut f32, hint: u8, dist: usize) {
    use common::q4::{GROUP, TILE, WBIAS};
    use std::arch::x86_64::*;
    #[inline(always)]
    unsafe fn dot64(w: *const u8, a_lo: __m256i, a_hi: __m256i, mask: __m256i, ones: __m256i, acc: __m256i) -> __m256i {
        let p = _mm256_loadu_si256(w as *const __m256i);
        let lo = _mm256_and_si256(p, mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(p), mask);
        let s = _mm256_add_epi16(_mm256_maddubs_epi16(lo, a_lo), _mm256_maddubs_epi16(hi, a_hi));
        _mm256_add_epi32(acc, _mm256_madd_epi16(s, ones))
    }
    #[inline(always)]
    unsafe fn hsum4(a: __m256i, b: __m256i, c: __m256i, d: __m256i) -> __m128i {
        let ab = _mm256_hadd_epi32(a, b);
        let cd = _mm256_hadd_epi32(c, d);
        let x = _mm256_hadd_epi32(ab, cd);
        _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256::<1>(x))
    }
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
            // the group's 256 bytes are four lines: each prefetched `dist` ahead
            if hint != 0 {
                let mut l = 0;
                while l < 256 {
                    let p = wg.add(dist + l) as *const i8;
                    if hint == 1 { _mm_prefetch::<_MM_HINT_NTA>(p) } else { _mm_prefetch::<_MM_HINT_T0>(p) }
                    l += 64;
                }
            }
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
            let sums = hsum4(c0, c1, c2, c3);
            let bias = _mm_set1_epi32(WBIAS * *act.gsum.get_unchecked(g));
            let s = _mm_cvtepi32_ps(_mm_sub_epi32(sums, bias));
            let sc = _mm_mul_ps(_mm_loadu_ps(st.add(g * TILE)), _mm_set1_ps(*act.scale.get_unchecked(g)));
            y = _mm_add_ps(y, _mm_mul_ps(s, sc));
        }
        _mm_storeu_ps(out.add(t * TILE), y);
    }
}
