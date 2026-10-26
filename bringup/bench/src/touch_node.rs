//! The node's prefetch touch (construct.rs `touch_avx2`, copied: it is private there): a 32-byte load a line, sixteen
//! lines in flight a round, the deadline checked every 4 KB. Shared by the cache benches so their touches are the node's.
#[target_feature(enable = "avx2")]
pub unsafe fn touch(p: *const u8, len: usize, until: Option<std::time::Instant>) -> bool {
    use std::arch::x86_64::*;
    const K: usize = 16;
    let mut acc = [_mm256_setzero_si256(); K];
    let mut i = 0;
    let mut next_check = 0;
    while i + 64 * K <= len {
        if i >= next_check {
            if until.is_some_and(|u| std::time::Instant::now() >= u) {
                return false;
            }
            next_check = i + (4 << 10);
        }
        for k in 0..K {
            acc[k] = _mm256_xor_si256(acc[k], _mm256_loadu_si256(p.add(i + 64 * k) as *const __m256i));
        }
        i += 64 * K;
    }
    while i + 32 <= len {
        acc[0] = _mm256_xor_si256(acc[0], _mm256_loadu_si256(p.add(i) as *const __m256i));
        i += 64;
    }
    let mut s = acc[0];
    for k in 1..K {
        s = _mm256_xor_si256(s, acc[k]);
    }
    std::hint::black_box(s);
    true
}
