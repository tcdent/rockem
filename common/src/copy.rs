//! Bulk copies as ops. A context swap-in, a fork, a weight upload: each is
//! bytes moved once at some ceiling — DRAM, the PCIe link — and is measured
//! against it like any kernel. `stream` writes with non-temporal 256-bit
//! stores, so a large destination (a BAR window, a fresh context region) is
//! never read into cache first; `copy_scalar` is the twin.

pub fn copy_scalar(dst: &mut [u8], src: &[u8]) {
    dst.copy_from_slice(src);
}

/// `dst = src`, streamed. Lengths must match; any alignment.
#[inline(always)]
pub fn stream(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avx2::stream(dst.as_mut_ptr(), src.as_ptr(), dst.len())
    }
    #[cfg(not(target_arch = "x86_64"))]
    copy_scalar(dst, src)
}

/// `dst = src` where `src` is write-combining memory (a card's host-memory buffer mapped through
/// KFD): non-temporal 256-bit loads, which read WC memory at its bandwidth where plain loads go a
/// cache line at a time (a 32 KB handoff read cost the route lap ~15 µs a layer, 2026-10-19). Lengths
/// must match; `src` any alignment (the aligned middle streams, the ends copy).
#[inline(always)]
pub fn stream_read(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avx2::stream_read(dst.as_mut_ptr(), src.as_ptr(), dst.len())
    }
    #[cfg(not(target_arch = "x86_64"))]
    copy_scalar(dst, src)
}

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use core::arch::x86_64::*;

    /// Head to 32-byte alignment of `src` with plain loads, then four non-temporal loads per
    /// iteration (MOVNTDQA: the WC buffer's lines streamed through the load path), then the tail.
    #[target_feature(enable = "avx2")]
    pub unsafe fn stream_read(dst: *mut u8, src: *const u8, n: usize) {
        let head = (32 - (src as usize & 31)) & 31;
        let head = head.min(n);
        core::ptr::copy_nonoverlapping(src, dst, head);
        let mut i = head;
        while i + 128 <= n {
            let a = _mm256_stream_load_si256(src.add(i) as *const __m256i);
            let b = _mm256_stream_load_si256(src.add(i + 32) as *const __m256i);
            let c = _mm256_stream_load_si256(src.add(i + 64) as *const __m256i);
            let d = _mm256_stream_load_si256(src.add(i + 96) as *const __m256i);
            _mm256_storeu_si256(dst.add(i) as *mut __m256i, a);
            _mm256_storeu_si256(dst.add(i + 32) as *mut __m256i, b);
            _mm256_storeu_si256(dst.add(i + 64) as *mut __m256i, c);
            _mm256_storeu_si256(dst.add(i + 96) as *mut __m256i, d);
            i += 128;
        }
        while i + 32 <= n {
            _mm256_storeu_si256(dst.add(i) as *mut __m256i, _mm256_stream_load_si256(src.add(i) as *const __m256i));
            i += 32;
        }
        core::ptr::copy_nonoverlapping(src.add(i), dst.add(i), n - i);
    }

    /// Head to 32-byte alignment of `dst` with plain stores, then four
    /// non-temporal stores per iteration, then the tail; one fence at the end
    /// so the stores are visible before the caller hands the memory over.
    #[target_feature(enable = "avx2")]
    pub unsafe fn stream(dst: *mut u8, src: *const u8, n: usize) {
        let head = (32 - (dst as usize & 31)) & 31;
        let head = head.min(n);
        core::ptr::copy_nonoverlapping(src, dst, head);
        let mut i = head;
        while i + 128 <= n {
            let a = _mm256_loadu_si256(src.add(i) as *const __m256i);
            let b = _mm256_loadu_si256(src.add(i + 32) as *const __m256i);
            let c = _mm256_loadu_si256(src.add(i + 64) as *const __m256i);
            let d = _mm256_loadu_si256(src.add(i + 96) as *const __m256i);
            _mm256_stream_si256(dst.add(i) as *mut __m256i, a);
            _mm256_stream_si256(dst.add(i + 32) as *mut __m256i, b);
            _mm256_stream_si256(dst.add(i + 64) as *mut __m256i, c);
            _mm256_stream_si256(dst.add(i + 96) as *mut __m256i, d);
            i += 128;
        }
        while i + 32 <= n {
            _mm256_stream_si256(dst.add(i) as *mut __m256i, _mm256_loadu_si256(src.add(i) as *const __m256i));
            i += 32;
        }
        core::ptr::copy_nonoverlapping(src.add(i), dst.add(i), n - i);
        _mm_sfence();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Rng;

    #[test]
    fn stream_matches_scalar_at_every_alignment() {
        let mut rng = Rng::new(9);
        let src: std::vec::Vec<u8> = (0..4096).map(|_| rng.next() as u8).collect();
        for off in 0..40 {
            for n in [0usize, 1, 31, 32, 33, 127, 128, 129, 1000, 4000] {
                let mut a = std::vec![0u8; 4096 + 64];
                let mut b = a.clone();
                stream(&mut a[off..off + n], &src[..n]);
                copy_scalar(&mut b[off..off + n], &src[..n]);
                assert_eq!(a, b, "off {off} n {n}");
            }
        }
    }
}
