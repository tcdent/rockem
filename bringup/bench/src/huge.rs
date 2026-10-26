//! Bench memory in 2 MiB pages, as the node's experts are (hugetlb). metatron's THP is `madvise`: a plain allocation is
//! 4 KiB pages, and a DRAM stream over them pays TLB misses the node does not (every bench before 2026-10-26 did).
extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
}
/// `len` bytes (rounded up to 2 MiB) at a 2 MiB boundary, advised huge before the first touch; never freed.
pub fn alloc(len: usize) -> &'static mut [u8] {
    alloc_pages(len, true)
}
/// The same, in 2 MiB pages (`huge`) or left to 4 KiB pages (`MADV_NOHUGEPAGE`).
pub fn alloc_pages(len: usize, huge: bool) -> &'static mut [u8] {
    const H: usize = 2 << 20;
    let len = len.next_multiple_of(H);
    unsafe {
        let p = mmap(std::ptr::null_mut(), len + H, 3, 0x22, -1, 0); // PROT_READ|WRITE, MAP_PRIVATE|ANONYMOUS
        assert!(p as isize != -1, "mmap {len}");
        let a = (p as usize).next_multiple_of(H) as *mut u8;
        assert_eq!(madvise(a, len, if huge { 14 } else { 15 }), 0, "madvise"); // MADV_HUGEPAGE / MADV_NOHUGEPAGE
        std::slice::from_raw_parts_mut(a, len)
    }
}
/// The process's anonymous memory in huge pages, MiB (`/proc/self/smaps_rollup`).
pub fn huge_mib() -> usize {
    std::fs::read_to_string("/proc/self/smaps_rollup").unwrap_or_default().lines().find(|l| l.starts_with("AnonHugePages:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<usize>().ok()).unwrap_or(0) >> 10
}
