//! Bring a bundle file into memory the way the design says: a hugetlbfs
//! file the size of the input, mapped, filled by the calling thread, locked.
//! The hugetlbfs pages are interleaved over every NUMA node (`set_mempolicy`
//! around the fill): a thread pool spanning both sockets then streams them at
//! both sockets' bandwidth, where first-touch from one thread had put 177 GB
//! on one node (metatron, 2026-10-15). Per-CCD placement is the later, measured
//! refinement (l3-intentional expert path). The
//! mapping outlives the process's use of it by construction — nothing here
//! is ever unmapped. `Mem::Anon` is the dev-host substitute when there is no
//! hugetlbfs; it is chosen explicitly, never fallen back to.

use std::fs;
use std::os::unix::fs::FileExt;
use std::io::Read;
use std::os::unix::io::AsRawFd;

#[derive(Clone, Copy, PartialEq)]
pub enum Mem {
    Hugetlb,
    Anon,
}

extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
    fn mlock(addr: *const u8, len: usize) -> i32;
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
    fn syscall(nr: i64, ...) -> i64;
}
const SYS_SET_MEMPOLICY: i64 = 238;
const SYS_MOVE_PAGES: i64 = 279;
const SYS_MBIND: i64 = 237;
const MPOL_BIND: i32 = 2;
const MPOL_DEFAULT: i32 = 0;
const MPOL_INTERLEAVE: i32 = 3;
/// Interleave this thread's new pages over every online node (all 64 bits set names more nodes than exist, which the kernel masks to the possible ones).
fn interleave(on: bool) {
    let mask: u64 = if on { u64::MAX } else { 0 };
    let r = unsafe { syscall(SYS_SET_MEMPOLICY, if on { MPOL_INTERLEAVE } else { MPOL_DEFAULT }, if on { &mask as *const u64 } else { std::ptr::null() }, if on { 64u64 } else { 0 }) };
    assert!(r == 0 || !on, "set_mempolicy: {}", std::io::Error::last_os_error());
}
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_SHARED: i32 = 1;
const MAP_PRIVATE: i32 = 2;
const MAP_ANONYMOUS: i32 = 0x20;
const HUGEPAGES: &str = "/dev/hugepages";
const GB: usize = 1 << 30;

/// The NUMA node of every 2 MiB page of a loaded region (`construct::PAGE`), asked of the
/// kernel (`move_pages` with no destination): the truth a placed pool runs its items by.
pub fn placement(buf: &[u8]) -> Vec<u8> {
    const PAGE: usize = glm_5_3_flash::construct::PAGE;
    let n = buf.len().div_ceil(PAGE);
    let mut nodes = Vec::with_capacity(n);
    for chunk in (0..n).collect::<Vec<_>>().chunks(4096) {
        let pages: Vec<*const u8> = chunk.iter().map(|&p| unsafe { buf.as_ptr().add(p * PAGE) }).collect();
        let mut status = vec![-1i32; pages.len()];
        let r = unsafe { syscall(SYS_MOVE_PAGES, 0i64, pages.len() as u64, pages.as_ptr(), std::ptr::null::<i32>(), status.as_mut_ptr(), 0i32) };
        assert_eq!(r, 0, "move_pages: {}", std::io::Error::last_os_error());
        nodes.extend(status.iter().map(|&s| u8::try_from(s).unwrap_or_else(|_| panic!("page not resident: status {s}"))));
    }
    nodes
}

/// The expert stream, placed: one region per NUMA node, each item copied into the region of
/// the node `node_of_item` names (`layout::placement`, from the nodes' measured stream
/// rates), packed in file order, first-touched by a thread pinned to that node. The items'
/// addresses come back in the stream's table; the file's layout stays what it is.
pub fn load_placed(path: &std::path::Path, mem: Mem, node_of_item: &[u8]) -> glm_5_3_flash::expert::Stream<'static> {
    use glm_5_3_flash::expert::{item_index, ITEMS};
    use glm_5_3_flash::layout::{self, PAGE};
    use glm_5_3_flash::{E, FIRST_K_DENSE_REPLACE, NUM_HIDDEN_LAYERS};
    let src = fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let len = src.metadata().unwrap().len() as usize;
    assert_eq!(len, layout::STREAM, "{}: not the expert stream", path.display());
    let nodes = glm_5_3_flash::construct::node_count();
    assert_eq!(node_of_item.len(), layout::MOE_LAYERS * E * ITEMS);
    // each node's region: its items' bytes, the region starting on a page of its own
    let mut bytes = vec![0usize; nodes];
    let mut addr = vec![0u64; node_of_item.len()];
    for l in FIRST_K_DENSE_REPLACE..NUM_HIDDEN_LAYERS {
        for e in 0..E {
            for (j, r) in layout::file_items(l, e) {
                let n = node_of_item[item_index(l, e, j)] as usize;
                addr[item_index(l, e, j)] = bytes[n] as u64;
                bytes[n] += r.len();
            }
        }
    }
    // each region with room for the items a re-placement moves in (`Stream::migrate`)
    const SLACK: usize = 10;
    let mut base = vec![0usize; nodes];
    let mut cap = vec![0usize; nodes];
    let mut total = 0;
    for n in 0..nodes {
        base[n] = total;
        cap[n] = (bytes[n] + bytes[n] / SLACK).div_ceil(PAGE) * PAGE;
        total += cap[n];
    }
    for l in FIRST_K_DENSE_REPLACE..NUM_HIDDEN_LAYERS {
        for e in 0..E {
            for j in 0..ITEMS {
                addr[item_index(l, e, j)] += base[node_of_item[item_index(l, e, j)] as usize] as u64;
            }
        }
    }
    let dst = map_memory(path, total, mem);
    // each region bound to its node by policy: whoever touches a page, it is allocated there
    if mem == Mem::Hugetlb {
        for n in 0..nodes {
            let mask: u64 = 1 << n;
            let r = unsafe { syscall(SYS_MBIND, dst as usize + base[n], cap[n], MPOL_BIND, &mask as *const u64, 64u64, 0u64) };
            assert_eq!(r, 0, "mbind node {n}: {}", std::io::Error::last_os_error());
        }
    }
    // each region's pages faulted in order by a thread on its node, before the readers touch them in
    // whatever order they reach them: the region's pages come off the node's free list in sequence
    if mem == Mem::Hugetlb {
        std::thread::scope(|sc| {
            for n in 0..nodes {
                let (b, l) = (dst as usize + base[n], cap[n]);
                sc.spawn(move || {
                    glm_5_3_flash::construct::pin(glm_5_3_flash::construct::first_core_of(n));
                    const MADV_POPULATE_WRITE: i32 = 23;
                    let r = unsafe { madvise(b as *mut u8, l, MADV_POPULATE_WRITE) };
                    assert_eq!(r, 0, "madvise populate node {n}: {}", std::io::Error::last_os_error());
                });
            }
        });
    }
    let t0 = std::time::Instant::now();
    let name = path.file_name().unwrap().to_string_lossy().into_owned();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let dst_addr = dst as usize;
    // readers over the file's runs (a block's 32 experts' share `r`, RUN bytes contiguous): one read
    // a run, its items scattered to their nodes from the staging buffer
    const READERS: usize = 8;
    let runs = layout::MOE_LAYERS * (E / layout::XBLOCK) * layout::RUNS;
    std::thread::scope(|sc| {
        for t in 0..READERS {
            let (src, done, name, addr) = (&src, &done, &name, &addr);
            sc.spawn(move || {
                let mut staging = vec![0u8; layout::RUN];
                let mut next_mark = GB;
                for u in (t..runs).step_by(READERS) {
                    let (l, b, r) = (FIRST_K_DENSE_REPLACE + u / (E / layout::XBLOCK * layout::RUNS), u / layout::RUNS % (E / layout::XBLOCK), u % layout::RUNS);
                    let run_start = layout::share(l, b * layout::XBLOCK, r);
                    src.read_exact_at(&mut staging, run_start as u64).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
                    for e in b * layout::XBLOCK..(b + 1) * layout::XBLOCK {
                        for (j, rg) in layout::file_items(l, e) {
                            if rg.start < run_start || rg.start >= run_start + layout::RUN {
                                continue;
                            }
                            let a = addr[item_index(l, e, j)] as usize;
                            let dst_item = unsafe { std::slice::from_raw_parts_mut((dst_addr + a) as *mut u8, rg.len()) };
                            dst_item.copy_from_slice(&staging[rg.start - run_start..rg.end - run_start]);
                        }
                    }
                    let d = done.fetch_add(layout::RUN, std::sync::atomic::Ordering::Relaxed) + layout::RUN;
                    if t == 0 && d >= next_mark {
                        println!("loading  : {name} {:.1}/{:.1} GB ({:.0} MB/s, placed by item)", d as f64 / 1e9, len as f64 / 1e9, d as f64 / 1e6 / t0.elapsed().as_secs_f64());
                        next_mark += GB;
                    }
                }
            });
        }
    });
    assert_eq!(done.load(std::sync::atomic::Ordering::Relaxed), len, "placed load: bytes copied");
    println!("loading  : {name} {:.1} GB in {:.0} s ({:.0} MB/s, placed by item; regions {} GB of {} mapped)", len as f64 / 1e9, t0.elapsed().as_secs_f64(), len as f64 / 1e6 / t0.elapsed().as_secs_f64(), bytes.iter().map(|b| format!("{:.1}", *b as f64 / 1e9)).collect::<Vec<_>>().join(" / "), total as f64 / 1e9);
    if mem == Mem::Hugetlb {
        assert_eq!(unsafe { mlock(dst, total) }, 0, "mlock: {}", std::io::Error::last_os_error());
    }
    let regions = (0..nodes).map(|n| glm_5_3_flash::expert::Region { base: base[n], cap: cap[n], used: bytes[n] }).collect();
    glm_5_3_flash::expert::Stream::placed(unsafe { std::slice::from_raw_parts(dst, total) }, addr, node_of_item.to_vec(), regions)
}
/// A mapping of `len` bytes for the file: hugetlbfs (a file of the same name under the mount) or anonymous.
fn map_memory(path: &std::path::Path, len: usize, mem: Mem) -> *mut u8 {
    match mem {
        Mem::Hugetlb => {
            let name = format!("{HUGEPAGES}/{}", path.file_name().unwrap().to_string_lossy());
            let f = fs::OpenOptions::new().read(true).write(true).create(true).open(&name).unwrap_or_else(|e| panic!("{name}: {e} (is hugetlbfs mounted?)"));
            let rounded = len.div_ceil(GB) * GB;
            f.set_len(rounded as u64).unwrap();
            let p = unsafe { mmap(std::ptr::null_mut(), rounded, PROT_READ | PROT_WRITE, MAP_SHARED, f.as_raw_fd(), 0) };
            assert!(p as isize != -1, "mmap hugetlbfs {name} ({} GB): {}", rounded / GB, std::io::Error::last_os_error());
            p
        }
        Mem::Anon => {
            let p = unsafe { mmap(std::ptr::null_mut(), len.max(1), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) };
            assert!(p as isize != -1, "mmap anon: {}", std::io::Error::last_os_error());
            p
        }
    }
}
pub fn load(path: &std::path::Path, mem: Mem) -> &'static [u8] {
    load_with(path, mem)
}
fn load_with(path: &std::path::Path, mem: Mem) -> &'static [u8] {
    let mut src = fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let len = src.metadata().unwrap().len() as usize;
    let dst = map_memory(path, len, mem);
    let buf = unsafe { std::slice::from_raw_parts_mut(dst, len) };
    let t0 = std::time::Instant::now();
    let name = path.file_name().unwrap().to_string_lossy().into_owned();
    {
        {
            // fill from this thread: this is the first touch, interleaved over the nodes
            if mem == Mem::Hugetlb {
                interleave(true);
            }
            let (mut done, mut next_mark) = (0, GB);
            while done < len {
                let n = src.read(&mut buf[done..(done + 64 << 20).min(len)]).unwrap();
                assert!(n > 0, "short read on {}", path.display());
                done += n;
                if done >= next_mark || done == len {
                    println!("loading  : {name} {:.1}/{:.1} GB ({:.0} MB/s)", done as f64 / 1e9, len as f64 / 1e9, done as f64 / 1e6 / t0.elapsed().as_secs_f64());
                    next_mark += GB;
                }
            }
            if mem == Mem::Hugetlb {
                interleave(false);
            }
        }
    }
    if mem == Mem::Hugetlb {
        assert_eq!(unsafe { mlock(dst, len) }, 0, "mlock: {}", std::io::Error::last_os_error());
    }
    unsafe { std::slice::from_raw_parts(dst, len) }
}
