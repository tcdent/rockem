//! Bring a bundle file into memory the way the design says: a hugetlbfs
//! file the size of the input, mapped, filled by the calling thread (so
//! first-touch places every page on that thread's socket), locked. The
//! mapping outlives the process's use of it by construction — nothing here
//! is ever unmapped. `Mem::Anon` is the dev-host substitute when there is no
//! hugetlbfs; it is chosen explicitly, never fallen back to.

use std::fs;
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
}
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_SHARED: i32 = 1;
const MAP_PRIVATE: i32 = 2;
const MAP_ANONYMOUS: i32 = 0x20;
const MAP_POPULATE: i32 = 0x8000;
const HUGEPAGES: &str = "/dev/hugepages";
const GB: usize = 1 << 30;

pub fn load(path: &std::path::Path, mem: Mem) -> &'static [u8] {
    let mut src = fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let len = src.metadata().unwrap().len() as usize;
    let dst: *mut u8 = match mem {
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
            let p = unsafe { mmap(std::ptr::null_mut(), len.max(1), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0) };
            assert!(p as isize != -1, "mmap anon: {}", std::io::Error::last_os_error());
            p
        }
    };
    // fill from this thread: this is the first touch
    let buf = unsafe { std::slice::from_raw_parts_mut(dst, len) };
    let (mut done, mut next_mark) = (0, GB);
    let t0 = std::time::Instant::now();
    while done < len {
        let n = src.read(&mut buf[done..(done + 64 << 20).min(len)]).unwrap();
        assert!(n > 0, "short read on {}", path.display());
        done += n;
        if done >= next_mark || done == len {
            println!("loading  : {} {:.1}/{:.1} GB ({:.0} MB/s)", path.file_name().unwrap().to_string_lossy(), done as f64 / 1e9, len as f64 / 1e9, done as f64 / 1e6 / t0.elapsed().as_secs_f64());
            next_mark += GB;
        }
    }
    if mem == Mem::Hugetlb {
        assert_eq!(unsafe { mlock(dst, len) }, 0, "mlock: {}", std::io::Error::last_os_error());
    }
    unsafe { std::slice::from_raw_parts(dst, len) }
}
