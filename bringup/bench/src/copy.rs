//! bench-copy — the transfer ops against their ceilings: DRAM → DRAM (a
//! fork) and, on a box with a card, DRAM → host-visible VRAM through the BAR
//! (a context swap-in, a weight upload). `memcpy` against `common::copy::stream`,
//! one thread and several, on a context-sized buffer.
//!
//!   bench-copy [--mb 128] [--threads 1,2,4] [--bar]

#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;

use harness::{arg, flag};

fn main() {
    let mb: usize = arg("--mb").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(128);
    let threads: Vec<usize> = arg("--threads").map(|p| p.to_str().unwrap().split(',').map(|t| t.parse().unwrap()).collect()).unwrap_or(vec![1, 2, 4]);
    let n = mb << 20;
    let src: Vec<u8> = (0..n).map(|i| (i * 2654435761usize >> 13) as u8).collect();
    let mut dram = vec![0u8; n];
    let k = flag("--bar").then(|| kfd::Kfd::new(&std::fs::read(arg("--bundle").expect("--bar needs --bundle").join("kernels.gfx1100.elf")).unwrap()));
    let bar = k.as_ref().map(|k| k.vram_visible(n));
    let dsts: Vec<(&str, *mut u8)> = [Some(("dram", dram.as_mut_ptr())), bar.as_ref().map(|m| ("bar", m.ptr() as *mut u8))].into_iter().flatten().collect();
    for (name, dst) in dsts {
        for &t in &threads {
            for (way, f) in [("memcpy", common::copy::copy_scalar as fn(&mut [u8], &[u8])), ("stream", common::copy::stream as fn(&mut [u8], &[u8]))] {
                let chunk = n / t;
                let run = || {
                    std::thread::scope(|sc| {
                        for i in 0..t {
                            let (src, dst) = (&src[i * chunk..(i + 1) * chunk], unsafe { std::slice::from_raw_parts_mut(dst.add(i * chunk), chunk) });
                            sc.spawn(move || {
                                harness::bench::pin(i);
                                f(dst, src)
                            });
                        }
                    })
                };
                run();
                let b = harness::bench(5, run);
                println!("{name:<5} {way:<7} {t} thread{}: {:>6.1} GB/s  ({:.1} ms for {mb} MB)", if t == 1 { " " } else { "s" }, b.gbps(n), b.median.as_secs_f64() * 1e3);
            }
        }
    }
    if let Some(k) = &k {
        // first touch of a BAR window: what populating the mapping costs, one thread against
        // several, and madvise(MADV_POPULATE_WRITE) — the driver maps 16 pages per fault
        extern "C" {
            fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
        }
        for t in [1usize, 2, 4, 8] {
            let m = k.vram_visible(n);
            let t0 = std::time::Instant::now();
            std::thread::scope(|sc| {
                for i in 0..t {
                    let base = m.ptr() as usize + i * (n / t);
                    sc.spawn(move || {
                        harness::bench::pin(i);
                        for off in (0..n / t).step_by(4096) {
                            unsafe { std::ptr::write_volatile((base + off) as *mut u8, 1) };
                        }
                    });
                }
            });
            println!("touch {t} thread{}: {:>6.1} GB/s of window populated ({:.0} ms for {mb} MB)", if t == 1 { " " } else { "s" }, n as f64 / t0.elapsed().as_secs_f64() / 1e9, t0.elapsed().as_secs_f64() * 1e3);
        }
        let m = k.vram_visible(n);
        let t0 = std::time::Instant::now();
        let r = unsafe { madvise(m.ptr() as *mut u8, n, 23) };
        println!("madvise POPULATE_WRITE: r {r} {:>6.1} GB/s ({:.0} ms)", n as f64 / t0.elapsed().as_secs_f64() / 1e9, t0.elapsed().as_secs_f64() * 1e3);
    }
    if let Some(m) = bar {
        // the check: what the card holds is what was written
        let back: Vec<u8> = unsafe { std::slice::from_raw_parts(m.ptr() as *const u8, n).to_vec() };
        assert!(back == src, "BAR readback differs");
        println!("bar   readback identical");
    }
}
