//! bench-dram — what this box's memory system streams: every thread owns a
//! private buffer it first-touched itself (so the pages are local to it),
//! all threads start together, and the aggregate is bytes over the wall
//! time of the whole pass. The number the expert kernel is measured against.
//!
//!   bench-dram [--threads N] [--mb 256] [--iters 5] [--stride 1]   (thread t pinned to CPU t·stride)

use harness::arg;
use std::sync::Barrier;

fn num(name: &str, default: usize) -> usize {
    arg(name).map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(default)
}

#[target_feature(enable = "avx2")]
unsafe fn stream_sum(p: *const u8, len: usize) -> u64 {
    use core::arch::x86_64::*;
    let mut a = _mm256_setzero_si256();
    let mut b = _mm256_setzero_si256();
    let mut c = _mm256_setzero_si256();
    let mut d = _mm256_setzero_si256();
    let mut i = 0;
    while i + 128 <= len {
        a = _mm256_add_epi64(a, _mm256_loadu_si256(p.add(i) as *const __m256i));
        b = _mm256_add_epi64(b, _mm256_loadu_si256(p.add(i + 32) as *const __m256i));
        c = _mm256_add_epi64(c, _mm256_loadu_si256(p.add(i + 64) as *const __m256i));
        d = _mm256_add_epi64(d, _mm256_loadu_si256(p.add(i + 96) as *const __m256i));
        i += 128;
    }
    let s = _mm256_add_epi64(_mm256_add_epi64(a, b), _mm256_add_epi64(c, d));
    let mut o = [0u64; 4];
    _mm256_storeu_si256(o.as_mut_ptr() as *mut __m256i, s);
    o.iter().sum()
}

fn main() {
    let threads = num("--threads", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    let mb = num("--mb", 256);
    let iters = num("--iters", 5);
    let stride = num("--stride", 1);
    let len = mb << 20;
    let start = Barrier::new(threads);
    let done = Barrier::new(threads);
    let times = std::sync::Mutex::new(vec![0f64; iters]);
    std::thread::scope(|sc| {
        for t in 0..threads {
            let (start, done, times) = (&start, &done, &times);
            sc.spawn(move || {
                harness::bench::pin(t * stride);
                let mut buf = vec![0u8; len];
                for (i, b) in buf.iter_mut().enumerate().step_by(4096) {
                    *b = (i + t) as u8;
                }
                let mut sink = 0u64;
                for it in 0..iters {
                    start.wait();
                    let t0 = std::time::Instant::now();
                    sink = sink.wrapping_add(unsafe { stream_sum(buf.as_ptr(), len) });
                    let dt = t0.elapsed().as_secs_f64();
                    done.wait();
                    if t == 0 {
                        times.lock().unwrap()[it] = dt;
                    } else {
                        let mut g = times.lock().unwrap();
                        g[it] = g[it].max(dt);
                    }
                }
                std::hint::black_box(sink);
            });
        }
    });
    let mut t = times.into_inner().unwrap();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total = (len * threads) as f64;
    println!("{threads} threads × {mb} MB private: {:.1} GB/s aggregate (median pass), best {:.1} GB/s", total / t[iters / 2] / 1e9, total / t[0] / 1e9);
}
