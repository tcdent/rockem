//! bench-stream — STREAM as Dell ran it for the Milan 378 GB/s Triad figure (BIOS Characterization for HPC, 2021): each
//! thread owns three arrays of `--mb` MB of f64 each, far larger than any cache, re-read every iteration; Copy, Scale,
//! Add and Triad over the whole arrays, `--iters` iterations, the best time of each kernel reported as STREAM does;
//! bytes counted STREAM's way (16 an element for Copy/Scale, 24 for Add/Triad). `--nt` stores the result with
//! non-temporal stores (what the vendor compilers emit for STREAM: the written line costs no read-for-ownership).
//! Threads pinned to `--cpus`; one a CCD (Dell's 16 threads) is `1,9,17,...`.
//!
//!   bench-stream --cpus 1,9,17,25,33,41,49,57,65,73,81,89 [--mb 64] [--iters 10] [--nt]
use harness::{arg, flag};
use std::sync::Barrier;

fn num(name: &str, default: usize) -> usize {
    arg(name).map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(default)
}
fn cpus(spec: &str) -> Vec<usize> {
    let mut v = Vec::new();
    for part in spec.split(',') {
        match part.split_once('-') {
            Some((a, b)) => v.extend(a.parse::<usize>().unwrap()..=b.parse::<usize>().unwrap()),
            None => v.push(part.parse().unwrap()),
        }
    }
    v
}

#[target_feature(enable = "avx2")]
unsafe fn copy(a: *mut f64, b: *const f64, n: usize, nt: bool) {
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 4 <= n {
        let v = _mm256_loadu_pd(b.add(i));
        if nt { _mm256_stream_pd(a.add(i), v) } else { _mm256_storeu_pd(a.add(i), v) }
        i += 4;
    }
    if nt { _mm_sfence() }
}
#[target_feature(enable = "avx2")]
unsafe fn scale(a: *mut f64, b: *const f64, n: usize, s: f64, nt: bool) {
    use core::arch::x86_64::*;
    let sv = _mm256_set1_pd(s);
    let mut i = 0;
    while i + 4 <= n {
        let v = _mm256_mul_pd(sv, _mm256_loadu_pd(b.add(i)));
        if nt { _mm256_stream_pd(a.add(i), v) } else { _mm256_storeu_pd(a.add(i), v) }
        i += 4;
    }
    if nt { _mm_sfence() }
}
#[target_feature(enable = "avx2")]
unsafe fn add(a: *mut f64, b: *const f64, c: *const f64, n: usize, nt: bool) {
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 4 <= n {
        let v = _mm256_add_pd(_mm256_loadu_pd(b.add(i)), _mm256_loadu_pd(c.add(i)));
        if nt { _mm256_stream_pd(a.add(i), v) } else { _mm256_storeu_pd(a.add(i), v) }
        i += 4;
    }
    if nt { _mm_sfence() }
}
#[target_feature(enable = "avx2")]
unsafe fn triad(a: *mut f64, b: *const f64, c: *const f64, n: usize, s: f64, nt: bool) {
    use core::arch::x86_64::*;
    let sv = _mm256_set1_pd(s);
    let mut i = 0;
    while i + 4 <= n {
        let v = _mm256_add_pd(_mm256_loadu_pd(b.add(i)), _mm256_mul_pd(sv, _mm256_loadu_pd(c.add(i))));
        if nt { _mm256_stream_pd(a.add(i), v) } else { _mm256_storeu_pd(a.add(i), v) }
        i += 4;
    }
    if nt { _mm_sfence() }
}

fn main() {
    let list = cpus(arg("--cpus").map(|p| p.to_str().unwrap().to_string()).as_deref().unwrap_or("1,9,17,25,33,41,49,57,65,73,81,89"));
    let mb = num("--mb", 64);
    let iters = num("--iters", 10);
    let nt = flag("--nt");
    let n = list.len();
    let elems = (mb << 20) / 8;
    let start = Barrier::new(n);
    let end = Barrier::new(n);
    // each kernel's best time (s) over the iterations, as thread 0 sees the round (all wait at the barrier)
    let best = std::sync::Mutex::new([f64::MAX; 4]);
    std::thread::scope(|sc| {
        for (t, &cpu) in list.iter().enumerate() {
            let (start, end, best) = (&start, &end, &best);
            sc.spawn(move || {
                harness::bench::pin(cpu);
                // STREAM's init: a = 1, b = 2, c = 0, first-touched by the thread that uses them; 64 B aligned (the
                // non-temporal store needs 32)
                let mut a = vec![1.0f64; elems + 8];
                let mut b = vec![2.0f64; elems + 8];
                let mut c = vec![0.0f64; elems + 8];
                let al = |v: &mut Vec<f64>| ((v.as_mut_ptr() as usize + 63) & !63) as *mut f64;
                let (ap, bp, cp) = (al(&mut a), al(&mut b), al(&mut c));
                for _ in 0..iters {
                    for k in 0..4 {
                        start.wait();
                        let t0 = std::time::Instant::now();
                        unsafe {
                            match k {
                                0 => copy(cp, ap, elems, nt),
                                1 => scale(bp, cp, elems, 3.0, nt),
                                2 => add(cp, ap, bp, elems, nt),
                                _ => triad(ap, bp, cp, elems, 3.0, nt),
                            }
                        }
                        end.wait();
                        if t == 0 {
                            let dt = t0.elapsed().as_secs_f64();
                            let mut bt = best.lock().unwrap();
                            bt[k] = bt[k].min(dt);
                        }
                    }
                }
                std::hint::black_box((&a[elems / 2], &b[elems / 2], &c[elems / 2]));
            });
        }
    });
    let bt = best.into_inner().unwrap();
    let bytes = |k: usize| (if k < 2 { 16.0 } else { 24.0 }) * elems as f64 * n as f64;
    println!(
        "stream threads {n} ({} MB an array a thread, {}): Copy {:.0}  Scale {:.0}  Add {:.0}  Triad {:.0} GB/s (best of {iters})",
        mb,
        if nt { "non-temporal stores" } else { "plain stores" },
        bytes(0) / bt[0] / 1e9,
        bytes(1) / bt[1] / 1e9,
        bytes(2) / bt[2] / 1e9,
        bytes(3) / bt[3] / 1e9
    );
}
