//! bench-experts — `deepseek_v2_lite::expert` on real q4g128 records from a bundle.
//!
//!   bench-experts --bundle <dir> [--threads N] [--experts-per-thread 32] [--iters 5] [--stride 1]
//!
//! Each thread owns a disjoint set of expert records and runs the CPU
//! realization over them back to back; the same threads then stream the same
//! bytes with plain AVX2 loads, which is the ceiling the kernel is reported
//! against. One record per thread is also checked against the scalar
//! reference on the same activation.

use deepseek_v2_lite::expert::{self, ScratchBuf, Weights};
use deepseek_v2_lite::layout::RECORD;
use deepseek_v2_lite::H as D_MODEL;
use harness::{arg, arg_required};
use std::sync::Barrier;

fn num(name: &str, default: usize) -> usize {
    arg(name).map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(default)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn stream_sum(p: *const u8, len: usize) -> u64 {
    use core::arch::x86_64::*;
    let mut a = _mm256_setzero_si256();
    let mut b = _mm256_setzero_si256();
    let mut i = 0;
    while i + 64 <= len {
        a = _mm256_add_epi64(a, _mm256_loadu_si256(p.add(i) as *const __m256i));
        b = _mm256_add_epi64(b, _mm256_loadu_si256(p.add(i + 32) as *const __m256i));
        i += 64;
    }
    let s = _mm256_add_epi64(a, b);
    let mut o = [0u64; 4];
    _mm256_storeu_si256(o.as_mut_ptr() as *mut __m256i, s);
    o.iter().sum()
}

fn main() {
    let bdir = arg_required("--bundle");
    let threads = num("--threads", 1);
    let per = num("--experts-per-thread", 32);
    let iters = num("--iters", 5);
    let f = std::fs::File::open(bdir.join("experts.q4g128.bin")).unwrap();
    let mm = unsafe { memmap2::Mmap::map(&f).unwrap() };
    let stream = &mm[..];
    let n_records = stream.len() / RECORD;
    assert!(threads * per <= n_records, "{n_records} records in the bundle");
    let x: Vec<f32> = (0..D_MODEL).map(|i| ((i * 7919) % 2000) as f32 * 1e-3 - 1.0).collect();

    // correctness: one record per thread, cpu vs reference
    for t in 0..threads {
        let rec = &stream[t * per * RECORD..t * per * RECORD + RECORD];
        let w = Weights::from_record(rec);
        let (mut a, mut b) = (ScratchBuf::new(), ScratchBuf::new());
        let (mut oa, mut ob) = (vec![0f32; D_MODEL], vec![0f32; D_MODEL]);
        expert::forward_scalar(&w, &x, &mut a.scratch(), &mut oa);
        expert::forward(&w, &x, &mut b.scratch(), &mut ob);
        let (abs, scale) = harness::diff::err(&ob, &oa);
        assert!(abs <= 1e-4 * scale.max(1.0), "thread {t}: cpu vs reference abs {abs:.3e} of {scale:.3e}");
    }

    let bytes_per_thread = per * RECORD;
    let stride = num("--stride", 1);
    // every pass is fenced by barriers on both sides and timed as a whole, so the
    // aggregate is bytes over the wall time all threads were busy; each thread is
    // pinned and reads a private copy it first-touched itself, like the node's loader
    let start = Barrier::new(threads);
    let done = Barrier::new(threads);
    let passes = std::sync::Mutex::new((vec![0f64; iters], vec![0f64; iters]));
    std::thread::scope(|sc| {
        for t in 0..threads {
            let (start, done, passes, stream, x) = (&start, &done, &passes, stream, &x);
            sc.spawn(move || {
                harness::bench::pin(t * stride);
                let mine: Vec<u8> = stream[t * bytes_per_thread..(t + 1) * bytes_per_thread].to_vec();
                let mut b = ScratchBuf::new();
                let mut out = vec![0f32; D_MODEL];
                let mut sink = 0u64;
                for it in 0..iters + 1 {
                    start.wait();
                    let t0 = std::time::Instant::now();
                    for e in 0..per {
                        let w = Weights::from_record(&mine[e * RECORD..(e + 1) * RECORD]);
                        expert::forward(&w, x, &mut b.scratch(), &mut out);
                    }
                    let dk = t0.elapsed().as_secs_f64();
                    done.wait();
                    start.wait();
                    let t0 = std::time::Instant::now();
                    sink = sink.wrapping_add(unsafe { stream_sum(mine.as_ptr(), mine.len()) });
                    let ds = t0.elapsed().as_secs_f64();
                    done.wait();
                    if it > 0 {
                        let mut g = passes.lock().unwrap();
                        g.0[it - 1] = g.0[it - 1].max(dk);
                        g.1[it - 1] = g.1[it - 1].max(ds);
                    }
                }
                std::hint::black_box(sink);
            });
        }
    });
    let (mut pk, mut ps) = passes.into_inner().unwrap();
    pk.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let as_bench = |v: &Vec<f64>| harness::Bench { min: std::time::Duration::from_secs_f64(v[0]), median: std::time::Duration::from_secs_f64(v[v.len() / 2]), iters };
    let (k, s) = (as_bench(&pk), as_bench(&ps));
    let total = bytes_per_thread * threads;
    let ceiling = s.gbps(total);
    println!("threads {threads}, {per} experts/thread, {:.1} MB/thread, {iters} iters", bytes_per_thread as f64 / 1e6);
    harness::bench::report("streaming read (ceiling)", &s, Some(total), None);
    harness::bench::report("deepseek_v2_lite::expert", &k, Some(total), Some(ceiling));
    println!("per expert: {:.1} µs at {threads} threads", k.median.as_secs_f64() * 1e6 / per as f64);
}
