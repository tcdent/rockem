//! bench-reuse — what decode's L3 prefetch can deliver: every listed core runs the one-token expert `gemv` over a
//! region of weights at once, in rounds, each round a fresh region (cold in every cache) unless the mode says
//! otherwise, and the bench reports the gemv's rate a core over the rounds. The modes are how the region got where it
//! is when the gemv starts:
//!
//!   cold  — straight from DRAM (nothing touched it);
//!   self  — this core touched it (AVX loads, a line each) just before;
//!   peer  — the neighbouring core of the same CCD touched it just before (decode's pattern: the prefetch's items and
//!           the compute's run on different cores of the CCD that will use the bytes);
//!   warm  — the same region every round (whatever caches keep it).
//!
//!   bench-reuse --mode peer --cpus 1-23,25-47,49-71,73-95 [--kb 1536] [--rounds 40]
//!
//! `--kb` is a core's region a round (decode prefetches ~13 MB a CCD, ~1.6 MB a core). Measured to tell apart why a
//! fully prefetched decode expert ran 31 µs against 42 from DRAM (2026-10-20), not ~18 at L3 speed.

use common::q4::{gemv, Act, QAct, GROUP};
use harness::arg;
use std::sync::Barrier;

const K: usize = 4096;

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

/// A line a 64 bytes, with AVX loads, four in flight (the node's prefetch `touch`).
#[target_feature(enable = "avx2")]
unsafe fn touch(p: *const u8, len: usize) {
    use core::arch::x86_64::*;
    let (mut a, mut b, mut c, mut d) = (_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256());
    let mut i = 0;
    while i + 256 <= len {
        a = _mm256_xor_si256(a, _mm256_loadu_si256(p.add(i) as *const __m256i));
        b = _mm256_xor_si256(b, _mm256_loadu_si256(p.add(i + 64) as *const __m256i));
        c = _mm256_xor_si256(c, _mm256_loadu_si256(p.add(i + 128) as *const __m256i));
        d = _mm256_xor_si256(d, _mm256_loadu_si256(p.add(i + 192) as *const __m256i));
        i += 256;
    }
    std::hint::black_box(_mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_xor_si256(c, d)));
}

fn main() {
    let mode = arg("--mode").map(|p| p.to_str().unwrap().to_string()).unwrap_or("cold".into());
    let list = cpus(arg("--cpus").map(|p| p.to_str().unwrap().to_string()).as_deref().unwrap_or("1-2"));
    let rounds = num("--rounds", 40);
    let rows = (num("--kb", 1536) << 10) / (K / 2) / 8 * 8; // a round's region: rows of K nibbles
    let wbytes = rows * K / 2;
    let regions = rounds + 1;
    let threads = list.len();
    // every core's regions, first-touched by itself (its node's pages); a peer reads its neighbour's
    let bufs: Vec<std::sync::Mutex<usize>> = (0..threads).map(|_| std::sync::Mutex::new(0)).collect();
    let ready = Barrier::new(threads);
    let phase = Barrier::new(threads);
    let rates = std::sync::Mutex::new(vec![0f64; threads]);
    std::thread::scope(|sc| {
        for (t, &cpu) in list.iter().enumerate() {
            let (bufs, ready, phase, rates, mode) = (&bufs, &ready, &phase, &rates, mode.as_str());
            sc.spawn(move || {
                harness::bench::pin(cpu);
                let mut w = vec![0u8; regions * wbytes];
                for (i, x) in w.iter_mut().enumerate().step_by(64) {
                    *x = (i * 31 + t) as u8;
                }
                let ws = vec![0.01f32; rows * (K / GROUP)];
                let (mut q, mut s, mut g) = (vec![0i8; K], vec![0f32; K / GROUP], vec![0i32; K / GROUP]);
                let x: Vec<f32> = (0..K).map(|k| ((k * 7 + t) % 200) as f32 / 100.0 - 1.0).collect();
                QAct { q: &mut q, scale: &mut s, gsum: &mut g }.quantize(&x);
                let mut out = vec![0f32; rows];
                *bufs[t].lock().unwrap() = w.as_ptr() as usize;
                ready.wait();
                // the neighbour on the same CCD: pairs (0,1), (2,3), ... of the list
                let peer = if t ^ 1 < threads { t ^ 1 } else { t };
                let peer_base = *bufs[peer].lock().unwrap();
                let mut total = 0f64;
                for r in 1..=rounds {
                    let reg = if mode == "warm" { 0 } else { r };
                    match mode {
                        "self" => unsafe { touch(w.as_ptr().add(reg * wbytes), wbytes) },
                        "peer" => unsafe { touch((peer_base as *const u8).add(reg * wbytes), wbytes) },
                        _ => {}
                    }
                    phase.wait();
                    let t0 = std::time::Instant::now();
                    let act = Act { q: &q, scale: &s, gsum: &g };
                    gemv::<K>(rows, &w[reg * wbytes..(reg + 1) * wbytes], &ws, &act, &mut out);
                    total += t0.elapsed().as_secs_f64();
                    std::hint::black_box(&out);
                    phase.wait();
                }
                rates.lock().unwrap()[t] = (rounds * wbytes) as f64 / total / 1e9;
            });
        }
    });
    let mut r = rates.into_inner().unwrap();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("mode {mode} cores {threads} region {} KB a core: gemv {:.1}/{:.1}/{:.1} GB/s a core (min/med/max), {:.0} GB/s summed", wbytes >> 10, r[0], r[threads / 2], r[threads - 1], r.iter().sum::<f64>());
}
