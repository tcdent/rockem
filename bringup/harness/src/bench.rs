//! Timing against a ceiling. Every kernel number in this project is reported
//! as a ratio to the bytes it had to move at a stated bandwidth, so "fast"
//! has a denominator. `bench` runs a closure `iters` times after a warm-up
//! and returns min and median wall time.

use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct Bench {
    pub min: Duration,
    pub median: Duration,
    pub iters: usize,
}

impl Bench {
    /// GB/s implied by moving `bytes` once per iteration at the median.
    pub fn gbps(&self, bytes: usize) -> f64 {
        bytes as f64 / self.median.as_secs_f64() / 1e9
    }
    /// Fraction of a stated ceiling (GB/s) reached at the median.
    pub fn ratio(&self, bytes: usize, ceiling_gbps: f64) -> f64 {
        self.gbps(bytes) / ceiling_gbps
    }
}

pub fn bench(iters: usize, mut f: impl FnMut()) -> Bench {
    f();
    f();
    let mut t: Vec<Duration> = (0..iters)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed()
        })
        .collect();
    t.sort();
    Bench { min: t[0], median: t[t.len() / 2], iters }
}

/// One line: name, median, and — when a byte count and ceiling are given —
/// the achieved GB/s and the ratio to the ceiling.
pub fn report(name: &str, b: &Bench, bytes: Option<usize>, ceiling_gbps: Option<f64>) {
    let mut line = format!("{name:<40} median {:>9.1} µs  min {:>9.1} µs", b.median.as_secs_f64() * 1e6, b.min.as_secs_f64() * 1e6);
    if let Some(n) = bytes {
        line += &format!("  {:>7.1} GB/s", b.gbps(n));
        if let Some(c) = ceiling_gbps {
            line += &format!("  {:.3} of {c:.0}", b.ratio(n, c));
        }
    }
    println!("{line}");
}

extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
}
/// Pin the calling thread to one CPU, so a bandwidth number is that core's
/// and its first-touched memory stays local to it.
pub fn pin(cpu: usize) {
    let mut mask = [0u64; 16];
    mask[cpu / 64] |= 1 << (cpu % 64);
    let r = unsafe { sched_setaffinity(0, size_of_val(&mask), mask.as_ptr()) };
    assert!(r == 0, "sched_setaffinity({cpu})");
}
