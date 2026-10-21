//! bench-l3 — the L3 as a programmable cache (docs/expert-cache-architecture.md, phase 0). Per CCD, its first four
//! non-housekeeping cores are "keep" cores and the rest "stream" cores. A round, all CCDs in lockstep:
//!
//!   1. the keep cores touch X MB of fresh lines (the planned set, X/4 a core: AVX loads, a line each, or prefetcht0);
//!   2. the stream cores read Y MB of fresh lines from DRAM (the run's own misses passing through the same L3);
//!   3. each keep core reads its neighbouring keep core's slice of the planned set (a touch and its use on different
//!      cores, as in decode), its fill counters read around the read (`pmc`, programmed by the node: L2 prefetches
//!      that hit / missed L3, demand fills from L3 / DRAM);
//!   4. with `--reread`, each stream core then reads the planned slices again (does an L3 hit leave the line?);
//!   5. with `--concurrent`, step 3 runs while the stream cores read another fresh Y (do L3 and DRAM streams add?).
//!
//! `--keep-ways K` (0 = no split) sets a resctrl CAT split for the run: a `keep` group of the keep cores fills ways
//! 0..K, every other CPU (the root group) ways K..16; the split is undone at exit. resctrl must be mounted.
//!
//!   bench-l3 --ccds 0-11 --x-mb 16 --y-mb 8 [--keep-ways 8] [--rounds 12] [--touch loads|prefetch] [--reread] [--concurrent]
//!
//! Prints, over the rounds (after the first): the planned set's lines from L3 as read by the keep cores (and by the
//! stream cores with `--reread`), the read rates, and with `--concurrent` the stream side's rate.

use common::pmc::rdpmc;
use harness::arg;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Barrier;

fn num(name: &str, default: usize) -> usize {
    arg(name).map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(default)
}
fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}
fn list(spec: &str) -> Vec<usize> {
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
unsafe fn read(p: *const u8, len: usize) -> u64 {
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
    let s = _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_xor_si256(c, d));
    _mm256_extract_epi64::<0>(s) as u64
}

unsafe fn prefetch(p: *const u8, len: usize) {
    use core::arch::x86_64::*;
    let mut i = 0;
    while i < len {
        _mm_prefetch::<_MM_HINT_T0>(p.add(i) as *const i8);
        i += 64;
    }
}

/// The counters a phase: L2 prefetches that hit L3, that missed it; demand fills from L3, from DRAM.
fn counts() -> [u64; 4] {
    [rdpmc(4), rdpmc(5), rdpmc(0), rdpmc(1)]
}

struct Sum([AtomicU64; 6]); // hit, miss, dfill l3, dfill dram, bytes, ns
impl Sum {
    fn new() -> Self {
        Sum(std::array::from_fn(|_| AtomicU64::new(0)))
    }
    fn add(&self, c0: [u64; 4], c1: [u64; 4], bytes: usize, ns: u64) {
        for i in 0..4 {
            self.0[i].fetch_add(c1[i].wrapping_sub(c0[i]) & ((1 << 48) - 1), Ordering::Relaxed);
        }
        self.0[4].fetch_add(bytes as u64, Ordering::Relaxed);
        self.0[5].fetch_add(ns, Ordering::Relaxed);
    }
    fn line(&self, what: &str, threads: usize) -> String {
        let v: Vec<u64> = self.0.iter().map(|x| x.load(Ordering::Relaxed)).collect();
        let pf = (v[0] + v[1]).max(1) as f64;
        let lines = (v[4] / 64).max(1) as f64;
        let gbs = if v[5] > 0 { v[4] as f64 / (v[5] as f64 / threads.max(1) as f64) } else { 0.0 };
        // the lines served from L3, prefetched or demanded, of every line that came from L3 or DRAM
        let l3 = (v[0] + v[2]) as f64 / (v[0] + v[1] + v[2] + v[3]).max(1) as f64;
        let _ = pf;
        format!("{what}: {:.1} % from L3 (prefetch hit {:.2} miss {:.2}, demand L3 {:.2} DRAM {:.2} a line read), {:.1} GB/s a core", 100.0 * l3, v[0] as f64 / lines, v[1] as f64 / lines, v[2] as f64 / lines, v[3] as f64 / lines, gbs / threads.max(1) as f64)
    }
}

/// resctrl: the keep group's CPUs fill ways 0..k; the root group (every other CPU) ways k..16. Undone by `undo`.
fn split(keep_cpus: &[usize], k: usize) -> Option<Vec<String>> {
    let root = "/sys/fs/resctrl";
    let schemata = std::fs::read_to_string(format!("{root}/schemata")).ok()?;
    let l3 = schemata.lines().find(|l| l.trim_start().starts_with("L3:"))?.trim().to_string();
    let domains: Vec<String> = l3.trim_start_matches("L3:").split(';').map(|d| d.split('=').next().unwrap().to_string()).collect();
    let mask = |m: u32| format!("L3:{}", domains.iter().map(|d| format!("{d}={m:x}")).collect::<Vec<_>>().join(";"));
    let (keep, stream) = ((1u32 << k) - 1, 0xffffu32 & !((1u32 << k) - 1));
    std::fs::create_dir_all(format!("{root}/keep")).ok()?;
    std::fs::write(format!("{root}/keep/schemata"), mask(keep) + "\n").ok()?;
    let cpus = keep_cpus.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",");
    std::fs::write(format!("{root}/keep/cpus_list"), cpus + "\n").ok()?;
    std::fs::write(format!("{root}/schemata"), mask(stream) + "\n").ok()?;
    Some(vec![mask(0xffff)])
}
fn undo(full: &[String]) {
    let root = "/sys/fs/resctrl";
    let _ = std::fs::write(format!("{root}/schemata"), full[0].clone() + "\n");
    let _ = std::fs::remove_dir(format!("{root}/keep"));
}

fn main() {
    let ccds = list(arg("--ccds").map(|p| p.to_str().unwrap().to_string()).as_deref().unwrap_or("0"));
    let x = num("--x-mb", 16) << 20;
    let y = num("--y-mb", 8) << 20;
    let k = num("--keep-ways", 0);
    let rounds = num("--rounds", 12);
    let touch_prefetch = arg("--touch").map(|p| p.to_str().unwrap() == "prefetch").unwrap_or(false);
    let (reread, concurrent) = (flag("--reread"), flag("--concurrent"));
    const HOUSE: [usize; 4] = [0, 24, 48, 72];
    // per CCD: four keep cores, the rest stream cores
    let mut keep = Vec::new();
    let mut stream = Vec::new();
    for &c in &ccds {
        let cores: Vec<usize> = (8 * c..8 * c + 8).filter(|x| !HOUSE.contains(x)).collect();
        keep.push(cores[..4].to_vec());
        stream.push(cores[4..].to_vec());
    }
    let full = if k > 0 {
        let s = split(&keep.concat(), k);
        if s.is_none() {
            eprintln!("resctrl split failed (mounted?)");
            return;
        }
        s
    } else {
        None
    };
    let threads = keep.concat().len() + stream.concat().len();
    let bar = Barrier::new(threads);
    let (kread, sread, srun) = (Sum::new(), Sum::new(), Sum::new());
    let xs = x / 4;
    // every keep core's planned slices, a fresh one a round; the neighbour reads it
    let planned: Vec<Vec<std::sync::Mutex<usize>>> = keep.iter().map(|g| g.iter().map(|_| std::sync::Mutex::new(0)).collect()).collect();
    let ready = Barrier::new(threads);
    std::thread::scope(|sc| {
        for (gi, g) in keep.iter().enumerate() {
            for (ki, &cpu) in g.iter().enumerate() {
                let (bar, ready, planned, kread) = (&bar, &ready, &planned, &kread);
                let nstream = stream[gi].len();
                sc.spawn(move || {
                    harness::bench::pin(cpu);
                    let buf = vec![1u8; (rounds + 1) * xs];
                    *planned[gi][ki].lock().unwrap() = buf.as_ptr() as usize;
                    ready.wait();
                    let neighbour = *planned[gi][(ki + 1) % 4].lock().unwrap();
                    let mut sink = 0u64;
                    for r in 0..rounds {
                        let mine = unsafe { buf.as_ptr().add(r * xs) };
                        unsafe { if touch_prefetch { prefetch(mine, xs) } else { sink ^= read(mine, xs) } };
                        bar.wait(); // touched
                        bar.wait(); // streamed (or the concurrent read starts)
                        let p = (neighbour + r * xs) as *const u8;
                        let (c0, t0) = (counts(), std::time::Instant::now());
                        sink ^= unsafe { read(p, xs) };
                        let ns = t0.elapsed().as_nanos() as u64;
                        if r > 0 {
                            kread.add(c0, counts(), xs, ns);
                        }
                        bar.wait(); // read
                        bar.wait(); // reread
                        let _ = nstream;
                    }
                    std::hint::black_box(sink);
                });
            }
        }
        for (gi, g) in stream.iter().enumerate() {
            for (si, &cpu) in g.iter().enumerate() {
                let (bar, ready, planned, sread, srun) = (&bar, &ready, &planned, &sread, &srun);
                let n = g.len();
                sc.spawn(move || {
                    harness::bench::pin(cpu);
                    let ys = y / n;
                    let buf = vec![1u8; (2 * rounds + 2) * ys.max(64)];
                    ready.wait();
                    let mut sink = 0u64;
                    for r in 0..rounds {
                        bar.wait(); // touched
                        if ys > 0 {
                            sink ^= unsafe { read(buf.as_ptr().add(2 * r * ys), ys) };
                        }
                        bar.wait(); // streamed
                        if concurrent && ys > 0 {
                            let (c0, t0) = (counts(), std::time::Instant::now());
                            sink ^= unsafe { read(buf.as_ptr().add((2 * r + 1) * ys), ys) };
                            if r > 0 {
                                srun.add(c0, counts(), ys, t0.elapsed().as_nanos() as u64);
                            }
                        }
                        bar.wait(); // read
                        if reread {
                            // the planned slices again, a stream core a keep core's slice (does a hit keep the line?)
                            let owner = si % 4;
                            let p = (*planned[gi][owner].lock().unwrap() + r * xs) as *const u8;
                            let (c0, t0) = (counts(), std::time::Instant::now());
                            sink ^= unsafe { read(p, xs) };
                            if r > 0 {
                                sread.add(c0, counts(), xs, t0.elapsed().as_nanos() as u64);
                            }
                        }
                        bar.wait(); // reread
                    }
                    std::hint::black_box(sink);
                });
            }
        }
    });
    if let Some(f) = &full {
        undo(f);
    }
    let nk = keep.concat().len();
    let ns = stream.concat().len();
    println!("ccds {} x {} MB y {} MB keep-ways {k} touch {}{}{}", ccds.len(), x >> 20, y >> 20, if touch_prefetch { "prefetch" } else { "loads" }, if reread { " reread" } else { "" }, if concurrent { " concurrent" } else { "" });
    println!("  {}", kread.line("planned, read by the keep cores", nk));
    if reread {
        println!("  {}", sread.line("planned, reread by the stream cores", ns));
    }
    if concurrent {
        println!("  {}", srun.line("fresh, streamed alongside", ns));
    }
}
