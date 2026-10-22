//! bench-touch — the decode prefetch's touch, replicated rung by rung from the probe's read ceiling to the production
//! shape, so the rung where the rate falls names the gap (tcdent 2026-10-22: "a cleanroom replication will show us how
//! it works"). Every listed core streams node-local memory (its own buffer, first-touched by itself); all start
//! together; the aggregate GB/s over the touch, box-wide and a CCD, best of `--rounds`.
//!
//!   bench-touch --mode M --cpus 1-23,25-47,49-71,73-95 [--mb 12] [--rounds 8] [--inflight 4] [--chunk 4096]
//!
//! modes (each adds one production feature to the one before):
//!   probe     the boot probe's `stream_sum`: every 32 B loaded, four accumulators, over the whole buffer (metatron-cpu §2:
//!             326 GB/s box-wide, 2026-10-15)
//!   touch     the node's `touch_avx2`: one 32 B load a 64 B line, `--inflight` lines in flight, over the whole buffer
//!   chunks    the touch in `--chunk`-byte pieces with an atomic flag read between pieces (the node's abandon check)
//!   items     production items: a CCD's 13 items a round, each four 272 KB slices at four places in the buffer, the
//!             cores of the CCD claiming items off one shared counter (the pool's queue), the touch in chunks
//!   static    the same items assigned statically, no shared counter
//!   slices    items of one slice (272 KB), claimed off the CCD's counter
//!
//! Prints, per mode: GB/s box-wide, then a CCD (min/med/max over CCDs), then a core (min/med/max).

use harness::arg;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
/// The probe's loop: every 32 bytes loaded, four accumulators (128 B an iteration).
#[target_feature(enable = "avx2")]
unsafe fn stream_sum(p: *const u8, len: usize) -> u64 {
    use core::arch::x86_64::*;
    let (mut a, mut b, mut c, mut d) = (_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256());
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
/// The node's touch: one 32 B load a line, `K` lines in flight (K = 4 is `construct::touch_avx2`).
#[target_feature(enable = "avx2")]
unsafe fn touch<const K: usize>(p: *const u8, len: usize) {
    use core::arch::x86_64::*;
    let mut acc = [_mm256_setzero_si256(); K];
    let mut i = 0;
    while i + 64 * K <= len {
        for k in 0..K {
            acc[k] = _mm256_xor_si256(acc[k], _mm256_loadu_si256(p.add(i + 64 * k) as *const __m256i));
        }
        i += 64 * K;
    }
    while i + 32 <= len {
        acc[0] = _mm256_xor_si256(acc[0], _mm256_loadu_si256(p.add(i) as *const __m256i));
        i += 64;
    }
    let mut s = acc[0];
    for k in 1..K {
        s = _mm256_xor_si256(s, acc[k]);
    }
    std::hint::black_box(s);
}
unsafe fn touch_n(inflight: usize, p: *const u8, len: usize) {
    match inflight {
        1 => touch::<1>(p, len),
        2 => touch::<2>(p, len),
        8 => touch::<8>(p, len),
        16 => touch::<16>(p, len),
        32 => touch::<32>(p, len),
        _ => touch::<4>(p, len),
    }
}
/// The touch in chunks with the node's abandon check between them.
unsafe fn touch_chunks(inflight: usize, chunk: usize, p: *const u8, len: usize, stop: &AtomicBool) {
    let mut o = 0;
    while o < len {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        touch_n(inflight, p.add(o), chunk.min(len - o));
        o += chunk;
    }
}

const SLICE: usize = 278_528; // a bundle item: 64 rows of gate + up, int4, with their scales (layout::GU_ITEM)
const SLICES_PER_ITEM: usize = 4; // a CCD's share of one expert
const ITEMS_PER_CCD: usize = 13; // the plan's cap

fn main() {
    let mode = arg("--mode").map(|p| p.to_str().unwrap().to_string()).unwrap_or("probe".into());
    let list = cpus(arg("--cpus").map(|p| p.to_str().unwrap().to_string()).as_deref().unwrap_or("1-23,25-47,49-71,73-95"));
    let mb = num("--mb", 12);
    let rounds = num("--rounds", 8);
    let inflight = num("--inflight", 4);
    let chunk = num("--chunk", 4096);
    let n = list.len();
    let ccd_of = |cpu: usize| cpu / 8;
    let ccds: Vec<usize> = {
        let mut v: Vec<usize> = list.iter().map(|&c| ccd_of(c)).collect();
        v.sort();
        v.dedup();
        v
    };
    // a CCD's item queue a round: items (slice offsets into the touching core's buffer... but an item belongs to the CCD:
    // its slices live in the CCD's cores' buffers, so any core of the CCD can claim it — the region is the claimer's
    // CCD-mate's, local to the node either way (as the node's items are)
    let bytes = mb << 20;
    let per_ccd_items = ITEMS_PER_CCD;
    let queues: Vec<AtomicUsize> = (0..16).map(|_| AtomicUsize::new(0)).collect();
    let stop = AtomicBool::new(false);
    let start = Barrier::new(n);
    let round_end = Barrier::new(n);
    let per_core = std::sync::Mutex::new(vec![(0usize, 0f64); n]); // (cpu, best GB/s)
    let bufs: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
    let ready = Barrier::new(n);
    std::thread::scope(|sc| {
        for (t, &cpu) in list.iter().enumerate() {
            let (start, round_end, ready, per_core, queues, stop, bufs, mode, list) = (&start, &round_end, &ready, &per_core, &queues, &stop, &bufs, mode.as_str(), &list);
            sc.spawn(move || {
                harness::bench::pin(cpu);
                let mut buf = vec![0u8; bytes + (2 << 20)];
                for (i, b) in buf.iter_mut().enumerate().step_by(4096) {
                    *b = (i + t) as u8;
                }
                let base = (buf.as_ptr() as usize + (2 << 20) - 1) & !((2 << 20) - 1);
                bufs[t].store(base, Ordering::Release);
                ready.wait();
                // this CCD's cores (indices into the list), for items
                let mates: Vec<usize> = (0..list.len()).filter(|&u| ccd_of(list[u]) == ccd_of(cpu)).collect();
                let my_rank = mates.iter().position(|&u| u == t).unwrap();
                let ccd_q = &queues[ccd_of(cpu) % 16];
                let mut best = 0f64;
                for r in 0..rounds {
                    if my_rank == 0 {
                        ccd_q.store(0, Ordering::Relaxed);
                    }
                    start.wait();
                    let t0 = std::time::Instant::now();
                    let mut done = 0usize;
                    match mode {
                        "probe" => {
                            std::hint::black_box(unsafe { stream_sum(base as *const u8, bytes) });
                            done = bytes;
                        }
                        "touch" => {
                            unsafe { touch_n(inflight, base as *const u8, bytes) };
                            done = bytes;
                        }
                        "chunks" => {
                            unsafe { touch_chunks(inflight, chunk, base as *const u8, bytes, stop) };
                            done = bytes;
                        }
                        "items" | "static" | "slices" => {
                            // a round's items on fresh memory: a CCD's 13 experts × 4 slices spread over its mates' buffers,
                            // each round in its own slab of every buffer (a round's 52 slices / 8 buffers = 6.5 slices = 1.8 MB a
                            // buffer; with --mb 12, six rounds before any address repeats — a wrap would read L3 hits, which the
                            // first form of this mode did, 2026-10-22)
                            let item_slices = if mode == "slices" { 1 } else { SLICES_PER_ITEM };
                            let n_items = per_ccd_items * SLICES_PER_ITEM / item_slices;
                            let slab = (per_ccd_items * SLICES_PER_ITEM).div_ceil(mates.len()) * SLICE;
                            assert!((r + 1) * slab <= bytes, "--mb too small for {} rounds of fresh memory", rounds);
                            loop {
                                let k = if mode == "static" { my_rank + (done / (item_slices * SLICE)) * mates.len() } else { ccd_q.fetch_add(1, Ordering::AcqRel) };
                                if k >= n_items {
                                    break;
                                }
                                for s in 0..item_slices {
                                    let g = k * item_slices + s; // the round's slice number, 0..52
                                    let m = mates[g % mates.len()];
                                    let off = r * slab + (g / mates.len()) * SLICE;
                                    let p = (bufs[m].load(Ordering::Relaxed) + off) as *const u8;
                                    unsafe { touch_chunks(inflight, chunk, p, SLICE, stop) };
                                }
                                done += item_slices * SLICE;
                            }
                        }
                        _ => panic!("mode {mode}: probe, touch, chunks, items, static, slices"),
                    }
                    let dt = t0.elapsed().as_secs_f64();
                    round_end.wait();
                    // the round's rate for this core: its bytes over the round's wall (the slowest core's time: all wait)
                    let wall = t0.elapsed().as_secs_f64();
                    let _ = dt;
                    if r > 0 {
                        best = best.max(done as f64 / wall / 1e9);
                    }
                }
                per_core.lock().unwrap()[t] = (cpu, best);
            });
        }
    });
    let pc = per_core.into_inner().unwrap();
    let mut by_ccd = vec![0f64; 16];
    for &(cpu, g) in &pc {
        by_ccd[ccd_of(cpu)] += g;
    }
    let mut ccd_rates: Vec<f64> = ccds.iter().map(|&c| by_ccd[c]).collect();
    ccd_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut core_rates: Vec<f64> = pc.iter().map(|x| x.1).collect();
    core_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = |v: &Vec<f64>| v[v.len() / 2];
    println!(
        "mode {mode} cores {n} ccds {} mb {mb} inflight {inflight} chunk {chunk}: {:.0} GB/s box; a CCD {:.1}/{:.1}/{:.1}; a core {:.2}/{:.2}/{:.2} (min/med/max)",
        ccds.len(),
        core_rates.iter().sum::<f64>(),
        ccd_rates[0], med(&ccd_rates), ccd_rates[ccd_rates.len() - 1],
        core_rates[0], med(&core_rates), core_rates[core_rates.len() - 1]
    );
}
