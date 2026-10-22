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
//!   fresh     the touch over `--round-mb` MB a core of memory no round has read before (round r at r × round-mb): DRAM
//!             only. `touch`/`probe`/`chunks` re-read one buffer every round, and the L3 keeps a protected share of it
//!             between rounds — at two cores a CCD a 20 MB buffer read 600 GB/s box-wide, 40 MB 383, 72 MB 318 (2026-10-22)
//!   probe-fresh  the probe's loop likewise on fresh memory
//!   triad     STREAM's Triad, a[i] = b[i] + s·c[i] over three arrays of `--round-mb`/3 MB a core, fresh each round; the
//!             rate counts 24 bytes an element as STREAM does (Dell's Milan figure, 378 GB/s, is this number)
//!   touch2    both 32 B halves of every line loaded (the probe's pattern at the touch's structure)
//!   pf        the touch with a software prefetch `--ahead` lines ahead of each load
//!   chunks    the touch in `--chunk`-byte pieces with an atomic flag read between pieces (the node's abandon check)
//!   items     production items: a CCD's 13 items a round, each four 272 KB slices at four places in the buffer, the
//!             cores of the CCD claiming items off one shared counter (the pool's queue), the touch in chunks
//!   static    the same items assigned statically, no shared counter
//!   slices    items of one slice (272 KB), claimed off the CCD's counter
//!   ranges    the same bytes as `runs`, statically mine (every mates-th run), each in my own buffer: the stream cut into
//!             runs with a reduction each, nothing else changed
//!   alt       `ranges`, but run k in buffer (k / mates + my rank) % mates: the address alternates between the CCD's cores'
//!             buffers from run to run, still static
//!   runs      a CCD's 13 experts' bytes (52 slices) as contiguous runs of `--run` bytes each, claimed off the CCD's
//!             counter: the rate against the run's length (the node's items are runs of 272–544 KB, 2026-10-22)
//!   share     runs, and a core that finds the queue empty helps finish the runs still in progress: the owner takes
//!             `--piece`-byte pieces from the run's front, helpers from its back (two sequential streams meeting), so no
//!             core idles at the tail
//!
//! A round's rate is every core's bytes over the round's wall (the slowest core's); the median round (by box rate) is
//! printed, with that round's CCD and core rates — not the best: the kernel zeroes a buffer as it is first touched and
//! the last ~32 MB a CCD of that sit in its L3, so with one or two cores a CCD the last rounds' "fresh" memory reads
//! from L3 (Triad at one core a CCD read 496 GB/s, 2026-10-22). Each core also streams a 64 MB scratch after its
//! allocation to push the zeroed lines out before the rounds start. (A first form kept each core's best round and summed those: a mode whose cores
//! did different work each round summed rates no round had, 2026-10-22.)
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
/// STREAM Triad over `n` doubles: a = b + s·c.
#[target_feature(enable = "avx2")]
unsafe fn triad(a: *mut f64, b: *const f64, c: *const f64, n: usize, s: f64) {
    use core::arch::x86_64::*;
    let sv = _mm256_set1_pd(s);
    let mut i = 0;
    while i + 16 <= n {
        for k in 0..4 {
            let bv = _mm256_loadu_pd(b.add(i + 4 * k));
            let cv = _mm256_loadu_pd(c.add(i + 4 * k));
            _mm256_storeu_pd(a.add(i + 4 * k), _mm256_add_pd(bv, _mm256_mul_pd(sv, cv)));
        }
        i += 16;
    }
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
        64 => touch::<64>(p, len),
        _ => touch::<4>(p, len),
    }
}
/// Both halves of every line loaded, `K` lines in flight.
#[target_feature(enable = "avx2")]
unsafe fn touch2<const K: usize>(p: *const u8, len: usize) {
    use core::arch::x86_64::*;
    let mut acc = [_mm256_setzero_si256(); K];
    let mut i = 0;
    while i + 64 * K <= len {
        for k in 0..K {
            acc[k] = _mm256_xor_si256(acc[k], _mm256_loadu_si256(p.add(i + 64 * k) as *const __m256i));
            acc[k] = _mm256_xor_si256(acc[k], _mm256_loadu_si256(p.add(i + 64 * k + 32) as *const __m256i));
        }
        i += 64 * K;
    }
    let mut s = acc[0];
    for k in 1..K {
        s = _mm256_xor_si256(s, acc[k]);
    }
    std::hint::black_box(s);
}
/// The touch with a software prefetch `ahead` lines ahead of each load.
#[target_feature(enable = "avx2")]
unsafe fn touch_pf<const K: usize>(p: *const u8, len: usize, ahead: usize) {
    use core::arch::x86_64::*;
    let mut acc = [_mm256_setzero_si256(); K];
    let mut i = 0;
    while i + 64 * K <= len {
        for k in 0..K {
            _mm_prefetch(p.add(i + 64 * (k + ahead)) as *const i8, _MM_HINT_T0);
            acc[k] = _mm256_xor_si256(acc[k], _mm256_loadu_si256(p.add(i + 64 * k) as *const __m256i));
        }
        i += 64 * K;
    }
    let mut s = acc[0];
    for k in 1..K {
        s = _mm256_xor_si256(s, acc[k]);
    }
    std::hint::black_box(s);
}
/// The touch over a run with the abandon flag read every `chunk` bytes *inside* the streaming loop: the accumulators
/// are reduced once, at the end. (A touch called a chunk reduced its accumulators each call, so a chunk's loads all
/// returned before the next chunk's issued — one DRAM latency of bubble a chunk, hidden by eight cores a CCD, half
/// the rate at two: 216 against 383 GB/s, 2026-10-22.)
#[target_feature(enable = "avx2")]
unsafe fn touch_flag<const K: usize>(p: *const u8, len: usize, chunk: usize, stop: &AtomicBool) {
    use core::arch::x86_64::*;
    let mut acc = [_mm256_setzero_si256(); K];
    let mut i = 0;
    let mut next_check = 0;
    while i + 64 * K <= len {
        if i >= next_check {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            next_check = i + chunk;
        }
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
unsafe fn touch_chunks(inflight: usize, chunk: usize, p: *const u8, len: usize, stop: &AtomicBool) {
    match inflight {
        64 => touch_flag::<64>(p, len, chunk, stop),
        32 => touch_flag::<32>(p, len, chunk, stop),
        16 => touch_flag::<16>(p, len, chunk, stop),
        8 => touch_flag::<8>(p, len, chunk, stop),
        _ => touch_flag::<4>(p, len, chunk, stop),
    }
}

const SLICE: usize = 278_528; // a bundle item: 64 rows of gate + up, int4, with their scales (layout::GU_ITEM)
const SLICES_PER_ITEM: usize = 4; // a CCD's share of one expert
/// A piece of a run claimed: the owner's from the front (forward), a helper's from the back (backward), the two counts
/// packed in one word; `None` once they meet.
fn claim_piece(c: &std::sync::atomic::AtomicU64, n: usize, owner: bool) -> Option<usize> {
    let mut v = c.load(Ordering::Acquire);
    loop {
        let (front, back) = ((v & 0xffff_ffff) as usize, (v >> 32) as usize);
        if front + back >= n {
            return None;
        }
        let (nv, i) = if owner { (v + 1, front) } else { (v + (1 << 32), n - 1 - back) };
        match c.compare_exchange_weak(v, nv, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return Some(i),
            Err(cur) => v = cur,
        }
    }
}
const ITEMS_PER_CCD: usize = 13; // the plan's cap

fn main() {
    let mode = arg("--mode").map(|p| p.to_str().unwrap().to_string()).unwrap_or("probe".into());
    let list = cpus(arg("--cpus").map(|p| p.to_str().unwrap().to_string()).as_deref().unwrap_or("1-23,25-47,49-71,73-95"));
    let mb = num("--mb", 12);
    let rounds = num("--rounds", 8);
    let inflight = num("--inflight", 4);
    let chunk = num("--chunk", 4096);
    let run_len = num("--run", 544 << 10);
    let piece = num("--piece", 64 << 10);
    let ahead = num("--ahead", 32);
    let round_mb = num("--round-mb", 14);
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
    // the runs' cursors a CCD (share mode): pieces claimed, the owner's count in the low word, the helpers' in the high
    let cursors: Vec<Vec<std::sync::atomic::AtomicU64>> = (0..16).map(|_| (0..256).map(|_| std::sync::atomic::AtomicU64::new(0)).collect()).collect();
    // each core's bytes this round, for the round's rate
    let round_done: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
    // each core's own time this round (ns), for the round's busy share: Σ core time / (cores × wall) — a tail is idle cores
    let round_ns: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
    let summed = Barrier::new(n);
    let all_rounds = std::sync::Mutex::new(Vec::<(f64, Vec<f64>, Vec<f64>, f64)>::new());
    let stop = AtomicBool::new(false);
    let start = Barrier::new(n);
    let round_end = Barrier::new(n);
    let per_core = std::sync::Mutex::new(vec![(0usize, 0f64); n]); // (cpu, best GB/s)
    let bufs: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
    let ready = Barrier::new(n);
    std::thread::scope(|sc| {
        for (t, &cpu) in list.iter().enumerate() {
            let (start, round_end, ready, per_core, queues, stop, bufs, mode, list, cursors, round_done, summed, best_round) = (&start, &round_end, &ready, &per_core, &queues, &stop, &bufs, mode.as_str(), &list, &cursors, &round_done, &summed, &all_rounds);
            let round_ns = &round_ns;
            sc.spawn(move || {
                harness::bench::pin(cpu);
                let mut buf = vec![0u8; bytes + (2 << 20)];
                for (i, b) in buf.iter_mut().enumerate().step_by(4096) {
                    *b = (i + t) as u8;
                }
                let base = (buf.as_ptr() as usize + (2 << 20) - 1) & !((2 << 20) - 1);
                bufs[t].store(base, Ordering::Release);
                // the L3 cleared of this buffer's zeroing: a scratch of 64 MB streamed twice (its own zeroing lands in
                // the L3 instead, and the scratch is never read again)
                {
                    let scratch = vec![1u8; 64 << 20];
                    for _ in 0..2 {
                        unsafe { touch::<16>(scratch.as_ptr(), scratch.len()) };
                    }
                }
                ready.wait();
                // this CCD's cores (indices into the list), for items
                let mates: Vec<usize> = (0..list.len()).filter(|&u| ccd_of(list[u]) == ccd_of(cpu)).collect();
                let my_rank = mates.iter().position(|&u| u == t).unwrap();
                let ccd_q = &queues[ccd_of(cpu) % 16];
                let mut best = 0f64;
                let curs = &cursors[ccd_of(cpu) % 16];
                for r in 0..rounds {
                    if my_rank == 0 {
                        ccd_q.store(0, Ordering::Relaxed);
                        for c in curs {
                            c.store(0, Ordering::Relaxed);
                        }
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
                        "triad" => {
                            let per = round_mb << 20;
                            assert!((r + 1) * per <= bytes, "--mb too small for {} rounds of fresh memory", rounds);
                            let third = per / 3 / 64 * 64;
                            let n = third / 8;
                            let p = base + r * per;
                            unsafe { triad(p as *mut f64, (p + third) as *const f64, (p + 2 * third) as *const f64, n, 3.0) };
                            done = 3 * third;
                        }
                        "fresh" | "probe-fresh" => {
                            let per = round_mb << 20;
                            assert!((r + 1) * per <= bytes, "--mb too small for {} rounds of fresh memory", rounds);
                            let p = (base + r * per) as *const u8;
                            if mode == "fresh" {
                                unsafe { touch_n(inflight, p, per) };
                            } else {
                                std::hint::black_box(unsafe { stream_sum(p, per) });
                            }
                            done = per;
                        }
                        "touch2" => {
                            unsafe {
                                match inflight {
                                    32 => touch2::<32>(base as *const u8, bytes),
                                    _ => touch2::<16>(base as *const u8, bytes),
                                }
                            };
                            done = bytes;
                        }
                        "pf" => {
                            unsafe {
                                match inflight {
                                    32 => touch_pf::<32>(base as *const u8, bytes, ahead),
                                    _ => touch_pf::<16>(base as *const u8, bytes, ahead),
                                }
                            };
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
                        "ranges" | "alt" => {
                            let total = per_ccd_items * SLICES_PER_ITEM * SLICE;
                            let n_runs = total / run_len;
                            let slab = n_runs.div_ceil(mates.len()) * run_len;
                            assert!((r + 1) * slab <= bytes, "--mb too small for {} rounds of fresh memory", rounds);
                            let mut k = my_rank;
                            while k < n_runs {
                                let m = if mode == "alt" { mates[(k / mates.len() + my_rank) % mates.len()] } else { t };
                                let p = (bufs[m].load(Ordering::Relaxed) + r * slab + (k / mates.len()) * run_len) as *const u8;
                                unsafe { touch_n(inflight, p, run_len) };
                                done += run_len;
                                k += mates.len();
                            }
                        }
                        "runs" | "share" => {
                            // the CCD's bytes a round as runs of `run_len`, run g in mate g % mates' buffer, each round in a
                            // fresh slab
                            let total = per_ccd_items * SLICES_PER_ITEM * SLICE;
                            let n_runs = total / run_len;
                            assert!(n_runs <= 256 && n_runs >= 1, "--run: 1..=256 runs a CCD");
                            let slab = n_runs.div_ceil(mates.len()) * run_len;
                            assert!((r + 1) * slab <= bytes, "--mb too small for {} rounds of fresh memory", rounds);
                            let at = |g: usize| (bufs[mates[g % mates.len()]].load(Ordering::Relaxed) + r * slab + (g / mates.len()) * run_len) as *const u8;
                            loop {
                                let k = ccd_q.fetch_add(1, Ordering::AcqRel);
                                if k >= n_runs {
                                    break;
                                }
                                if mode == "runs" {
                                    unsafe { touch_chunks(inflight, chunk, at(k), run_len, stop) };
                                    done += run_len;
                                } else {
                                    // this run in pieces from its front (the owner alone until a helper comes from the back)
                                    let n_pieces = run_len.div_ceil(piece);
                                    while let Some(i) = claim_piece(&curs[k], n_pieces, true) {
                                        let o = i * piece;
                                        unsafe { touch_chunks(inflight, chunk, at(k).add(o), piece.min(run_len - o), stop) };
                                        done += piece.min(run_len - o);
                                    }
                                }
                            }
                            if mode == "share" {
                                // the queue is empty: help the runs still in progress from their backs, the last-claimed first
                                let n_pieces = run_len.div_ceil(piece);
                                for k in (0..n_runs).rev() {
                                    while let Some(i) = claim_piece(&curs[k], n_pieces, false) {
                                        let o = i * piece;
                                        unsafe { touch_chunks(inflight, chunk, at(k).add(o), piece.min(run_len - o), stop) };
                                        done += piece.min(run_len - o);
                                    }
                                }
                            }
                        }
                        _ => panic!("mode {mode}: probe, touch, fresh, probe-fresh, triad, touch2, pf, chunks, items, static, slices, ranges, alt, runs, share"),
                    }
                    round_done[t].store(done, Ordering::Relaxed);
                    round_ns[t].store(t0.elapsed().as_nanos() as usize, Ordering::Relaxed);
                    round_end.wait();
                    // the round's wall is the slowest core's (all wait); thread 0 sums the round
                    let wall = t0.elapsed().as_secs_f64();
                    if t == 0 && r > 0 {
                        let per: Vec<f64> = (0..n).map(|u| round_done[u].load(Ordering::Relaxed) as f64 / wall / 1e9).collect();
                        let box_rate: f64 = per.iter().sum();
                        let mut by_ccd = vec![0f64; 16];
                        for (u, g) in per.iter().enumerate() {
                            by_ccd[ccd_of(list[u])] += g;
                        }
                        let busy = (0..n).map(|u| round_ns[u].load(Ordering::Relaxed) as f64).sum::<f64>() / (n as f64 * wall * 1e9);
                        all_rounds.lock().unwrap().push((box_rate, by_ccd, per, busy));
                    }
                    summed.wait();
                    best = best.max(done as f64 / wall / 1e9);
                }
                per_core.lock().unwrap()[t] = (cpu, best);
            });
        }
    });
    let _ = per_core.into_inner().unwrap();
    let mut rounds_v = all_rounds.into_inner().unwrap();
    rounds_v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (lo, hi) = (rounds_v[0].0, rounds_v[rounds_v.len() - 1].0);
    let (box_rate, by_ccd, per, busy) = rounds_v.swap_remove(rounds_v.len() / 2);
    let mut ccd_rates: Vec<f64> = ccds.iter().map(|&c| by_ccd[c]).collect();
    ccd_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut core_rates: Vec<f64> = per.clone();
    core_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = |v: &Vec<f64>| v[v.len() / 2];
    println!(
        "mode {mode} cores {n} ccds {} mb {mb} inflight {inflight} chunk {chunk} run {}K piece {}K: {:.0} GB/s box (rounds {:.0}–{:.0}), cores busy {:.0} %; a CCD {:.1}/{:.1}/{:.1}; a core {:.2}/{:.2}/{:.2} (min/med/max)",
        ccds.len(),
        run_len >> 10,
        piece >> 10,
        box_rate,
        lo,
        hi,
        busy * 100.0,
        ccd_rates[0], med(&ccd_rates), ccd_rates[ccd_rates.len() - 1],
        core_rates[0], med(&core_rates), core_rates[core_rates.len() - 1]
    );
}
