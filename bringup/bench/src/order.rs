//! bench-order — does the order of a layer's expert run matter when part of it is in L3 and part is not (090/097, the
//! "missed experts first" question)? Each pinned core gets a warm set of items it touches just before the run (as the
//! prefetch does: a read of every line, on the core that will run them) and a cold set drawn from a pool far past the
//! caches (as the experts nobody prefetched); every core starts the run together (the bus shared as in a layer), and
//! the run applies the production kernel (`common::q4::avx2::gemm`, `--n` tokens) to every item in one of three orders:
//! warm first (today's: the plan's order puts prefetched experts first), cold first, and interleaved. Reported: the
//! iteration's time (its slowest core), median over `--iters`, and warm-only / cold-only runs for reference.
//!
//! Defaults match a verify step's layer at 18k on metatron (2026-10-25: ~198 MB of host experts a layer over 92
//! cores, 56 % of the run's lines from L3): `--warm 9 --cold 7` items of 64 rows × 4096 (≈ 1.2 MB warm, 0.95 MB cold a
//! core). Run on socket 1's pool cores (their memory local).
//!
//! Each item is timed by its class (warm, cold): what an order does to each.
//!
//! `--fresh`: each iteration's warm items are new — the pool's next, last read a pool's length ago (past every cache) —
//! touched from DRAM and then run, as the node's prefetch touches a new layer's experts; without it the warm set is the
//! same lines every iteration (L3-resident from the iterations before, the touch adding nothing). `--gap-us g`: the touch
//! and the run `g` µs apart; `--touches t`: the fresh set touched `t` times over (a second pass hits it in L3) (the cards' phase between the node's prefetch and its run).
//!
//! `--shared 4k|2m`: every core's warm set is a slice of ONE allocation (one mmap, in 4 KiB or 2 MiB pages), each slice
//! still first-touched by its own core — bench-run's memory shape; default: a private Vec a core.
//!
//! `--spin 1`: the cores released into each run by a spin barrier (within nanoseconds of each other, as the node's workers
//! start on a generation word) instead of std's futex barrier (tens of µs of stagger).
//!
//! `--gap-mode pause|busy|mwaitx|l3|dram`: how a core spends `--gap-us` — spinning on `pause`, a register-only busy
//! loop, the node's `wait_on` (monitorx/mwaitx with its ~30 µs timeout, re-armed until the gap ends), reading a 1 MB
//! L3-resident buffer of its own over and over (L3 traffic, as a run's), or streaming fresh lines from the pool (DRAM
//! traffic, as the node's window); `l3one`: only the first listed core reads its L3 buffer, the rest `pause`; `dramnta`:
//! fresh lines streamed with `prefetchnta` (DRAM traffic that leaves the L3 alone); `l2`: a 192 KB L2-resident buffer
//! read over and over (L2 traffic, no L3 requests); `l3duty`: 50 µs slots, `--duty` percent of them reading the L3 buffer,
//! the rest `pause` (NB: a slot reads the whole 1 MB buffer once whenever its share is nonzero — ~40 µs — so this mode
//! only distinguishes some L3 traffic from none); `l3ring`: 50 µs slots, `--duty-kb` KB of a rotating 1 MB ring read in
//! each (a line re-read after a full ring, past the L2: an L3 hit), the rest `pause` — the least L3 traffic that holds
//! the state; `pause_l3`: `--gap-us` of pause, then `--gap2-us` of full-rate L3 reads (the recovery time after a decay). With a gap, WarmOnly also
//! reports the core clock (APERF/MPERF) over the gap and over the run, and the warm items' µs by position (first third /
//! middle / last third).
//!
//! `--pretouch 0`: no touch of the warm set before each run (the fixed set stays L3-resident from the runs before).
//!
//! `--ballast-mb n [--ballast-pages 2m|4k]`: before the run, a thread on the first listed core allocates and fills `n` MB
//! of pool exactly as bench-run does (one mapping, `MADV_HUGEPAGE` or not, never read again): does its presence slow the
//! cores' L3-fed reads?
//!
//! `--warmup n`: `n` untimed WarmOnly iterations before the orders (the cores' L3 read rate recovers from idle over
//! milliseconds of hits, metatron-cpu §3.8; the orders run in sequence, so without this WarmFirst — first — is measured
//! in the recovery and WarmOnly — fourth — in the steady state).
//!
//! `--chase 1`: the warm items are dependent-load pointer chases instead of gemvs — each core a fixed 2 MB buffer of
//! lines holding the index of the next (a random cycle), 2,176 dependent loads an item — so a gap's effect on L3-hit
//! LATENCY is measured apart from bandwidth (prefetchers cannot help a dependent chain; docs/l3-literature.md §5 B2).
//!
//! `--pmc fills2`: the second counter set (common::pmc::FILLS2: L2 fill latency and fills pending; l3-literature §5 B3) in
//! place of the fill sources — the printed six columns follow. `--pf-off bits`: Prefetch Control bits set on every
//! listed core for the run (0x08 = L2 stream off, 0x20 = up/down, 0x07 = the L1 prefetchers; §5 B4), restored at exit.
//!
//!   bench-order [--pmc fills] [--pf-off 0] [--chase 0] [--warmup 0] [--ballast-mb 0] [--pretouch 1] [--gap-mode pause] [--spin 0] [--shared none] [--fresh] [--gap-us 0] [--cpus 49-71,73-95] [--n 3] [--rows 64] [--warm 9] [--cold 7] [--pool-mb 96] [--iters 200]
use harness::arg;
use std::sync::{Arc, Barrier, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A barrier: std's (a futex wake) or a spinning one on padded lines.
#[repr(align(64))]
struct Line(AtomicUsize);
enum Bar {
    Futex(Barrier),
    Spin { n: usize, arrived: Line, gen: Line },
}
impl Bar {
    fn new(n: usize, spin: bool) -> Self {
        if spin { Bar::Spin { n, arrived: Line(AtomicUsize::new(0)), gen: Line(AtomicUsize::new(0)) } } else { Bar::Futex(Barrier::new(n)) }
    }
    fn wait(&self) {
        match self {
            Bar::Futex(b) => {
                b.wait();
            }
            Bar::Spin { n, arrived, gen } => {
                let g = gen.0.load(Ordering::Acquire);
                if arrived.0.fetch_add(1, Ordering::AcqRel) + 1 == *n {
                    arrived.0.store(0, Ordering::Relaxed);
                    gen.0.store(g + 1, Ordering::Release);
                } else {
                    while gen.0.load(Ordering::Acquire) == g {
                        std::hint::spin_loop();
                    }
                }
            }
        }
    }
}

const K: usize = 4096;

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
fn num(name: &str, d: usize) -> usize {
    arg(name).map_or(d, |v| v.to_str().unwrap().parse().unwrap())
}
/// Every line of `b` read once (the prefetch's touch): its lines into this core's caches.
fn touch(b: &[u8]) -> u64 {
    let mut s = 0u64;
    let mut i = 0;
    while i < b.len() {
        s = s.wrapping_add(unsafe { std::ptr::read_volatile(b.as_ptr().add(i)) } as u64);
        i += 64;
    }
    s
}

#[path = "gemv_hint.rs"]
mod gemv_hint;
#[path = "huge.rs"]
mod huge;
use gemv_hint::gemv_hint;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Order {
    WarmFirst,
    ColdFirst,
    Interleave,
    WarmOnly,
    ColdOnly,
}

fn main() {
    let cpus = list(arg("--cpus").map(|p| p.to_str().unwrap().to_string()).as_deref().unwrap_or("49-71,73-95"));
    let n = num("--n", 3);
    // rows an item (64: 128 KB of codes; the served gate/up pair is ~544 KB, 256 rows)
    let rows = num("--rows", 64);
    let (nw0, nc) = (num("--warm", 9), num("--cold", 7));
    // `--warm-spread s`: core i gets nw ± s warm items (spread evenly over the cores, the mean kept) — the served run's
    // own deques are uneven (each core took what it claimed during the prefetch), which staggers when cores reach their
    // cold items
    let spread = num("--warm-spread", 0);
    // `--cold-hint nta|t0` with `--dist` bytes: the cold items' reads prefetched non-temporally (or ordinarily) ahead
    // `--warm-hint nta|t0`: the warm items' reads prefetched non-temporally (or ordinarily) `--dist` ahead: does a run that
    // reads its L3 lines non-temporally leave them unprotected for the next round's fresh lines (099)?
    let whint: u8 = match arg("--warm-hint").map(|v| v.to_str().unwrap().to_string()).as_deref() { Some("nta") => 1, Some("t0") => 2, _ => 0 };
    let hint: u8 = match arg("--cold-hint").map(|v| v.to_str().unwrap().to_string()).as_deref() { Some("nta") => 1, Some("t0") => 2, _ => 0 };
    let dist = num("--dist", 2048);
    let pool_mb = num("--pool-mb", 96);
    let iters = num("--iters", 200);
    let fresh = arg("--fresh").is_some();
    let gap_us = num("--gap-us", 0) as u64;
    let touches = num("--touches", 1);
    let shared = arg("--shared").map(|v| v.to_str().unwrap().to_string()).unwrap_or("none".into());
    let orders = [Order::WarmFirst, Order::ColdFirst, Order::Interleave, Order::WarmOnly, Order::ColdOnly];
    let t = cpus.len();
    // per iteration, per thread: the run's time (ns); per order: the warm and cold items' summed times and counts
    let times: Arc<Mutex<Vec<Vec<Vec<u64>>>>> = Arc::new(Mutex::new(vec![vec![vec![0; t]; iters]; orders.len()]));
    let class: Arc<Mutex<Vec<[u64; 4]>>> = Arc::new(Mutex::new(vec![[0; 4]; orders.len()]));
    // the fill counters (common::pmc::FILLS, the node's): per order and class (warm, cold), each item's six counts
    let fills: Arc<Mutex<Vec<[[u64; 6]; 2]>>> = Arc::new(Mutex::new(vec![[[0; 6]; 2]; orders.len()]));
    let fills2 = arg("--pmc").is_some_and(|v| v.to_str() == Some("fills2"));
    let pf_off = u64::from_str_radix(arg("--pf-off").map(|v| v.to_str().unwrap().trim_start_matches("0x").to_string()).as_deref().unwrap_or("0"), 16).unwrap();
    let pmc_ok = {
        let _ = std::fs::write("/proc/sys/kernel/nmi_watchdog", "0");
        let set = if fills2 { &common::pmc::FILLS2 } else { &common::pmc::FILLS };
        let ok = cpus.iter().all(|&c| common::pmc::Pmc::open(c).is_some_and(|p| p.program(set)));
        ok && common::pmc::allow_rdpmc()
    };
    let (ibytes, sbytes) = (rows * K / 2, rows * K / 128 * 4);
    let per = (nw0 + spread) * ibytes + 64;
    let shared_w: Option<usize> = match shared.as_str() {
        "4k" => Some(huge::alloc_pages(t * per, false).as_ptr() as usize),
        "2m" => Some(huge::alloc_pages(t * per, true).as_ptr() as usize),
        _ => None,
    };
    let shared_s: Option<usize> = shared_w.map(|_| huge::alloc_pages(t * ((nw0 + spread) * sbytes + 64), shared == "2m").as_ptr() as usize);
    let spin = num("--spin", 0) == 1;
    let ballast_mb = num("--ballast-mb", 0);
    let warmup_iters = num("--warmup", 0);
    let chase = num("--chase", 0) == 1;
    if ballast_mb > 0 {
        let hugep = !arg("--ballast-pages").is_some_and(|v| v.to_str() == Some("4k"));
        let core0 = cpus[0];
        std::thread::spawn(move || {
            harness::bench::pin(core0);
            let b = huge::alloc_pages(ballast_mb << 20, hugep);
            for (i, x) in b.iter_mut().enumerate() {
                *x = (i as u8) ^ 0x11;
            }
            std::hint::black_box(&b);
        })
        .join()
        .unwrap();
        println!("ballast: {ballast_mb} MB filled on core {core0} ({} MiB of the process in 2 MiB pages)", huge::huge_mib());
    }
    let gap_mode = arg("--gap-mode").map(|v| v.to_str().unwrap().to_string()).unwrap_or("pause".into());
    let duty = num("--duty", 10);
    let pretouch = num("--pretouch", 1) == 1;
    let duty_kb = num("--duty-kb", 16);
    let gap2_us = num("--gap2-us", 0) as u64;
    // clocks: [gap aperf, gap mperf, run aperf, run mperf]; items by position: [ns, n] × 3
    let clocks: Arc<Mutex<[u64; 4]>> = Arc::new(Mutex::new([0; 4]));
    let bypos: Arc<Mutex<[[u64; 2]; 3]>> = Arc::new(Mutex::new([[0; 2]; 3]));
    // --pf-off: the prefetchers' control bits on the bench cores, restored at the end
    let pf_old: Vec<(usize, u64)> = if pf_off != 0 { cpus.iter().filter_map(|&c| common::pmc::prefetch_control(c, pf_off).map(|o| (c, o))).collect() } else { Vec::new() };
    if pf_off != 0 {
        println!("prefetch control: bits {pf_off:#x} set on {} of {} cores (previous value {:#x})", pf_old.len(), cpus.len(), pf_old.first().map_or(0, |x| x.1));
    }
    let start = Arc::new(Bar::new(t, spin));
    let handles: Vec<_> = cpus
        .iter()
        .enumerate()
        .map(|(ti, &cpu)| {
            let (start, times, class, fills) = (start.clone(), times.clone(), class.clone(), fills.clone());
            let (clocks, bypos, gap_mode) = (clocks.clone(), bypos.clone(), gap_mode.clone());
            let nw = if spread == 0 { nw0 } else { (nw0 + spread).saturating_sub((ti * (2 * spread + 1)) / t.max(1)).max(1) };
            std::thread::spawn(move || {
                use common::q4::{pack, Act, QAct, GROUP};
                harness::bench::pin(cpu);
                let mut rng = common::testing::Rng::new(77 + ti as u64);
                let w: Vec<i8> = (0..rows * K).map(|_| rng.i4()).collect();
                let sc: Vec<f32> = vec![0.01; rows * K / GROUP];
                let (pw1, ps1) = pack(rows, K, &w, &sc);
                let (ib, sb) = (pw1.len(), ps1.len());
                let mk = |count: usize| -> (Vec<u8>, Vec<f32>) {
                    ((0..count).flat_map(|i| pw1.iter().map(move |&b| b ^ (i as u8 & 0x11))).collect(), (0..count).flat_map(|_| ps1.iter().copied()).collect())
                };
                let pool_items = ((pool_mb << 20) / ib).max(nc * 4);
                let (warm_w, warm_s) = mk(nw);
                // --shared: this core's slice of the one allocation, written (first-touched) here
                let (warm_w, warm_s): (Vec<u8>, Vec<f32>) = match shared_w {
                    None => (warm_w, warm_s),
                    Some(base) => {
                        let wp = unsafe { std::slice::from_raw_parts_mut((base + ti * per) as *mut u8, nw * ib) };
                        wp.copy_from_slice(&warm_w);
                        let sp = unsafe { std::slice::from_raw_parts_mut((shared_s.unwrap() + ti * ((nw0 + spread) * sbytes + 64)) as *mut f32, nw * sb) };
                        sp.copy_from_slice(&warm_s);
                        // Vecs over the slices, never freed (the bench ends with the process)
                        unsafe { (Vec::from_raw_parts(wp.as_mut_ptr(), wp.len(), wp.len()), Vec::from_raw_parts(sp.as_mut_ptr(), sp.len(), sp.len())) }
                    }
                };
                let (pool_w, pool_s) = mk(pool_items);
                let (mut q, mut s, mut g) = (vec![0i8; n * K], vec![0f32; n * K / GROUP], vec![0i32; n * K / GROUP]);
                for i in 0..n {
                    let x: Vec<f32> = (0..K).map(|_| rng.unit()).collect();
                    QAct { q: &mut q[i * K..(i + 1) * K], scale: &mut s[i * K / GROUP..(i + 1) * K / GROUP], gsum: &mut g[i * K / GROUP..(i + 1) * K / GROUP] }.quantize(&x);
                }
                let act = |i: usize| Act { q: &q[i * K..(i + 1) * K], scale: &s[i * K / GROUP..(i + 1) * K / GROUP], gsum: &g[i * K / GROUP..(i + 1) * K / GROUP] };
                let mut out = vec![0f32; n * rows];
                let op = out.as_mut_ptr() as usize;
                let item = |wp: &[u8], sp: &[f32], k: usize| unsafe {
                    if n == 1 {
                        common::q4::avx2::gemv::<K>(rows, wp.as_ptr().add(k * ib), sp.as_ptr().add(k * sb), &act(0), op as *mut f32);
                    } else {
                        common::q4::avx2::gemm::<K>(rows, wp.as_ptr().add(k * ib), sp.as_ptr().add(k * sb), n, act, |i| (op as *mut f32).add(i * rows));
                    }
                };
                // each item timed by its class: what interleaving does to the warm and to the cold items
                let (wsum, csum) = (std::cell::Cell::new([0u64; 2]), std::cell::Cell::new([0u64; 2]));
                let (wfill, cfill) = (std::cell::Cell::new([0u64; 6]), std::cell::Cell::new([0u64; 6]));
                let sink_chase = std::cell::Cell::new(0u64);
                let read6 = || -> [u64; 6] { if pmc_ok { std::array::from_fn(|i| common::pmc::rdpmc(i as u32)) } else { [0; 6] } };
                let timed = |cell: &std::cell::Cell<[u64; 2]>, fc: &std::cell::Cell<[u64; 6]>, f: &dyn Fn()| {
                    let p0 = read6();
                    let t = std::time::Instant::now();
                    f();
                    let dt = t.elapsed().as_nanos() as u64;
                    let p1 = read6();
                    let mut c = cell.get();
                    c[0] += dt;
                    c[1] += 1;
                    cell.set(c);
                    let mut x = fc.get();
                    for i in 0..6 {
                        x[i] += p1[i].wrapping_sub(p0[i]) & ((1 << 48) - 1);
                    }
                    fc.set(x);
                };
                let mut cursor = 0usize;
                let mut sink = 0u64;
                // --chase: a 2 MB buffer of u32 line indices forming one random cycle (64 B a line, 32768 lines), its
                // chase of 2,176 lines starting at line k·2176 of the cycle order
                let chase_lines = (2usize << 20) / 64;
                let chase_buf: Vec<u64> = if chase {
                    let mut order: Vec<u32> = (0..chase_lines as u32).collect();
                    let mut r = common::testing::Rng::new(1234 + ti as u64);
                    for i in (1..order.len()).rev() {
                        let j = (r.i4() as i64 + 8) as usize * 7919 % (i + 1);
                        order.swap(i, j);
                    }
                    let mut b = vec![0u64; chase_lines * 8];
                    for i in 0..chase_lines {
                        b[order[i] as usize * 8] = order[(i + 1) % chase_lines] as u64;
                    }
                    b
                } else {
                    Vec::new()
                };
                let chase_item = |k: usize| -> u64 {
                    let mut at = (k * 2176 % chase_lines) as u64;
                    for _ in 0..2176 {
                        at = unsafe { std::ptr::read_volatile(chase_buf.as_ptr().add(at as usize * 8)) };
                    }
                    at
                };
                let gapbuf: Vec<u8> = (0..(1usize << 20)).map(|i| i as u8).collect();
                let gap_cursor = std::cell::Cell::new(pool_items / 2);
                let ring_at = std::cell::Cell::new(0usize);
                // WarmOnly's touch passes: [first, later] × (ns, L3 hits, DRAM lines, passes)
                let pass_acc = std::cell::Cell::new([[0u64; 4]; 2]);
                for _ in 0..warmup_iters {
                    start.wait();
                    for k in 0..nw {
                        if fresh { item(&pool_w, &pool_s, k % pool_items) } else { item(&warm_w, &warm_s, k) }
                    }
                    start.wait();
                }
                for (oi, &order) in orders.iter().enumerate() {
                    for it in 0..iters + 1 {
                        // the prefetch: this core touches its warm items; the cold ones are the pool's next, untouched for
                        // a pool's length (past every cache)
                        let wsel: Vec<usize> = if fresh { (0..nw).map(|j| (cursor + j) % pool_items).collect() } else { Vec::new() };
                        if fresh {
                            cursor = (cursor + nw) % pool_items;
                            for pass in 0..touches {
                                let (p0, tp) = (read6(), std::time::Instant::now());
                                for &k in &wsel {
                                    sink = sink.wrapping_add(touch(&pool_w[k * ib..(k + 1) * ib]));
                                    sink = sink.wrapping_add(touch(unsafe { std::slice::from_raw_parts(pool_s.as_ptr().add(k * sb) as *const u8, sb * 4) }));
                                }
                                if oi == 3 && it > 0 {
                                    let p1 = read6();
                                    let mut x = pass_acc.get();
                                    let q = pass.min(1);
                                    x[q][0] += tp.elapsed().as_nanos() as u64;
                                    x[q][1] += (p1[4].wrapping_sub(p0[4])) & ((1 << 48) - 1);
                                    x[q][2] += (p1[5].wrapping_sub(p0[5])) & ((1 << 48) - 1);
                                    x[q][3] += 1;
                                    pass_acc.set(x);
                                }
                            }
                        } else if pretouch {
                            sink = sink.wrapping_add(touch(&warm_w));
                        }
                        let cold: Vec<usize> = (0..nc).map(|j| (cursor + j) % pool_items).collect();
                        cursor = (cursor + nc) % pool_items;
                        start.wait();
                        let msr = |reg: u64| -> u64 {
                            use std::os::unix::fs::FileExt;
                            let f = std::fs::File::open(format!("/dev/cpu/{cpu}/msr")).expect("msr");
                            let mut b = [0u8; 8];
                            f.read_exact_at(&mut b, reg).expect("msr read");
                            u64::from_le_bytes(b)
                        };
                        let timed_run = gap_us > 0 && order == Order::WarmOnly && it > 0;
                        if gap_us > 0 {
                            let (a0, m0) = if timed_run { (msr(0xE8), msr(0xE7)) } else { (0, 0) };
                            let g = std::time::Instant::now();
                            let d = std::time::Duration::from_micros(gap_us);
                            match gap_mode.as_str() {
                                "busy" => {
                                    let mut x = 1u64;
                                    while g.elapsed() < d {
                                        for _ in 0..64 {
                                            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                                        }
                                    }
                                    std::hint::black_box(x);
                                }
                                "l3one" if ti == 0 => {
                                    let mut x = 0u64;
                                    while g.elapsed() < d {
                                        x = x.wrapping_add(touch(&gapbuf));
                                    }
                                    std::hint::black_box(x);
                                }
                                "dramnta" => {
                                    use std::arch::x86_64::*;
                                    let mut x = 0u64;
                                    while g.elapsed() < d {
                                        let k = gap_cursor.get();
                                        let b = &pool_w[k * ib..(k + 1) * ib];
                                        let mut i = 0;
                                        while i + 4096 <= b.len() {
                                            // prefetch the next 4 KB non-temporally, then read this 4 KB (hits the prefetched lines)
                                            let mut l = 0;
                                            while l < 4096 && i + 4096 + l < b.len() {
                                                unsafe { _mm_prefetch::<_MM_HINT_NTA>(b.as_ptr().add(i + 4096 + l) as *const i8) };
                                                l += 64;
                                            }
                                            x = x.wrapping_add(touch(&b[i..i + 4096]));
                                            i += 4096;
                                        }
                                        gap_cursor.set((k + 1) % pool_items);
                                    }
                                    std::hint::black_box(x);
                                }
                                "l2" => {
                                    let mut x = 0u64;
                                    while g.elapsed() < d {
                                        x = x.wrapping_add(touch(&gapbuf[..192 << 10]));
                                    }
                                    std::hint::black_box(x);
                                }
                                "l3duty" => {
                                    let mut x = 0u64;
                                    let slot = std::time::Duration::from_micros(50);
                                    let on = std::time::Duration::from_micros(duty as u64 * 50 / 100);
                                    while g.elapsed() < d {
                                        let s0 = std::time::Instant::now();
                                        while s0.elapsed() < on {
                                            x = x.wrapping_add(touch(&gapbuf));
                                        }
                                        while s0.elapsed() < slot {
                                            std::hint::spin_loop();
                                        }
                                    }
                                    std::hint::black_box(x);
                                }
                                "l3ring" => {
                                    let mut x = 0u64;
                                    let slot = std::time::Duration::from_micros(50);
                                    let step = (duty_kb << 10).min(gapbuf.len());
                                    while g.elapsed() < d {
                                        let s0 = std::time::Instant::now();
                                        let at = ring_at.get();
                                        let end = (at + step).min(gapbuf.len());
                                        x = x.wrapping_add(touch(&gapbuf[at..end]));
                                        ring_at.set(if end >= gapbuf.len() { 0 } else { end });
                                        while s0.elapsed() < slot {
                                            std::hint::spin_loop();
                                        }
                                    }
                                    std::hint::black_box(x);
                                }
                                "pause_l3" => {
                                    while g.elapsed() < d {
                                        std::hint::spin_loop();
                                    }
                                    let d2 = std::time::Duration::from_micros(gap2_us);
                                    let g2 = std::time::Instant::now();
                                    let mut x = 0u64;
                                    while g2.elapsed() < d2 {
                                        x = x.wrapping_add(touch(&gapbuf));
                                    }
                                    std::hint::black_box(x);
                                }
                                "l3" => {
                                    // this core's own 1 MB, read in full each pass (L3-resident after the first)
                                    let mut x = 0u64;
                                    while g.elapsed() < d {
                                        x = x.wrapping_add(touch(&gapbuf));
                                    }
                                    std::hint::black_box(x);
                                }
                                "dram" => {
                                    let mut x = 0u64;
                                    while g.elapsed() < d {
                                        let k = gap_cursor.get();
                                        x = x.wrapping_add(touch(&pool_w[k * ib..(k + 1) * ib]));
                                        gap_cursor.set((k + 1) % pool_items);
                                    }
                                    std::hint::black_box(x);
                                }
                                "mwaitx" => {
                                    let word = std::sync::atomic::AtomicU64::new(0);
                                    while g.elapsed() < d {
                                        unsafe {
                                            core::arch::asm!("monitorx", in("rax") &word as *const _ as *const u8, in("ecx") 0u32, in("edx") 0u32, options(nostack, preserves_flags));
                                            core::arch::asm!("push rbx", "mov ebx, {t:e}", "mwaitx", "pop rbx", t = in(reg) 100_000u32, in("eax") 0u32, in("ecx") 2u32, options(nostack, preserves_flags));
                                        }
                                    }
                                }
                                _ => {
                                    while g.elapsed() < d {
                                        std::hint::spin_loop();
                                    }
                                }
                            }
                            if timed_run {
                                let (a1, m1) = (msr(0xE8), msr(0xE7));
                                let mut c = clocks.lock().unwrap();
                                c[0] += a1 - a0;
                                c[1] += m1 - m0;
                            }
                        }
                        let (ra0, rm0) = if timed_run { (msr(0xE8), msr(0xE7)) } else { (0, 0) };
                        let t0 = std::time::Instant::now();
                        let warm = |k: usize| {
                            if chase {
                                return timed(&wsum, &wfill, &|| {
                                    sink_chase.set(sink_chase.get().wrapping_add(chase_item(k)));
                                });
                            }
                            timed(&wsum, &wfill, &|| {
                                let (wp, sp, kk) = if fresh { (&pool_w, &pool_s, wsel[k]) } else { (&warm_w, &warm_s, k) };
                                if n == 1 && whint != 0 {
                                    unsafe { gemv_hint::<K>(rows, wp.as_ptr().add(kk * ib), sp.as_ptr().add(kk * sb), &act(0), op as *mut f32, whint, dist) };
                                } else {
                                    item(wp, sp, kk);
                                }
                            })
                        };
                        let cold_item = |k: usize| unsafe {
                            if n == 1 && hint != 0 {
                                gemv_hint::<K>(rows, pool_w.as_ptr().add(cold[k] * ib), pool_s.as_ptr().add(cold[k] * sb), &act(0), op as *mut f32, hint, dist);
                            } else {
                                item(&pool_w, &pool_s, cold[k]);
                            }
                        };
                        let cold_run = |k: usize| timed(&csum, &cfill, &|| cold_item(k));
                        if it == 1 {
                            wsum.set([0; 2]);
                            csum.set([0; 2]);
                            wfill.set([0; 6]);
                            cfill.set([0; 6]);
                        }
                        match order {
                            Order::WarmFirst => {
                                (0..nw).for_each(warm);
                                (0..nc).for_each(cold_run);
                            }
                            Order::ColdFirst => {
                                (0..nc).for_each(cold_run);
                                (0..nw).for_each(warm);
                            }
                            Order::Interleave => {
                                let (mut a, mut b) = (0, 0);
                                while a < nw || b < nc {
                                    if b < nc && (b * nw <= a * nc || a >= nw) {
                                        cold_run(b);
                                        b += 1;
                                    } else {
                                        warm(a);
                                        a += 1;
                                    }
                                }
                            }
                            Order::WarmOnly => {
                                for k in 0..nw {
                                    let t1 = std::time::Instant::now();
                                    warm(k);
                                    if timed_run {
                                        let p = (k * 3 / nw).min(2);
                                        let mut b = bypos.lock().unwrap();
                                        b[p][0] += t1.elapsed().as_nanos() as u64;
                                        b[p][1] += 1;
                                    }
                                }
                            }
                            Order::ColdOnly => (0..nc).for_each(cold_run),
                        }
                        let dt = t0.elapsed().as_nanos() as u64;
                        if timed_run {
                            let (ra1, rm1) = (msr(0xE8), msr(0xE7));
                            let mut c = clocks.lock().unwrap();
                            c[2] += ra1 - ra0;
                            c[3] += rm1 - rm0;
                        }
                        if it > 0 {
                            times.lock().unwrap()[oi][it - 1][ti] = dt;
                        }
                        start.wait();
                    }
                    let (w2, c2) = (wsum.get(), csum.get());
                    let mut cl = class.lock().unwrap();
                    cl[oi][0] += w2[0];
                    cl[oi][1] += w2[1];
                    cl[oi][2] += c2[0];
                    cl[oi][3] += c2[1];
                    drop(cl);
                    let mut fl = fills.lock().unwrap();
                    let (wf, cf) = (wfill.get(), cfill.get());
                    for i in 0..6 {
                        fl[oi][0][i] += wf[i];
                        fl[oi][1][i] += cf[i];
                    }
                    drop(fl);
                    wsum.set([0; 2]);
                    csum.set([0; 2]);
                    wfill.set([0; 6]);
                    cfill.set([0; 6]);
                }
                if ti == 0 && fresh {
                    let x = pass_acc.get();
                    for (q, name) in [(0, "first"), (1, "later")] {
                        if x[q][3] > 0 {
                            let n = x[q][3] as f64;
                            println!("core {cpu}, WarmOnly, {name} touch pass: {:.1} µs, {:.0} L3 hits, {:.0} DRAM lines ({:.2} MB)", x[q][0] as f64 / n / 1e3, x[q][1] as f64 / n, x[q][2] as f64 / n, nw as f64 * (ib + 4 * sb) as f64 / 1e6);
                        }
                    }
                }
                std::hint::black_box((sink, out));
                if shared_w.is_some() {
                    // the Vecs over the shared slices are not malloc's: never freed
                    std::mem::forget(warm_w);
                    std::mem::forget(warm_s);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let times = times.lock().unwrap();
    let mb = |items: usize| (items * (rows * K / 2 + rows * K / 128 * 4)) as f64 / 1e6;
    let cl = class.lock().unwrap();
    if gap_us > 0 {
        let c = clocks.lock().unwrap();
        let b = bypos.lock().unwrap();
        let ghz = |a: u64, m: u64| 2.65 * a as f64 / m.max(1) as f64;
        let us = |x: [u64; 2]| x[0] as f64 / x[1].max(1) as f64 / 1e3;
        println!("gap {gap_us} µs as {gap_mode}: clock over the gap {:.2} GHz, over the WarmOnly run {:.2} GHz; warm item µs by position: first third {:.2}, middle {:.2}, last third {:.2}", ghz(c[0], c[1]), ghz(c[2], c[3]), us(b[0]), us(b[1]), us(b[2]));
    }
    if chase {
        println!("warm items: dependent-load pointer chases, 2,176 lines each over a fixed 2 MB cycle a core (latency, not bandwidth)");
    }
    println!("release: {}", if spin { "spin barrier" } else { "std (futex) barrier" });
    println!("warm memory: {}", match shared.as_str() { "4k" => "one shared allocation, 4 KiB pages", "2m" => "one shared allocation, 2 MiB pages", _ => "a private Vec a core" });
    println!("warm set: {}", if fresh { format!("fresh each iteration (touched from DRAM, {touches}×), gap {gap_us} µs") } else { "the same lines every iteration".into() });
    println!("bench-order: {t} cores, {n} tokens, items of {rows} rows, {nw0} ± {spread} warm ({:.2} MB) + {nc} cold ({:.2} MB) a core, {iters} iterations, cold hint {} at {dist} B", mb(nw0), mb(nc), ["none", "nta", "t0"][hint as usize]);
    println!("{:<12} {:>9} {:>9} {:>9} {:>15} {:>14} {:>14}", "order", "p50 µs", "p10 µs", "p90 µs", "cold GB/s (all)", "warm item µs", "cold item µs");
    let fl = fills.lock().unwrap();
    let lines = (rows * K / 2 + rows * K / 128 * 4) as f64 / 64.0;
    if pmc_ok {
        println!("lines an item from each source (an item is {lines:.0} lines): {}", if fills2 { common::pmc::FILL2_NAMES.join(" / ") } else { common::pmc::FILL_NAMES.join(" / ") });
        for (oi, o) in orders.iter().enumerate() {
            for (c, name) in [(0, "warm"), (1, "cold")] {
                let items = if c == 0 { cl[oi][1] } else { cl[oi][3] };
                if items == 0 {
                    continue;
                }
                let v: Vec<String> = (0..6).map(|i| format!("{:.0}", fl[oi][c][i] as f64 / items as f64)).collect();
                println!("  {:<12} {name}: {}", format!("{o:?}"), v.join(" / "));
            }
        }
    }
    for (oi, o) in orders.iter().enumerate() {
        let mut span: Vec<u64> = times[oi].iter().map(|v| *v.iter().max().unwrap()).collect();
        span.sort();
        let p = |q: f64| span[((span.len() - 1) as f64 * q) as usize] as f64 / 1e3;
        let cold = if matches!(o, Order::WarmOnly) { 0.0 } else { mb(nc) * t as f64 / (p(0.5) * 1e-6) / 1e3 };
        let per = |a: u64, b: u64| if b == 0 { 0.0 } else { a as f64 / b as f64 / 1e3 };
        println!("{:<12} {:>9.1} {:>9.1} {:>9.1} {:>15.1} {:>14.2} {:>14.2}", format!("{o:?}"), p(0.5), p(0.1), p(0.9), cold, per(cl[oi][0], cl[oi][1]), per(cl[oi][2], cl[oi][3]));
    }
    for (c, old) in pf_old {
        common::pmc::prefetch_control(c, old);
    }
}
