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
//!   bench-order [--fresh] [--gap-us 0] [--cpus 49-71,73-95] [--n 3] [--rows 64] [--warm 9] [--cold 7] [--pool-mb 96] [--iters 200]
use harness::arg;
use std::sync::{Arc, Barrier, Mutex};

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
    let orders = [Order::WarmFirst, Order::ColdFirst, Order::Interleave, Order::WarmOnly, Order::ColdOnly];
    let t = cpus.len();
    // per iteration, per thread: the run's time (ns); per order: the warm and cold items' summed times and counts
    let times: Arc<Mutex<Vec<Vec<Vec<u64>>>>> = Arc::new(Mutex::new(vec![vec![vec![0; t]; iters]; orders.len()]));
    let class: Arc<Mutex<Vec<[u64; 4]>>> = Arc::new(Mutex::new(vec![[0; 4]; orders.len()]));
    // the fill counters (common::pmc::FILLS, the node's): per order and class (warm, cold), each item's six counts
    let fills: Arc<Mutex<Vec<[[u64; 6]; 2]>>> = Arc::new(Mutex::new(vec![[[0; 6]; 2]; orders.len()]));
    let pmc_ok = {
        let _ = std::fs::write("/proc/sys/kernel/nmi_watchdog", "0");
        let ok = cpus.iter().all(|&c| common::pmc::Pmc::open(c).is_some_and(|p| p.program(&common::pmc::FILLS)));
        ok && common::pmc::allow_rdpmc()
    };
    let start = Arc::new(Barrier::new(t));
    let handles: Vec<_> = cpus
        .iter()
        .enumerate()
        .map(|(ti, &cpu)| {
            let (start, times, class, fills) = (start.clone(), times.clone(), class.clone(), fills.clone());
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
                // WarmOnly's touch passes: [first, later] × (ns, L3 hits, DRAM lines, passes)
                let pass_acc = std::cell::Cell::new([[0u64; 4]; 2]);
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
                        } else {
                            sink = sink.wrapping_add(touch(&warm_w));
                        }
                        let cold: Vec<usize> = (0..nc).map(|j| (cursor + j) % pool_items).collect();
                        cursor = (cursor + nc) % pool_items;
                        start.wait();
                        if gap_us > 0 {
                            let g = std::time::Instant::now();
                            while g.elapsed().as_micros() < gap_us as u128 {
                                std::hint::spin_loop();
                            }
                        }
                        let t0 = std::time::Instant::now();
                        let warm = |k: usize| {
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
                            Order::WarmOnly => (0..nw).for_each(warm),
                            Order::ColdOnly => (0..nc).for_each(cold_run),
                        }
                        let dt = t0.elapsed().as_nanos() as u64;
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
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let times = times.lock().unwrap();
    let mb = |items: usize| (items * (rows * K / 2 + rows * K / 128 * 4)) as f64 / 1e6;
    let cl = class.lock().unwrap();
    println!("warm set: {}", if fresh { format!("fresh each iteration (touched from DRAM, {touches}×), gap {gap_us} µs") } else { "the same lines every iteration".into() });
    println!("bench-order: {t} cores, {n} tokens, items of {rows} rows, {nw0} ± {spread} warm ({:.2} MB) + {nc} cold ({:.2} MB) a core, {iters} iterations, cold hint {} at {dist} B", mb(nw0), mb(nc), ["none", "nta", "t0"][hint as usize]);
    println!("{:<12} {:>9} {:>9} {:>9} {:>15} {:>14} {:>14}", "order", "p50 µs", "p10 µs", "p90 µs", "cold GB/s (all)", "warm item µs", "cold item µs");
    let fl = fills.lock().unwrap();
    let lines = (rows * K / 2 + rows * K / 128 * 4) as f64 / 64.0;
    if pmc_ok {
        println!("lines an item from each source (an item is {lines:.0} lines): {}", common::pmc::FILL_NAMES.join(" / "));
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
}
