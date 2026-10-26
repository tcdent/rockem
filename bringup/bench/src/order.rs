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
//!   bench-order [--cpus 49-71,73-95] [--n 3] [--warm 9] [--cold 7] [--pool-mb 96] [--iters 200]
use harness::arg;
use std::sync::{Arc, Barrier, Mutex};

const K: usize = 4096;
const ROWS: usize = 64;

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
    let (nw, nc) = (num("--warm", 9), num("--cold", 7));
    let pool_mb = num("--pool-mb", 96);
    let iters = num("--iters", 200);
    let orders = [Order::WarmFirst, Order::ColdFirst, Order::Interleave, Order::WarmOnly, Order::ColdOnly];
    let t = cpus.len();
    // per iteration, per thread: the run's time (ns)
    let times: Arc<Mutex<Vec<Vec<Vec<u64>>>>> = Arc::new(Mutex::new(vec![vec![vec![0; t]; iters]; orders.len()]));
    let start = Arc::new(Barrier::new(t));
    let handles: Vec<_> = cpus
        .iter()
        .enumerate()
        .map(|(ti, &cpu)| {
            let (start, times) = (start.clone(), times.clone());
            std::thread::spawn(move || {
                use common::q4::{pack, Act, QAct, GROUP};
                harness::bench::pin(cpu);
                let mut rng = common::testing::Rng::new(77 + ti as u64);
                let w: Vec<i8> = (0..ROWS * K).map(|_| rng.i4()).collect();
                let sc: Vec<f32> = vec![0.01; ROWS * K / GROUP];
                let (pw1, ps1) = pack(ROWS, K, &w, &sc);
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
                let mut out = vec![0f32; n * ROWS];
                let op = out.as_mut_ptr() as usize;
                let item = |wp: &[u8], sp: &[f32], k: usize| unsafe {
                    common::q4::avx2::gemm::<K>(ROWS, wp.as_ptr().add(k * ib), sp.as_ptr().add(k * sb), n, act, |i| (op as *mut f32).add(i * ROWS));
                };
                let mut cursor = 0usize;
                let mut sink = 0u64;
                for (oi, &order) in orders.iter().enumerate() {
                    for it in 0..iters + 1 {
                        // the prefetch: this core touches its warm items; the cold ones are the pool's next, untouched for
                        // a pool's length (past every cache)
                        sink = sink.wrapping_add(touch(&warm_w));
                        let cold: Vec<usize> = (0..nc).map(|j| (cursor + j) % pool_items).collect();
                        cursor = (cursor + nc) % pool_items;
                        start.wait();
                        let t0 = std::time::Instant::now();
                        let warm = |k: usize| item(&warm_w, &warm_s, k);
                        let cold_run = |k: usize| item(&pool_w, &pool_s, cold[k]);
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
                }
                std::hint::black_box((sink, out));
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let times = times.lock().unwrap();
    let mb = |items: usize| (items * (ROWS * K / 2 + ROWS * K / 128 * 4)) as f64 / 1e6;
    println!("bench-order: {t} cores, {n} tokens, {nw} warm items ({:.2} MB) + {nc} cold ({:.2} MB) a core, {iters} iterations", mb(nw), mb(nc));
    println!("{:<12} {:>12} {:>12} {:>12} {:>16}", "order", "p50 µs", "p10 µs", "p90 µs", "cold GB/s (all)");
    for (oi, o) in orders.iter().enumerate() {
        let mut span: Vec<u64> = times[oi].iter().map(|v| *v.iter().max().unwrap()).collect();
        span.sort();
        let p = |q: f64| span[((span.len() - 1) as f64 * q) as usize] as f64 / 1e3;
        let cold = if matches!(o, Order::WarmOnly) { 0.0 } else { mb(nc) * t as f64 / (p(0.5) * 1e-6) / 1e3 };
        println!("{:<12} {:>12.1} {:>12.1} {:>12.1} {:>16.1}", format!("{o:?}"), p(0.5), p(0.1), p(0.9), cold);
    }
}
