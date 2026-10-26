//! bench-window — the node's prefetch window and run, with the re-touch (099). Each pinned core, every iteration (a
//! layer): `--plan` fresh items (new lines, from a pool far past the caches) are its share of the layer's predicted
//! experts; for `--window-us` it touches them in order (the abandon check every 4 KB, as the node's touch), and with
//! `--retouch-kb lag` touches again every range it touched at least `lag` KB earlier (out of its L2 by then: an L3 hit);
//! then every core runs, together: its touched items newest first (the node's owner pops its deque from the back), then
//! its untouched planned items, then `--cold` items nobody planned. Reported: the run's time (its slowest core), the
//! items a core touched in the window, the touched items' lines from DRAM at the run, the run's DRAM lines.
//!
//! `--oldest-first 1`: the run takes its touched items oldest first (the most bytes inserted after them: the most at risk).
//!
//! `--cold-nta 1`: the items nobody touched (read once, from DRAM) read with `prefetchnta` `--dist` bytes ahead: their
//! lines kept out of the L3 they would otherwise churn.
//!
//! `--smt a`: each core's SMT sibling (cpu + 96) streams the core's untouched items (the plan's rest, then the cold ones)
//! into their shared L2, up to `a` items ahead of the core, from the run's start (beside its warm items).
//!
//!   bench-window [--smt 0] [--cold-nta 0] [--dist 2048] [--oldest-first 0] [--cpus 49-71,73-95] [--plan 16] [--cold 4] [--window-us 440] [--retouch-kb 0] [--iters 100]
use harness::arg;
use std::sync::{Arc, Barrier, Mutex};
#[path = "gemv_hint.rs"]
mod gemv_hint;

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
/// Every line of `b` read once, the deadline checked every 4 KB; false when it passed first.
fn touch(b: &[u8], until: std::time::Instant) -> bool {
    let mut s = 0u64;
    let mut i = 0;
    while i < b.len() {
        if i % 4096 == 0 && std::time::Instant::now() >= until {
            std::hint::black_box(s);
            return false;
        }
        s = s.wrapping_add(unsafe { std::ptr::read_volatile(b.as_ptr().add(i)) } as u64);
        i += 64;
    }
    std::hint::black_box(s);
    true
}

fn main() {
    let cpus = list(arg("--cpus").map(|p| p.to_str().unwrap().to_string()).as_deref().unwrap_or("49-71,73-95"));
    let plan = num("--plan", 16);
    let nc = num("--cold", 4);
    let window = num("--window-us", 440) as u64;
    let lag = num("--retouch-kb", 0) << 10;
    let iters = num("--iters", 100);
    let oldest = num("--oldest-first", 0) == 1;
    let cold_nta = num("--cold-nta", 0) == 1;
    let smt = num("--smt", 0);
    // `--smt-idle 1`: the siblings spawned and spinning, never streaming (the control for what a busy sibling does)
    let smt_idle = num("--smt-idle", 0) == 1;
    let dist = num("--dist", 2048);
    let pool_mb = num("--pool-mb", 160);
    let t = cpus.len();
    let pmc_ok = {
        let _ = std::fs::write("/proc/sys/kernel/nmi_watchdog", "0");
        cpus.iter().all(|&c| common::pmc::Pmc::open(c).is_some_and(|p| p.program(&common::pmc::FILLS))) && common::pmc::allow_rdpmc()
    };
    // per iteration and core: the run's ns; per core: touched items, touched items' (L3 hits, DRAM lines), the run's DRAM lines
    let runs: Arc<Mutex<Vec<Vec<u64>>>> = Arc::new(Mutex::new(vec![vec![0; t]; iters]));
    let acc: Arc<Mutex<[u64; 5]>> = Arc::new(Mutex::new([0; 5]));
    let start = Arc::new(Barrier::new(t));
    let handles: Vec<_> = cpus
        .iter()
        .enumerate()
        .map(|(ti, &cpu)| {
            let (start, runs, acc) = (start.clone(), runs.clone(), acc.clone());
            std::thread::spawn(move || {
                use common::q4::{pack, Act, QAct, GROUP};
                use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
                harness::bench::pin(cpu);
                // the sibling streamer: a round's list (base address, item bytes, ids) and the core's position in it
                let sib_list: Arc<Mutex<Vec<(usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
                let (sib_gen, sib_pos, sib_quit) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicUsize::new(0)), Arc::new(std::sync::atomic::AtomicBool::new(false)));
                if smt_idle {
                    let quit = sib_quit.clone();
                    std::thread::spawn(move || {
                        harness::bench::pin(cpu + 96);
                        while !quit.load(Ordering::Relaxed) {
                            std::hint::spin_loop();
                        }
                    });
                }
                if smt > 0 {
                    let (list, gen, pos, quit) = (sib_list.clone(), sib_gen.clone(), sib_pos.clone(), sib_quit.clone());
                    std::thread::spawn(move || {
                        harness::bench::pin(cpu + 96);
                        let mut seen = 0;
                        while !quit.load(Ordering::Relaxed) {
                            let g = gen.load(Ordering::Acquire);
                            if g == seen {
                                std::hint::spin_loop();
                                continue;
                            }
                            seen = g;
                            let l = list.lock().unwrap().clone();
                            for (i, &(a, n)) in l.iter().enumerate() {
                                while i >= pos.load(Ordering::Acquire) + smt && gen.load(Ordering::Relaxed) == g {
                                    std::hint::spin_loop();
                                }
                                if gen.load(Ordering::Relaxed) != g {
                                    break;
                                }
                                let p = a as *const u8;
                                let mut x = 0u64;
                                let mut o = 0;
                                while o < n {
                                    x = x.wrapping_add(unsafe { std::ptr::read_volatile(p.add(o)) } as u64);
                                    o += 64;
                                }
                                std::hint::black_box(x);
                            }
                        }
                    });
                }
                let mut rng = common::testing::Rng::new(31 + ti as u64);
                let w: Vec<i8> = (0..ROWS * K).map(|_| rng.i4()).collect();
                let (pw1, ps1) = pack(ROWS, K, &w, &vec![0.01; ROWS * K / GROUP]);
                let (ib, sb) = (pw1.len(), ps1.len());
                let items = (pool_mb << 20) / ib;
                let pool_w: Vec<u8> = (0..items).flat_map(|i| pw1.iter().map(move |&b| b ^ (i as u8 & 0x11))).collect();
                let pool_s: Vec<f32> = (0..items).flat_map(|_| ps1.iter().copied()).collect();
                let x: Vec<f32> = (0..K).map(|_| rng.unit()).collect();
                let (mut q, mut s, mut g) = (vec![0i8; K], vec![0f32; K / GROUP], vec![0i32; K / GROUP]);
                QAct { q: &mut q, scale: &mut s, gsum: &mut g }.quantize(&x);
                let a = Act { q: &q, scale: &s, gsum: &g };
                let mut out = vec![0f32; ROWS];
                let read2 = || if pmc_ok { (common::pmc::rdpmc(4), common::pmc::rdpmc(5)) } else { (0, 0) };
                let mut cursor = 0usize;
                let mut mine = [0u64; 5];
                for it in 0..iters + 1 {
                    let planned: Vec<usize> = (0..plan).map(|j| (cursor + j) % items).collect();
                    let cold: Vec<usize> = (0..nc).map(|j| (cursor + plan + j) % items).collect();
                    cursor = (cursor + plan + nc) % items;
                    let item_bytes = |k: usize| (&pool_w[k * ib..(k + 1) * ib], unsafe { std::slice::from_raw_parts(pool_s.as_ptr().add(k * sb) as *const u8, sb * 4) });
                    start.wait();
                    // the window: touch the plan in order, re-touching what is `lag` behind
                    let until = std::time::Instant::now() + std::time::Duration::from_micros(window);
                    let mut done = 0usize;
                    let (mut touched, mut queue): (usize, std::collections::VecDeque<(usize, usize)>) = (0, Default::default());
                    'win: for &k in &planned {
                        let (bw, bs) = item_bytes(k);
                        if !(touch(bw, until) && touch(bs, until)) {
                            break;
                        }
                        done += 1;
                        touched += ib + sb * 4;
                        if lag > 0 {
                            queue.push_back((k, touched));
                            while let Some(&(r, at)) = queue.front() {
                                if at + lag > touched {
                                    break;
                                }
                                queue.pop_front();
                                let (rw, rs) = item_bytes(r);
                                if !(touch(rw, until) && touch(rs, until)) {
                                    break 'win;
                                }
                            }
                        }
                    }
                    while std::time::Instant::now() < until {
                        std::hint::spin_loop();
                    }
                    start.wait();
                    // the run: touched items newest first, then the plan's rest, then the cold ones
                    let rest: Vec<usize> = planned[done..].iter().chain(cold.iter()).copied().collect();
                    if smt > 0 {
                        *sib_list.lock().unwrap() = rest.iter().map(|&k| (pool_w.as_ptr() as usize + k * ib, ib)).collect();
                        sib_pos.store(0, Ordering::Release);
                        sib_gen.fetch_add(1, Ordering::AcqRel);
                    }
                    let t0 = std::time::Instant::now();
                    let mut run_dram = 0u64;
                    let mut item = |k: usize, warm: bool| {
                        let c0 = read2();
                        if !warm && cold_nta {
                            unsafe { gemv_hint::gemv_hint::<K>(ROWS, pool_w.as_ptr().add(k * ib), pool_s.as_ptr().add(k * sb), &a, out.as_mut_ptr(), 1, dist) };
                        } else {
                            unsafe { common::q4::avx2::gemv::<K>(ROWS, pool_w.as_ptr().add(k * ib), pool_s.as_ptr().add(k * sb), &a, out.as_mut_ptr()) };
                        }
                        let c1 = read2();
                        let (h, m) = ((c1.0.wrapping_sub(c0.0)) & ((1 << 48) - 1), (c1.1.wrapping_sub(c0.1)) & ((1 << 48) - 1));
                        run_dram += m;
                        if warm && it > 0 {
                            mine[1] += h;
                            mine[2] += m;
                        }
                    };
                    if oldest {
                        for &k in planned[..done].iter() {
                            item(k, true);
                        }
                    } else {
                        for &k in planned[..done].iter().rev() {
                            item(k, true);
                        }
                    }
                    for (i, &k) in rest.iter().enumerate() {
                        sib_pos.store(i, Ordering::Release);
                        item(k, false);
                    }
                    sib_pos.store(usize::MAX / 2, Ordering::Release);
                    let dt = t0.elapsed().as_nanos() as u64;
                    if it > 0 {
                        runs.lock().unwrap()[it - 1][ti] = dt;
                        mine[0] += done as u64;
                        mine[3] += run_dram;
                        mine[4] += 1;
                    }
                }
                let mut a2 = acc.lock().unwrap();
                for i in 0..5 {
                    a2[i] += mine[i];
                }
                sib_quit.store(true, Ordering::Relaxed);
                std::hint::black_box(out);
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let runs = runs.lock().unwrap();
    let mut span: Vec<u64> = runs.iter().map(|v| *v.iter().max().unwrap()).collect();
    span.sort();
    let a = acc.lock().unwrap();
    let n = a[4] as f64;
    println!(
        "bench-window: {t} cores, plan {plan} + cold {nc} items (136 KB) a core, window {window} µs, {} first, re-touch {}, cold reads {}, smt {smt}{} | run p50 {:.1} µs p90 {:.1} | touched {:.2} items a core | touched lines from DRAM {:.1} % | run DRAM lines a core {:.0}",
        if oldest { "oldest" } else { "newest" },
        if lag == 0 { "off".to_string() } else { format!("{} KB behind", lag >> 10) },
        if cold_nta { "nta" } else { "plain" },
        if smt_idle { " (idle siblings spinning)" } else { "" },
        span[span.len() / 2] as f64 / 1e3,
        span[span.len() * 9 / 10] as f64 / 1e3,
        a[0] as f64 / n,
        100.0 * a[2] as f64 / (a[1] + a[2]).max(1) as f64,
        a[3] as f64 / n
    );
}
