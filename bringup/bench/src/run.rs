//! bench-run — the node's expert run as the node schedules it (099), on socket 1: its six CCDs, two NUMA nodes, each
//! CCD's cores in the pool's roles. A round (a layer):
//!
//! 1. **The window** (`--window-us`): each CCD's planned slices (`--plan` a CCD, two 136 KB items each, new lines from
//!    the CCD's node) are claimed from the CCD's queue by its cores and touched with the node's touch; a claimed slice's
//!    items belong to its taker (construct.rs `slot_tab`: a planned slice to the worker that touched it).
//! 2. **The run**, two phases with a join between (gate/up, then down: two thirds of the items, then one third): each
//!    worker claims in the node's slot order (`slot_order`) — its own deque from the back, its CCD mates' deques from the
//!    front, its CCD's queue (the planned slices nobody claimed, then `--cold` items a CCD nobody planned), the other
//!    CCDs of its node, then any. An item is the production one-token `gemv` over 64 rows × 4096.
//!
//! Reported over the rounds: each phase's wall (release to the last item done), the workers' end times (p10 / p50 / p90 /
//! max), and by claim (own, stolen, CCD queue, another CCD's): items a round, µs an item, lines from DRAM.
//!
//! `--cpus` the workers (default socket 1's pool cores; both sockets need the memory a resident node holds:
//! bringup/box/dev.cmdline.bench); `--steal node|socket|any`: how far a worker claims past its own CCD after its node's
//! CCDs — the node's own slot order ends at any queue (`ANY`), across the sockets.
//!
//!   bench-run [--cpus 49-71,73-95] [--steal any] [--plan 45] [--cold 28] [--window-us 440] [--rounds 60] [--pool-mb 600]
use harness::arg;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
#[path = "touch_node.rs"]
mod touch_node;

const K: usize = 4096;
const ROWS: usize = 64;

fn num(name: &str, d: usize) -> usize {
    arg(name).map_or(d, |v| v.to_str().unwrap().parse().unwrap())
}
fn sys(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_default().trim().to_string()
}
/// A barrier that spins (the node's workers watch a generation word and wake in well under a microsecond; a futex
/// barrier's wake is tens of microseconds and would be the run's start).
/// The arrivals and the generation on lines of their own: on one line every arrival invalidated it under 92 spinning
/// cores across the sockets (~75 µs a barrier).
#[repr(align(64))]
struct Line(AtomicUsize);
struct Spin {
    n: usize,
    arrived: Line,
    gen: Line,
}
impl Spin {
    fn wait(&self) {
        let g = self.gen.0.load(Ordering::Acquire);
        if self.arrived.0.fetch_add(1, Ordering::AcqRel) + 1 == self.n {
            self.arrived.0.store(0, Ordering::Relaxed);
            self.gen.0.store(g + 1, Ordering::Release);
        } else {
            while self.gen.0.load(Ordering::Acquire) == g {
                std::hint::spin_loop();
            }
        }
    }
}
/// A phase's lists, built by the main thread and shared read-only: each worker's deque, each CCD's queue.
struct PhaseData {
    own: Vec<Vec<usize>>,
    ccd: Vec<Vec<usize>>,
}
/// A CCD's queue: its items in order and a claim counter.
struct Queue {
    items: Mutex<Vec<usize>>,
    next: AtomicUsize,
    len: AtomicUsize,
}
impl Queue {
    fn new() -> Self {
        Queue { items: Mutex::new(Vec::new()), next: AtomicUsize::new(0), len: AtomicUsize::new(0) }
    }
    fn set(&self, v: Vec<usize>) {
        self.len.store(v.len(), Ordering::Relaxed);
        *self.items.lock().unwrap() = v;
        self.next.store(0, Ordering::Release);
    }
    fn claim(&self, snapshot: &[usize]) -> Option<usize> {
        let i = self.next.fetch_add(1, Ordering::AcqRel);
        snapshot.get(i).copied()
    }
}
/// A worker's deque: items in touch order, front and back packed (the pool's `own`).
struct Deque {
    items: Mutex<Vec<usize>>,
    fb: AtomicU64,
}
impl Deque {
    fn set(&self, v: Vec<usize>) {
        self.fb.store((v.len() as u64) << 32, Ordering::Release);
        *self.items.lock().unwrap() = v;
    }
    fn claim(&self, owner: bool) -> Option<usize> {
        let mut v = self.fb.load(Ordering::Acquire);
        loop {
            let (front, back) = (v as u32, (v >> 32) as u32);
            if front >= back {
                return None;
            }
            let (nv, i) = if owner { (v - (1 << 32), back - 1) } else { (v + 1, front) };
            match self.fb.compare_exchange_weak(v, nv, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Some(i as usize),
                Err(cur) => v = cur,
            }
        }
    }
}

fn main() {
    let plan = num("--plan", 45);
    let cold = num("--cold", 28);
    let window = num("--window-us", 440) as u64;
    let rounds = num("--rounds", 60);
    let pool_mb = num("--pool-mb", 600);
    // socket 1's pool cores (its housekeeping cores 48 and 72 left out, as in the node), grouped by L3 and node
    let cpus: Vec<usize> = {
        let spec = arg("--cpus").map(|p| p.to_str().unwrap().to_string()).unwrap_or("49-71,73-95".into());
        let mut v = Vec::new();
        for part in spec.split(',') {
            match part.split_once('-') {
                Some((a, b)) => v.extend(a.parse::<usize>().unwrap()..=b.parse::<usize>().unwrap()),
                None => v.push(part.parse().unwrap()),
            }
        }
        v
    };
    let steal = arg("--steal").map(|p| p.to_str().unwrap().to_string()).unwrap_or("any".into());
    let socket_of = |node: usize| node / 2;
    let ccd_of_cpu = |c: usize| sys(&format!("/sys/devices/system/cpu/cpu{c}/cache/index3/shared_cpu_list")).split([',', '-']).next().unwrap().parse::<usize>().unwrap();
    let mut keys: Vec<usize> = cpus.iter().map(|&c| ccd_of_cpu(c)).collect();
    keys.sort_unstable();
    keys.dedup();
    let nccd = keys.len();
    let ccd: Vec<usize> = cpus.iter().map(|&c| keys.iter().position(|&k| k == ccd_of_cpu(c)).unwrap()).collect();
    let node_of = |c: usize| (0..8).find(|n| std::path::Path::new(&format!("/sys/devices/system/node/node{n}/cpu{c}")).exists()).unwrap();
    let ccd_node: Vec<usize> = (0..nccd).map(|k| node_of(cpus[ccd.iter().position(|&x| x == k).unwrap()])).collect();
    let mates: Vec<Vec<usize>> = (0..nccd).map(|k| (0..cpus.len()).filter(|&w| ccd[w] == k).collect()).collect();
    let t = cpus.len();
    let pmc_ok = {
        let _ = std::fs::write("/proc/sys/kernel/nmi_watchdog", "0");
        cpus.iter().all(|&c| common::pmc::Pmc::open(c).is_some_and(|p| p.program(&common::pmc::FILLS))) && common::pmc::allow_rdpmc()
    };
    println!("bench-run: steal {steal}, {t} workers on {nccd} CCDs {:?} (nodes {:?}), plan {plan} slices + {cold} cold items a CCD, window {window} µs, {rounds} rounds; pmc {}", mates.iter().map(|m| m.len()).collect::<Vec<_>>(), ccd_node, if pmc_ok { "on" } else { "off" });

    // the item bytes: a pool a CCD, allocated and first-touched by one of its cores (its node's memory)
    let mut rng = common::testing::Rng::new(9);
    let w: Vec<i8> = (0..ROWS * K).map(|_| rng.i4()).collect();
    let (pw1, ps1) = common::q4::pack(ROWS, K, &w, &vec![0.01; ROWS * K / common::q4::GROUP]);
    let (ib, sb) = (pw1.len(), ps1.len());
    let ipool = (pool_mb << 20) / ib;
    let pools: Vec<(usize, usize)> = (0..nccd)
        .map(|k| {
            let cpu = cpus[mates[k][0]];
            let (pw1, ps1) = (pw1.clone(), ps1.clone());
            std::thread::spawn(move || {
                harness::bench::pin(cpu);
                let wv: Vec<u8> = (0..ipool).flat_map(|i| pw1.iter().map(move |&b| b ^ (i as u8 & 0x11))).collect();
                let sv: Vec<f32> = (0..ipool).flat_map(|_| ps1.iter().copied()).collect();
                let (a, b) = (wv.as_ptr() as usize, sv.as_ptr() as usize);
                std::mem::forget(wv);
                std::mem::forget(sv);
                (a, b)
            })
            .join()
            .unwrap()
        })
        .collect();
    let pools: &'static [(usize, usize)] = Box::leak(pools.into_boxed_slice());
    // an item id: ccd << 32 | index in its pool
    let wptr = move |id: usize| (pools[id >> 32].0 + (id & 0xffff_ffff) * ib) as *const u8;
    let sptr = move |id: usize| (pools[id >> 32].1 + (id & 0xffff_ffff) * sb * 4) as *const f32;

    let ccdq: Arc<Vec<Queue>> = Arc::new((0..nccd).map(|_| Queue::new()).collect());
    let planq: Arc<Vec<Queue>> = Arc::new((0..nccd).map(|_| Queue::new()).collect());
    let deques: Arc<Vec<Deque>> = Arc::new((0..t).map(|_| Deque { items: Mutex::new(Vec::new()), fb: AtomicU64::new(0) }).collect());
    // a slice's taker this round (usize::MAX: none), by (ccd, slice)
    let takers: Arc<Vec<Vec<AtomicUsize>>> = Arc::new((0..nccd).map(|_| (0..plan).map(|_| AtomicUsize::new(usize::MAX)).collect()).collect());
    let bar = Arc::new(Spin { n: t + 1, arrived: Line(AtomicUsize::new(0)), gen: Line(AtomicUsize::new(0)) });
    let done = Arc::new(AtomicUsize::new(0));
    // per phase: worker end ns a round; per claim class (own, stolen, ccd, other): items, ns, L3, DRAM
    let ends: Arc<Mutex<Vec<[Vec<u64>; 2]>>> = Arc::new(Mutex::new(Vec::new()));
    let cls: Arc<Vec<[AtomicU64; 4]>> = Arc::new((0..6).map(|_| [const { AtomicU64::new(0) }; 4]).collect());
    let walls: Arc<Mutex<Vec<[u64; 2]>>> = Arc::new(Mutex::new(Vec::new()));
    let measuring = Arc::new(AtomicBool::new(false));
    // each worker's end in each phase, a line a worker
    let round_end: Arc<Vec<[Line; 2]>> = Arc::new((0..t).map(|_| [Line(AtomicUsize::new(0)), Line(AtomicUsize::new(0))]).collect());
    // the phase's start (ns after `epoch`) and its lists, published through atomics as the node publishes a job (a mutex
    // here staggered 92 workers' starts by their lock handoffs across the sockets)
    let epoch = std::time::Instant::now();
    let t0s: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
    let phase_data: Arc<std::sync::atomic::AtomicPtr<PhaseData>> = Arc::new(std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()));

    let hs: Vec<_> = (0..t)
        .map(|wi| {
            let (ccdq, planq, deques, takers, bar, done, cls, measuring, round_end, t0s, phase_data) = (ccdq.clone(), planq.clone(), deques.clone(), takers.clone(), bar.clone(), done.clone(), cls.clone(), measuring.clone(), round_end.clone(), t0s.clone(), phase_data.clone());
            let (cpu, my_ccd, my_node) = (cpus[wi], ccd[wi], ccd_node[ccd[wi]]);
            let mates_w: Vec<usize> = mates[my_ccd].iter().copied().filter(|&m| m != wi).collect();
            let node_ccds: Vec<usize> = (0..nccd).filter(|&k| k != my_ccd && ccd_node[k] == my_node).collect();
            let ccd_node_w = ccd_node.clone();
            let far_ccds: Vec<usize> = (0..nccd)
                .filter(|&k| ccd_node[k] != my_node && match steal.as_str() { "node" => false, "socket" => socket_of(ccd_node[k]) == socket_of(my_node), _ => true })
                .collect();
            std::thread::spawn(move || {
                use common::q4::{Act, QAct, GROUP};
                harness::bench::pin(cpu);
                let mut rng = common::testing::Rng::new(100 + wi as u64);
                let x: Vec<f32> = (0..K).map(|_| rng.unit()).collect();
                let (mut q, mut s, mut g) = (vec![0i8; K], vec![0f32; K / GROUP], vec![0i32; K / GROUP]);
                QAct { q: &mut q, scale: &mut s, gsum: &mut g }.quantize(&x);
                let a = Act { q: &q, scale: &s, gsum: &g };
                let mut out = vec![0f32; ROWS];
                // this worker's per-class counts, its own (shared atomics across 92 cores bounced between the sockets
                // inside the phase), added to the totals once at the end
                let local = std::cell::RefCell::new([[0u64; 4]; 6]);
                let read2 = || if pmc_ok { (common::pmc::rdpmc(0).wrapping_add(common::pmc::rdpmc(4)), common::pmc::rdpmc(1).wrapping_add(common::pmc::rdpmc(2)).wrapping_add(common::pmc::rdpmc(5))) } else { (0, 0) };
                loop {
                    // round start: the window
                    bar.wait();
                    if done.load(Ordering::Acquire) == usize::MAX {
                        for (c, v) in local.borrow().iter().enumerate() {
                            for x in 0..4 {
                                cls[c][x].fetch_add(v[x], Ordering::Relaxed);
                            }
                        }
                        return;
                    }
                    let snap = planq[my_ccd].items.lock().unwrap().clone();
                    let until = std::time::Instant::now() + std::time::Duration::from_micros(window);
                    while std::time::Instant::now() < until {
                        let Some(sl) = planq[my_ccd].claim(&snap) else { break };
                        takers[my_ccd][sl & 0xffff].store(wi, Ordering::Relaxed);
                        let id = sl >> 16;
                        let ok = unsafe { touch_node::touch(wptr(id), 2 * ib, Some(until)) } && unsafe { touch_node::touch(sptr(id) as *const u8, 2 * sb * 4, Some(until)) };
                        if !ok {
                            break;
                        }
                    }
                    while std::time::Instant::now() < until {
                        std::hint::spin_loop();
                    }
                    bar.wait(); // the main thread builds the run's queues
                    for phase in 0..2 {
                        bar.wait();
                        let t0 = t0s.load(Ordering::Acquire);
                        let pd: &PhaseData = unsafe { &*phase_data.load(Ordering::Acquire) };
                        let (own_s, ccd_s) = (&pd.own[wi], &pd.ccd);
                        let mut run = |id: usize, class: usize| {
                            let c0 = read2();
                            let ts = std::time::Instant::now();
                            unsafe { common::q4::avx2::gemv::<K>(ROWS, wptr(id), sptr(id), &a, out.as_mut_ptr()) };
                            let dt = ts.elapsed().as_nanos() as u64;
                            let c1 = read2();
                            if measuring.load(Ordering::Relaxed) {
                                let c = &mut local.borrow_mut()[class];
                                c[0] += 1;
                                c[1] += dt;
                                c[2] += c1.0.wrapping_sub(c0.0) & ((1 << 48) - 1);
                                c[3] += c1.1.wrapping_sub(c0.1) & ((1 << 48) - 1);
                            }
                        };
                        while let Some(i) = deques[wi].claim(true) {
                            run(own_s[i], 0);
                        }
                        for &m in mates_w.iter() {
                            while let Some(i) = deques[m].claim(false) {
                                run(pd.own[m][i], 1);
                            }
                        }
                        while let Some(id) = ccdq[my_ccd].claim(&ccd_s[my_ccd]) {
                            run(id, 2);
                        }
                        for &k in node_ccds.iter() {
                            while let Some(id) = ccdq[k].claim(&ccd_s[k]) {
                                run(id, 3);
                            }
                        }
                        for &k in far_ccds.iter() {
                            let class = if socket_of(ccd_node_w[k]) == socket_of(my_node) { 4 } else { 5 };
                            while let Some(id) = ccdq[k].claim(&ccd_s[k]) {
                                run(id, class);
                            }
                        }
                        let end = (epoch.elapsed().as_nanos() as u64).saturating_sub(t0);
                        round_end[wi][phase].0.store(end as usize, Ordering::Release);
                        bar.wait(); // the join
                    }
                }
            })
        })
        .collect();

    let mut cursor = vec![0usize; nccd];
    for r in 0..rounds + 1 {
        // this round's items a CCD: plan slices (two items each), then the cold ones; gate/up two thirds, down a third
        let mut plan_slices: Vec<Vec<usize>> = Vec::new();
        let mut cold_items: Vec<Vec<usize>> = Vec::new();
        for k in 0..nccd {
            let mut pl = Vec::new();
            for s in 0..plan {
                let i = cursor[k] % (ipool - 1);
                cursor[k] += 2;
                pl.push((k << 32) | i);
                let _ = s;
            }
            let mut co = Vec::new();
            for _ in 0..cold {
                co.push((k << 32) | (cursor[k] % ipool));
                cursor[k] += 1;
            }
            plan_slices.push(pl);
            cold_items.push(co);
            for s in 0..plan {
                takers[k][s].store(usize::MAX, Ordering::Relaxed);
            }
            planq[k].set((0..plan).map(|s| (plan_slices[k][s] << 16) | s).collect());
        }
        measuring.store(r > 0, Ordering::Relaxed);
        bar.wait(); // window
        bar.wait(); // window over: build the run
        let mut phase_items: [Vec<(usize, usize, Option<usize>)>; 2] = [Vec::new(), Vec::new()];
        for k in 0..nccd {
            for s in 0..plan {
                let tk = takers[k][s].load(Ordering::Relaxed);
                let ph = if s % 3 == 2 { 1 } else { 0 };
                for h in 0..2 {
                    phase_items[ph].push((k, plan_slices[k][s] + h, if tk == usize::MAX { None } else { Some(tk) }));
                }
            }
            for (j, &id) in cold_items[k].iter().enumerate() {
                phase_items[if j % 3 == 2 { 1 } else { 0 }].push((k, id, None));
            }
        }
        let mut wall = [0u64; 2];
        for (ph, items) in phase_items.iter().enumerate() {
            let mut own: Vec<Vec<usize>> = vec![Vec::new(); t];
            let mut q: Vec<Vec<usize>> = vec![Vec::new(); nccd];
            for &(k, id, tk) in items {
                match tk {
                    Some(w) => own[w].push(id),
                    None => q[k].push(id),
                }
            }
            for (wi, v) in own.iter().enumerate() {
                deques[wi].fb.store((v.len() as u64) << 32, Ordering::Release);
            }
            for (k, v) in q.iter().enumerate() {
                ccdq[k].len.store(v.len(), Ordering::Relaxed);
                ccdq[k].next.store(0, Ordering::Release);
            }
            let old = phase_data.swap(Box::into_raw(Box::new(PhaseData { own, ccd: q })), Ordering::AcqRel);
            if !old.is_null() {
                drop(unsafe { Box::from_raw(old) });
            }
            t0s.store(epoch.elapsed().as_nanos() as u64, Ordering::Release);
            bar.wait(); // release the phase
            bar.wait(); // the join
            wall[ph] = (epoch.elapsed().as_nanos() as u64).saturating_sub(t0s.load(Ordering::Relaxed));
        }
        if r > 0 {
            walls.lock().unwrap().push(wall);
            let re: [Vec<u64>; 2] = std::array::from_fn(|ph| (0..t).map(|w| round_end[w][ph].0.load(Ordering::Acquire) as u64).collect());
            ends.lock().unwrap().push(re);
        }
    }
    done.store(usize::MAX, Ordering::Release);
    bar.wait();
    for h in hs {
        h.join().unwrap();
    }
    let walls = walls.lock().unwrap();
    let ends = ends.lock().unwrap();
    for ph in 0..2 {
        let mut wv: Vec<u64> = walls.iter().map(|w| w[ph]).collect();
        wv.sort_unstable();
        // the workers' end quantiles a round, then their medians over the rounds
        let mut qs = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for e in ends.iter() {
            let mut v = e[ph].clone();
            v.sort_unstable();
            for (j, f) in [0.1, 0.5, 0.9, 1.0].iter().enumerate() {
                qs[j].push(v[((v.len() - 1) as f64 * f) as usize]);
            }
        }
        let med = |v: &mut Vec<u64>| {
            v.sort_unstable();
            v[v.len() / 2] as f64 / 1e3
        };
        println!("phase {}: wall p50 {:.1} µs | worker ends p10 {:.1} p50 {:.1} p90 {:.1} max {:.1}", ["gate/up", "down"][ph], wv[wv.len() / 2] as f64 / 1e3, med(&mut qs[0]), med(&mut qs[1]), med(&mut qs[2]), med(&mut qs[3]));
    }
    let n = rounds as f64;
    for (c, name) in ["own", "stolen", "CCD queue", "node's CCD", "socket's CCD", "far socket"].iter().enumerate() {
        let v: Vec<u64> = cls[c].iter().map(|x| x.load(Ordering::Relaxed)).collect();
        if v[0] > 0 {
            println!("  {name:<10} {:>7.1} items a round, {:>6.2} µs an item, {:>5.1} % of lines from DRAM", v[0] as f64 / n, v[1] as f64 / v[0] as f64 / 1e3, 100.0 * v[3] as f64 / (v[2] + v[3]).max(1) as f64);
        }
    }
}
