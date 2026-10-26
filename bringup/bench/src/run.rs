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
//! `--multi 1`: items run for the node's verify-step mix of tokens (`expert_items`, 2026-10-26: 86 % one token, 12 % two,
//! 2 % three), `q4::gemm` over n tokens, the count fixed by the item.
//!
//! `--smt 1`: each worker claims its next item before computing the current one, and when that next item is a cold one
//! (a CCD queue's) its SMT sibling (cpu + 96) pulls it into their shared L2 meanwhile with the node's touch; the sibling
//! waits in `mwaitx` on its slot between items (a spinning sibling takes ~10 % of the core's compute, metatron-cpu §3).
//!
//! `--fixed 1`: the same items every round (the planned slices and cold items do not advance through the pool): lines
//! read every round are protected in L3 (metatron-cpu §3), so the run's L3 delivery without DRAM lines.
//!
//! `--pages 4k|2m`: the pools' page size (default 2m, the node's).
//!
//! `--private 1`: slice s of a CCD always belongs to its worker s mod n (with `--fixed`, the same lines every round on the
//! same core, as `bench-order`'s private sets) and nobody steals: what ownership and stealing cost.
//!
//! Bisecting against `bench-order` (an L3-resident item 15 µs here, 5.5–5.9 there): `--notime 1` skips the per-item
//! counters and clock; `--plain 1` (with `--private 1`) runs a worker's own items as a plain loop, no deque claims.
//!
//! The ladder toward `bench-order`'s shape: `--notouch 1` (no window touching), `--front 1` (own items oldest first),
//! `--one-phase 1` (every item in one phase, no join), `--regions 1` (with `--private`: worker w's slices are a
//! contiguous region of the CCD's pool, as bench-order's private sets).
//!
//! `--stagger-ns x`: worker w (by its rank in its CCD) starts each phase w·x ns after the release.
//!
//! `--hold kb`: each worker keeps its L3-hit rate up through the window and its waits — `kb` KB of its own 1 MB ring every
//! 50 µs between touches (metatron-cpu §3.7: ~5 GB/s of L3 hits holds a core's L3 read rate; a run after a window of
//! DRAM touching reads its warm lines up to 4× slower otherwise).
//!
//! `--ring-kb n`: the hold ring's size (default 1024; must exceed the L2's 512 KB for its reads to be L3 hits).
//! `--retouch 1`: each worker re-reads every slice it touched once 512 KB of touching later (the node lab's re-touch,
//! construct.rs `RETOUCH_LAG`): L3 hits on the planned lines themselves — the hold with no footprint, promoting them too.
//!
//! `--ccd-barrier 1`: in each phase a CCD's workers finish every warm item (own, mates') before any of them claims a cold
//! one (a spin barrier a CCD): the warm reads with the CCD's DRAM path quiet, then the cold ones at full rate — the CCD
//! delivers the sum either way (099), so the overlap is worth nothing and the interference is.
//!
//! `--pad-kb n`: `n` KB of padding after every item in the pool (item stride 128 KB + n): eight cores reading regions or
//! queue-claimed slices at a 2 MiB stride in the same rhythm meet in the same L3 sets; a pad breaks the stride.
//!
//! `--main-pin c`: the main thread (it builds each phase's lists and spins in the barriers) pinned to core `c` (default:
//! unpinned, wherever the kernel puts it). Reported: the gap from a round's last worker end to the next release.
//!
//!   bench-run [--main-pin -] [--pad-kb 0] [--ccd-barrier 0] [--retouch 0] [--ring-kb 1024] [--hold 0] [--stagger-ns 0] [--notouch 0] [--front 0] [--one-phase 0] [--regions 0] [--notime 0] [--plain 0] [--private 0] [--pages 2m] [--fixed 0] [--smt 0] [--multi 0] [--cpus 49-71,73-95] [--steal any] [--plan 45] [--cold 28] [--window-us 440] [--rounds 60] [--pool-mb 600]
use harness::arg;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
#[path = "touch_node.rs"]
mod touch_node;
#[path = "huge.rs"]
mod huge;

const K: usize = 4096;
const ROWS: usize = 64;

fn num(name: &str, d: usize) -> usize {
    arg(name).map_or(d, |v| v.to_str().unwrap().parse().unwrap())
}
fn sys(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_default().trim().to_string()
}
/// Rest until `w` changes from `seen` or ~30 µs pass (the node's `wait_on`: monitorx, re-check, mwaitx with a timeout).
fn wait_change(w: &AtomicUsize, seen: usize) {
    unsafe {
        core::arch::asm!("monitorx", in("rax") w as *const _ as *const u8, in("ecx") 0u32, in("edx") 0u32, options(nostack, preserves_flags));
        if w.load(Ordering::Acquire) == seen {
            core::arch::asm!("push rbx", "mov ebx, {t:e}", "mwaitx", "pop rbx", t = in(reg) 100_000u32, in("eax") 0u32, in("ecx") 2u32, options(nostack, preserves_flags));
        }
    }
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
    let multi = num("--multi", 0) == 1;
    let smt = num("--smt", 0) == 1;
    let fixed = num("--fixed", 0) == 1;
    let private = num("--private", 0) == 1;
    let notime = num("--notime", 0) == 1;
    let plain = num("--plain", 0) == 1;
    let notouch = num("--notouch", 0) == 1;
    let front = num("--front", 0) == 1;
    let one_phase = num("--one-phase", 0) == 1;
    let regions = num("--regions", 0) == 1;
    let stagger_ns = num("--stagger-ns", 0) as u64;
    let hold_kb = num("--hold", 0);
    let ring_kb = num("--ring-kb", 1024);
    let retouch = num("--retouch", 0) == 1;
    let ccd_barrier = num("--ccd-barrier", 0) == 1;
    let pages_huge = !arg("--pages").is_some_and(|v| v.to_str() == Some("4k"));
    // `--share cores`: a CCD's planned slices and cold items in proportion to its workers (the node gives every CCD the
    // same bytes of every expert, whatever its count of workers)
    let share_cores = arg("--share").is_some_and(|v| v.to_str() == Some("cores"));
    // 256 MB a CCD: a round reads ≤ 16 MB a CCD, so a line comes back after ≥ 16 rounds through a 32 MB L3 — cold
    let pool_mb = num("--pool-mb", 256);
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
    let ccd_bars: Arc<Vec<Spin>> = Arc::new((0..nccd).map(|k| Spin { n: mates[k].len(), arrived: Line(AtomicUsize::new(0)), gen: Line(AtomicUsize::new(0)) }).collect());
    let t = cpus.len();
    let pmc_ok = {
        let _ = std::fs::write("/proc/sys/kernel/nmi_watchdog", "0");
        cpus.iter().all(|&c| common::pmc::Pmc::open(c).is_some_and(|p| p.program(&common::pmc::FILLS))) && common::pmc::allow_rdpmc()
    };
    println!("bench-run: tokens {}, siblings {}, steal {steal},", if multi { "the node's mix" } else { "one" }, if smt { "stream the next cold item" } else { "halted" });
    println!("bench-run: steal {steal}, {t} workers on {nccd} CCDs {:?} (nodes {:?}), plan {plan} slices + {cold} cold items a CCD, window {window} µs, {rounds} rounds; pmc {}", mates.iter().map(|m| m.len()).collect::<Vec<_>>(), ccd_node, if pmc_ok { "on" } else { "off" });

    // the item bytes: a pool a CCD, allocated and first-touched by one of its cores (its node's memory)
    let mut rng = common::testing::Rng::new(9);
    let w: Vec<i8> = (0..ROWS * K).map(|_| rng.i4()).collect();
    let (pw1, ps1) = common::q4::pack(ROWS, K, &w, &vec![0.01; ROWS * K / common::q4::GROUP]);
    let (ib, sb) = (pw1.len(), ps1.len());
    let pad = num("--pad-kb", 0) << 10;
    if let Some(c) = arg("--main-pin") {
        harness::bench::pin(c.to_str().unwrap().parse().unwrap());
    }
    let mut gaps: Vec<u64> = Vec::new();
    let mut last_end: Option<u64> = None;
    let stride = ib + pad;
    let ipool = (pool_mb << 20) / stride;
    // every CCD's pool filled at once (serially, 12 × 600 MB by one core each was most of a run's wall)
    let pools: Vec<(usize, usize)> = (0..nccd)
        .map(|k| {
            let cpu = cpus[mates[k][0]];
            let (pw1, ps1) = (pw1.clone(), ps1.clone());
            std::thread::spawn(move || {
                harness::bench::pin(cpu);
                // 2 MiB pages, first-touched here (this CCD's node), as the node's experts
                let wv = huge::alloc_pages(ipool * stride, pages_huge);
                for i in 0..ipool {
                    for (d, &b) in wv[i * stride..i * stride + pw1.len()].iter_mut().zip(pw1.iter()) {
                        *d = b ^ (i as u8 & 0x11);
                    }
                }
                let sv = huge::alloc_pages(ipool * ps1.len() * 4, pages_huge);
                let sf = unsafe { std::slice::from_raw_parts_mut(sv.as_mut_ptr() as *mut f32, ipool * ps1.len()) };
                for i in 0..ipool {
                    sf[i * ps1.len()..(i + 1) * ps1.len()].copy_from_slice(&ps1);
                }
                (wv.as_ptr() as usize, sv.as_ptr() as usize)
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    let pools: &'static [(usize, usize)] = Box::leak(pools.into_boxed_slice());
    println!("pools: {} MiB in 2 MiB pages of {} MiB", huge::huge_mib(), nccd * pool_mb * 129 / 128);
    // an item id: ccd << 32 | index in its pool
    let wptr = move |id: usize| (pools[id >> 32].0 + (id & 0xffff_ffff) * stride) as *const u8;
    let sptr = move |id: usize| (pools[id >> 32].1 + (id & 0xffff_ffff) * sb * 4) as *const f32;

    let ccdq: Arc<Vec<Queue>> = Arc::new((0..nccd).map(|_| Queue::new()).collect());
    let planq: Arc<Vec<Queue>> = Arc::new((0..nccd).map(|_| Queue::new()).collect());
    let deques: Arc<Vec<Deque>> = Arc::new((0..t).map(|_| Deque { items: Mutex::new(Vec::new()), fb: AtomicU64::new(0) }).collect());
    // a slice's taker this round (usize::MAX: none), by (ccd, slice)
    // a CCD's share: the same for every CCD (the node's), or in proportion to its workers
    let share = |k: usize, n: usize| if share_cores { (n * nccd * mates[k].len() + t / 2) / t } else { n };
    let plan_k: Vec<usize> = (0..nccd).map(|k| share(k, plan)).collect();
    let cold_k: Vec<usize> = (0..nccd).map(|k| share(k, cold)).collect();
    let takers: Arc<Vec<Vec<AtomicUsize>>> = Arc::new((0..nccd).map(|k| (0..plan_k[k]).map(|_| AtomicUsize::new(usize::MAX)).collect()).collect());
    let bar = Arc::new(Spin { n: t + 1, arrived: Line(AtomicUsize::new(0)), gen: Line(AtomicUsize::new(0)) });
    let done = Arc::new(AtomicUsize::new(0));
    // per phase: worker end ns a round; per claim class (own, stolen, ccd, other): items, ns, L3, DRAM
    let ends: Arc<Mutex<Vec<[Vec<u64>; 2]>>> = Arc::new(Mutex::new(Vec::new()));
    let cls: Arc<Vec<[AtomicU64; 5]>> = Arc::new((0..6).map(|_| [const { AtomicU64::new(0) }; 5]).collect());
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
            let ccd_bars = ccd_bars.clone();
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
                let (mut q, mut s, mut g) = (vec![0i8; 3 * K], vec![0f32; 3 * K / GROUP], vec![0i32; 3 * K / GROUP]);
                for i in 0..3 {
                    QAct { q: &mut q[i * K..(i + 1) * K], scale: &mut s[i * K / GROUP..(i + 1) * K / GROUP], gsum: &mut g[i * K / GROUP..(i + 1) * K / GROUP] }.quantize(&x);
                }
                let a = Act { q: &q[..K], scale: &s[..K / GROUP], gsum: &g[..K / GROUP] };
                let act = |i: usize| Act { q: &q[i * K..(i + 1) * K], scale: &s[i * K / GROUP..(i + 1) * K / GROUP], gsum: &g[i * K / GROUP..(i + 1) * K / GROUP] };
                let mut out = vec![0f32; 3 * ROWS];
                // the sibling: streams the item whose id (+ 1) is posted in its slot, into the L2 it shares with this core
                let slot: Arc<Line> = Arc::new(Line(AtomicUsize::new(0)));
                if smt {
                    let slot = slot.clone();
                    let quit_on = done.clone();
                    std::thread::spawn(move || {
                        harness::bench::pin(cpu + 96);
                        let mut seen = 0usize;
                        loop {
                            let v = slot.0.load(Ordering::Acquire);
                            if quit_on.load(Ordering::Relaxed) == usize::MAX {
                                return;
                            }
                            if v == seen {
                                wait_change(&slot.0, seen);
                                continue;
                            }
                            seen = v;
                            if v != 0 {
                                let id = v - 1;
                                unsafe {
                                    touch_node::touch(wptr(id), ib, None);
                                    touch_node::touch(sptr(id) as *const u8, sb * 4, None);
                                }
                            }
                        }
                    });
                }
                // this worker's per-class counts, its own (shared atomics across 92 cores bounced between the sockets
                // inside the phase), added to the totals once at the end
                let local = std::cell::RefCell::new([[0u64; 5]; 6]);
                // the hold: `hold_kb` KB of this ring (1 MB, L3-resident after the first pass) when 50 µs have passed
                let ring: Vec<u8> = (0..(ring_kb << 10)).map(|i| i as u8).collect();
                let ring_at = std::cell::Cell::new(0usize);
                let last_hold = std::cell::Cell::new(std::time::Instant::now());
                let hold = || {
                    if hold_kb > 0 && last_hold.get().elapsed() >= std::time::Duration::from_micros(50) {
                        let at = ring_at.get();
                        let end = (at + (hold_kb << 10)).min(ring.len());
                        unsafe { touch_node::touch(ring.as_ptr().add(at), end - at, None) };
                        ring_at.set(if end >= ring.len() { 0 } else { end });
                        last_hold.set(std::time::Instant::now());
                    }
                };
                // (demand fills from L3, L2-prefetch hits in L3, lines from DRAM)
                let read3 = || if pmc_ok { (common::pmc::rdpmc(0), common::pmc::rdpmc(4), common::pmc::rdpmc(1).wrapping_add(common::pmc::rdpmc(2)).wrapping_add(common::pmc::rdpmc(5))) } else { (0, 0, 0) };
                loop {
                    // round start: the window
                    bar.wait();
                    if done.load(Ordering::Acquire) == usize::MAX {
                        for (c, v) in local.borrow().iter().enumerate() {
                            for x in 0..5 {
                                cls[c][x].fetch_add(v[x], Ordering::Relaxed);
                            }
                        }
                        return;
                    }
                    let snap = planq[my_ccd].items.lock().unwrap().clone();
                    let (mut rt_queue, mut rt_bytes): (std::collections::VecDeque<(usize, usize)>, usize) = (Default::default(), 0);
                    let until = std::time::Instant::now() + std::time::Duration::from_micros(window);
                    while !notouch && std::time::Instant::now() < until {
                        let Some(sl) = planq[my_ccd].claim(&snap) else { break };
                        takers[my_ccd][sl & 0xffff].store(wi, Ordering::Relaxed);
                        let id = sl >> 16;
                        if retouch {
                            rt_queue.push_back((id, rt_bytes + 2 * ib));
                            rt_bytes += 2 * ib;
                            // the slices touched ≥ 512 KB ago, read again
                            while let Some(&(rid, at)) = rt_queue.front() {
                                if at + (512 << 10) > rt_bytes {
                                    break;
                                }
                                rt_queue.pop_front();
                                if !(unsafe { touch_node::touch(wptr(rid), ib, Some(until)) } && unsafe { touch_node::touch(wptr(rid + 1), ib, Some(until)) }) {
                                    break;
                                }
                            }
                        }
                        // a slice touched in 64 KB pieces with the hold between them
                        let mut ok = true;
                        let mut off = 0;
                        for h in 0..2 {
                            while ok && off < ib {
                                let n = (64 << 10).min(ib - off);
                                ok = unsafe { touch_node::touch(wptr(id + h).add(off), n, Some(until)) };
                                off += n;
                                hold();
                            }
                            off = 0;
                        }
                        ok = ok && unsafe { touch_node::touch(sptr(id) as *const u8, 2 * sb * 4, Some(until)) };
                        if !ok {
                            break;
                        }
                    }
                    while std::time::Instant::now() < until {
                        hold();
                        std::hint::spin_loop();
                    }
                    bar.wait(); // the main thread builds the run's queues
                    for phase in 0..2 {
                        bar.wait();
                        let t0 = t0s.load(Ordering::Acquire);
                        if stagger_ns > 0 {
                            let rank = mates_w.iter().filter(|&&m| m < wi).count() as u64;
                            let d = std::time::Duration::from_nanos(rank * stagger_ns);
                            let s0 = std::time::Instant::now();
                            while s0.elapsed() < d {
                                std::hint::spin_loop();
                            }
                        }
                        let pd: &PhaseData = unsafe { &*phase_data.load(Ordering::Acquire) };
                        let (own_s, ccd_s) = (&pd.own[wi], &pd.ccd);
                        let mut run = |id: usize, class: usize| {
                            let c0 = if notime { (0, 0, 0) } else { read3() };
                            let ts = std::time::Instant::now();
                            // the item's token count, fixed by its id: 86 / 12 / 2 %
                            let n = if !multi { 1 } else { match (id.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 57) % 50 { 0 => 3, 1..=6 => 2, _ => 1 } };
                            if n == 1 {
                                unsafe { common::q4::avx2::gemv::<K>(ROWS, wptr(id), sptr(id), &a, out.as_mut_ptr()) };
                            } else {
                                let op = out.as_mut_ptr() as usize;
                                unsafe { common::q4::avx2::gemm::<K>(ROWS, wptr(id), sptr(id), n, act, |i| (op as *mut f32).add(i * ROWS)) };
                            }
                            if notime {
                                return;
                            }
                            let dt = ts.elapsed().as_nanos() as u64;
                            let c1 = read3();
                            if measuring.load(Ordering::Relaxed) {
                                let c = &mut local.borrow_mut()[class];
                                c[0] += 1;
                                c[1] += dt;
                                c[2] += c1.0.wrapping_sub(c0.0) & ((1 << 48) - 1);
                                c[3] += c1.1.wrapping_sub(c0.1) & ((1 << 48) - 1);
                                c[4] += c1.2.wrapping_sub(c0.2) & ((1 << 48) - 1);
                            }
                        };
                        // the slot order as a sequence of claims: own (back), mates (front), this CCD's queue, the node's
                        // other CCDs, then farther as `--steal` allows
                        if plain {
                            if front {
                                for &id in own_s.iter() {
                                    run(id, 0);
                                }
                            } else {
                                for &id in own_s.iter().rev() {
                                    run(id, 0);
                                }
                            }
                        }
                        let mut stage = if plain { 1 } else { 0usize };
                        let mut claim_next = || -> Option<(usize, usize)> {
                            loop {
                                let n_m = mates_w.len();
                                let (n_node, n_far) = (node_ccds.len(), far_ccds.len());
                                let got = if stage == 0 {
                                    deques[wi].claim(!front).map(|i| (own_s[i], 0))
                                } else if stage <= n_m && private {
                                    None
                                } else if stage <= n_m {
                                    let m = mates_w[stage - 1];
                                    deques[m].claim(false).map(|i| (pd.own[m][i], 1))
                                } else if stage == n_m + 1 {
                                    ccdq[my_ccd].claim(&ccd_s[my_ccd]).map(|id| (id, 2))
                                } else if stage <= n_m + 1 + n_node {
                                    let k = node_ccds[stage - n_m - 2];
                                    ccdq[k].claim(&ccd_s[k]).map(|id| (id, 3))
                                } else if stage <= n_m + 1 + n_node + n_far {
                                    let k = far_ccds[stage - n_m - 2 - n_node];
                                    let class = if socket_of(ccd_node_w[k]) == socket_of(my_node) { 4 } else { 5 };
                                    ccdq[k].claim(&ccd_s[k]).map(|id| (id, class))
                                } else {
                                    return None;
                                };
                                if got.is_some() {
                                    return got;
                                }
                                stage += 1;
                                if ccd_barrier && stage == n_m + 1 {
                                    ccd_bars[my_ccd].wait();
                                }
                            }
                        };
                        if smt {
                            // one ahead: the next item claimed before this one runs; a cold next one to the sibling
                            let mut cur = claim_next();
                            while let Some((id, class)) = cur {
                                let next = claim_next();
                                if let Some((nid, ncl)) = next {
                                    if ncl >= 2 {
                                        slot.0.store(nid + 1, Ordering::Release);
                                    }
                                }
                                run(id, class);
                                cur = next;
                            }
                        } else {
                            while let Some((id, class)) = claim_next() {
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
        if fixed {
            cursor.iter_mut().for_each(|c| *c = 0);
        }
        // this round's items a CCD: plan slices (two items each), then the cold ones; gate/up two thirds, down a third
        let mut plan_slices: Vec<Vec<usize>> = Vec::new();
        let mut cold_items: Vec<Vec<usize>> = Vec::new();
        for k in 0..nccd {
            let mut pl = Vec::new();
            for s in 0..plan_k[k] {
                let i = cursor[k] % (ipool - 1);
                cursor[k] += 2;
                pl.push((k << 32) | i);
                let _ = s;
            }
            let mut co = Vec::new();
            for _ in 0..cold_k[k] {
                co.push((k << 32) | (cursor[k] % ipool));
                cursor[k] += 1;
            }
            plan_slices.push(pl);
            cold_items.push(co);
            for s in 0..plan_k[k] {
                takers[k][s].store(usize::MAX, Ordering::Relaxed);
            }
            planq[k].set((0..plan_k[k]).map(|s| (plan_slices[k][s] << 16) | s).collect());
        }
        measuring.store(r > 0, Ordering::Relaxed);
        bar.wait(); // window
        bar.wait(); // window over: build the run
        let mut phase_items: [Vec<(usize, usize, Option<usize>)>; 2] = [Vec::new(), Vec::new()];
        for k in 0..nccd {
            for s in 0..plan_k[k] {
                let n_m = mates[k].len();
                let tk = if private && regions { mates[k][(s * n_m / plan_k[k]).min(n_m - 1)] } else if private { mates[k][s % n_m] } else { takers[k][s].load(Ordering::Relaxed) };
                let ph = if s % 3 == 2 && !one_phase { 1 } else { 0 };
                for h in 0..2 {
                    phase_items[ph].push((k, plan_slices[k][s] + h, if tk == usize::MAX { None } else { Some(tk) }));
                }
            }
            for (j, &id) in cold_items[k].iter().enumerate() {
                phase_items[if j % 3 == 2 && !one_phase { 1 } else { 0 }].push((k, id, None));
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
            let rel = epoch.elapsed().as_nanos() as u64;
            if let Some(e) = last_end {
                if r > 0 {
                    gaps.push(rel.saturating_sub(e));
                }
            }
            t0s.store(rel, Ordering::Release);
            bar.wait(); // release the phase
            bar.wait(); // the join
            wall[ph] = (epoch.elapsed().as_nanos() as u64).saturating_sub(t0s.load(Ordering::Relaxed));
            last_end = Some(epoch.elapsed().as_nanos() as u64);
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
    gaps.sort_unstable();
    if !gaps.is_empty() {
        println!("gap from a phase's last end to the next release: p50 {:.1} µs, p90 {:.1}, max {:.1}", gaps[gaps.len() / 2] as f64 / 1e3, gaps[gaps.len() * 9 / 10] as f64 / 1e3, gaps[gaps.len() - 1] as f64 / 1e3);
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
        // each CCD's last worker, median over the rounds
        let per: Vec<String> = (0..nccd)
            .map(|k| {
                let mut v: Vec<u64> = ends.iter().map(|e| mates[k].iter().map(|&w| e[ph][w]).max().unwrap()).collect();
                format!("{}:{:.0}", mates[k].len(), med(&mut v))
            })
            .collect();
        println!("  each CCD's last end (workers:µs): {}", per.join(" "));
    }
    let n = rounds as f64;
    for (c, name) in ["own", "stolen", "CCD queue", "node's CCD", "socket's CCD", "far socket"].iter().enumerate() {
        let v: Vec<u64> = cls[c].iter().map(|x| x.load(Ordering::Relaxed)).collect();
        if v[0] > 0 {
            let (items, lines) = (v[0] as f64, (v[2] + v[3] + v[4]) as f64);
            println!("  {name:<10} {:>7.1} items a round, {:>6.2} µs an item, {:>5.1} % of lines from DRAM | lines an item: {:.0} demand L3 / {:.0} prefetch L3 / {:.0} DRAM", items / n, v[1] as f64 / items / 1e3, 100.0 * v[4] as f64 / lines.max(1.0), v[2] as f64 / items, v[3] as f64 / items, v[4] as f64 / items);
        }
    }
}
