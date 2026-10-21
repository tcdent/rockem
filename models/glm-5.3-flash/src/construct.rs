//! Construction: a sequence's state per layer, the scratch allocated once,
//! the routed experts from the bundle's stream, the thread pool, and the CPU
//! realization — every op in this process, the reference placement every card
//! kernel is diffed against — with the step that ties the fused kernels
//! together one token at a time. `std` only; nothing on the token path allocates.

use crate::hc::gfx11_consts::ROUTE_ROWS;
use crate::ffn::Routed;
use crate::layout::{self, KDA_MEMORY, KDA_WINDOW, MLA_ROW, MLA_ROW_U16, POSITION, POSITION_W};
use crate::weights::{Attn, Ffn, Weights};
use crate::*;
use common::gemv::{self, widen};
use common::pool::Pool;
use common::rmsnorm::rmsnorm;
use common::trace::{Trace, NONE};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Worker threads pinned to `cores`, spinning for work. A job is one
/// allocation — the function, its items in node order with a counter per
/// node, and the done counter — published through one atomic pointer, so
/// whatever job a worker loads, it draws items from that job's own counters
/// and counts completions there: a worker that wakes late to a finished job
/// finds the counters spent and leaves. `run_at` sorts the items by the NUMA
/// node of the memory each streams, through the placement `place` was told
/// (the node of every 2 MiB page of a loaded file); a worker drains its own
/// node's items first, then the other nodes' — the weights stream at every
/// socket's local bandwidth, and the tail still balances. `run` publishes a
/// new job only when no worker is registered (`active`), and frees a job one
/// generation after replacing it, so a worker that registered during the swap
/// never reads freed memory. The pool has two states: busy — a step is in
/// flight, and a worker between jobs waits on the job pointer's cache line
/// with MWAITX (the core rests, and wakes within a microsecond of the next
/// publish; 95 cores spinning through the card phases heated the package for
/// nothing, 2026-10-15) — and idle, entered by `idle()` when the node is about
/// to wait on the wire, in which a worker sleeps between looks. `run` makes
/// the pool busy again.
pub struct Threads {
    shared: std::sync::Arc<Shared>,
    /// Jobs posted and superseded, freed at `idle` once no worker is inside any (a post never
    /// waits for the previous job's workers: that wait was 0.4 ms a token, the frees 3 ms on the
    /// card-driving thread, 2026-10-16).
    retired: std::sync::Mutex<Vec<usize>>,
    /// The slot order the calling thread drains (`slot_order`).
    order: Vec<usize>,
    /// The CCDs of each node, as global CCD indices.
    node_ccds: Vec<Vec<usize>>,
    /// The placed regions: base, length, the node of each `PAGE` of it.
    placed: std::sync::Mutex<Vec<(usize, usize, Vec<u8>)>>,
    /// The last run's (local items, items) and the last prefetch's (done, items), for the profile.
    locality: std::sync::Mutex<(usize, usize)>,
    prefetched: std::sync::Mutex<(usize, usize)>,
    /// The calling thread's node.
    node: usize,
    /// The workers, the calling thread counted first.
    workers: usize,
    /// Since the last `node_ends`: each node's items and the sum of its run-end times (s), over the placed runs.
    node_stats: std::sync::Mutex<([usize; MAX_NODES], [f64; MAX_NODES])>,
    /// Since the last `overheads`, over the placed runs: seconds slotting (entry to post), to the
    /// first claim (wake), and after the last node's end (tail); and the runs.
    overheads: std::sync::Mutex<(f64, f64, f64, usize)>,
    /// Since the last `overheads`: the sum over the placed runs of the last node's end (s).
    last_ends: std::sync::Mutex<f64>,
    /// Since the last `post_stats`, over the runs' posts: seconds sorting the items into slots, draining
    /// the previous job's workers, and allocating the job and freeing the retired one.
    post_stats: std::sync::Mutex<(f64, f64, f64)>,
}
/// The grain of placement: a region's node is known per 2 MiB page.
pub use crate::layout::PAGE;
const MAX_NODES: usize = 8;
const MAX_CCDS: usize = 16;
/// A job's items sit in slots: slot `c < MAX_CCDS` is CCD `c`'s (an item with a
/// preferred core complex), slot `MAX_CCDS + d` is node `d`'s (an item placed by its
/// rows' node only), slot `ANY` is for an item nothing places (no placed region holds
/// its rows): every worker drains it, last.
const ANY: usize = MAX_CCDS + MAX_NODES;
const SLOTS: usize = ANY + 1;
/// The decode prefetch's experts a layer at most, its best-ranked first (a CCD's L3 retains about this many; see the
/// prefetch's comment).
const PREFETCH_CAP: usize = 13;
struct Job {
    f: *const (dyn Fn(usize) + Sync),
    /// The items in slot order: slot `s`'s are `order[start[s]..start[s + 1]]`.
    order: Vec<u32>,
    start: [usize; SLOTS + 1],
    next: [Counter; SLOTS],
    done: Counter,
    /// Items done by a thread from its own first slot (its CCD's): the run's locality.
    local: Counter,
    /// Set when the job is abandoned (a prefetch the next run overtakes): no more items are claimed.
    stop: std::sync::atomic::AtomicBool,
    n: usize,
    /// Per node: items its workers completed, and when the last of them finished (ns after the
    /// job was posted) — the run's end on each node, the measurement the placement is tuned by.
    node_done: [Counter; MAX_NODES],
    node_end: [std::sync::atomic::AtomicU64; MAX_NODES],
    /// When the first item was claimed (ns after the job was posted): the workers' wake.
    first: std::sync::atomic::AtomicU64,
    /// `track-experts`: when each worker finished its last item of this job (ns after the post; 0: none taken),
    /// the main thread first — a run's tail read per core (074).
    #[cfg(feature = "track-experts")]
    core_end: Box<[std::sync::atomic::AtomicU64]>,
    /// `track-experts`, a prefetch: the workers' time inside its items summed, and how many workers took any.
    #[cfg(feature = "track-experts")]
    busy_ns: std::sync::atomic::AtomicU64,
    #[cfg(feature = "track-experts")]
    takers: AtomicUsize,
    /// `track-experts`: the post's wall clock and the last item's end (ns after the post), for the job's span.
    #[cfg(feature = "track-experts")]
    ts0: prow::Timestamp,
    #[cfg(feature = "track-experts")]
    last: std::sync::atomic::AtomicU64,
    t0: std::time::Instant,
    /// A prefetch's closure, owned by its job (the runs' closures are the caller's, alive through the run).
    _owned: Option<Box<dyn Fn(usize) + Send + Sync>>,
    /// A prefetch: only the first `PREFETCH_CORES` cores of each CCD take its items, so the pull
    /// leaves DRAM and the fabric room for the thread driving the cards.
    light: bool,
}
/// Cores a CCD lends a prefetch (of its eight): a CCD's L3 fills at ~60 GB/s and four cores
/// stream ~22 GB/s from DRAM, enough to pull a layer's share in the attention window.
const PREFETCH_CORES: usize = 4;
/// A counter on its own cache line: a node's cores share its `next`, nothing else bounces with it.
#[repr(align(64))]
struct Counter(AtomicUsize);
struct Shared {
    job: std::sync::atomic::AtomicPtr<Job>,
    active: AtomicUsize,
    busy: std::sync::atomic::AtomicBool,
    quit: std::sync::atomic::AtomicBool,
}
extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
}
pub fn pin(core: usize) {
    let mut mask = [0u64; 16];
    mask[core / 64] |= 1 << (core % 64);
    unsafe { sched_setaffinity(0, 128, mask.as_ptr()) };
}
/// Rest the core until a store lands on the line holding `job`, or ~30 µs pass:
/// arm the monitor, re-check (a store between the check and the wait would be
/// missed), then MWAITX in C1 with a cycle timeout. The x86-64 form; elsewhere a spin.
#[inline(always)]
fn wait_on(job: &std::sync::atomic::AtomicPtr<Job>, seen: *mut Job) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!("monitorx", in("rax") job as *const _ as *const u8, in("ecx") 0u32, in("edx") 0u32, options(nostack, preserves_flags));
        if job.load(Ordering::Acquire) == seen {
            core::arch::asm!("push rbx", "mov ebx, {t:e}", "mwaitx", "pop rbx", t = in(reg) 100_000u32, in("eax") 0u32, in("ecx") 2u32, options(nostack, preserves_flags));
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (job, seen);
        std::hint::spin_loop();
    }
}

thread_local! {
    /// The job this thread is inside, for `Pool::abandoned`: an item asks about its own job, not the pool's current one.
    static INSIDE: std::cell::Cell<*const Job> = const { std::cell::Cell::new(std::ptr::null()) };
}

/// The slots a thread on CCD `ccd` of node `node` drains, in order: its own CCD's, its node's,
/// its node's other CCDs'. Never another node's: a stream from across the fabric is served at
/// the cross-node rate (metatron: 165 GB/s for the box against 326 node-local), so idle cores
/// stealing a straggling node's items slowed the run they meant to finish (NPS2, 2026-10-15:
/// experts 30.5 → 23.1 ms a token without it). Work lands where its pages are.
fn slot_order(node: usize, ccd: usize, node_ccds: &[Vec<usize>]) -> Vec<usize> {
    let mut o = vec![ccd, MAX_CCDS + node];
    o.extend(node_ccds[node].iter().filter(|&&c| c != ccd));
    o.push(ANY);
    o
}
/// Every item of `job` this thread can claim, slot by slot in its order; the completions
/// counted once, at the end.
/// A zone on the expert pool's lane, `cpu` (prow's `zone!` takes the calling thread's lane; the pool's runs and the
/// expert phases are the pool's work, whichever thread drives them — lanes are devices: `main`, `cpu`, `card0`, `card1`).
#[cfg(feature = "track-zones")]
struct CpuZone {
    name: &'static str,
    start: prow::Timestamp,
}
#[cfg(feature = "track-zones")]
impl Drop for CpuZone {
    fn drop(&mut self) {
        prow::zone_at!("track-zones", self.name, "cpu", self.start, prow::Timestamp::now());
    }
}
macro_rules! cpu_zone {
    ($name:expr) => {
        #[cfg(feature = "track-zones")]
        let _cpu_zone = CpuZone { name: $name, start: prow::Timestamp::now() };
    };
}
unsafe fn work(job: *const Job, order: &[usize], node: usize, who: usize) {
    let j = &*job;
    // a prefetch's items only from this core's own CCD: a line touched here lands in this CCD's L3, so another CCD's item
    // done here is wasted for its run and evicts this CCD's own (the steal was counted done, 2026-10-21)
    let order = if j.light { &order[..1] } else { order };
    let f = &*j.f;
    INSIDE.with(|c| c.set(job));
    let (mut mine, mut local) = (0, 0);
    // a prefetch's time inside its items on this worker (`track-experts`)
    #[cfg(feature = "track-experts")]
    let (mut taken, mut busy0) = (false, std::time::Instant::now());
    for (k, &s) in order.iter().enumerate() {
        loop {
            if j.stop.load(Ordering::Relaxed) {
                break;
            }
            let i = j.next[s].0.fetch_add(1, Ordering::AcqRel);
            if i >= j.start[s + 1] {
                break;
            }
            if mine == 0 {
                j.first.fetch_min(j.t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                #[cfg(feature = "track-experts")]
                if j.light {
                    (taken, busy0) = (true, std::time::Instant::now());
                }
            }
            f(j.order[i] as usize);
            mine += 1;
            local += (k == 0) as usize;
        }
    }
    if mine > 0 {
        // this thread's last item ends its node's run as far as it is concerned (no cross-node stealing: its items were its node's)
        let end = j.t0.elapsed().as_nanos() as u64;
        #[cfg(feature = "track-experts")]
        j.core_end[who].store(end, Ordering::Relaxed);
        #[cfg(not(feature = "track-experts"))]
        let _ = who;
        j.node_end[node].fetch_max(end, Ordering::Relaxed);
        j.node_done[node].0.fetch_add(mine, Ordering::Relaxed);
        #[cfg(feature = "track-experts")]
        if taken {
            j.busy_ns.fetch_add(busy0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            j.takers.fetch_add(1, Ordering::Relaxed);
            j.last.fetch_max(end, Ordering::Relaxed);
        }
        j.done.0.fetch_add(mine, Ordering::AcqRel);
        j.local.0.fetch_add(local, Ordering::Relaxed);
    }
}
impl Threads {
    /// One worker per core after the first; the calling thread, pinned to `cores[0]`, is the other.
    pub fn new(cores: &[usize]) -> Self {
        assert!(!cores.is_empty());
        pin(cores[0]);
        // the core complexes: cores sharing an L3, numbered by their first core, grouped by node
        let mut ccd_keys: Vec<usize> = cores.iter().map(|&c| ccd_key(c)).collect();
        ccd_keys.sort_unstable();
        ccd_keys.dedup();
        assert!(ccd_keys.len() <= MAX_CCDS, "{} core complexes", ccd_keys.len());
        let ccd_of = |core: usize| ccd_keys.iter().position(|&k| k == ccd_key(core)).unwrap();
        let mut node_ccds: Vec<Vec<usize>> = vec![Vec::new(); MAX_NODES];
        for &c in cores {
            let (d, cc) = (node_of_core(c), ccd_of(c));
            if !node_ccds[d].contains(&cc) {
                node_ccds[d].push(cc);
            }
        }
        let shared = std::sync::Arc::new(Shared { job: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()), active: AtomicUsize::new(0), busy: std::sync::atomic::AtomicBool::new(false), quit: std::sync::atomic::AtomicBool::new(false) });
        let mut seen_in_ccd = vec![0usize; ccd_keys.len()];
        seen_in_ccd[ccd_of(cores[0])] += 1;
        for (who, &core) in cores.iter().enumerate().skip(1) {
            let sh = shared.clone();
            let node = node_of_core(core);
            let order = slot_order(node, ccd_of(core), &node_ccds);
            let rank = seen_in_ccd[ccd_of(core)];
            seen_in_ccd[ccd_of(core)] += 1;
            std::thread::spawn(move || {
                pin(core);
                let mut seen: *mut Job = std::ptr::null_mut();
                loop {
                    if sh.job.load(Ordering::Acquire) == seen {
                        if sh.quit.load(Ordering::Relaxed) {
                            return;
                        }
                        if sh.busy.load(Ordering::Relaxed) {
                            wait_on(&sh.job, seen);
                        } else {
                            // idle: a look a millisecond, so 95 sleeping threads cost the kernel nothing and
                            // the first run of a request waits at most that long for them
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                        continue;
                    }
                    // registered first, then the job as it is now: whatever it is, it is alive and its counters are its own
                    sh.active.fetch_add(1, Ordering::AcqRel);
                    let job = sh.job.load(Ordering::Acquire);
                    seen = job;
                    if !job.is_null() && !(unsafe { (*job).light } && rank >= PREFETCH_CORES) {
                        unsafe { work(job, &order, node, who) };
                    }
                    sh.active.fetch_sub(1, Ordering::AcqRel);
                }
            });
        }
        let node = node_of_core(cores[0]);
        let order = slot_order(node, ccd_of(cores[0]), &node_ccds);
        Threads { workers: cores.len(), shared, retired: std::sync::Mutex::new(Vec::new()), order, node_ccds, placed: std::sync::Mutex::new(Vec::new()), locality: std::sync::Mutex::new((0, 0)), prefetched: std::sync::Mutex::new((0, 0)), node, node_stats: std::sync::Mutex::new(([0; MAX_NODES], [0.0; MAX_NODES])), overheads: std::sync::Mutex::new((0.0, 0.0, 0.0, 0)), last_ends: std::sync::Mutex::new(0.0), post_stats: std::sync::Mutex::new((0.0, 0.0, 0.0)) }
    }
    /// Tell the pool where a region's memory lives: `nodes[p]` is the node of its `p`th `PAGE`.
    pub fn place(&self, base: usize, len: usize, nodes: Vec<u8>) {
        assert!(nodes.len() == len.div_ceil(PAGE) && nodes.iter().all(|&d| (d as usize) < MAX_NODES), "placement of {len} bytes: {} pages", nodes.len());
        self.placed.lock().unwrap().push((base, len, nodes));
    }
    /// The node is about to wait: the workers may sleep between looks until the next `run`; the
    /// jobs superseded since the last idle are freed once no worker is inside any.
    pub fn idle(&self) {
        self.shared.busy.store(false, Ordering::Relaxed);
        let retired: Vec<usize> = std::mem::take(&mut *self.retired.lock().unwrap());
        if !retired.is_empty() {
            while self.shared.active.load(Ordering::Acquire) != 0 {
                std::hint::spin_loop();
            }
            for j in retired {
                drop(unsafe { Box::from_raw(j as *mut Job) });
            }
        }
    }
    /// The items by slot (a counting sort over the placement: the node of each item's rows,
    /// and its preferred CCD within that node when it names one; every item is node 0's when
    /// nothing is placed or named), published, worked and waited for.
    fn publish(&self, n: usize, at: Option<&(dyn Fn(usize) -> *const u8 + Sync)>, node: Option<&(dyn Fn(usize) -> usize + Sync)>, ccd: Option<&(dyn Fn(usize) -> usize + Sync)>, slot: Option<&(dyn Fn(usize) -> usize + Sync)>, f: &(dyn Fn(usize) + Sync)) {
        if n <= 1 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        cpu_zone!("pool.run");
        let entered = std::time::Instant::now();
        let job = self.post(n, at, node, ccd, slot, f, None);
        unsafe { work(job, &self.order, self.node, 0) };
        while unsafe { (*job).done.0.load(Ordering::Acquire) } != n {
            std::hint::spin_loop();
        }
        #[cfg(feature = "track-experts")]
        if slot.is_some() {
            // the run's tail by core: when each worker finished, against the run's wall (074)
            let j = unsafe { &*job };
            let total = j.t0.elapsed().as_nanos() as u64;
            let mut ends: Vec<u64> = j.core_end.iter().map(|e| e.load(Ordering::Relaxed)).filter(|&e| e > 0).collect();
            ends.sort_unstable();
            let at = |q: f64| ends.get(((ends.len() as f64 - 1.0) * q) as usize).copied().unwrap_or(0) as f64 / 1e3;
            let ne = |d: usize| j.node_end[d].load(Ordering::Relaxed) as f64 / 1e3;
            prow::row!("track-experts", "pool_run", {
                "items" => n, "wall_us" => total as f64 / 1e3, "wake_us" => j.first.load(Ordering::Relaxed).min(total) as f64 / 1e3,
                "busy_cores" => ends.len(), "end_p10_us" => at(0.1), "end_p50_us" => at(0.5), "end_p90_us" => at(0.9), "end_max_us" => at(1.0),
                "node0_us" => ne(0), "node1_us" => ne(1), "node2_us" => ne(2), "node3_us" => ne(3),
            });
        }
        if ccd.is_some() || slot.is_some() {
            *self.locality.lock().unwrap() = (unsafe { (*job).local.0.load(Ordering::Relaxed) }, n);
            let j = unsafe { &*job };
            let mut st = self.node_stats.lock().unwrap();
            let mut end_max = 0u64;
            for d in 0..MAX_NODES {
                let done = j.node_done[d].0.load(Ordering::Relaxed);
                if done > 0 {
                    st.0[d] += done;
                    let e = j.node_end[d].load(Ordering::Relaxed);
                    st.1[d] += e as f64 * 1e-9;
                    end_max = end_max.max(e);
                }
            }
            let total = j.t0.elapsed().as_nanos() as u64;
            let mut o = self.overheads.lock().unwrap();
            o.0 += (j.t0 - entered).as_secs_f64();
            o.1 += j.first.load(Ordering::Relaxed).min(total) as f64 * 1e-9;
            o.2 += total.saturating_sub(end_max) as f64 * 1e-9;
            o.3 += 1;
            *self.last_ends.lock().unwrap() += end_max as f64 * 1e-9;
        }
    }
    /// The job slotted and published: the workers take it up; the job before it is stopped
    /// (a prefetch still running gives up its unclaimed items) and freed a generation late.
    fn post(&self, n: usize, at: Option<&(dyn Fn(usize) -> *const u8 + Sync)>, node: Option<&(dyn Fn(usize) -> usize + Sync)>, ccd: Option<&(dyn Fn(usize) -> usize + Sync)>, given: Option<&(dyn Fn(usize) -> usize + Sync)>, f: &(dyn Fn(usize) + Sync), owned: Option<Box<dyn Fn(usize) + Send + Sync>>) -> *mut Job {
        let entered = std::time::Instant::now();
        let mut start = [0usize; SLOTS + 1];
        let mut order = vec![0u32; n];
        let mut item_slot = Vec::new();
        let slot = |i: usize, d: usize| match ccd {
            Some(c) if !self.node_ccds[d].is_empty() => self.node_ccds[d][c(i) % self.node_ccds[d].len()] as u8,
            _ => (MAX_CCDS + d) as u8,
        };
        if let Some(given) = given {
            item_slot = (0..n).map(|i| given(i) as u8).collect();
        } else if let Some(node) = node {
            item_slot = (0..n).map(|i| slot(i, node(i))).collect();
        } else if let Some(at) = at {
            let placed = self.placed.lock().unwrap();
            if !placed.is_empty() {
                item_slot = (0..n)
                    .map(|i| {
                        let p = at(i) as usize;
                        let d = placed.iter().find(|(base, len, _)| p >= *base && p < base + len).map_or(0, |(base, _, nodes)| nodes[(p - base) / PAGE] as usize);
                        slot(i, d)
                    })
                    .collect();
            }
        }
        let slot_of = |i: usize| item_slot.get(i).map_or(ANY, |&s| s as usize);
        for i in 0..n {
            start[slot_of(i) + 1] += 1;
        }
        for s in 0..SLOTS {
            start[s + 1] += start[s];
        }
        let mut fill = start;
        for i in 0..n {
            let s = slot_of(i);
            order[fill[s]] = i as u32;
            fill[s] += 1;
        }
        let sh = &*self.shared;
        sh.busy.store(true, Ordering::Relaxed);
        let sorted = std::time::Instant::now();
        // the job before this one stops handing out items (it finished, or it was a prefetch); a worker
        // still inside it finishes its item on the job's own memory, which lives until `idle`
        let current = sh.job.load(Ordering::Acquire);
        if !current.is_null() {
            unsafe { (*current).stop.store(true, Ordering::Relaxed) };
        }
        let drained = std::time::Instant::now();
        if !current.is_null() && unsafe { (*current)._owned.is_some() } {
            // a prefetch it overtook: what it managed so far
            *self.prefetched.lock().unwrap() = unsafe { ((*current).done.0.load(Ordering::Relaxed), (*current).n) };
            // `track-experts`: the prefetch as a row when the next job overtakes it — its window (post to now), its items
            // done of its items, the workers' time inside them summed and how many took part (its cores' utilization)
            #[cfg(feature = "track-experts")]
            {
                let c = unsafe { &*current };
                let window = c.t0.elapsed().as_nanos() as u64;
                let first = c.first.load(Ordering::Relaxed);
                // its span on the pool's lane, `cpu` (the host's devices are lanes: `main` drives, `cpu` is the pool,
                // `card0`, `card1` the cards): its first item's start to its last item's end
                if first != u64::MAX {
                    let at = |ns: u64| prow::Timestamp::from_unix_nanos(c.ts0.unix_nanos() + ns as i64);
                    prow::zone_at!("track-experts", "prefetch", "cpu", at(first), at(c.last.load(Ordering::Relaxed).max(first)));
                }
                // its queues: how many had items, the fullest, and the most items a queue still held when overtaken
                let (mut q_used, mut q_max, mut q_left) = (0usize, 0usize, 0usize);
                for q in 0..SLOTS {
                    let len = c.start[q + 1] - c.start[q];
                    if len > 0 {
                        q_used += 1;
                        q_max = q_max.max(len);
                        q_left = q_left.max(c.start[q + 1].saturating_sub(c.next[q].0.load(Ordering::Relaxed)));
                    }
                }
                prow::row!("track-experts", "prefetch_job", { "q_used" => q_used, "q_max" => q_max, "q_left" => q_left, "items" => c.n, "done" => c.done.0.load(Ordering::Relaxed), "window_us" => window as f64 / 1e3, "first_us" => if first == u64::MAX { -1.0 } else { first as f64 / 1e3 }, "busy_us" => c.busy_ns.load(Ordering::Relaxed) as f64 / 1e3, "cores" => c.takers.load(Ordering::Relaxed) });
            }
        }
        let fp: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        let light = owned.is_some();
        let job = Box::into_raw(Box::new(Job { f: fp, order, next: std::array::from_fn(|s| Counter(AtomicUsize::new(start[s]))), start, done: Counter(AtomicUsize::new(0)), local: Counter(AtomicUsize::new(0)), stop: std::sync::atomic::AtomicBool::new(false), n, node_done: std::array::from_fn(|_| Counter(AtomicUsize::new(0))), node_end: std::array::from_fn(|_| std::sync::atomic::AtomicU64::new(0)), #[cfg(feature = "track-experts")] core_end: (0..self.workers).map(|_| std::sync::atomic::AtomicU64::new(0)).collect(), #[cfg(feature = "track-experts")] busy_ns: std::sync::atomic::AtomicU64::new(0), #[cfg(feature = "track-experts")] takers: AtomicUsize::new(0), #[cfg(feature = "track-experts")] ts0: prow::Timestamp::now(), #[cfg(feature = "track-experts")] last: std::sync::atomic::AtomicU64::new(0), first: std::sync::atomic::AtomicU64::new(u64::MAX), t0: std::time::Instant::now(), _owned: owned, light }));
        let old = sh.job.swap(job, Ordering::AcqRel);
        if !old.is_null() {
            self.retired.lock().unwrap().push(old as usize);
        }
        if !light {
            let mut ps = self.post_stats.lock().unwrap();
            ps.0 += (sorted - entered).as_secs_f64();
            ps.1 += (drained - sorted).as_secs_f64();
            ps.2 += drained.elapsed().as_secs_f64();
        }
        job
    }
}
impl Drop for Threads {
    fn drop(&mut self) {
        self.shared.quit.store(true, Ordering::Relaxed);
    }
}
impl Pool for Threads {
    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        self.publish(n, None, None, None, None, f)
    }
    fn run_at_ccd(&self, n: usize, at: &(dyn Fn(usize) -> *const u8 + Sync), ccd: &(dyn Fn(usize) -> usize + Sync), f: &(dyn Fn(usize) + Sync)) {
        self.publish(n, Some(at), None, Some(ccd), None, f)
    }
    fn run_on(&self, n: usize, node: &(dyn Fn(usize) -> usize + Sync), ccd: &(dyn Fn(usize) -> usize + Sync), f: &(dyn Fn(usize) + Sync)) {
        self.publish(n, None, Some(node), Some(ccd), None, f)
    }
    fn slot(&self, node: usize, ccd: usize) -> usize {
        if self.node_ccds.get(node).is_some_and(|c| !c.is_empty()) { self.node_ccds[node][ccd % self.node_ccds[node].len()] } else { MAX_CCDS + node }
    }
    fn run_slotted(&self, n: usize, slot: &(dyn Fn(usize) -> usize + Sync), f: &(dyn Fn(usize) + Sync)) {
        self.publish(n, None, None, None, Some(slot), f)
    }
    fn prefetch_slotted(&self, n: usize, slot: Box<dyn Fn(usize) -> usize + Send + Sync>, f: Box<dyn Fn(usize) + Send + Sync>) {
        if n > 1 {
            let fr: &(dyn Fn(usize) + Sync) = unsafe { &*(&*f as *const (dyn Fn(usize) + Send + Sync)) };
            self.post(n, None, None, None, Some(&*slot), fr, Some(f));
        }
    }
    fn prefetch_on(&self, n: usize, node: Box<dyn Fn(usize) -> usize + Send + Sync>, ccd: Box<dyn Fn(usize) -> usize + Send + Sync>, f: Box<dyn Fn(usize) + Send + Sync>) {
        if n > 1 {
            let fr: &(dyn Fn(usize) + Sync) = unsafe { &*(&*f as *const (dyn Fn(usize) + Send + Sync)) };
            self.post(n, None, Some(&*node), Some(&*ccd), None, fr, Some(f));
        }
    }
    fn node_of(&self, p: *const u8) -> usize {
        let p = p as usize;
        self.placed.lock().unwrap().iter().find(|(base, len, _)| p >= *base && p < base + len).map_or(0, |(base, _, nodes)| nodes[(p - base) / PAGE] as usize)
    }
    fn ccds_of(&self, node: usize) -> usize {
        self.node_ccds.get(node).map_or(1, |c| c.len().max(1))
    }
    fn abandoned(&self) -> bool {
        let job = INSIDE.with(|c| c.get());
        !job.is_null() && unsafe { (*job).stop.load(Ordering::Relaxed) }
    }
    fn locality(&self) -> (usize, usize) {
        *self.locality.lock().unwrap()
    }
    fn node_ends(&self) -> ([usize; MAX_NODES], [f64; MAX_NODES]) {
        std::mem::replace(&mut *self.node_stats.lock().unwrap(), ([0; MAX_NODES], [0.0; MAX_NODES]))
    }
    fn overheads(&self) -> (f64, f64, f64, usize) {
        std::mem::replace(&mut *self.overheads.lock().unwrap(), (0.0, 0.0, 0.0, 0))
    }
    fn last_ends(&self) -> f64 {
        std::mem::replace(&mut *self.last_ends.lock().unwrap(), 0.0)
    }
    fn post_stats(&self) -> (f64, f64, f64) {
        std::mem::replace(&mut *self.post_stats.lock().unwrap(), (0.0, 0.0, 0.0))
    }
    fn prefetched(&self) -> (usize, usize) {
        *self.prefetched.lock().unwrap()
    }
    fn prefetch(&self, n: usize, at: Box<dyn Fn(usize) -> *const u8 + Send + Sync>, ccd: Box<dyn Fn(usize) -> usize + Send + Sync>, f: Box<dyn Fn(usize) + Send + Sync>) {
        if n > 1 {
            // the job owns `f`; `at` and `ccd` are needed only to slot the items here
            let fr: &(dyn Fn(usize) + Sync) = unsafe { &*(&*f as *const (dyn Fn(usize) + Send + Sync)) };
            self.post(n, Some(&*at), None, Some(&*ccd), None, fr, Some(f));
        }
    }
    fn run_at(&self, n: usize, at: &(dyn Fn(usize) -> *const u8 + Sync), f: &(dyn Fn(usize) + Sync)) {
        self.publish(n, Some(at), None, None, None, f)
    }
}

/// A core's core complex, named by the first core sharing its L3 (from /sys; the core itself without the file).
pub fn ccd_key(core: usize) -> usize {
    std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{core}/cache/index3/shared_cpu_list")).ok().and_then(|s| s.trim().split([',', '-']).next().and_then(|v| v.parse().ok())).unwrap_or(core)
}

/// How many NUMA nodes /sys lists (1 where there is no NUMA).
pub fn node_count() -> usize {
    (0..MAX_NODES).take_while(|n| std::path::Path::new(&format!("/sys/devices/system/node/node{n}")).exists()).count().max(1)
}
/// The first core of node `node` (core 0 where there is no NUMA).
pub fn first_core_of(node: usize) -> usize {
    physical_cores().into_iter().find(|&c| node_of_core(c) == node).unwrap_or(0)
}
/// A kernel cpu list ("1-23,25-47") as the cores it names.
pub fn cpu_list(s: &str) -> Vec<usize> {
    let mut v = Vec::new();
    for part in s.trim().split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => v.extend(a.parse::<usize>().unwrap()..=b.parse::<usize>().unwrap()),
            None => v.push(part.parse().unwrap()),
        }
    }
    v
}
/// The NUMA node of a core, from /sys (0 where there is no NUMA).
pub fn node_of_core(core: usize) -> usize {
    for n in 0..MAX_NODES {
        let Ok(list) = std::fs::read_to_string(format!("/sys/devices/system/node/node{n}/cpulist")) else { break };
        for part in list.trim().split(',') {
            let (a, b) = part.split_once('-').map(|(a, b)| (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap())).unwrap_or_else(|| (part.parse().unwrap(), part.parse().unwrap()));
            if (a..=b).contains(&core) {
                return n;
            }
        }
    }
    0
}

/// The cores a pool on this machine should take: one per physical core (the
/// first SMT sibling of each), every socket, as /sys lists them — within the kernel's
/// tickless set when the image isolates cores (`nohz_full=` on the cmdline: the kernel's
/// own work stays on the housekeeping cores, the pool's never see a tick or an interrupt).
pub fn physical_cores() -> Vec<usize> {
    let isolated = cpu_list(&std::fs::read_to_string("/sys/devices/system/cpu/nohz_full").unwrap_or_default());
    let mut cores = Vec::new();
    let mut i = 0;
    while let Ok(s) = std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list")) {
        let first: usize = s.trim().split([',', '-']).next().unwrap().parse().unwrap();
        if first == i && (isolated.is_empty() || isolated.contains(&i)) {
            cores.push(i);
        }
        i += 1;
    }
    if cores.is_empty() {
        cores = (0..std::thread::available_parallelism().map_or(1, |t| t.get())).collect();
    }
    cores
}

/// The routed experts from the bundle's expert stream in this process's memory,
/// each expert's rows split over the pool: first the gate/up rows in chunks of
/// `CHUNK` hidden channels — a chunk's clamped SwiGLU is its own, written as
/// f32 — then the down rows in chunks of `DOWN_CHUNK`, each item quantizing
/// its expert's whole hidden state to INT8 groups itself (the same groups and
/// bits whoever cuts them; 2 µs against an item's 100 µs of streaming — so the
/// gate/up items are not held to a group's width: 128 of them over 48 cores a
/// node left a whole item of tail, 2026-10-15). Every item names the rows it
/// streams, so a placed pool runs it on their node. Every row's arithmetic is
/// `expert::forward`'s exactly (the same tiles, groups and flush); only the
/// cores differ.
pub struct CpuExperts<'a> {
    stream: &'a expert::Stream<'a>,
    pool: &'a dyn Pool,
    /// The node of every slice of every expert record (`[MOE_LAYERS][E][SLICES]`), asked of the
    /// pool once at construction, so slotting a prefetch is lookups, not a lock a slice.
    slice_node: Vec<u8>,
    /// The pool's queue for every slice (`[MOE_LAYERS][E][SLICES]`): its node's, with the core
    /// complex the slice's hash names — a map, so a run's slotting is a byte a part (tcdent
    /// 2026-10-16: "hold that as a map of offsets in memory for the lookup").
    slice_slot: Vec<u8>,
    /// Every expert's slices ordered by their queue (`[MOE_LAYERS][E][SLICES]`, slice ids), so a prefetch's items — an
    /// (expert, queue) and its run of slices — are cut from a table instead of grouped a call (the grouping's
    /// allocations were ~31 µs of a decode layer's 53 µs `slot` lap, 2026-10-20).
    slice_order: Vec<u8>,
    /// The last prefetch's plan: its layer and experts in the order posted, so that layer's run takes them in the order
    /// they arrive (the planned first, then the rest).
    planned: std::sync::Mutex<(usize, Vec<usize>)>,
    /// A step's activations quantized once each (`[STEP][H]` and their groups), the experts' hidden
    /// states (`[STEP][TOP_K][D_FF]`) and those quantized once each for the down rows.
    xq: Vec<i8>,
    xs: Vec<f32>,
    xg: Vec<i32>,
    h: Vec<f32>,
    hq: Vec<i8>,
    hs: Vec<f32>,
    hg: Vec<i32>,
    elapsed: f64,
    /// `track-experts`: a run's items by token-count bucket, accumulated a pool slot (a core complex's
    /// line, so eight cores share a counter and nothing bounces across the box), summed into rows after the run.
    #[cfg(feature = "track-experts")]
    item_acc: Vec<[ItemAcc; BUCKETS]>,
    /// `track-experts`: the last prefetch — its layer, the experts it predicted, each one's items and how many of them
    /// finished touching their slices — read when that layer's run starts (`prefetch_use` rows).
    #[cfg(feature = "track-experts")]
    pf: std::sync::Mutex<Option<(usize, Vec<usize>, Vec<u16>, std::sync::Arc<Vec<std::sync::atomic::AtomicU16>>)>>,
    /// `track-experts`: every core's performance counters, programmed once with `pmc::FILLS` (where a run's bytes come
    /// from); the items read them themselves (`rdpmc`).
    #[cfg(feature = "track-experts")]
    pmcs: Vec<common::pmc::Pmc>,
    /// `track-experts`: whether items read their core's fill counters (`rdpmc` allowed and the counters programmed),
    /// and those counts a pool slot, by the item's expert's prefetch status (complete, partial, missed) — the L2
    /// prefetcher's lines that hit L3 and that missed it.
    #[cfg(feature = "track-experts")]
    rdpmc: bool,
    #[cfg(feature = "track-experts")]
    fill_acc: Vec<FillAcc>,
}
#[cfg(feature = "track-experts")]
#[repr(align(64))]
#[derive(Default)]
struct FillAcc([std::sync::atomic::AtomicU64; 6]);
#[cfg(feature = "track-experts")]
const BUCKETS: usize = 8;
/// The token-count buckets of an item: 1, 2, 3–4, 5–8, 9–16, 17–32, 33–64, 65+.
#[cfg(feature = "track-experts")]
const BUCKET_NAMES: [&str; BUCKETS] = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+"];
#[cfg(feature = "track-experts")]
fn bucket_of(tokens: usize) -> usize {
    match tokens {
        0 | 1 => 0,
        2 => 1,
        3..=4 => 2,
        5..=8 => 3,
        9..=16 => 4,
        17..=32 => 5,
        33..=64 => 6,
        _ => 7,
    }
}
#[cfg(feature = "track-experts")]
#[repr(align(64))]
#[derive(Default)]
struct ItemAcc {
    ns: std::sync::atomic::AtomicU64,
    items: std::sync::atomic::AtomicU64,
    pairs: std::sync::atomic::AtomicU64,
}
/// An item is the record's (`layout::gu_item`, `layout::down_item`): 32 gate/up items and 20 down items an expert.
use crate::layout::{DOWN_ITEMS, GU_ITEMS, GU_PER_RUN};
use crate::STEP;

impl<'a> CpuExperts<'a> {
    pub fn new(stream: &'a expert::Stream<'a>, pool: &'a dyn Pool) -> Self {
        use common::q4::GROUP;
        #[cfg(feature = "track-experts")]
        let item_acc = (0..SLOTS).map(|_| std::array::from_fn(|_| ItemAcc::default())).collect();
        let mut this = CpuExperts { #[cfg(feature = "track-experts")] item_acc, #[cfg(feature = "track-experts")] pf: std::sync::Mutex::new(None), #[cfg(feature = "track-experts")] pmcs: Vec::new(), #[cfg(feature = "track-experts")] rdpmc: false, #[cfg(feature = "track-experts")] fill_acc: (0..SLOTS).map(|_| FillAcc::default()).collect(), stream, pool, slice_node: Vec::new(), slice_slot: Vec::new(), slice_order: Vec::new(), planned: std::sync::Mutex::new((usize::MAX, Vec::new())), xq: vec![0; STEP * H], xs: vec![0.0; STEP * H / GROUP], xg: vec![0; STEP * H / GROUP], h: vec![0.0; STEP * TOP_K * D_FF], hq: vec![0; STEP * TOP_K * D_FF], hs: vec![0.0; STEP * TOP_K * D_FF / GROUP], hg: vec![0; STEP * TOP_K * D_FF / GROUP], elapsed: 0.0 };
        this.relearn();
        // the fill counters on every core (the NMI watchdog holds a counter on each CPU: off first, 2026-10-20)
        #[cfg(feature = "track-experts")]
        {
            let _ = std::fs::write("/proc/sys/kernel/nmi_watchdog", "0");
            this.pmcs = (0..crate::construct::physical_cores().len()).filter_map(common::pmc::Pmc::open).filter(|p| p.program(&common::pmc::FILLS)).collect();
            // read by the items themselves with `rdpmc` (reading 96 cores' counters through msr took ~8.5 ms a run)
            this.rdpmc = !this.pmcs.is_empty() && common::pmc::allow_rdpmc();
        }
        this
    }
    /// The slices' nodes read again from the placement: after the stream's items were re-placed.
    pub fn relearn(&mut self) {
        let (stream, pool) = (self.stream, self.pool);
        let mut slice_node = Vec::with_capacity(layout::MOE_LAYERS * E * (GU_ITEMS + DOWN_ITEMS));
        let mut slice_slot = Vec::with_capacity(layout::MOE_LAYERS * E * (GU_ITEMS + DOWN_ITEMS));
        for l in FIRST_K_DENSE_REPLACE..layout::LAYERS_IN_FILE {
            for e in 0..E {
                let w = expert::Weights::at(stream, l, e);
                // a slice's core complex within its node: a function of the expert and the slice, never of the
                // expert's position among a token's picks, so a prefetch and the compute meet in one L3
                for j in 0..GU_ITEMS {
                    let d = pool.node_of(w.gu_bytes(j).as_ptr());
                    slice_node.push(d as u8);
                    slice_slot.push(pool.slot(d, e * GU_ITEMS + j) as u8);
                }
                for c in 0..DOWN_ITEMS {
                    let d = pool.node_of(w.down_bytes(c).as_ptr());
                    slice_node.push(d as u8);
                    slice_slot.push(pool.slot(d, e * DOWN_ITEMS + c) as u8);
                }
            }
        }
        self.slice_node = slice_node;
        // each expert's slices in queue order (stable: within a queue, the compute's order)
        const SLICES: usize = GU_ITEMS + DOWN_ITEMS;
        let mut order = Vec::with_capacity(slice_slot.len());
        for rec in slice_slot.chunks(SLICES) {
            let mut ix: Vec<u8> = (0..SLICES as u8).collect();
            ix.sort_by_key(|&j| rec[j as usize]);
            order.extend_from_slice(&ix);
        }
        self.slice_order = order;
        self.slice_slot = slice_slot;
    }
}
/// Pull every cache line of `b` through this core: into its L2 and, as the core streams on,
/// its CCD's L3. One 32-byte AVX2 load a line, four lines in flight.
#[inline(always)]
fn touch(b: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        touch_avx2(b.as_ptr(), b.len())
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut acc = 0u64;
        let mut i = 0;
        while i + 8 <= b.len() {
            acc ^= u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
            i += 64;
        }
        std::hint::black_box(acc);
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn touch_avx2(p: *const u8, len: usize) {
    use std::arch::x86_64::*;
    let (mut a, mut b, mut c, mut d) = (_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256());
    let mut i = 0;
    while i + 256 <= len {
        a = _mm256_xor_si256(a, _mm256_loadu_si256(p.add(i) as *const __m256i));
        b = _mm256_xor_si256(b, _mm256_loadu_si256(p.add(i + 64) as *const __m256i));
        c = _mm256_xor_si256(c, _mm256_loadu_si256(p.add(i + 128) as *const __m256i));
        d = _mm256_xor_si256(d, _mm256_loadu_si256(p.add(i + 192) as *const __m256i));
        i += 256;
    }
    while i + 32 <= len {
        a = _mm256_xor_si256(a, _mm256_loadu_si256(p.add(i) as *const __m256i));
        i += 64;
    }
    std::hint::black_box(_mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_xor_si256(c, d)));
}
impl Routed for CpuExperts<'_> {
    fn elapsed(&mut self) -> f64 {
        std::mem::take(&mut self.elapsed)
    }
    fn prefetch(&self, l: usize, idx: &[usize]) {
        {
            let mut p = self.planned.lock().unwrap();
            p.0 = l;
            p.1.clear();
            p.1.extend_from_slice(idx);
        }
        // the stream lives as long as the process (the node leaks it; the gate's outlives every token):
        // the job that owns these closures is gone within the token
        let stream: &'static expert::Stream<'static> = unsafe { std::mem::transmute::<&expert::Stream, &'static expert::Stream<'static>>(self.stream) };
        let pool = self.pool;
        // an item an (expert, node, CCD): the slices of that expert the compute will slot there, in the
        // compute's order — a slice's node from its rows' page, its CCD from the expert's id and the slice
        let (n_gu, n_down) = (GU_ITEMS, DOWN_ITEMS);
        let idx: Vec<usize> = idx.to_vec();
        // the items cut from `slice_order` (each expert's slices already in queue order): an item is (expert position,
        // queue, its run in `slices`), every item's slice ids in one flat copy the job owns
        let mut items: Vec<(usize, usize, u16, u16)> = Vec::with_capacity(idx.len() * SLOTS);
        let mut slices: Vec<u8> = Vec::with_capacity(idx.len() * (n_gu + n_down));
        for (k, &e) in idx.iter().enumerate() {
            let rec = ((l - FIRST_K_DENSE_REPLACE) * E + e) * (n_gu + n_down);
            let (order, slot_of) = (&self.slice_order[rec..rec + n_gu + n_down], &self.slice_slot[rec..rec + n_gu + n_down]);
            let mut a = 0;
            while a < order.len() {
                let q = slot_of[order[a] as usize];
                let mut b = a + 1;
                while b < order.len() && slot_of[order[b] as usize] == q {
                    b += 1;
                }
                items.push((k, q as usize, slices.len() as u16, (b - a) as u16));
                slices.extend_from_slice(&order[a..b]);
                a = b;
            }
        }
        let n = items.len();
        #[cfg(feature = "track-experts")]
        let finished = {
            let mut per = vec![0u16; idx.len()];
            for it in &items {
                per[it.0] += 1;
            }
            let fin = std::sync::Arc::new((0..idx.len()).map(|_| std::sync::atomic::AtomicU16::new(0)).collect::<Vec<_>>());
            *self.pf.lock().unwrap() = Some((l, idx.clone(), per, fin.clone()));
            fin
        };
        let items = std::sync::Arc::new((items, slices));
        let (i1, i3) = (items.clone(), items);
        let slot = move |i: usize| i1.0[i].1;
        // the touching yields between 4 KB chunks once the job is overtaken: the run that overtakes it
        // waits for the items in flight (the run's slotting time was 13 µs with 64 KB chunks, 2026-10-16)
        let pool_s: &'static dyn Pool = unsafe { std::mem::transmute::<&dyn Pool, &'static dyn Pool>(pool) };
        let f = move |i: usize| {
            let (it, slices) = (&i3.0[i], &i3.1);
            let w = expert::Weights::at(stream, l, idx[it.0]);
            let touch_yielding = |b: &[u8]| -> bool {
                for chunk in b.chunks(4 << 10) {
                    if pool_s.abandoned() {
                        return false;
                    }
                    touch(chunk);
                }
                true
            };
            for &j in &slices[it.2 as usize..(it.2 + it.3) as usize] {
                let j = j as usize;
                if !touch_yielding(if j < n_gu { w.gu_bytes(j) } else { w.down_bytes(j - n_gu) }) {
                    return;
                }
            }
            #[cfg(feature = "track-experts")]
            finished[it.0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        };
        self.pool.prefetch_slotted(n, Box::new(slot), Box::new(f));
    }
    fn run(&mut self, l: usize, idx: &[usize; TOP_K], x: &[f32], out: &mut [f32]) {
        self.run_batch(l, std::slice::from_ref(idx), x, out)
    }
    /// The step grouped by expert: every expert any token picked is one group of parts, each part
    /// streamed once and applied to every (token, slot) that picked the expert. Three runs over the
    /// pool: the gate/up parts (each token's input quantized once before), the hidden states
    /// quantized once each, the down parts. One token is a step of one.
    fn run_batch(&mut self, l: usize, idx: &[[usize; TOP_K]], x: &[f32], out: &mut [f32]) {
        use common::q4::{QAct, GROUP};
        const HG: usize = H / GROUP;
        let (mut xq, mut xs, mut xg) = (std::mem::take(&mut self.xq), std::mem::take(&mut self.xs), std::mem::take(&mut self.xg));
        for i in 0..idx.len() {
            QAct { q: &mut xq[i * H..(i + 1) * H], scale: &mut xs[i * HG..(i + 1) * HG], gsum: &mut xg[i * HG..(i + 1) * HG] }.quantize(&x[i * H..(i + 1) * H]);
        }
        self.run_batch_q(l, idx, &xq, &xs, &xg, out);
        (self.xq, self.xs, self.xg) = (xq, xs, xg);
    }
    fn run_batch_act(&mut self, l: usize, idx: &[[usize; TOP_K]], xq: &[i8], xs: &[f32], xg: &[i32], out: &mut [f32]) {
        self.run_batch_q(l, idx, xq, xs, xg, out)
    }
}
impl CpuExperts<'_> {
    /// The step's experts over quantized activations (`run_batch_act`'s form; `run_batch` quantizes first).
    fn run_batch_q(&mut self, l: usize, idx: &[[usize; TOP_K]], xq: &[i8], xs: &[f32], xg: &[i32], out: &mut [f32]) {
        /// Pair `i`'s (token, slot) as the index of its hidden state and output row.
        fn blk_slot(pairs: &[(u16, u8)], i: usize) -> usize {
            pairs[i].0 as usize * TOP_K + pairs[i].1 as usize
        }
        use common::q4::{self, Act, QAct, GEMM_TOKENS, GROUP};
        use common::swiglu::swiglu_clamped;
        use expert::{DOWN_PARTS, DOWN_ROWS, GU_PARTS, GU_ROWS, SUB};
        let t0 = std::time::Instant::now();
        let n = idx.len();
        assert!(n <= STEP);
        const HG: usize = H / GROUP;
        const DG: usize = D_FF / GROUP;
        // the experts picked, each with the (token, slot) pairs that picked it: a counting sort into one
        // list, `first[e]..first[e + 1]` an expert's pairs (no allocation an expert)
        // a pick of `E` is resident on a card and runs there (`Placed::xpt_dispatch`): skipped here, its
        // output slot filled by the join
        let mut first = [0usize; E + 1];
        for picks in idx {
            for &e in picks {
                if e < E {
                    first[e + 1] += 1;
                }
            }
        }
        for e in 0..E {
            first[e + 1] += first[e];
        }
        let mut fill = first;
        let mut pairs = vec![(0u16, 0u8); n * TOP_K];
        for (i, picks) in idx.iter().enumerate() {
            for (k, &e) in picks.iter().enumerate() {
                if e < E {
                    pairs[fill[e]] = (i as u16, k as u8);
                    fill[e] += 1;
                }
            }
        }
        // the run's experts in the order the prefetch posted them (its plan for this layer), then the rest by id: each
        // CCD's queue takes its items in this order, so the run reads first what arrived first, and the experts it streams
        // from DRAM come after the prefetched ones are read (in id order, the run streamed a high-id unplanned expert
        // between planned ones). The outputs land in fixed rows: the order changes no bits.
        let mut experts: Vec<usize> = (0..E).filter(|&e| first[e + 1] > first[e]).collect();
        {
            let mut p = self.planned.lock().unwrap();
            if p.0 == l {
                let rank = |e: &usize| p.1.iter().position(|q| q == e).unwrap_or(E + *e);
                experts.sort_by_key(rank);
                p.0 = usize::MAX;
            }
        }
        // `track-experts`: this run's host experts against the prefetch made for this layer — predicted and every item
        // touched, predicted but not finished when the run began, not predicted (decode's L3 question, 2026-10-20)
        #[cfg(feature = "track-experts")]
        let mut pf_use: Option<[usize; 8]> = None;
        // each expert's prefetch status for the items' fill counts: 0 complete, 1 partial, 2 missed
        #[cfg(feature = "track-experts")]
        let mut status = [2u8; E];
        #[cfg(feature = "track-experts")]
        if let Some((pl, pidx, per, fin)) = self.pf.lock().unwrap().take() {
            if pl == l {
                let (mut complete, mut partial, mut missed, mut items, mut items_done) = (0usize, 0usize, 0usize, 0usize, 0usize);
                for &e in &experts {
                    match pidx.iter().position(|&p| p == e) {
                        Some(k) => {
                            let d = fin[k].load(Ordering::Relaxed) as usize;
                            items += per[k] as usize;
                            items_done += d;
                            if d == per[k] as usize {
                                complete += 1;
                                status[e] = 0;
                            } else {
                                partial += 1;
                                status[e] = 1;
                            }
                        }
                        None => missed += 1,
                    }
                }
                let all_done: usize = fin.iter().map(|f| f.load(Ordering::Relaxed) as usize).sum();
                let all: usize = per.iter().map(|&x| x as usize).sum();
                pf_use = Some([experts.len(), complete, partial, missed, items, items_done, pidx.len(), all * 1_000_000 + all_done]);
            }
        }
        // an item's L3 hits and misses (its core's counters around it), to its expert's status, when this run is counted
        #[cfg(feature = "track-experts")]
        let (counted, fill_acc, status) = (pf_use.is_some() && self.rdpmc, &self.fill_acc, &status);
        #[cfg(feature = "track-experts")]
        let fill = move |slot: usize, e: usize, c0: (u64, u64)| {
            if counted {
                let (h, m) = (common::pmc::rdpmc(4).wrapping_sub(c0.0), common::pmc::rdpmc(5).wrapping_sub(c0.1));
                let s = status[e] as usize;
                fill_acc[slot].0[2 * s].fetch_add(h & ((1 << 48) - 1), Ordering::Relaxed);
                fill_acc[slot].0[2 * s + 1].fetch_add(m & ((1 << 48) - 1), Ordering::Relaxed);
            }
        };
        #[cfg(feature = "track-experts")]
        let fill0 = move || if counted { (common::pmc::rdpmc(4), common::pmc::rdpmc(5)) } else { (0, 0) };
        let ne = experts.len();
        let uses = |e: usize| &pairs[first[e]..first[e + 1]];
        let stream = self.stream;
        let slice_slot = &self.slice_slot;
        let slot_of = |e: usize, j: usize| slice_slot[((l - FIRST_K_DENSE_REPLACE) * E + e) * (GU_ITEMS + DOWN_ITEMS) + j] as usize;
        let h_p = self.h.as_mut_ptr() as usize;
        // the gate/up rows: part (j, c) owns expert experts[j]'s hidden channels c·GU_ROWS.., for all of
        // its tokens at once (`q4::gemm`: the part's bytes read once, every token's dot on them), in
        // blocks of GEMM_TOKENS
        let uses_r = &uses;
        let experts_r = &experts;
        // a step past SPEC_TOKENS: a part is a quantize group's channels (QP parts: two bundle items, adjacent in one
        // run, so one node's), and quantizes the group for its tokens right after SwiGLU — the separate quantize run read
        // `h` lines every CCD of both sockets had written (~2 ms a 512-token layer-chunk, 2026-10-20). A few tokens keep
        // the fine parts (a decode step's few experts need many items) and the quantize below.
        const QP: usize = GROUP / GU_ROWS;
        const { assert!(GROUP % GU_ROWS == 0 && GU_PARTS % QP == 0 && GU_PER_RUN % (QP / SUB) == 0) };
        let fold = n > SPEC_TOKENS;
        let parts = if fold { GU_PARTS / QP } else { GU_PARTS };
        let first_c = move |p: usize| if fold { p * QP } else { p };
        let gate_slot = move |i: usize| slot_of(experts_r[i / parts], first_c(i % parts) / SUB);
        let (hq_p, hs_p, hg_p) = (self.hq.as_mut_ptr() as usize, self.hs.as_mut_ptr() as usize, self.hg.as_mut_ptr() as usize);
        #[cfg(feature = "track-experts")]
        let acc = &self.item_acc;
        {
            cpu_zone!("experts.gu");
            self.pool.run_slotted(ne * parts, &gate_slot, &move |i| {
            #[cfg(feature = "track-experts")]
            let t_item = std::time::Instant::now();
            let (j, p) = (i / parts, i % parts);
            let e = experts_r[j];
            #[cfg(feature = "track-experts")]
            let c0 = fill0();
            let w = expert::Weights::at(stream, l, e);
            let pairs = uses_r(e);
            for c in first_c(p)..first_c(p) + if fold { QP } else { 1 } {
            let (gw, uw, gs, usc) = w.gu(c);
            let mut b0 = 0;
            while b0 < pairs.len() {
                let blk = &pairs[b0..(b0 + GEMM_TOKENS).min(pairs.len())];
                let act = |i: usize| {
                    let t = blk[i].0 as usize;
                    Act { q: &xq[t * H..(t + 1) * H], scale: &xs[t * HG..(t + 1) * HG], gsum: &xg[t * HG..(t + 1) * HG] }
                };
                let mut gu = [[0f32; 2 * GU_ROWS]; GEMM_TOKENS];
                let gp = gu.as_mut_ptr() as *mut f32;
                q4::gemm::<H>(GU_ROWS, gw, gs, blk.len(), act, |i| unsafe { gp.add(i * 2 * GU_ROWS) });
                q4::gemm::<H>(GU_ROWS, uw, usc, blk.len(), act, |i| unsafe { gp.add(i * 2 * GU_ROWS + GU_ROWS) });
                for (i, &(t, k)) in blk.iter().enumerate() {
                    let h = unsafe { std::slice::from_raw_parts_mut((h_p as *mut f32).add((t as usize * TOP_K + k as usize) * D_FF + c * GU_ROWS), GU_ROWS) };
                    swiglu_clamped::<GU_ROWS>(&gu[i], SWIGLU_LIMIT, h);
                }
                b0 += GEMM_TOKENS;
            }
            }
            if fold {
                // this part's group of every token's hidden state, quantized as the whole vector's would be (per group)
                for &(t, k) in pairs {
                    let s = t as usize * TOP_K + k as usize;
                    let (x, q, sc, g) = unsafe {
                        (
                            std::slice::from_raw_parts((h_p as *const f32).add(s * D_FF + p * GROUP), GROUP),
                            std::slice::from_raw_parts_mut((hq_p as *mut i8).add(s * D_FF + p * GROUP), GROUP),
                            std::slice::from_raw_parts_mut((hs_p as *mut f32).add(s * DG + p), 1),
                            std::slice::from_raw_parts_mut((hg_p as *mut i32).add(s * DG + p), 1),
                        )
                    };
                    QAct { q, scale: sc, gsum: g }.quantize(x);
                }
            }
            #[cfg(feature = "track-experts")]
            acc[gate_slot(i)][bucket_of(pairs.len())].add(t_item.elapsed().as_nanos() as u64, pairs.len());
            #[cfg(feature = "track-experts")]
            fill(gate_slot(i), e, c0);
            });
        }
        #[cfg(feature = "track-experts")]
        self.item_rows(l, "gu", (2 * GU_ROWS * H / 2 + 2 * GU_ROWS * HG * 4) * if fold { QP } else { 1 });
        // a few tokens: every (token, slot)'s hidden state quantized once, here for a token (2 µs each, less than a
        // pool run's posting), over the pool otherwise
        let h = &self.h;
        let quantize_h = move |i: usize| {
            let (q, sc, g) = unsafe { (std::slice::from_raw_parts_mut((hq_p as *mut i8).add(i * D_FF), D_FF), std::slice::from_raw_parts_mut((hs_p as *mut f32).add(i * DG), DG), std::slice::from_raw_parts_mut((hg_p as *mut i32).add(i * DG), DG)) };
            QAct { q, scale: sc, gsum: g }.quantize(&h[i * D_FF..(i + 1) * D_FF]);
        };
        if !fold {
            if n * TOP_K <= 16 {
                for i in 0..n * TOP_K {
                    quantize_h(i);
                }
            } else {
                cpu_zone!("experts.quant");
                self.pool.run(n * TOP_K, &quantize_h);
            }
        }
        // the down rows: part (j, c) owns rows c·DOWN_ROWS.. of expert experts[j]'s output, for all of its tokens, likewise
        let (hq, hs, hg) = (&self.hq, &self.hs, &self.hg);
        let out_p = out.as_mut_ptr() as usize;
        let down_slot = move |i: usize| slot_of(experts_r[i / DOWN_PARTS], GU_ITEMS + i % DOWN_PARTS / SUB);
        {
            cpu_zone!("experts.down");
            self.pool.run_slotted(ne * DOWN_PARTS, &down_slot, &move |i| {
            #[cfg(feature = "track-experts")]
            let t_item = std::time::Instant::now();
            let (j, c) = (i / DOWN_PARTS, i % DOWN_PARTS);
            let e = experts_r[j];
            #[cfg(feature = "track-experts")]
            let c0 = fill0();
            let w = expert::Weights::at(stream, l, e);
            let (dw, ds) = w.down(c);
            let r0 = c * DOWN_ROWS;
            let pairs = uses_r(e);
            let slot = |i: usize| blk_slot(pairs, i);
            let mut b0 = 0;
            while b0 < pairs.len() {
                let blk = &pairs[b0..(b0 + GEMM_TOKENS).min(pairs.len())];
                let act = |i: usize| {
                    let s = slot(b0 + i);
                    Act { q: &hq[s * D_FF..(s + 1) * D_FF], scale: &hs[s * DG..(s + 1) * DG], gsum: &hg[s * DG..(s + 1) * DG] }
                };
                q4::gemm::<D_FF>(DOWN_ROWS, dw, ds, blk.len(), act, |i| unsafe { (out_p as *mut f32).add(slot(b0 + i) * H + r0) });
                b0 += GEMM_TOKENS;
            }
            // the rows may be a card's VRAM through the BAR (002): this core's write-combining stores out before the
            // run ends and the main thread writes the leader's word
            unsafe { core::arch::x86_64::_mm_sfence() };
            #[cfg(feature = "track-experts")]
            acc[down_slot(i)][bucket_of(pairs.len())].add(t_item.elapsed().as_nanos() as u64, pairs.len());
            #[cfg(feature = "track-experts")]
            fill(down_slot(i), e, c0);
            });
        }
        #[cfg(feature = "track-experts")]
        self.item_rows(l, "down", DOWN_ROWS * D_FF / 2 + DOWN_ROWS * DG * 4);
        self.elapsed += t0.elapsed().as_secs_f64();
        #[cfg(feature = "track-experts")]
        if let Some(u) = pf_use {
            // the items' fills by status, summed over the slots and cleared
            let mut f = [0u64; 6];
            for a in self.fill_acc.iter() {
                for (i, x) in a.0.iter().enumerate() {
                    f[i] += x.swap(0, Ordering::Relaxed);
                }
            }
            prow::row!("track-experts", "prefetch_use", { "layer" => l, "tokens" => n, "host_experts" => u[0], "complete" => u[1], "partial" => u[2], "missed" => u[3], "used_items" => u[4], "used_items_done" => u[5], "predicted" => u[6], "items" => u[7] / 1_000_000, "items_done" => u[7] % 1_000_000, "counted" => self.rdpmc as u8, "c_hit" => f[0], "c_miss" => f[1], "p_hit" => f[2], "p_miss" => f[3], "m_hit" => f[4], "m_miss" => f[5], "run_us" => t0.elapsed().as_secs_f64() * 1e6 });
        }
    }
}
#[cfg(feature = "track-experts")]
impl CpuExperts<'_> {
    /// A run's items summed over the slots into a row a bucket (`expert_items`), and the accumulators cleared.
    #[cfg(feature = "track-experts")]
    fn item_rows(&self, l: usize, phase: &'static str, item_bytes: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        for b in 0..BUCKETS {
            let (mut ns, mut items, mut pairs) = (0u64, 0u64, 0u64);
            for sl in &self.item_acc {
                ns += sl[b].ns.swap(0, Relaxed);
                items += sl[b].items.swap(0, Relaxed);
                pairs += sl[b].pairs.swap(0, Relaxed);
            }
            if items > 0 {
                prow::row!("track-experts", "expert_items", { "layer" => l, "phase" => phase, "tokens" => BUCKET_NAMES[b], "items" => items, "pairs" => pairs, "ms" => ns as f64 / 1e6, "mb" => (items as usize * item_bytes) as f64 / 1e6 });
            }
        }
    }
}
#[cfg(feature = "track-experts")]
impl ItemAcc {
    #[inline]
    fn add(&self, ns: u64, pairs: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        self.ns.fetch_add(ns, Relaxed);
        self.items.fetch_add(1, Relaxed);
        self.pairs.fetch_add(pairs as u64, Relaxed);
    }
}

/// Every block but the routed experts, placed away from this CPU — on the
/// cards — behind one interface, the residual streams with them. A token:
/// `begin` puts the embedding into the streams; a layer's attention half is
/// posted with `attention` (a KDA layer) or `mla_post`, `mla_wait`,
/// `mla_attend` (an MLA layer, the host selecting the positions between);
/// `early_logits` reads the FFN router run early on the attention block's
/// input (the prefetch's predictor, a one-token step's: a step of many tokens
/// names most experts and runs no early router); `ffn` posts the FFN half, which for a
/// MoE layer collapses the block input to the host (`ffn_input`) and folds
/// the routed experts' rows (the leader card sums them on `ffn_done`) into the streams with
/// the shared expert's rows; `logits` is the head over the streams' mean.
/// Nothing is waited for but what the host reads. `seed` and `snapshot` move
/// a KDA layer's state; `mla_seed` installs a latent; `layer_out` fetches a
/// layer's output off the streams for bring-up.
pub trait Placed {
    /// The step's `n` tokens' embeddings (`[n][H]`) into every card's streams; the handoffs restart.
    fn begin(&mut self, rows: &[f32], n: usize);
    /// The calls that follow address chunk `c` of the step (037): its tokens from `t0`, its own handoff words.
    /// Their `n` is the chunk's width and their token indices are the chunk's own; `begin` starts at chunk 0.
    fn chunk(&mut self, c: usize, t0: usize);
    /// A KDA layer's block over the step; a step of a verify's size (`2..=SPEC_TOKENS` tokens) keeps
    /// the state after each token for `kda_rewind`.
    fn attention(&mut self, l: usize, n: usize);
    /// A KDA layer's state back to what it was after token `slot` of the last verify-sized step.
    fn kda_rewind(&mut self, l: usize, slot: usize);
    fn mla_post(&mut self, l: usize, n: usize);
    /// The step's projections once the launch is done: every token's latent before its norm, its indexer key
    /// before its norm and its gate, in the host memory the card wrote them to.
    fn mla_proj(&mut self, l: usize) -> MlaProj;
    /// The pool keys `from..to` into the card's mirror (every pool completed, by a step or a seed, goes
    /// there before any token scores among them).
    fn ix_pools(&mut self, l: usize, pools: &[u16], from: usize, to: usize);
    /// Every token's selection on the card (`mla::score_pools` and `mla::pick`'s results, from the
    /// mirrored pools) into the attend's lists; the first token's count back when asked (bring-up).
    fn ix_select(&mut self, l: usize, pos: usize, n: usize, count: Option<&mut usize>);
    /// The step's `n` tokens (at `pos..`) attend over their selections (the card's, `ix_select`), one
    /// launch; their latents (`[n][KV_LORA_RANK]`) into the mirror first.
    fn mla_attend(&mut self, l: usize, n: usize, pos: usize, latents: &[u16]);
    /// The output projection over the step's heads, into the streams.
    fn mla_out(&mut self, l: usize, n: usize);
    /// The draft layer (`layout::MTP_LAYER`): its input `[2·H]` (the next token's normed embedding, the
    /// normed hidden state) through `eh_proj` and the attention's projections (then `mla_wait`,
    /// `ix_pools`, `ix_select`, `mla_attend` as for any MLA layer);
    /// A MoE layer's routed experts resident on the cards: every pick `(token i, slot k)` whose expert is
    /// resident becomes a job on its card over the token's quantized activation, launched at once, and is
    /// marked `E` in `picks` so the host's run skips it; a pick not resident may claim a slot (a free one,
    /// or a less-used resident's past a margin), which the loader fills for the steps to come. Returns the jobs.
    fn xpt_dispatch(&mut self, l: usize, n: usize, picks: &mut [[usize; TOP_K]]) -> usize;
    /// MoE layer `l`'s routed experts' outputs, `[STEP · TOP_K][H]` f32 by (token, pick) in the VRAM of the layer's
    /// leader card (its FFN card, 002), mapped for the host and the other card: the cards' jobs land there
    /// directly and the host's experts write theirs through the BAR, so the join is a wait and no copy.
    fn expert_out(&self, l: usize) -> *mut f32;
    /// The loader the residency fills slots with, off the step: `fetch(l, e, out)` gathers expert `e` of
    /// layer `l` as a slot holds it (`expert::Weights::slot_bytes`). Set once before the first step.
    fn xpt_loader(&mut self, fetch: Box<dyn Fn(usize, usize, &mut [u8]) + Send + 'static>);
    /// A MoE layer's FFN output half (`hc_post`, which waits for the host's experts), posted after the resident
    /// experts' launches so they are not queued behind it.
    fn ffn_post(&mut self, l: usize, n: usize);
    /// The cards' expert outputs of `xpt_dispatch`'s jobs into `out[(i · TOP_K + k) · H..]`, waiting for them.
    fn xpt_join(&mut self, l: usize, out: &mut [f32]);
    /// Whether expert `e` of layer `l` is resident on a card (the prefetch leaves those to the cards).
    fn resident(&self, l: usize, e: usize) -> bool;
    /// The residency's slots claimed, slots in all, and loads made since the start.
    fn resident_stats(&self) -> (usize, usize, usize) {
        (0, 0, 0)
    }
    /// Whether the bring-up checks run (the node built with `bringup-checks`).
    fn checks(&self) -> bool;
    fn mtp_begin(&mut self, cat: &[f32], n: usize);
    /// its output projection and residual, the blocks' states out (`[n][H]`) for the host's routing;
    fn mtp_attn_out(&mut self, hidden: &mut [f32], n: usize);
    /// the shared expert, the host's routed experts' sums `routed` (`[n][H]`) and the residual, the states out;
    fn mtp_ffn(&mut self, hidden: &mut [f32], n: usize);
    /// The draft layer's routed sum, `[STEP][H]` f32 in host memory the draft's card reads: the host writes each
    /// position's weighted sum there and `mtp_ffn` folds it in.
    fn mtp_routed(&self) -> *mut f32;
    /// the shared head over token `i`'s state: its draft's logits.
    fn mtp_head(&mut self, i: usize, logits: &mut [f32]);
    /// The last step's tokens' hidden states (the streams' means before the final norm), `[n][H]`.
    fn hidden_all(&mut self, n: usize, out: &mut [f32]);
    fn mla_seed(&mut self, l: usize, pos: usize, latent: &[u16]);
    fn ffn(&mut self, l: usize, n: usize);
    /// A MoE layer's FFN input for every token (`[n][H]`) and the FFN router's logits (`[n][E]`).
    fn ffn_input(&mut self, l: usize, n: usize, xq: &mut [i8], xs: &mut [f32], xg: &mut [i32], picks: &mut [[usize; TOP_K]], wts: &mut [[f32; TOP_K]], pool: &dyn Pool);
    /// The prefetch predictor's step on the cards (`plan::gfx11`), posted right ahead of a MoE layer's output half: pair
    /// l predicts the next layer (a decode step's prediction is a plan, `plan`), pair l − 1 learns from the picks.
    fn plan_step(&mut self, l: usize, n: usize, picks: &[[usize; TOP_K]]);
    /// The step's predictor updates, once its layers are done: every pair over the step's `n` tokens at once, a
    /// workgroup a pair (in the layers' waits a decode step's took 23 µs a layer of a ~180 µs wait, 2026-10-21).
    fn plan_learn(&mut self, n: usize);
    /// A decode step's plan for the layer after l, best first (`plan::PLAN` experts), once the card's word says it is in.
    fn plan(&mut self, l: usize, out: &mut [usize]);
    /// Bring-up: a MoE layer's FFN input (`x: [n][H]`) and its router's logits (`[n][ROUTE_ROWS]`) as the card
    /// made them, for forced picks and the traces' edges; serving reads `ffn_input`'s quantized form instead.
    fn ffn_logits(&mut self, l: usize, n: usize, x: &mut [f32], logits: &mut [f32]);
    /// Lab (`dump-pred`): a MoE layer's shared expert output (`[n][H]`) and its routed experts' rows (`[n][TOP_K][H]`,
    /// pick order) as the leader card holds them, once the layer's output half is done.
    fn ffn_parts(&mut self, l: usize, n: usize, shared: &mut [f32], rows: &mut [f32]);
    /// Every routed expert row of the chunk is in the leader's VRAM (the host's run and the residents' join done):
    /// the leader's word, after which it sums the rows in pick order into the streams and forwards the sums (002).
    fn ffn_done(&mut self, l: usize);
    /// Token `i`'s layer output off the streams (bring-up).
    fn layer_out(&mut self, l: usize, i: usize, mean: &mut [f32]);
    /// The head over tokens `from..from + n`'s streams, `[n][VOCAB_SIZE]`; the step ends.
    fn logits(&mut self, from: usize, n: usize, logits: &mut [f32]);
    /// The step's end without logits: every card done with its streams.
    fn finish(&mut self);
    fn seed(&mut self, l: usize, window: &[f32], mem: &[f32]);
    fn snapshot(&self, l: usize, window: &mut [f32], mem: &mut [f32]);
}

/// One layer's sequence state: a KDA layer's convolution window and memory
/// (the snapshot's), or an MLA layer's cache rows and pool keys (the rows').
pub enum LayerState {
    Kda { window: Vec<f32>, mem: Vec<f32> },
    Mla { cache: Vec<u16>, pools: Vec<u16> },
}

/// A token's time by phase, summed over steps until reset (bring-up, and the node's `step` rows
/// under its `track-step` category). The experts are the routed realization's own count.
#[derive(Default, Clone, Copy)]
pub struct Profile {
    pub tokens: usize,
    /// The hyper-connections on the host, or, with cards, the posting of a layer's launches.
    pub hc: f64,
    /// With cards: the next layer's predicted picks and the prefetch's slotting.
    pub slot: f64,
    pub kda_card: f64,
    pub kda_cpu: f64,
    /// The MoE's time but the experts': the router and the routed sum (`route`), the shared expert's finish (`shared`).
    pub route: f64,
    pub shared: f64,
    pub mla_proj: f64,
    pub mla_attn: f64,
    pub mla_out: f64,
    pub dense_ffn: f64,
    pub experts: f64,
    pub head: f64,
    /// The draft layer after a step (`draft_batch`), cards and host.
    pub draft: f64,
    /// The host's wait for the cards' resident experts after its own run (`xpt_join`): what a
    /// card-controlled expert phase (glm-next-steps §2) could remove.
    pub join: f64,
    /// Over the MoE layers: how many of a layer's eight experts were also the previous token's at that layer
    /// (the ceiling of a same-layer prefetch), and the layers counted.
    pub reuse: usize,
    /// Picks run on the cards' resident experts, and picks in all, over the steps profiled (verify-sized steps).
    pub resident: usize,
    pub picks: usize,
    pub moe_layers: usize,
    /// Expert items (gate/up and down) that ran on their preferred CCD, and all of them.
    pub local_items: usize,
    pub items: usize,
    /// How many of a decode layer's picks the plan for it named (the prefetch predictor's `plan::PLAN` experts, the
    /// resident ones among them).
    pub pre_agree: usize,
    /// Prefetch slices done before the router's run overtook the prefetch, and the slices asked for.
    pub prefetched: usize,
    pub prefetch_items: usize,
    /// Per node over the expert runs: items, and the sum of its run ends (s) — the nodes' measured rates.
    pub node_items: [usize; MAX_NODES],
    pub node_end: [f64; MAX_NODES],
    /// Over the expert runs: seconds slotting, waking, and after the last node's end; the runs.
    pub run_slot: f64,
    pub run_wake: f64,
    pub run_tail: f64,
    pub runs: usize,
    /// The sum over the runs of the last node's end: what the phase waits for.
    pub run_last: f64,
    /// The posts' parts: sorting into slots, draining the previous job, allocating and freeing.
    pub post_sort: f64,
    pub post_drain: f64,
    pub post_free: f64,
}
pub struct CpuModel<'p, R: Routed> {
    pub routed: R,
    pub pool: &'p dyn Pool,
    pub profile: Profile,
    /// Blocks that run elsewhere; `None` runs every block here.
    pub placed: Option<Box<dyn Placed>>,
    /// The previous token's expert picks by layer (`E` where none yet): the prefetch's predictor and its score.
    pub last_picks: Vec<[usize; TOP_K]>,
    /// How often each expert of each layer was picked (`[NUM_HIDDEN_LAYERS][E]`): the data for placing the hot ones.
    pub freq: Vec<u32>,
    pub max_ctx: usize,
    pub layers: Vec<LayerState>,
    /// The draft layer's sequence state (its MLA cache rows and pool keys, positions as the main
    /// model's), when the weights carry one.
    pub mtp: Option<LayerState>,
    /// The prefix positions' hidden states (`[FIRST_POS][H]`, the prefix file's): the draft's input there.
    prefix_hidden: Vec<f32>,
    d_cat: Vec<f32>,
    d_cats: Vec<f32>,
    /// The draft layer's output at the last position it drafted, `[H]`: the next depth's hidden input.
    d_hid: Vec<f32>,
    /// A few tokens' FFN inputs quantized for the cards' resident experts (`[SPEC_TOKENS][H]` i8, the scales and Σq).
    b_xq: Vec<i8>,
    b_xs: Vec<f32>,
    b_xg: Vec<i32>,
    /// A step's picks and weights from the router, and the picks as routed, kept beside `b_picks` (which the
    /// dispatch marks) for the accounting and the checks.
    b_picks: Vec<[usize; TOP_K]>,
    b_wts: Vec<[f32; TOP_K]>,
    b_routed: Vec<[usize; TOP_K]>,
    d_x: Vec<f32>,
    d_xn: Vec<f32>,
    d_y: Vec<f32>,
    /// The last step's logits, `[VOCAB_SIZE]`.
    /// `[STEP][VOCAB_SIZE]`: token `i` of the last step's logits at row `i` (`Logits`); a single token's at row 0.
    pub logits: Vec<f32>,
    streams: Vec<f32>,
    x: Vec<f32>,
    xn: Vec<f32>,
    y: Vec<f32>,
    mean: Vec<f32>,
    hc_flat: Vec<f32>,
    hc_coef: Vec<f32>,
    k_mixed: Vec<f32>,
    k_conv: Vec<f32>,
    k_fa: Vec<f32>,
    k_f: Vec<f32>,
    k_beta: Vec<f32>,
    k_ga: Vec<f32>,
    k_gate: Vec<f32>,
    k_normed: Vec<f32>,
    k_heads: Vec<f32>,
    m_qr: Vec<f32>,
    m_qa: Vec<f32>,
    m_q: Vec<f32>,
    m_kva: Vec<f32>,
    m_ik: Vec<f32>,
    m_iq: Vec<f32>,
    m_iw: Vec<f32>,
    m_pscore: Vec<f32>,
    m_pidx: Vec<u32>,
    m_sel: Vec<usize>,
    // a step's MLA selection, a token a row: indexer queries, head weights, pool scores, the pick's scratch, the selections and their lengths
    b_lat: Vec<u16>,
    m_heads: Vec<f32>,
    m_hs: Vec<f32>,
    f_gu: Vec<f32>,
    f_h: Vec<f32>,
    f_expert_out: Vec<f32>,
    f_shared: Vec<f32>,
    f_score: Vec<f32>,
    /// The draft's router scores, `[STEP][E]`: every position's at once (081).
    d_scores: Vec<f32>,
    /// A step's per-token rows with the cards: the block inputs, their norms, the routed outputs,
    /// the routers' logits, the experts' outputs (`[STEP][…]`).
    b_x: Vec<f32>,
    b_xn: Vec<f32>,
    b_y: Vec<f32>,
    b_score: Vec<f32>,
    b_out: Vec<f32>,
}

impl<'p, R: Routed> CpuModel<'p, R> {
    pub fn new(routed: R, pool: &'p dyn Pool, max_ctx: usize) -> Self {
        #[cfg(target_arch = "x86_64")]
        assert!(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"), "the shipped realizations need AVX2+FMA");
        assert!(max_ctx >= FIRST_POS + 1);
        let layers = (0..NUM_HIDDEN_LAYERS)
            .map(|l| if LINEAR_ATTENTION[l] { LayerState::Kda { window: vec![0.0; KDA_WINDOW], mem: vec![0.0; KDA_MEMORY] } } else { LayerState::Mla { cache: vec![0u16; max_ctx * MLA_ROW_U16], pools: vec![0u16; max_ctx.div_ceil(INDEX_KPOOL) * INDEX_HEAD_DIM] } })
            .collect();
        CpuModel {
            routed,
            pool,
            profile: Profile::default(),
            placed: None,
            last_picks: vec![[E; TOP_K]; NUM_HIDDEN_LAYERS],
            freq: vec![0; NUM_HIDDEN_LAYERS * E],
            max_ctx,
            mtp: (NUM_NEXTN == 1).then(|| LayerState::Mla { cache: vec![0u16; max_ctx * MLA_ROW_U16], pools: vec![0u16; max_ctx.div_ceil(INDEX_KPOOL) * INDEX_HEAD_DIM] }),
            prefix_hidden: vec![0.0; FIRST_POS * H],
            d_cat: vec![0.0; 2 * H],
            d_cats: vec![0.0; STEP * 2 * H],
            d_hid: vec![0.0; H],
            b_xq: vec![0; STEP * H],
            b_xs: vec![0.0; STEP * (H / common::q4::GROUP)],
            b_xg: vec![0; STEP * (H / common::q4::GROUP)],
            b_picks: vec![[0; TOP_K]; STEP],
            b_wts: vec![[0.0; TOP_K]; STEP],
            b_routed: vec![[E; TOP_K]; STEP],
            d_x: vec![0.0; H],
            d_xn: vec![0.0; H],
            d_y: vec![0.0; H],
            layers,
            logits: vec![0.0; STEP * VOCAB_SIZE],
            streams: vec![0.0; HC * H],
            x: vec![0.0; H],
            xn: vec![0.0; H],
            y: vec![0.0; H],
            mean: vec![0.0; H],
            hc_flat: vec![0.0; HC * H],
            hc_coef: vec![0.0; HC_COEFS],
            k_mixed: vec![0.0; CONV_DIM],
            k_conv: vec![0.0; CONV_DIM],
            k_fa: vec![0.0; KDA_DIM],
            k_f: vec![0.0; KDA_QKV],
            k_beta: vec![0.0; KDA_HEADS],
            k_ga: vec![0.0; KDA_DIM],
            k_gate: vec![0.0; KDA_QKV],
            k_normed: vec![0.0; KDA_QKV],
            k_heads: vec![0.0; KDA_HEADS * kda::HEAD_SCRATCH],
            m_qr: vec![0.0; Q_LORA_RANK],
            m_qa: vec![0.0; Q_LORA_RANK],
            m_q: vec![0.0; MLA_HEADS * Q_HEAD],
            m_kva: vec![0.0; KV_LORA_RANK],
            m_ik: vec![0.0; INDEX_HEAD_DIM],
            m_iq: vec![0.0; INDEX_N_HEADS * INDEX_HEAD_DIM],
            m_iw: vec![0.0; INDEX_N_HEADS],
            m_pscore: vec![0.0; max_ctx.div_ceil(INDEX_KPOOL) + 1],
            m_pidx: vec![0; max_ctx.div_ceil(INDEX_KPOOL) + 1],
            m_sel: vec![0; DENSE_BELOW],
            b_lat: vec![0u16; STEP * KV_LORA_RANK],
            m_heads: vec![0.0; MLA_HEADS * V_HEAD_DIM],
            m_hs: vec![0.0; MLA_HEADS * mla::HEAD_SCRATCH],
            f_gu: vec![0.0; 2 * INTERMEDIATE_SIZE],
            f_h: vec![0.0; INTERMEDIATE_SIZE],
            f_expert_out: vec![0.0; TOP_K * H],
            f_shared: vec![0.0; H],
            f_score: vec![0.0; E],
            d_scores: vec![0.0; STEP * E],
            b_x: vec![0.0; STEP * H],
            b_xn: vec![0.0; STEP * H],
            b_y: vec![0.0; STEP * H],
            b_score: vec![0.0; STEP * ROUTE_ROWS],
            b_out: vec![0.0; STEP * TOP_K * H],
        }
    }

    /// Of every pick so far, the share that went to each layer's `n` most-picked experts, for each `n` given:
    /// how much of the routed stream a card holding the hot experts would serve.
    pub fn coverage(&self, ns: &[usize]) -> Vec<f64> {
        let mut covered = vec![0u64; ns.len()];
        let mut total = 0u64;
        for l in 0..NUM_HIDDEN_LAYERS {
            let mut f: Vec<u32> = self.freq[l * E..(l + 1) * E].to_vec();
            f.sort_unstable_by(|a, b| b.cmp(a));
            total += f.iter().map(|&v| v as u64).sum::<u64>();
            for (i, &n) in ns.iter().enumerate() {
                covered[i] += f[..n.min(E)].iter().map(|&v| v as u64).sum::<u64>();
            }
        }
        covered.iter().map(|&c| c as f64 / total.max(1) as f64).collect()
    }

    /// Install `rows` (`[n][POSITION]`) as positions `0..n` and `snap` (`[SNAPSHOT]`)
    /// as the KDA layers' state at `n`; the MLA layers' pool keys are rebuilt from the rows.
    pub fn seed(&mut self, w: &Weights, rows: &[f32], snap: &[f32]) {
        assert_eq!(rows.len() % POSITION_W, 0);
        assert_eq!(snap.len(), layout::SNAPSHOT);
        let n = rows.len() / POSITION_W;
        let words: &[u16] = unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u16, rows.len() * 2) };
        assert!(n <= self.max_ctx);
        let (mut ki, mut mi) = (0, 0);
        for (l, st) in self.layers.iter_mut().enumerate() {
            match (st, &w.layers[l].attn) {
                (LayerState::Kda { window, mem }, _) => {
                    let base = ki * layout::KDA_STATE;
                    window.copy_from_slice(&snap[base..base + KDA_WINDOW]);
                    mem.copy_from_slice(&snap[base + KDA_WINDOW..base + layout::KDA_STATE]);
                    if let Some(p) = self.placed.as_mut() {
                        p.seed(l, window, mem);
                    }
                    ki += 1;
                }
                (LayerState::Mla { cache, pools }, Attn::Mla(mw)) => {
                    for p in 0..n {
                        let at = p * POSITION_W * 2 + mi * MLA_ROW_U16;
                        cache[p * MLA_ROW_U16..(p + 1) * MLA_ROW_U16].copy_from_slice(&words[at..at + MLA_ROW_U16]);
                    }
                    for j in 0..n / INDEX_KPOOL {
                        let (c, pk) = (&*cache, &mut pools[j * INDEX_HEAD_DIM..(j + 1) * INDEX_HEAD_DIM]);
                        mla::pool_key(mw, c, j, pk);
                    }
                    if let Some(pl) = self.placed.as_mut() {
                        for p in 0..n {
                            pl.mla_seed(l, p, mla::latent(cache, p));
                        }
                        pl.ix_pools(l, pools, 0, n / INDEX_KPOOL);
                    }
                    mi += 1;
                }
                _ => unreachable!("a layer's state and weights disagree"),
            }
        }
    }
    /// The bundle's prefix file: its rows, then its snapshot.
    pub fn seed_prefix(&mut self, w: &Weights, prefix: &[f32]) {
        assert_eq!(prefix.len(), layout::PREFIX_LEN);
        let (rows, rest) = prefix.split_at(FIRST_POS * POSITION);
        let (snap, hidden) = rest.split_at(layout::SNAPSHOT);
        self.prefix_hidden.copy_from_slice(hidden);
        self.seed(w, &pack_rows(rows), snap)
    }
}
/// The bundle's f32 rows (`[n][POSITION]`) in the node's packed form (`[n][POSITION_W]` words; the
/// latent and key to bf16): what `seed` and the context store take.
pub fn pack_rows(rows: &[f32]) -> Vec<f32> {
    assert_eq!(rows.len() % POSITION, 0);
    let n = rows.len() / POSITION;
    let mut packed = vec![0f32; n * POSITION_W];
    let words: &mut [u16] = unsafe { std::slice::from_raw_parts_mut(packed.as_mut_ptr() as *mut u16, packed.len() * 2) };
    for p in 0..n {
        for mi in 0..layout::MLA_LAYERS {
            mla::pack_row(&rows[p * POSITION + mi * MLA_ROW..p * POSITION + (mi + 1) * MLA_ROW], &mut words[p * POSITION_W * 2 + mi * MLA_ROW_U16..p * POSITION_W * 2 + (mi + 1) * MLA_ROW_U16]);
        }
    }
    packed
}
impl<'p, R: Routed> CpuModel<'p, R> {
    /// The KDA layers' states back to what they were after token `slot` of the last verify-sized step
    /// (its drafts past `slot` rejected). The cards' only.
    pub fn rewind(&mut self, slot: usize) {
        let pl = self.placed.as_mut().expect("rewind: the KDA ring is the cards'");
        for l in 0..NUM_HIDDEN_LAYERS {
            if LINEAR_ATTENTION[l] {
                pl.kda_rewind(l, slot);
            }
        }
    }
    /// Prefix position `pos`'s hidden state (the streams' mean before the final norm).
    pub fn prefix_hidden(&self, pos: usize) -> &[f32] {
        &self.prefix_hidden[pos * H..(pos + 1) * H]
    }
    /// The last computed token's hidden state (the streams' mean before the final norm, `[H]`): the
    /// cards' when placed (token 0 of the last step), else this realization's.
    pub fn hidden(&mut self, out: &mut [f32]) {
        match self.placed.as_mut() {
            Some(pl) => pl.layer_out(NUM_HIDDEN_LAYERS - 1, 0, out),
            None => out.copy_from_slice(&self.mean),
        }
    }
    /// The draft layer (multi-token prediction) at position `pos`: `eh_proj` over the next token's
    /// normed embedding and the normed hidden state `hidden` (the main model's at `pos`, or the
    /// draft's own `hidden_out` of the position before when drafting further), then its MLA attention
    /// over its own cache (its row at `pos` written) and its MoE MLP as a plain residual block, the
    /// shared head's norm and the shared head into `logits` — the draft of the token after the next.
    /// `hidden_out` receives the block's output. The CPU realization; the experts through `routed`
    /// as layer `MTP_LAYER`.
    pub fn draft(&mut self, w: &Weights, hidden: &[f32], next_tok: u32, pos: usize, logits: &mut [f32], hidden_out: &mut [f32], tr: &mut impl Trace) {
        let s = self;
        let m = w.mtp.as_ref().expect("the weights carry no draft layer");
        let LayerState::Mla { cache, pools } = s.mtp.as_mut().expect("no draft state") else { unreachable!() };
        let l = layout::MTP_LAYER;
        // eh_proj over [enorm(embed(next)) | hnorm(hidden)]
        let row = &w.embed[next_tok as usize * H..(next_tok as usize + 1) * H];
        for d in 0..H {
            s.d_y[d] = widen(row[d]);
        }
        {
            let (e, h) = s.d_cat.split_at_mut(H);
            rmsnorm::<H>(&s.d_y, m.enorm, RMS_NORM_EPS, e);
            rmsnorm::<H>(hidden, m.hnorm, RMS_NORM_EPS, h);
        }
        gemv::bf16_par::<H, { 2 * H }>(m.eh_proj, &s.d_cat, &mut s.d_x, s.pool);
        tr.edge(l, "mtp.eh", &s.d_x);
        // the attention block
        rmsnorm::<H>(&s.d_x, m.input_ln, RMS_NORM_EPS, &mut s.d_xn);
        {
            let mut ms = mla::Scratch { qr: &mut s.m_qr, qa: &mut s.m_qa, q: &mut s.m_q, kva: &mut s.m_kva, ik: &mut s.m_ik, iq: &mut s.m_iq, iw: &mut s.m_iw, pscore: &mut s.m_pscore, pidx: &mut s.m_pidx, sel: &mut s.m_sel, heads: &mut s.m_heads, hs: &mut s.m_hs };
            {
                let (latent, ikey, igate) = mla::row(cache, pos);
                mla::project(&m.attn, &s.d_xn, &mut ms, igate, s.pool);
                mla::finish_row(&m.attn, ms.kva, ms.ik, latent, ikey);
            }
            let n = mla::select_at(l, &m.attn, cache, pools, pos, &mut ms, tr);
            mla::heads(l, &m.attn, cache, &mut ms, n, s.pool, tr);
            mla::output(&m.attn, ms.heads, &mut s.d_y, s.pool);
        }
        for d in 0..H {
            s.d_x[d] += s.d_y[d];
        }
        tr.edge(l, "mtp.attn", &s.d_x);
        // the MoE block
        rmsnorm::<H>(&s.d_x, m.post_ln, RMS_NORM_EPS, &mut s.d_xn);
        {
            let mut fs = ffn::Scratch { gu: &mut s.f_gu, h: &mut s.f_h, expert_out: &mut s.f_expert_out, shared: &mut s.f_shared, score: &mut s.f_score };
            ffn::moe_routed(l, &m.ffn, &s.d_xn, &mut fs, s.pool, &mut s.routed, None, tr, &mut s.d_y);
            ffn::moe_shared(&m.ffn, &s.d_xn, &mut fs, s.pool);
            tr.edge(l, "mtp.shared", fs.shared);
            for d in 0..H {
                s.d_x[d] += s.d_y[d] + fs.shared[d];
            }
        }
        tr.edge(l, "mtp.ffn", &s.d_x);
        hidden_out.copy_from_slice(&s.d_x);
        // the shared head
        rmsnorm::<H>(&s.d_x, m.head_norm, RMS_NORM_EPS, &mut s.d_xn);
        gemv::bf16_par::<VOCAB_SIZE, H>(w.lm_head, &s.d_xn, &mut logits[..VOCAB_SIZE], s.pool);
    }
    /// The draft layer on the cards over the last step's `n` positions (`pos..`): each position's
    /// hidden state (`hiddens: [n][H]`, the main model's there) with the token that follows it
    /// (`nexts[i]`, the step's next token or the draw after it) through `eh_proj`, the attention over
    /// the draft's own cache (its rows at these positions written), the MoE (routed experts here as
    /// layer `MTP_LAYER`, the shared expert there), and for the last position the shared head:
    /// `logits` is the draft of the token after `nexts[n - 1]`. Run after every step so the draft's
    /// cache has every position.
    pub fn draft_batch(&mut self, w: &Weights, hiddens: &[f32], nexts: &[u32], pos: usize, n: usize, logits: &mut [f32], tr: &mut impl Trace) {
        let s = self;
        prow::zone!("track-zones", "draft");
        let t0 = std::time::Instant::now();
        let m = w.mtp.as_ref().expect("the weights carry no draft layer");
        let LayerState::Mla { cache, pools } = s.mtp.as_mut().expect("no draft state") else { unreachable!() };
        let pl = s.placed.as_mut().expect("draft_batch: the cards'");
        let l = layout::MTP_LAYER;
        assert!(n >= 1 && n <= STEP && nexts.len() >= n && hiddens.len() >= n * H);
        // the draft's phases as zones inside `draft` (081): its input, attention, router, experts, FFN, head
        // every position's [enorm(embed(next)) | hnorm(hidden)]
        // a position a pool item (a prefill step's 1024 went one at a time on this thread, ~16 ms, 2026-10-20)
        let z = prow::Timestamp::now();
        {
            let (cp, hp, emb, en, hn) = (s.d_cats.as_mut_ptr() as usize, hiddens.as_ptr() as usize, w.embed, m.enorm, m.hnorm);
            s.pool.run(n, &move |i| {
                let row = &emb[nexts[i] as usize * H..(nexts[i] as usize + 1) * H];
                let mut y = [0f32; H];
                for d in 0..H {
                    y[d] = widen(row[d]);
                }
                let cat = unsafe { std::slice::from_raw_parts_mut((cp as *mut f32).add(i * 2 * H), 2 * H) };
                let (e, h) = cat.split_at_mut(H);
                rmsnorm::<H>(&y, en, RMS_NORM_EPS, e);
                rmsnorm::<H>(unsafe { std::slice::from_raw_parts((hp as *const f32).add(i * H), H) }, hn, RMS_NORM_EPS, h);
            });
        }
        pl.mtp_begin(&s.d_cats[..n * 2 * H], n);
        let z = draft_zone("draft.in", z);
        // inside `draft.attn`: the wait for the projections, the cache rows, the selection, the attend and output
        let proj = pl.mla_proj(l);
        let y = draft_zone("draft.attn.proj", z);
        mla_rows(&m.attn, cache, pools, pos, n, proj);
        let y = draft_zone("draft.attn.rows", y);
        pl.ix_pools(l, pools, pos / INDEX_KPOOL, (pos + n) / INDEX_KPOOL);
        pl.ix_select(l, pos, n, None);
        let y = draft_zone("draft.attn.select", y);
        for i in 0..n {
            s.b_lat[i * KV_LORA_RANK..(i + 1) * KV_LORA_RANK].copy_from_slice(mla::latent(cache, pos + i));
        }
        pl.mla_attend(l, n, pos, &s.b_lat[..n * KV_LORA_RANK]);
        let y = draft_zone("draft.attn.attend", y);
        pl.mtp_attn_out(&mut s.b_x[..n * H], n);
        draft_zone("draft.attn.out", y);
        tr.edge(l, "mtp.attn", &s.b_x[(n - 1) * H..n * H]);
        let z = draft_zone("draft.attn", z);
        // the routing and the routed experts here, every token's
        let mut picks = [[0usize; TOP_K]; STEP];
        let mut wts = [[0f32; TOP_K]; STEP];
        // every position's router rows in one pool run, an item a (position, `pool::ROWS` rows): a prefill step's
        // positions went through a run each (~33 ms a 1024-token step, 2026-10-20). Each item norms its position
        // itself (the first block keeps it); the rows are `bf16_par`'s, so the scores are the same bits
        {
            const BLOCKS: usize = E.div_ceil(common::pool::ROWS);
            let (xp, xnp, sp, rw, ln) = (s.b_x.as_ptr() as usize, s.b_xn.as_mut_ptr() as usize, s.d_scores.as_mut_ptr() as usize, m.ffn.router, m.post_ln);
            s.pool.run_at(n * BLOCKS, &move |j| (rw.as_ptr() as usize + (j % BLOCKS) * common::pool::ROWS * H * 2) as *const u8, &move |j| {
                let (i, b) = (j / BLOCKS, j % BLOCKS);
                let x = unsafe { std::slice::from_raw_parts((xp as *const f32).add(i * H), H) };
                let mut xn = [0f32; H];
                rmsnorm::<H>(x, ln, RMS_NORM_EPS, &mut xn);
                if b == 0 {
                    unsafe { std::slice::from_raw_parts_mut((xnp as *mut f32).add(i * H), H) }.copy_from_slice(&xn);
                }
                let out = unsafe { std::slice::from_raw_parts_mut((sp as *mut f32).add(i * E), E) };
                gemv::bf16_rows::<H>(rw, &xn, out, b * common::pool::ROWS, ((b + 1) * common::pool::ROWS).min(E));
            });
        }
        {
            // each position's picks a pool item (serially, ~9 µs a position: 9.6 ms a 1024-token step)
            let (sp, pp, wp, f) = (s.d_scores.as_mut_ptr() as usize, picks.as_mut_ptr() as usize, wts.as_mut_ptr() as usize, &m.ffn);
            s.pool.run(n, &move |i| {
                let (idx, wt) = ffn::route_scores(f, unsafe { std::slice::from_raw_parts_mut((sp as *mut f32).add(i * E), E) }, None);
                unsafe {
                    *(pp as *mut [usize; TOP_K]).add(i) = idx;
                    *(wp as *mut [f32; TOP_K]).add(i) = wt;
                }
            });
        }
        let z = draft_zone("draft.route", z);
        s.routed.run_batch(l, &picks[..n], &s.b_xn[..n * H], &mut s.b_out[..n * TOP_K * H]);
        let z = draft_zone("draft.experts", z);
        // each position's weighted sum a pool item, straight into the card's staging (a prefill step's 1024 went
        // serially on this thread and were then copied, ~24 ms with the shared expert, 2026-10-20); the same adds
        let routed = pl.mtp_routed();
        {
            let (op, yp, wp) = (s.b_out.as_ptr() as usize, routed as usize, wts.as_ptr() as usize);
            s.pool.run(n, &move |i| {
                let wt = unsafe { *(wp as *const [f32; TOP_K]).add(i) };
                let mut y = [0f32; H];
                for k in 0..TOP_K {
                    let eo = unsafe { std::slice::from_raw_parts((op as *const f32).add((i * TOP_K + k) * H), H) };
                    for d in 0..H {
                        y[d] += wt[k] * eo[d];
                    }
                }
                unsafe { std::slice::from_raw_parts_mut((yp as *mut f32).add(i * H), H) }.copy_from_slice(&y);
            });
        }
        tr.edge(l, "routed", unsafe { std::slice::from_raw_parts(routed.add((n - 1) * H), H) });
        let pl = s.placed.as_mut().expect("draft_batch: the cards'");
        pl.mtp_ffn(&mut s.b_x[..n * H], n);
        tr.edge(l, "mtp.ffn", &s.b_x[(n - 1) * H..n * H]);
        s.d_hid.copy_from_slice(&s.b_x[(n - 1) * H..n * H]);
        let z = draft_zone("draft.ffn", z);
        pl.mtp_head(n - 1, logits);
        draft_zone("draft.head", z);
        // the draft's expert time is the draft's (the step's `experts` lap takes what `elapsed` holds)
        s.routed.elapsed();
        s.profile.draft += t0.elapsed().as_secs_f64();
    }
    /// The next depth of drafting: the draft layer at `pos` (the position after the last drafted) from its
    /// own output there and `next` (the token the last draft proposed, standing for an accepted draft);
    /// `logits` is the draft of the token after `next`.
    pub fn draft_next(&mut self, w: &Weights, next: u32, pos: usize, logits: &mut [f32], tr: &mut impl Trace) {
        prow::zone!("track-zones", "draft.next");
        let hid = self.d_hid.clone(); // `draft_batch` writes `d_hid` for the depth after this one
        self.draft_batch(w, &hid, &[next], pos, 1, logits, tr);
    }
    /// The KDA layers' state now, `[SNAPSHOT]`.
    pub fn snapshot(&self, out: &mut [f32]) {
        let mut ki = 0;
        for (l, st) in self.layers.iter().enumerate() {
            if let LayerState::Kda { window, mem } = st {
                let base = ki * layout::KDA_STATE;
                let (w, m) = out[base..base + layout::KDA_STATE].split_at_mut(KDA_WINDOW);
                match self.placed.as_ref() {
                    Some(p) => p.snapshot(l, w, m),
                    None => {
                        w.copy_from_slice(window);
                        m.copy_from_slice(mem);
                    }
                }
                ki += 1;
            }
        }
    }
    /// Position `pos`'s rows in the node's packed form, `[POSITION_W]` words.
    pub fn rows(&self, pos: usize, out: &mut [f32]) {
        let words: &mut [u16] = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u16, out.len() * 2) };
        let mut mi = 0;
        for st in &self.layers {
            if let LayerState::Mla { cache, .. } = st {
                words[mi * MLA_ROW_U16..(mi + 1) * MLA_ROW_U16].copy_from_slice(&cache[pos * MLA_ROW_U16..(pos + 1) * MLA_ROW_U16]);
                mi += 1;
            }
        }
    }

    /// Token `tok` at position `pos` — the state holds `0..pos` — leaving its
    /// logits in `self.logits`. `forced`: the MoE layers' picks to take instead
    /// of the router's, `[MoE layer][TOP_K]` (bring-up only). With cards, the
    /// token runs there (`step_cards`); here is the CPU realization.
    pub fn step(&mut self, w: &Weights, tok: u32, pos: usize, forced: Option<&[[usize; TOP_K]]>, tr: &mut impl Trace, logits: bool) {
        assert!(pos < self.max_ctx);
        if self.placed.is_some() {
            return self.step_batch(w, &[tok], pos, forced, tr, if logits { Logits::Last } else { Logits::None });
        }
        // the CPU realization's own zone (with cards, step_batch names the step)
        prow::zone!("track-zones", "token");
        let s = self;
        let row = &w.embed[tok as usize * H..(tok as usize + 1) * H];
        for j in 0..HC {
            for d in 0..H {
                s.streams[j * H + d] = widen(row[d]);
            }
        }
        tr.edge(NONE, "embed", &s.streams[..H]);
        let mut t = std::time::Instant::now();
        // a lap is a phase from the previous lap to now: its time goes to the profile and, with
        // track-zones, it is a zone of that name on this thread's lane
        #[cfg(feature = "track-zones")]
        let mut mark = prow::Timestamp::now();
        let mut lap = |acc: &mut f64, _name: &'static str| {
            let now = std::time::Instant::now();
            *acc += (now - t).as_secs_f64();
            t = now;
            #[cfg(feature = "track-zones")]
            {
                let end = prow::Timestamp::now();
                prow::zone_at!("track-zones", _name, prow::lane(), mark, end);
                mark = end;
            }
        };
        let p = &mut s.profile;
        p.tokens += 1;
        for l in 0..NUM_HIDDEN_LAYERS {
            let lw = &w.layers[l];
            let mut hcs = hc::Scratch { flat: &mut s.hc_flat, coef: &mut s.hc_coef };
            let mix = hc::pre(&lw.hc_attn, &s.streams, &mut hcs, s.pool, &mut s.x);
            rmsnorm::<H>(&s.x, &lw.input_ln, RMS_NORM_EPS, &mut s.xn);
            tr.edge(l, "attn_in", &s.xn);
            lap(&mut p.hc, "hc");
            match (&lw.attn, &mut s.layers[l]) {
                (Attn::Kda(a), LayerState::Kda { window, mem }) => {
                    let mut ks = kda::Scratch { mixed: &mut s.k_mixed, conv: &mut s.k_conv, fa: &mut s.k_fa, f: &mut s.k_f, beta: &mut s.k_beta, ga: &mut s.k_ga, gate: &mut s.k_gate, normed: &mut s.k_normed, heads: &mut s.k_heads };
                    kda::step(l, a, &s.xn, window, mem, &mut ks, s.pool, tr, &mut s.y);
                    lap(&mut p.kda_cpu, "kda_cpu");
                }
                (Attn::Mla(a), LayerState::Mla { cache, pools }) => {
                    let mut ms = mla::Scratch { qr: &mut s.m_qr, qa: &mut s.m_qa, q: &mut s.m_q, kva: &mut s.m_kva, ik: &mut s.m_ik, iq: &mut s.m_iq, iw: &mut s.m_iw, pscore: &mut s.m_pscore, pidx: &mut s.m_pidx, sel: &mut s.m_sel, heads: &mut s.m_heads, hs: &mut s.m_hs };
                    {
                        let (latent, ikey, igate) = mla::row(cache, pos);
                        mla::project(a, &s.xn, &mut ms, igate, s.pool);
                        mla::finish_row(a, ms.kva, ms.ik, latent, ikey);
                    }
                    lap(&mut p.mla_proj, "mla_proj");
                    let n = mla::select_at(l, a, cache, pools, pos, &mut ms, tr);
                    mla::heads(l, a, cache, &mut ms, n, s.pool, tr);
                    lap(&mut p.mla_attn, "mla_attn");
                    mla::output(a, ms.heads, &mut s.y, s.pool);
                    lap(&mut p.mla_out, "mla_out");
                }
                _ => unreachable!("a layer's state and weights disagree"),
            }
            tr.edge(l, "attn_out", &s.y);
            hc::post(&mix, &s.y, &mut s.streams);

            let mut hcs = hc::Scratch { flat: &mut s.hc_flat, coef: &mut s.hc_coef };
            let mix = hc::pre(&lw.hc_ffn, &s.streams, &mut hcs, s.pool, &mut s.x);
            rmsnorm::<H>(&s.x, &lw.post_ln, RMS_NORM_EPS, &mut s.xn);
            tr.edge(l, "ffn_in", &s.xn);
            lap(&mut p.hc, "hc");
            let mut fs = ffn::Scratch { gu: &mut s.f_gu, h: &mut s.f_h, expert_out: &mut s.f_expert_out, shared: &mut s.f_shared, score: &mut s.f_score };
            match &lw.ffn {
                Ffn::Dense(d) => {
                    ffn::dense(l, d, &s.xn, &mut fs, s.pool, tr, &mut s.y);
                    lap(&mut p.dense_ffn, "dense_ffn");
                }
                Ffn::Moe(m) => {
                    let picks = ffn::moe_routed(l, m, &s.xn, &mut fs, s.pool, &mut s.routed, forced.map(|f| &f[l - FIRST_K_DENSE_REPLACE]), tr, &mut s.y);
                    account(p, &mut s.last_picks, &mut s.freq, s.pool, l, picks, &[]);
                    let e = s.routed.elapsed();
                    p.experts += e;
                    lap(&mut p.route, "route");
                    p.route -= e;
                    ffn::moe_shared(m, &s.xn, &mut fs, s.pool);
                    tr.edge(l, "shared", fs.shared);
                    for i in 0..H {
                        s.y[i] += fs.shared[i];
                    }
                    lap(&mut p.shared, "shared");
                }
            }
            tr.edge(l, "ffn_out", &s.y);
            hc::post(&mix, &s.y, &mut s.streams);
            for d in 0..H {
                let mut acc = 0f32;
                for j in 0..HC {
                    acc += s.streams[j * H + d];
                }
                s.mean[d] = acc / HC as f32;
            }
            tr.edge(l, "out", &s.mean);
            lap(&mut p.hc, "hc");
        }
        if logits {
            rmsnorm::<H>(&s.mean, &w.norm, RMS_NORM_EPS, &mut s.xn);
            gemv::bf16_par::<VOCAB_SIZE, H>(w.lm_head, &s.xn, &mut s.logits[..VOCAB_SIZE], s.pool);
            tr.edge(NONE, "logits", &s.logits[..VOCAB_SIZE]);
        }
        lap(&mut p.head, "head");
    }

    /// A step of `toks` (at `pos..pos + n`, `n ≤ STEP`) on the cards: the host posts each layer's
    /// halves over the step and works only where the routed experts are its own — it waits for
    /// the FFN half's inputs, runs the step's experts grouped by expert, hands their outputs back —
    /// and reads the last token's logits at the end when asked. One token is a step of one.
    /// Profile: `hc` is the posting, `kda_card` the wait for the cards (the attention block and
    /// the hyper-connections), `route` the host's router and sum around the experts, the MLA laps
    /// the host's selection, `head` the wait for the logits.
    pub fn step_batch(&mut self, w: &Weights, toks: &[u32], pos: usize, forced: Option<&[[usize; TOP_K]]>, tr: &mut impl Trace, logits: Logits) {
        // the step's zone is named by what the step is: a prefill step (wider than a verify step), a
        // SCORE step (prefill-shaped, every token's logits), a verify step (a token and its drafts) or
        // a single token with no draft to verify
        prow::zone!("track-zones", match toks.len() {
            n if n > crate::SPEC_TOKENS && matches!(logits, Logits::All) => "score",
            n if n > crate::SPEC_TOKENS => "prefill",
            1 => "token",
            _ => "verify",
        });
        let s = self;
        let n = toks.len();
        assert!(n >= 1 && n <= STEP && pos + n <= s.max_ctx);
        for (i, &tok) in toks.iter().enumerate() {
            let row = &w.embed[tok as usize * H..(tok as usize + 1) * H];
            for d in 0..H {
                s.b_x[i * H + d] = widen(row[d]);
            }
        }
        if n == 1 {
            tr.edge(NONE, "embed", &s.b_x[..H]);
        }
        let pl = s.placed.as_mut().unwrap();
        pl.begin(&s.b_x[..n * H], n);
        let mut t = std::time::Instant::now();
        // a lap is a phase from the previous lap to now: its time goes to the profile and, with
        // track-zones, it is a zone of that name on this thread's lane
        #[cfg(feature = "track-zones")]
        let mut mark = prow::Timestamp::now();
        let mut lap = |acc: &mut f64, _name: &'static str| {
            let now = std::time::Instant::now();
            *acc += (now - t).as_secs_f64();
            t = now;
            #[cfg(feature = "track-zones")]
            {
                let end = prow::Timestamp::now();
                prow::zone_at!("track-zones", _name, prow::lane(), mark, end);
                mark = end;
            }
        };
        let p = &mut s.profile;
        p.tokens += n;
        let mut pre_next = [E; plan::PLAN];
        let mut pre_picks = [E; plan::PLAN];
        // the step as chunks in flight (037), a layer of a chunk in three phases: its card half (the attention and
        // the FFN's card launches, posted), its dispatch (its FFN input waited for, its resident experts launched)
        // and its run (the FFN's output half posted, the host's experts, the join, the sum and the handoff). The
        // order: A's card half of layer l, B's run of l - 1, A's dispatch of l, B's card half of l, A's run of l,
        // B's dispatch of l — so a chunk's residents sit ahead of the other's attention in the cards' queues, and the
        // host's experts of one chunk overlap the cards' half of the other. Each card's queue is in order, so a
        // chunk's attention follows the other's on the KDA state and the latents. One chunk is the plain order.
        const CARD: u8 = 0;
        const DISPATCH: u8 = 1;
        const RUN: u8 = 2;
        let chunks = crate::chunk_count(n);
        let mut order: Vec<(u8, usize, usize)> = Vec::with_capacity(3 * chunks * NUM_HIDDEN_LAYERS);
        for l in 0..NUM_HIDDEN_LAYERS {
            order.push((CARD, 0, l));
            if chunks > 1 && l > 0 {
                order.push((RUN, 1, l - 1));
            }
            order.push((DISPATCH, 0, l));
            if chunks > 1 {
                order.push((CARD, 1, l));
            }
            order.push((RUN, 0, l));
            if chunks > 1 {
                order.push((DISPATCH, 1, l));
            }
        }
        if chunks > 1 {
            order.push((RUN, 1, NUM_HIDDEN_LAYERS - 1));
        }
        let mut jobs_of = [0usize; crate::CHUNKS];
        for (phase, ch, l) in order {
            let (t0, n) = crate::chunk_of(toks.len(), ch);
            let pos = pos + t0;
            pl.chunk(ch, t0);
            let lw = &w.layers[l];
            let moe = match &lw.ffn {
                Ffn::Moe(m) => Some(m),
                Ffn::Dense(_) => None,
            };
            if phase == CARD {
                let is_mla = matches!(lw.attn, Attn::Mla(_));
                if is_mla {
                    pl.mla_post(l, n);
                } else {
                    pl.attention(l, n);
                }
                lap(&mut p.hc, "hc");
                // the plan for this layer, made by the layer before (the prefetch predictor on the cards); `pre_picks` scores it
                pre_picks = std::mem::replace(&mut pre_next, [E; plan::PLAN]);
                if let (Attn::Mla(a), LayerState::Mla { cache, pools }) = (&lw.attn, &mut s.layers[l]) {
                    // every token's projections off the card into its cache row, its indexer query and head
                    // weights kept; the pools completed along the way gain their keys
                    let proj = pl.mla_proj(l);
                    mla_rows(a, cache, pools, pos, n, proj);
                    lap(&mut p.mla_proj, "mla_proj");
                    // every token's selection on the card (this step's completed pools mirrored first): the
                    // scores, the picks, straight into the attend's lists — the one-token path had scored and
                    // picked on one thread, 150 ms a token at 17k positions (2026-10-17); then the host's round
                    // trip for the picks, 1.8 ms a token at 18k
                    pl.ix_pools(l, pools, pos / INDEX_KPOOL, (pos + n) / INDEX_KPOOL);
                    let mut count = 0usize;
                    pl.ix_select(l, pos, n, (n == 1 && tr.wants_edges()).then_some(&mut count));
                    if n == 1 && tr.wants_edges() {
                        tr.edge(l, "self_attn.indexer.count", &[count as f32]);
                    }
                    lap(&mut p.mla_attn, "mla_attn");
                    // the step's latents gathered off the cache rows; every token's attention in one launch
                    for i in 0..n {
                        s.b_lat[i * KV_LORA_RANK..(i + 1) * KV_LORA_RANK].copy_from_slice(mla::latent(cache, pos + i));
                    }
                    pl.mla_attend(l, n, pos, &s.b_lat[..n * KV_LORA_RANK]);
                    pl.mla_out(l, n);
                    lap(&mut p.mla_out, "mla_out");
                }
                pl.ffn(l, n);
                continue;
            }
            if phase == DISPATCH && moe.is_none() {
                continue;
            }
            if let Some(m) = moe {
                // the FFN input as the card made it (001/064): the activations quantized once for both tiers, the
                // router's picks and weights — the host's norm, quantize and routing moved to the FFN card's `hc_pre`
                const HG: usize = H / common::q4::GROUP;
                if phase == DISPATCH {
                    pl.ffn_input(l, n, &mut s.b_xq[t0 * H..(t0 + n) * H], &mut s.b_xs[t0 * HG..(t0 + n) * HG], &mut s.b_xg[t0 * HG..(t0 + n) * HG], &mut s.b_picks[t0..t0 + n], &mut s.b_wts[t0..t0 + n], s.pool);
                    lap(&mut p.kda_card, "kda_card");
                    let (picks, wts) = (&mut s.b_picks[t0..t0 + n], &mut s.b_wts[t0..t0 + n]);
                    let (xq, xs, xg) = (&s.b_xq[t0 * H..(t0 + n) * H], &s.b_xs[t0 * HG..(t0 + n) * HG], &s.b_xg[t0 * HG..(t0 + n) * HG]);
                    // bring-up: forced picks and the traces take the card's logits and input through the host's own routing and norm
                    if forced.is_some() || tr.wants_edges() || pl.checks() {
                        pl.ffn_logits(l, n, &mut s.b_x, &mut s.b_score);
                        let forced_l = forced.map(|f| &f[l - FIRST_K_DENSE_REPLACE]);
                        for i in 0..n {
                            common::rmsnorm::rmsnorm_tree::<H>(&s.b_x[i * H..(i + 1) * H], lw.post_ln, RMS_NORM_EPS, &mut s.b_xn[i * H..(i + 1) * H]);
                            let (idx, wt) = ffn::route_scores(m, &mut s.b_score[i * ROUTE_ROWS..i * ROUTE_ROWS + E], forced_l);
                            if forced.is_some() {
                                picks[i] = idx;
                                wts[i] = wt;
                            } else {
                                // the card's act phase (`hc_pre`) against the host's own arithmetic: picks, weights, the quantized activation
                                let (mut q, mut sc, mut g) = (vec![0i8; H], vec![0f32; HG], vec![0i32; HG]);
                                common::q4::QAct { q: &mut q, scale: &mut sc, gsum: &mut g }.quantize(&s.b_xn[i * H..(i + 1) * H]);
                                let dq = (0..H).filter(|&d| q[d] != xq[i * H + d]).count();
                                let ds = (0..HG).filter(|&d| sc[d].to_bits() != xs[i * HG + d].to_bits() || g[d] != xg[i * HG + d]).count();
                                let dw = (0..TOP_K).filter(|&k| wts[i][k].to_bits() != wt[k].to_bits()).count();
                                if picks[i] != idx || dq + ds + dw > 0 {
                                    let d0 = (0..H).find(|&d| q[d] != xq[i * H + d]);
                                    eprintln!("act check: layer {l} token {i}: picks card {:?} host {idx:?}; weights differ {dw} (card {:?} host {wt:?}); q differ {dq} of {H} (first {:?}: card {:?} host {:?}), scales/sums {ds} of {HG} (card s0 {:e} g0 {} host {:e} {})", picks[i], wts[i], d0, d0.map(|d| xq[i * H + d]), d0.map(|d| q[d]), xs[i * HG], xg[i * HG], sc[0], g[0]);
                                }
                            }
                        }
                    }
                    if n == 1 {
                        tr.edge(l, "ffn_in", &s.b_xn[..H]);
                        tr.topk(l, &picks[0], &wts[0]);
                    }
                    // the picks resident on the cards run there, concurrently with the host's (the dispatch marks them
                    // `E` in `picks`); the picks as routed are kept for the accounting
                    s.b_routed[t0..t0 + n].copy_from_slice(&picks[..n]);
                    let jobs = pl.xpt_dispatch(l, n, &mut picks[..n]);
                    p.resident += jobs;
                    p.picks += n * TOP_K;
                    jobs_of[ch] = jobs;
                    lap(&mut p.route, "route");
                    continue;
                }
                // the run: the FFN's output half posted behind the residents' launches (it waits for the host's word)
                let jobs = jobs_of[ch];
                let (picks, wts) = (&mut s.b_picks[t0..t0 + n], &mut s.b_wts[t0..t0 + n]);
                let (xq, xs, xg) = (&s.b_xq[t0 * H..(t0 + n) * H], &s.b_xs[t0 * HG..(t0 + n) * HG], &s.b_xg[t0 * HG..(t0 + n) * HG]);
                // the prefetch predictor on the cards, right ahead of the FFN's output half, which spins until the host's
                // experts are in: this layer's pair predicts the next (a decode step's prediction is the plan); the pairs
                // learn from the picks at the step's end
                pl.plan_step(l, n, &s.b_routed[t0..t0 + n]);
                pl.ffn_post(l, n);
                // the outputs: the layer's leader card's rows in its VRAM (002) — the host's experts write theirs through
                // the BAR, the cards' jobs theirs directly; the leader sums them
                let out: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(pl.expert_out(l).add(t0 * TOP_K * H), n * TOP_K * H) };
                // the host's FFN phase as laps: the route (the picks, the residents' dispatch), the experts' run,
                // the join of the cards' jobs, the word to the leader and the accounting — each a zone of its own
                s.routed.run_batch_act(l, &picks[..n], xq, xs, xg, out);
                lap(&mut p.experts, "experts");
                s.routed.elapsed(); // the pool's own timer, taken so it does not accumulate (the lap is the measure here)
                if jobs > 0 || n > SPEC_TOKENS {
                    pl.xpt_join(l, out);
                    lap(&mut p.join, "join");
                    // bring-up (a traced run): every resident pick's output against the host's on the same bytes
                    if tr.wants_edges() || pl.checks() {
                        let mut host = vec![0f32; TOP_K * H];
                        for i in 0..n {
                            for k in 0..TOP_K {
                                if picks[i][k] == E {
                                    let e = s.b_routed[t0 + i][k];
                                    s.routed.run(l, &[e; TOP_K], &s.b_xn[i * H..(i + 1) * H], &mut host);
                                    let card = &out[(i * TOP_K + k) * H..(i * TOP_K + k + 1) * H];
                                    if let Some(d) = (0..H).find(|&d| card[d].to_bits() != host[d].to_bits()) {
                                        let worst = (0..H).map(|d| (card[d] - host[d]).abs()).fold(0f32, f32::max);
                                        eprintln!("xpt mismatch: layer {l} expert {e} token {i} pick {k}: first at {d} card {:e} host {:e}, max |Δ| {worst:e}", card[d], host[d]);
                                    }
                                }
                            }
                        }
                    }
                }
                // the leader card sums the rows in pick order and folds them in (002); the host only says they are all there
                {
                    prow::zone!("track-zones", "done.word");
                    pl.ffn_done(l);
                }
                if n == 1 && tr.wants_edges() {
                    // bring-up: the routed sum as the leader makes it, from its rows, for the trace's edges
                    let y = &mut s.b_y[..H];
                    y.fill(0.0);
                    for k in 0..TOP_K {
                        for d in 0..H {
                            y[d] += wts[0][k] * out[k * H + d];
                        }
                    }
                    tr.edge(l, "ffn_in", &s.b_xn[..H]);
                    for k in 0..TOP_K {
                        tr.routed(l, k, &out[k * H..(k + 1) * H]);
                    }
                    tr.edge(l, "routed", &s.b_y[..H]);
                }
                {
                    prow::zone!("track-zones", "done.account");
                    for i in 0..n {
                        account(p, &mut s.last_picks, &mut s.freq, s.pool, l, s.b_routed[t0 + i], if n <= SPEC_TOKENS { &pre_picks[..] } else { &[] });
                    }
                }
                lap(&mut p.route, "done");
                // the next MoE layer's picks, predicted by its router over this layer's FFN input (the second
                // half of each token's scores), prefetched now that this layer's run is done (a run abandons
                // the prefetches in flight); a step of many tokens names most experts and runs none
                if let (true, Some(Ffn::Moe(_))) = (n <= SPEC_TOKENS, w.layers.get(l + 1).map(|lw| &lw.ffn)) {
                    // inside `slot`: the plan off the card (the learned predictor, `plan.rs`), its experts not resident, the
                    // prefetch's post. At most PREFETCH_CAP: a CCD's 32 MiB L3 kept less of a completed prefetch the more the
                    // layer prefetched (86 % of its lines at <= 10 experts, 56 % at 23+): past what the L3 retains a
                    // prefetched expert costs a DRAM read and evicts an earlier one (2026-10-20)
                    let z = prow::Timestamp::now();
                    pl.plan(l, &mut pre_next);
                    #[cfg(feature = "dump-pred")]
                    {
                        pl.ffn_logits(l, n, &mut s.b_x, &mut s.b_score);
                        let ahead = more_ahead(&mut **pl, w, l, n, &s.b_x, &s.b_picks[t0..t0 + n], &s.b_wts[t0..t0 + n]);
                        dump_pred(l, n, &s.b_routed[t0..t0 + n], &s.b_xq[t0 * H..(t0 + n) * H], &s.b_xs[t0 * HG..(t0 + n) * HG], &s.b_score, &ahead);
                    }
                    let z = draft_zone("slot.ahead", z);
                    let ranked: Vec<usize> = pre_next.iter().copied().filter(|&e| !pl.resident(l + 1, e)).take(PREFETCH_CAP).collect();
                    let z = draft_zone("slot.pick", z);
                    s.routed.prefetch(l + 1, &ranked);
                    draft_zone("slot.post", z);
                    lap(&mut p.slot, "slot");
                }
            }
            if tr.wants_edges() && n == 1 {
                pl.layer_out(l, 0, &mut s.mean);
                tr.edge(l, "out", &s.mean);
            }
        }
        // the head, the draft layer and anything after the layers address the step whole again
        pl.chunk(0, 0);
        // the predictor's updates (`plan_learn`): queued behind the head once its logits are read, so the cards run them
        // while the host samples and drafts (~1 ms they idle there); queued ahead of the head they held the logits,
        // `head` 1.93 → 2.31 ms a verify step (2026-10-21). A step without logits learns before its finish.
        match logits {
            Logits::None => {
                pl.plan_learn(n);
                pl.finish()
            }
            Logits::Last => pl.logits(n - 1, 1, &mut s.logits[(n - 1) * VOCAB_SIZE..n * VOCAB_SIZE]),
            Logits::All => pl.logits(0, n, &mut s.logits[..n * VOCAB_SIZE]),
        }
        if !matches!(logits, Logits::None) {
            pl.plan_learn(n);
        }
        if n == 1 && !matches!(logits, Logits::None) {
            tr.edge(NONE, "logits", &s.logits[..VOCAB_SIZE]);
        }
        lap(&mut p.head, "head");
    }
}

/// Which of a step's logits `step_batch` computes into `CpuModel::logits` (`[STEP][VOCAB_SIZE]`,
/// token `i`'s at row `i`): none (the step ends without the head), the last token's (a reply's
/// draw), or every token's (scoring, tracing).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Logits {
    None,
    Last,
    All,
}

/// A MoE layer's bookkeeping after its experts ran: the picks against the previous token's and the
/// early router's, their frequencies, the pool's locality and the prefetch's yield.
fn account(p: &mut Profile, last_picks: &mut [[usize; TOP_K]], freq: &mut [u32], pool: &dyn Pool, l: usize, picks: [usize; TOP_K], pre_picks: &[usize]) {
    p.reuse += picks.iter().filter(|e| last_picks[l].contains(e)).count();
    p.pre_agree += picks.iter().filter(|e| pre_picks.contains(e)).count();
    p.moe_layers += 1;
    last_picks[l] = picks;
    for &e in &picks {
        if e < E {
            freq[l * E + e] += 1;
        }
    }
    let (local, items) = pool.locality();
    p.local_items += local;
    p.items += items;
    let (done, of) = pool.prefetched();
    p.prefetched += done;
    p.prefetch_items += of;
    let (items, ends) = pool.node_ends();
    for d in 0..MAX_NODES {
        p.node_items[d] += items[d];
        p.node_end[d] += ends[d];
    }
    let (slot, wake, tail, runs) = pool.overheads();
    p.run_slot += slot;
    p.run_wake += wake;
    p.run_tail += tail;
    p.runs += runs;
    p.run_last += pool.last_ends();
    let (sort, drain, free) = pool.post_stats();
    p.post_sort += sort;
    p.post_drain += drain;
    p.post_free += free;
}

/// A draft phase from `since` to now as a zone on this lane (`track-zones`); returns now, the next phase's start.
fn draft_zone(_name: &'static str, since: prow::Timestamp) -> prow::Timestamp {
    let now = prow::Timestamp::now();
    prow::zone_at!("track-zones", _name, prow::lane(), since, now);
    now
}

/// The step's MLA projections in host memory as the card wrote them (`Placed::mla_proj`): token `i`'s latent before
/// its norm at `kva + i · KV_LORA_RANK`, its indexer key and gate at `ix + i · ix_rows` (the key, then the gate).
#[derive(Clone, Copy)]
pub struct MlaProj {
    pub kva: *const f32,
    pub ix: *const f32,
    pub ix_rows: usize,
}
unsafe impl Send for MlaProj {}
unsafe impl Sync for MlaProj {}

/// Every token's cache row from the card's projections, then the pool it completes gains its key. On this thread:
/// a token a pool item measured no faster in prefill and cost decode a round trip a layer (2026-10-20).
fn mla_rows(a: &weights::Mla, cache: &mut [u16], pools: &mut [u16], pos: usize, n: usize, p: MlaProj) {
    let (kp, xp, rows) = (p.kva as usize, p.ix as usize, p.ix_rows);
    for i in 0..n {
        let (latent, ikey, igate) = mla::row(cache, pos + i);
        let (mut kva, mut ik) = ([0f32; KV_LORA_RANK], [0f32; INDEX_HEAD_DIM]);
        let src = |at: *const f32, k: usize| unsafe { std::slice::from_raw_parts(at as *const u8, 4 * k) };
        let dst = |v: &mut [f32]| unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, 4 * v.len()) };
        unsafe {
            common::copy::stream_read(dst(&mut kva), src((kp as *const f32).add(i * KV_LORA_RANK), KV_LORA_RANK));
            common::copy::stream_read(dst(&mut ik), src((xp as *const f32).add(i * rows), INDEX_HEAD_DIM));
            common::copy::stream_read(dst(&mut igate[..INDEX_HEAD_DIM]), src((xp as *const f32).add(i * rows + INDEX_HEAD_DIM), INDEX_HEAD_DIM));
        }
        mla::finish_row(a, &kva, &ik, latent, ikey);
        mla::pool_done(a, cache, pools, pos + i);
    }
}


/// `dump-pred`: a decode layer's record appended to /tmp/pred.bin — u16 layer, u16 tokens, then per token its picks at
/// this layer (8 × u16), its quantized FFN input's group scales (H/128 × f32) and values (H × i8), and the next
/// layer's route-ahead logits (E × f32, before the sigmoid).
/// The candidate route-ahead inputs the lab dump records beyond the FFN input (`MORE` a token): layer l + 1's router
/// over layer l's FFN input plus its shared expert's output; plus the resident experts' rows too (what the cards hold
/// before the host's experts are in); plus every routed row (the layer's whole FFN output, after the host's experts:
/// the bound). Normed as the route-ahead is (layer l's post-norm). `[n][MORE][E]` logits.
#[cfg(feature = "dump-pred")]
const MORE: usize = 3;
#[cfg(feature = "dump-pred")]
fn more_ahead(pl: &mut dyn Placed, w: &Weights, l: usize, n: usize, x: &[f32], marked: &[[usize; TOP_K]], wts: &[[f32; TOP_K]]) -> Vec<f32> {
    let Ffn::Moe(m2) = &w.layers[l + 1].ffn else { unreachable!() };
    let (mut shared, mut rows) = (vec![0f32; n * H], vec![0f32; n * TOP_K * H]);
    pl.ffn_parts(l, n, &mut shared, &mut rows);
    let mut out = vec![0f32; n * MORE * E];
    let (mut v, mut vn) = (vec![0f32; H], vec![0f32; H]);
    for i in 0..n {
        for c in 0..MORE {
            for d in 0..H {
                v[d] = x[i * H + d] + shared[i * H + d];
            }
            for k in 0..TOP_K {
                // the dispatch marks a resident pick `E`
                if c == 2 || (c == 1 && marked[i][k] == E) {
                    for d in 0..H {
                        v[d] += wts[i][k] * rows[(i * TOP_K + k) * H + d];
                    }
                }
            }
            common::rmsnorm::rmsnorm_tree::<H>(&v, w.layers[l].post_ln, RMS_NORM_EPS, &mut vn);
            for e in 0..E {
                let r = &m2.router[e * H..(e + 1) * H];
                let mut acc = [0f32; 8];
                for d in (0..H).step_by(8) {
                    for j in 0..8 {
                        acc[j] += widen(r[d + j]) * vn[d + j];
                    }
                }
                out[(i * MORE + c) * E + e] = acc.iter().sum();
            }
        }
    }
    out
}

#[cfg(feature = "dump-pred")]
fn dump_pred(l: usize, n: usize, picks: &[[usize; TOP_K]], xq: &[i8], xs: &[f32], score: &[f32], more: &[f32]) {
    use std::io::Write;
    static OUT: std::sync::Mutex<Option<std::io::BufWriter<std::fs::File>>> = std::sync::Mutex::new(None);
    let mut g = OUT.lock().unwrap();
    if g.is_none() {
        *g = std::fs::File::create("/tmp/pred.bin").ok().map(|f| std::io::BufWriter::with_capacity(1 << 20, f));
    }
    let Some(w) = g.as_mut() else { return };
    const HG: usize = H / common::q4::GROUP;
    let mut b: Vec<u8> = Vec::with_capacity(4 + n * (16 + HG * 4 + H + E * 4));
    b.extend_from_slice(&(l as u16).to_le_bytes());
    b.extend_from_slice(&(n as u16).to_le_bytes());
    for i in 0..n {
        for &e in &picks[i] {
            b.extend_from_slice(&(e as u16).to_le_bytes());
        }
        for &v in &xs[i * HG..(i + 1) * HG] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend(xq[i * H..(i + 1) * H].iter().map(|&v| v as u8));
        for &v in &score[i * ROUTE_ROWS + E..(i + 1) * ROUTE_ROWS] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &more[i * MORE * E..(i + 1) * MORE * E] {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    let _ = w.write_all(&b);
    let _ = w.flush();
}
