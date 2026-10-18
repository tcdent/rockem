//! Construction: a sequence's state per layer, the scratch allocated once,
//! the routed experts from the bundle's stream, the thread pool, and the CPU
//! realization — every op in this process, the reference placement every card
//! kernel is diffed against — with the step that ties the fused kernels
//! together one token at a time. `std` only; nothing on the token path allocates.

use crate::hc::gfx11_consts::ROUTE_ROWS;
use crate::ffn::Routed;
use crate::layout::{self, KDA_MEMORY, KDA_WINDOW, MLA_ROW, POSITION};
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
unsafe fn work(job: *const Job, order: &[usize], node: usize) {
    let j = &*job;
    let f = &*j.f;
    INSIDE.with(|c| c.set(job));
    let (mut mine, mut local) = (0, 0);
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
            }
            f(j.order[i] as usize);
            mine += 1;
            local += (k == 0) as usize;
        }
    }
    if mine > 0 {
        // this thread's last item ends its node's run as far as it is concerned (no cross-node stealing: its items were its node's)
        j.node_end[node].fetch_max(j.t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        j.node_done[node].0.fetch_add(mine, Ordering::Relaxed);
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
        for &core in &cores[1..] {
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
                        unsafe { work(job, &order, node) };
                    }
                    sh.active.fetch_sub(1, Ordering::AcqRel);
                }
            });
        }
        let node = node_of_core(cores[0]);
        let order = slot_order(node, ccd_of(cores[0]), &node_ccds);
        Threads { shared, retired: std::sync::Mutex::new(Vec::new()), order, node_ccds, placed: std::sync::Mutex::new(Vec::new()), locality: std::sync::Mutex::new((0, 0)), prefetched: std::sync::Mutex::new((0, 0)), node, node_stats: std::sync::Mutex::new(([0; MAX_NODES], [0.0; MAX_NODES])), overheads: std::sync::Mutex::new((0.0, 0.0, 0.0, 0)), last_ends: std::sync::Mutex::new(0.0), post_stats: std::sync::Mutex::new((0.0, 0.0, 0.0)) }
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
        let entered = std::time::Instant::now();
        let job = self.post(n, at, node, ccd, slot, f, None);
        unsafe { work(job, &self.order, self.node) };
        while unsafe { (*job).done.0.load(Ordering::Acquire) } != n {
            std::hint::spin_loop();
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
        }
        let fp: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        let light = owned.is_some();
        let job = Box::into_raw(Box::new(Job { f: fp, order, next: std::array::from_fn(|s| Counter(AtomicUsize::new(start[s]))), start, done: Counter(AtomicUsize::new(0)), local: Counter(AtomicUsize::new(0)), stop: std::sync::atomic::AtomicBool::new(false), n, node_done: std::array::from_fn(|_| Counter(AtomicUsize::new(0))), node_end: std::array::from_fn(|_| std::sync::atomic::AtomicU64::new(0)), first: std::sync::atomic::AtomicU64::new(u64::MAX), t0: std::time::Instant::now(), _owned: owned, light }));
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
}
/// An item is the record's (`layout::gu_item`, `layout::down_item`): 32 gate/up items and 20 down items an expert.
use crate::layout::{DOWN_ITEMS, GU_ITEMS};
use crate::STEP;

impl<'a> CpuExperts<'a> {
    pub fn new(stream: &'a expert::Stream<'a>, pool: &'a dyn Pool) -> Self {
        use common::q4::GROUP;
        let mut this = CpuExperts { stream, pool, slice_node: Vec::new(), slice_slot: Vec::new(), xq: vec![0; STEP * H], xs: vec![0.0; STEP * H / GROUP], xg: vec![0; STEP * H / GROUP], h: vec![0.0; STEP * TOP_K * D_FF], hq: vec![0; STEP * TOP_K * D_FF], hs: vec![0.0; STEP * TOP_K * D_FF / GROUP], hg: vec![0; STEP * TOP_K * D_FF / GROUP], elapsed: 0.0 };
        this.relearn();
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
        // the stream lives as long as the process (the node leaks it; the gate's outlives every token):
        // the job that owns these closures is gone within the token
        let stream: &'static expert::Stream<'static> = unsafe { std::mem::transmute::<&expert::Stream, &'static expert::Stream<'static>>(self.stream) };
        let pool = self.pool;
        // an item an (expert, node, CCD): the slices of that expert the compute will slot there, in the
        // compute's order — a slice's node from its rows' page, its CCD from the expert's id and the slice
        let (n_gu, n_down) = (GU_ITEMS, DOWN_ITEMS);
        let idx: Vec<usize> = idx.to_vec();
        let mut items: Vec<(usize, usize, Vec<u32>)> = Vec::new(); // expert position, the pool's queue, slice ids (gate/up first, then down + n_gu)
        let mut group = vec![usize::MAX; idx.len() * SLOTS]; // (k, queue) → its item
        for (k, &e) in idx.iter().enumerate() {
            for j in 0..n_gu + n_down {
                let q = self.slice_slot[((l - FIRST_K_DENSE_REPLACE) * E + e) * (n_gu + n_down) + j] as usize;
                let g = &mut group[k * SLOTS + q];
                if *g == usize::MAX {
                    *g = items.len();
                    items.push((k, q, Vec::with_capacity(16)));
                }
                items[*g].2.push(j as u32);
            }
        }
        let n = items.len();
        let items = std::sync::Arc::new(items);
        let (i1, i3) = (items.clone(), items);
        let slot = move |i: usize| i1[i].1;
        // the touching yields between 4 KB chunks once the job is overtaken: the run that overtakes it
        // waits for the items in flight (the run's slotting time was 13 µs with 64 KB chunks, 2026-10-16)
        let pool_s: &'static dyn Pool = unsafe { std::mem::transmute::<&dyn Pool, &'static dyn Pool>(pool) };
        let f = move |i: usize| {
            let it = &i3[i];
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
            for &j in &it.2 {
                let j = j as usize;
                if !touch_yielding(if j < n_gu { w.gu_bytes(j) } else { w.down_bytes(j - n_gu) }) {
                    return;
                }
            }
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
        for i in 0..n {
            QAct { q: &mut self.xq[i * H..(i + 1) * H], scale: &mut self.xs[i * HG..(i + 1) * HG], gsum: &mut self.xg[i * HG..(i + 1) * HG] }.quantize(&x[i * H..(i + 1) * H]);
        }
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
        let experts: Vec<usize> = (0..E).filter(|&e| first[e + 1] > first[e]).collect();
        let ne = experts.len();
        let uses = |e: usize| &pairs[first[e]..first[e + 1]];
        let stream = self.stream;
        let slice_slot = &self.slice_slot;
        let slot_of = |e: usize, j: usize| slice_slot[((l - FIRST_K_DENSE_REPLACE) * E + e) * (GU_ITEMS + DOWN_ITEMS) + j] as usize;
        let (xq, xs, xg) = (&self.xq, &self.xs, &self.xg);
        let h_p = self.h.as_mut_ptr() as usize;
        // the gate/up rows: part (j, c) owns expert experts[j]'s hidden channels c·GU_ROWS.., for all of
        // its tokens at once (`q4::gemm`: the part's bytes read once, every token's dot on them), in
        // blocks of GEMM_TOKENS
        let uses_r = &uses;
        let experts_r = &experts;
        let gate_slot = move |i: usize| slot_of(experts_r[i / GU_PARTS], i % GU_PARTS / SUB);
        self.pool.run_slotted(ne * GU_PARTS, &gate_slot, &move |i| {
            let (j, c) = (i / GU_PARTS, i % GU_PARTS);
            let e = experts_r[j];
            let w = expert::Weights::at(stream, l, e);
            let (gw, uw, gs, usc) = w.gu(c);
            let pairs = uses_r(e);
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
        });
        // every (token, slot)'s hidden state quantized once: here for a token (2 µs each, less than a
        // pool run's posting), over the pool for a step
        let (hq_p, hs_p, hg_p) = (self.hq.as_mut_ptr() as usize, self.hs.as_mut_ptr() as usize, self.hg.as_mut_ptr() as usize);
        let h = &self.h;
        let quantize_h = move |i: usize| {
            let (q, sc, g) = unsafe { (std::slice::from_raw_parts_mut((hq_p as *mut i8).add(i * D_FF), D_FF), std::slice::from_raw_parts_mut((hs_p as *mut f32).add(i * DG), DG), std::slice::from_raw_parts_mut((hg_p as *mut i32).add(i * DG), DG)) };
            QAct { q, scale: sc, gsum: g }.quantize(&h[i * D_FF..(i + 1) * D_FF]);
        };
        if n * TOP_K <= 16 {
            for i in 0..n * TOP_K {
                quantize_h(i);
            }
        } else {
            self.pool.run(n * TOP_K, &quantize_h);
        }
        // the down rows: part (j, c) owns rows c·DOWN_ROWS.. of expert experts[j]'s output, for all of its tokens, likewise
        let (hq, hs, hg) = (&self.hq, &self.hs, &self.hg);
        let out_p = out.as_mut_ptr() as usize;
        let down_slot = move |i: usize| slot_of(experts_r[i / DOWN_PARTS], GU_ITEMS + i % DOWN_PARTS / SUB);
        self.pool.run_slotted(ne * DOWN_PARTS, &down_slot, &move |i| {
            let (j, c) = (i / DOWN_PARTS, i % DOWN_PARTS);
            let e = experts_r[j];
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
        });
        self.elapsed += t0.elapsed().as_secs_f64();
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
/// the routed experts the host computes (`ffn_routed`) into the streams with
/// the shared expert's rows; `logits` is the head over the streams' mean.
/// Nothing is waited for but what the host reads. `seed` and `snapshot` move
/// a KDA layer's state; `mla_seed` installs a latent; `layer_out` fetches a
/// layer's output off the streams for bring-up.
pub trait Placed {
    /// The step's `n` tokens' embeddings (`[n][H]`) into every card's streams; the handoffs restart.
    fn begin(&mut self, rows: &[f32], n: usize);
    /// A KDA layer's block over the step; a step of a verify's size (`2..=SPEC_TOKENS` tokens) keeps
    /// the state after each token for `kda_rewind`.
    fn attention(&mut self, l: usize, n: usize);
    /// A KDA layer's state back to what it was after token `slot` of the last verify-sized step.
    fn kda_rewind(&mut self, l: usize, slot: usize);
    fn mla_post(&mut self, l: usize, n: usize);
    /// Token `i`'s projections of the step (the launch waited for at `i == 0`): the latent before its
    /// norm, the indexer's key before its norm, the position's gate.
    fn mla_wait(&mut self, l: usize, i: usize, kva: &mut [f32], ik: &mut [f32], igate: &mut [f32]);
    /// The pool keys `from..to` into the card's mirror (every pool completed, by a step or a seed, goes
    /// there before any token scores among them).
    fn ix_pools(&mut self, l: usize, pools: &[f32], from: usize, to: usize);
    /// Every token's selection on the card (`mla::score_pools` and `mla::pick`'s results, from the
    /// mirrored pools) into the attend's lists; the first token's count back when asked (bring-up).
    fn ix_select(&mut self, l: usize, pos: usize, n: usize, count: Option<&mut usize>);
    /// The step's `n` tokens (at `pos..`) attend over their selections (the card's, `ix_select`), one
    /// launch; their latents (`[n][KV_LORA_RANK]`) into the mirror first.
    fn mla_attend(&mut self, l: usize, n: usize, pos: usize, latents: &[f32]);
    /// The output projection over the step's heads, into the streams.
    fn mla_out(&mut self, l: usize, n: usize);
    /// The draft layer (`layout::MTP_LAYER`): its input `[2·H]` (the next token's normed embedding, the
    /// normed hidden state) through `eh_proj` and the attention's projections (then `mla_wait`,
    /// `ix_pools`, `ix_select`, `mla_attend` as for any MLA layer);
    /// A MoE layer's routed experts resident on the cards: every pick `(token i, slot k)` whose expert is
    /// resident becomes a job on its card over the token's quantized activation, launched at once, and is
    /// marked `E` in `picks` so the host's run skips it; a pick not resident may claim a slot (a free one,
    /// or a less-used resident's past a margin), which the loader fills for the steps to come. Returns the jobs.
    fn xpt_dispatch(&mut self, l: usize, n: usize, picks: &mut [[usize; TOP_K]], xq: &[i8], xs: &[f32], xg: &[i32]) -> usize;
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
    /// Whether the bring-up checks run this boot (the cards instrumented).
    fn checks(&self) -> bool;
    fn mtp_begin(&mut self, cat: &[f32], n: usize);
    /// its output projection and residual, the blocks' states out (`[n][H]`) for the host's routing;
    fn mtp_attn_out(&mut self, hidden: &mut [f32], n: usize);
    /// the shared expert, the host's routed experts' sums `routed` (`[n][H]`) and the residual, the states out;
    fn mtp_ffn(&mut self, routed: &[f32], hidden: &mut [f32], n: usize);
    /// the shared head over token `i`'s state: its draft's logits.
    fn mtp_head(&mut self, i: usize, logits: &mut [f32]);
    /// The last step's tokens' hidden states (the streams' means before the final norm), `[n][H]`.
    fn hidden_all(&mut self, n: usize, out: &mut [f32]);
    fn mla_seed(&mut self, l: usize, pos: usize, latent: &[f32]);
    fn ffn(&mut self, l: usize, n: usize);
    /// A MoE layer's FFN input for every token (`[n][H]`) and the FFN router's logits (`[n][E]`).
    fn ffn_input(&mut self, l: usize, n: usize, x: &mut [f32], logits: &mut [f32]);
    /// The routed experts' output for every token (`[n][H]`), into the streams.
    fn ffn_routed(&mut self, l: usize, n: usize, y: &[f32]);
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
    Mla { cache: Vec<f32>, pools: Vec<f32> },
}

/// A token's time by phase, summed over steps until `take`n (bring-up and the node's
/// `rockem.profile` line). The experts are the routed realization's own count.
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
    /// Over the MoE layers: how many of a layer's eight experts were also the previous token's at that layer
    /// (the ceiling of a same-layer prefetch), and the layers counted.
    pub reuse: usize,
    pub moe_layers: usize,
    /// Expert items (gate/up and down) that ran on their preferred CCD, and all of them.
    pub local_items: usize,
    pub items: usize,
    /// How many of a layer's picks the router already names on the attention block's input (the prefetch's predictor).
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
impl Profile {
    /// The expert runs by node: each node's share of the items, when its last item finished (ms a
    /// token, over the gate/up and down runs of every MoE layer), and its rate in items a ms — the
    /// shares a placement should give the nodes are in proportion to these rates.
    fn nodes(&self) -> String {
        let used: Vec<usize> = (0..MAX_NODES).filter(|&d| self.node_items[d] > 0).collect();
        if used.len() < 2 {
            return String::new();
        }
        let n = self.tokens.max(1) as f64;
        let total: usize = used.iter().map(|&d| self.node_items[d]).sum();
        let share: Vec<String> = used.iter().map(|&d| format!("{:.1}", 100.0 * self.node_items[d] as f64 / total as f64)).collect();
        let end: Vec<String> = used.iter().map(|&d| format!("{:.2}", 1e3 * self.node_end[d] / n)).collect();
        let rate: Vec<String> = used.iter().map(|&d| format!("{:.1}", self.node_items[d] as f64 / (1e3 * self.node_end[d]).max(1e-9))).collect();
        format!("\nprofile  : nodes — items {} %, run ends {} ms a token, {} items/ms; a token's {:.1} runs end at {:.2} ms (the last node each run) and spend {:.2} ms slotting (sort {:.2}, drain {:.2}, alloc+free {:.2}), {:.2} waking, {:.2} in the tail", share.join(" / "), end.join(" / "), rate.join(" / "), self.runs as f64 / n, 1e3 * self.run_last / n, 1e3 * self.run_slot / n, 1e3 * self.post_sort / n, 1e3 * self.post_drain / n, 1e3 * self.post_free / n, 1e3 * self.run_wake / n, 1e3 * self.run_tail / n)
    }
    /// One line of milliseconds a token, the phases in order and their sum.
    pub fn line(&self) -> String {
        let n = self.tokens.max(1) as f64 / 1e3;
        let mla = self.mla_proj + self.mla_attn + self.mla_out;
        let rs = self.route + self.shared;
        let sum = self.hc + self.slot + self.kda_card + self.kda_cpu + mla + self.dense_ffn + rs + self.experts + self.head + self.draft;
        format!("{} tokens: hc {:.1}  slot {:.1}  kda card {:.1}  kda cpu {:.1}  mla {:.1} (proj {:.1} attn {:.1} out {:.1})  dense ffn {:.1}  router+shared {:.1} (route {:.1} shared {:.1})  experts {:.1} (reused {:.2} of {} from the previous token, {:.2} named on the attention input; {:.0}% of the predicted slices pulled; {:.0}% of items on their CCD)  head {:.1}  draft {:.1}  = {:.1} ms a token{}", self.tokens, self.hc / n, self.slot / n, self.kda_card / n, self.kda_cpu / n, mla / n, self.mla_proj / n, self.mla_attn / n, self.mla_out / n, self.dense_ffn / n, rs / n, self.route / n, self.shared / n, self.experts / n, self.reuse as f64 / self.moe_layers.max(1) as f64, TOP_K, self.pre_agree as f64 / self.moe_layers.max(1) as f64, 100.0 * self.prefetched as f64 / self.prefetch_items.max(1) as f64, 100.0 * self.local_items as f64 / self.items.max(1) as f64, self.head / n, self.draft / n, sum / n, self.nodes())
    }
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
    b_lat: Vec<f32>,
    m_heads: Vec<f32>,
    m_hs: Vec<f32>,
    f_gu: Vec<f32>,
    f_h: Vec<f32>,
    f_expert_out: Vec<f32>,
    f_shared: Vec<f32>,
    f_score: Vec<f32>,
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
            .map(|l| if LINEAR_ATTENTION[l] { LayerState::Kda { window: vec![0.0; KDA_WINDOW], mem: vec![0.0; KDA_MEMORY] } } else { LayerState::Mla { cache: vec![0.0; max_ctx * MLA_ROW], pools: vec![0.0; max_ctx.div_ceil(INDEX_KPOOL) * INDEX_HEAD_DIM] } })
            .collect();
        CpuModel {
            routed,
            pool,
            profile: Profile::default(),
            placed: None,
            last_picks: vec![[E; TOP_K]; NUM_HIDDEN_LAYERS],
            freq: vec![0; NUM_HIDDEN_LAYERS * E],
            max_ctx,
            mtp: (NUM_NEXTN == 1).then(|| LayerState::Mla { cache: vec![0.0; max_ctx * MLA_ROW], pools: vec![0.0; max_ctx.div_ceil(INDEX_KPOOL) * INDEX_HEAD_DIM] }),
            prefix_hidden: vec![0.0; FIRST_POS * H],
            d_cat: vec![0.0; 2 * H],
            d_cats: vec![0.0; STEP * 2 * H],
            d_hid: vec![0.0; H],
            b_xq: vec![0; SPEC_TOKENS * H],
            b_xs: vec![0.0; SPEC_TOKENS * (H / common::q4::GROUP)],
            b_xg: vec![0; SPEC_TOKENS * (H / common::q4::GROUP)],
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
            b_lat: vec![0.0; STEP * KV_LORA_RANK],
            m_heads: vec![0.0; MLA_HEADS * V_HEAD_DIM],
            m_hs: vec![0.0; MLA_HEADS * mla::HEAD_SCRATCH],
            f_gu: vec![0.0; 2 * INTERMEDIATE_SIZE],
            f_h: vec![0.0; INTERMEDIATE_SIZE],
            f_expert_out: vec![0.0; TOP_K * H],
            f_shared: vec![0.0; H],
            f_score: vec![0.0; E],
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
        assert_eq!(rows.len() % POSITION, 0);
        assert_eq!(snap.len(), layout::SNAPSHOT);
        let n = rows.len() / POSITION;
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
                        cache[p * MLA_ROW..(p + 1) * MLA_ROW].copy_from_slice(&rows[p * POSITION + mi * MLA_ROW..p * POSITION + (mi + 1) * MLA_ROW]);
                        // the bundle's rows are f32: the latent and the key at the stored precision (`finish_row`)
                        for v in &mut cache[p * MLA_ROW..p * MLA_ROW + KV_LORA_RANK + INDEX_HEAD_DIM] {
                            *v = common::math::bf16_round(*v);
                        }
                    }
                    for j in 0..n / INDEX_KPOOL {
                        let (c, pk) = (&*cache, &mut pools[j * INDEX_HEAD_DIM..(j + 1) * INDEX_HEAD_DIM]);
                        mla::pool_key(mw, c, j, pk);
                    }
                    if let Some(pl) = self.placed.as_mut() {
                        for p in 0..n {
                            pl.mla_seed(l, p, &cache[p * MLA_ROW..p * MLA_ROW + KV_LORA_RANK]);
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
        self.seed(w, rows, snap)
    }
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
        let t0 = std::time::Instant::now();
        let m = w.mtp.as_ref().expect("the weights carry no draft layer");
        let LayerState::Mla { cache, pools } = s.mtp.as_mut().expect("no draft state") else { unreachable!() };
        let pl = s.placed.as_mut().expect("draft_batch: the cards'");
        let l = layout::MTP_LAYER;
        assert!(n >= 1 && n <= STEP && nexts.len() >= n && hiddens.len() >= n * H);
        // every position's [enorm(embed(next)) | hnorm(hidden)]
        for i in 0..n {
            let row = &w.embed[nexts[i] as usize * H..(nexts[i] as usize + 1) * H];
            for d in 0..H {
                s.d_y[d] = widen(row[d]);
            }
            let (e, h) = s.d_cats[i * 2 * H..(i + 1) * 2 * H].split_at_mut(H);
            rmsnorm::<H>(&s.d_y, m.enorm, RMS_NORM_EPS, e);
            rmsnorm::<H>(&hiddens[i * H..(i + 1) * H], m.hnorm, RMS_NORM_EPS, h);
        }
        pl.mtp_begin(&s.d_cats[..n * 2 * H], n);
        for i in 0..n {
            let (latent, ikey, igate) = mla::row(cache, pos + i);
            pl.mla_wait(l, i, &mut s.m_kva, &mut s.m_ik, igate);
            mla::finish_row(&m.attn, &s.m_kva, &s.m_ik, latent, ikey);
            mla::pool_done(&m.attn, cache, pools, pos + i);
        }
        pl.ix_pools(l, pools, pos / INDEX_KPOOL, (pos + n) / INDEX_KPOOL);
        pl.ix_select(l, pos, n, None);
        for i in 0..n {
            s.b_lat[i * KV_LORA_RANK..(i + 1) * KV_LORA_RANK].copy_from_slice(&cache[(pos + i) * MLA_ROW..(pos + i) * MLA_ROW + KV_LORA_RANK]);
        }
        pl.mla_attend(l, n, pos, &s.b_lat[..n * KV_LORA_RANK]);
        pl.mtp_attn_out(&mut s.b_x[..n * H], n);
        tr.edge(l, "mtp.attn", &s.b_x[(n - 1) * H..n * H]);
        // the routing and the routed experts here, every token's
        let mut picks = [[0usize; TOP_K]; STEP];
        let mut wts = [[0f32; TOP_K]; STEP];
        for i in 0..n {
            rmsnorm::<H>(&s.b_x[i * H..(i + 1) * H], m.post_ln, RMS_NORM_EPS, &mut s.b_xn[i * H..(i + 1) * H]);
            gemv::bf16_par::<E, H>(m.ffn.router, &s.b_xn[i * H..(i + 1) * H], &mut s.f_score, s.pool);
            let (idx, wt) = ffn::route_scores(&m.ffn, &mut s.f_score, None);
            picks[i] = idx;
            wts[i] = wt;
        }
        s.routed.run_batch(l, &picks[..n], &s.b_xn[..n * H], &mut s.b_out[..n * TOP_K * H]);
        for i in 0..n {
            let y = &mut s.b_y[i * H..(i + 1) * H];
            y.fill(0.0);
            for k in 0..TOP_K {
                let eo = &s.b_out[(i * TOP_K + k) * H..(i * TOP_K + k + 1) * H];
                for d in 0..H {
                    y[d] += wts[i][k] * eo[d];
                }
            }
        }
        tr.edge(l, "routed", &s.b_y[(n - 1) * H..n * H]);
        pl.mtp_ffn(&s.b_y[..n * H], &mut s.b_x[..n * H], n);
        tr.edge(l, "mtp.ffn", &s.b_x[(n - 1) * H..n * H]);
        s.d_hid.copy_from_slice(&s.b_x[(n - 1) * H..n * H]);
        pl.mtp_head(n - 1, logits);
        // the draft's expert time is the draft's (the step's `experts` lap takes what `elapsed` holds)
        s.routed.elapsed();
        s.profile.draft += t0.elapsed().as_secs_f64();
    }
    /// The next depth of drafting: the draft layer at `pos` (the position after the last drafted) from its
    /// own output there and `next` (the token the last draft proposed, standing for an accepted draft);
    /// `logits` is the draft of the token after `next`.
    pub fn draft_next(&mut self, w: &Weights, next: u32, pos: usize, logits: &mut [f32], tr: &mut impl Trace) {
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
    /// Position `pos`'s rows, `[POSITION]`.
    pub fn rows(&self, pos: usize, out: &mut [f32]) {
        let mut mi = 0;
        for st in &self.layers {
            if let LayerState::Mla { cache, .. } = st {
                out[mi * MLA_ROW..(mi + 1) * MLA_ROW].copy_from_slice(&cache[pos * MLA_ROW..(pos + 1) * MLA_ROW]);
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
        let s = self;
        let row = &w.embed[tok as usize * H..(tok as usize + 1) * H];
        for j in 0..HC {
            for d in 0..H {
                s.streams[j * H + d] = widen(row[d]);
            }
        }
        tr.edge(NONE, "embed", &s.streams[..H]);
        let mut t = std::time::Instant::now();
        let mut lap = |acc: &mut f64| {
            let now = std::time::Instant::now();
            *acc += (now - t).as_secs_f64();
            t = now;
        };
        let p = &mut s.profile;
        p.tokens += 1;
        for l in 0..NUM_HIDDEN_LAYERS {
            let lw = &w.layers[l];
            let mut hcs = hc::Scratch { flat: &mut s.hc_flat, coef: &mut s.hc_coef };
            let mix = hc::pre(&lw.hc_attn, &s.streams, &mut hcs, s.pool, &mut s.x);
            rmsnorm::<H>(&s.x, &lw.input_ln, RMS_NORM_EPS, &mut s.xn);
            tr.edge(l, "attn_in", &s.xn);
            lap(&mut p.hc);
            match (&lw.attn, &mut s.layers[l]) {
                (Attn::Kda(a), LayerState::Kda { window, mem }) => {
                    let mut ks = kda::Scratch { mixed: &mut s.k_mixed, conv: &mut s.k_conv, fa: &mut s.k_fa, f: &mut s.k_f, beta: &mut s.k_beta, ga: &mut s.k_ga, gate: &mut s.k_gate, normed: &mut s.k_normed, heads: &mut s.k_heads };
                    kda::step(l, a, &s.xn, window, mem, &mut ks, s.pool, tr, &mut s.y);
                    lap(&mut p.kda_cpu);
                }
                (Attn::Mla(a), LayerState::Mla { cache, pools }) => {
                    let mut ms = mla::Scratch { qr: &mut s.m_qr, qa: &mut s.m_qa, q: &mut s.m_q, kva: &mut s.m_kva, ik: &mut s.m_ik, iq: &mut s.m_iq, iw: &mut s.m_iw, pscore: &mut s.m_pscore, pidx: &mut s.m_pidx, sel: &mut s.m_sel, heads: &mut s.m_heads, hs: &mut s.m_hs };
                    {
                        let (latent, ikey, igate) = mla::row(cache, pos);
                        mla::project(a, &s.xn, &mut ms, igate, s.pool);
                        mla::finish_row(a, ms.kva, ms.ik, latent, ikey);
                    }
                    lap(&mut p.mla_proj);
                    let n = mla::select_at(l, a, cache, pools, pos, &mut ms, tr);
                    mla::heads(l, a, cache, &mut ms, n, s.pool, tr);
                    lap(&mut p.mla_attn);
                    mla::output(a, ms.heads, &mut s.y, s.pool);
                    lap(&mut p.mla_out);
                }
                _ => unreachable!("a layer's state and weights disagree"),
            }
            tr.edge(l, "attn_out", &s.y);
            hc::post(&mix, &s.y, &mut s.streams);

            let mut hcs = hc::Scratch { flat: &mut s.hc_flat, coef: &mut s.hc_coef };
            let mix = hc::pre(&lw.hc_ffn, &s.streams, &mut hcs, s.pool, &mut s.x);
            rmsnorm::<H>(&s.x, &lw.post_ln, RMS_NORM_EPS, &mut s.xn);
            tr.edge(l, "ffn_in", &s.xn);
            lap(&mut p.hc);
            let mut fs = ffn::Scratch { gu: &mut s.f_gu, h: &mut s.f_h, expert_out: &mut s.f_expert_out, shared: &mut s.f_shared, score: &mut s.f_score };
            match &lw.ffn {
                Ffn::Dense(d) => {
                    ffn::dense(l, d, &s.xn, &mut fs, s.pool, tr, &mut s.y);
                    lap(&mut p.dense_ffn);
                }
                Ffn::Moe(m) => {
                    let picks = ffn::moe_routed(l, m, &s.xn, &mut fs, s.pool, &mut s.routed, forced.map(|f| &f[l - FIRST_K_DENSE_REPLACE]), tr, &mut s.y);
                    account(p, &mut s.last_picks, &mut s.freq, s.pool, l, picks, [E; TOP_K]);
                    let e = s.routed.elapsed();
                    p.experts += e;
                    lap(&mut p.route);
                    p.route -= e;
                    ffn::moe_shared(m, &s.xn, &mut fs, s.pool);
                    tr.edge(l, "shared", fs.shared);
                    for i in 0..H {
                        s.y[i] += fs.shared[i];
                    }
                    lap(&mut p.shared);
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
            lap(&mut p.hc);
        }
        if logits {
            rmsnorm::<H>(&s.mean, &w.norm, RMS_NORM_EPS, &mut s.xn);
            gemv::bf16_par::<VOCAB_SIZE, H>(w.lm_head, &s.xn, &mut s.logits[..VOCAB_SIZE], s.pool);
            tr.edge(NONE, "logits", &s.logits[..VOCAB_SIZE]);
        }
        lap(&mut p.head);
    }

    /// A step of `toks` (at `pos..pos + n`, `n ≤ STEP`) on the cards: the host posts each layer's
    /// halves over the step and works only where the routed experts are its own — it waits for
    /// the FFN half's inputs, runs the step's experts grouped by expert, hands their outputs back —
    /// and reads the last token's logits at the end when asked. One token is a step of one.
    /// Profile: `hc` is the posting, `kda_card` the wait for the cards (the attention block and
    /// the hyper-connections), `route` the host's router and sum around the experts, the MLA laps
    /// the host's selection, `head` the wait for the logits.
    pub fn step_batch(&mut self, w: &Weights, toks: &[u32], pos: usize, forced: Option<&[[usize; TOP_K]]>, tr: &mut impl Trace, logits: Logits) {
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
        let mut lap = |acc: &mut f64| {
            let now = std::time::Instant::now();
            *acc += (now - t).as_secs_f64();
            t = now;
        };
        let p = &mut s.profile;
        p.tokens += n;
        let mut pre_next = [[E; TOP_K]; SPEC_TOKENS];
        for l in 0..NUM_HIDDEN_LAYERS {
            let lw = &w.layers[l];
            let moe = match &lw.ffn {
                Ffn::Moe(m) => Some(m),
                Ffn::Dense(_) => None,
            };
            let is_mla = matches!(lw.attn, Attn::Mla(_));
            if is_mla {
                pl.mla_post(l, n);
            } else {
                pl.attention(l, n);
            }
            lap(&mut p.hc);
            // the prefetch's predictor for this layer: the layer before named its picks over its own FFN
            // input (the next layer's router rows ride in that router launch); `pre_picks` scores it
            let pre_picks = std::mem::replace(&mut pre_next, [[E; TOP_K]; SPEC_TOKENS]);
            if let (Attn::Mla(a), LayerState::Mla { cache, pools }) = (&lw.attn, &mut s.layers[l]) {
                // every token's projections off the card into its cache row, its indexer query and head
                // weights kept; the pools completed along the way gain their keys
                for i in 0..n {
                    let (latent, ikey, igate) = mla::row(cache, pos + i);
                    pl.mla_wait(l, i, &mut s.m_kva, &mut s.m_ik, igate);
                    mla::finish_row(a, &s.m_kva, &s.m_ik, latent, ikey);
                    mla::pool_done(a, cache, pools, pos + i);
                }
                lap(&mut p.mla_proj);
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
                lap(&mut p.mla_attn);
                // the step's latents gathered off the cache rows; every token's attention in one launch
                for i in 0..n {
                    s.b_lat[i * KV_LORA_RANK..(i + 1) * KV_LORA_RANK].copy_from_slice(&cache[(pos + i) * MLA_ROW..(pos + i) * MLA_ROW + KV_LORA_RANK]);
                }
                pl.mla_attend(l, n, pos, &s.b_lat[..n * KV_LORA_RANK]);
                pl.mla_out(l, n);
                lap(&mut p.mla_out);
            }
            pl.ffn(l, n);
            if let Some(m) = moe {
                pl.ffn_input(l, n, &mut s.b_x, &mut s.b_score);
                lap(&mut p.kda_card);
                // every token's norm, picks and weights from the card's router logits, a token an item over
                // the pool (one token runs inline); the step's experts as one grouped run
                let mut picks = [[0usize; TOP_K]; STEP];
                let mut wts = [[0f32; TOP_K]; STEP];
                {
                    let (xn_p, sc_p, pk_p, wt_p) = (s.b_xn.as_mut_ptr() as usize, s.b_score.as_mut_ptr() as usize, picks.as_mut_ptr() as usize, wts.as_mut_ptr() as usize);
                    let (x, post_ln, forced_l) = (&s.b_x, lw.post_ln, forced.map(|f| &f[l - FIRST_K_DENSE_REPLACE]));
                    s.pool.run(n, &move |i| unsafe {
                        let xn = std::slice::from_raw_parts_mut((xn_p as *mut f32).add(i * H), H);
                        rmsnorm::<H>(&x[i * H..(i + 1) * H], post_ln, RMS_NORM_EPS, xn);
                        let score = std::slice::from_raw_parts_mut((sc_p as *mut f32).add(i * ROUTE_ROWS), E);
                        let (idx, wt) = ffn::route_scores(m, score, forced_l);
                        *(pk_p as *mut [usize; TOP_K]).add(i) = idx;
                        *(wt_p as *mut [f32; TOP_K]).add(i) = wt;
                    });
                }
                if n == 1 {
                    tr.edge(l, "ffn_in", &s.b_xn[..H]);
                    tr.topk(l, &picks[0], &wts[0]);
                }
                // a step of a few tokens: the picks resident on the cards run there, concurrently with the host's;
                // the picks as routed are kept for the accounting (the dispatch marks the resident ones)
                let mut routed_picks = [[E; TOP_K]; SPEC_TOKENS];
                if n <= SPEC_TOKENS {
                    routed_picks[..n].copy_from_slice(&picks[..n]);
                }
                let jobs = if n <= SPEC_TOKENS {
                    const HG: usize = H / common::q4::GROUP;
                    for i in 0..n {
                        common::q4::QAct { q: &mut s.b_xq[i * H..(i + 1) * H], scale: &mut s.b_xs[i * HG..(i + 1) * HG], gsum: &mut s.b_xg[i * HG..(i + 1) * HG] }.quantize(&s.b_xn[i * H..(i + 1) * H]);
                    }
                    pl.xpt_dispatch(l, n, &mut picks[..n], &s.b_xq[..n * H], &s.b_xs[..n * HG], &s.b_xg[..n * HG])
                } else {
                    0
                };
                pl.ffn_post(l, n);
                s.routed.run_batch(l, &picks[..n], &s.b_xn[..n * H], &mut s.b_out[..n * TOP_K * H]);
                if jobs > 0 {
                    pl.xpt_join(l, &mut s.b_out[..n * TOP_K * H]);
                    // bring-up (a traced run): every resident pick's output against the host's on the same bytes
                    if tr.wants_edges() || pl.checks() {
                        let mut host = vec![0f32; TOP_K * H];
                        for i in 0..n {
                            for k in 0..TOP_K {
                                if picks[i][k] == E {
                                    let e = routed_picks[i][k];
                                    s.routed.run(l, &[e; TOP_K], &s.b_xn[i * H..(i + 1) * H], &mut host);
                                    let card = &s.b_out[(i * TOP_K + k) * H..(i * TOP_K + k + 1) * H];
                                    if let Some(d) = (0..H).find(|&d| card[d].to_bits() != host[d].to_bits()) {
                                        let worst = (0..H).map(|d| (card[d] - host[d]).abs()).fold(0f32, f32::max);
                                        eprintln!("xpt mismatch: layer {l} expert {e} token {i} pick {k}: first at {d} card {:e} host {:e}, max |Δ| {worst:e}", card[d], host[d]);
                                    }
                                }
                            }
                        }
                    }
                }
                // each token's weighted sum of its experts' outputs, likewise
                {
                    let (y_p, out, wts_r) = (s.b_y.as_mut_ptr() as usize, &s.b_out, &wts);
                    s.pool.run(n, &move |i| {
                        let y = unsafe { std::slice::from_raw_parts_mut((y_p as *mut f32).add(i * H), H) };
                        y.fill(0.0);
                        for k in 0..TOP_K {
                            let eo = &out[(i * TOP_K + k) * H..(i * TOP_K + k + 1) * H];
                            for d in 0..H {
                                y[d] += wts_r[i][k] * eo[d];
                            }
                        }
                    });
                }
                if n == 1 {
                    tr.edge(l, "ffn_in", &s.b_xn[..H]);
                    for k in 0..TOP_K {
                        tr.routed(l, k, &s.b_out[k * H..(k + 1) * H]);
                    }
                    tr.edge(l, "routed", &s.b_y[..H]);
                }
                pl.ffn_routed(l, n, &s.b_y[..n * H]);
                for i in 0..n {
                    let as_routed = if n <= SPEC_TOKENS { routed_picks[i] } else { picks[i] };
                    account(p, &mut s.last_picks, &mut s.freq, s.pool, l, as_routed, if n <= SPEC_TOKENS { pre_picks[i] } else { [E; TOP_K] });
                }
                let e = s.routed.elapsed();
                p.experts += e;
                lap(&mut p.route);
                p.route -= e;
                // the next MoE layer's picks, predicted by its router over this layer's FFN input (the second
                // half of each token's scores), prefetched now that this layer's run is done (a run abandons
                // the prefetches in flight); a step of many tokens names most experts and runs none
                if let (true, Some(Ffn::Moe(m2))) = (n <= SPEC_TOKENS, w.layers.get(l + 1).map(|lw| &lw.ffn)) {
                    let mut union = [E; SPEC_TOKENS * TOP_K];
                    let mut nu = 0;
                    for i in 0..n {
                        pre_next[i] = ffn::picks(m2, &mut s.b_score[i * ROUTE_ROWS + E..(i + 1) * ROUTE_ROWS]);
                        for &e in &pre_next[i] {
                            if !union[..nu].contains(&e) && !pl.resident(l + 1, e) {
                                union[nu] = e;
                                nu += 1;
                            }
                        }
                    }
                    s.routed.prefetch(l + 1, &union[..nu]);
                    lap(&mut p.slot);
                }
            }
            if tr.wants_edges() && n == 1 {
                pl.layer_out(l, 0, &mut s.mean);
                tr.edge(l, "out", &s.mean);
            }
        }
        match logits {
            Logits::None => pl.finish(),
            Logits::Last => pl.logits(n - 1, 1, &mut s.logits[(n - 1) * VOCAB_SIZE..n * VOCAB_SIZE]),
            Logits::All => pl.logits(0, n, &mut s.logits[..n * VOCAB_SIZE]),
        }
        if n == 1 && !matches!(logits, Logits::None) {
            tr.edge(NONE, "logits", &s.logits[..VOCAB_SIZE]);
        }
        lap(&mut p.head);
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
fn account(p: &mut Profile, last_picks: &mut [[usize; TOP_K]], freq: &mut [u32], pool: &dyn Pool, l: usize, picks: [usize; TOP_K], pre_picks: [usize; TOP_K]) {
    p.reuse += picks.iter().filter(|e| last_picks[l].contains(e)).count();
    p.pre_agree += picks.iter().filter(|e| pre_picks.contains(e)).count();
    p.moe_layers += 1;
    last_picks[l] = picks;
    for &e in &picks {
        freq[l * E + e] += 1;
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
