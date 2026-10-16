//! Construction: a sequence's state per layer, the scratch allocated once,
//! the routed experts from the bundle's stream, the thread pool, and the CPU
//! realization — every op in this process, the reference placement every card
//! kernel is diffed against — with the step that ties the fused kernels
//! together one token at a time. `std` only; nothing on the token path allocates.

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
    /// The job before the current one, freed at the next publish.
    retired: std::sync::atomic::AtomicPtr<Job>,
    /// The slot order the calling thread drains (`slot_order`).
    order: Vec<usize>,
    /// The CCDs of each node, as global CCD indices.
    node_ccds: Vec<Vec<usize>>,
    /// The placed regions: base, length, the node of each `PAGE` of it.
    placed: std::sync::Mutex<Vec<(usize, usize, Vec<u8>)>>,
    /// The last run's (local items, items) and the last prefetch's (done, items), for the profile.
    locality: std::sync::Mutex<(usize, usize)>,
    prefetched: std::sync::Mutex<(usize, usize)>,
    /// Called by the calling thread between its own items of a run while set (`tick`): the
    /// card's heartbeat during the expert phase.
    tick: std::sync::Mutex<Option<Tick>>,
}
#[derive(Clone, Copy)]
struct Tick(*const dyn Fn());
unsafe impl Send for Tick {}
/// The grain of placement: a region's node is known per 2 MiB page.
pub const PAGE: usize = 2 << 20;
const MAX_NODES: usize = 8;
const MAX_CCDS: usize = 16;
/// A job's items sit in slots: slot `c < MAX_CCDS` is CCD `c`'s (an item with a
/// preferred core complex), slot `MAX_CCDS + d` is node `d`'s (an item placed by its
/// rows' node only, or every item when nothing is placed).
const SLOTS: usize = MAX_CCDS + MAX_NODES;
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
    /// A prefetch's closure, owned by its job (the runs' closures are the caller's, alive through the run).
    _owned: Option<Box<dyn Fn(usize) + Send + Sync>>,
}
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
fn pin(core: usize) {
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

/// The slots a thread on CCD `ccd` of node `node` drains, in order: its own CCD's, its node's,
/// its node's other CCDs', then the other nodes' (their node slot, then their CCDs').
fn slot_order(node: usize, ccd: usize, node_ccds: &[Vec<usize>]) -> Vec<usize> {
    let mut o = vec![ccd, MAX_CCDS + node];
    o.extend(node_ccds[node].iter().filter(|&&c| c != ccd));
    for d in (0..MAX_NODES).filter(|&d| d != node) {
        o.push(MAX_CCDS + d);
        if let Some(cs) = node_ccds.get(d) {
            o.extend(cs.iter().copied());
        }
    }
    o
}
/// `work` with `tick` called between items.
unsafe fn work_ticking(job: *const Job, order: &[usize], tick: &dyn Fn()) {
    let j = &*job;
    let f = &*j.f;
    let (mut mine, mut local) = (0, 0);
    for (k, &s) in order.iter().enumerate() {
        loop {
            tick();
            if j.stop.load(Ordering::Relaxed) {
                break;
            }
            let i = j.next[s].0.fetch_add(1, Ordering::AcqRel);
            if i >= j.start[s + 1] {
                break;
            }
            f(j.order[i] as usize);
            mine += 1;
            local += (k == 0) as usize;
        }
    }
    if mine > 0 {
        j.done.0.fetch_add(mine, Ordering::AcqRel);
        j.local.0.fetch_add(local, Ordering::Relaxed);
    }
}

/// Every item of `job` this thread can claim, slot by slot in its order; the completions
/// counted once, at the end.
unsafe fn work(job: *const Job, order: &[usize]) {
    let j = &*job;
    let f = &*j.f;
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
            f(j.order[i] as usize);
            mine += 1;
            local += (k == 0) as usize;
        }
    }
    if mine > 0 {
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
        for &core in &cores[1..] {
            let sh = shared.clone();
            let order = slot_order(node_of_core(core), ccd_of(core), &node_ccds);
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
                            std::thread::sleep(std::time::Duration::from_micros(50));
                        }
                        continue;
                    }
                    // registered first, then the job as it is now: whatever it is, it is alive and its counters are its own
                    sh.active.fetch_add(1, Ordering::AcqRel);
                    let job = sh.job.load(Ordering::Acquire);
                    seen = job;
                    if !job.is_null() {
                        unsafe { work(job, &order) };
                    }
                    sh.active.fetch_sub(1, Ordering::AcqRel);
                }
            });
        }
        let order = slot_order(node_of_core(cores[0]), ccd_of(cores[0]), &node_ccds);
        Threads { shared, retired: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()), order, node_ccds, placed: std::sync::Mutex::new(Vec::new()), tick: std::sync::Mutex::new(None), locality: std::sync::Mutex::new((0, 0)), prefetched: std::sync::Mutex::new((0, 0)) }
    }
    /// Tell the pool where a region's memory lives: `nodes[p]` is the node of its `p`th `PAGE`.
    pub fn place(&self, base: usize, len: usize, nodes: Vec<u8>) {
        assert!(nodes.len() == len.div_ceil(PAGE) && nodes.iter().all(|&d| (d as usize) < MAX_NODES), "placement of {len} bytes: {} pages", nodes.len());
        self.placed.lock().unwrap().push((base, len, nodes));
    }
    /// The node is about to wait: the workers may sleep between looks until the next `run`.
    pub fn idle(&self) {
        self.shared.busy.store(false, Ordering::Relaxed);
    }
    /// The items by slot (a counting sort over the placement: the node of each item's rows,
    /// and its preferred CCD within that node when it names one; every item is node 0's when
    /// nothing is placed or named), published, worked and waited for.
    fn publish(&self, n: usize, at: Option<&(dyn Fn(usize) -> *const u8 + Sync)>, ccd: Option<&(dyn Fn(usize) -> usize + Sync)>, f: &(dyn Fn(usize) + Sync)) {
        if n <= 1 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        let job = self.post(n, at, ccd, f, None);
        let tick = *self.tick.lock().unwrap();
        match tick {
            Some(t) => unsafe { work_ticking(job, &self.order, &*t.0) },
            None => unsafe { work(job, &self.order) },
        }
        while unsafe { (*job).done.0.load(Ordering::Acquire) } != n {
            if let Some(t) = tick {
                unsafe { (*t.0)() };
            }
            std::hint::spin_loop();
        }
        if ccd.is_some() {
            *self.locality.lock().unwrap() = (unsafe { (*job).local.0.load(Ordering::Relaxed) }, n);
        }
    }
    /// The job slotted and published: the workers take it up; the job before it is stopped
    /// (a prefetch still running gives up its unclaimed items) and freed a generation late.
    fn post(&self, n: usize, at: Option<&(dyn Fn(usize) -> *const u8 + Sync)>, ccd: Option<&(dyn Fn(usize) -> usize + Sync)>, f: &(dyn Fn(usize) + Sync), owned: Option<Box<dyn Fn(usize) + Send + Sync>>) -> *mut Job {
        let mut start = [0usize; SLOTS + 1];
        let mut order = vec![0u32; n];
        let mut item_slot = Vec::new();
        if let Some(at) = at {
            let placed = self.placed.lock().unwrap();
            if !placed.is_empty() {
                item_slot = (0..n)
                    .map(|i| {
                        let p = at(i) as usize;
                        let d = placed.iter().find(|(base, len, _)| p >= *base && p < base + len).map_or(0, |(base, _, nodes)| nodes[(p - base) / PAGE] as usize);
                        match ccd {
                            Some(c) if !self.node_ccds[d].is_empty() => self.node_ccds[d][c(i) % self.node_ccds[d].len()] as u8,
                            _ => (MAX_CCDS + d) as u8,
                        }
                    })
                    .collect();
            }
        }
        let slot_of = |i: usize| item_slot.get(i).map_or(MAX_CCDS, |&s| s as usize);
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
        // the job before this one stops handing out items (it finished, or it was a prefetch), then no worker is inside it
        let current = sh.job.load(Ordering::Acquire);
        if !current.is_null() {
            unsafe { (*current).stop.store(true, Ordering::Relaxed) };
        }
        while sh.active.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        if !current.is_null() && unsafe { (*current)._owned.is_some() } {
            // a prefetch it overtook: what it managed
            *self.prefetched.lock().unwrap() = unsafe { ((*current).done.0.load(Ordering::Relaxed), (*current).n) };
        }
        // no worker is inside any job: the one retired at the last publish can go
        let fp: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        let job = Box::into_raw(Box::new(Job { f: fp, order, next: std::array::from_fn(|s| Counter(AtomicUsize::new(start[s]))), start, done: Counter(AtomicUsize::new(0)), local: Counter(AtomicUsize::new(0)), stop: std::sync::atomic::AtomicBool::new(false), n, _owned: owned }));
        let old = sh.job.swap(job, Ordering::AcqRel);
        let older = self.retired.swap(old, Ordering::AcqRel);
        if !older.is_null() {
            drop(unsafe { Box::from_raw(older) });
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
        self.publish(n, None, None, f)
    }
    fn run_at_ccd(&self, n: usize, at: &(dyn Fn(usize) -> *const u8 + Sync), ccd: &(dyn Fn(usize) -> usize + Sync), f: &(dyn Fn(usize) + Sync)) {
        self.publish(n, Some(at), Some(ccd), f)
    }
    fn node_of(&self, p: *const u8) -> usize {
        let p = p as usize;
        self.placed.lock().unwrap().iter().find(|(base, len, _)| p >= *base && p < base + len).map_or(0, |(base, _, nodes)| nodes[(p - base) / PAGE] as usize)
    }
    fn ccds_of(&self, node: usize) -> usize {
        self.node_ccds.get(node).map_or(1, |c| c.len().max(1))
    }
    fn abandoned(&self) -> bool {
        let job = self.shared.job.load(Ordering::Acquire);
        !job.is_null() && unsafe { (*job).stop.load(Ordering::Relaxed) }
    }
    fn locality(&self) -> (usize, usize) {
        *self.locality.lock().unwrap()
    }
    fn prefetched(&self) -> (usize, usize) {
        *self.prefetched.lock().unwrap()
    }
    fn prefetch(&self, n: usize, at: Box<dyn Fn(usize) -> *const u8 + Send + Sync>, ccd: Box<dyn Fn(usize) -> usize + Send + Sync>, f: Box<dyn Fn(usize) + Send + Sync>) {
        if n > 1 {
            // the job owns `f`; `at` and `ccd` are needed only to slot the items here
            let fr: &(dyn Fn(usize) + Sync) = unsafe { &*(&*f as *const (dyn Fn(usize) + Send + Sync)) };
            self.post(n, Some(&*at), Some(&*ccd), fr, Some(f));
        }
    }
    fn tick(&self, f: Option<&dyn Fn()>) {
        // only the calling thread ever runs it, between its own items
        *self.tick.lock().unwrap() = f.map(|f| Tick(unsafe { std::mem::transmute::<&dyn Fn(), *const dyn Fn()>(f) }));
    }
    fn run_at(&self, n: usize, at: &(dyn Fn(usize) -> *const u8 + Sync), f: &(dyn Fn(usize) + Sync)) {
        self.publish(n, Some(at), None, f)
    }
}

/// A core's core complex, named by the first core sharing its L3 (from /sys; the core itself without the file).
pub fn ccd_key(core: usize) -> usize {
    std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{core}/cache/index3/shared_cpu_list")).ok().and_then(|s| s.trim().split([',', '-']).next().and_then(|v| v.parse().ok())).unwrap_or(core)
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
/// first SMT sibling of each), every socket, as /sys lists them.
pub fn physical_cores() -> Vec<usize> {
    let mut cores = Vec::new();
    let mut i = 0;
    while let Ok(s) = std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list")) {
        let first: usize = s.trim().split([',', '-']).next().unwrap().parse().unwrap();
        if first == i {
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
    stream: &'a [u8],
    pool: &'a dyn Pool,
    /// The node of every slice of every expert record (`[MOE_LAYERS][E][SLICES]`), asked of the
    /// pool once at construction, so slotting a prefetch is lookups, not a lock a slice.
    slice_node: Vec<u8>,
    xq: Vec<i8>,
    xs: Vec<f32>,
    xg: Vec<i32>,
    h: Vec<f32>, // [TOP_K][D_FF]: the experts' hidden states
    elapsed: f64,
}
/// Hidden channels an item takes: 32 items an expert.
const CHUNK: usize = 64;
/// Output rows an item takes for the down projection: 20 items an expert.
const DOWN_CHUNK: usize = 256;
const _: () = assert!(D_FF % CHUNK == 0 && CHUNK % common::q4::TILE == 0 && D_FF % common::q4::GROUP == 0 && H % DOWN_CHUNK == 0 && DOWN_CHUNK % common::q4::TILE == 0);

impl<'a> CpuExperts<'a> {
    pub fn new(stream: &'a [u8], pool: &'a dyn Pool) -> Self {
        use common::q4::GROUP;
        assert_eq!(stream.len(), layout::STREAM);
        use common::q4::TILE;
        let (n_gu, n_down) = (D_FF / CHUNK, H / DOWN_CHUNK);
        let mut slice_node = Vec::with_capacity(layout::MOE_LAYERS * E * (n_gu + n_down));
        for l in FIRST_K_DENSE_REPLACE..NUM_HIDDEN_LAYERS {
            for e in 0..E {
                let w = expert::Weights::from_record(&stream[layout::record(l, e)]);
                for j in 0..n_gu {
                    slice_node.push(pool.node_of(w.gate_up_w[j * CHUNK / TILE * 2 * H..].as_ptr()) as u8);
                }
                for c in 0..n_down {
                    slice_node.push(pool.node_of(w.down_w[c * DOWN_CHUNK / TILE * 2 * D_FF..].as_ptr()) as u8);
                }
            }
        }
        CpuExperts { stream, pool, slice_node, xq: vec![0; H], xs: vec![0.0; H / GROUP], xg: vec![0; H / GROUP], h: vec![0.0; TOP_K * D_FF], elapsed: 0.0 }
    }
    /// The node of slice `j` (gate/up slices first, then down) of expert `e` of layer `l`.
    fn slice_node(&self, l: usize, e: usize, j: usize) -> usize {
        self.slice_node[((l - FIRST_K_DENSE_REPLACE) * E + e) * (D_FF / CHUNK + H / DOWN_CHUNK) + j] as usize
    }
}
fn raw_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
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
    fn prefetch(&self, l: usize, idx: [usize; TOP_K]) {
        use common::q4::{GROUP, TILE};
        // the stream lives as long as the process (the node leaks it; the gate's outlives every token):
        // the job that owns these closures is gone within the token
        let stream: &'static [u8] = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(self.stream) };
        let pool = self.pool;
        // an item an (expert, node, CCD): the slices of that expert the compute will slot there, in the
        // compute's order — a slice's node from its rows' page, its CCD from the expert's id and the slice
        let (n_gu, n_down) = (D_FF / CHUNK, H / DOWN_CHUNK);
        let mut items: Vec<(usize, usize, usize, Vec<u32>)> = Vec::new(); // expert position, node, ccd, slice ids (gate/up first, then down + n_gu)
        for k in 0..TOP_K {
            for j in 0..n_gu + n_down {
                let hash = if j < n_gu { idx[k] * n_gu + j } else { idx[k] * n_down + (j - n_gu) };
                let d = self.slice_node(l, idx[k], j);
                let c = hash % pool.ccds_of(d);
                match items.iter_mut().find(|(kk, dd, cc, _)| *kk == k && *dd == d && *cc == c) {
                    Some(it) => it.3.push(j as u32),
                    None => items.push((k, d, c, vec![j as u32])),
                }
            }
        }
        let n = items.len();
        let items = std::sync::Arc::new(items);
        let (i1, i2, i3) = (items.clone(), items.clone(), items);
        let first_slice = move |it: &(usize, usize, usize, Vec<u32>)| -> *const u8 {
            let w = expert::Weights::from_record(&stream[layout::record(l, idx[it.0])]);
            let j = it.3[0] as usize;
            if j < n_gu { w.gate_up_w[j * CHUNK / TILE * 2 * H..].as_ptr() } else { w.down_w[(j - n_gu) * DOWN_CHUNK / TILE * 2 * D_FF..].as_ptr() }
        };
        let at = move |i: usize| first_slice(&i1[i]);
        let ccd = move |i: usize| i2[i].2;
        // the touching yields between 64 KB chunks once the job is overtaken: the run that overtakes it
        // waits for the items in flight, and a megabyte of touching would be a quarter millisecond of stall
        let pool_s: &'static dyn Pool = unsafe { std::mem::transmute::<&dyn Pool, &'static dyn Pool>(pool) };
        let f = move |i: usize| {
            let it = &i3[i];
            let w = expert::Weights::from_record(&stream[layout::record(l, idx[it.0])]);
            let touch_yielding = |b: &[u8]| -> bool {
                for chunk in b.chunks(64 << 10) {
                    if pool_s.abandoned() {
                        return false;
                    }
                    touch(chunk);
                }
                true
            };
            for &j in &it.3 {
                let j = j as usize;
                let go = if j < n_gu {
                    let mut go = true;
                    for r0 in [j * CHUNK, D_FF + j * CHUNK] {
                        go = go && touch_yielding(&w.gate_up_w[r0 / TILE * 2 * H..(r0 + CHUNK) / TILE * 2 * H]) && touch_yielding(raw_f32(&w.gate_up_s[r0 / TILE * (H / GROUP) * TILE..(r0 + CHUNK) / TILE * (H / GROUP) * TILE]));
                    }
                    go
                } else {
                    let r0 = (j - n_gu) * DOWN_CHUNK;
                    touch_yielding(&w.down_w[r0 / TILE * 2 * D_FF..(r0 + DOWN_CHUNK) / TILE * 2 * D_FF]) && touch_yielding(raw_f32(&w.down_s[r0 / TILE * (D_FF / GROUP) * TILE..(r0 + DOWN_CHUNK) / TILE * (D_FF / GROUP) * TILE]))
                };
                if !go {
                    return;
                }
            }
        };
        self.pool.prefetch(n, Box::new(at), Box::new(ccd), Box::new(f));
    }
    fn run(&mut self, l: usize, idx: &[usize; TOP_K], x: &[f32], out: &mut [f32]) {
        use common::q4::{self, Act, QAct, GROUP, TILE};
        use common::swiglu::swiglu_clamped;
        let t0 = std::time::Instant::now();
        QAct { q: &mut self.xq, scale: &mut self.xs, gsum: &mut self.xg }.quantize(&x[..H]);
        let xa = Act { q: &self.xq, scale: &self.xs, gsum: &self.xg };
        let stream = self.stream;
        let (h_p, out_p) = (self.h.as_mut_ptr() as usize, out.as_mut_ptr() as usize);
        // the gate/up rows: item (k, c) owns expert k's hidden channels c·CHUNK..
        let gate_rows = move |i: usize| {
            let (k, c) = (i / (D_FF / CHUNK), i % (D_FF / CHUNK));
            expert::Weights::from_record(&stream[layout::record(l, idx[k])]).gate_up_w[c * CHUNK / TILE * 2 * H..].as_ptr()
        };
        // a slice's core complex within its node: a function of the EXPERT and the slice, never of the
        // expert's position among the picks, so a prefetch and the compute meet in one L3
        let gate_ccd = move |i: usize| idx[i / (D_FF / CHUNK)] * (D_FF / CHUNK) + i % (D_FF / CHUNK);
        self.pool.run_at_ccd(TOP_K * (D_FF / CHUNK), &gate_rows, &gate_ccd, &move |i| {
            let (k, c) = (i / (D_FF / CHUNK), i % (D_FF / CHUNK));
            let w = expert::Weights::from_record(&stream[layout::record(l, idx[k])]);
            let rows = |r0: usize| (&w.gate_up_w[r0 / TILE * 2 * H..(r0 + CHUNK) / TILE * 2 * H], &w.gate_up_s[r0 / TILE * (H / GROUP) * TILE..(r0 + CHUNK) / TILE * (H / GROUP) * TILE]);
            let mut gu = [0f32; 2 * CHUNK];
            let (gw, gs) = rows(c * CHUNK);
            q4::gemv::<H>(CHUNK, gw, gs, &xa, &mut gu[..CHUNK]);
            let (uw, us) = rows(D_FF + c * CHUNK);
            q4::gemv::<H>(CHUNK, uw, us, &xa, &mut gu[CHUNK..]);
            let h = unsafe { std::slice::from_raw_parts_mut((h_p as *mut f32).add(k * D_FF + c * CHUNK), CHUNK) };
            swiglu_clamped::<CHUNK>(&gu, SWIGLU_LIMIT, h);
        });
        // the down rows: item (k, c) owns rows c·DOWN_CHUNK.. of expert k's output over its whole hidden state, quantized here
        let h = &self.h;
        let down_rows = move |i: usize| {
            let (k, c) = (i / (H / DOWN_CHUNK), i % (H / DOWN_CHUNK));
            expert::Weights::from_record(&stream[layout::record(l, idx[k])]).down_w[c * DOWN_CHUNK / TILE * 2 * D_FF..].as_ptr()
        };
        let down_ccd = move |i: usize| idx[i / (H / DOWN_CHUNK)] * (H / DOWN_CHUNK) + i % (H / DOWN_CHUNK);
        self.pool.run_at_ccd(TOP_K * (H / DOWN_CHUNK), &down_rows, &down_ccd, &move |i| {
            let (k, c) = (i / (H / DOWN_CHUNK), i % (H / DOWN_CHUNK));
            let w = expert::Weights::from_record(&stream[layout::record(l, idx[k])]);
            let (mut hq, mut hs, mut hg) = ([0i8; D_FF], [0f32; D_FF / GROUP], [0i32; D_FF / GROUP]);
            QAct { q: &mut hq, scale: &mut hs, gsum: &mut hg }.quantize(&h[k * D_FF..(k + 1) * D_FF]);
            let ha = Act { q: &hq, scale: &hs, gsum: &hg };
            let r0 = c * DOWN_CHUNK;
            let (dw, ds) = (&w.down_w[r0 / TILE * 2 * D_FF..(r0 + DOWN_CHUNK) / TILE * 2 * D_FF], &w.down_s[r0 / TILE * (D_FF / GROUP) * TILE..(r0 + DOWN_CHUNK) / TILE * (D_FF / GROUP) * TILE]);
            let o = unsafe { std::slice::from_raw_parts_mut((out_p as *mut f32).add(k * H + r0), DOWN_CHUNK) };
            q4::gemv::<D_FF>(DOWN_CHUNK, dw, ds, &ha, o);
        });
        self.elapsed += t0.elapsed().as_secs_f64();
    }
}

/// Blocks placed away from this CPU — on the cards — behind one interface.
/// `kda_step` runs placed KDA layer `l` over the block input `x` (`[H]`,
/// before the input norm: the placement norms it) into `out`; `seed` and
/// `snapshot` move a placed layer's state (its convolution window and
/// memory). `ffn_start` begins a placed FFN — a dense layer's MLP, or a MoE
/// layer's shared expert — over the block input (normed there with the post
/// norm) and returns at once, so the routed experts run here meanwhile;
/// `ffn_finish` waits for its output. `logits` is the final norm and the
/// head over the streams' mean. A block the placement does not hold runs here.
pub trait Placed {
    fn kda(&self, l: usize) -> bool;
    fn seed(&mut self, l: usize, window: &[f32], mem: &[f32]);
    fn snapshot(&self, l: usize, window: &mut [f32], mem: &mut [f32]);
    fn ffn(&self, l: usize) -> bool;
    fn ffn_start(&mut self, l: usize, x: &[f32]);
    fn ffn_finish(&mut self, l: usize, out: &mut [f32]);
    /// Layer `l`'s router over the block input `x` normed with the FFN half's norm — the FFN
    /// half's router run early, on the attention block's input, as the prefetch's predictor:
    /// the logits into `logits` (`[E]`), waited for.
    fn route_early(&mut self, l: usize, x: &[f32]);
    fn early_logits(&mut self, l: usize, logits: &mut [f32]);
    /// The attention blocks posted, then waited for: between the two the host reads the early
    /// router and starts the prefetch while the card works.
    fn kda_post(&mut self, l: usize, x: &[f32]);
    fn kda_wait(&mut self, l: usize, out: &mut [f32]);
    fn mla_post(&mut self, l: usize, x: &[f32]);
    fn mla_wait(&mut self, l: usize, kva: &mut [f32], ik: &mut [f32], igate: &mut [f32], iq: &mut [f32], iw: &mut [f32]);
    /// Keep layer `l`'s card awake while the host works: a trivial launch once it has been
    /// quiet a while (a card idle ≥ 400 µs pays ~100 µs on its next packet, probe 2026-10-15).
    fn heartbeat(&self, l: usize);
    fn head(&self) -> bool;
    fn logits(&mut self, mean: &[f32], logits: &mut [f32]);
    /// The MLA projections of block input `x` (`mla::project`'s outputs but
    /// the query, which stays there: the latent and the indexer's key before
    /// their norms, the gate, the indexer's query and unscaled head weights),
    /// waited for; then the attention over the selected positions `sel` with
    /// position `pos`'s latent just made, and the output projection, into
    /// `out`. The latents live there too: `mla_seed` installs position
    /// `pos`'s when a sequence is seeded.
    fn mla(&self, l: usize) -> bool;
    fn mla_attend(&mut self, l: usize, pos: usize, latent: &[f32], sel: &[usize], out: &mut [f32]);
    fn mla_seed(&mut self, l: usize, pos: usize, latent: &[f32]);
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
    pub hc: f64,
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
}
impl Profile {
    /// One line of milliseconds a token, the phases in order and their sum.
    pub fn line(&self) -> String {
        let n = self.tokens.max(1) as f64 / 1e3;
        let mla = self.mla_proj + self.mla_attn + self.mla_out;
        let rs = self.route + self.shared;
        let sum = self.hc + self.kda_card + self.kda_cpu + mla + self.dense_ffn + rs + self.experts + self.head;
        format!("{} tokens: hc {:.1}  kda card {:.1}  kda cpu {:.1}  mla {:.1} (proj {:.1} attn {:.1} out {:.1})  dense ffn {:.1}  router+shared {:.1} (route {:.1} shared {:.1})  experts {:.1} (reused {:.2} of {} from the previous token, {:.2} named on the attention input; {:.0}% of the predicted slices pulled; {:.0}% of items on their CCD)  head {:.1}  = {:.1} ms a token", self.tokens, self.hc / n, self.kda_card / n, self.kda_cpu / n, mla / n, self.mla_proj / n, self.mla_attn / n, self.mla_out / n, self.dense_ffn / n, rs / n, self.route / n, self.shared / n, self.experts / n, self.reuse as f64 / self.moe_layers.max(1) as f64, TOP_K, self.pre_agree as f64 / self.moe_layers.max(1) as f64, 100.0 * self.prefetched as f64 / self.prefetch_items.max(1) as f64, 100.0 * self.local_items as f64 / self.items.max(1) as f64, self.head / n, sum / n)
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
    /// The last step's logits, `[VOCAB_SIZE]`.
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
    m_sel: Vec<usize>,
    m_heads: Vec<f32>,
    m_hs: Vec<f32>,
    f_gu: Vec<f32>,
    f_h: Vec<f32>,
    f_expert_out: Vec<f32>,
    f_shared: Vec<f32>,
    f_score: Vec<f32>,
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
            layers,
            logits: vec![0.0; VOCAB_SIZE],
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
            m_sel: vec![0; DENSE_BELOW],
            m_heads: vec![0.0; MLA_HEADS * V_HEAD_DIM],
            m_hs: vec![0.0; MLA_HEADS * mla::HEAD_SCRATCH],
            f_gu: vec![0.0; 2 * INTERMEDIATE_SIZE],
            f_h: vec![0.0; INTERMEDIATE_SIZE],
            f_expert_out: vec![0.0; TOP_K * H],
            f_shared: vec![0.0; H],
            f_score: vec![0.0; E],
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
                    if let Some(p) = self.placed.as_mut().filter(|p| p.kda(l)) {
                        p.seed(l, window, mem);
                    }
                    ki += 1;
                }
                (LayerState::Mla { cache, pools }, Attn::Mla(mw)) => {
                    for p in 0..n {
                        cache[p * MLA_ROW..(p + 1) * MLA_ROW].copy_from_slice(&rows[p * POSITION + mi * MLA_ROW..p * POSITION + (mi + 1) * MLA_ROW]);
                    }
                    for j in 0..n / INDEX_KPOOL {
                        let (c, pk) = (&*cache, &mut pools[j * INDEX_HEAD_DIM..(j + 1) * INDEX_HEAD_DIM]);
                        mla::pool_key(mw, c, j, pk);
                    }
                    if let Some(pl) = self.placed.as_mut().filter(|p| p.mla(l)) {
                        for p in 0..n {
                            pl.mla_seed(l, p, &cache[p * MLA_ROW..p * MLA_ROW + KV_LORA_RANK]);
                        }
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
        let (rows, snap) = prefix.split_at(FIRST_POS * POSITION);
        self.seed(w, rows, snap)
    }
    /// The KDA layers' state now, `[SNAPSHOT]`.
    pub fn snapshot(&self, out: &mut [f32]) {
        let mut ki = 0;
        for (l, st) in self.layers.iter().enumerate() {
            if let LayerState::Kda { window, mem } = st {
                let base = ki * layout::KDA_STATE;
                let (w, m) = out[base..base + layout::KDA_STATE].split_at_mut(KDA_WINDOW);
                match self.placed.as_ref().filter(|p| p.kda(l)) {
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
    /// of the router's, `[MoE layer][TOP_K]` (bring-up only).
    pub fn step(&mut self, w: &Weights, tok: u32, pos: usize, forced: Option<&[[usize; TOP_K]]>, tr: &mut impl Trace) {
        assert!(pos < self.max_ctx);
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
            // the prefetch's predictor: the FFN half's router on the attention block's input, posted on the card where
            // the FFN is placed; read once the attention launches are posted, its picks, likeliest first, start the pull
            let early = matches!(&lw.ffn, Ffn::Moe(_)) && s.placed.as_ref().is_some_and(|p| p.ffn(l));
            if early {
                s.placed.as_mut().unwrap().route_early(l, &s.x);
            }
            let mut pre_picks = [E; TOP_K];
            lap(&mut p.hc);
            match (&lw.attn, &mut s.layers[l]) {
                (Attn::Kda(_), _) if s.placed.as_ref().is_some_and(|p| p.kda(l)) => {
                    let pl = s.placed.as_mut().unwrap();
                    pl.kda_post(l, &s.x);
                    if early {
                        if let Ffn::Moe(m) = &lw.ffn {
                            pl.early_logits(l, &mut s.f_score);
                            pre_picks = ffn::picks(m, &mut s.f_score);
                            s.routed.prefetch(l, pre_picks);
                        }
                    }
                    pl.kda_wait(l, &mut s.y);
                    lap(&mut p.kda_card);
                }
                (Attn::Kda(a), LayerState::Kda { window, mem }) => {
                    let mut ks = kda::Scratch { mixed: &mut s.k_mixed, conv: &mut s.k_conv, fa: &mut s.k_fa, f: &mut s.k_f, beta: &mut s.k_beta, ga: &mut s.k_ga, gate: &mut s.k_gate, normed: &mut s.k_normed, heads: &mut s.k_heads };
                    kda::step(l, a, &s.xn, window, mem, &mut ks, s.pool, tr, &mut s.y);
                    lap(&mut p.kda_cpu);
                }
                (Attn::Mla(a), LayerState::Mla { cache, pools }) => {
                    let mut ms = mla::Scratch { qr: &mut s.m_qr, qa: &mut s.m_qa, q: &mut s.m_q, kva: &mut s.m_kva, ik: &mut s.m_ik, iq: &mut s.m_iq, iw: &mut s.m_iw, pscore: &mut s.m_pscore, sel: &mut s.m_sel, heads: &mut s.m_heads, hs: &mut s.m_hs };
                    let mut placed_mla = s.placed.as_mut().filter(|p| p.mla(l));
                    {
                        let (latent, ikey, igate) = mla::row(cache, pos);
                        match &mut placed_mla {
                            Some(pl) => {
                                pl.mla_post(l, &s.x);
                                if early {
                                    if let Ffn::Moe(m) = &lw.ffn {
                                        pl.early_logits(l, &mut s.f_score);
                                        pre_picks = ffn::picks(m, &mut s.f_score);
                                        s.routed.prefetch(l, pre_picks);
                                    }
                                }
                                pl.mla_wait(l, ms.kva, ms.ik, igate, ms.iq, ms.iw);
                            }
                            None => mla::project(a, &s.xn, &mut ms, igate, s.pool),
                        }
                        mla::finish_row(a, &ms, latent, ikey);
                    }
                    lap(&mut p.mla_proj);
                    let n = mla::select_at(l, a, cache, pools, pos, &mut ms, tr);
                    match placed_mla {
                        Some(pl) => {
                            lap(&mut p.mla_attn);
                            pl.mla_attend(l, pos, &cache[pos * MLA_ROW..pos * MLA_ROW + KV_LORA_RANK], &ms.sel[..n], &mut s.y);
                        }
                        None => {
                            mla::heads(l, a, cache, &mut ms, n, s.pool, tr);
                            lap(&mut p.mla_attn);
                            mla::output(a, ms.heads, &mut s.y, s.pool);
                        }
                    }
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
            let mut placed = s.placed.as_mut().filter(|p| p.ffn(l));
            match &lw.ffn {
                Ffn::Dense(d) => {
                    match placed {
                        Some(p) => {
                            p.ffn_start(l, &s.x);
                            p.ffn_finish(l, &mut s.y);
                            tr.edge(l, "mlp.out", &s.y);
                        }
                        None => ffn::dense(l, d, &s.xn, &mut fs, s.pool, tr, &mut s.y),
                    }
                    lap(&mut p.dense_ffn);
                }
                Ffn::Moe(m) => {
                    // the shared expert runs on its card while the routed experts run here
                    if let Some(p) = &mut placed {
                        p.ffn_start(l, &s.x);
                    }
                    // the heartbeat through the routed experts, from the main thread between its items
                    let beat = placed.as_deref().map(|p| move || p.heartbeat(l));
                    if let Some(b) = &beat {
                        s.pool.tick(Some(b));
                    }
                    let picks = ffn::moe_routed(l, m, &s.xn, &mut fs, s.pool, &mut s.routed, forced.map(|f| &f[l - FIRST_K_DENSE_REPLACE]), tr, &mut s.y);
                    if beat.is_some() {
                        s.pool.tick(None);
                    }
                    p.reuse += picks.iter().filter(|e| s.last_picks[l].contains(e)).count();
                    p.pre_agree += picks.iter().filter(|e| pre_picks.contains(e)).count();
                    p.moe_layers += 1;
                    s.last_picks[l] = picks;
                    for &e in &picks {
                        s.freq[l * E + e] += 1;
                    }
                    let (local, items) = s.pool.locality();
                    p.local_items += local;
                    p.items += items;
                    let (done, of) = s.pool.prefetched();
                    p.prefetched += done;
                    p.prefetch_items += of;
                    let e = s.routed.elapsed();
                    p.experts += e;
                    lap(&mut p.route);
                    p.route -= e;
                    match placed {
                        Some(p) => p.ffn_finish(l, fs.shared),
                        None => ffn::moe_shared(m, &s.xn, &mut fs, s.pool),
                    }
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
        match s.placed.as_mut().filter(|p| p.head()) {
            Some(p) => p.logits(&s.mean, &mut s.logits),
            None => {
                rmsnorm::<H>(&s.mean, &w.norm, RMS_NORM_EPS, &mut s.xn);
                gemv::bf16_par::<VOCAB_SIZE, H>(w.lm_head, &s.xn, &mut s.logits, s.pool);
            }
        }
        tr.edge(NONE, "logits", &s.logits);
        lap(&mut p.head);
    }
}
