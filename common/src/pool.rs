//! Where a kernel's independent work items run: a `Pool` takes `n` items and
//! a function of the index and returns when every item has run, on whatever
//! threads it owns. Chosen at construction and threaded through the fused
//! kernels like the trace: `Serial` runs the items inline, in order, and is
//! the realization the GPU-bound node and every test use; a threaded pool
//! (`std`, the model crate's `construct`) spreads them over pinned cores. A
//! kernel never asks which. Items must be independent: they are given disjoint
//! rows or heads to write, never the same word.

pub trait Pool: Sync {
    /// Run `f(i)` for every `i < n`, returning when all have.
    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync));
    /// The same, each item naming the address of the weights it streams
    /// (`at(i)`): a pool that knows where memory lives runs the item on a core
    /// local to it. The default is `run`; the items are still independent.
    fn run_at(&self, n: usize, at: &(dyn Fn(usize) -> *const u8 + Sync), f: &(dyn Fn(usize) + Sync)) {
        let _ = at;
        self.run(n, f)
    }
    /// `run_at` with a preferred core complex (CCD) for each item within its rows' node:
    /// `ccd(i)` is any stable number of the item (taken modulo the node's CCDs), so the
    /// cores whose L3 prefetched an item's rows are the ones that stream them. The
    /// default ignores the preference.
    fn run_at_ccd(&self, n: usize, at: &(dyn Fn(usize) -> *const u8 + Sync), ccd: &(dyn Fn(usize) -> usize + Sync), f: &(dyn Fn(usize) + Sync)) {
        let _ = ccd;
        self.run_at(n, at, f)
    }
    /// `run_at_ccd` for a caller that already knows each item's node: `node(i)` instead of an
    /// address to look up, so slotting an item costs nothing. The default ignores both.
    fn run_on(&self, n: usize, node: &(dyn Fn(usize) -> usize + Sync), ccd: &(dyn Fn(usize) -> usize + Sync), f: &(dyn Fn(usize) + Sync)) {
        let _ = (node, ccd);
        self.run(n, f)
    }
    /// The queue an item on `node` with core-complex preference `ccd` is slotted into — what
    /// `run_on` computes per item; a caller holds it in a map and runs `run_slotted`.
    fn slot(&self, node: usize, ccd: usize) -> usize {
        let _ = (node, ccd);
        0
    }
    /// `run_on` with each item's queue given outright (`slot(i)`, from `Pool::slot`).
    fn run_slotted(&self, n: usize, slot: &(dyn Fn(usize) -> usize + Sync), f: &(dyn Fn(usize) + Sync)) {
        let _ = slot;
        self.run(n, f)
    }
    /// Whether the pool can build a run's job ahead (`prepare_slotted`) and announce it later (`run_prepared`).
    #[cfg(feature = "std")]
    fn prepares(&self) -> bool {
        false
    }
    /// A run's job built — items sorted into their queues (`slots[i]`, from `Pool::slot`), the job allocated — but not
    /// announced: `run_prepared` runs it. The closure is the job's own. Returns a handle (0: not supported).
    #[cfg(feature = "std")]
    fn prepare_slotted(&self, slots: &[u8], f: std::boxed::Box<dyn Fn(usize) + Send + Sync>) -> usize {
        let _ = (slots, f);
        0
    }
    /// Announce a job `prepare_slotted` built, work on it with this thread too, and return when it is done.
    #[cfg(feature = "std")]
    fn run_prepared(&self, job: usize) {
        let _ = job;
        unreachable!("run_prepared: this pool does not prepare jobs");
    }
    /// `prefetch_on` with each item's queue given outright. Returns, for a pool that records it, which worker took each
    /// item (`u8::MAX`: not taken), filled in as the items are claimed: a later run can send the item's bytes to the
    /// core whose caches hold them.
    #[cfg(feature = "std")]
    fn prefetch_slotted(&self, n: usize, slot: std::boxed::Box<dyn Fn(usize) -> usize + Send + Sync>, f: std::boxed::Box<dyn Fn(usize) + Send + Sync>) -> Option<std::sync::Arc<std::vec::Vec<core::sync::atomic::AtomicU8>>> {
        let _ = (n, slot, f);
        None
    }
    /// `prefetch` for a caller that knows each item's node (as `run_on`).
    #[cfg(feature = "std")]
    fn prefetch_on(&self, n: usize, node: std::boxed::Box<dyn Fn(usize) -> usize + Send + Sync>, ccd: std::boxed::Box<dyn Fn(usize) -> usize + Send + Sync>, f: std::boxed::Box<dyn Fn(usize) + Send + Sync>) {
        let _ = (n, node, ccd, f);
    }
    /// Start `f` over `n` items (slotted as `run_at_ccd`) on the pool's threads and return at
    /// once; the items are a hint — the next run abandons whatever has not been claimed. The
    /// pool owns the closures until the job is gone. A pool without its own threads does nothing.
    #[cfg(feature = "std")]
    fn prefetch(&self, n: usize, at: std::boxed::Box<dyn Fn(usize) -> *const u8 + Send + Sync>, ccd: std::boxed::Box<dyn Fn(usize) -> usize + Send + Sync>, f: std::boxed::Box<dyn Fn(usize) + Send + Sync>) {
        let _ = (n, at, ccd, f);
    }
    /// The node whose memory holds `p` (0 when nothing is placed), and how many core complexes
    /// a node has (1 where the pool does not tell them apart): what a caller needs to group
    /// slices the way `run_at_ccd` will slot them.
    fn node_of(&self, p: *const u8) -> usize {
        let _ = p;
        0
    }
    /// The cores of queue `slot` (from `slot`) that touch a prefetch: the bytes a queue is given follow it.
    fn prefetchers(&self, slot: usize) -> usize {
        let _ = slot;
        1
    }
    fn ccds_of(&self, node: usize) -> usize {
        let _ = node;
        1
    }
    /// Every (node, core complex) the pool has cores on, in a stable order — the `node` and `ccd` arguments
    /// `run_on` slots by: a caller gives items that share an operand the same complex, so its L3 holds it once.
    #[cfg(feature = "std")]
    fn complexes(&self) -> std::vec::Vec<(usize, usize)> {
        std::vec![(0, 0)]
    }
    /// Of the last run's items, how many ran on a core of their preferred CCD, and how many
    /// there were (0, 0 for a pool that has no CCDs to prefer).
    fn locality(&self) -> (usize, usize) {
        (0, 0)
    }
    /// Whether the job now running has been overtaken: a long prefetch item asks between chunks and returns.
    fn abandoned(&self) -> bool {
        false
    }
    /// Of the last prefetch's items, how many were done before a run overtook it, and how many there were.
    fn prefetched(&self) -> (usize, usize) {
        (0, 0)
    }
    /// Since the last call, over the placed runs: each node's items, and the sum over the runs
    /// of the time (s) its last item finished — a node's end of each run. What the placement
    /// of the items across the nodes is tuned by. All zero for a pool without nodes.
    fn node_ends(&self) -> ([usize; 8], [f64; 8]) {
        ([0; 8], [0.0; 8])
    }
    /// Since the last call, over the placed runs: seconds spent slotting (entry to post), waiting for
    /// the first claim (the workers' wake) and after the last node's end (the tail); and the runs.
    fn overheads(&self) -> (f64, f64, f64, usize) {
        (0.0, 0.0, 0.0, 0)
    }
    /// Since the last call: the sum over the placed runs of the last node's end (s).
    fn last_ends(&self) -> f64 {
        0.0
    }
    /// Since the last call, over the runs' posts: seconds sorting items into slots, draining the
    /// previous job's workers, allocating the job and freeing the retired one.
    fn post_stats(&self) -> (f64, f64, f64) {
        (0.0, 0.0, 0.0)
    }
}

pub struct Serial;
impl Pool for Serial {
    #[inline(always)]
    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        for i in 0..n {
            f(i)
        }
    }
}

/// Rows of a projection a thread takes at once: enough that the weights
/// streamed per item (`ROWS · K · bytes`) dwarf the item's scheduling cost,
/// few enough that a 1,536-row projection still splits over a socket's cores.
pub const ROWS: usize = 16;
