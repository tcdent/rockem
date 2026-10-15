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
    /// Run `f` between the calling thread's own items of the runs until `tick(None)`: a
    /// heartbeat for a device that must not fall asleep while the pool works. `f` must
    /// outlive those runs. A pool without a calling thread's loop ignores it.
    fn tick(&self, f: Option<&dyn Fn()>) {
        let _ = f;
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
