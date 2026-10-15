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
