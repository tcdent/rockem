//! The node's allocator: size classes, thread magazines, slabs that are never unmapped. musl's malloc (mallocng) hands
//! a freed medium block back to the kernel and page-faults it in again at the next allocation: the node's per-layer
//! allocations — a pool job, a run plan, a prefetch's items, ~20 blocks of 1–8 KB — cost 77 µs a layer under it against
//! 1.9 under glibc (bench-alloc, 2026-10-22), 3.5 ms a verify step on the main thread. This one has no such policy and
//! no behaviour we did not choose:
//!
//! - a block is a power of two from 16 B to 1 MiB (`CLASSES`), its class recovered from the `Layout` Rust hands back
//!   at `dealloc` (no header); larger or over-aligned requests go to the libc allocator (the model's buffers, at boot);
//! - every thread keeps a magazine a class — an intrusive queue, freed blocks at the tail, allocations from the head —
//!   and allocates and frees there without a lock; a queue past `KEEP` bytes spills its older half to the class's
//!   central list, an empty one refills `BATCH` blocks from it. First in, first out, and deep: a block a plan wrote is
//!   read by every core of the pool, and its lines sit shared in their caches; handed straight back (last in, first
//!   out) the next plan's stores into it each invalidated a line across the box — the plan took 35 µs for ~25 KB of
//!   writes (2026-10-22). Reused ~`KEEP`/size allocations later, the expert stream has long evicted them everywhere;
//! - the central list carves fresh blocks from 2 MiB slabs mapped populated (no first-touch faults on the hot path)
//!   and never unmaps them: the appliance's working set is fixed, what a step allocates it allocates every step.
//!
//! Blocks migrate with the threads that free them (a builder-core plan freed by the main thread lands in the main
//! thread's magazine, where its next plan is allocated from). Threads here are permanent; a thread that exits leaves
//! its magazines' blocks where they are.
use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const MIN_SHIFT: usize = 4;
const MAX_SHIFT: usize = 20;
const CLASSES: usize = MAX_SHIFT - MIN_SHIFT + 1;
/// A slab: 2 MiB, mapped populated.
const SLAB: usize = 2 << 20;
/// Blocks a refill takes from the central list.
const BATCH: usize = 16;
/// Bytes of one class a thread keeps before spilling the older half to the central list.
const KEEP: usize = 1 << 20;

/// The allocator, for `#[global_allocator]`.
pub struct Classes;

#[inline(always)]
fn class_of(layout: &Layout) -> Option<usize> {
    let s = layout.size().max(layout.align()).max(1 << MIN_SHIFT);
    if s > 1 << MAX_SHIFT {
        return None;
    }
    Some(s.next_power_of_two().trailing_zeros() as usize - MIN_SHIFT)
}

struct Magazine {
    head: [*mut u8; CLASSES],
    tail: [*mut u8; CLASSES],
    count: [u32; CLASSES],
}
std::thread_local! {
    static MAG: UnsafeCell<Magazine> = const { UnsafeCell::new(Magazine { head: [ptr::null_mut(); CLASSES], tail: [ptr::null_mut(); CLASSES], count: [0; CLASSES] }) };
}

/// One class's central state: its free list and the slab it is carving blocks from.
struct Central {
    lock: AtomicBool,
    head: UnsafeCell<*mut u8>,
    bump: UnsafeCell<*mut u8>,
    end: UnsafeCell<*mut u8>,
}
unsafe impl Sync for Central {}
#[allow(clippy::declare_interior_mutable_const)]
const EMPTY: Central = Central { lock: AtomicBool::new(false), head: UnsafeCell::new(ptr::null_mut()), bump: UnsafeCell::new(ptr::null_mut()), end: UnsafeCell::new(ptr::null_mut()) };
static CENTRAL: [Central; CLASSES] = [EMPTY; CLASSES];
static MAPPED: AtomicUsize = AtomicUsize::new(0);

/// Bytes mapped in slabs so far.
pub fn mapped() -> usize {
    MAPPED.load(Ordering::Relaxed)
}

/// Write the blocks this thread will allocate next (the head of each magazine, up to `bytes` a class) so they sit
/// modified in its own cache: a store into a line another core read, or one evicted long ago, costs ~35 ns for the
/// invalidation or the fetch, and a plan written into such lines was ~25 µs of ~25 KB (2026-10-22); into lines this
/// core owns it is the stores. For a thread with idle time ahead of a burst of allocation.
pub fn prewarm(bytes: usize) {
    let _ = MAG.try_with(|m| unsafe {
        let m = &mut *m.get();
        for c in 0..CLASSES {
            let size = 1 << (c + MIN_SHIFT);
            let mut p = m.head[c];
            let mut left = bytes;
            while !p.is_null() && left >= size {
                let next = *(p as *mut *mut u8);
                ptr::write_bytes(p.add(8), 0, size - 8);
                left -= size;
                p = next;
            }
        }
    });
}

extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
}
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_PRIVATE: i32 = 0x02;
const MAP_ANONYMOUS: i32 = 0x20;
const MAP_POPULATE: i32 = 0x8000;

impl Central {
    #[inline(always)]
    fn lock(&self) {
        while self.lock.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            core::hint::spin_loop();
        }
    }
    #[inline(always)]
    fn unlock(&self) {
        self.lock.store(false, Ordering::Release);
    }
    /// Up to `n` blocks of `size` as a linked list (the next pointer in each block's first word): its head, its last
    /// block, and how many.
    unsafe fn take(&self, size: usize, n: usize) -> (*mut u8, *mut u8, u32) {
        self.lock();
        let mut list: *mut u8 = ptr::null_mut();
        let mut last: *mut u8 = ptr::null_mut();
        let mut got = 0u32;
        let head = self.head.get();
        while (got as usize) < n && !(*head).is_null() {
            let p = *head;
            *head = *(p as *mut *mut u8);
            *(p as *mut *mut u8) = list;
            if list.is_null() {
                last = p;
            }
            list = p;
            got += 1;
        }
        let (bump, end) = (self.bump.get(), self.end.get());
        while (got as usize) < n {
            if (*bump).is_null() || (*end as usize) - (*bump as usize) < size {
                // a fresh slab, its blocks aligned to their size
                let len = SLAB + size;
                let m = mmap(ptr::null_mut(), len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
                if m as isize == -1 {
                    break;
                }
                MAPPED.fetch_add(len, Ordering::Relaxed);
                let base = (m as usize + size - 1) & !(size - 1);
                *bump = base as *mut u8;
                *end = (m as usize + len) as *mut u8;
            }
            let p = *bump;
            *bump = p.add(size);
            *(p as *mut *mut u8) = list;
            if list.is_null() {
                last = p;
            }
            list = p;
            got += 1;
        }
        self.unlock();
        (list, last, got)
    }
    /// A linked list of blocks given back.
    unsafe fn give(&self, list: *mut u8, last: *mut u8) {
        self.lock();
        let head = self.head.get();
        *(last as *mut *mut u8) = *head;
        *head = list;
        self.unlock();
    }
}

unsafe impl GlobalAlloc for Classes {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let Some(c) = class_of(&layout) else { return std::alloc::System.alloc(layout) };
        let size = 1 << (c + MIN_SHIFT);
        let Ok(p) = MAG.try_with(|m| {
            let m = &mut *m.get();
            if m.head[c].is_null() {
                let (list, last, got) = CENTRAL[c].take(size, BATCH);
                m.head[c] = list;
                m.tail[c] = last;
                m.count[c] = got;
            }
            let p = m.head[c];
            if !p.is_null() {
                m.head[c] = *(p as *mut *mut u8);
                if m.head[c].is_null() {
                    m.tail[c] = ptr::null_mut();
                }
                m.count[c] -= 1;
            }
            p
        }) else {
            let (list, _, _) = CENTRAL[c].take(size, 1);
            return list;
        };
        p
    }
    #[inline]
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        let Some(c) = class_of(&layout) else { return std::alloc::System.dealloc(p, layout) };
        let size = 1 << (c + MIN_SHIFT);
        if MAG
            .try_with(|m| {
                let m = &mut *m.get();
                *(p as *mut *mut u8) = ptr::null_mut();
                if m.tail[c].is_null() {
                    m.head[c] = p;
                } else {
                    *(m.tail[c] as *mut *mut u8) = p;
                }
                m.tail[c] = p;
                m.count[c] += 1;
                if m.count[c] as usize * size > KEEP && m.count[c] >= 2 {
                    // the older half of the queue, from its head, to the central list
                    let spill = m.count[c] / 2;
                    let (first, mut last) = (m.head[c], m.head[c]);
                    for _ in 1..spill {
                        last = *(last as *mut *mut u8);
                    }
                    m.head[c] = *(last as *mut *mut u8);
                    CENTRAL[c].give(first, last);
                    m.count[c] -= spill;
                }
            })
            .is_err()
        {
            CENTRAL[c].give(p, p);
        }
    }
    #[inline]
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = Layout::from_size_align_unchecked(new_size, layout.align());
        match (class_of(&layout), class_of(&new)) {
            (Some(a), Some(b)) if a == b => p,
            (None, None) => std::alloc::System.realloc(p, layout, new_size),
            _ => {
                let q = self.alloc(new);
                if !q.is_null() {
                    ptr::copy_nonoverlapping(p, q, layout.size().min(new_size));
                    self.dealloc(p, layout);
                }
                q
            }
        }
    }
}
