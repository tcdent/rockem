//! The runtime GPU path: the kernel's KFD interface, directly. No HIP, no
//! HSA runtime, nothing dlopened — the node owns its queue.
//!
//! What the card needs from us, in order: a process bound to the GPU's
//! virtual memory (`/dev/kfd` + the render node, `ACQUIRE_VM`); memory the
//! GPU can see, allocated through KFD and mapped at the *same* address on
//! both sides so a pointer is a pointer; one AQL queue — a ring of 64-byte
//! packets in host memory, its read/write indices, an end-of-pipe buffer and
//! a context-save area sized by the kernel's own numbers from sysfs, and a
//! doorbell page; the code object copied into executable VRAM (the ELF has
//! no relocations); and a dispatch packet per launch. Completion is a signal
//! word the command processor decrements in host memory; the host waits on
//! that cache line with MWAITX (analysis: kfd-direct decisions).
//!
//! Memory policy, one path: the weights, the cache and the tables are
//! host-visible VRAM through the resizable BAR (the host writes them once);
//! what crosses between host and card per token (residual, expert outputs,
//! logits, the quantized input, the code object) is host memory the card
//! caches, made visible by the fences — a system-scope acquire on every
//! dispatch, a system-scope release on the barrier; and the intermediates
//! neither side needs to see live in device-only VRAM. The queue's own words
//! (indices, signal, kernargs) are coherent host memory the card never
//! caches. With a small BAR the kernel refuses host-visible VRAM outright
//! ("Alloc host visible vram on small bar is not allowed"); the node says so
//! and stops. There is no copy path.

use std::cell::Cell;
use std::ffi::c_void;
use std::os::fd::AsRawFd;
use std::sync::atomic::{fence, AtomicI64, AtomicU32, AtomicU64, Ordering};

extern "C" {
    fn ioctl(fd: i32, req: u64, arg: *mut c_void) -> i32;
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
}
const PROT_NONE: i32 = 0;
const PROT_RW: i32 = 3;
const MAP_SHARED: i32 = 1;
const MAP_PRIVATE: i32 = 2;
const MAP_FIXED: i32 = 0x10;
const MAP_ANONYMOUS: i32 = 0x20;
const MAP_NORESERVE: i32 = 0x4000;
const MAP_FAILED: *mut u8 = !0usize as *mut u8;

// --- the KFD ioctl interface (include/uapi/linux/kfd_ioctl.h) ---
const fn ioc(dir: u64, nr: u64, size: usize) -> u64 {
    (dir << 30) | ((size as u64) << 16) | ((b'K' as u64) << 8) | nr
}
const W: u64 = 1;
const R: u64 = 2;
const WR: u64 = 3;

#[repr(C)]
#[derive(Default)]
struct VersionArgs {
    major: u32,
    minor: u32,
}
#[repr(C)]
#[derive(Default)]
struct CreateQueueArgs {
    ring_base_address: u64,
    write_pointer_address: u64,
    read_pointer_address: u64,
    doorbell_offset: u64,
    ring_size: u32,
    gpu_id: u32,
    queue_type: u32,
    queue_percentage: u32,
    queue_priority: u32,
    queue_id: u32,
    eop_buffer_address: u64,
    eop_buffer_size: u64,
    ctx_save_restore_address: u64,
    ctx_save_restore_size: u32,
    ctl_stack_size: u32,
    sdma_engine_id: u32,
    metadata_ring_size: u32,
}
#[repr(C)]
#[derive(Default)]
struct AllocArgs {
    va_addr: u64,
    size: u64,
    handle: u64,
    mmap_offset: u64,
    gpu_id: u32,
    flags: u32,
}
#[repr(C)]
#[derive(Default)]
struct MapArgs {
    handle: u64,
    device_ids_array_ptr: u64,
    n_devices: u32,
    n_success: u32,
}
#[repr(C)]
#[derive(Default)]
struct AcquireVmArgs {
    drm_fd: u32,
    gpu_id: u32,
}
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Apertures {
    lds_base: u64,
    lds_limit: u64,
    scratch_base: u64,
    scratch_limit: u64,
    gpuvm_base: u64,
    gpuvm_limit: u64,
    gpu_id: u32,
    pad: u32,
}
#[repr(C)]
#[derive(Default)]
struct AperturesArgs {
    ptr: u64,
    num_of_nodes: u32,
    pad: u32,
}
const IOC_GET_VERSION: u64 = ioc(R, 0x01, size_of::<VersionArgs>());
const IOC_CREATE_QUEUE: u64 = ioc(WR, 0x02, size_of::<CreateQueueArgs>());
const IOC_GET_APERTURES: u64 = ioc(WR, 0x14, size_of::<AperturesArgs>());
const IOC_ACQUIRE_VM: u64 = ioc(W, 0x15, size_of::<AcquireVmArgs>());
const IOC_ALLOC: u64 = ioc(WR, 0x16, size_of::<AllocArgs>());
const IOC_MAP: u64 = ioc(WR, 0x18, size_of::<MapArgs>());
const QUEUE_TYPE_COMPUTE_AQL: u32 = 2;
const MEM_VRAM: u32 = 1 << 0;
const MEM_GTT: u32 = 1 << 1;
const MEM_WRITABLE: u32 = 1 << 31;
const MEM_EXECUTABLE: u32 = 1 << 30;
const MEM_PUBLIC: u32 = 1 << 29;
const MEM_AQL_QUEUE: u32 = 1 << 27;
const MEM_COHERENT: u32 = 1 << 26;
const MEM_UNCACHED: u32 = 1 << 25;

// --- what the command processor expects in memory (ROCR amd_hsa_queue.h, amd_hsa_signal.h, hsa.h) ---
/// Completion lines on a queue's signal page (a cache line each): line 0 is `fence`/`wait`'s.
const FENCE_LINES: usize = 4;
/// `amd_queue_t`: the CP is told where `write_dispatch_id` and `read_dispatch_id` live.
#[repr(C, align(64))]
struct AmdQueue {
    queue_type: u32, // hsa_queue_t
    features: u32,
    base_address: u64,
    doorbell_signal: u64,
    size: u32,
    reserved1: u32,
    id: u64,
    caps: u32,
    reserved2: [u32; 3],
    write_dispatch_id: AtomicU64, // 56
    group_segment_aperture_base_hi: u32,
    private_segment_aperture_base_hi: u32,
    max_cu_id: u32,
    max_wave_id: u32,
    max_legacy_doorbell_dispatch_id_plus_1: u64,
    legacy_doorbell_lock: u32,
    reserved3: [u32; 9],
    read_dispatch_id: AtomicU64, // 128
    read_dispatch_id_field_base_byte_offset: u32,
    compute_tmpring_size: u32,
    scratch_resource_descriptor: [u32; 4],
    scratch_backing_memory_location: u64,
    reserved4: [u32; 2],
    scratch_wave64_lane_byte_size: u32,
    queue_properties: u32, // 180
    reserved5: [u32; 2],
    queue_inactive_signal: u64,
    reserved6: [u32; 14],
}
const _: () = assert!(size_of::<AmdQueue>() == 256);
const QUEUE_PROPERTIES_IS_PTR64: u32 = 1 << 1;
const QUEUE_PROPERTIES_ENABLE_PROFILING: u32 = 1 << 3;

/// `amd_signal_t`: the CP decrements `value` when the packet that names it retires,
/// and stamps `start_ts`/`end_ts` (100 MHz) when the queue profiles.
#[repr(C, align(64))]
pub struct Signal {
    kind: i64,
    value: AtomicI64,
    event_mailbox_ptr: u64,
    event_id: u32,
    reserved1: u32,
    pub start_ts: u64,
    pub end_ts: u64,
    reserved2: u64,
    reserved3: [u32; 2],
}
const _: () = assert!(size_of::<Signal>() == 64);
const SIGNAL_KIND_USER: i64 = 1;

/// The AQL kernel dispatch packet (hsa.h); the barrier-AND packet shares the
/// header and the completion signal at the end.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Packet {
    header: u16,
    setup: u16,
    workgroup_size: [u16; 3],
    reserved0: u16,
    grid_size: [u32; 3],
    private_segment_size: u32,
    group_segment_size: u32,
    kernel_object: u64,
    kernarg_address: u64,
    reserved2: u64,
    completion_signal: u64,
}
const _: () = assert!(size_of::<Packet>() == 64);
const PACKET_INVALID: u16 = 1;
const PACKET_KERNEL_DISPATCH: u16 = 2;
const PACKET_BARRIER_AND: u16 = 3;
const HEADER_BARRIER: u16 = 1 << 8;
const FENCE_AGENT: u16 = 1;
const FENCE_SYSTEM: u16 = 2;
const HEADER_ACQUIRE_SHIFT: u16 = 9;
const HEADER_RELEASE_SHIFT: u16 = 11;
const RING_PACKETS: usize = 1024;
const KERNARG_SLOT: usize = 256;
const KERNARG_SLOTS: usize = RING_PACKETS;

/// A GPU allocation: `va` is its address on both sides.
#[derive(Clone, Copy)]
pub struct Mem {
    pub va: usize,
    pub size: usize,
    /// KFD's handle, for mapping the same memory on another card (`host_shared`).
    handle: u64,
}
impl Mem {
    pub fn ptr(&self) -> *mut c_void {
        self.va as *mut c_void
    }
    pub fn at(&self, bytes: usize) -> usize {
        debug_assert!(bytes <= self.size);
        self.va + bytes
    }
}

/// A kernel of the loaded code object: its descriptor's address, which is
/// what the dispatch packet names.
#[derive(Clone, Copy)]
pub struct Kernel {
    object: u64,
    kernarg_size: u32,
    group_segment: u32,
    pub name: &'static str,
}

/// Kernel arguments laid out as the compiler expects: natural alignment, in
/// order. Pointers are 8-aligned, scalars 4.
pub struct Args {
    buf: [u8; KERNARG_SLOT],
    len: usize,
}
impl Args {
    pub fn new() -> Self {
        Args { buf: [0; KERNARG_SLOT], len: 0 }
    }
    fn put(&mut self, bytes: &[u8]) {
        let a = bytes.len();
        self.len = (self.len + a - 1) & !(a - 1);
        self.buf[self.len..self.len + a].copy_from_slice(bytes);
        self.len += a;
    }
    pub fn ptr(&mut self, va: usize) -> &mut Self {
        self.put(&(va as u64).to_le_bytes());
        self
    }
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.put(&v.to_le_bytes());
        self
    }
    pub fn f32(&mut self, v: f32) -> &mut Self {
        self.put(&v.to_le_bytes());
        self
    }
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.put(&v.to_le_bytes());
        self
    }
    pub fn f64(&mut self, v: f64) -> &mut Self {
        self.put(&v.to_le_bytes());
        self
    }
}

/// One process, one card, one queue.
pub struct Kfd {
    /// VRAM allocated through this handle so far (bytes, page-rounded): what a late allocation may still take.
    pub allocated: std::cell::Cell<usize>,
    kfd: std::fs::File,
    drm: std::fs::File,
    gpu_id: u32,
    /// This card's place among the gfx1100 cards, in topology order.
    pub index: usize,
    pub visible_vram: usize,
    /// The card's compute units: what a launch that needs its whole grid resident is sized against.
    pub cu: usize,
    ring: Mem,
    queue: Mem, // AmdQueue
    kernargs: Mem,
    signal: Mem,
    doorbell: *mut u64,
    write_index: Cell<u64>,
    /// The barrier packet each completion line's last fence went out as.
    fenced: [Cell<u64>; FENCE_LINES],
    /// When a packet was last posted or a completion seen: how long the card has been quiet.
    /// The host has written GPU-visible memory since the last launch: the next
    /// dispatch acquires at system scope (an L2 invalidate); the ones after it
    /// at agent scope, since kernels only hand each other what the card wrote.
    host_wrote: Cell<bool>,
    /// Instrumented, every dispatch carries a signal from this ring, so the
    /// command processor stamps its start and end (100 MHz): the token's
    /// timeline, per launch, with nothing in any kernel — `timeline()` after
    /// a wait. The signal costs the processor 1.7 µs a launch (measured
    /// 2026-09-19: 0.5 ms of a 7.9 ms token), so the node runs without it
    /// and the benches with it.
    instrument: bool,
    stamps: Mem,
    names: std::cell::RefCell<Vec<&'static str>>,
    grids: std::cell::RefCell<Vec<u32>>,
    /// The statistics slab (`common::stats`), when the code object has `ROCKEM_STATS`.
    stats: Option<Mem>,
    /// The card's 100 MHz clock as UTC (`utc`), fitted on first use.
    clock: std::cell::OnceCell<prow::clock::Card>,
    token_start: Cell<u64>,
    last_span: Cell<(u64, u64)>, // the launches between the previous two waits
    code: Mem,
    elf: Vec<u8>,
}

fn sysfs_u64(path: &str) -> u64 {
    std::fs::read_to_string(path).unwrap_or_default().trim().parse().unwrap_or(0)
}
fn sysfs_prop(props: &str, key: &str) -> u64 {
    props.lines().find_map(|l| l.strip_prefix(key).and_then(|r| r.trim().parse().ok())).unwrap_or(0)
}

impl Kfd {
    /// Open the first gfx1100 node, bind the process to it, build the queue,
    /// and load the code object.
    pub fn new(elf: &[u8], instrument: bool) -> std::rc::Rc<Kfd> {
        Self::open(elf, instrument, 0)
    }
    /// The gfx1100 nodes in the KFD topology, in topology order: (gpu_id, properties).
    pub fn cards() -> Vec<(u64, String)> {
        let mut dirs: Vec<_> = std::fs::read_dir("/sys/class/kfd/kfd/topology/nodes").expect("KFD topology").map(|e| e.unwrap().path()).collect();
        dirs.sort();
        dirs.into_iter()
            .filter_map(|dir| {
                let props = std::fs::read_to_string(dir.join("properties")).unwrap_or_default();
                (sysfs_prop(&props, "gfx_target_version ") == 110000).then(|| (sysfs_u64(dir.join("gpu_id").to_str().unwrap()), props))
            })
            .collect()
    }
    /// Card `index` among the gfx1100 nodes, in topology order.
    pub fn open(elf: &[u8], instrument: bool, index: usize) -> std::rc::Rc<Kfd> {
        let cards = Self::cards();
        let (gpu_id, props) = cards.get(index).cloned().unwrap_or_else(|| panic!("gfx1100 card {index}: the topology has {}", cards.len()));
        let gpu_id = gpu_id as u32;
        let render = sysfs_prop(&props, "drm_render_minor ");
        let kfd = std::fs::OpenOptions::new().read(true).write(true).open("/dev/kfd").expect("/dev/kfd");
        let drm = std::fs::OpenOptions::new().read(true).write(true).open(format!("/dev/dri/renderD{render}")).expect("render node");
        let mut v = VersionArgs::default();
        Self::call(&kfd, IOC_GET_VERSION, &mut v, "get version");
        let mut vm = AcquireVmArgs { drm_fd: drm.as_raw_fd() as u32, gpu_id };
        Self::call(&kfd, IOC_ACQUIRE_VM, &mut vm, "acquire vm");
        // apertures: ask how many nodes, then ask for them (more than the topology has is EINVAL)
        let mut aa = AperturesArgs::default();
        Self::call(&kfd, IOC_GET_APERTURES, &mut aa, "apertures count");
        let mut aps = vec![Apertures::default(); aa.num_of_nodes as usize];
        aa.ptr = aps.as_mut_ptr() as u64;
        Self::call(&kfd, IOC_GET_APERTURES, &mut aa, "apertures");
        let ap = aps[..aa.num_of_nodes as usize].iter().find(|a| a.gpu_id == gpu_id).copied().expect("apertures for the gpu");
        let visible_vram = {
            let card = format!("/sys/class/drm/renderD{render}/device/mem_info_vis_vram_total");
            sysfs_u64(&card) as usize
        };
        assert!(mwaitx_supported(), "the wait is MWAITX; this CPU has none");

        let mut k = Kfd {
            allocated: std::cell::Cell::new(0),
            kfd,
            drm,
            gpu_id,
            index,
            visible_vram,
            cu: 0,
            ring: Mem { va: 0, size: 0, handle: 0 },
            queue: Mem { va: 0, size: 0, handle: 0 },
            kernargs: Mem { va: 0, size: 0, handle: 0 },
            signal: Mem { va: 0, size: 0, handle: 0 },
            doorbell: std::ptr::null_mut(),
            write_index: Cell::new(0),
            fenced: std::array::from_fn(|_| Cell::new(0)),
            host_wrote: Cell::new(true),
            instrument,
            stamps: Mem { va: 0, size: 0, handle: 0 },
            names: std::cell::RefCell::new(vec![""; RING_PACKETS]),
            grids: std::cell::RefCell::new(vec![0; RING_PACKETS]),
            stats: None,
            clock: std::cell::OnceCell::new(),
            token_start: Cell::new(0),
            last_span: Cell::new((0, 0)),
            code: Mem { va: 0, size: 0, handle: 0 },
            elf: elf.to_vec(),
        };
        // the queue's memory: ring (doubled by KFD so the wrap is contiguous), the queue
        // struct with the indices, the kernarg slots, the wait signal, the EOP buffer in
        // VRAM, and the context-save area the kernel sized for this device
        k.ring = k.alloc(RING_PACKETS * size_of::<Packet>() * 2, MEM_GTT | MEM_AQL_QUEUE | MEM_UNCACHED | MEM_COHERENT | MEM_WRITABLE | MEM_EXECUTABLE, true, RING_PACKETS * size_of::<Packet>());
        k.queue = k.alloc(4096, MEM_GTT | MEM_COHERENT | MEM_WRITABLE, true, 4096); // one page: KFD checks the index words' page is exactly one mapping
        k.kernargs = k.alloc(KERNARG_SLOTS * KERNARG_SLOT, MEM_GTT | MEM_COHERENT | MEM_WRITABLE, true, KERNARG_SLOTS * KERNARG_SLOT);
        k.signal = k.alloc(4096, MEM_GTT | MEM_COHERENT | MEM_WRITABLE, true, 4096);
        k.stamps = k.alloc(RING_PACKETS * size_of::<Signal>(), MEM_GTT | MEM_COHERENT | MEM_WRITABLE, true, RING_PACKETS * size_of::<Signal>());
        unsafe {
            std::ptr::write_bytes(k.stamps.ptr() as *mut u8, 0, k.stamps.size);
            for i in 0..RING_PACKETS {
                (*(k.stamps.ptr() as *mut Signal).add(i)).kind = SIGNAL_KIND_USER;
            }
        }
        let eop = k.alloc(4096, MEM_VRAM | MEM_WRITABLE | MEM_EXECUTABLE, false, 4096);
        // the context-save area: the kernel's sizes for this device, and the debug
        // area the kernel adds (32 bytes per wave, 32 waves per CU on gfx11); KFD
        // requires the mapping to be exactly this big
        let cwsr_size = sysfs_prop(&props, "cwsr_size ") as usize;
        let ctl_stack = sysfs_prop(&props, "ctl_stack_size ") as u32;
        let cu = (sysfs_prop(&props, "simd_count ") / sysfs_prop(&props, "simd_per_cu ").max(1)) as usize;
        k.cu = cu;
        let debug = (cu * 32 * 32 + 63) & !63;
        let cwsr = k.alloc(cwsr_size + debug, MEM_GTT | MEM_COHERENT | MEM_WRITABLE | MEM_EXECUTABLE, true, cwsr_size + debug);
        unsafe {
            // HsaUserContextSaveAreaHeader: DebugOffset, DebugSize; the rest zero
            std::ptr::write_bytes(cwsr.ptr() as *mut u8, 0, 64);
            *(cwsr.at(16) as *mut u32) = cwsr_size as u32;
            *(cwsr.at(20) as *mut u32) = debug as u32;
            let ring = k.ring.ptr() as *mut Packet;
            for i in 0..RING_PACKETS {
                std::ptr::write_volatile(&mut (*ring.add(i)).header, PACKET_INVALID);
            }
            let q = k.queue.ptr() as *mut AmdQueue;
            std::ptr::write_bytes(q as *mut u8, 0, size_of::<AmdQueue>());
            (*q).queue_type = 1; // HSA_QUEUE_TYPE_MULTI
            (*q).features = 1; // KERNEL_DISPATCH
            (*q).base_address = k.ring.va as u64;
            (*q).size = RING_PACKETS as u32;
            (*q).group_segment_aperture_base_hi = (ap.lds_base >> 32) as u32;
            (*q).private_segment_aperture_base_hi = (ap.scratch_base >> 32) as u32;
            (*q).max_cu_id = cu as u32 - 1;
            (*q).max_wave_id = (sysfs_prop(&props, "max_waves_per_simd ") * sysfs_prop(&props, "simd_per_cu ")) as u32 - 1;
            (*q).read_dispatch_id_field_base_byte_offset = 128;
            (*q).queue_properties = QUEUE_PROPERTIES_IS_PTR64 | QUEUE_PROPERTIES_ENABLE_PROFILING;
            let s = k.signal.ptr() as *mut Signal;
            std::ptr::write_bytes(s as *mut u8, 0, size_of::<Signal>());
            (*s).kind = SIGNAL_KIND_USER;
        }
        let mut cq = CreateQueueArgs {
            ring_base_address: k.ring.va as u64,
            write_pointer_address: k.queue.at(56) as u64,
            read_pointer_address: k.queue.at(128) as u64,
            ring_size: (RING_PACKETS * size_of::<Packet>()) as u32,
            gpu_id,
            queue_type: QUEUE_TYPE_COMPUTE_AQL,
            queue_percentage: 100,
            queue_priority: 7,
            eop_buffer_address: eop.va as u64,
            eop_buffer_size: eop.size as u64,
            ctx_save_restore_address: cwsr.va as u64,
            ctx_save_restore_size: cwsr_size as u32,
            ctl_stack_size: ctl_stack,
            ..Default::default()
        };
        Self::call(&k.kfd, IOC_CREATE_QUEUE, &mut cq, "create queue");
        // the doorbell page: 1024 × 8-byte doorbells; KFD's offset names the page and the slot
        let page = 1024 * 8;
        let base = unsafe { mmap(std::ptr::null_mut(), page, PROT_RW, MAP_SHARED, k.kfd.as_raw_fd(), (cq.doorbell_offset & !(page as u64 - 1)) as i64) };
        assert!(base != MAP_FAILED, "doorbell mmap");
        k.doorbell = (base as usize + (cq.doorbell_offset & (page as u64 - 1)) as usize) as *mut u64;
        k.load();
        println!("gpu      : kfd {}.{}, gpu {gpu_id}, queue {}, {} MiB of VRAM host-visible", v.major, v.minor, cq.queue_id, visible_vram >> 20);
        std::rc::Rc::new(k)
    }

    fn call<T>(fd: &std::fs::File, req: u64, args: &mut T, what: &str) {
        let r = unsafe { ioctl(fd.as_raw_fd(), req, args as *mut T as *mut c_void) };
        assert!(r == 0, "kfd {what}: {}", std::io::Error::last_os_error());
    }

    /// Every card at its highest clocks. A token leaves a card idle between
    /// launches and the driver clocks it down, so every launch paid the climb
    /// back: on metatron a token went from 70 to 65 ms with the level forced
    /// (2026-10-15). Set after the cards are open — set before, from the init,
    /// the file read "high" but the clocks still fell.
    pub fn clocks_high() {
        let mut n = 0;
        for e in std::fs::read_dir("/sys/class/drm").into_iter().flatten().flatten() {
            let p = e.path().join("device/power_dpm_force_performance_level");
            if p.exists() && std::fs::write(&p, "high").is_ok() {
                n += 1;
            }
        }
        println!("gpu      : {n} card{} held at their highest clocks", if n == 1 { "" } else { "s" });
    }

    /// GPU memory at an address the CPU also owns. `host` maps it into this
    /// process (through the BAR for VRAM); `host_bytes` is how much of it (the
    /// ring's alias half is GPU-only).
    fn alloc(&self, size: usize, flags: u32, host: bool, host_bytes: usize) -> Mem {
        const ALIGN: usize = 2 << 20;
        let size = (size + 4095) & !4095;
        if flags & MEM_VRAM != 0 {
            self.allocated.set(self.allocated.get() + size);
        }
        if flags & MEM_PUBLIC != 0 {
            assert!(size <= self.visible_vram, "{} MiB host-visible VRAM requested, {} MiB visible: enable resizable BAR", size >> 20, self.visible_vram >> 20);
        }
        let reserve = unsafe { mmap(std::ptr::null_mut(), size + ALIGN, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0) };
        assert!(reserve != MAP_FAILED, "va reservation");
        let va = (reserve as usize + ALIGN - 1) & !(ALIGN - 1);
        let mut a = AllocArgs { va_addr: va as u64, size: size as u64, gpu_id: self.gpu_id, flags, ..Default::default() };
        Self::call(&self.kfd, IOC_ALLOC, &mut a, "alloc memory");
        if host {
            let p = unsafe { mmap(va as *mut u8, host_bytes, PROT_RW, MAP_SHARED | MAP_FIXED, self.drm.as_raw_fd(), a.mmap_offset as i64) };
            assert!(p as usize == va, "map to host: {}", std::io::Error::last_os_error());
            populate(va, host_bytes);
        }
        let ids = [self.gpu_id];
        let mut m = MapArgs { handle: a.handle, device_ids_array_ptr: ids.as_ptr() as u64, n_devices: 1, n_success: 0 };
        Self::call(&self.kfd, IOC_MAP, &mut m, "map to gpu");
        Mem { va, size, handle: a.handle }
    }
    /// VRAM the host writes (weights, cache, tables): through the BAR.
    /// VRAM not yet allocated through this handle (bytes).
    pub fn vram_free(&self) -> usize {
        self.visible_vram.saturating_sub(self.allocated.get())
    }
    pub fn vram_visible(&self, size: usize) -> Mem {
        self.alloc(size, MEM_VRAM | MEM_WRITABLE | MEM_PUBLIC, true, size)
    }
    /// VRAM only the card touches (intermediates).
    pub fn vram(&self, size: usize) -> Mem {
        self.alloc(size, MEM_VRAM | MEM_WRITABLE, false, size)
    }
    /// Host memory both sides touch per token, cached by the card; the
    /// fences make each side's writes visible to the other at the boundary.
    pub fn host(&self, size: usize) -> Mem {
        self.alloc(size, MEM_GTT | MEM_WRITABLE, true, size)
    }
    /// Host memory the card reads uncached: what a kernel polls or reads
    /// while the host is still writing (the host's expert slots and their
    /// valid words), every load a trip over the link.
    pub fn host_coherent(&self, size: usize) -> Mem {
        self.alloc(size, MEM_GTT | MEM_COHERENT | MEM_WRITABLE, true, size)
    }
    /// VRAM of this card every card of `others` sees too (their writes come over the link as peer
    /// writes): the receiver's side of a card-to-card handoff (docs/queue/001). Fine-grained
    /// (`MEM_COHERENT`: gfx11 maps it MTYPE UC), so this card's loads of what a peer wrote bypass its
    /// L2; host-visible, so the host seeds and reads it. The peer map failing is the test of peer access.
    #[allow(dead_code)] // the node keeps its handoffs in host memory (docs/queue/001); bench-p2p uses it
    pub fn vram_shared(&self, size: usize, others: &[&Kfd]) -> Mem {
        let m = self.alloc(size, MEM_VRAM | MEM_COHERENT | MEM_WRITABLE | MEM_PUBLIC, true, size);
        for o in others {
            let ids = [o.gpu_id];
            let mut a = MapArgs { handle: m.handle, device_ids_array_ptr: ids.as_ptr() as u64, n_devices: 1, n_success: 0 };
            Self::call(&self.kfd, IOC_MAP, &mut a, "map vram to the other gpu");
        }
        m
    }
    /// `host_coherent` memory every card of `others` sees too, at the same address: the
    /// cards' handoffs (block outputs a card produces and another consumes, and the words
    /// that say they are there). One process holds every card, so KFD maps one handle on each.
    pub fn host_shared(&self, size: usize, others: &[&Kfd]) -> Mem {
        let m = self.host_coherent(size);
        for o in others {
            let ids = [o.gpu_id];
            let mut a = MapArgs { handle: m.handle, device_ids_array_ptr: ids.as_ptr() as u64, n_devices: 1, n_success: 0 };
            Self::call(&self.kfd, IOC_MAP, &mut a, "map to the other gpu");
        }
        m
    }
    /// Host → card through the BAR, streamed (`common::copy`): a weight
    /// upload, a context swap-in.
    pub fn upload(&self, dst: &Mem, off: usize, bytes: &[u8]) {
        assert!(off + bytes.len() <= dst.size);
        common::copy::stream(unsafe { std::slice::from_raw_parts_mut(dst.at(off) as *mut u8, bytes.len()) }, bytes);
        self.host_wrote.set(true);
    }

    /// The code object into executable host memory: LOAD segments at their
    /// virtual addresses (no relocations), the rest zero. 48 KB read through
    /// the instruction cache; VRAM would buy nothing.
    fn load(&mut self) {
        let e = &self.elf;
        let rd = |o: usize, n: usize| -> u64 {
            let mut v = 0u64;
            for i in 0..n {
                v |= (e[o + i] as u64) << (8 * i);
            }
            v
        };
        let (phoff, phnum) = (rd(0x20, 8) as usize, rd(0x38, 2) as usize);
        let mut top = 0;
        for i in 0..phnum {
            let p = phoff + i * 56;
            if rd(p, 4) == 1 {
                top = top.max(rd(p + 16, 8) as usize + rd(p + 40, 8) as usize);
            }
        }
        // the code object in host-visible VRAM: the card fetches instructions from its own
        // memory. In host memory (the first placement, "VRAM would buy nothing" at 48 KB of
        // small kernels) every launch of the larger n-token kernels refetched its code over
        // the link after a token's weights had streamed through L2 — measured 2026-09-19 as
        // 8–14 µs of dead time inside every launch, 140 → 89 tok/s
        let code = self.alloc((top + 4095) & !4095, MEM_VRAM | MEM_WRITABLE | MEM_EXECUTABLE | MEM_PUBLIC, true, (top + 4095) & !4095);
        unsafe { std::ptr::write_bytes(code.ptr() as *mut u8, 0, code.size) };
        for i in 0..phnum {
            let p = phoff + i * 56;
            if rd(p, 4) == 1 {
                let (off, va, filesz) = (rd(p + 8, 8) as usize, rd(p + 16, 8) as usize, rd(p + 32, 8) as usize);
                self.upload(&code, va, &e[off..off + filesz]);
            }
        }
        self.code = code;
        // relocations: the release object has none; a stats object has one absolute
        // address per static it reaches through the GOT (R_AMDGPU_ABS64 = 3)
        let (shoff, shnum) = (rd(0x28, 8) as usize, rd(0x3c, 2) as usize);
        for i in 0..shnum {
            let sh = shoff + i * 64;
            if rd(sh + 4, 4) != 4 {
                continue; // SHT_RELA
            }
            let (off, size, link) = (rd(sh + 24, 8) as usize, rd(sh + 32, 8) as usize, rd(sh + 40, 4) as usize);
            let symtab = shoff + link * 64;
            let symoff = rd(symtab + 24, 8) as usize;
            for r in 0..size / 24 {
                let e0 = off + r * 24;
                let (r_offset, r_info, r_addend) = (rd(e0, 8) as usize, rd(e0 + 8, 8), rd(e0 + 16, 8) as i64);
                let (sym, kind) = ((r_info >> 32) as usize, (r_info & 0xffff_ffff) as u32);
                assert!(kind == 3, "relocation type {kind} at {r_offset:#x}: only R_AMDGPU_ABS64 is applied");
                let sym_value = rd(symoff + sym * 24 + 8, 8) as usize;
                let value = (self.code.va + sym_value) as i64 + r_addend;
                unsafe { *(self.code.at(r_offset) as *mut u64) = value as u64 };
            }
        }
        // a code object built with stats has the slab pointer as a symbol: fill it in
        if let Some(va) = self.symbol("ROCKEM_STATS") {
            use common::stats::{STAT_RING, STAT_SLOTS, STAT_WAVES};
            // in the card's VRAM: a kernel's stamps must not cross the link (host memory would add the very
            // PCIe writes a phase is being timed against); the host reads it back streamed
            let slab = self.vram_visible(STAT_RING * STAT_SLOTS * STAT_WAVES * 4);
            unsafe { *(self.code.at(va) as *mut u64) = slab.va as u64 };
            self.stats = Some(slab);
            println!("gpu      : kernel statistics on ({} MB slab)", slab.size >> 20);
        }
    }
    /// A symbol's address in the code object, if it has one.
    fn symbol(&self, want: &str) -> Option<usize> {
        let e = &self.elf;
        let rd = |o: usize, n: usize| -> u64 {
            let mut v = 0u64;
            for i in 0..n {
                v |= (e[o + i] as u64) << (8 * i);
            }
            v
        };
        let (shoff, shnum) = (rd(0x28, 8) as usize, rd(0x3c, 2) as usize);
        let sh = |i: usize| shoff + i * 64;
        let symtab = (0..shnum).find(|&i| rd(sh(i) + 4, 4) == 2).or_else(|| (0..shnum).find(|&i| rd(sh(i) + 4, 4) == 11))?;
        let strtab = sh(rd(sh(symtab) + 40, 4) as usize);
        let (symoff, symsize, stroff) = (rd(sh(symtab) + 24, 8) as usize, rd(sh(symtab) + 32, 8) as usize, rd(strtab + 24, 8) as usize);
        (0..symsize / 24).find_map(|i| {
            let s = symoff + i * 24;
            let n0 = stroff + rd(s, 4) as usize;
            let n1 = n0 + e[n0..].iter().position(|&b| b == 0).unwrap();
            (&e[n0..n1] == want.as_bytes()).then(|| rd(s + 8, 8) as usize)
        })
    }

    /// A kernel by name: its descriptor (`name.kd`) in the loaded image.
    pub fn kernel(&self, name: &str) -> Kernel {
        let e = &self.elf;
        let rd = |o: usize, n: usize| -> u64 {
            let mut v = 0u64;
            for i in 0..n {
                v |= (e[o + i] as u64) << (8 * i);
            }
            v
        };
        let (shoff, shnum) = (rd(0x28, 8) as usize, rd(0x3c, 2) as usize);
        let sh = |i: usize| shoff + i * 64;
        let symtab = (0..shnum).find(|&i| rd(sh(i) + 4, 4) == 2).or_else(|| (0..shnum).find(|&i| rd(sh(i) + 4, 4) == 11)).expect("symtab");
        let strtab = sh(rd(sh(symtab) + 40, 4) as usize);
        let (symoff, symsize, stroff) = (rd(sh(symtab) + 24, 8) as usize, rd(sh(symtab) + 32, 8) as usize, rd(strtab + 24, 8) as usize);
        let want = format!("{name}.kd");
        for i in 0..symsize / 24 {
            let s = symoff + i * 24;
            let n0 = stroff + rd(s, 4) as usize;
            let n1 = n0 + e[n0..].iter().position(|&b| b == 0).unwrap();
            if &e[n0..n1] == want.as_bytes() {
                let va = rd(s + 8, 8) as usize;
                let kd = self.code.at(va);
                let (group, private, kernarg) = unsafe { (*(kd as *const u32), *((kd + 4) as *const u32), *((kd + 8) as *const u32)) };
                assert!(private == 0, "{name}: private segment (scratch) is not supported");
                return Kernel { object: kd as u64, kernarg_size: kernarg, group_segment: group, name: Box::leak(name.to_string().into_boxed_str()) };
            }
        }
        panic!("kernel {name} not in the code object");
    }

    /// One dispatch: `grid` workgroups of `wg` items, `lds` dynamic bytes,
    /// launched after everything before it has completed (the barrier bit);
    /// nothing waits here.
    pub fn launch(&self, k: &Kernel, grid: u32, wg: u32, lds: u32, args: &Args) {
        self.dispatch(k, grid, wg, lds, args, true)
    }
    /// The same, launched as soon as the dispatch before it has launched: it
    /// may run concurrently with that one, so its inputs must already have
    /// been complete when that one launched (a barrier of its own before it).
    pub fn launch_free(&self, k: &Kernel, grid: u32, wg: u32, lds: u32, args: &Args) {
        self.dispatch(k, grid, wg, lds, args, false)
    }
    fn dispatch(&self, k: &Kernel, grid: u32, wg: u32, lds: u32, args: &Args, barrier: bool) {
        let idx = self.write_index.get();
        let (pkt, header) = self.packet(idx, k, grid, wg, lds, args, barrier);
        self.post(idx, pkt, header);
    }
    /// The dispatch packet and header for `idx`: kernargs and stamp in the slot, the fences by what the host wrote.
    fn packet(&self, idx: u64, k: &Kernel, grid: u32, wg: u32, lds: u32, args: &Args, barrier: bool) -> (Packet, u16) {
        assert_eq!(args.len as u32, k.kernarg_size, "kernarg size");
        let slot = (idx as usize) % RING_PACKETS;
        let ka = self.kernargs.at(slot * KERNARG_SLOT);
        unsafe { std::ptr::copy_nonoverlapping(args.buf.as_ptr(), ka as *mut u8, args.len) };
        let stamp = unsafe { &*(self.stamps.ptr() as *const Signal).add(slot) };
        stamp.value.store(1, Ordering::Relaxed);
        self.names.borrow_mut()[slot] = k.name;
        self.grids.borrow_mut()[slot] = grid;
        let pkt = Packet {
            header: PACKET_INVALID,
            setup: 1,
            workgroup_size: [wg as u16, 1, 1],
            reserved0: 0,
            grid_size: [grid * wg, 1, 1],
            private_segment_size: 0,
            group_segment_size: lds + k.group_segment,
            kernel_object: k.object,
            kernarg_address: ka as u64,
            reserved2: 0,
            completion_signal: if self.instrument { self.stamps.at(slot * size_of::<Signal>()) as u64 } else { 0 },
        };
        // acquire at system scope only when the host wrote since the last launch (an L2
        // invalidate, once per token in practice); agent scope between kernels — the
        // agent acquire (L0/GL1/K$ invalidates) costs the processor 1.1 µs a launch, and
        // without it the step gate fails on stale reads (2026-09-19). Release at agent
        // scope always: the barrier's system-scope release publishes to the host.
        let acquire = if self.host_wrote.replace(false) { FENCE_SYSTEM } else { FENCE_AGENT };
        (pkt, PACKET_KERNEL_DISPATCH | (acquire << HEADER_ACQUIRE_SHIFT) | (FENCE_AGENT << HEADER_RELEASE_SHIFT) | if barrier { HEADER_BARRIER } else { 0 })
    }

    /// Wait for everything posted so far: a barrier packet carrying the
    /// signal, then MWAITX on the signal's line until the CP decrements it.
    pub fn sync(&self) {
        self.fence();
        self.wait();
    }
    /// Post the completion of everything so far — a barrier packet carrying
    /// the signal — without waiting: the CP decrements the signal the moment
    /// the work is done, while it is still awake, and `wait` later finds it
    /// decremented. A card left idle 400 µs pays ~100 µs on its next packet
    /// (probe, 2026-10-15), so the packet that completes a launch sequence
    /// goes out with it. One completion outstanding at a time.
    pub fn fence(&self) {
        self.fence_at(0);
    }
    /// Wait for the completion `fence` posted.
    pub fn wait(&self) {
        self.wait_at(0);
    }
    /// `fence` and `wait` on completion line `line` (1..FENCE_LINES) instead of the queue's own: a sequence
    /// whose completion is waited for later, while other fences go out and are waited for in between, takes
    /// a line of its own — a line holds one completion outstanding (a second arms the signal again and the
    /// counts cross: the wait never ends).
    pub fn fence_line(&self, line: usize) {
        assert!(line > 0 && line < FENCE_LINES);
        self.fence_at(line * 64);
    }
    pub fn wait_line(&self, line: usize) {
        assert!(line > 0 && line < FENCE_LINES);
        self.wait_at(line * 64);
    }
    /// The signal at byte `off` of the signal page (its own cache line), armed and carried by a barrier packet.
    fn fence_at(&self, off: usize) {
        let s = self.signal.at(off) as *const Signal;
        unsafe { (*s).value.store(1, Ordering::Release) };
        let idx = self.write_index.get();
        // the slot's stamp and name are a launch's: cleared so a timeline skips this packet
        let slot = (idx as usize) % RING_PACKETS;
        self.names.borrow_mut()[slot] = "";
        unsafe {
            let st = &mut *(self.stamps.ptr() as *mut Signal).add(slot);
            st.start_ts = 0;
            st.end_ts = 0;
        }
        let pkt = Packet { header: PACKET_INVALID, setup: 0, workgroup_size: [0; 3], reserved0: 0, grid_size: [0; 3], private_segment_size: 0, group_segment_size: 0, kernel_object: 0, kernarg_address: 0, reserved2: 0, completion_signal: self.signal.at(off) as u64 };
        self.post(idx, pkt, PACKET_BARRIER_AND | HEADER_BARRIER | (FENCE_SYSTEM << HEADER_ACQUIRE_SHIFT) | (FENCE_SYSTEM << HEADER_RELEASE_SHIFT));
        self.fenced[off / 64].set(idx);
    }
    fn wait_at(&self, off: usize) {
        let s = self.signal.at(off) as *const Signal;
        let idx = self.fenced[off / 64].get();
        let value = unsafe { &(*s).value };
        let t0 = std::time::Instant::now();
        while value.load(Ordering::Acquire) != 0 {
            unsafe { wait_on(value) };
            if t0.elapsed().as_secs() >= 10 {
                // the packets from the read index on: the one the queue stopped at first (a fence has no name)
                let read = unsafe { (*(self.queue.ptr() as *const AmdQueue)).read_dispatch_id.load(Ordering::Relaxed) };
                let names = self.names.borrow();
                let at: Vec<&str> = (read.min(idx)..=idx).map(|i| names[(i as usize) % RING_PACKETS]).collect();
                panic!("gpu hang: gpu {} packet {idx} never completed (read index {read}, write index {}, {} launched; from the read index: {at:?})", self.gpu_id, unsafe { (*(self.queue.ptr() as *const AmdQueue)).write_dispatch_id.load(Ordering::Relaxed) }, self.write_index.get());
            }
        }
        self.host_wrote.set(true); // the host reads and writes freely until the next launch
        if off == 0 {
            self.last_span.set((self.token_start.get(), idx));
            self.token_start.set(idx + 1);
        }
    }
    /// The last token's statistics, per launch in order: kernel name, then
    /// per slot the values every wave wrote (`common::gfx11` stats; empty
    /// without a stats code object). Valid after `sync`, until the ring wraps.
    pub fn stats(&self) -> Vec<(&'static str, [Vec<f32>; 4])> {
        use common::stats::{STAT_RING, STAT_SLOTS, STAT_WAVES, STAT_WAVES_PER_WG};
        let Some(slab) = &self.stats else { return Vec::new() };
        let (from, to) = self.last_span.get();
        let (names, grids) = (self.names.borrow(), self.grids.borrow());
        (from..to)
            .map(|i| {
                let slot = (i as usize) % RING_PACKETS;
                let block = (i as usize) % STAT_RING;
                let waves = ((grids[slot] * STAT_WAVES_PER_WG) as usize).min(STAT_WAVES);
                let read = |s: usize| unsafe { std::slice::from_raw_parts(slab.at(((block * STAT_SLOTS + s) * STAT_WAVES) * 4) as *const f32, waves).to_vec() };
                (names[slot], [read(0), read(1), read(2), read(3)])
            })
            .collect()
    }
    /// The statistics of the launches from packet `from` to the last posted whose kernel is in `kernels`: the
    /// packet index, the kernel's name, and per slot the values every wave wrote. Empty without a stats
    /// code object. Valid after `sync`, until the ring wraps.
    pub fn stats_from(&self, from: u64, kernels: &[&str]) -> Vec<(u64, &'static str, [Vec<f32>; 4])> {
        use common::stats::{STAT_RING, STAT_SLOTS, STAT_WAVES, STAT_WAVES_PER_WG};
        let Some(slab) = &self.stats else { return Vec::new() };
        let to = self.write_index.get();
        let from = from.max(to.saturating_sub(STAT_RING.min(RING_PACKETS) as u64));
        let (names, grids) = (self.names.borrow(), self.grids.borrow());
        (from..to)
            .filter(|&i| kernels.contains(&names[(i as usize) % RING_PACKETS]))
            .map(|i| {
                let slot = (i as usize) % RING_PACKETS;
                let block = (i as usize) % STAT_RING;
                let waves = ((grids[slot] * STAT_WAVES_PER_WG) as usize).min(STAT_WAVES);
                let rd = |s: usize| {
                    let mut v = vec![0f32; waves];
                    read(slab, ((block * STAT_SLOTS + s) * STAT_WAVES) * 4, &mut v);
                    v
                };
                (i, names[slot], [rd(0), rd(1), rd(2), rd(3)])
            })
            .collect()
    }
    /// The launches since the previous wait, in order: name, start and end
    /// on the card's 100 MHz clock. Valid after `sync`, until the ring wraps.
    pub fn timeline(&self) -> Vec<(&'static str, u64, u64)> {
        assert!(self.instrument, "timeline: the launcher was opened without instrumentation");
        let (from, to) = self.last_span.get();
        assert!(to - from <= RING_PACKETS as u64, "timeline: more launches than the ring holds");
        let names = self.names.borrow();
        (from..to)
            .map(|i| {
                let slot = (i as usize) % RING_PACKETS;
                let s = unsafe { &*(self.stamps.ptr() as *const Signal).add(slot) };
                (names[slot], s.start_ts, s.end_ts)
            })
            .collect()
    }
    /// The launches from packet `from` to the last posted, in order: name, start and end on the
    /// card's 100 MHz clock (a barrier packet's name is ""). Valid after `sync`; past the ring's
    /// depth (a step of hundreds of tokens) the earliest are gone and the last `RING_PACKETS` come.
    pub fn timeline_from(&self, from: u64) -> Vec<(&'static str, u64, u64)> {
        assert!(self.instrument, "timeline: the launcher was opened without instrumentation");
        let to = self.write_index.get();
        let from = from.max(to.saturating_sub(RING_PACKETS as u64));
        let names = self.names.borrow();
        (from..to)
            .map(|i| {
                let slot = (i as usize) % RING_PACKETS;
                let s = unsafe { &*(self.stamps.ptr() as *const Signal).add(slot) };
                (names[slot], s.start_ts, s.end_ts)
            })
            .collect()
    }
    /// A stamp of the card's 100 MHz clock (a timeline's start or end) in UTC.
    pub fn utc(&self, ticks: u64) -> prow::Timestamp {
        use std::os::fd::AsFd;
        self.clock.get_or_init(|| prow::clock::Card::new(self.kfd.as_fd(), self.gpu_id).expect("the card's clock counters")).at(ticks)
    }
    /// Whether the launches are stamped (`open`'s `instrument`).
    pub fn instrumented(&self) -> bool {
        self.instrument
    }
    /// The index the next launch takes.
    pub fn posted(&self) -> u64 {
        self.write_index.get()
    }
    /// Device timestamps of the last `sync`'s barrier packet, 100 MHz.
    #[allow(dead_code)]
    pub fn last_ts(&self) -> (u64, u64) {
        let s = self.signal.ptr() as *const Signal;
        unsafe { ((*s).start_ts, (*s).end_ts) }
    }

    fn post(&self, idx: u64, pkt: Packet, header: u16) {
        let ring = self.ring.ptr() as *mut Packet;
        let slot = unsafe { ring.add((idx as usize) % RING_PACKETS) };
        let q = self.queue.ptr() as *const AmdQueue;
        // the ring is full when the CP has not retired the packet this slot held
        while idx - unsafe { (*q).read_dispatch_id.load(Ordering::Acquire) } >= RING_PACKETS as u64 {
            std::hint::spin_loop();
        }
        unsafe {
            let mut p = pkt;
            p.header = PACKET_INVALID;
            std::ptr::write_volatile(slot, p);
            fence(Ordering::SeqCst);
            let hs = slot as *const AtomicU32;
            (*hs).store((pkt.setup as u32) << 16 | header as u32, Ordering::Release);
            (*q).write_dispatch_id.store(idx + 1, Ordering::Release);
            fence(Ordering::SeqCst);
            std::ptr::write_volatile(self.doorbell, idx);
        }
        self.write_index.set(idx + 1);
    }
}

/// Fault a host mapping in before it is used. The driver maps 16 pages per
/// fault and ignores MAP_POPULATE (and refuses MADV_POPULATE) on its
/// mappings, so a window touched for the first time inside a copy runs at
/// the fault rate, 6 GB/s, instead of the link's 22. Measured: one thread
/// populates at 7.7 GB/s, two at 14, more no faster (`bench-copy`). Every
/// window is populated once here, and every copy after that is at the link.
/// Assumed mean: four threads, from lasenora's saturation at two; a box with
/// more channels may keep scaling and want more.
fn populate(va: usize, bytes: usize) {
    let threads = if bytes >= 8 << 20 { 4 } else { 1 };
    let chunk = (bytes / threads + 4095) & !4095;
    std::thread::scope(|sc| {
        for t in 0..threads {
            sc.spawn(move || {
                let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(bytes));
                let mut off = lo;
                while off < hi {
                    unsafe { std::ptr::write_volatile((va + off) as *mut u8, 0) };
                    off += 4096;
                }
            });
        }
    });
}

fn mwaitx_supported() -> bool {
    let r = core::arch::x86_64::__cpuid(0x8000_0001);
    r.ecx & (1 << 29) != 0
}

/// Sleep the core until a store lands on the line holding `value`, or ~30 µs
/// pass. Arm the monitor, re-check (a store between the check and the wait
/// would otherwise be missed), then wait.
#[inline(always)]
/// Wait until the word at `w` (host memory a card writes with `hc_signal`) reads `value`:
/// MWAITX on its line, as the completion signals are waited for.
pub fn wait_word(w: *const AtomicU32, value: u32) {
    let w = unsafe { &*w };
    let t0 = std::time::Instant::now();
    while w.load(Ordering::Acquire) != value {
        assert!(t0.elapsed().as_secs() < 10, "handoff hang: the word at {:#x} reads {}, {value} expected", w as *const AtomicU32 as usize, w.load(Ordering::Relaxed));
        unsafe {
            core::arch::asm!("monitorx", in("rax") w as *const AtomicU32, in("ecx") 0u32, in("edx") 0u32, options(nostack, preserves_flags));
            if w.load(Ordering::Acquire) != value {
                core::arch::asm!("push rbx", "mov ebx, {t:e}", "mwaitx", "pop rbx", t = in(reg) 100_000u32, in("eax") 0u32, in("ecx") 2u32, options(nostack, preserves_flags));
            }
        }
    }
}

unsafe fn wait_on(value: &AtomicI64) {
    core::arch::asm!("monitorx", in("rax") value as *const AtomicI64, in("ecx") 0u32, in("edx") 0u32, options(nostack, preserves_flags));
    if value.load(Ordering::Acquire) != 0 {
        // eax = C-state hint (C1), ecx bit 1 = timeout enabled, ebx = timeout in cycles
        core::arch::asm!("push rbx", "mov ebx, {t:e}", "mwaitx", "pop rbx", t = in(reg) 100_000u32, in("eax") 0u32, in("ecx") 2u32, options(nostack, preserves_flags));
    }
}

// --- shared by the models' drivers on this launcher ---

pub(crate) const WG: u32 = 256;
pub fn raw<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, size_of_val(v)) }
}
pub fn raw_u32(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, 4 * v.len()) }
}
/// The host's read of a card's host-memory buffer (write-combining for the CPU): streamed.
pub fn read<T: Copy>(m: &Mem, off: usize, out: &mut [T]) {
    let bytes = out.len() * size_of::<T>();
    assert!(off + bytes <= m.size);
    let src = unsafe { std::slice::from_raw_parts(m.at(off) as *const u8, bytes) };
    let dst = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, bytes) };
    common::copy::stream_read(dst, src);
}
