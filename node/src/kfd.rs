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
/// The AQL barrier-AND packet: retires when every `dep_signal` reads zero.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct BarrierPacket {
    header: u16,
    reserved0: u16,
    reserved1: u32,
    dep_signal: [u64; 5],
    reserved2: u64,
    completion_signal: u64,
}
const _: () = assert!(size_of::<BarrierPacket>() == 64);
/// Signals in the signal page: 0 is the wait's, the rest a host's to name.
pub const SIGNALS: usize = 64;
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
    kfd: std::fs::File,
    drm: std::fs::File,
    gpu_id: u32,
    pub visible_vram: usize,
    ring: Mem,
    queue: Mem, // AmdQueue
    kernargs: Mem,
    signal: Mem,
    doorbell: *mut u64,
    write_index: Cell<u64>,
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
        let nodes = "/sys/class/kfd/kfd/topology/nodes";
        let mut found = None;
        for e in std::fs::read_dir(nodes).expect("KFD topology") {
            let dir = e.unwrap().path();
            let props = std::fs::read_to_string(dir.join("properties")).unwrap_or_default();
            if sysfs_prop(&props, "gfx_target_version ") == 110000 {
                found = Some((sysfs_u64(dir.join("gpu_id").to_str().unwrap()), props));
                break;
            }
        }
        let (gpu_id, props) = found.expect("no gfx1100 node in the KFD topology");
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
            kfd,
            drm,
            gpu_id,
            visible_vram,
            ring: Mem { va: 0, size: 0 },
            queue: Mem { va: 0, size: 0 },
            kernargs: Mem { va: 0, size: 0 },
            signal: Mem { va: 0, size: 0 },
            doorbell: std::ptr::null_mut(),
            write_index: Cell::new(0),
            host_wrote: Cell::new(true),
            instrument,
            stamps: Mem { va: 0, size: 0 },
            names: std::cell::RefCell::new(vec![""; RING_PACKETS]),
            grids: std::cell::RefCell::new(vec![0; RING_PACKETS]),
            stats: None,
            token_start: Cell::new(0),
            last_span: Cell::new((0, 0)),
            code: Mem { va: 0, size: 0 },
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
            std::ptr::write_bytes(s as *mut u8, 0, SIGNALS * size_of::<Signal>());
            for i in 0..SIGNALS {
                (*s.add(i)).kind = SIGNAL_KIND_USER;
            }
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

    /// GPU memory at an address the CPU also owns. `host` maps it into this
    /// process (through the BAR for VRAM); `host_bytes` is how much of it (the
    /// ring's alias half is GPU-only).
    fn alloc(&self, size: usize, flags: u32, host: bool, host_bytes: usize) -> Mem {
        const ALIGN: usize = 2 << 20;
        let size = (size + 4095) & !4095;
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
        Mem { va, size }
    }
    /// VRAM the host writes (weights, cache, tables): through the BAR.
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
            let slab = self.host(STAT_RING * STAT_SLOTS * STAT_WAVES * 4);
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
        assert_eq!(args.len as u32, k.kernarg_size, "kernarg size");
        let idx = self.write_index.get();
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
        self.post(idx, pkt, PACKET_KERNEL_DISPATCH | (acquire << HEADER_ACQUIRE_SHIFT) | (FENCE_AGENT << HEADER_RELEASE_SHIFT) | if barrier { HEADER_BARRIER } else { 0 });
    }

    /// Wait for everything posted so far: a barrier packet carrying the
    /// signal, then MWAITX on the signal's line until the CP decrements it.
    pub fn sync(&self) {
        let s = self.signal.ptr() as *const Signal;
        unsafe { (*s).value.store(1, Ordering::Release) };
        let idx = self.write_index.get();
        let pkt = Packet { header: PACKET_INVALID, setup: 0, workgroup_size: [0; 3], reserved0: 0, grid_size: [0; 3], private_segment_size: 0, group_segment_size: 0, kernel_object: 0, kernarg_address: 0, reserved2: 0, completion_signal: self.signal.va as u64 };
        self.post(idx, pkt, PACKET_BARRIER_AND | HEADER_BARRIER | (FENCE_SYSTEM << HEADER_ACQUIRE_SHIFT) | (FENCE_SYSTEM << HEADER_RELEASE_SHIFT));
        let value = unsafe { &(*s).value };
        let t0 = std::time::Instant::now();
        while value.load(Ordering::Acquire) != 0 {
            unsafe { wait_on(value) };
            assert!(t0.elapsed().as_secs() < 10, "gpu hang: packet {idx} never completed (read index {})", unsafe { (*(self.queue.ptr() as *const AmdQueue)).read_dispatch_id.load(Ordering::Relaxed) });
        }
        self.host_wrote.set(true); // the host reads and writes freely until the next launch
        self.last_span.set((self.token_start.get(), idx));
        self.token_start.set(idx + 1);
    }
    fn sig(&self, i: usize) -> &Signal {
        assert!(i > 0 && i < SIGNALS, "signal {i}");
        unsafe { &*(self.signal.ptr() as *const Signal).add(i) }
    }
    /// A barrier packet with its ring slot's signal as completion, so the
    /// processor stamps it like a launch: on the timeline a mark's span is
    /// its own cost and a gate's span is the wait behind it. Returns the packet's index.
    fn post_barrier(&self, mut pkt: BarrierPacket, header: u16, name: &'static str) -> u64 {
        let idx = self.write_index.get();
        let slot = (idx as usize) % RING_PACKETS;
        let stamp = unsafe { &*(self.stamps.ptr() as *const Signal).add(slot) };
        stamp.value.store(1, Ordering::Release);
        self.names.borrow_mut()[slot] = name;
        self.grids.borrow_mut()[slot] = 0;
        pkt.completion_signal = self.stamps.at(slot * size_of::<Signal>()) as u64;
        // the same 64 bytes as a dispatch packet; `post` writes the header word last
        self.post(idx, unsafe { std::mem::transmute::<BarrierPacket, Packet>(pkt) }, header);
        idx
    }
    /// A mark in the queue: its signal reads zero once everything posted
    /// before it has completed and its writes are in memory (a system-scope
    /// release) — `wait(mark)` on the host. What lets the host pick up a
    /// layer's routing and integers while the card runs on.
    pub fn mark(&self) -> u64 {
        let pkt = BarrierPacket { header: PACKET_INVALID, reserved0: 0, reserved1: 0, dep_signal: [0; 5], reserved2: 0, completion_signal: 0 };
        self.post_barrier(pkt, PACKET_BARRIER_AND | HEADER_BARRIER | (FENCE_SYSTEM << HEADER_RELEASE_SHIFT), "mark")
    }
    /// The host waits for a mark (its packet index; the ring must not have
    /// wrapped since, which a token's worth of packets never does).
    pub fn wait(&self, mark: u64) {
        assert!(self.write_index.get() - mark < RING_PACKETS as u64, "wait: the ring has wrapped past mark {mark}");
        let value = unsafe { &(*(self.stamps.ptr() as *const Signal).add((mark as usize) % RING_PACKETS)).value };
        let t0 = std::time::Instant::now();
        while value.load(Ordering::Acquire) != 0 {
            unsafe { wait_on(value) };
            assert!(t0.elapsed().as_secs() < 10, "gpu hang: mark {mark} never completed");
        }
    }
    /// A gate in the queue on signal `i`: nothing posted after it runs until
    /// the host `release(i)`s. The launch after the gate acquires at system
    /// scope, since the host will have written by then.
    pub fn gate(&self, i: usize) {
        self.sig(i).value.store(1, Ordering::Release);
        let pkt = BarrierPacket { header: PACKET_INVALID, reserved0: 0, reserved1: 0, dep_signal: [self.signal.at(i * size_of::<Signal>()) as u64, 0, 0, 0, 0], reserved2: 0, completion_signal: 0 };
        self.post_barrier(pkt, PACKET_BARRIER_AND | HEADER_BARRIER, "gate");
        self.host_wrote.set(true);
    }
    /// The host opens gate `i`: its writes before this are what the card sees after.
    pub fn release(&self, i: usize) {
        fence(Ordering::SeqCst);
        self.sig(i).value.store(0, Ordering::Release);
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
unsafe fn wait_on(value: &AtomicI64) {
    core::arch::asm!("monitorx", in("rax") value as *const AtomicI64, in("ecx") 0u32, in("edx") 0u32, options(nostack, preserves_flags));
    if value.load(Ordering::Acquire) != 0 {
        // eax = C-state hint (C1), ecx bit 1 = timeout enabled, ebx = timeout in cycles
        core::arch::asm!("push rbx", "mov ebx, {t:e}", "mwaitx", "pop rbx", t = in(reg) 100_000u32, in("eax") 0u32, in("ecx") 2u32, options(nostack, preserves_flags));
    }
}

// --- the model on this launcher: the weights in host-visible VRAM, the
// intermediates in device VRAM, and what crosses per token in host memory ---

use common::q4::GROUP;
use common::trace::Trace;
use deepseek_v2_lite::layout::{self, dense as dl};
use deepseek_v2_lite::{attn, ffn, Attention, FeedForward, Head, Model, Rope, Sampling, D_FF, H, HEADS, INTERMEDIATE_SIZE, KVA_DIM, KV_ROW, NUM_HIDDEN_LAYERS, N_ROUTED_EXPERTS, O_IN_DIM, QK_ROPE_HEAD_DIM, Q_DIM, SHARED_FF, TOKENS, TOP_K, VOCAB_SIZE};
use std::rc::Rc;

pub(crate) const WG: u32 = 256;
fn raw<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, size_of_val(v)) }
}
fn raw_u32(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, 4 * v.len()) }
}
pub fn read<T: Copy>(m: &Mem, off: usize, out: &mut [T]) {
    unsafe { std::ptr::copy_nonoverlapping(m.at(off) as *const T, out.as_mut_ptr(), out.len()) }
}

impl Kfd {
    /// The bundle's dense file, resident: every realization indexes it by `layout::dense` offsets.
    pub fn upload_dense(&self, dense: &[u16]) -> Mem {
        let t0 = std::time::Instant::now();
        let d = self.vram_visible(dense.len() * 2);
        let t1 = std::time::Instant::now();
        self.upload(&d, 0, raw(dense));
        println!("gpu      : {:.1} GB of dense weights resident in VRAM (mapped in {:.2}s, written at {:.1} GB/s)", dense.len() as f64 * 2.0 / 1e9, (t1 - t0).as_secs_f32(), dense.len() as f64 * 2.0 / t1.elapsed().as_secs_f64() / 1e9);
        d
    }
}

/// The model on the card, one sync per token. The residual stays in VRAM
/// from the embedding to the head, the router runs on the card (`route`),
/// every layer's launches are posted as they come, and the host waits once,
/// in `logits`. What the host needs afterwards — the position's cache rows
/// for the store, and under `rockem.debug` each layer's residual and routing
/// — the kernels mirror into host memory as they go, read after that wait.
pub struct KfdModel<'a> {
    k: Rc<Kfd>,
    proj: Kernel,
    absorb: Kernel,
    mla: Kernel,
    merge: Kernel,
    out: Kernel,
    rows: Kernel,
    dense_down: Kernel,
    shared_down: Kernel,
    route: Kernel,
    up: Kernel,
    down: Kernel,
    sample: Kernel,
    score: Kernel,
    d_dense: Mem,
    d_stream: Mem,
    d_cache: Mem, // [layers][max_ctx][KV_ROW]
    d_cos: Mem,
    d_sin: Mem,
    // per-step buffers, TOKENS deep
    d_h: Mem,  // [T][H], the residual; host-visible for the embedding write
    d_q: Mem,
    d_kva: Mem,
    d_heads: Mem,
    d_qabs: Mem, // [T][HEADS][KV_LORA_RANK], the absorbed queries
    d_part: Mem,
    d_wt: Mem,
    d_xq: Mem,
    d_xs: Mem,
    d_xg: Mem,
    d_expert: Mem, // [T][TOP_K] u32, for `group`
    d_count: Mem,  // COUNT_WORDS u32: counts, the active experts, the arrival words
    d_logit: Mem,  // [T][E], the router rows' dots
    d_list: Mem,   // [E][T] u32
    d_egu: Mem,    // [T][TOP_K][2·D_FF]
    d_eout: Mem,   // [T][TOP_K][H]
    d_gu: Mem,     // [T][2·INTERMEDIATE_SIZE] f32, the widest
    d_mirror: Mem, // host: each token's cache rows, [T][layer][KV_ROW]
    d_layers: Mem, // host: every layer's residual out, [layer][T][H]
    d_topk: Mem,   // host: every layer's routing, [layer][T][TOP_K] u32
    d_logits: Mem, // VRAM, host-visible: [T][VOCAB_SIZE]
    d_next: Mem,   // host: the drawn token, or a score
    max_ctx: usize,
    /// The placement table, a word per expert: 0 on the card, 1 in this host's DRAM
    /// (`d_place` is its copy for the router). The host's share of the experts runs
    /// from `stream` in this process's memory, on the card's integers mirrored to
    /// `d_xq_m`/`d_xs_m`/`d_xg_m` (`[layer][T]…`), into its slots of `d_eout`
    /// (host-visible), between a mark after each layer's router and a gate before
    /// its combine (`host_experts`).
    place: Vec<u32>,
    d_place: Mem,
    /// Layers whose gate is posted and not yet released: (layer, the step's token count, the mark to wait for).
    pending: Vec<(usize, usize, u64)>,
    /// The host's expert time this token: computing, and waiting for marks (seconds; `bench-token` reads and resets).
    pub host_busy: f64,
    pub host_wait: f64,
    stream: &'a [u8],
    scratch: deepseek_v2_lite::expert::ScratchBuf,
    d_xq_m: Mem, // host: [layer][T][H] i8
    d_xs_m: Mem, // host: [layer][T][H / GROUP] f32
    d_xg_m: Mem, // host: [layer][T][H / GROUP] i32
}
/// The signal a layer's gate waits on (marks use their ring slot's signal).
const fn gate_of(l: usize) -> usize {
    1 + l
}
const _: () = assert!(gate_of(NUM_HIDDEN_LAYERS - 1) < SIGNALS);

impl<'a> KfdModel<'a> {
    pub fn new(k: Rc<Kfd>, dense: &[u16], stream: &'a [u8], rope: &Rope, max_ctx: usize, place: &[u32]) -> Self {
        assert_eq!(stream.len(), layout::STREAM);
        assert_eq!(place.len(), N_ROUTED_EXPERTS);
        let d_place = k.vram_visible(N_ROUTED_EXPERTS * 4);
        k.upload(&d_place, 0, raw_u32(place));
        println!("experts  : {} of {} in this host's DRAM, the rest on the card", place.iter().filter(|&&p| p == 1).count(), N_ROUTED_EXPERTS);
        let d_dense = k.upload_dense(dense);
        let t0 = std::time::Instant::now();
        let d_stream = k.vram_visible(stream.len());
        let t1 = std::time::Instant::now();
        k.upload(&d_stream, 0, stream);
        println!("gpu      : {:.1} GB of experts resident in VRAM (mapped in {:.2}s, written at {:.1} GB/s)", stream.len() as f64 / 1e9, (t1 - t0).as_secs_f32(), stream.len() as f64 / t1.elapsed().as_secs_f64() / 1e9);
        let d_cache = k.vram_visible(NUM_HIDDEN_LAYERS * max_ctx * KV_ROW * 4);
        let d_cos = k.vram_visible(rope.cos.len() * 4);
        k.upload(&d_cos, 0, raw(&rope.cos));
        let d_sin = k.vram_visible(rope.sin.len() * 4);
        k.upload(&d_sin, 0, raw(&rope.sin));
        println!("gpu      : attention cache for {max_ctx} positions in VRAM, steps of up to {TOKENS} tokens");
        const T: usize = TOKENS;
        let d_count = k.vram_visible(ffn::gfx11_consts::COUNT_WORDS * 4);
        k.upload(&d_count, 0, &vec![0u8; ffn::gfx11_consts::COUNT_WORDS * 4]); // the arrival word starts at zero
        KfdModel {
            proj: k.kernel("attn_proj"),
            absorb: k.kernel("attn_absorb"),
            mla: k.kernel("attn_mla"),
            merge: k.kernel("attn_merge"),
            out: k.kernel("attn_out"),
            rows: k.kernel("bf16_rows"),
            dense_down: k.kernel("dense_down"),
            shared_down: k.kernel("shared_down"),
            route: k.kernel("route"),
            up: k.kernel("experts_up"),
            down: k.kernel("experts_down"),
            sample: k.kernel("sample"),
            score: k.kernel("score"),
            d_dense,
            d_stream,
            d_cache,
            d_cos,
            d_sin,
            d_h: k.vram_visible(T * H * 4),
            d_q: k.vram(T * Q_DIM * 4),
            d_kva: k.vram(T * KVA_DIM * 4),
            d_heads: k.vram(T * O_IN_DIM * 4),
            d_qabs: k.vram(T * attn::gfx11_consts::QABS * 4),
            d_part: k.vram(T * attn::gfx11_consts::partials(max_ctx) * 4),
            d_wt: k.vram(T * TOP_K * 4),
            d_xq: k.vram(T * H),
            d_xs: k.vram(T * 4 * H / GROUP),
            d_xg: k.vram(T * 4 * H / GROUP),
            d_expert: k.vram(T * TOP_K * 4),
            d_count,
            d_logit: k.vram(T * N_ROUTED_EXPERTS * 4),
            d_list: k.vram(N_ROUTED_EXPERTS * T * 4),
            d_egu: k.vram(T * TOP_K * 2 * D_FF * 4),
            d_eout: k.vram_visible(T * TOP_K * H * 4), // the host writes its slots
            d_gu: k.vram(T * 2 * INTERMEDIATE_SIZE * 4),
            d_mirror: k.host(T * layout::POSITION * 4),
            d_layers: k.host(NUM_HIDDEN_LAYERS * T * H * 4),
            d_topk: k.host(NUM_HIDDEN_LAYERS * T * TOP_K * 4),
            d_logits: k.vram_visible(T * VOCAB_SIZE * 4), // the card draws from it; the host reads it only to trace
            d_next: k.host(4),
            max_ctx,
            place: place.to_vec(),
            d_place,
            pending: Vec::new(),
            host_busy: 0.0,
            host_wait: 0.0,
            stream,
            scratch: deepseek_v2_lite::expert::ScratchBuf::new(),
            d_xq_m: k.host(NUM_HIDDEN_LAYERS * T * H),
            d_xs_m: k.host(NUM_HIDDEN_LAYERS * T * (H / GROUP) * 4),
            d_xg_m: k.host(NUM_HIDDEN_LAYERS * T * (H / GROUP) * 4),
            k,
        }
    }
    fn dense_at(&self, elems: usize) -> usize {
        self.d_dense.at(elems * 2)
    }
    /// Wait for everything posted (benches; the node waits in `draw`).
    #[allow(dead_code)]
    pub fn sync(&mut self) {
        self.host_experts(); // the gates still posted, before waiting behind them
        self.k.sync()
    }
    /// The last token's launches with their card timestamps (see `Kfd::timeline`).
    #[allow(dead_code)]
    pub fn timeline(&self) -> Vec<(&'static str, u64, u64)> {
        self.k.timeline()
    }
    /// The last token's kernel statistics (see `Kfd::stats`).
    #[allow(dead_code)]
    pub fn stats(&self) -> Vec<(&'static str, [Vec<f32>; 4])> {
        self.k.stats()
    }
    /// `xn = rmsnorm(h) ⊙ ln` for the step's `n` tokens: what every rows kernel reads.
    /// `out[n][m] = W[m][H] · (rmsnorm(h[n]) ⊙ ln)`, posted; `free` lets it launch alongside the dispatch before it.
    fn rows(&self, ln: usize, w: usize, out: &Mem, m: usize, n: usize, free: bool) {
        let mut a = Args::new();
        a.ptr(self.d_h.va).ptr(ln).ptr(w).ptr(out.va).u32(m as u32).u32(n as u32);
        let lds = ffn::gfx11_consts::normed_rows_lds(H, n) as u32;
        if free {
            self.k.launch_free(&self.rows, ffn::gfx11_consts::rows_grid(m), WG, lds, &a);
        } else {
            self.k.launch(&self.rows, ffn::gfx11_consts::rows_grid(m), WG, lds, &a);
        }
    }
}

impl Attention for KfdModel<'_> {
    fn seed(&mut self, rows: &[f32]) {
        assert_eq!(rows.len() % layout::POSITION, 0);
        // the store is position-major and the card layer-major: gather each layer's rows in
        // DRAM, then write the layer as one run — a BAR write is at the link only in long runs
        let n = rows.len() / layout::POSITION;
        let mut run = vec![0f32; n * KV_ROW];
        for l in 0..NUM_HIDDEN_LAYERS {
            for p in 0..n {
                run[p * KV_ROW..(p + 1) * KV_ROW].copy_from_slice(&rows[p * layout::POSITION + l * KV_ROW..p * layout::POSITION + (l + 1) * KV_ROW]);
            }
            self.k.upload(&self.d_cache, attn::gfx11_consts::cache_layer(l, self.max_ctx), raw(&run));
        }
    }
    fn rows(&self, i: usize, out: &mut [f32]) {
        read(&self.d_mirror, i * layout::POSITION * 4, out);
    }
    /// Layer 0 carries the embeddings in; after that `h` on the host is not read again until the trace.
    fn step(&mut self, l: usize, _w: &attn::Weights, h: &mut [f32], pos: usize, n: usize, _cos: &[f32], _sin: &[f32], scale: f32, _tr: &mut dyn Trace) {
        assert!(n >= 1 && n <= TOKENS && pos + n <= self.max_ctx);
        if l == 0 {
            self.k.upload(&self.d_h, 0, raw(&h[..n * H]));
        }
        let b = dl::layer(l);
        let (ln, wq, wkva, kvaln, kvb, wo) = (self.dense_at(b + dl::INPUT_LN), self.dense_at(b + dl::Q_PROJ), self.dense_at(b + dl::KV_A), self.dense_at(b + dl::KV_A_LN), self.dense_at(b + dl::KV_B), self.dense_at(b + dl::O_PROJ));
        let cache = self.d_cache.at(attn::gfx11_consts::cache_layer(l, self.max_ctx));
        let cos = self.d_cos.at(pos * QK_ROPE_HEAD_DIM * 4);
        let sin = self.d_sin.at(pos * QK_ROPE_HEAD_DIM * 4);
        let max_splits = attn::gfx11_consts::max_splits(self.max_ctx) as u32;
        let mut a1 = Args::new();
        a1.ptr(self.d_h.va).ptr(ln).ptr(wq).ptr(wkva).ptr(self.d_q.va).ptr(self.d_kva.va).u32(n as u32);
        self.k.launch(&self.proj, attn::gfx11_consts::PROJ_GRID, WG, attn::gfx11_consts::normed_rows_lds(H, n) as u32, &a1);
        // the absorbed queries and the cache rows, one launch after the projections
        let mut ab = Args::new();
        ab.ptr(self.d_q.va).ptr(kvb).ptr(self.d_qabs.va).u32(n as u32).ptr(self.d_kva.va).ptr(kvaln).ptr(cache).u32(pos as u32).ptr(cos).ptr(sin).ptr(self.d_mirror.at(l * KV_ROW * 4));
        self.k.launch(&self.absorb, attn::gfx11_consts::absorb_grid(n), WG, attn::gfx11_consts::ABSORB_LDS as u32, &ab);
        let mut a2 = Args::new();
        a2.ptr(self.d_qabs.va).ptr(cache).u32(pos as u32).ptr(self.d_q.va).ptr(cos).ptr(sin).f32(scale).ptr(self.d_part.va).u32(max_splits);
        self.k.launch(&self.mla, attn::gfx11_consts::mla_grid(pos, n), WG, attn::gfx11_consts::MLA_LDS as u32, &a2);
        let mut am = Args::new();
        am.ptr(self.d_part.va).u32(pos as u32).u32(max_splits).ptr(kvb).ptr(self.d_heads.va);
        self.k.launch(&self.merge, (n * HEADS) as u32, WG, attn::gfx11_consts::MERGE_LDS as u32, &am);
        let mut a3 = Args::new();
        a3.ptr(self.d_heads.va).ptr(wo).ptr(self.d_h.va).u32(n as u32);
        self.k.launch(&self.out, attn::gfx11_consts::OUT_GRID, WG, attn::gfx11_consts::rows_lds(O_IN_DIM, n) as u32, &a3);
    }
}

impl FeedForward for KfdModel<'_> {
    fn dense(&mut self, l: usize, _w: &ffn::Dense, _h: &mut [f32], n: usize, _tr: &mut dyn Trace) {
        let b = dl::layer(l);
        let (ln, gate, down) = (self.dense_at(b + dl::POST_LN), self.dense_at(b + dl::DENSE_GATE), self.dense_at(b + dl::DENSE_DOWN));
        self.rows(ln, gate, &self.d_gu, 2 * INTERMEDIATE_SIZE, n, false); // gate rows then up rows, contiguous in the file
        let mut a2 = Args::new();
        a2.ptr(self.d_gu.va).ptr(down).ptr(self.d_h.va).ptr(self.d_layers.at(l * TOKENS * H * 4)).u32(n as u32);
        self.k.launch(&self.dense_down, ffn::gfx11_consts::rows_grid(H), WG, ffn::gfx11_consts::rows_lds(INTERMEDIATE_SIZE, n) as u32, &a2);
    }

    fn moe(&mut self, l: usize, _w: &ffn::Moe, _h: &mut [f32], n: usize, _tr: &mut dyn Trace) {
        use deepseek_v2_lite::expert::gfx11_consts as ex;
        let b = dl::layer(l);
        let (ln, router, sg, sd) = (self.dense_at(b + dl::POST_LN), self.dense_at(b + dl::ROUTER), self.dense_at(b + dl::SHARED_GATE), self.dense_at(b + dl::SHARED_DOWN));
        let base = layout::record(l, 0).start as u64;
        let mut ar = Args::new();
        ar.ptr(self.d_h.va).ptr(ln).ptr(router).ptr(self.d_logit.va).ptr(self.d_wt.va).ptr(self.d_xq.va).ptr(self.d_xs.va).ptr(self.d_xg.va).ptr(self.d_expert.va).ptr(self.d_topk.at(l * TOKENS * TOP_K * 4)).u32(n as u32).ptr(self.d_count.va).ptr(self.d_list.va);
        ar.ptr(self.d_place.va).ptr(self.d_xq_m.at(l * TOKENS * H)).ptr(self.d_xs_m.at(l * TOKENS * (H / GROUP) * 4)).ptr(self.d_xg_m.at(l * TOKENS * (H / GROUP) * 4));
        self.k.launch(&self.route, (n * ffn::gfx11_consts::ROUTE_WGS) as u32, WG, ffn::gfx11_consts::ROUTE_LDS as u32, &ar);
        let mark = self.k.mark(); // the host may pick up this layer's routing and integers
        let mut au = Args::new();
        au.ptr(self.d_stream.va).u64(base).ptr(self.d_count.va).ptr(self.d_list.va).ptr(self.d_xq.va).ptr(self.d_xs.va).ptr(self.d_xg.va).ptr(self.d_egu.va);
        self.k.launch(&self.up, ex::up_grid(n), WG, ex::experts_lds(H, n) as u32, &au);
        // the shared experts' rows need only the residual, complete since the router
        // launched: free to overlap the routed experts' up
        self.rows(ln, sg, &self.d_gu, 2 * SHARED_FF, n, true);
        let mut ad = Args::new();
        ad.ptr(self.d_stream.va).u64(base).ptr(self.d_count.va).ptr(self.d_list.va).ptr(self.d_egu.va).ptr(self.d_eout.va);
        self.k.launch(&self.down, ex::down_grid(n), WG, ex::experts_lds(D_FF, n) as u32, &ad);
        let mut a2 = Args::new();
        a2.ptr(self.d_gu.va).ptr(sd).ptr(self.d_eout.va).ptr(self.d_wt.va).ptr(self.d_h.va).ptr(self.d_layers.at(l * TOKENS * H * 4)).u32(n as u32);
        self.k.gate(gate_of(l)); // the host's expert slots, before the combine
        self.pending.push((l, n, mark));
        self.k.launch(&self.shared_down, ffn::gfx11_consts::rows_grid(H), WG, ffn::gfx11_consts::rows_lds(SHARED_FF, n) as u32, &a2);
    }
}

impl KfdModel<'_> {
    /// This host's share of the step's experts, layer by layer behind the card:
    /// for every layer whose gate is posted, wait for its mark, run every
    /// (token, slot) whose expert is placed here from the mirrored integers into
    /// its slot of `d_eout`, open the gate. An empty share is an empty loop body
    /// — the marks and gates are posted either way, one path. Runs once the
    /// step is posted (`logits`), so the card is ahead the whole time, and
    /// before any wait (`sync`), so a driver that posts layers alone cannot
    /// deadlock on its own gates.
    pub fn host_experts(&mut self) {
        use deepseek_v2_lite::expert;
        let mut out = vec![0f32; H];
        for (l, n, mark) in std::mem::take(&mut self.pending) {
            let t = std::time::Instant::now();
            self.k.wait(mark);
            self.host_wait += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            for i in 0..n {
                for kk in 0..TOP_K {
                    let mut e = [0u32; 1];
                    read(&self.d_topk, ((l * TOKENS + i) * TOP_K + kk) * 4, &mut e);
                    let e = e[0] as usize;
                    if self.place[e] == 1 {
                        let ew = expert::Weights::from_record(&self.stream[layout::record(l, e)]);
                        let (xq, xs, xg) = unsafe {
                            (
                                std::slice::from_raw_parts(self.d_xq_m.at((l * TOKENS + i) * H) as *const i8, H),
                                std::slice::from_raw_parts(self.d_xs_m.at((l * TOKENS + i) * (H / GROUP) * 4) as *const f32, H / GROUP),
                                std::slice::from_raw_parts(self.d_xg_m.at((l * TOKENS + i) * (H / GROUP) * 4) as *const i32, H / GROUP),
                            )
                        };
                        expert::forward_q(&ew, xq, xs, xg, &mut self.scratch.scratch(), &mut out);
                        self.k.upload(&self.d_eout, (i * TOP_K + kk) * H * 4, raw(&out));
                    }
                }
            }
            self.host_busy += t.elapsed().as_secs_f64();
            self.k.release(gate_of(l));
        }
    }
}

impl Head for KfdModel<'_> {
    /// Posts the head for every token of the step; `logits` is filled on the card, row per token.
    fn logits(&mut self, _norm: &[f32], _lm_head: &[u16], _h: &[f32], n: usize, _logits: &mut [f32]) {
        let (ln, w) = (self.dense_at(dl::NORM), self.dense_at(dl::LM_HEAD));
        self.rows(ln, w, &self.d_logits, VOCAB_SIZE, n, false);
        // the step is posted: the host's experts follow the card through the layers
        self.host_experts();
    }
}

impl Model for KfdModel<'_> {
    fn token_trace(&self, i: usize, layers: &mut [f32], topk: &mut [u32], logits: &mut [f32]) {
        for l in 0..NUM_HIDDEN_LAYERS {
            read(&self.d_layers, (l * TOKENS + i) * H * 4, &mut layers[l * H..(l + 1) * H]);
            read(&self.d_topk, (l * TOKENS + i) * TOP_K * 4, &mut topk[l * TOP_K..(l + 1) * TOP_K]);
        }
        read(&self.d_logits, i * VOCAB_SIZE * 4, logits); // through the BAR: uncached reads, the trace's price
    }
    /// ln p(id) on the card, one wait.
    fn score(&mut self, i: usize, id: usize, _logits: &[f32]) -> f32 {
        let mut a = Args::new();
        a.ptr(self.d_logits.at(i * VOCAB_SIZE * 4)).ptr(self.d_next.va).u32(id as u32);
        self.k.launch(&self.score, 1, WG, ffn::gfx11_consts::SCORE_LDS as u32, &a);
        self.k.sync();
        let mut v = [0f32; 1];
        read(&self.d_next, 0, &mut v);
        v[0]
    }
    /// The one wait of the step: the draw on the card from token `i`'s row, then
    /// the token read from host memory. The logits stay on the card (`token_trace` reads them).
    fn draw(&mut self, i: usize, recipe: &Sampling, u: f64, _logits: &[f32]) -> usize {
        let mut a = Args::new();
        a.ptr(self.d_logits.at(i * VOCAB_SIZE * 4)).ptr(self.d_next.va).f64(u).f32(recipe.temperature).u32(recipe.top_k as u32).f32(recipe.top_p);
        self.k.launch(&self.sample, 1, WG, ffn::gfx11_consts::SAMPLE_LDS as u32, &a);
        self.k.sync();
        let mut next = [0u32; 1];
        read(&self.d_next, 0, &mut next);
        next[0] as usize
    }
}
