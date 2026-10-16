//! gfx11 wave and workgroup primitives every GPU kernel composes: launch-
//! sized LDS, the workgroup barrier that is also an LDS fence, and wave
//! reductions over `ds_bpermute`. Wave32 throughout.

use core::arch::amdgpu::ds_bpermute;

/// The workgroup's launch-sized LDS, as a pointer of any type.
#[inline(always)]
pub unsafe fn lds<T>() -> *mut T {
    core::intrinsics::gpu::gpu_launch_sized_workgroup_mem::<u64>() as *mut T
}

/// Workgroup barrier that is also a memory fence for LDS: wait for this
/// wave's LDS traffic, then barrier. The barrier is the LLVM intrinsic,
/// which is `convergent`: written as `asm!`, the compiler once duplicated
/// it into both arms of a divergent branch (the sample kernel's rank
/// placement), and a wave with lanes on both sides then barriered twice.
/// The intrinsic is also `IntrNoMem`, so loads could move across it; the
/// `asm!` blocks on either side, without `nomem`, are compiler barriers
/// that pin memory traffic to its side of the wall.
#[inline(always)]
pub unsafe fn workgroup_sync() {
    core::arch::asm!("s_waitcnt lgkmcnt(0)", options(nostack));
    core::arch::amdgpu::s_barrier();
    core::arch::asm!("", options(nostack));
}

/// Spin until the word (VRAM, polled through L2) reads `value`, a short sleep between looks:
/// hundreds of waves polling one line back to back starve the workgroup they wait for
/// (`hc_pre`, 2026-10-16).
#[inline(always)]
pub unsafe fn await_word(w: *const u32, value: u32) {
    while core::sync::atomic::AtomicU32::from_ptr(w as *mut u32).load(core::sync::atomic::Ordering::Acquire) != value {
        nap();
    }
}

/// A short sleep between looks at a word another workgroup will write.
#[inline(always)]
pub unsafe fn nap() {
    core::arch::asm!("s_sleep 1", options(nostack, nomem));
}

/// Sum of one f32 per lane across the wave; every lane gets the result.
#[inline(always)]
pub unsafe fn wave_sum(lane: u32, v: f32) -> f32 {
    let mut v = v;
    let mut m = 1;
    while m < 32 {
        v += f32::from_bits(ds_bpermute((lane ^ m) * 4, v.to_bits()));
        m <<= 1;
    }
    v
}

/// Sum over the `n`-lane group a lane belongs to (`n` a power of two): lanes `lane & !(n-1) ..+n`.
#[inline(always)]
pub unsafe fn group_sum(lane: u32, v: f32, n: u32) -> f32 {
    let mut v = v;
    let mut m = 1;
    while m < n {
        v += f32::from_bits(ds_bpermute((lane ^ m) * 4, v.to_bits()));
        m <<= 1;
    }
    v
}
/// Max over the `n`-lane group a lane belongs to.
#[inline(always)]
pub unsafe fn group_max(lane: u32, v: f32, n: u32) -> f32 {
    let mut v = v;
    let mut m = 1;
    while m < n {
        let o = f32::from_bits(ds_bpermute((lane ^ m) * 4, v.to_bits()));
        if o > v {
            v = o;
        }
        m <<= 1;
    }
    v
}
/// The value lane `src` holds, to every lane of the wave.
#[inline(always)]
pub unsafe fn lane_read(src: u32, v: f32) -> f32 {
    f32::from_bits(ds_bpermute(src * 4, v.to_bits()))
}

/// Max of one f32 per lane across the wave; every lane gets the result.
#[inline(always)]
pub unsafe fn wave_max(lane: u32, v: f32) -> f32 {
    let mut v = v;
    let mut m = 1;
    while m < 32 {
        let o = f32::from_bits(ds_bpermute((lane ^ m) * 4, v.to_bits()));
        if o > v {
            v = o;
        }
        m <<= 1;
    }
    v
}

/// Sum of one f32 per thread across a 256-thread workgroup, through eight
/// wave partials in `red`; every thread gets the result. Two barriers.
#[inline(always)]
pub unsafe fn workgroup_sum(tid: u32, v: f32, red: *mut f32) -> f32 {
    let (lane, wave) = (tid % 32, (tid / 32) as usize);
    let s = wave_sum(lane, v);
    if lane == 0 {
        *red.add(wave) = s;
    }
    workgroup_sync();
    let mut total = 0f32;
    let mut w = 0;
    while w < 8 {
        total += *red.add(w);
        w += 1;
    }
    workgroup_sync();
    total
}

/// Max, likewise.
#[inline(always)]
pub unsafe fn workgroup_max(tid: u32, v: f32, red: *mut f32) -> f32 {
    let (lane, wave) = (tid % 32, (tid / 32) as usize);
    let s = wave_max(lane, v);
    if lane == 0 {
        *red.add(wave) = s;
    }
    workgroup_sync();
    let mut m = *red;
    let mut w = 1;
    while w < 8 {
        if *red.add(w) > m {
            m = *red.add(w);
        }
        w += 1;
    }
    workgroup_sync();
    m
}
