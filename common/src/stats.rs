//! Statistics from inside kernels, into a slab the launcher owns; a
//! pre-defined structure, so what a kernel calls a number is what the host
//! reads it back as. The slab is `[STAT_RING][STAT_SLOTS][STAT_WAVES]` u32:
//! a block per launch, found by the packet's own dispatch id (so it is the
//! timeline's index by construction), a slot per named statistic of that
//! kernel, an entry per wave — every wave writes its own value and the
//! host averages or ranks after the fact, so no wave is a representative and
//! nothing is atomic. Values are f32 bits; times are `stat_clock!()` deltas
//! in ticks of the 100 MHz clock the timeline uses. Under the `stats`
//! feature only: `stat!` expands to nothing otherwise, and this is the one
//! thing allowed to differ inside a kernel body between the two builds — a
//! store, never a branch that changes what the kernel computes. The
//! launcher enables it by writing the slab's address into `ROCKEM_STATS`
//! in the loaded image; a code object without that symbol has no stats.
pub const STAT_RING: usize = 512;
pub const STAT_SLOTS: usize = 4;
pub const STAT_WAVES: usize = 16384;
pub const STAT_WAVES_PER_WG: u32 = 8;

#[cfg(all(target_arch = "amdgpu", feature = "stats"))]
#[no_mangle]
pub static mut ROCKEM_STATS: u64 = 0;

/// The real-time counter (100 MHz, 64 bits): a returning send-message on gfx11, about a microsecond a read
/// (measured through this channel). The 20-bit SHADER_CYCLES register it replaced wraps every ~0.5 ms, so a prefill
/// launch's phases (milliseconds, 2026-10-20) came out modulo it; a microsecond a stamp is noise beside them, and
/// the few-token phases (tens of µs) carry it as a known bias.
#[cfg(all(target_arch = "amdgpu", feature = "stats"))]
#[inline(always)]
pub unsafe fn clock() -> u64 {
    let c: u64;
    core::arch::asm!("s_sendmsg_rtn_b64 {c}, sendmsg(MSG_RTN_GET_REALTIME)", "s_waitcnt lgkmcnt(0)", c = out(sgpr64) c, options(nostack, nomem, preserves_flags));
    c
}
/// The shader clock in core cycles: the 20-bit SHADER_CYCLES register, wrapping every ~0.5 ms at 2 GHz. Beside a
/// `clock()` pair over the same span it gives the core clock the launch ran at (`stat_cycles!`).
#[cfg(all(target_arch = "amdgpu", feature = "stats"))]
#[inline(always)]
pub unsafe fn cycles() -> u32 {
    let c: u32;
    core::arch::asm!("s_getreg_b32 {c}, hwreg(HW_REG_SHADER_CYCLES, 0, 20)", c = out(sgpr32) c, options(nostack, nomem, preserves_flags));
    c
}
/// The difference of two `cycles()` readings, modulo the register's width.
pub const CYCLES_MASK: u32 = (1 << 20) - 1;
/// Every load of this wave landed (`s_waitcnt vmcnt(0)`), so a stamp after it times the loads and not the compiler's
/// placement of the wait; the compiler treats it as touching memory, so no load moves across it.
#[cfg(all(target_arch = "amdgpu", feature = "stats"))]
#[inline(always)]
pub unsafe fn loads_landed() {
    core::arch::asm!("s_waitcnt vmcnt(0)", options(nostack, preserves_flags));
}

/// Where the wave runs: HW_ID1 with the wave and SIMD fields masked off — the WGP, the shader array and the shader
/// engine (gfx11: wave 0–4, SIMD 8–9, WGP 10–13, SA 16, SE 18–20). Equal values from two workgroups: the same WGP.
#[cfg(all(target_arch = "amdgpu", feature = "stats"))]
#[inline(always)]
pub unsafe fn hw_id() -> u32 {
    let c: u32;
    core::arch::asm!("s_getreg_b32 {c}, hwreg(HW_REG_HW_ID1)", c = out(sgpr32) c, options(nostack, nomem, preserves_flags));
    c & 0x001F_FC00
}

/// The difference of two `clock()` readings: the 64-bit real-time counter never wraps in a launch.
pub const CLOCK_MASK: u64 = u64::MAX;
/// `clock()` ticks a microsecond (the 100 MHz real-time counter).
pub const CLOCK_PER_US: f64 = 100.0;

#[cfg(all(target_arch = "amdgpu", feature = "stats"))]
#[inline(always)]
pub unsafe fn stat_write(slot: usize, value: f32) {
    use core::arch::amdgpu::{dispatch_id, workgroup_id_x, workitem_id_x};
    let base = core::ptr::read_volatile(core::ptr::addr_of!(ROCKEM_STATS)) as *mut u32;
    let wave = (workgroup_id_x() * STAT_WAVES_PER_WG + workitem_id_x() / 32) as usize;
    if !base.is_null() && wave < STAT_WAVES {
        let block = (dispatch_id() as usize) % STAT_RING;
        *base.add((block * STAT_SLOTS + slot) * STAT_WAVES + wave) = value.to_bits();
    }
}

/// `stat!(slot, value)`: this wave's value for the kernel's named slot.
#[macro_export]
macro_rules! stat {
    ($slot:expr, $value:expr) => {
        #[cfg(feature = "stats")]
        {
            $crate::stats::stat_write($slot, ($value) as f32)
        }
        #[cfg(not(feature = "stats"))]
        {
            let _ = ($value);
        }
    };
}
/// `stat_loads_landed!()`: wait for this wave's loads (see `loads_landed`); nothing without stats.
#[macro_export]
macro_rules! stat_loads_landed {
    () => {{
        #[cfg(feature = "stats")]
        {
            $crate::stats::loads_landed()
        }
    }};
}

/// `stat_hwid!()`: the wave's placement (see `hw_id`); 0 without stats.
#[macro_export]
macro_rules! stat_hwid {
    () => {{
        #[cfg(feature = "stats")]
        {
            $crate::stats::hw_id()
        }
        #[cfg(not(feature = "stats"))]
        {
            0u32
        }
    }};
}

/// `stat_cycles!()`: the shader clock in core cycles (see `cycles`); 0 without stats.
#[macro_export]
macro_rules! stat_cycles {
    () => {{
        #[cfg(feature = "stats")]
        {
            $crate::stats::cycles()
        }
        #[cfg(not(feature = "stats"))]
        {
            0u32
        }
    }};
}

/// `stat_clock!()`: the real-time counter (see `clock`).
#[macro_export]
macro_rules! stat_clock {
    () => {{
        #[cfg(feature = "stats")]
        {
            $crate::stats::clock()
        }
        #[cfg(not(feature = "stats"))]
        {
            0u64
        }
    }};
}
