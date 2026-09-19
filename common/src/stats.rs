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

/// The shader clock, in core cycles: `s_getreg` of SHADER_CYCLES, one
/// scalar instruction, 20 bits wide (it wraps every ~0.4 ms at 2.5 GHz, so
/// a phase must be shorter than that; `stat_clock!` deltas are taken modulo
/// the width). The real-time clock the timeline uses is a returning
/// send-message on gfx11 and costs about a microsecond a read — measured
/// through this very channel — which is too coarse for phases inside a
/// kernel. The host converts cycles to time with the clock rate it infers
/// per kernel from the timeline (cycles per real-time tick).
#[cfg(all(target_arch = "amdgpu", feature = "stats"))]
#[inline(always)]
pub unsafe fn clock() -> u64 {
    let c: u32;
    core::arch::asm!("s_getreg_b32 {c}, hwreg(HW_REG_SHADER_CYCLES, 0, 20)", c = out(sgpr32) c, options(nostack, nomem, preserves_flags));
    c as u64
}
/// The difference of two `clock()` readings, modulo the counter's width.
pub const CLOCK_MASK: u64 = (1 << 20) - 1;

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
/// `stat_clock!()`: the 100 MHz clock, for phase deltas; 0 without stats.
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
