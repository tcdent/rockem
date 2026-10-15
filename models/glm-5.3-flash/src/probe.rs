//! Bring-up probes carried in this model's code object: `stream`, a kernel
//! that reads a buffer and does nothing else, for measuring what the card
//! streams from its own VRAM and from host memory across the link
//! (`bench-box`). Not a model kernel; here because the ELF the node loads is
//! this model's, and a probe that lives in it needs no second code object.

/// Launch geometry, visible to hosts.
pub mod gfx11_consts {
    pub const WG: u32 = 256;
    /// u32 words each workgroup streams: 256 threads × WORDS_PER_THREAD, contiguous per step.
    pub const WORDS_PER_THREAD: usize = 4096;
    pub const WORDS_PER_WG: usize = 256 * WORDS_PER_THREAD;
}

#[cfg(target_arch = "amdgpu")]
pub mod gfx11 {
    pub use super::gfx11_consts::*;
    use core::arch::amdgpu::{workgroup_id_x, workitem_id_x};

    /// Every workgroup sums its `WORDS_PER_WG` words of `buf` (a step of 256
    /// consecutive words a round, coalesced) into `out[workgroup]`, so the
    /// reads cannot be elided. Grid: `words / WORDS_PER_WG`.
    #[no_mangle]
    pub extern "gpu-kernel" fn stream(buf: *const u32, out: *mut u32) {
        unsafe {
            let g = workgroup_id_x() as usize;
            let t = workitem_id_x() as usize;
            let base = buf.add(g * WORDS_PER_WG);
            let mut acc = 0u32;
            let mut s = 0;
            while s < WORDS_PER_THREAD {
                acc = acc.wrapping_add(*base.add(s * 256 + t));
                s += 1;
            }
            let o = out.add(g * 256 + t);
            *o = acc;
        }
    }
}
