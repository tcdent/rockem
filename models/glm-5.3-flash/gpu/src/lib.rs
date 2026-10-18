//! gfx1100 code object for GLM-5.3-Flash. The kernel entries live in the
//! model crate next to their CPU twins; this crate only gives them a home
//! the linker turns into an ELF.
#![no_std]

pub use glm_5_3_flash::expert::gfx11 as xpt;
pub use glm_5_3_flash::ffn::gfx11 as ffn;
pub use glm_5_3_flash::hc::gfx11 as hc;
pub use glm_5_3_flash::kda::gfx11 as kda;
pub use glm_5_3_flash::mla::gfx11 as mla;
pub use glm_5_3_flash::probe::gfx11 as probe;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
