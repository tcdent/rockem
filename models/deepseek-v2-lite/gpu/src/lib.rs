//! gfx1100 code object for DeepSeek-V2-Lite. The kernel entries live in the
//! model crate next to their CPU twins; this crate only gives them a home
//! the linker turns into an ELF.
#![no_std]

pub use deepseek_v2_lite::attn::gfx11 as attn;
pub use deepseek_v2_lite::expert::gfx11 as expert;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

