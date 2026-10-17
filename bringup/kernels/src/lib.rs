//! Rate probes for gfx1100: how fast the card's matrix path and its FMA path run with nothing
//! but registers in the way. The notes' question (docs/glm-next-steps.md §5): is the iu8 WMMA
//! rate far enough above the rows kernels' rate for an Ozaki-form prefill kernel (bf16 weights
//! and f32 activations sliced into int8, exact int32 accumulation) to pay? Each kernel runs
//! `iters` rounds of four independent chains per wave (the issue latency hidden by the chains)
//! and writes the accumulators out so nothing is dead.
#![no_std]
#![feature(abi_gpu_kernel, stdarch_amdgpu, link_llvm_intrinsics, repr_simd, simd_ffi, core_intrinsics)]
#![allow(internal_features, non_camel_case_types, improper_ctypes)]

use core::arch::amdgpu::{workgroup_id_x, workitem_id_x};

#[repr(simd)]
#[derive(Clone, Copy)]
pub struct v16i16([i16; 16]);
#[repr(simd)]
#[derive(Clone, Copy)]
pub struct v8f32([f32; 8]);
#[repr(simd)]
#[derive(Clone, Copy)]
pub struct v4i32([i32; 4]);
#[repr(simd)]
#[derive(Clone, Copy)]
pub struct v8i32([i32; 8]);

unsafe extern "C" {
    #[link_name = "llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16"]
    fn wmma_f32_bf16(a: v16i16, b: v16i16, c: v8f32) -> v8f32;
    #[link_name = "llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32"]
    fn wmma_i32_iu8(sa: bool, a: v4i32, sb: bool, b: v4i32, c: v8i32, clamp: bool) -> v8i32;
}

/// `iters` rounds of four iu8 WMMA chains a wave: 4 · 2 · 16³ integer ops a round a wave.
#[no_mangle]
pub extern "gpu-kernel" fn rate_wmma_iu8(out: *mut v8i32, iters: u32) {
    unsafe {
        let g = (workgroup_id_x() * 256 + workitem_id_x()) as usize;
        let a = v4i32([g as i32, 0x0102_0304, 0x0506_0708, 0x090a_0b0c]);
        let b = v4i32([0x0c0b_0a09, 0x0807_0605, 0x0403_0201, g as i32]);
        let (mut c0, mut c1, mut c2, mut c3) = (v8i32([0; 8]), v8i32([1; 8]), v8i32([2; 8]), v8i32([3; 8]));
        let mut i = 0;
        while i < iters {
            c0 = wmma_i32_iu8(true, a, true, b, c0, false);
            c1 = wmma_i32_iu8(true, a, true, b, c1, false);
            c2 = wmma_i32_iu8(true, a, true, b, c2, false);
            c3 = wmma_i32_iu8(true, a, true, b, c3, false);
            i += 1;
        }
        *out.add(g * 4) = c0;
        *out.add(g * 4 + 1) = c1;
        *out.add(g * 4 + 2) = c2;
        *out.add(g * 4 + 3) = c3;
    }
}

/// The same with bf16 WMMA (f32 accumulation): 4 · 2 · 16³ flops a round a wave.
#[no_mangle]
pub extern "gpu-kernel" fn rate_wmma_bf16(out: *mut v8f32, iters: u32) {
    unsafe {
        let g = (workgroup_id_x() * 256 + workitem_id_x()) as usize;
        let a = v16i16([0x3f80, 0x3f00, 0x3e80, 0x3e00, 0x3d80, 0x3d00, 0x3c80, 0x3c00, g as i16, 1, 2, 3, 4, 5, 6, 7]);
        let b = v16i16([7, 6, 5, 4, 3, 2, 1, g as i16, 0x3c00, 0x3c80, 0x3d00, 0x3d80, 0x3e00, 0x3e80, 0x3f00, 0x3f80]);
        let (mut c0, mut c1, mut c2, mut c3) = (v8f32([0.0; 8]), v8f32([1.0; 8]), v8f32([2.0; 8]), v8f32([3.0; 8]));
        let mut i = 0;
        while i < iters {
            c0 = wmma_f32_bf16(a, b, c0);
            c1 = wmma_f32_bf16(a, b, c1);
            c2 = wmma_f32_bf16(a, b, c2);
            c3 = wmma_f32_bf16(a, b, c3);
            i += 1;
        }
        *out.add(g * 4) = c0;
        *out.add(g * 4 + 1) = c1;
        *out.add(g * 4 + 2) = c2;
        *out.add(g * 4 + 3) = c3;
    }
}

/// Scalar f32 FMA, four chains a lane: 4 · 2 · 32 flops a round a wave (the rows kernels' arithmetic).
#[no_mangle]
pub extern "gpu-kernel" fn rate_fma_f32(out: *mut f32, iters: u32) {
    unsafe {
        let g = (workgroup_id_x() * 256 + workitem_id_x()) as usize;
        let x = g as f32 * 1e-9 + 0.999;
        let (mut c0, mut c1, mut c2, mut c3) = (0.0f32, 1.0f32, 2.0f32, 3.0f32);
        let mut i = 0;
        while i < iters {
            c0 = core::intrinsics::fmaf32(c0, x, 0.5);
            c1 = core::intrinsics::fmaf32(c1, x, 0.25);
            c2 = core::intrinsics::fmaf32(c2, x, 0.125);
            c3 = core::intrinsics::fmaf32(c3, x, 0.0625);
            i += 1;
        }
        *out.add(g) = c0 + c1 + c2 + c3;
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
