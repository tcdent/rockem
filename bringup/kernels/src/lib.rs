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

/// `rate_wmma_iu8` with eight independent chains a wave (docs/queue/011: four may not cover the
/// iu8 issue latency — iu8 measured no faster than bf16 at four): 8 · 2 · 16³ ops a round a wave.
#[no_mangle]
pub extern "gpu-kernel" fn rate_wmma_iu8_8(out: *mut v8i32, iters: u32) {
    unsafe {
        let g = (workgroup_id_x() * 256 + workitem_id_x()) as usize;
        let a = v4i32([g as i32, 0x0102_0304, 0x0506_0708, 0x090a_0b0c]);
        let b = v4i32([0x0c0b_0a09, 0x0807_0605, 0x0403_0201, g as i32]);
        let mut c = [v8i32([0; 8]), v8i32([1; 8]), v8i32([2; 8]), v8i32([3; 8]), v8i32([4; 8]), v8i32([5; 8]), v8i32([6; 8]), v8i32([7; 8])];
        let mut i = 0;
        while i < iters {
            let mut j = 0;
            while j < 8 {
                c[j] = wmma_i32_iu8(true, a, true, b, c[j], false);
                j += 1;
            }
            i += 1;
        }
        let mut j = 0;
        while j < 8 {
            *out.add(g * 8 + j) = c[j];
            j += 1;
        }
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

/// The wave32 bf16 WMMA's operand layout, found empirically: lane `l` loads A as `a[l * 16 .. l * 16 + 16]`
/// and B as `b[l * 16 .. ]` (16 bf16 each, as the compiler's fragments want them), the product's 8 f32 a
/// lane go out as `out[l * 8 + r]`; the host knows what it put in and solves for which (row, col) each
/// (lane, register) holds. Grid: one wave (a workgroup of 32).
#[no_mangle]
pub extern "gpu-kernel" fn wmma_layout(a: *const u16, b: *const u16, out: *mut f32) {
    unsafe {
        let l = workitem_id_x() as usize;
        if l >= 32 {
            return;
        }
        let mut av = [0i16; 16];
        let mut bv = [0i16; 16];
        let mut i = 0;
        while i < 16 {
            av[i] = *a.add(l * 16 + i) as i16;
            bv[i] = *b.add(l * 16 + i) as i16;
            i += 1;
        }
        let d = wmma_f32_bf16(v16i16(av), v16i16(bv), v8f32([0.0; 8]));
        let dv: [f32; 8] = core::mem::transmute(d);
        let mut r = 0;
        while r < 8 {
            *out.add(l * 8 + r) = dv[r];
            r += 1;
        }
    }
}

/// Ping-pong over a link (docs/queue/001): one wave. `rounds` times, the opener (`first` = 1) writes
/// `words` payload words (`r ^ i`) into the peer's buffer, then the round number into the peer's word;
/// the other side spins on its own word, checks the payload it finds (mismatches counted into
/// `errors`: the visibility test), and answers the same way. The words and buffers may be the peer's
/// VRAM (peer writes), host memory (today's handoffs) or, with the host as the other side, a host word.
/// Grid 1 × 32.
#[no_mangle]
pub extern "gpu-kernel" fn pingpong(mine: *mut u32, theirs: *mut u32, my_data: *mut u32, their_data: *mut u32, words: u32, rounds: u32, first: u32, errors: *mut u32) {
    use core::sync::atomic::{fence, AtomicU32, Ordering};
    let t = workitem_id_x() as usize;
    let mut bad = 0u32;
    unsafe {
        let send = |r: u32| {
            let mut i = t;
            while i < words as usize {
                *their_data.add(i) = r ^ i as u32;
                i += 32;
            }
            fence(Ordering::Release);
            if t == 0 {
                AtomicU32::from_ptr(theirs).store(r, Ordering::Release);
            }
        };
        let recv = |r: u32, bad: &mut u32| {
            if t == 0 {
                while AtomicU32::from_ptr(mine).load(Ordering::Acquire) != r {}
            }
            fence(Ordering::Acquire);
            let mut i = t;
            while i < words as usize {
                if AtomicU32::from_ptr(my_data.add(i)).load(Ordering::Relaxed) != r ^ i as u32 {
                    *bad += 1;
                }
                i += 32;
            }
        };
        let mut r = 1;
        while r <= rounds {
            if first != 0 {
                send(r);
                recv(r, &mut bad);
            } else {
                recv(r, &mut bad);
                send(r);
            }
            r += 1;
        }
        if bad != 0 {
            AtomicU32::from_ptr(errors).fetch_add(bad, Ordering::Relaxed);
        }
    }
}
