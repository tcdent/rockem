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

/// A wave's own LDS traffic settled before its lanes read what other lanes of the same wave
/// wrote (the LDS serves one wave's requests in order; this pins the compiler's order too).
#[inline(always)]
pub unsafe fn wave_lds_sync() {
    core::arch::asm!("s_waitcnt lgkmcnt(0)", options(nostack));
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

/// The lanes of the wave (wave32) where `b` holds, a bit a lane.
#[inline(always)]
pub unsafe fn ballot(b: bool) -> u32 {
    core::arch::amdgpu::ballot(b) as u32
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
/// Max of `N` u32s a lane across the wave by xor butterflies over `ds_bpermute`, the `N` in lockstep (each step's
/// `N` permutes issued before one wait); every lane gets each. Exact (a max has no order), and five steps where
/// `wave_reduce_umax` compiles to a loop over the 32 lanes one at a time (13 µs for a token's quantize,
/// 2026-10-19). Non-negative f32 bits compare as their floats.
#[inline(always)]
pub unsafe fn wave_max_u32<const N: usize>(lane: u32, v: [u32; N]) -> [u32; N] {
    let mut v = v;
    let mut m = 1;
    while m < 32 {
        let mut o = [0u32; N];
        let mut i = 0;
        while i < N {
            o[i] = ds_bpermute((lane ^ m) * 4, v[i]);
            i += 1;
        }
        let mut i = 0;
        while i < N {
            v[i] = if o[i] > v[i] { o[i] } else { v[i] };
            i += 1;
        }
        m <<= 1;
    }
    v
}
/// Sums of `N` i32s a lane across the wave, likewise (integer: exact in any order).
#[inline(always)]
pub unsafe fn wave_sum_i32<const N: usize>(lane: u32, v: [i32; N]) -> [i32; N] {
    let mut v = v;
    let mut m = 1;
    while m < 32 {
        let mut o = [0u32; N];
        let mut i = 0;
        while i < N {
            o[i] = ds_bpermute((lane ^ m) * 4, v[i] as u32);
            i += 1;
        }
        let mut i = 0;
        while i < N {
            v[i] = v[i].wrapping_add(o[i] as i32);
            i += 1;
        }
        m <<= 1;
    }
    v
}

/// Sum of `N` f32 a lane across the wave, the `N` butterflies in lockstep: each step's `N` permutes issued together, so
/// the wave waits ~5 round trips for all of them, not 5 a value (`wave_sum` a value at a time was 325 dependent
/// permutes for 65 values, ~20 µs, the analog's phase A, 2026-10-27).
#[inline(always)]
pub unsafe fn wave_sum_f32<const N: usize>(lane: u32, v: [f32; N]) -> [f32; N] {
    let mut v = v;
    let mut m = 1;
    while m < 32 {
        let mut o = [0u32; N];
        let mut i = 0;
        while i < N {
            o[i] = ds_bpermute((lane ^ m) * 4, v[i].to_bits());
            i += 1;
        }
        let mut i = 0;
        while i < N {
            v[i] += f32::from_bits(o[i]);
            i += 1;
        }
        m <<= 1;
    }
    v
}

unsafe extern "C" {
    #[link_name = "llvm.amdgcn.rcp.f32"]
    fn amdgcn_rcp(x: f32) -> f32;
    #[link_name = "llvm.amdgcn.rsq.f32"]
    fn amdgcn_rsq(x: f32) -> f32;
}
/// `v_rcp_f32`: the hardware reciprocal (1 ulp), one instruction — a `/` lowers to the IEEE division sequence of ~10
/// (`v_div_scale` … `v_div_fixup`), a dependent chain where a few divisions sit on one wave's critical path (the
/// analog's register Sinkhorn ran 1,280 of them a token: +20 µs over the permute chain it replaced, 2026-10-27).
#[inline(always)]
pub unsafe fn rcp(x: f32) -> f32 {
    amdgcn_rcp(x)
}
/// `v_rsq_f32`, one instruction (`math::rsqrt` is `1.0 / sqrt(x)`: the IEEE division sequence after the sqrt; the
/// bit-identical kernels keep it, the analog does not need it).
#[inline(always)]
pub unsafe fn rsq(x: f32) -> f32 {
    amdgcn_rsq(x)
}
/// `a * b + c` as one `v_fma_f32` (rustc never contracts the pair; the bit-identical kernels keep their pairs —
/// this is for arithmetic no forward output depends on, the analog's phase B first).
#[inline(always)]
pub unsafe fn fma(a: f32, b: f32, c: f32) -> f32 {
    core::intrinsics::fmaf32(a, b, c)
}

/// The value lane `src` holds, to every lane of the wave.
#[inline(always)]
pub unsafe fn lane_read(src: u32, v: f32) -> f32 {
    f32::from_bits(ds_bpermute(src * 4, v.to_bits()))
}

/// `v_mov_b32_dpp row_xmask:M`: the value lane `l ^ M` holds within its row of 16 — a butterfly step on the VALU,
/// no LDS and no wait (a `ds_bpermute` is a round trip through LDS, ~a hundred cycles each in a dependent chain).
/// `CTRL` = `0x160 | M`.
#[inline(always)]
pub unsafe fn dpp_xmask<const CTRL: u32>(v: f32) -> f32 {
    f32::from_bits(core::arch::amdgpu::update_dpp::<CTRL, 0xF, 0xF, true>(0, v.to_bits()))
}
pub const XMASK1: u32 = 0x161;
pub const XMASK2: u32 = 0x162;
pub const XMASK4: u32 = 0x164;
pub const XMASK8: u32 = 0x168;
/// The sum over the lane's row of 16, on every lane of the row: four butterflies.
#[inline(always)]
pub unsafe fn row_sum16(v: f32) -> f32 {
    let v = v + dpp_xmask::<XMASK1>(v);
    let v = v + dpp_xmask::<XMASK2>(v);
    let v = v + dpp_xmask::<XMASK4>(v);
    v + dpp_xmask::<XMASK8>(v)
}
/// A compiler fence: no memory operation moves across it (an empty `asm!` with a memory clobber, no instruction
/// emitted). For a hand-pipelined body whose loads the scheduler would otherwise hoist to the register bound.
#[inline(always)]
pub unsafe fn compiler_fence() {
    core::arch::asm!("", options(nostack, preserves_flags));
}
/// The max over the lane's row of 16, on every lane of the row.
#[inline(always)]
pub unsafe fn row_max16(v: f32) -> f32 {
    let o = dpp_xmask::<XMASK1>(v);
    let v = if o > v { o } else { v };
    let o = dpp_xmask::<XMASK2>(v);
    let v = if o > v { o } else { v };
    let o = dpp_xmask::<XMASK4>(v);
    let v = if o > v { o } else { v };
    let o = dpp_xmask::<XMASK8>(v);
    if o > v { o } else { v }
}
/// The value the same-numbered lane of the other row of 16 holds (`v_permlanex16`, a VALU op).
#[inline(always)]
pub unsafe fn other_row(v: f32) -> f32 {
    f32::from_bits(core::arch::amdgpu::permlanex16_u32::<false, false>(0, v.to_bits(), 0x7654_3210, 0xFEDC_BA98))
}
/// The wave's sum on every lane with no LDS: four butterflies and one cross-row permute, against `wave_sum`'s five
/// dependent `ds_bpermute` round trips.
#[inline(always)]
pub unsafe fn wave_sum_dpp(v: f32) -> f32 {
    let r = row_sum16(v);
    r + other_row(r)
}
/// `wave_sum_dpp` of every value of an array: the chains are independent, so they interleave on the VALU; against
/// `wave_sum_f32`'s five round trips through LDS for all values (phase A of the analog, 2026-10-27).
#[inline(always)]
pub unsafe fn wave_sum_dpp_n<const N: usize>(v: [f32; N]) -> [f32; N] {
    let mut out = [0f32; N];
    let mut k = 0;
    while k < N {
        out[k] = wave_sum_dpp(v[k]);
        k += 1;
    }
    out
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

/// Exclusive prefix sum of one u32 per thread across a 256-thread workgroup (thread order), and
/// the workgroup's total to every thread: a wave scan over `ds_bpermute`, the eight wave totals
/// through `red` (eight u32). Two barriers.
#[inline(always)]
pub unsafe fn workgroup_scan_excl(tid: u32, v: u32, red: *mut u32) -> (u32, u32) {
    let (lane, wave) = (tid % 32, (tid / 32) as usize);
    // the wave's inclusive scan: shifts of 1, 2, 4, 8, 16 lanes
    let mut incl = v;
    let mut d = 1;
    while d < 32 {
        let o = ds_bpermute((lane.wrapping_sub(d)) * 4, incl);
        if lane >= d {
            incl += o;
        }
        d <<= 1;
    }
    let wave_total = ds_bpermute(31 * 4, incl);
    if lane == 31 {
        *red.add(wave) = incl;
    }
    workgroup_sync();
    let mut before = 0u32;
    let mut total = 0u32;
    let mut w = 0;
    while w < 8 {
        let t = *red.add(w);
        if w < wave {
            before += t;
        }
        total += t;
        w += 1;
    }
    workgroup_sync();
    let _ = wave_total;
    (before + incl - v, total)
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
