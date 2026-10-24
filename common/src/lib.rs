//! rockem `common` — primitives with no model in them.
//!
//! Everything here is `#[inline(always)]` and none of it is a kernel: a
//! primitive only ever exists inside a model's fused per-layer kernel, where
//! the compiler flattens it into straight-line code with the model's
//! constants as literals. That is the composition mechanism on both targets;
//! on `amdgcn` the pieces are lane- and wave-shaped, on `x86_64` they are
//! vector-shaped, and a model's kernel picks by `cfg`.
//!
//! Every shape is a const, every buffer a caller-owned slice, every function
//! has one signature and no runtime branch.
//!
//! # Conventions, for every kernel file in this tree
//!
//! - **Nouns:** `Weights<'a>` borrows slices into the bundle; `Scratch<'a>`
//!   borrows caller-owned buffers; a `ScratchBuf` (std) is its owning twin
//!   and is allocated in exactly one place per binary. One verb per role:
//!   `step`, `forward`, `dense`, `moe`. The verb says what the kernel does
//!   for the model; the nouns never vary.
//! - **Realizations:** the bare name (`gemv`, `bf16`, `forward`) is the
//!   realization this target ships and selects by `cfg`; `<name>_scalar` is
//!   the target-neutral twin it is compared to, byte for byte. Fast twins
//!   live in modules named after the ISA they need — `avx2`, `gfx11` —
//!   never after a tier. A new ISA is a new module, not a flag.
//! - **The scalar twin is written in the fast form:** it commits to the
//!   same summation order and the same flush expression, because it exists
//!   to be bit-matched, not to be naive.
//! - **`cfg` appears on `mod` and `fn` declarations only**, never inside a
//!   body. Placement is decided at construction; nothing probes at runtime.
//! - **Constants cross a boundary once.** `config.json` is transcribed into
//!   consts by `build.rs`; anything a script or a launcher needs is emitted
//!   by Rust into a meta file and read from it, never retyped.
//! - **The header comment is the spec**, and the one non-obvious choice in
//!   the file is stated there. Structural facts are `const _: () =
//!   assert!(…)` next to what they guard.
//! - **Test data comes from `common::testing`** (a seeded xorshift), so two
//!   tests never disagree about what "random" meant.
#![no_std]
#![cfg_attr(target_arch = "amdgpu", feature(link_llvm_intrinsics, stdarch_amdgpu, core_intrinsics, gpu_intrinsics, gpu_launch_sized_workgroup_mem, asm_experimental_arch, repr_simd, simd_ffi))]
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments, internal_features)]

#[cfg(any(test, feature = "std"))]
extern crate std;

#[cfg(target_arch = "amdgpu")]
pub mod gfx11;
#[cfg(all(feature = "std", target_os = "linux"))]
pub mod alloc;
pub mod copy;
pub mod gemv;
#[cfg(any(test, feature = "std"))]
pub mod linalg;
pub mod math;
#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub mod pmc;
pub mod pool;
#[cfg(any(test, feature = "std"))]
pub mod testing;
pub mod q4;
pub mod rmsnorm;
pub mod rope;
pub mod sample;
#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub mod sgemm;
pub mod softmax;
pub mod stats;
pub mod swiglu;
pub mod trace;
