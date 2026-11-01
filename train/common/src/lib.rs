//! Training primitives with no model in them. The inference crates own the
//! byte formats (`common::q4`, FP8 blocks, bf16, exp4) and the forward
//! products over them; this crate owns the *transposed* products — given the
//! gradient at a matrix's output, the gradient at its input, over the same
//! bytes, read in the same order — the backward steps of the elementwise
//! pieces (the clamped SwiGLU, the RMS norm), and the small dense steps an
//! adapter adds. Nothing here is a kernel signature; a model's training crate
//! composes these into its layer backwards, as `models/<name>` composes
//! `common` into its forwards.
//!
//! This crate has `std` (it runs on the host); the GPU twins, when they come,
//! take the `no_std` pieces of the same files under `cfg`, as `common` does.

pub mod bf16;
pub mod fp8;
pub mod lowrank;
pub mod q4;
pub mod rmsnorm;
pub mod swiglu;
