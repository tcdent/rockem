# 022 — hc_pre for prefill

**Status:** Deferred — small gain
**Area:** prefill
**Gain:** est. ~0.2 ms a prompt token (0.35 measured at 512, a per-launch cost)

## Proposal
`hc_pre` uses decode's shape (a device-wide arrival, the last workgroup builds every token's mix).
For prefill: the 24 × 16384 coefficient GEMV as a WMMA GEMM over the step's tokens, then a
Sinkhorn wave a token.

## Source
tinygrad `a76fc2c6b` (the coefficient linear on int8 tensor cores — a precision deviation for us;
bf16 WMMA would follow the reference), `5ac204148` (one kernel for coefficients, mix and norm).

## History
- 2026-10-16/17: hc_pre made multi-token (`d075a4d`), rounds of 128 for LDS residency (`340bd2d`).

## Next
A prefill pair when sub-millisecond gains are picked up.
