# 054 — LUT GEMM for the CPU experts in prefill

**Status:** Deferred — needs 021's answer first
**Area:** prefill
**Gain:** unknown

## Proposal
Bit-plane lookup GEMM (T-MAC, LUT-GEMM): activations' partial sums precomputed once a token,
each weight a `vpshufb` lookup instead of a multiply. Declined for decode (DRAM-bound); prefill's
CPU phase is partly compute.

## Source
T-MAC, LUT Tensor Core, BitNet.cpp; the Method of Four Russians (1970).

## History
- 2026-09-20: LUT kernels declined for decode.
- 2026-10-17: prefill recheck noted; reported gains are at 1–2 bits, not 4.

## Next
Only if 021 shows the kernel's compute is the bound.
