# 010 — Ozaki scheme: exact products from int8 slices

**Status:** Rejected (for the rows pair) — kept for any future order-independence need
**Area:** prefill, determinism
**Gain:** not measured; 9–16 iu8 products a multiply

## Proposal
Slice bf16 weights and f32 activations into int8 pieces, multiply on iu8 WMMA with exact int32
accumulation: results independent of reduction order (any tiling, any split agrees).

## Source
Ozaki, Ogita, Oishi, Rump (2012); recent LLM batch-invariance use (arXiv 2609.11356); RIKEN GEMMul8.

## History
- 2026-10-17: the rate headroom measured (iu8 WMMA 87 TOP/s vs rows kernels ~1.4 TFLOPS).
- 2026-10-18 (`c014153`): rejected for the rows pair — the reference runs these projections in
  bf16, so plain bf16 WMMA is the reference's arithmetic; prefill no longer asks for bit-identity.

## Next
Returns if a tensor split or a decode kernel needs order-independent sums.
