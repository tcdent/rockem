# 011 — iu8 WMMA rate recheck

**Status:** Resolved (measured 2026-10-18) — eight chains 87.4 TOP/s, four 87.1, bf16 84.8: iu8 WMMA runs at the bf16 rate on gfx1100
**Area:** prefill
**Gain:** informs 038; RDNA3 lists int8 at twice fp16

## Proposal
`bench-wmma` measured iu8 87 TOP/s, no faster than bf16 85. Four independent chains a wave may
not cover iu8's latency; rerun with eight.

## Source
`1e39425`, 2026-10-17.

## History
- 2026-10-17: measured at four chains.

## History (cont.)
- 2026-10-18 11:05 (`rate_wmma_iu8_8`, bench-wmma): 87.4 TOP/s at eight chains against 87.1 at four and
  84.8 TFLOP/s bf16. Not a latency artefact: the iu8 16×16×16 WMMA issues at the bf16 rate on this
  card (RDNA3's "2× int8" is the dense-vs-sparse figure, not iu8 vs f16).

## Next
None. For 038: an int8 expert kernel on WMMA buys the bytes (int8 activations, q4 weights unpacked
to int8 in registers), not the matrix rate — size it at 85 T-ops/s.
