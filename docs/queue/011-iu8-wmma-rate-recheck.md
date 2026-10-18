# 011 — iu8 WMMA rate recheck

**Status:** Quantified, on the shelf — a measurement (eight chains in `rate_wmma_iu8`), minutes of work; sizes 038
**Area:** prefill
**Gain:** informs 038; RDNA3 lists int8 at twice fp16

## Proposal
`bench-wmma` measured iu8 87 TOP/s, no faster than bf16 85. Four independent chains a wave may
not cover iu8's latency; rerun with eight.

## Source
`1e39425`, 2026-10-17.

## History
- 2026-10-17: measured at four chains.

## Next
Rerun at eight chains before sizing the grouped iu8 expert kernel (038).
