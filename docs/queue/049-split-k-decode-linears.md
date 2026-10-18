# 049 — Split-K decode linears reduced through LDS

**Status:** Deferred — small gain
**Area:** decode
**Gain:** unmeasured

## Proposal
Split-K waves of one output share a workgroup and add partials through LDS; neighbouring rows
packed up to 4; a wave serves 32/lanes outputs with 32, 16 or 8 lanes each.

## Source
tinygrad `a55bef5a2`, `d8d21c29f`; vLLM PR #61868 (low-batch BF16 split-K GEMM).

## History
- 2026-10-18: vLLM PR #61868 measured split-K for skinny low-row GEMMs (compressor, indexer weights,
  indexer `wk`): −60 µs of a 5.4 ms step.
- 2026-10-17: listed.

## Next
Compare against the rows kernels' one-token width on the decode timeline.
