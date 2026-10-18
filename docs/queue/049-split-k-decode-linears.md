# 049 — Split-K decode linears reduced through LDS

**Status:** Deferred — small gain
**Area:** decode
**Gain:** unmeasured

## Proposal
Split-K waves of one output share a workgroup and add partials through LDS; neighbouring rows
packed up to 4; a wave serves 32/lanes outputs with 32, 16 or 8 lanes each.

## Source
tinygrad `a55bef5a2`, `d8d21c29f`.

## History
- 2026-10-17: listed.

## Next
Compare against the rows kernels' one-token width on the decode timeline.
