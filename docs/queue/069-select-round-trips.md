# 069 — The indexer selection's round trips at long context

**Status:** Quantified, on the shelf — ~0.7 ms a verify step past 32k positions (~70 µs a layer)
**Area:** long context (decode)
**Gain:** measured context — `ix_select` 92 µs a layer at 262k, flat to 1M, of which the kernel's own work is a fraction

## Proposal
The cooperative `ix_select` runs its radix passes with a global histogram and a last-arriving-workgroup digit
pick a pass: each pass is a round trip through global memory and `await_word`. Fewer passes (a wider radix on
the first pass from the score range), or the selection fused into the scorer's tail for the single-slice case.

## Source
019's history (2026-10-18).

## History
- 2026-10-19: given its own entry (it lived in 019's Next). Below 32k positions the selection is one slice and
  the round trips are within a workgroup.

## Next
bench-mla-attend's selection timing at 65k–1M with the pass count varied.
