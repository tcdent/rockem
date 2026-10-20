# 080 — Launch and timeline granularity: steps cut too fine

**Status:** Open — noted, not scheduled (tcdent, 2026-10-20: "we should go back at some point and clean up some of
the steps that we incorporate too much granularity in")
**Area:** prefill (launch counts), profiling (the card lanes' readability)
**Gain:** unknown; fewer launches a prefill step and a legible timeline

## Proposal
Some work shows as many small launches on the card lanes where one would do:
- `rows_d2` (the KDA gates' second halves, f_b and g_b) runs in passes of `TOKENS` (32): a 512-token chunk is 16
  launches a KDA part a layer, ~530 a card a prefill step (`cards.sh`, 2026-10-20). It has no matrix-path twin for
  steps past `SPEC_TOKENS`, as the other rows kernels do (`*_w`) — a twin makes it one launch.
- Others seen at the same grain on the lanes: the `copy_f32` handoffs, the split attend's `word_wait` / `word_post`.
Where the work itself should stay split, the timeline can group it (one zone a phase) instead of a zone a launch.

## Source
tcdent reading the full-step prefill card lanes, 2026-10-20.

## History
- 2026-10-20: listed.
