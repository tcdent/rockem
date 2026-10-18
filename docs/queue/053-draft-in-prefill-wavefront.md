# 053 — The draft layer as one more layer of the prefill wavefront

**Status:** Deferred — small gain
**Area:** prefill
**Gain:** est. ~0.15 ms a prompt token (~3 % of the work, serialized)

## Proposal
The draft layer runs over every prompt position as its own pass after each step (its cache needs
every position). In a wavefront it overlaps like any other layer.

## Source
Our inventory, 2026-10-18; tinygrad's `_mtp_prefill` does the same work.

## History
- 2026-10-18: listed.

## Next
With 037.
