# 050 — DeltaLog for the KDA state

**Status:** Deferred — small gain
**Area:** decode
**Gain:** est. < 0.5 ms a token (our state traffic)

## Proposal
Keep a dense base state plus a log of compact rank-1 updates, merged periodically: exact, less
state write traffic.

## Source
arXiv 2608.15533 (state-update kernel up to 1.86×, write traffic up to 7.83× lower).

## History
- 2026-10-17: listed.

## Next
Only if a timeline shows the state's read/write traffic.
