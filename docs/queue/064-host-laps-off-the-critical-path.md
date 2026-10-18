# 064 — The host's between-phase laps off the critical path

**Status:** Open
**Area:** decode
**Gain:** est. 2–3 ms a two-token verify step (the `slot` lap 1.9 and part of `route` 2.7, measured at 18k)

## Proposal
Between a layer's card phases the main thread does host work that the next phase waits on: the
next-layer prefetch (the `slot` lap: the next layer's picks from the layer-ahead rows, the union against
the residents, the L3 prefetch issue — 1.9 ms a step), and the router lap (`route` 2.7 a step: the
host's router, the join's sum, the accounting, the layer-ahead rows). The cards wait while it runs.
Move what is not on the dependency chain to a side thread or the card: the prefetch issue and the
accounting can trail the step; the layer-ahead picks can be computed on the card that already holds the
rows (005's rows are its output), handed back with the layer's result.

## Source
The prow zones of run e03735b6 (`zones.sh` over a decode window: `slot` 0.95 a step token, `route`
1.35); 041 (the route lap's growth when it took on the layer-ahead rows); 059 measures the host loop's
total occupancy, this entry is the fix for its two largest laps.

## History
- 2026-10-18: proposed from the baseline's zones.

## Next
1. 059's read: the main thread's busy time against the step, to see what else hides in the laps.
2. The prefetch issue to a side thread (an exact change: the prefetch is advisory); measure the `slot`
   lap and the expert lap together, since a late prefetch costs where it was meant to save.
3. The router's host work: what must precede the experts (the picks) and what can trail (the
   accounting, the layer-ahead rows into the next layer's prefetch).
