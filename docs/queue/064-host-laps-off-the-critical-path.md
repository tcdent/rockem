# 064 — The host's between-phase laps off the critical path

**Status:** Measured (2026-10-19); the laps named and sized, one experiment without gain, the next on the card
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
- 2026-10-19 00:07: the `route` zone had spanned the whole host FFN phase (the experts inside it), 17 ms a
  step in the chart against 3.7 in the profile. The phase is now four laps — `route` (the picks and the
  residents' dispatch, with a `dispatch` zone inside), `experts`, `join`, `sum` (the weighted sum, the
  handoff back, the accounting) — and the profile's `join` is no longer counted inside `route`. At 18k, a
  two-token verify step (run ecdb1cb5): route 1.4 ms (dispatch 0.37 of it), sum 1.1, join 0.8 — the host
  work the cards wait on — and slot 1.6, hc 0.3, which run while the cards work (the host's wait for the
  cards, `kda_card`, is 19.2 of the 44 ms step). So 3.3 ms a step is on the host's critical path, in laps
  of 15–35 µs a layer: launch-and-upload granularity, not compute.
- 2026-10-19 00:23: the two tokens' weighted sum on the main thread instead of the pool: 25 against 22 µs a
  layer — reverted; the pool's round trip is not the cost.

## Next
1. The route lap's host quantization and its three uploads (xq, xs, xg a layer): the card holds the normed
   FFN input already, so `hc_pre` can quantize it there and the dispatch upload only the job list — est.
   ~0.5 ms a step, exact (the quantizer's arithmetic is integer).
2. The join (0.8 ms a step): the cards' jobs posted before the host's run begins already; what remains is
   the fence and the host-memory read — a word the kernel's last workgroup writes would replace the fence.
3. The slot lap's overlap with the cards is inferred from the structure (the card's next attention block runs
   while it computes); a run with the prefetch issued from a side thread would confirm it.
