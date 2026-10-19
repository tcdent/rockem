# 074 — The decode expert runs' idle

**Status:** Open — measured 2026-10-19, the largest unexplained slice of a verify step's expert phase
**Area:** decode
**Gain:** up to ~4 ms a verify step (~2 ms an emitted token): the pool is 73 % busy over the gate/up runs and 62 % over the down runs at a verify step

## Proposal
At a verify step a layer's expert run is ~700 gate/up items of one or two tokens (128 KB each at the bandwidth
floor, ~20 µs) over 92 cores, 7–8 items a core: the run's wall is set by the last core, and the items a core
gets vary with the slotting (a slice's node and core complex are fixed by its placement; the pool's slot order
steals within a node, never across). `track-experts` reads the items' summed time at 73 % (gu) and 62 % (down)
of 92 cores over the runs' wall — 2.2 + 2.0 ms a step of cores waiting for the last item. Finer items for a
step of a few tokens (`SUB` 4 at decode: half the item, twice the count), cross-node stealing at a run's tail,
or the two runs merged so the down items start as their gate/up parts finish.

## Source
021's instrument over the 328-token decode of run 6274d8fa (2026-10-19 14:00): gu items 103,630 ms over 185
steps against 1,540 ms of run wall; down 55,334 against 964.

## History
- 2026-10-19: found while closing 021; the prefill runs are 94 % busy, so this is decode's shape alone.

## Next
1. The tail per run: the last item's end against the median core's (the pool's `node_end` by node is in the
   `step_node` rows; a per-core end would say whether it is one node or one core).
2. `SUB` 4 for steps of a few tokens (derived from the step, no toggle) as the first measure.
