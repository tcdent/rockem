# 074 — The decode expert runs' idle

**Status:** Resolved (measured 2026-10-19) — not idle: the run is at the box's DRAM ceiling; finer items measured no gain
**Area:** decode
**Gain:** none by balancing — a verify step's expert run streams 60 MB in 160 µs, 375 GB/s, the socket pair's ceiling; the 27 % of cores done early are the shape of a bandwidth-bound run, not a lever

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

- 2026-10-19 08:00 (`pool_run` rows: each worker's end a slotted run, run 2e68…): a verify step's expert run
  at 18k is 467 items over 92 cores in 160 µs of wall; the workers wake in 3 µs, the 10th-percentile core is done
  at 112 µs, the median at 127, the 90th at 151, the last at 160 — and every node ends alike (140–143 µs), so
  it is not a slow node or a slow core but the grain: five items a core of 20–30 µs each, the last core drawing
  one more. A prefill run (11,300 items, 15 ms) has the same shape at 6 % (14.1 → 15.0 ms). So the tail is
  the item size at a few tokens: `SUB_FEW` = 4 (16-row gate/up parts, 64-row down parts, derived from the step
  width) doubles a run's items and should halve the tail, ~1.3 ms a step est.

- 2026-10-19 08:17 (`SUB_FEW` 4: 16-row gate/up parts and 64-row down parts for a step of a few tokens, run
  a6707520): twice the items a run (924), the tail 20.4 → 15.6 % (the median core done at 133 of 158 µs
  against 127 of 160) — and the run's wall unchanged, the experts lap 15.6 → 16.3 ms a step, the verify step
  46.1 → 46.8. The finer items cut the tail and cost it back in per-item overhead and activation re-reads.
  Reverted; one item split for every step width. The arithmetic that closes this: a run's items are 467 ×
  128 KB = 60 MB in 160 µs = 375 GB/s, the two sockets' DRAM ceiling (300 measured with the slab bench, the
  loader's and the handoffs' share beside it): the run is bandwidth-bound end to end, and the early-finished
  cores are what a bandwidth-bound run looks like when the last items are in flight. The decode expert phase
  is bytes over the ceiling — 2.5 GB a verify step at 375 GB/s ≈ 6.7 + down's 5.8 of the 15.6 ms lap — and
  its levers are bytes (more residents, 031; a tighter format, 035/036/055) and the ceiling (`[[milan-bandwidth-targets]]`), not the pool.

## Next
None. The `pool_run` rows (each worker's end a slotted run) and `bringup/reads/pool.sh` stay as the read.
