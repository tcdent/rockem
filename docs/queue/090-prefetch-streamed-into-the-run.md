# 090 — The prefetch streamed into the run

**Status:** Open — a direction (tcdent 2026-10-21), not built
**Area:** decode
**Gain:** unknown; the bound is the share of a layer's posted prefetch done when its run starts (48.6 % of the items,
card13)

## Proposal
tcdent: the decode prefetch is an ordered problem — the order in which bytes arrive matters, and it should be as
sequential as the run that uses them. Today the prefetch is posted at the slot (~600 µs before the layer's run), and
the run, when it starts, overtakes it: whatever is not done is abandoned. Of a layer's 13 posted experts (~156
items), ~76 items are done by then (card13, 2026-10-21); a posted, used expert gets ~72–75 % of its lines from L3
whether complete or partial, an unposted one ~7 %.

The direction: intentionally overlap the prefetch with the run, so the prefetch's span covers the run too and each item
arrives just ahead of the core that will read it — the prefetch a stream timed to the run's consumption order, not a
batch that must finish before it. Prerequisite: arrival order equal to consumption order (the run taking its experts
in the plan's order, built 2026-10-21), so "ahead" is well defined per CCD queue.

## Source
tcdent (2026-10-21); bringup/reads/prefetch.sh (done share, window) and prefetch_use.sh (status, L3 shares).

## Next
Per CCD: how far ahead of a run's item its prefetch must be (the L3 fill latency at the CCD's delivery rate, ~60–70
GB/s shared with the run's own DRAM reads); whether the touch can run on the same cores between items or needs the
CCD's other cores; what happens to the run's own rate when its CCD also streams the prefetch (phase 0: L3 and DRAM
share the CCD's delivery).
