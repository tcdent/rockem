# 064 — The host's between-phase laps off the critical path

**Status:** Done (2026-10-20) — the input side moved to the cards by 001, the output side by 002 (the leader card sums the expert rows)
**Area:** decode
**Gain:** measured — 3.6 ms a verify step of host laps on the critical path (route 1.5, sum 2.1) in pieces of 3–40 µs a layer; taken: the join 0.86 → 0.07 (038), the prefill expert handoff 0.2 → 0.11 ms a prompt token (streamed)

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

- 2026-10-19 09:00–09:20 (zones inside the laps, runs 33e4… and 29db…): a verify step at 18k —

  | piece | µs a layer | ms a step | what it is |
  |---|---|---|---|
  | route.pick | 20 | 0.94 | two tokens' post-norm and router pick on the host (inline or on the pool alike: not a round trip) |
  | dispatch | 7.5 | 0.41 | the resident jobs' table and three uploads, two launches, a fence |
  | route.quant | 2.5 | 0.11 | two tokens quantized |
  | sum.y | 41 | 1.77 | the weighted sum over 16 expert rows the cards and the host wrote (DMA'd rows read from DRAM) |
  | sum.handoff | 2–4 | 0.29 | the FFN output into the cards' handoff buffer |
  | join | 1 | 0.07 | the wait for the cards' jobs (was 0.86 before 038's shared output rows) |

  and a 1024-token prefill step: sum.handoff 196 → 113 ms a layer-step with the copy streamed (a plain copy
  into write-combining memory ran at 3.4 GB/s; prefill 4.58 → 4.42 ms a prompt token, the cold 18k prompt
  86.3 → 83.7 s), experts.quant 150–160 ms (066), route.pick 53 ms, sum.y 125 ms. The host's reads of the
  cards' buffers streamed with non-temporal loads changed nothing measurable (they are small). A few tokens'
  routing inline instead of on the pool: 19 → 21 µs — the pool's round trip was never the cost; the pieces
  are host arithmetic over data the cards produced, which is 001's point: every one of these laps exists
  because the host sits between two card phases with the data crossing the link twice.

## Next
The pieces left, each quantified: route.pick 0.94 ms a step (the card computes the router scores already; its
top-k and norm there, the picks handed down), sum.y 1.77 (the weighted sum on the card into its own FFN
input — the host's rows would travel up instead, 002's shape), dispatch 0.41 and route.quant 0.11 (the card
quantizing in `hc_pre`), sum.handoff 0.29. Together 3.5 ms a step, 1.9 an emitted token — and they are
one design, 001's: the host's work between card phases moved to the cards with the data staying there.

- 2026-10-20: the rest taken. 001 moved route.pick, route.quant and the dispatch's uploads onto the FFN card;
  002 moved sum.y and sum.handoff (the leader card sums the rows the producers write into its VRAM): a verify
  step at 18k 45.9 → 43.5 ms. What remains of the host's between-phase laps: `route` 0.31 (the residents'
  dispatch 0.30), `done` 0.05, `slot` 1.67 (beside the cards' work).
