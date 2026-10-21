# 082 — Expert prediction on the cards' idle time

**Status:** Open — proposed (tcdent, 2026-10-20); the prefetch's yield is now measured
**Area:** decode (the CPU experts' L3 prefetch), prefill (idle card time)
**Gain:** est. ~2 ms a verify step per +10 % of prediction precision (decode agent's estimate, 2026-10-20); up to ~4.3
ms a verify step if every host expert were fully prefetched at today's measured worth

> "prefill certainly still has room for GPU operations. But perhaps we task the GPU with helping us decide which
> expert we're going to need. That's obviously hand-wavy and abstract, but it could be an interesting optimization.
> To incorporate with the spare processing time that we have later" — tcdent, 2026-10-20

## Proposal
The CPU experts' L3 prefetch is only as good as its prediction. Today's predictor is the layer-ahead router the
cards already run (`hc_route_ahead`: layer l+1's router rows over layer l's FFN input). The cards have idle time (in
prefill ~7 ms a card a layer-chunk; in decode the waits inside a step), so a stronger predictor can run there:
candidates include the next layer's router over a better estimate of its input (e.g. the next layer's attention
approximated), a wider candidate list ranked by confidence, learned co-routing, and the draft tokens' picks.

## What is measured (2026-10-20, `prefetch_use` rows, run of build pfuse, 18k context, 9,471 decode layer runs)
- A layer run has ~16.6 host experts: 7.2 fully prefetched, 4.5 partly, 4.9 (30 %) not predicted; the prefetch
  predicts 16.9 a layer and ~31 % of them go unused.
- Each layer's expert time fitted on those counts (R² 0.67): 31.0 µs an expert fully prefetched, 36.4 partly, 42.4
  not predicted.
- `bench-reuse` on 92 cores: an L3-served (neighbour-touched) gemv runs 2.2× a cold one, so a prefetched expert
  could be ~19 µs if its lines survived to the run; today they mostly do not (docs/metatron-cpu.md, §5).

- 2026-10-20, build fills2 (per-item fill counters by prefetch status): complete experts' lines 71.5 % from L3, partial
  51 %, not predicted 6.9 %. Estimated worth a verify step: ~2.5 ms if the 30 % unpredicted experts were predicted and
  completed (this entry), ~3.6 ms if completed prefetches kept their lines (the L3's own question), ~1 ms partial.

- 2026-10-20, builds sets: the L3 keeps less of a completed prefetch the more the layer prefetches (86 % at ≤ 10 experts,
  56 % at 23+; 87 / 81 / 71 / 62 % at 1 / 2 / 3 / 4 tokens) — capacity, not set conflicts (1.2 % of lines past 16 in
  their set). So precision is worth more than its share of picks: every unused prefetched expert also evicts used ones.

## Next
First make a completed prefetch worth its 2.2× (why the lines are gone by the run: eviction or timing), since better
prediction multiplies that worth; then the predictor's precision against the `prefetch_use` rows.
