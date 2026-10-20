# 034 — Expert deferral

**Status:** Experiment (approximate) — not evaluated
**Area:** decode
**Gain:** reported +33 % decode at −0.5 % LiveBench (DeepSeek-V3)

## Proposal
Run the top-I experts at layer k; add the rest's outputs at layer k+2, so the CPU's experts
overlap the cards' next layer.

## Source
KTransformers, SOSP '25.

## History
- 2026-10-17: listed. With 003 the CPU's share of the expert phase is smaller, which changes the
  overlap available.

- 2026-10-20: tcdent tied it to 002 (the leader card). The leader protocol is the mechanism: producers already
  write their rows into the leader's VRAM and raise a word, and the leader sums them in pick order with the
  weights in VRAM. Deferral is a second word: the on-time picks waited for at layer k, the deferred ones at k + 2,
  summed there the same way. The overlap to take is the cards' wait on the host's experts (14.65 ms of a 43.5 ms
  verify step at 18k, run 34f5563b). The split to try first is by where an expert runs (the host's picks deferred,
  the residents on time) rather than by rank.

## Next
Work out where the deferred add lands in mHC's four streams before any run: layer k's output enters the streams
through layer k's post coefficients, and two comb mixes have run by k + 2, so the late add is approximate by
construction. The first form: the deferred sum scaled by layer k's post coefficients (the leader keeps them),
added to the streams at k + 2, judged by the long SCORE NLL distance and the evals.
