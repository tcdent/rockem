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

## Next
Work out where the deferred add lands in mHC's four streams before any run.
