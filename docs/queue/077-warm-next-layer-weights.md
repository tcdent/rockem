# 077 — Warm the next layer's weights while the cards wait for the host

**Status:** Open — an experiment
**Area:** decode (the cards' phase of a layer)
**Gain:** unknown; bounded by the share of the cards' ~0.48 ms a layer that is weight streaming

## Proposal
During a MoE layer's host phase the cards sit idle (~0.35 ms a layer of the host's experts at a verify step,
14.65 ms of a 43.5 ms step at 18k, run 34f5563b), and in exact arithmetic layer k + 1 cannot start: its first
operation (`hc_post`'s fold, then `hc_pre`'s norms and coefficient dots over the four streams) needs layer k's
routed sum. Its weights, though, do not depend on the token. A prefetch launch posted at the start of the wait pulls
layer k + 1's projection rows (the KDA part's `sq`, `fg`, `o`; an MLA layer's q_a, kv_a, q_b, o) into the card's
on-die cache, so the phase after the host's word reads them from cache instead of VRAM. Exact: no model change.

## What decides it
- How much of a layer's card phase is weight streaming rather than launch latency and device barriers.
- Whether a layer's rows a card fit its on-die cache (~80 MB on the 7900 XT from memory, not measured here) and
  whether a prefetch's lines survive until the phase (the residents' jobs and the shared expert run in between).

## First measurement
One layer's card phase twice back to back, the second over warm weights (a bench beside the node, or a lab build
that repeats a layer), against the cold phase; then a prefetch kernel's own time to warm a layer.

## Source
tcdent and our discussion of how late the host's experts may arrive, 2026-10-20.

## History
- 2026-10-20: listed.
