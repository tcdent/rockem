# 036 — Bit-plane expert storage

**Status:** Experiment (approximate) — not evaluated
**Area:** decode
**Gain:** est. ~19 % fewer expert bytes (low-weight picks read 2–3 planes)

## Proposal
Store experts high bit-plane first. One layout gives precision per pick by gate weight, progressive
prefetch (high planes of uncertain predictions first), and a bit-serial kernel's input.

## Source
Embedded zerotree coding (Shapiro 1993), Stripes (2016); recent DPA-MoE, SliceMoE, MatQuant.

## History
- 2026-10-17: reached independently and found in the literature. Risk: coarser low-weight outputs
  feed later routers and may add flips.

## Next
Measure on the oracle (KL, picks flipped) before a layout change.
