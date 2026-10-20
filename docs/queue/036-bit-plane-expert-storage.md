# 036 — Bit-plane expert storage

**Status:** Experiment (approximate) — reviewed 2026-10-20: no exact form gains; held until accuracy trades open
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

- 2026-10-20: reviewed for an exact form (accuracy held fixed, tcdent). Findings:
  - Every plane read is bit-identical but the same 4 bits a weight: no bytes saved, and the kernel loses — the
    CPU's nibble kernel is ~0.15 instructions a weight; four planes recombined (bit-serial passes) is plausibly
    0.6–1 against a Zen 3 budget of ~1, which could turn the DRAM-bound expert run compute-bound.
  - The one exact benefit is progressive prefetch: the layer-ahead predictor (~5 of 8 right) could fetch only the
    high planes of uncertain predictions. In decode the prefetch runs while the cards work and DRAM is otherwise
    idle, so the misses' bytes cost ~nothing today; it pays only if prefetch comes to compete with the run.
  - Lossless coding of the planes is 055's question (deferred, likely a dead end).
  - The approximate form is truncation: two planes of a 4-bit RTN weight are a worse 2-bit than a rounded one
    (MatQuant trains for nesting); ds4 measured proper Q2 at NLL 0.458 against 0.300 for Q4 on this model.
    Ceiling ~19 % of the host's experts, ~2.8 ms of a 43.5 ms verify step at 18k; prefill reads every expert
    whole either way.
  Verdict: nothing to build while accuracy is fixed.

## Next
If accuracy trades open: measure on the oracle (KL, picks flipped) before a layout change.
