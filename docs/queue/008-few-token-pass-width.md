# 008 — A pass width for 2–4 tokens (verify steps)

**Status:** Resolved; the FP8 sub-gap is Deferred — small gain
**Area:** decode (verify)
**Gain:** measured — a verify step 60 → 58 ms; FP8 launches still 1.7–2× a token's

## Proposal
A two-token step ran the eight-wide pass: bf16 launches 1.35× a token's, FP8 2–3×. A third
width `FEW = 4` with four-step rounds for 2..=4 tokens.

## Source
Our timeline of the verify step, 2026-10-17.

## History
- 2026-10-17 15:23 (`bc45af0`): the card's two-token step 21.7 → 17.6 ms of kernels; bf16 1.18×.
- 2026-10-17 15:51 (`1e39425`): one or two rows a wave at that width measured worse (FP8
  `ffn_down_shared` 1277 / 1208 µs a step vs 994 at four rows) — the FP8 gap is not occupancy.

## Next
Small-gain item: find what holds FP8 at 1.7–2× (decode table, the 16-slot slab, registers) when
sub-millisecond work is picked up.
