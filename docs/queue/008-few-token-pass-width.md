# 008 — A pass width for 2–4 tokens (verify steps)

**Status:** Reopened 2026-10-18 — the FP8 rows kernels run at 275 GB/s at one token (decode-bound, ~2.5 ms a step token on the table)
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

## Measured 2026-10-18 10:30 (`bench-fp8-rows`, the kernels alone on the deployed code object)
`ffn_gu` (the shared experts' gate|up, 4096 × 4096 FP8 = 16.8 MB, norm folded in) and `ffn_down_shared`
(4096 × 2048 = 8.4 MB), µs a launch (a token):

| tokens | ffn_gu | ffn_down_shared |
|---|---|---|
| 1 | 61.0 (61.0) | 40.1 (40.1) |
| 2 | 73.2 (36.6) | 52.9 (26.4) |
| 4 | 73.5 (18.4) | 57.8 (14.5) |
| 8 | 108.9 (13.6) | 77.0 (9.6) |
| 16 | 186.9 (11.7) | 134.0 (8.4) |
| 32 | 327.5 (10.2) | 243.0 (7.6) |

Alone, a two- or four-token FP8 launch costs 1.2–1.4× a token's, not the 1.7–2× the served timeline
showed (the rest is the step's launch gaps, not the pass width). The larger number is the one-token
rate itself: **16.8 MB in 61 µs is 275 GB/s** and 8.4 MB in 40 µs is 210 GB/s, against ~800 GB/s of
VRAM — the FP8 rows kernels are decode-bound at every width, not bandwidth-bound. A step's dense FP8
weights (~1.1 GB a token across the layers) at 275 GB/s are ~4 ms of the ~8.8 ms `kda card` lap; at
the memory rate they would be ~1.5. **That is the item: the FP8 decode (e4m3 → f32 through the LDS
table, the block scale) at one to four tokens, ~2.5 ms a step token.** Reopened on that basis.