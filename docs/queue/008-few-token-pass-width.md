# 008 — A pass width for 2–4 tokens (verify steps)

**Status:** Open, one step taken 2026-10-18 — the f16-transplant decode deployed (−15–18 % on `ffn_gu`, bit-identical); `ffn_down_shared`'s limiter still to find
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

- 2026-10-18 10:50, two decodes tried (`bench-fp8-rows`, median µs, `ffn_gu` / `ffn_down_shared`; the
  bf16 reference `rows_h` 4096² at one token 34.7 µs — the bench's weights sit in the 80 MB Infinity
  Cache, so absolute GB/s are cache rates; the ratios hold):

  | tokens | LDS table | f16 bit transplant | transplant + 16 B a lane over a step pair |
  |---|---|---|---|
  | 1 | 59.9 / 37.5 | **48.8** / 36.5 | 42.9 / 36.1 |
  | 2 | 70.3 / 50.9 | **59.3** / 49.2 | 62.8 / 49.3 |
  | 4 | 70.5 / 51.0 | **59.7** / 49.6 | 63.1 / 49.6 |
  | 8 | 98.6 / 75.5 | **92.0** / 74.4 | 103.2 / 73.2 |
  | 32 | 319.6 / 243.2 | **291.0** / 239.0 | 310.5 / 234.4 |

  The transplant (sign to bit 15, the seven magnitude bits one down; the f16 is the e4m3 value × 2⁻⁸
  exactly, subnormals included; the 2⁸ folded into the block scale, so `w · scale` rounds as before —
  bit-identical to the table except at the NaN codes) is kept: −18 % at one token, −15 % at a verify
  step's two to four, on `ffn_gu`; `ffn_down_shared` barely moves (−3 %), so its limiter is not the
  decode. The step pair wins only at one token and loses at every other width; not kept. Deployed
  as `kernels-fp8t` 10:51: the long SCORE **bit-identical**, decode text and acceptance unchanged; served, the
  MLA projection lap 2.8 → 2.6 ms a step token and `kda card` 8.8 → 8.7 (the FP8 shapes are a part of
  each).

## Next
`ffn_down_shared` at 8.4 MB in 36 µs: not the decode — find what holds it (its SwiGLU staging, the
emit, the short K = 2048 rows) with the bench; the same question for the other FP8 shapes (mla_qb,
mla_o). The ceiling on the table stays ~2 ms a step token.