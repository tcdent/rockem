# 019 — Indexer scoring on WMMA

**Status:** Deployed 2026-10-18 13:21 (`ix_score_w`: 27–34× the scalar at a 512-step, 2.6–5.3× at a two-token step past 18k; selections within 0.03–0.3 %; SCORE vs the attend-only build: NLL 0.04365 → 0.04362, mean |Δ| 5e-5); the decode selection's split still open
**Area:** long context (prefill and decode)
**Gain:** measured — the whole scoring cost above, ~10× on WMMA; the selection needs its own split (below)

## Proposal
`ix_score` is a thread a (token, pool), scalar. Scoring is
`[32 heads × 128] · [pools × 128]ᵀ` (rows of 16 queries in prefill), then ReLU and the head
weights: WMMA-shaped. With the pool keys in bf16 (030) the operands are already bf16; scores
accumulate in f32 as the reference does.

## Source
tinygrad PR #18863 (`indexer_scores`, rows of 16 queries); vLLM #60618 (scores and top-k in one
launch); our 1M arithmetic (~13 GMAC and ~1.6 GB of f32 pool keys read a decode token).

## History
- 2026-10-18 (`971f5df`): measured — the card wait that grows with position (attend and scorer)
  grows ~0.08 ms a prompt token per 1k positions: ~80 ms a prompt token at 1M as the kernels stand.
- 2026-10-17: proposed.

## History (cont.)
- 2026-10-18 11:15 (`bench-mla-attend`, µs a token a layer): `ix_score` at a 512-step 1.3 / 27 / 106 /
  427 at 2k / 18k / 65k / 262k; at a two-token step 9.6 / 31 / 137 / 640 — linear in the pools, a
  thread a (token, pool) with 32 × 128 MACs. `ix_select` at a 512-step 0.2–2.9 (cheap), at two tokens
  9.7 / 35 / 66 / 182: one workgroup a token scanning every pool — at 1M that is ~0.7 ms a layer a
  decode step, 8 ms a step over 11 layers, so the selection needs a split-row form as well
  (SGLang #43775 / vLLM #60618).

- 2026-10-18 13:15, built (`ix_score_w`: a workgroup scores 16 tokens against 128 pools, a pool tile
  a wave with its eight key fragments in registers for the whole kernel, the queries staged as bf16 a
  round of 8 heads at a time, ReLU and the head weight on each head's tile, summed in registers).
  `bench-mla-attend`, µs a token a layer, scalar → WMMA:

  | positions | 512-step | two-token step |
  |---|---|---|
  | 2k | 1.25 → 0.22 | 9.5 → 24 |
  | 18k | 27.6 → 1.0 | 31.5 → 24 |
  | 65k | 109 → 3.3 | 141 → 49 |
  | 262k | 435 → 12.8 | 652 → 123 |

  At 1M, extrapolated: ~0.6 ms a prompt token over 11 layers (18 before) and ~5 ms a decode step (56).
  Selections against the scalar scorer's: 276–2,048 of 1.05 M positions differ at a 512-step
  (0.03–0.2 %), 0–12 of 4,099 at two tokens — near-tie flips from the bf16 queries (the reference's
  dtype). The two-token form pads 2 tokens to a 16-row tile and is 5 µs a layer slower than the scalar
  below ~4k positions (**Quantified, on the shelf — 0.05 ms a step token at short context**: a
  position-derived twin). One kernel serves both.

- 2026-10-18 13:21, deployed: `ix_select` launches it over the pools the step's last token sees. Served
  at 18k: the 512-step 5.2 ms a prompt token steady, the decode step 28.9 — the scorer's share at 18k is
  small (0.3 ms a prompt token, 0.1 a step); its gain is the long tail (65k–1M). SCORE vs the attend-only
  build: NLL 0.04365 → 0.04362, 42 % of positions moved by a mean 5e-5 (the selection's near-tie flips).

## Next
The selection at decode: `ix_select` is one workgroup a token over every pool (182 µs a layer at 262k,
~0.7 ms at 1M — 8 ms a step over 11 layers): a split-row select. Build after 018 shares its WMMA staging: scores as `[32 × 128] · [128 × pools]` a token (rows of 16
queries at prefill), ReLU and the head weights in the epilogue, f32 scores; then a split-row select.
