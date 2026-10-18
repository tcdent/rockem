# 019 — Indexer scoring on WMMA

**Status:** Quantified (bench, 2026-10-18) — `ix_score` is 1.6 µs a token per 1k positions a layer, linear: 0.3 ms a prompt token at 18k, 4.7 at 262k, ~18 at 1M; a decode step at 262k 14 ms of scoring, at 1M ~56; `ix_select` at decode 0.07 ms a layer at 65k, ~0.7 at 1M
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

## Next
Build after 018 shares its WMMA staging: scores as `[32 × 128] · [128 × pools]` a token (rows of 16
queries at prefill), ReLU and the head weights in the epilogue, f32 scores; then a split-row select.
