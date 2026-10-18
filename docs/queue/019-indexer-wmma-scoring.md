# 019 — Indexer scoring on WMMA

**Status:** Open
**Area:** long context (prefill and decode)
**Gain:** est. — at 256k ~1–2 ms a prompt token; at 1M decode several ms a token avoided

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

## Next
After 030's second phase; a timeline at 64k–256k to size it.
