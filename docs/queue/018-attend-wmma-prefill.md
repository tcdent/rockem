# 018 — Sparse attend on WMMA for prefill (heads as rows)

**Status:** Open
**Area:** prefill, long context
**Gain:** est. ~1–1.5 ms a prompt token at long positions (the 18k vs short gap is 1.5 ms)

## Proposal
All 64 heads share a token's latents and selection, so scoring is
`[64 heads × 512] · [selected × 512]ᵀ` — WMMA-shaped even for one query — followed by softmax and
the latent mix. Each latent loaded once for all heads. A prefill kernel pair.

## Source
DeepSeek FlashMLA's decode on tensor cores; tinygrad's `flash_attention_mla_sparse`; the attend
experiments 014–016 (every scalar shape tried; this one untried).

## History
- 2026-10-18 (`971f5df`): measured — the card wait that grows with position (attend and scorer)
  grows ~0.08 ms a prompt token per 1k positions: ~80 ms a prompt token at 1M as the kernels stand.
- 2026-10-17: proposed; not built.

## Next
Build as the prefill attend pair, gated numerically.
