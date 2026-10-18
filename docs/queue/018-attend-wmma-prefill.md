# 018 — Sparse attend on WMMA for prefill (heads as rows)

**Status:** Quantified (bench, 2026-10-18) — the attend is 151–160 µs a token a layer at a 512-step from 2k positions on: ~1.7 ms a prompt token over 11 layers, a quarter of the 6.7; ready to build
**Area:** prefill, long context
**Gain:** measured ceiling ~1.7 ms a prompt token at any context past 2,050 positions (the kernel runs at ~1.8 TFLOP/s; WMMA takes the same work ~10× down)

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

## History (cont.)
- 2026-10-18 11:15 (`bench-mla-attend`): `mla_attend` over a 512-token step 151 µs a token a layer
  at 2k positions, 160 at 18k, 161 at 65k, 155 at 262k — constant once 2,048 positions are selected.
  Per token a layer the work is 64 heads × 2,048 positions × 512 for the scores and again for the
  mix, 0.27 GFLOP, so the kernel runs at ~1.8 TFLOP/s against the rows pair's 18–26 on WMMA.
  Scoring is `[64 × 512] · [512 × 2048]` a token: 4 head tiles × 128 position tiles × 32 k-steps,
  the latents staged once for all heads. The scalar kernel at four heads a workgroup (016) gains 6 %,
  so the latent reads are not where the time goes; the per-thread dots are.

## Next
Build as the prefill attend pair (and 017's decode form), gated numerically: top-1/KL on the step
gate text at long positions and the long SCORE.
