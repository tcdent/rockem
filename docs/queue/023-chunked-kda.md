# 023 — Chunked KDA (WY form, FlashKDA)

**Status:** Deferred — small gain for prefill; returns with longer verify steps
**Area:** prefill, decode (verify)
**Gain:** measured context — `kda_step` is 0.13 ms a prompt token

## Proposal
The delta rule as a product of `(I − β k kᵀ)` compresses into `I − W Yᵀ`: a 16-token chunk
becomes small matmuls plus a 16×16 triangular solve. CHUNK 16 is gfx1100's WMMA shape, and GLM's
gate lower bound (−5) keeps `exp(cumsum g)` in range without rescaling.

## Source
FlashKDA v1 (MoonshotAI, docs/20260420-flashkda-v1-deep-dive.md: 1.85–2.3× over FLA's
`chunk_kda`); compact WY, Bischof–Van Loan 1987 / Schreiber–Van Loan 1989.

## History
- 2026-10-17: deprioritized after the timeline showed `kda_step` small.

## Next
Revisit with deeper drafting (025), where a verify step is a k-token chunk.
