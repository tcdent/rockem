# 024 — Speculative decoding with the draft layer (MTP), depth 1

**Status:** Resolved
**Area:** decode
**Gain:** measured — 36 → ~30.2 ms an emitted token (41 → 34 at 18k); later stacked with 003

## Proposal
The checkpoint's layer 45 drafts one token after every step; the next step verifies two tokens
with rejection sampling (accept with min(1, p/q), else the residual, under the sampler's
truncation); the KDA state rewinds on rejection.

## Source
vLLM #56983 (MTP acceptance 0.81/0.57/0.37), ik_llama.cpp #2548 (+27 % on code with CPU experts),
tcdent's direction (2026-10-17: "I'm not willing to accept that we've hit the floor").

## History
- 2026-10-17 (`bf224d2`, `1855eb7`): built and served; a scale-layout bug in the draft's shared
  expert at first (nothing accepted) — the acceptance gate now runs card and host drafts side by side.
  Acceptance 0.92 on the gate text (draft top-1 0.86), 0.84–0.90 in real sessions.
- The L3 prefetch extended to a verify step's union of picks (expert phase 32 → 23 ms).
- 2026-10-18: tinygrad's verify accepts a draft only if it equals the model's own sample; ours
  accepts with min(1, p/q), at least as often under temperature.

## Next
Depth (025), shared indexer picks (026), model-free drafts (027).
