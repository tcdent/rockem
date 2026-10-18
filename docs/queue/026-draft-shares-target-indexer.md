# 026 — The draft layer reuses the target's indexer picks

**Status:** Deferred — small gain
**Area:** decode
**Gain:** vLLM measured +2.7 % decode (~0.6 ms an emitted token, est.)

## Proposal
The checkpoint sets `index_share_for_mtp_iteration: true`: draft steps reuse the target's indexer
top-k instead of running their own.

## Source
vLLM #61079 (accepted tokens per draft step 3.28 → 3.42); SGLang `IndexTopKShareState.from_mtp_carry`.

## History
- 2026-10-17: our draft uses its own indexer (it has the weights).

## Next
When sub-millisecond gains are picked up; larger at long context, where the indexer grows.
