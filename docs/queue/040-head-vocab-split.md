# 040 — Split the head's vocabulary across both cards

**Status:** Open
**Area:** decode
**Gain:** est. ~1 ms an emitted token

## Proposal
`head` (1.1 ms a step token) and `draft` (1.8) both stream the 1.26 GB lm_head on one card. Split
the vocabulary across the cards; merge the argmax / sampling across the halves.

## Source
Our decode profile, 2026-10-18; tinygrad's argmax per vocab shard exchanged through host memory.

## History
- 2026-10-17: the draft's sync trims measured nothing — the draft lap's floor is the head's 1.26 GB.

## Next
Sampling over a split vocabulary must stay the sampler's exact distribution (top-k/top-p across halves).
