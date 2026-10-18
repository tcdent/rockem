# 051 — Exact top-2048 indexer with block-max pruning

**Status:** Deferred — speculative gain; matters at long context
**Area:** long context
**Gain:** depends on how keys cluster (unmeasured)

## Proposal
Per-block per-dimension min/max of the pooled keys bound each block's ReLU-weighted score; skip
blocks below the running 2048th score, seeded from the previous token. Exact (the dense==sparse
oracle holds).

## Source
Fagin–Lotem–Naor threshold algorithm (1996/2003); Block-Max WAND (Ding–Suel 2011).

## History
- 2026-10-17: proposed by the old-results research.

## Next
Measure key clustering at 64k–256k before building; weigh against 019.
