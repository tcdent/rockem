# 015 — Attend scores register-tiled (four positions a thread)

**Status:** Rejected (measured)
**Area:** prefill, decode
**Gain:** measured — 2× slower (3k: kda card 4.4 → 9.2 ms a prompt token)

## Proposal
Each q_abs element read from LDS once for four positions' dots.

## Source
Our design, 2026-10-17.

## History
- 2026-10-17 11:28: bit-identical, 2× slower; not timelined. Likely four concurrent 2 KB
  latent-row streams a thread thrashing L2, or the per-thread score array.

## Next
None; see 018.
