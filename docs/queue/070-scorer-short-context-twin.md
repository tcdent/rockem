# 070 — The indexer scorer's short-context form

**Status:** Quantified, on the shelf — 0.05 ms a verify step
**Area:** decode
**Gain:** measured context — `ix_score_w` at two tokens stages 16 token columns for two; the scalar `ix_score` is kept beside it

## Proposal
At a verify step the WMMA scorer runs its 16-token tiles over two real tokens; a two-token form (a wave a pool
block with the two queries in registers) would read the pools once and skip the staging. 0.05 ms a step at 18k.

## Source
019's history (2026-10-18): scorer 2.6–5.3× at two tokens against 27–34× at a 512 step.

## History
- 2026-10-19: given its own entry (it lived in 019's Next).

## Next
Only with the other sub-millisecond decode items.
