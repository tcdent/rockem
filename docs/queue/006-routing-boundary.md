# 006 — The 8th/9th routing boundary: resolve exactly or exploit

**Status:** Decision (tcdent)
**Area:** decode, determinism
**Gain:** exploit — some resident hits; resolve — reproducibility across realizations

## Proposal
4–13 % of positions a layer have 8th/9th margins under 1e-3; flips there are arithmetic noise,
and an unforced flip gives a different path of equal quality (corpus NLL 2.3156 vs 2.3162).
Two mutually exclusive uses:
- Resolve: compute the router with an error bound, recompute only boundary scores in compensated
  arithmetic (Shewchuk's adaptive predicates, 1997; Ogita–Rump–Oishi, 2005). Removes
  arithmetic-caused flips; also the indexer's 2048th key.
- Exploit: inside the noise band, pick whichever candidate is resident.

## Source
Our measurements (2026-09-30, 2026-10-15); the old-results research, 2026-10-17. The GLM-5
report: non-deterministic top-k "caused drastic performance degradation during RL".

## History
- 2026-10-17: posed as a fork; not decided.

## Next
tcdent's call. Exploiting has something to prefer now that 003 exists.
