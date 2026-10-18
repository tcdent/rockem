# 006 — The 8th/9th routing boundary: resolve exactly or exploit

**Status:** Decision (tcdent) — the speed half estimated 2026-10-18 at ~0.02 ms a step token; the determinism half is not a speed item
**Area:** decode, determinism
**Gain:** exploit — ~0.3 % more resident picks (4–13 % of positions near a tie × ~23 % chance the 9th is resident and the 8th not at 35 % residency × one pick of eight): ~0.02 ms a step token, an estimate (the routing trace keeps only the top eight, so no replay); resolve — reproducibility, at a cost in router arithmetic

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
tcdent's call on determinism. Exploiting is on the shelf by its estimate (~0.02 ms a step token); a trace that records the 9th pick and the margin would make it a replay.
