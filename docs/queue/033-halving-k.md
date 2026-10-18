# 033 — Halving k with a fixed reference mass

**Status:** Rejected (measured)
**Area:** decode
**Gain:** measured −3 ms an emitted token (~21 vs 24–25), at −8 GSM8K points

## Proposal
Run the top 4 of the router's 8 picks, weights normalized by all 8 scores.

## Source
arXiv 2609.04575 (Qwen3.5-397B 10 → 5: −0.55 MMLU points).

## History
- 2026-10-17 22:16 (`a0225a4`): `ROUTED_K`; expert phase 6.0 → 3.9 ms a step token; long SCORE
  NLL 0.0426 vs 0.0446 (invisible on that text); acceptance 0.78–0.85.
- 2026-10-18 (`ade33d7`): GSM8K on the same 408 problems, k = 4 83.3 % vs k = 8 91.2 % (64
  discordant, 48 to k = 8, z ≈ 4). `TOP_K` stays 8. The paired comparison needed ~400 problems;
  the harness's 320-token cap is wrong for a thinking model.

## Next
None.
