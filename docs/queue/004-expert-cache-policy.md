# 004 — The expert cache's replacement policy (ARC)

**Status:** Done (aged counts + a layer turn for the one load a step; measured 2026-10-18 10:29)
**Area:** decode
**Gain:** measured — after sixteen warm-up rounds 30–40 % of picks resident (was 3–4 %), the expert lap 9.3–10.0 → 6.7–7.4 ms a step token, a verify step 26.1–26.9 → 23.6–24.3

## Proposal
LFU wins on code, LRU on chat (003's replay). An adaptive policy between recency and frequency
(ARC) may track both. Replay it on the two traces before changing the residency.

## Source
003's routing replay, 2026-10-17; ARC is Megiddo & Modha, 2003.

## History
- 2026-10-17: proposed; not replayed. The residency today: claim a free slot, or the layer's
  least-picked resident once the miss leads it by `RESIDENT_MARGIN`.
- 2026-10-18 09:10: replayed (`routing_replay.py`, columns `node` and `ARC` added) at the served
  sizes, 21 / 18 slots a layer, hit share code / chat:

  | rule | code@21 | chat@21 | code@18 | chat@18 |
  |---|---|---|---|---|
  | LRU | 0.294 | 0.348 | 0.258 | 0.318 |
  | LFU | 0.405 | 0.285 | 0.372 | 0.259 |
  | node (lifetime counts, margin 4) | 0.406 | 0.285 | 0.374 | 0.260 |
  | ARC | 0.345 | 0.366 | 0.312 | 0.336 |
  | **node, counts halved every 32 positions, margin 2** | **0.438** | **0.364** | **0.405** | **0.335** |
  | Belady | 0.552 | 0.577 | 0.519 | 0.547 |

  ARC trades code for chat (−0.06 / +0.08) and is not a replacement. Aging the node's own counts
  (half-life swept 8–512 positions; 16–48 flat, 32 taken; margin 2 over 4 and 1) beats every online
  rule on both traces, so one path with one more constant. The replay assumes a free load every
  miss; the node loads one a step.
- 2026-10-18 09:12: built — `RESIDENT_HALF_LIFE = 32`, `RESIDENT_MARGIN = 2`, the counts shifted
  right by the half-lives a step crosses (`glm_kfd.rs`). Deployed with the bf16-state code object.

- 2026-10-18 09:50: the profile line now carries the resident share of picks (`N% of picks resident`).
  On the deploy's short decodes after one 79-position warm-up it reads **0–1 %**: the slots fill one
  load a step (one expert of 1,770 slots a step), so the residency is empty for the first ~1,800 steps
  of a boot, and no short measure sees a policy. The 6.0 ms / 24–25 ms figures (003) need the
  sixteen-round warm-up; the policy's served number is taken that way (`deploy14.sh`).

- 2026-10-18 10:05, sixteen warm-up rounds (~2,600 steps) on the aged counts: **3–4 % of picks
  resident**, expert lap 9.3–10.0. Not the policy: the one load a step goes to the *earliest* MoE
  layer with a miss (`inflight` is global, the claim loop runs layer by layer), so the first layers took
  every load and the rest stayed empty — true of yesterday's code too; the 6.0 ms figure in 003 did not
  reproduce under this warm-up.
- 2026-10-18 10:29: a rotating turn (`Resident::turn = ticks % MOE_LAYERS`; layers before the turn
  skip claiming that step) and a console line `N of 1770 resident slots claimed, M loads` every profile
  window. Same warm-up: 1,521 of 1,770 slots claimed after ~2,700 steps, 2,763 loads; **30–40 % of
  picks resident** (the replay's 0.36–0.44 for these texts), expert lap **6.7–7.4 ms a step token**,
  a verify step **23.6–24.3** (0.78–0.83 acceptance on the warm text). SCORE bit-identical, decode
  text unchanged.

## Next
None. The fill takes ~1,800 steps from a cold boot (one load a step); 031 (sizing by the context in
use) and 045 (loads by the card's copy engine, more than one a step) are the levers on that. The gap to Belady (0.11 / 0.21) is the residency's remaining
ceiling; 031 (sizing by context) and 038 (prefill) change what the cache holds, not how it replaces.
