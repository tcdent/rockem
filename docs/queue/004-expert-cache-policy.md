# 004 — The expert cache's replacement policy (ARC)

**Status:** Built (aged counts) — the replay done, the served number pending the 2026-10-18 09:30 deploy
**Area:** decode
**Gain:** unknown; bounded by Belady 0.74 against today's hit rates

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

## Next
The expert lap and resident share after sixteen warm-up rounds (lifetime counts: 6.0 ms a step
token); then Done. The gap to Belady (0.11 / 0.21) is the residency's remaining
ceiling; 031 (sizing by context) and 038 (prefill) change what the cache holds, not how it replaces.
