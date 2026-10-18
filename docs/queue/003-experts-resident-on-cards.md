# 003 — Routed experts resident on the cards

**Status:** Resolved (decode)
**Area:** decode
**Gain:** measured — expert phase 10.5 → 6.0 ms a step token; ~24–25 ms an emitted token (from 29.7)

## Proposal
The cards idle through the expert phase and have ~12 GB free each. Keep a cache of experts in
VRAM, refreshed within the conversation; run a step's resident picks there while the host runs
the rest; join on the host in pick order.

## Source
Our measurements (free VRAM, routing concentration), 2026-10-15 and 2026-10-17.

## History
- 2026-10-17 15:32 (`9c2a066`): routing trace and replay — at 48 experts a layer LRU 0.49/0.53,
  LFU 0.62/0.47, Belady 0.74/0.74 (code/chat); a static hotlist from code on chat 0.19.
- 2026-10-17 17:20 (`f4a1f97`): `xpt_gu`/`xpt_down`, bit-identical to the CPU (32768/32768
  outputs), 19 µs an expert at 4 jobs.
- 2026-10-17 19:02 (`9ddf9d2`): integrated — jobs posted before the output half (queued behind
  it they deadlock), the loader paced to one load a step (flat out it doubled every card lap).
- 2026-10-17 21:23 (`396c07f`): settled — loader on a housekeeping core, a self-check for traced
  runs (zero mismatches over ~4,650 steps). Warmed: 23–24 ms a step token (kda card 8.6, mla 2.7,
  experts 6.0, router 2.6, draft 1.8, head 1.1, slot 0.7).
- 2026-10-18 (`ade33d7`): `rockem.resident=<GB>` sizes the residency.
- 2026-10-18 09:00 (boot of `971f5df`): 938 / 832 experts resident (12.5 / 11.1 GB; 21 / 19 a layer). After a single 79-position warm-up the expert lap on the short decode text reads 9.0–9.9 ms a step token (26–30 ms a step token, 0.88 acceptance → ~28 an emitted token); the 6.0 / 24–25 figures are after sixteen warm-up rounds. The gap between the two is the cache filling (one load a step), i.e. 004 and 031 — and 004 found the loads all went to the first layers (fixed 10:29: 30–40 % of picks resident after the warm-up, expert lap 6.7–7.4).

## Next
- The cache policy (004), sizing by the context in use (031), use in prefill (038), loading by
  the card's copy engine (045).
