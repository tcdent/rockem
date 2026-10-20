# 041 — Trace the router lap

**Status:** Resolved (2026-10-20): the host's router lap is gone; the router's cost is on the FFN card
**Area:** decode
**Gain:** unknown, possibly ~1 ms a step token

## Proposal
The router lap grew from 1.8 to 2.6 ms a step token when it took on the next layer's rows (005).
Understand where the time goes before optimizing.

## Source
The 2026-10-17 profiles.

## History
- 2026-10-18: listed.

- 2026-10-20: resolved by the work after it. 064 split the lap into zones (route, dispatch, experts, join, sum);
  001 moved the norm, quantize and pick onto the FFN card's `hc_pre`; 002 moved the sum to the leader card. The
  host's `route` lap is 0.31 ms a verify step at 18k, all of it the residents' dispatch. The router's rows (this
  layer's and the next's, 005's predictor) run inside the FFN card's `hc_pre`: ~85 µs a launch with them against
  ~34 without (stamped labs, 2026-10-19), ~2 ms a verify step over 42 layers. In prefill the same launch is the
  card-side cost the host waits on (078).

## Instruments in place
- the host: `main`'s zones (`laps.sh`, `trace.sh`);
- the cards: every launch on `card0` / `card1` with `track-launches` (`trace.sh`, `cards.sh`), at served speed in
  prefill and ~3 % in decode since the bring-up checks became their own feature (`bringup-checks`);
- inside `hc_pre`: the phase stamps of a stats code object (`kstats.sh`).
