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

- 2026-10-20: the next layer's router rows (the prefetch's predictor) off the FFN card's critical path. On the card
  lanes the host left `kda_card` ~20 µs after the FFN card's `hc_pre` ended, never during it. A two-pass `hc_pre`
  (this layer's rows, the act phase and its word, then the predictor rows) changed nothing: the host still saw the
  act word only after the launch ended (2 / 9 / 31 µs after it, p5 / p50 / p95), though the word was written ~19 µs
  before the end. Not the clocks (the leader's `hc_post` ends 5–6 µs after the host's done word, p1 −4 µs), not the
  host's MWAITX (a 2,000-cycle timeout moved nothing), not the reads (`act.read` 3–6 µs), and mapping the host-shared
  memory uncached on the cards moved nothing either: a kernel's store to host memory became visible at its launch's
  end-of-kernel release, and gfx11 has no shader instruction to write the L2 back. So the predictor rows run in their
  own launch, `hc_route_ahead`, behind `hc_pre` (which writes the act word and ends) and ahead of the shared expert;
  it writes the router's word, which the host waits for only at the prefetch (`ffn_ahead`). A verify step at 18k
  44.74 → 44.05 ms (`kda_card` 20.5 → 19.8, the FFN card's router `hc_pre` 87 → 70 µs, `hc_route_ahead` 29 µs off
  the path), long SCORE bit-identical, the prefetch unchanged (75–81 % of its slices).

## Instruments in place
- the host: `main`'s zones (`laps.sh`, `trace.sh`), with `act.wait` / `act.read` inside `kda_card`;
- the cards: every launch on `card0` / `card1` with `track-launches` (`trace.sh`, `cards.sh`), at served speed in
  prefill and ~3 % in decode since the bring-up checks became their own feature (`bringup-checks`);
- inside `hc_pre`: the phase stamps of a stats code object (`kstats.sh`).
