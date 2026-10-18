# 059 — How much of the decode step the host loop holds

**Status:** Open — a measurement
**Area:** decode
**Gain:** decides whether card-side work can show at all

## Proposal
vLLM found its host loop busy 4.95 ms of a 5.37 ms step: "a GPU-only speed-up beyond about 0.4 ms
would just expose the host." Our main thread posts launches, routes, slots prefetches, runs the
join, and drives the experts. Measure its busy time against the step's, separately from the pool's
expert work, before trusting a card-side saving to reach the token.

## Source
vLLM PR #61868, "Where the time goes".

## History
- 2026-10-18: proposed. Known pieces: slot 0.7 ms and the router lap 2.6 ms a step token (041).

## Next
A main-thread busy lap in the profile (wall time minus its waits), read with 041.
