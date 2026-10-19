# 073 — The grouped expert kernel's own rate

**Status:** Quantified, on the shelf — matters only when the cards become the prefill expert phase's longer side; today they finish inside the host's phase
**Area:** prefill (the cards' resident experts)
**Gain:** none today; needed before more picks move to the cards (031, 039)

## Proposal
`xpt_gemm_gu` / `xpt_gemm_down` (038) take one memory latency a k-step (two batches of four loads a group, no
LDS, direct 16-byte loads a lane) and run a 16-token tile over whatever a group has. LDS-staged activations for
a workgroup's token tile, more tiles in flight a wave, and a token-tile loop that skips empty columns would lift
the rate several-fold. At two tokens the grouped kernel is slower than the per-token one, which is why the
verify step keeps `xpt_gu` / `xpt_down` (twins at `SPEC_TOKENS`).

## Source
038's history (2026-10-19): the join waited 0.7–1.3 ms a step on the grouped kernel at two tokens.

## History
- 2026-10-19: given its own entry.

## Next
bench-xpt over the grouped kernels at 16–1024 tokens an expert; then LDS staging.
