# 045 — Fill residency slots with the card's copy engine

**Status:** Open
**Area:** decode, prefill (039)
**Gain:** frees a CPU core and keeps the loads off the critical path (unmeasured)

## Proposal
The residency loader streams experts through the BAR from a CPU core; flat out that doubled every
card lap (003). A KFD SDMA queue lets the card pull from host memory, scheduled into its idle
window during the CPU's experts.

## Source
tinygrad `88abb356f` (both SDMA engines brought up, copies moved onto them).

## History
- 2026-10-18: proposed.

## Next
A KFD SDMA queue in `kfd.rs`; measure its rate and its effect on the handoffs.
