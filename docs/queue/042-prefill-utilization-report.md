# 042 — A per-step utilization report for prefill

**Status:** Open — an instrument
**Area:** prefill
**Gain:** turns 043's estimates into measurements

## Proposal
Busy fraction of card 0, card 1 and the CPU pool across one prefill step, from the timeline's card
laps and the pool's run spans: a Gantt of who waits on whom.

## Source
Our inventory of inherited patterns, 2026-10-18.

## History
- 2026-10-18: proposed.

## Next
Build from the existing timeline and pool profile; run on a 512 step at short and 64k–256k positions.
