# 012 — The MLA attend in one launch a step

**Status:** Resolved
**Area:** prefill
**Gain:** measured — 5,632 launches a 512-step → 1; at 3k positions 10.4 → 9.4 ms a token; bit-identical

## Proposal
A token's attend was 128 workgroups launched once a token a layer. One launch over the step's
tokens: a workgroup a (token, head), the step's latents and selections uploaded first.

## Source
Our prefill timeline, 2026-10-17.

## History
- 2026-10-17 08:52 (`4e220bf`).

## Next
None (the remaining attend cost: 017, 018).
