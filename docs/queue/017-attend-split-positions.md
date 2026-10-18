# 017 — Attend: positions split across workgroups (decode at long context)

**Status:** Open — a timeline at 18k first
**Area:** decode, long context
**Gain:** unknown; the attend is ~2.3 ms a two-token step at 18k on the head's card

## Proposal
At decode the attend is occupancy-bound (016). Split a token's 2,048 selected positions across
workgroups and merge the partial softmaxes in a fixed order (flash-decoding's split-K). Not
bit-identical to today's single-workgroup order.

## Source
016's measurement; flash-decoding / Lean attention (stream-K decode).

## History
- 2026-10-17: proposed.

## Next
A timeline at 18k for the attend's real time a launch, then the kernel.
