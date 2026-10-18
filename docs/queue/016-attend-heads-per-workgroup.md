# 016 — Attend: four heads a workgroup

**Status:** Rejected (measured) for decode; Deferred — small gain for prefill
**Area:** decode, prefill
**Gain:** measured — decode at 18k worse (kda card 14.5 → 20.3); 3k prefill 4.4 → 4.3

## Proposal
`ATTEND_H` = 4: a latent row read once for four heads' dots and mixes.

## Source
Our design, 2026-10-17.

## History
- 2026-10-17 12:04: bit-identical. At decode one token is 32 workgroups on 84 CUs: the attend is
  occupancy-bound, not L2-bound. Kept at 1 (the kernel keeps the constant).

## Next
Small-gain: a prefill-only pair at 4 heads (~0.1 ms a prompt token at 3k).
