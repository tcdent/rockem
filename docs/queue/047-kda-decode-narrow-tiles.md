# 047 — KDA decode: narrow value tiles to widen the grid

**Status:** Deferred — small gain
**Area:** decode
**Gain:** est. ≤ ~0.2 ms an emitted token

## Proposal
`kda_step` runs a workgroup a head: 32 workgroups a card a layer on 84 CUs. Narrower value tiles
(a wave per 4–16 value rows) widen the grid; the delta rule is independent across value rows, so
no barriers in the token loop.

## Source
SGLang #44416 (verify recurrence BV 32 → 4/8/16: 663 → 353 µs a step on MI355X, bit-identical);
fla #1415; tinygrad's scan (a wave per 4 rows).

## History
- 2026-10-17: listed; `kda_step` ~16 µs against a ~10 µs launch floor.

## Next
When sub-millisecond decode gains are picked up.
