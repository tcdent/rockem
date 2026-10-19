# 068 — The WMMA attend's tile loop

**Status:** Quantified, on the shelf — ~5 µs a 16-position tile where its arithmetic is ~1.5; ~1.3 ms a verify step at 18k
**Area:** decode, prefill (long context)
**Gain:** measured context — `mla_attend_w` 119 µs a layer at a verify step at 18k, 34 µs a token a layer at a 512 step

## Proposal
The attend's iteration over a 16-position tile is ~5 µs where the WMMA, the LDS traffic and the VALU add to ~1.5:
the barriers (two an iteration), the k-split reduction through LDS and the single tile in flight. Two tiles an
iteration (double the register prefetch), the reduction folded into the next tile's staging, or a wave-level
softmax that avoids the LDS round trip.

## Source
018's history (2026-10-18, bench-mla-attend); 017's Next.

## History
- 2026-10-19: given its own entry (it lived in 017's and 018's Next).

## Next
bench-mla-attend with two tiles an iteration; the gate is bit-identity against the current kernel (the
reduction order kept) or the step gate's NLL otherwise.
