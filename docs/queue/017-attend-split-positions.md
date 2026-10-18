# 017 — Attend: positions split across workgroups (decode at long context)

**Status:** Built 2026-10-18 with 018 (bench: 380–430 → 226 µs a layer a two-token step, 1.7–1.9×); the node's integration and gate next
**Area:** decode, long context
**Gain:** measured ceiling ~4 ms a verify step at any context past 2,050 positions

## Proposal
At decode the attend is occupancy-bound (016). Split a token's 2,048 selected positions across
workgroups and merge the partial softmaxes in a fixed order (flash-decoding's split-K). Not
bit-identical to today's single-workgroup order.

## Source
016's measurement; flash-decoding / Lean attention (stream-K decode).

## History
- 2026-10-17: proposed.

## History (cont.)
- 2026-10-18 11:15 (`bench-mla-attend`, the kernel alone): a two-token step's `mla_attend` 383 µs a
  layer at 2k positions, 421 at 18k, 440 at 65k, 430 at 262k — flat past the 2,048-selection point,
  so the long-context decode cliff is the attend's shape (64 workgroups a token, one a head, each
  walking 2,048 latent rows), not the position count. ×11 layers ≈ 4.4 ms a step; the served step
  reads 24 ms short and 31 at 18k.

- 2026-10-18 13:30, built as 018's chunked form: a verify step's token is 4 head tiles × 9 chunks of
  256 positions (72 workgroups for two tokens), each chunk's (m, l, O) to a partials buffer,
  `mla_attend_merge` (all heads' weights in parallel, the chunk count a constant) into the mix,
  then the scalar `mla_uv` (a row a wave). **226 µs a layer** = q_abs 39 + attend 119 + merge 32 +
  W_uv 37, against the scalar's 380–430. Smaller chunks lost (the per-workgroup fixed cost), the
  scalar q_abs (39) is the next item: it wants a transposed `W_uk` copy or the rows shape.

## Next
With 018's integration. Expected served: ~2 ms a verify step at long context (31 → ~29 ms at 18k).
