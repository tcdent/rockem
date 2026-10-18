# 017 — Attend: positions split across workgroups (decode at long context)

**Status:** Quantified (bench, 2026-10-18) — the attend is 0.38–0.44 ms a layer a verify step from 2,050 positions on (~4.4 ms a step over 11 layers): the 24 → 31 ms step at long context; ready to build with 018's shape
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

## Next
Build with 018: the same WMMA scoring/mix kernel, positions split across workgroups for a one- or
two-token step, partial softmaxes merged in a fixed order (not bit-identical to today's single
workgroup; gated numerically like the prefill pairs).
