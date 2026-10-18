# 043 — What prefill inherits from generation

**Status:** Open — a living inventory
**Area:** prefill
**Gain:** the sum of its rows

## Proposal
Prefill was built on decode's patterns. tcdent, 2026-10-18: "we should be very conscious of the
ways in which we inherit patterns from generation into prefill because it was designed first, and
we should be able to quantify what those are to be able to identify ways that we can improve."
Decode minimizes synchronization on one token's path; prefill knows its whole input and should
keep every resource busy.

| Inherited pattern | Prefill cost (per prompt token at 512) | Prefill-native | Entry |
|---|---|---|---|
| One step through all layers; cards and CPU alternate | ~2.2 ms (est.) | chunk wavefront | 037 |
| Layer ownership by card | one card idle per MLA layer / FFN half (unmeasured) | pipeline across layer groups | 037 |
| Handoffs through uncached host memory | small (est. ~0.03 ms) | bulk copies / peer writes | 001, 045 |
| Routing, sum and accounting on the host | 0.3 ms (measured) | on the card | — |
| Expert items tuned for DRAM streaming | ~0.5 ms (est.) | prefetch next tile while computing | 021 |
| Residency only for steps ≤ `SPEC_TOKENS` | ~1 ms (est.) | grouped WMMA on residents | 038 |
| `STEP`-deep buffers sized like decode | later | steps sized by prompt length | 039 |
| `kda_step` one workgroup a head | 0.13 ms (measured) | chunked WY | 023 |
| `hc_pre` decode shape | 0.35 ms (measured) | WMMA GEMM + Sinkhorn waves | 022 |
| Attend: workgroup per (token, head), scalar | ~1.5 ms at long positions | WMMA heads as rows | 018 |
| Indexer: thread per (token, pool) | grows with context | WMMA scoring | 019 |
| Draft layer as its own pass after each step | ~0.15 ms (est.) | one more layer in the wavefront | 053 |
| Per-token host loops in the MLA layers | inside 0.9 ms (unmeasured) | on the card | — |

Already de-inherited: the head for the last token only, WMMA rows twins (009), the attend once a
step (012), the indexer on the card (013).

## Source
Our review, 2026-10-18.

## Next
042 to measure the unmeasured rows.
