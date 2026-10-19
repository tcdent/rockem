# 021 — What bounds the CPU expert phase in prefill

**Status:** Measured (2026-10-19) — the phase is the sum of streaming and compute, neither alone; the lever is the prefetch
**Area:** prefill
**Gain:** est. 2.1 → ~1.3–1.5 ms a prompt token if the next item's slab streams while the current one computes (0.6–0.8 ms, 12–15 % of prefill)

## Proposal
The doc called the phase compute-bound (~7.7 of ~12.5 TOPS), but 12.5 was ik_llama's achieved
whole-model rate, not Zen 3's peak (~29 TOPS by the port model). Per core at 512: streaming its
share ~1.0 ms, computing ~1.2 ms (est., from `gemm_rate` on this host's 7551P at ~25 GMAC/s a core,
scaled ~3× for Zen 3); measured ~1.75 — between max and sum: the two only partly overlap. DRAM
reads at ~180 GB/s, 56 % of the ceiling.

## Source
Our analysis, 2026-10-18; the 512-step profile (run ends 1.72–1.78 ms a token, no scheduling gap).

## History
- 2026-10-17: traced to the gemm, "compute-bound" (the doc's §5).
- 2026-10-18: that conclusion questioned; `gemm_rate` measured on the 7551P: 25.0 GMAC/s at 64 tokens.

- 2026-10-19 01:10 (`gemm-rate`, the `common` test binary built static for the box, one core, the lab boot): on a
  Zen 3 core of metatron the expert kernel runs 58 GMAC/s a token at a time and 85 GMAC/s as the multi-token
  gemm at 64 tokens (76 at 16), cache-resident; with 33.6 MB of rows streaming from DRAM, 42 GMAC/s a token at
  a time (the core pulls 21 GB/s) and 71–78 GMAC/s at 16–64 tokens — one core's prefetchers hide its own
  stream almost entirely. The 7551P's 25 GMAC/s scaled ×3 was right within 10 %. So at 512: compute
  42 × 256 items × ~16 tokens × 25.2 MMAC / (92 cores × 76 GMAC/s) ≈ 1.2 ms a prompt token, the stream
  162 GB a step / 512 ≈ 1.05 ms at 300 GB/s or 1.75 at the 180 GB/s the phase achieves; measured 2.1 —
  the sum, not the max: with 92 cores streaming at once each core's share is ~2–3 GB/s, too little for its
  prefetchers to run ahead of a 5 ms item, so every item waits for its bytes and then computes. The kernel's
  own compute is not the bound (15 TOPS aggregate against ~7 needed).

## Next
1. The prefetch of the next item's slab into L2 while the current one computes (decode's L3 prefetch
   machinery applied inside the pool's run): the phase toward max(compute, stream) ≈ 1.3–1.5 ms.
2. The achieved 180 GB/s against the socket ceiling ([[milan-bandwidth-targets]]): the item order's locality
   to each node's channels (the placement already keeps 94 % of items on their CCD).
3. `gemm_rate` on metatron, one Zen 3 core, cache-resident and DRAM-streamed at the pool's share.
2. If cache-resident × 96 ≫ ~10 TOPS: prefetch the next item into L2/L3 while computing (the
   decode prefetch machinery applied to prefill). If not: the kernel's per-core rate.
