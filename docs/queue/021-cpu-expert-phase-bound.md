# 021 — What bounds the CPU expert phase in prefill

**Status:** Open — a measurement first
**Area:** prefill
**Gain:** est. 1.75 → ~1.2 ms a prompt token if streaming and compute overlap

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

## Next
1. `gemm_rate` on metatron, one Zen 3 core, cache-resident and DRAM-streamed at the pool's share.
2. If cache-resident × 96 ≫ ~10 TOPS: prefetch the next item into L2/L3 while computing (the
   decode prefetch machinery applied to prefill). If not: the kernel's per-core rate.
