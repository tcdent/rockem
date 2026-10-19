# 021 — What bounds the CPU expert phase in prefill

**Status:** Measured (2026-10-19) — the kernel reaches 6.3 TMAC/s over the phase's own access pattern on all 92 cores (1.35 ms a prompt token of the 2.1); the loss is the light experts at the bandwidth floor and the pool's runs, so the levers are bytes a token (the step width, 038), not a prefetch
**Area:** prefill
**Gain:** measured — `STEP` 1024: prefill 5.16 → 4.82 ms a prompt token at 18k (experts 2.12 → 1.93); the floor 1.35 at the kernel's all-core rate, 038 the next bytes lever

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
  a time (the core pulls 21 GB/s) and 71–78 GMAC/s at 16–64 tokens: one core's prefetchers hide its own
  stream. The 7551P's 25 GMAC/s scaled ×3 was right within 10 %.
- 2026-10-19 03:30 (`gemm-rate` on all 92 pool cores at once, the served boot idle): the aggregate over the
  33.6 MB slab — 1 token a slab 624 GMAC/s (6.8 a core: 313 GB/s of weights, the DRAM ceiling for this
  pattern), 4 tokens 2,369 (25.7 a core, 295 GB/s: still the ceiling), 16 tokens 5,468 (59.4 a core,
  171 GB/s), 64 tokens 6,931 (75.3 a core: compute at all-core clocks). And the phase's own shape — 2,048
  slabs of 32 rows (64 KB, an item's gate or up rows) each applied to n tokens then the next, from a
  134 MB pool: solo 73.6 GMAC/s at 16 tokens (the fresh stream an item costs nothing against the long
  slab's 71), all 92 cores 6,327 GMAC/s at 16 tokens (68.8 a core, 197 GB/s of weights) and 7,100 at 64.
  So the kernel over the phase's access pattern reaches 6.3 TMAC/s; a 512-step's experts (4.35 TMAC)
  would take 0.69 s at that rate, 1.35 ms a prompt token, against the 2.1 measured. The bytes alone
  (162 GB a step) take 0.54 s at the 300 GB/s ceiling, 1.05 ms a prompt token. Neither the kernel's
  compute nor an item's stream start is the gap; what is: (a) the picks' skew — the top 64 experts take
  44 % of a layer's picks, so a third of the bytes belong to experts with ~8 tokens or fewer, which run at
  the bandwidth floor (25–40 GMAC/s a core all-core) while the heavy items run at compute; (b) the pool's
  three runs a layer (posting, slot imbalance, the quantize between). A software prefetch inside a core
  gains nothing: a core's own stream is already hidden, and across the box the light items are at the
  aggregate ceiling. What moves the phase is bytes a token: the step width (bytes halve a token at 1024;
  the earlier sweep 128/256/512 read experts 4.0/2.9/2.4) and 038 (the resident experts' tokens to the
  cards, their bytes off the CPU — the heavy experts, which 063 now has resident from the first prefill
  steps).

- 2026-10-19 03:52 (`STEP` 1024, run 03f082a1): prefill 5.16–5.20 → 4.82–4.84 ms a prompt token at 18k
  (experts 2.12 → 1.93, kda card 2.13 → 2.05, the rest unchanged), the cold 18k prompt 96.5 → 90.6 s; decode
  unchanged (48.4–48.8 ms a verify step on the hits, 26.4–27.5 ms an emitted token), the long SCORE
  bit-identical (the step width changes no token's arithmetic), zero PCIe records. Kept. The experts' fall
  with the width — 4.0 / 2.9 / 2.4 (2.1 with the multi-token kernel) / 1.93 at 128 / 256 / 512 / 1024 — is
  the bytes-a-token half of the phase; the kernel's 1.35 is the floor it approaches. One read to know: the
  first decode step after a prompt carries the last prefill step's draft pass in its row (the draft lands in
  the step after it), so that step reads ~230 ms of `draft` at 1024 against 115 at 512 — prefill's time,
  in decode's row.

## Next
1. 038: the resident experts' tokens to the cards in prefill.
3. The pool's three runs a layer as zones (`pool.run` is one already): the posting and tail imbalance sized.
4. `gemm_rate` on metatron, one Zen 3 core, cache-resident and DRAM-streamed at the pool's share.
2. If cache-resident × 96 ≫ ~10 TOPS: prefetch the next item into L2/L3 while computing (the
   decode prefetch machinery applied to prefill). If not: the kernel's per-core rate.
