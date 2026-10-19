# 021 — What bounds the CPU expert phase in prefill

**Status:** Resolved (instrumented and measured 2026-10-19) — at a 1024 step with the residents on the cards the host's phase is compute-bound: 94 % pool utilization, 83 % of the items' time in experts of 17+ tokens at 56–61 GMAC/s a core, the light experts 6 % of it
**Area:** prefill
**Gain:** measured — `STEP` 1024 (5.16 → 4.82 ms a prompt token at 18k) and 038 (→ 4.55; experts 2.12 → 1.41); what is left is the kernel's all-core rate on real items (58 against the microbench's 69–75 GMAC/s a core)

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

- 2026-10-19 14:00 (`track-experts`, run 6274d8fa: a row a pool run a token-count bucket — items, (token, expert)
  pairs, their summed time, their bytes — with the runs as `experts.gu` / `experts.quant` / `experts.down`
  zones; `bringup/reads/experts.sh`). The cold 18k prompt at `STEP` 1024, 27 % of picks on the cards:

  | items by tokens | share of gu time | GMAC/s a core | GB/s of weights a core |
  |---|---|---|---|
  | 1 | 0.4 % | 18 | 9.5 |
  | 2 | 0.5 % | 27 | 7.1 |
  | 3–4 | 1.2 % | 39 | 5.9 |
  | 5–8 | 3.6 % | 49 | 4.0 |
  | 9–16 | 11.8 % | 56 | 2.4 |
  | 17–32 | 29.8 % | 60 | 1.4 |
  | 33–64 | 34.1 % | 60 | 0.7 |
  | 65+ | 18.7 % | 61 | 0.3 |

  The down run reads the same way (56 GMAC/s from 17 tokens up, 20 at one). Utilization: the items' summed
  time is 94.4 % (gu) and 93.2 % (down) of 92 cores over the runs' wall — the pool is busy, not waiting.
  Items a prompt token: gu 0.83 ms, down 0.45, the lap 1.41 (the quantize run and the posting are the rest).
  So the light experts are 6 % of the time, not the gap the 03:30 note supposed: with the residents taking
  the heavy experts and the step at 1024, what is left on the host is compute-bound at 56–61 GMAC/s a core —
  85 % of the microbench's all-core 69–75 on the same shapes (an item's activations stream beside its weights
  here, and its outputs scatter into the hidden states; the bench reused one activation set). Decode's items
  (one or two tokens an expert) run at 12–24 GMAC/s a core, the bandwidth floor, with the pool 62–73 % busy
  over the runs' wall: the host's expert phase at a verify step has a quarter of idle in it (059/064).

## Next
1. The kernel's rate on real items against the bench's (58 vs 69): the per-tile re-read of the item's
   activations (eight 4-row tiles each walk every token's groups) and the scattered SwiGLU writes — a tile
   order that keeps a token's activation in registers across the rows, or wider tiles.
2. More picks on the cards: 038 takes 27 % at the residency the VRAM allows; the latent mirror's size (031)
   decides the rest.
3. The decode runs' 27–38 % idle: the items' tail and the post (059).
