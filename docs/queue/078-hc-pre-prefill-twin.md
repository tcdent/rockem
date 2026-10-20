# 078 — `hc_pre`'s matrix-path twin for prefill

**Status:** Open — measured, the kernel to write
**Area:** prefill
**Gain:** est. most of the host's 0.63 ms a prompt token wait on the FFN input (prefill 2.78 → ~2.2 ms a prompt token at 18k; an estimate)

## Proposal
`hc_pre` (the hyper-connection's input half: the coefficient dots over the four streams, the mix, the collapse, and on
a MoE layer's FFN card the router and the act phase) has one shape for every width: a fixed grid, the tokens looped in
rounds, a wave a token. The rows kernels got matrix-path twins for steps past `SPEC_TOKENS` (the `*_w` kernels,
018/017); `hc_pre` did not. A twin past `SPEC_TOKENS`: the coefficient dots and the router as WMMA GEMMs over the
chunk's tokens, the act phase a workgroup a token over the whole grid. Decode keeps today's kernel.

## Measured (2026-10-20, run of the track-launches gate, build 5f3e850 + the checks split)
The last ~1,000 launches a card of an 18k prompt's 1024-token step (`cards.sh`):

| card | busy | `hc_pre` launches | average | time | share of the card |
|---|---|---|---|---|---|
| card0 | 99 % | 89 | 2.1 ms | 186 ms | 14 % (`hc_post`'s wait 64 %) |
| card1 (the FFN card) | 98 % | 69 | 5.9 ms | 410 ms | 39 % |

The FFN card's `hc_pre` (router and act phase over a 512-token chunk) is what the host's FFN-input wait waits on
(`kda_card` 0.63 ms a prompt token): its act phase writes the word `ffn_input` polls. card0 mostly waits.

## Source
038's open question (are the cards prefill's longer side under 037's wavefront), answered by the first track-launches
timeline without the bring-up checks (2026-10-20).

## History
- 2026-10-20: measured and listed.

## After 058 (2026-10-20): where prefill stands, and what 078 is for
With the MLA attend split (058), prefill at 18k is 2.65 ms a prompt token: the host's experts 1.55, its wait on a
chunk's FFN input from the cards 0.49, its wait on the MLA projections 0.37. Each card is ~97 % busy in a prefill step,
but ~26 % of that is `hc_post` spinning (waiting on the host or the other card) and 28–30 % is `hc_pre` (cards.sh, run
of mla2f). 078 is the speed of `hc_pre` at prefill widths on both cards, not a balance between them: both run it on
their own streams; the FFN card's also carries the router and the act phase. Balancing that (the router's rows split
across the cards, the scores exchanged) is a separate, smaller step, to look at after 078 shows what is left.

## Context for later (tcdent, 2026-10-20: to think about before investing in it)
- **The per-chunk FFN card** (chunk A's FFN half on card0, chunk B's on card1, at once): exact, ~0.9 GB a card (every
  layer's shared expert and router rows on both, the three dense FFNs ~156 MB each; fits 1M at ~17.2 of 20.5 GB a card).
  It fills a card's spin with the other chunk's FFN, which pays when the cards are prefill's bottleneck — after 058 the
  host's experts are, so it waits.
- **Moving routed-expert work from the CPU to the cards in prefill.** The movable work is the routed experts' picks:
  today ~27 % of prefill picks hit experts resident in VRAM (~18 a layer a card, ~10.4 GB a card, full; 1M shrinks it).
  Beyond residency, 039 streams the next layer's non-resident experts through a staging area (~25 GB/s a card over PCIe:
  ~70 more experts a layer across both cards in a ~19 ms window, about a quarter of the CPU's). The catch: the cards'
  copies read the same host DRAM the CPU would, so the gain is at most the CPU's non-DRAM share of the experts lap —
  by arithmetic (not measured) a layer-chunk reads ~3.3 GB of non-resident experts, ~11 ms at ~300 GB/s, against ~19 ms
  measured, so ~8 ms a layer-chunk is compute and the pool's overhead.
  **First, before any build:** how close the CPU's prefill expert run sits to its DRAM floor (`pool.sh`, `experts.sh`,
  the pool's per-run rows). Close to it, streaming buys little and the CPU side needs another lever.
- **The CPU's expert phases interleaved instead** (tcdent, 2026-10-20: gate/up and down "as a single, somewhat
  interleaved kernel"). A token's down needs its whole hidden vector (every output row is a dot over all 2,048 values),
  so down cannot overtake gate/up on the same rows. Two forms:
  - *down split along the hidden dimension*: the core that made a slice of the hidden vector (whole 128-groups, so its
    quantize is local and the quantize run goes) multiplies it by down's matching columns into a partial output summed
    with the other slices' — one interleaved kernel a core, no barriers; numeric (the partials' sum reorders the adds),
    down re-laid out by column slices in the bundle, a 4,096-value partial written a slice;
  - *whole experts a CCD* (exact): a CCD's cores split an expert's gate/up rows, sync among themselves (the hidden
    vector stays in that L3), split its down rows, take the next expert — the same arithmetic a row, the global
    barriers and the 92-core run tails gone, the pipelining across the 11–12 CCDs; the hot experts' skew the sizing to
    watch. The preferred form.
  Both answer to the same first measurement: the run against its DRAM floor, and how much of the rest is barriers and
  tails (the pool's per-run rows: `pool.sh`).
- **The idle windows' prefetch in prefill** (tcdent, 2026-10-20: "if we pick them right we can use the idle CPU time to
  stream in as much as we can fit in the cache"). While the host waits on the cards (the FFN input ~6 ms a layer-chunk,
  an MLA layer's projections ~9 ms) the pool and DRAM idle. A 512-token chunk picks nearly every expert, so before the
  picks arrive each CCD can warm the first ~32 MB of its own queue in run order (an expert slice's CCD and slot are fixed;
  the unpicked few are cheap waste) — four cores a CCD at ~22 GB/s fill it in ~1.5 ms. The cap is the cache, not the
  window: ~384 MB of L3 against ~3.3 GB of non-resident experts a layer-chunk, so ~12 % of the run from cache — est.
  ~1–1.3 ms of a ~19 ms run, ~0.06–0.08 ms a prompt token, if the run is DRAM-bound at its start. The cheapest of these
  to try: the decode prefetch's job posted at each dispatch wait over each CCD's first slices; read `prefetch_job`
  (done before overtaken) and the experts lap.
  **Measured 2026-10-20 (run 778aa269 against 7c985d71): no gain; reverted.** Built as `Routed::prefetch_lead`: at each
  prefill dispatch wait, the non-resident experts' gate/up slices in the run's order, each slot's first 32 MB (tcdent:
  "24 is too conservative"). The warm-up worked — 98.6 % of a job's slices pulled before the run overtook it, in a
  ~9.4 ms window, ~46 cores — and the run did not get faster: the experts lap a prefill layer-chunk 17.78 → 18.24 ms
  median (16.34 → 18.20 average), prefill 2.65 → 2.68 ms a prompt token, long SCORE bit-identical. So either the run is
  not DRAM-bound at its start or the warmed lines are not served from L3 when it reaches them; the DRAM-floor read and
  an L3 hit count (the fill-source counters, l3-intentional-expert-path) would say which before trying it again.

## Next
The twin's arithmetic is the step gate's numeric kind (WMMA order) for the coefficients and the router; the act phase
stays exact. Then the card lanes again: card1's `hc_pre` share and the FFN-input wait.
