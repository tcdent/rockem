# 078 — `hc_pre`'s matrix-path twin for prefill

**Status:** Built (2026-10-20) — the exact fixes and the router's matrix path; prefill now CPU-expert bound
**Area:** prefill
**Gain:** measured so far (the exact fixes, 2026-10-20): prefill 2.65 → 2.19–2.23 ms a prompt token at 18k; the router's matrix path 2.19–2.23 → 2.14–2.21 (the FFN card's `hc_pre` 6.16 → 2.54 ms a chunk; the host's FFN-input wait gone)

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

## The launch to the phase (2026-10-20, stats object on the real-time counter, `kstats.sh`)
`hc_pre`'s four stamps on its phases, wave 0 of workgroup 0, medians in µs (wide = a 512-token chunk):

| `hc_pre` | total | dots | mix + collapse | router | act |
|---|---|---|---|---|---|
| few tokens, plain | 33 | 14 | 15 | — | — |
| few tokens, FFN card | 74 | 14 | 19 | 22 | 17 |
| wide, plain | 2,067 | 269 | 1,803 | — | — |
| wide, FFN card | 9,758 | 246 | 4,360 | 4,112 | 982 |

- **The mix is one workgroup's.** The last workgroup to arrive makes every token's mix (sigmoids, softmax rows,
  Sinkhorn) for all 512 tokens while the other 191 wait: ~1.5–1.8 ms in every wide launch, both halves of every layer,
  both cards. Each workgroup making its own tokens' mixes, then one barrier, is exact and parallel.
- **The FFN card's collapse writes the block input to host memory too** (`x_host`: 8 MB a chunk over the link), a
  leftover from before 001 that serving never reads (only bring-up's `ffn_logits`): the likely reason its mix + collapse
  is 4.4 ms against the plain 1.8. Bring-up can copy the card's `x` out when it asks.
- **The router** (4.1 ms): the decode-shaped loop; the WMMA path is the core of 078.
- **The act phase** (~1 ms for workgroup 0's three tokens, ~330 µs a token against 17 in decode): not explained yet.

## The exact fixes (2026-10-20, run 6725d575, build mix1)
Three changes from the phase table, all exact (long SCORE bit-identical to the run before, NLL 0.04678):
- **The mix distributed.** Every workgroup makes the mixes of its own tokens (`wg + k·GRID`, the same arithmetic),
  then one barrier (`PRE_BARRIERS` 4, the mixes' ready word `+5`) before the collapse.
- **`x_host` gone.** The FFN card's collapse no longer writes the block input to host memory; bring-up's
  `ffn_logits` copies the card's `x` out when it asks.
- **The other card's activations card to card.** The resident jobs on the non-FFN card read the act phase's
  output copied from the FFN card's `xq`/`xs`/`xg` into `Peer::axq/axs/axg` behind a `W_XQ` word, instead of the
  host uploading what it had just read: `xpt_dispatch(l, n, picks)`.

| a 512-token chunk, medians | before | after |
|---|---|---|
| FFN card's wide `hc_pre` | 9.76 ms | 6.16 |
| host `kda_card` | ~5 | 1.36 |
| `act.wait` inside it | — | 1.26 |
| `act.read` | 208 µs | 98 |
| the residents' dispatch | 974 µs | 73 |

Prefill at 18k 2.61–2.69 → 2.19–2.23 ms a prompt token; the cold 18k prompt ~51 → 43.5 s, warm 42.4; a verify step
44.65 ms (neutral).

**Does `hc_pre` need to publish mid-launch?** No, as long as the act phase is its last work. The host's `act.wait`
ends 23 µs after the FFN card's `hc_pre` launch ends (the wait closest to the timeline's clock anchor); it waits on
the launch, not on a held-back flush. A mid-launch flush would pay only if work stayed behind the act word in the
same launch — none does. What is left of `act.wait` is `hc_pre`'s length: the router (the matrix path below).
(The card lanes drift against the host's by ~230 ppm across a step — the same wait reads 347 µs at the step's start
and 23 at its end, linearly — so a card launch's position in the timeline is off by up to ~0.3 ms early in a step.)

## The router on the matrix path (2026-10-20, run 86c15b29, build rw1)
A launch past `SPEC_TOKENS` runs this layer's 288 router rows through `rows_wmma_into` (the rows kernels' twin: the
normed activations bf16 as staged, f32 accumulation): `wmma_grid(288, n)` workgroups, 12 for a 512-token chunk, a
128-row × 128-token block each; decode keeps the loop and its bits. 128 VGPRs (the grid stays resident), two more
barriers (the staging).

| a prefill step at 18k | mix1 | rw1 |
|---|---|---|
| `hc_pre` over a chunk, median (card0 / card1) | 6.16 ms (FFN card) | 2.31 / 2.54 |
| host `kda_card` median | 1.36 ms | 0.17 |
| `act.wait` in the last step | 93.7 ms | 26.7 (median 0) |
| the last step's main lane | 1,611 ms | 1,537 |
| prefill, ms a prompt token | 2.19–2.23 | 2.14–2.21 |
| cold 18k prompt | 43.5 s | 41.6 |

Long SCORE numeric, as the rows twins: NLL 0.04678 → 0.04505, mean |Δ logprob| 1.8e-2, one position at 6.3 (a pick
flipped at a near-tie). A verify step 24.53 → 24.42 ms (neutral).

**What this leaves:** the host no longer waits on the cards' FFN input. A prefill step's main lane is now 79 % the
host's experts (1,210 of 1,537 ms) and 15 % the MLA projections' wait (238 ms). Prefill is CPU-expert bound, which is
this entry's "Context for later" below — the DRAM-floor read first.

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

**The down items' stores (2026-10-20, runs 01969130 → 4cdf8832 → 07d1390e, 1M window).** `experts.sh` showed down a
core ~35 % below gate/up at 9–32 tokens an item (36.8 / 40.7 against 55.8 / 62.5 GMAC/s). Since 002 the host's down
items write into the leader card's VRAM through the BAR, and `q4::gemm` stores an eight-row tile (32 B) across every
token of the block in turn, so each token's line went out in halves. Staging a block's rows in a cached buffer and
streaming each token's 512 B run as whole lines (one fence an item) brought down to 46.1 / 53.8 / 56.9 / 56.7 GMAC/s a
core at 5–8 / 9–16 / 17–32 / 33–64 tokens — level with gate/up — exact (long SCORE bit-identical, the decode's tokens
unchanged). **The run's wall did not move:** prefill `experts.down` 6.82 → 6.74 ms median a layer-chunk, the experts
lap 21.35 → 21.77, prefill 2.25–2.27 → 2.26–2.27 ms a prompt token. Reverted. The gate/up and down walls keep the
proportion of their work (1.8 : 1, at 32k — 10.4 / 5.8 ms — and at 1M — 12.5 / 6.8), and by arithmetic
(not measured: ~3.3 GB of non-resident experts a layer-chunk, above) the phase streams ~165–175 GB/s against the
323–375 measured read ceilings: the per-item rate is not what sets a prefill expert run's length. The
DRAM-floor read above, and what bounds a run's aggregate stream, come first.

## Next
The twin's arithmetic is the step gate's numeric kind (WMMA order) for the coefficients and the router; the act phase
stays exact. Then the card lanes again: card1's `hc_pre` share and the FFN-input wait.
