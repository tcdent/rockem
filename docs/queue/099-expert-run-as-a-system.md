# 099 — The expert run as a system: what the bench does not model

**Status:** Open — an investigation (tcdent, 2026-10-25), with measurements below
**Area:** decode, the host's expert run, the cache hierarchy, the pool's scheduling
**Gain:** unknown; the warm-item eviction tax below bounds one lever at ~3 ms a verify step if it is real

> yeah hang on don't give up on it yet unless you can explain why it's not working then we still have some more
> discovery to do and modeling the hardware like you're doing at the very edge of our understanding is super valuable
> — tcdent

> let's think about ways in which it differs … quantifying the different systems that are running and the way that it's
> being incorporated i'm okay with this spawning off into an effort for us to rethink parts of our architecture
> fundamentally as well — tcdent

## What started it
090's `bench-order` (46 cores, per-core warm and cold sets sized as a verify step's layer at 18k) found a run that
interleaves its cold experts with its warm ones 25 % faster than warm-first. In the node the same change was 10 %
slower (A/B on one boot). The two disagree in sign, so the bench is missing part of the system.

## Measured, per item, by class (2026-10-25)
The node (lab-order3: `pool_run` rows with each item timed by where it was claimed — the core's own deque, its CCD's
queue, a mate's deque; orders cycled per step, 480 verify steps each, decodes at 18k):

| per item | warm first | interleaved | interleaved, a third of the CCD queue held back |
|---|---|---|---|
| own deque (warm) | 11.66 µs | 12.81 | 12.79 |
| CCD queue (cold) | 34.3 µs | 32.0 | 31.8 |
| stolen from a mate (warm) | 13.07 µs | 13.6 | 14.5 |
| a run's last core done (median) | 144 µs | 166 | 184 (more items: wider steps) |
| `experts`, ms a token | 6.81 | 7.55 | 7.06 |

The bench (1 token, 64-row items ≈ 136 KB, the node's decode item size):

| per item | warm only | warm first | interleaved |
|---|---|---|---|
| warm | 5.55 µs | 12.37 | 7.81 |
| cold | — | 36.87 | 30.07 |

What this says:
- **In the node interleaving costs the warm items (+10 %) and buys the cold ones little (−7 %).** In the bench it
  bought both (warm −37 %, cold −20 %). The balance it removes (the CCD queue is the run's balancer) widens the end
  spread; holding a reserve recovers part of it.
- **The warm-item interference tax.** In the bench a warm item costs 5.55 µs alone and 12.4 when the CCD's other cores
  stream cold data — not eviction (the counters below: its lines come from L3 either way) but interference. The node's
  warm items cost 11.7 µs in every order: it may be paying this tax all the time. With ~470 warm items a pool run at ~6 µs
  each, that is ~30 µs a run on 92 cores, ~3 ms a verify step — if the bench's 5.55 is reachable in the node.

## Where the lines come from (bench, fill counters, 2026-10-25)
`bench-order` now reads the node's six fill counters (`common::pmc::FILLS`) around every item. Lines an item (2,176),
L2-prefetch hits in L3 / L2-prefetch misses in L3 (DRAM): the warm items **~2,190–2,200 / 2–11 in every order**,
warm-first included; the cold items ~19–33 / ~2,170–2,185. Demand fills are negligible (the L2 prefetcher fetches every
line ahead of the kernel). So **the warm items are L3-resident in every order: nothing evicts them.** Yet a warm item
costs 12.85 µs in warm-first against 5.67 alone (and 7.78 interleaved, 8.75 cold-first) with the same source: an L3-fed
item runs ~2.3× slower while its CCD's other cores stream from DRAM. The non-temporal prefetch result agrees (it changes
nothing: there was no eviction to prevent). Candidates, to measure: the L3's own bandwidth shared with the cold lines'
victim installs (every DRAM line a core streams is written into L3 when its L2 evicts it); the CCD's fabric port; clocks
under the 280 W cap while DRAM streams.

## The interference is the CCD's memory path, not power or the socket (bench-interfere, 2026-10-25)
One core (56) runs warm items — the production one-token kernel over a set it touched, L3-resident — while j cores
stream the same kernel over a pool past the caches, on its own CCD (57–63, one L3: `shared_cpu_list` 56-63,152-159) or
another (65–71). Median warm item, the warm core's clock (APERF/MPERF), the streamers' aggregate:

| j streamers | same CCD: warm item | GHz | streamers GB/s | other CCD: warm item | streamers GB/s |
|---|---|---|---|---|---|
| 0 | 4.56 µs | 3.45 | — | — | — |
| 1 | 4.55 | 3.45 | 17.6 | 4.60 | 18.5 |
| 2 | 4.59 | 3.45 | 31.0 | 4.56 | 31.2 |
| 3 | 4.58 | 3.45 | 37.3 | 4.60 | 37.9 |
| 4 | 5.17 | 3.45 | 38.7 | 4.58 | 40.1 |
| 5 | 7.11 | 3.45 | 39.5 | 4.57 | 40.3 |
| 6 | 10.25 | 3.45 | 39.6 | 4.57 | 40.7 |
| 7 | 12.70 | 3.45 | 40.0 | 4.58 | 40.6 |

- **Not power:** the clock holds at 3.45 GHz throughout.
- **Not the socket:** streamers on another CCD leave the warm core at 4.6 µs.
- **The CCD's own path to memory:** a CCD's DRAM reads saturate at ~40 GB/s with 3–4 streaming cores (the GMI2 port,
  ~51 GB/s by spec, epyc-7r13-topology); every streamer past that adds no bandwidth and slows the CCD's L3-fed cores up
  to 2.8× (inference: L3 hits queued behind outstanding misses in the CCD's shared request path — which queue is not
  measured).
- **Why interleaving lost in the node:** the node's run has all 8 cores of a CCD streaming cold items at once (warm
  first or interleaved alike), so its warm items run at the clogged ~11.7 µs in every order; interleaving per core keeps
  every core streaming, and costs the balance. The bench's even, synchronized shape happened to reward it.
- The same mechanism, seen before from the other side: the main thread's route/dispatch doubled with seven cores
  streaming beside it on its CCD (8.5 → 18 µs a layer, `PREFETCH_CORES_MAIN`, construct.rs).

**With every CCD of the socket streaming (the production shape; `--bg`: 38 streamers on socket 1's other five CCDs,
134.9 GB/s):** the warm CCD's own sweep — j streamers: warm item µs / the CCD's streamers' GB/s — 0: 4.67 / —; 1: 4.60 /
8.8; 2: 4.67 / 14.0; 3: 4.74 / 16.8; 4: 5.56 / 17.7; **5: 9.48 / 18.3; 6: 16.28 / 18.5; 7: 22.96 / 18.7** (clock 3.45 GHz
throughout). With the socket loaded, a CCD's share saturates at ~17–18 GB/s with **3** streamers, and past that its
L3-fed cores slow 2–5×. The node's run streams cold items on all 8 cores of every CCD.

## What it suggests (architecture)
- **A CCD's DRAM streaming capped at its saturation point.** No more than ~3 cores of a CCD stream cold data at once
  (a per-CCD semaphore on the cold queue); the others take warm items (own or stolen) meanwhile, at 4.6 µs instead of
  ~12. Measured with the socket loaded: the saturation point is 3 streamers a CCD (~17–18 GB/s each CCD).
- **The prefetch window the same way:** 8 touchers a CCD saturate the port as 3 would; fewer touchers leave cores free
  (for 098's analog, for the main thread's phases) at the same delivery.
- **Streamers and computers by role:** a CCD's few streamers pull the cold items into the CCD's L3 just ahead of the
  computing cores (who then never wait on DRAM). The catch to design around: a line a core touches sits in its own L2
  until evicted (l3-core-boundary), so a streamer's lines reach L3 only ~512 KB later.


- **Token count:** at 1 token (the served ~1.15 a verify-step expert) the bench still favours interleaving (350 vs 432 µs).
- **Item size:** at the served gate/up pair size (~544 KB, 256 rows) still (416 vs 469).
- **Uneven warm sets** (cores reach their cold items at different times): at 9 ± 6 warm items a core still (357 vs 450).
- **A warm set inside L2:** interleaving's gain shrinks (330 vs 350) and the warm items slow (7.65 → 11.25 µs).
- **Eviction of the warm lines:** the fill counters show them L3-resident in every order; `prefetchnta` on the cold
  reads (1 and 4 KB ahead) changes nothing (warm 13.5–13.7 µs in warm-first, as without).

## A per-CCD cap on cold streamers, in the node (lab-cap, 2026-10-25): no gain
Built: a semaphore a CCD on the run's shared (cold) queues — the first `cap` workers to arrive start on the CCD's queue at
once, the rest on their own and their mates' warm items, streaming only with a slot; A/B on one clean boot, the cap
cycling 0 (none, today's) / 3 / 2 step by step, 24 decodes at 18k, ~480 verify steps each:

| | no cap | cap 3 | cap 2 |
|---|---|---|---|
| `experts` a step, p50, width 2 / 3 / 4 (ms) | 13.6 / 21.4 / 29.1 | 14.8 / 21.3 / 32.2 | 14.3 / 21.0 / 28.5 |
| own (warm) item | 11.84 µs | 13.15 | 12.9 |
| CCD-queue (cold) item | 34.8 µs | 19.6 | 14.9 |
| stolen (warm) item | 13.39 µs | 14.33 | 14.21 |
| a pool run, p50 wall | 150 µs | 165 | 185 |

- **Each cold item streams faster, but the CCD delivers fewer:** 8 streamers complete ~0.23 cold items a µs (~31 GB/s a
  CCD), 2 complete ~0.13. In the node a CCD is not saturated at 3 streamers as the bench's loaded socket was (17–18 GB/s
  a CCD): the node's CCDs do not all stream at once, or not for the whole run.
- **The warm items do not get faster.** Whatever holds the node's warm item at ~12 µs (the bench's: 4.6) is not other
  cores' streaming. Refuted for the node: the CCD-path explanation of its warm items. Long SCORE bit-identical; not taken.
- **Not the kernel either:** a node gate/up item is two 32-row `gemv`s (one token: `q4::gemm` drops to `gemv`) over
  ~128 KB plus a 32-wide SwiGLU — the bench's warm item is one 64-row `gemv` over ~128 KB. Same kernel, same bytes, 2.6×
  the time. Next suspects: where its lines come from by claim class (today's counters are by expert status: 85.7 % of a
  complete expert's lines from L3, so ~14 % from DRAM), the core clocks under the socket cap with all cores busy, and the
  touch's age (~400 µs, other cores touching meanwhile).

## The node's warm items are not warm: the cost is the DRAM lines inside them (bench-item, 2026-10-25)
`bench-item` (one core, 56, a set it touched just before) runs the node's item shapes one element at a time. Alone, every
shape is the bench's 4.6 µs: the gate/up part as `gu_f` runs it (two 32-row `gemm`s over the layout's halves, SwiGLU into
`h`) 4.71, with its group quantize 4.76, the down part with its `sfence` 4.55; the activations and `h` rewritten each rep
by a core on its CCD, another CCD or socket 0 change nothing. **The item's code is not the cost.**

The node's own counters (`prefetch_use`, verify steps) say what is: lines from DRAM by the expert's prefetch status —

| tokens | complete | partial | missed |
|---|---|---|---|
| 2 | 14.0 % | 43.8 % | 94.5 % |
| 3 | 16.4 % | 32.9 % | 94.2 % |
| 4 | 18.3 % | 27.4 % | 93.8 % |

Every slice of a complete expert was touched by the core that runs it (`slot_tab`: a planned slice goes to its taker's
own deque), yet a seventh of its lines come from DRAM at the run. With `--cold-pct p` (p % of the item's lines flushed
after the touch, runs of 16 lines) and `--bg` (streamers through the run), the median gemv64 item:

| load | 0 % cold | 14 % | 40 % | 100 % |
|---|---|---|---|---|
| none | 4.5 µs | 5.4 | 5.9 | 7.0 |
| 7 streamers on its CCD (43 GB/s) | 11.8 | 15.2 | 20.3 | 23.7 |
| 45 streamers over the socket (149 GB/s) | 17.5 | 25.7 | 40.7 | 48.4 |

- Both mechanisms are real and they stack: an L3-fed item slows while its CCD streams (above), and each DRAM line in it
  costs the loaded rate. The node's cold items (34 µs) sit between the two loads.
- **Why the cap gave nothing:** the uncapped "warm" cores pull 14–44 % of their lines from DRAM — they are streamers too.
  The cap moved who streams; the CCD's DRAM bytes stayed the same.
- **The lever is the DRAM bytes, and a fifth of them are prefetched lines that went missing.** In expert-equivalents of
  DRAM lines a layer: tokens 2 — complete 0.97, partial 1.38, missed 2.56 (the leak 20 % of the total); tokens 3 — 1.36,
  1.09, 5.87 (16 %). If the run is bound by its DRAM bytes, removing the leak is −16–20 % of `experts` (inference, not
  measured: ~2.7–3.4 ms a verify step at widths 2–3).
- Open: where a touched line goes before its run. Candidates: L3 eviction between the touch and the run (the touch's age,
  the CCD's other lines meanwhile), the lines of stolen slices (a mate's L2, l3-core-boundary), the partial experts'
  touched slices counted with them.

## The leak reproduced: Milan's L3 drops fresh lines first, and one L3 hit protects them (bench-order --fresh, 2026-10-25)
The bench's warm sets had been the same lines every iteration (L3-resident from the iterations before, the touch adding
nothing). `--fresh 1` takes each iteration's warm items from the pool — new lines, touched from DRAM, then run — as the
node's prefetch touches a new layer's experts. The leak appears. Warm-only (no cold traffic in the run), 8 cores of one
CCD (56–63), the CCD's touched bytes a round against the warm lines then read from DRAM:

| CCD touched | 3.3 MB | 6.5 MB | 9.8 MB | 15.2 MB | 21.8 MB |
|---|---|---|---|---|---|
| warm lines from DRAM | 0.3 % | 2.6 % | 7.2 % | 15.8 % | 33 % |

- **One core alone loses nothing** (2.7 MB touched, 0.1 %); 8 on one CCD lose 7 % at 10 MB — a third of the 32 MB L3.
  Not LRU capacity: the replacement drops lines with no hit history first. The fixed sets (every line hit each
  iteration) lost nothing at 46 cores.
- **One L3 hit protects a line.** The fresh set touched twice or three times before the run:

| CCD touched | order | 1 touch | 2 touches | 3 touches |
|---|---|---|---|---|
| 9.8 MB | warm only | 7.0 % | 0.6 % | 0.2 % |
| 9.8 MB | cold first | 16.3 % | 0.9 % | 0.1 % |
| 15.2 MB | warm only | 16.4 % | 2.0 % | 0.2 % |
| 15.2 MB | cold first | 35.0 % | 4.5 % | 0.4 % |
| 21.8 MB | cold first | 53 % | 11 % | 2.2 % |

- **The node's shape (46 cores, 11 fresh items a core ≈ 1.5 MB, the node's ~11.6 MB a CCD):** one touch loses 6.3 % (warm
  first) to 21.8 % (cold first) of the warm lines — the node's 14–18 % sits inside; two touches 0.8–1.3 %. The warm item
  15.6 → 9.3 µs interleaved, 12.4 → 8.1 warm only; the interleaved run 438 → 371 µs (−15 %); warm-first's run is set by its
  cold phase and does not move (538 → 543).
- **The second pass's cost:** ~110 µs a core for 1.5–2 MB, L3-fed (11,471 L3 hits, 873 DRAM lines: the lines the first
  pass already lost); the first pass ~440 µs, DRAM-fed. It is core time in the window, not DRAM bytes.
- Where it fits the node: the window's DRAM delivery saturates a CCD with ~3 streamers (above), so a CCD's other cores
  are free to re-touch from L3 what the streamers brought — the re-touch must reach a line after it left its toucher's L2
  (a mate's read of a line in the toucher's L2 refetches from DRAM, l3-core-boundary).

## The re-touch in the node: the leak falls a third, the step does not move (lab-retouch, 2026-10-26)
Built (lab): each prefetch worker touches again every range it touched at least 512 KB earlier (`RETOUCH_LAG`, out of
its L2), after each item; on alternate decode steps (`LAB_RETOUCH`, `retouch` on the `step` and `prefetch_use` rows);
24 decodes at 18k on one boot (`bringup/reads/retouch_ab.sh`):

| | width 2 | width 3 | width 4 |
|---|---|---|---|
| `experts` p50 a step, off → on (ms) | 12.27 → 12.21 | 17.97 → 18.35 | 26.55 → 24.99 |
| complete experts' lines from DRAM | 12.2 → 9.0 % | 14.5 → 9.3 % | 15.4 → 9.9 % |
| partial experts' lines from DRAM | 40.2 → 42.6 % | 32.4 → 37.3 % | 27.1 → 31.3 % |
| DRAM lines a layer run, per host expert (k) | 52.0 → 49.9 | 74.4 → 75.1 | 97.9 → 92.7 |

- The complete experts' leak falls by a third, not to the bench's ~1 %: a worker's last 512 KB is never re-touched (a
  third of its ~1.5 MB), and the amend's new job drops the early plan's pending re-touches.
- **The re-touch costs window time:** the partial experts' first touches fall behind by about what the complete ones
  save; the run's DRAM lines per expert do not move, nor does the step.

**The window modelled (`bench-window`, 46 cores):** each core touches fresh planned items for 440 µs (the node's
window; it reaches 10.4 items ≈ 1.4 MB a core, the node's ~11.6 MB a CCD), then the run — its touched items newest first,
the plan's rest, 4 cold items:

| re-touch | run p50 | touched a core | touched lines from DRAM | run DRAM lines a core |
|---|---|---|---|---|
| off | 645 µs | 10.43 items | 6.2 % | 21,032 |
| 512 KB behind | 655 µs | 10.11 | 1.4 % | 21,122 |
| 1024 KB behind | 632 µs | 10.35 | 3.5 % | 21,234 |

The re-touch removes the leak and buys nothing: the window is the limit, and what a re-touch protects it pays for in
first touches not made. **Correction:** the "−16–20 % of `experts`, ~3 ms a verify step" estimate above assumed the
leaked lines could be recovered at no window cost; at the node's window they cannot, by re-touching on the touching
cores. Recovering them is worth something only from capacity the window does not use (cores past a CCD's DRAM
saturation, re-touching what the streamers brought — not yet built or measured), or without a touch at all.
The node's leak (12–15 %) is larger than the model's (6 %): its ordering and the amend's churn, not yet separated.

## The node's loss map (lab-lossmap, 2026-10-26)
Built (lab, `lossmap` in construct.rs): every prefetch touch stamps its slice with its CCD's running count of
first-touched bytes; every run item (counted runs) adds its L3 hits, DRAM lines and time to a histogram over the
step's width, re-touch on/off, its claim (own deque / a mate's / the CCD queue), who touched its slice (early plan /
amended plan / no finished touch), its **age** (bytes its CCD first-touched after the slice) and the bytes its CCD's run
had read before it. 40 decodes at 18k, ~80k layer runs; `bringup/reads/lossmap.sh`.

Own touched items, re-touch off — lines from DRAM, by age (rows) and the CCD's run reads before the item (columns):

| age \ run before | <1 MB | 1–2 | 2–4 | 4–8 | 8+ |
|---|---|---|---|---|---|
| <1 MB | 10.7 % | 1.1 | 1.2 | 2.0 | 6.7 |
| 2–3 MB | 3.8 | 1.0 | 1.5 | 2.3 | 8.0 |
| 4–6 MB | 3.0 | 1.5 | 2.5 | 3.5 | 11.0 |
| 6–8 MB | 3.5 | 2.6 | 4.3 | 5.9 | 15.0 |
| 8–12 MB | 5.5 | 5.8 | 9.3 | 11.4 | 22.9 |
| 12–16 MB | 10.8 | 13.0 | 18.7 | 22.3 | 35.1 |

and their time (µs an item): 7.8–9.8 at young ages past the run's first MB, 10.7–14 at 8+ MB of run reads before, and
10–17 in the run's first MB at every age (the run's start, not its lines: those items lose 3–5 %).

- **The node's loss is the bytes model's:** a touched line's risk grows with the fresh bytes inserted after it — the
  window's touches (age) and the run's own reads (columns) alike. The oldest slices (the early plan's; the front of
  each deque, which thieves take) are the ones lost.
- **A layer run's budget** (width 3, re-touch off; per run: items, µs an item, lines from DRAM, core-µs a core, DRAM MB):

| kind | items | µs | DRAM | core-µs | DRAM MB |
|---|---|---|---|---|---|
| own, touched | 720 | 10.6 | 9.8 % | 82.7 | 8.7 |
| stolen, touched | 254 | 13.3 | 21.7 % | 36.7 | 8.0 |
| own, touch abandoned mid-item | 111 | 13.3 | 55 % | 16.1 | 5.9 |
| CCD queue (cold) | 335 | 33.5 | 93.9 % | 121.7 | 45.2 |

  At width 2 the touched items carry 36 % of the run's DRAM bytes, at 3 25 %, at 4 16 %. The cold items are ~half the
  run's core time at width 3 and two thirds at width 4.
- **What the leak is worth (inference):** touched items at the young, low-churn cells' cost (~8–9.4 µs) instead of their
  mean (10.6 own, 13.3 stolen) is ~20 % of their time, ~8–11 % of `experts` (~1.4 ms a verify step) — an upper bound:
  part of that excess is the CCD's streaming beside them, not their own lines.
- **The re-touch arm** (same boot): touched items 11.3 → 9.7 µs (width 3), their DRAM 13.3 → 9.1 %; per item over the
  whole run −1.4 to −4 %; the arms' item mixes differ (steps alternate), so the step-level difference is within noise.

## Ordering and hints do not reduce the loss; they move it (2026-10-26)
- **Oldest first** (lab-fifo: the owner claims its deque from the front on alternate steps, re-touch off, the same loss
  map): own items lose more (8.8 → 12.4 % at width 2, 10.2 → 13.4 % at 3, 11.0 → 15.0 % at 4) and stolen ones less (17.4 →
  10.3 %, 22.2 → 12.0 %, 24.4 → 14.2 %); the newest slices, safe in their toucher's L2 under newest-first, now wait out the
  run once evicted. Within a layer the loss is set by the bytes a CCD inserts; the order picks which lines pay. Not
  taken.
- **Cold reads non-temporal** (`bench-window --cold-nta 1`, the items nobody touched read with `prefetchnta`): the
  touched lines' loss rises (4.2 → 8.4–9.3 %; with 8 cold items 3.3 → 5.7 %), the run unchanged. The cold lines in the L3
  are the cheapest victims the next window's fresh lines find; without them the fresh lines displace each other. (The
  fill counters under `prefetchnta` are not comparable for the hinted reads themselves.) Not taken.

## The run as quantified (2026-10-26)
- **A touched line's survival** is the CCD's fresh bytes after it (the window's touches and the run's own reads), the
  once-referenced class replaced oldest first, the previous layer's read lines protected (metatron-cpu §3).
- **The leak's ceiling** is ~8–11 % of `experts` (inference, the loss map's budget); every way of buying it back measured
  so far costs about what it returns: a re-touch costs window time (first touches), a flush 12.4 ns a line, the
  non-temporal warm read ~35 % of the gemv, ordering only moves it, non-temporal cold reads make it worse.
- **A warm item that loses nothing** costs ~7.8 µs in the node against 4.6 alone in the bench: the socket's power cap
  (3.04 vs 3.44 GHz under the gemm, metatron-cpu §4: → ~5.2 µs) and the CCD's streaming beside it (~2.6 µs; the
  interference section above).
- **The cold items are the run:** ~half its core time at width 3, two thirds at width 4, at 30–36 µs an item, the CCD's
  DRAM path. The levers on them are fewer cold bytes (the predictor, 098; residency; formats) and the clock (power
  per MAC).

## Where the run leaves the hardware idle, and the SMT siblings (2026-10-26)
- **The run against its floor:** at width 3 a layer's run is ~420 µs and reads ~68 MB from DRAM (~165 GB/s); the box's
  ~290 GB/s would read them in ~235 µs. Per verify step the pool runs' cores wait ~3.2 ms for each run's last core
  (`pool_run`: end_max − end_p50, decode runs; 27.3 s of pool runs over 1,840 steps, 5.9 s of it that tail).
- **The idle hardware:** the 96 SMT siblings. `bench-window --smt a` gives each core's sibling the core's untouched items
  (the plan's rest, then the cold ones) to stream into their shared L2 up to `a` items ahead, from the run's start:

| siblings | run p50 (4 cold a core) | touched a core | run p50 (8 cold) |
|---|---|---|---|
| halted (C2) | 587–598 µs | 10.6–11.1 | 786 µs |
| spinning, no streaming (control) | 564–575 | 12.0–12.3 | |
| streaming 1 ahead | 518 | 11.0 | |
| 2 ahead | 537 | 12.3 | 735 |
| 3 ahead | 505 | 12.4 | |

  Against the spinning control (the fair one: a busy sibling alone speeds a core's DRAM touches 10–15 %, metatron-cpu
  §3), streaming is worth ~8–10 % of the bench's run. Fill counters do not compare across these rows (the sibling's
  fetches count on its own thread). Streaming raises the touched lines' loss (6 → 11–15 %): its bytes land in the L3
  while the warm items wait, as the bytes model says.

## How the bench and the node differ (to quantify, one at a time)
1. **The shared CCD queue and stealing.** The node's cold items are claimed from one queue by the CCD's 8 cores and its
   warm items can be stolen by mates; the bench's cores own private sets. The queue is the node's load balancer.
2. **Two phases.** The node runs gate/up then down as separate pool runs with a join between; the bench one.
3. **Both sockets and NUMA placement.** The node's experts are placed across the four nodes and its 92 cores span both
   sockets; the bench used socket 1 (socket 0's memory is full while the node is resident).
4. **The prefetch window before the run.** The node's warm lines were touched up to ~460 µs earlier by the same core,
   during a window when other cores touched theirs; the bench touches immediately before.
5. **What else runs.** The node's main thread drives the cards and the step (route, dispatch, join) on core 1; the
   cards' BAR traffic and the loader's expert loads cross the fabric; prow rows and zones are written.
6. **Power and clocks.** The node runs steps back to back under the 280 W socket cap; the bench in bursts.

## Directions this opens (architecture)
- **Keep cold streams out of the warm lines' way.** `prefetchnta` on the cold items' reads (testing, below); Zen 3's
  L3 QoS (resctrl CAT) cannot tell a core's warm reads from its cold ones, but SMT siblings can be given different
  classes: the 96 idle hyperthreads as DRAM streamers with a small L3 mask, the cores computing on what they stream into
  L2.
- **Balance with the bus in mind.** Keep a shared queue for the end of the run (the reserve variant) and spread the cold
  demand over the run's length without taking the balancer away.
- **Fewer cold bytes at all.** The predictor (098), the cards' residency, and formats (the trellis) all act on how much
  of a layer is cold.

## Next
1. Done: `prefetchnta` (no effect) and the fill counters (no eviction).
1a. Done: the interference matrix (above) — the CCD's memory path.
1b. Done: with the socket loaded, 3 streamers a CCD saturate it; 5–7 slow its L3-fed cores 2–5×.
1c. Done: the cap in the node — no gain (above); the bench's mechanism is not the node's limit.
1d. Done in part: the item's code is not the cost; the DRAM lines inside prefetched items are (above).
1f. Done: the leak reproduced in the bench (fresh sets) and a second touch removes it (above).
1g. Done: the re-touch in the node (lab) and the window model — the leak falls, the step does not (above).
1h. Done: the node's loss map by age, run churn, claim and plan (above).
1i. Done: oldest first (moves the loss) and non-temporal cold reads (raise it) — above.
1e. The node's fill counters by item, with its class (own / CCD / stolen) and its slice's touch time (lab): where a
    complete expert's 14–18 % DRAM lines come from — evicted (own items) or a mate's L2 (stolen).
2. The node's warm item at its best: a lab run with the prefetch posted but no cold traffic (the run's misses forced to
   the cards or skipped) to read the node's own floor for a warm item.
3. The differences above, one at a time in the bench, until it reproduces the node's sign.
