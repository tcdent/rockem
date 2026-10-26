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

## The run's reference: `bench-run` (2026-10-26, a bench boot)
> youre trying to get the hode to match the bench not the bench to match the node — tcdent

The bench is the reference for what the hardware delivers under a schedule; where the node falls short, the shortfall is
the node's, and the bench measures each node element's cost by adding it as a variable — the bench is not tuned to look
like the node.

`bench-run` runs the node's schedule (construct.rs `work`/`slot_order`): each CCD's planned slices touched in the window
by whichever of its cores claims them, then two phases (gate/up, down) with a join, each worker claiming its own deque
from the back, its mates' from the front, its CCD's queue, its node's other CCDs. Run on a bench boot
(`bringup/box/dev.cmdline.bench`: the serving command line without the bundle and the hugepage reservation, so the benches
have all four nodes' memory). At the node's width-3 shape (45 slices touched + 28 cold items a CCD ≈ 118 items):

| | gate/up | down |
|---|---|---|
| wall | 214 µs | 122 µs |
| workers' ends p10 / p50 / p90 / max | 155 / 168 / 206 / 213 | 76 / 88 / 116 / 121 |

per item: own 11.2 µs (6 % DRAM), stolen 14.2 (17 %), CCD queue 31.5 (96 %), the node's other CCDs 30.6 (98 %). Repeats
agree to a microsecond; one socket alone gives the same walls (214 / 122): the sockets do not interact.

What building it measured on the way (each a bench flaw first, then fixed — and each a property the node must keep):
- **Workers that start through a mutex** (each taking the phase's lists under a lock): 92 handoffs across the sockets
  staggered the starts; walls 287 → 214. The node publishes its job through an atomic pointer.
- **A futex barrier's wake** (tens of µs) as the phase's start; the node's workers spin on a generation word.
- **Shared per-item counters** (one set of atomics, 4 adds an item from 92 cores) and a barrier's arrivals on the
  spinners' line: each cost ~35–75 µs a phase across the sockets.
- **Stealing across the sockets** (`--steal any`): a far-socket cold item 625 µs, the gate/up wall 1,084 µs. The node
  never does (its last slot is the unplaced-item queue, not the other socket's CCDs).
- **The resident node slows these benches:** socket 1 alone, the same run, walls 287 / 155 µs with the node resident
  against 197 / 125 µs with its memory free (before the barrier fixes; the cause — its memory, its pages, its threads —
  is not separated). Bench numbers taken beside the resident node earlier in 099 carry this.

Against the node: at width 3 the node's run is ~390–420 µs a layer for the same item count against the bench's 336.
Before the difference is the node's to close, one workload element is still missing from the bench: experts picked by
two or three of the step's tokens (`gemm` over n tokens, more compute an item).

## `bench-run` on 2 MiB pages, and the SMT siblings in the reference (2026-10-26)
- **Every bench before this used 4 KiB pages:** metatron's THP is `madvise` and no bench advised; the node's experts are
  hugetlb 2 MiB pages. `huge.rs` (mmap, `MADV_HUGEPAGE`, verified in `smaps_rollup`) is now `bench-run`'s pool. On 4 KiB
  pages the NUMA nodes' runs differed by up to 28 % and the pattern moved between allocations (node 3 slowest in one
  session, node 2 in another); on 2 MiB pages every CCD ends within 212–225 µs and repeats agree to 1 µs. The per-item
  times rise (own 11.6 → 15.2 µs, the walls 218 → 228 / 124 → 127 µs) — not explained. **Reference, the node's width-3
  shape, 2 MiB pages: gate/up 225–229 µs, down 127–128 µs.**
- **Per-CCD shares by worker count** (`--share cores`; four CCDs have seven workers): no change — those CCDs were not
  the stragglers.
- **Stealing across the socket's two nodes** (`--steal socket`, 4 KiB pages): every CCD ends together, the wall does
  not move (218 µs).
- **Siblings streaming the next cold item** (`--smt 1`: each worker claims one ahead and its sibling, waiting in
  `mwaitx`, touches a cold next item into their L2): the cold items become L2-fed (33.6 → 18.8 µs, 95 → 6 % DRAM) and the
  warm ones slow (own 15.2 → 17.3, stolen 18.8 → 29.6); walls 225–227 → 229 / 127–128 → 132 µs. Zero-sum: the sibling
  moves bytes under a ceiling it cannot raise. In `bench-window` (private lists, 4 KiB pages) the same idea gave −9 %;
  the faithful schedule does not reproduce it. Not a lever as built.

## A CCD delivers through one path: the run's time is the sum of its sources (2026-10-26, `bench-run`, 2 MiB pages)
> page sizes and many other things can change on the node if you find gains in performance. we're trying to match best
> case cache hits and dram throughput and the stack can adjust in any way necessary — tcdent

The same schedule, both sockets, ~118 items (16 MB) a CCD a round, with every item warm, every item cold, and the mix:

| items a CCD | gate/up + down | a CCD's delivery |
|---|---|---|
| all warm, fresh (59 slices touched, the window long enough) | 174 + 104 = 278 µs | ~58 GB/s |
| all warm, **fixed** (the same lines every round: protected in L3, 0.1 % DRAM) | 179 + 103 = 282 µs | ~57 GB/s |
| all cold | 448 + 220 = 668 µs | ~24 GB/s (×12 ≈ 288 GB/s, the box's DRAM ceiling) |
| the node's mix (45 slices warm + 28 cold) | 225 + 126 = 351 µs | |

- **The mix costs the sum, not the maximum:** its 12.2 MB warm at 58 GB/s (211 µs) plus 3.8 MB cold at 24 (159 µs) is
  370 µs; measured 351. L3-fed and DRAM-fed reads do not overlap on a CCD: they share one delivery path (phase 0's "a
  CCD delivers ~60–70 GB/s whatever the source", now on the node's schedule and pages). This is why the cap, the
  re-touch and the siblings' streaming were zero-sum: they move bytes between the sources under one path.
- **A warm item's 15 µs is that path, not its lines:** the fixed set (no DRAM lines) costs 15.0 µs an item, the fresh set
  (8.5 % DRAM) 15.35. **Superseded the same day:** the ~58 GB/s is `bench-run`'s window-decayed state (below), not the
  CCD's L3 delivery — `bench-order` reads the same lines on the same 8 cores at 4.3 µs an item, ~250 GB/s a CCD. The
  cold (DRAM) rows stand.
- **The design consequence:** a layer's run is bounded by Σ bytes ÷ the CCD's rate for their source. Every byte a CCD
  reads in the run costs ~17 ns a KB from L3 and ~42 from DRAM; the levers are fewer bytes in the run (into the window,
  onto the cards, smaller formats) and the rates themselves (pages: below).

## A core's L3 read rate decays when its L3 hits stop (2026-10-26, `bench-order --gap-us`, CCD 7)
> youre trying to get the hode to match the bench not the bench to match the node — tcdent

Found by taking the two benches' disagreement to the bottom. `bench-run` read L3-resident items at 15–17 µs on 8 cores of
a CCD, `bench-order` the same kernel over the same lines at 4.3; neither per-item timing, deque claims, stealing, core
count, memory shape (one shared allocation, 4 KiB or 2 MiB pages), start synchronization, read order, phases nor
regions moved either (`bringup/box/logs/bench-run-{bisect,ladder,cores}.txt`, `bench-order-shared.txt`,
`bench-sync.txt`). One core alone: 4.27 µs in both. What `bench-run` had was a 3 ms window before every run.

**The effect.** Eight cores of CCD 7, each reading its fixed 2 MB set (15 items, L3-resident: ~2,190 prefetch hits in L3
an item in every row below, 0 from DRAM), after a gap in which the cores do no L3 reading:

| gap before the run | item µs | the run's clock |
|---|---|---|
| none | 4.35 | 3.4 GHz (the cap) |
| 50 µs (no pre-touch) | 4.86 | |
| 100 µs | 4.48 | 3.57 |
| 300 µs | 4.53 | 3.52 |
| 1 ms | 5.94 | 3.67 |
| 3 ms | 7.3–7.6 | 3.72 |
| 10 ms | 17.4 (4×) | 3.72 |

- **Uniform over the run** (first third / middle / last third 7.37 / 7.72 / 7.58 at 3 ms; 11.5 flat over 30 items, 345 µs,
  at 10 ms): a rate, not a wake-up; it does not recover inside a run.
- **Not the clock:** the cores run faster after a gap (3.72 vs 3.52 GHz) and the items are slower.
- **How the gap is spent does not matter:** `pause`, a register-only busy loop, the node's `mwaitx` wait — all 7.3–7.6 at
  3 ms. L2-resident reads (192 KB, no L3 requests): 7.81. DRAM streaming with `prefetchnta` (L3 misses, no pollution):
  7.09. **Full-rate L3 reads of any data (a 1 MB buffer of its own): 4.30 — no decay.**
- **Per core:** one core reading L3 does not hold its seven mates (7.28); CCD 8 reading L3 flat out does not hold CCD 7
  (7.44, CCD 8 itself 4.36). Not a package or fabric state the others can keep up.
- **What holds it:** a rotating 1 MB ring read in 50 µs slots (lines re-read after a full ring: L3 hits) — 1, 4, 16 KB a
  slot: 7.5–7.7 (nothing held); 64 KB (~1.3 GB/s): 5.94; 128 KB (2.6 GB/s): 4.67; **256 KB (~5 GB/s): 4.40**; 512 KB: 4.37.
  About a fifth of a core's L3 read rate, as hits, holds the state. (An earlier "2 % duty holds it" reading was a bench
  error: that mode read the whole buffer whenever its share was nonzero.)
- **Recovery after a decay** (10 ms idle, then full-rate L3 reads): 100 µs of reads → 16.6 µs an item; 300 µs → 10.9;
  **1 ms → 4.35**. Recovery takes ~0.5–1 ms of hits — longer than a node run (~400 µs), so a decayed core stays decayed
  through its run.
- **DRAM-fed reads are unaffected:** cold items 24.34 µs without a gap, 24.40 after 3 ms (the same run's warm items 4.37 →
  12.76). The state sets the L3-hit delivery rate only.
- **L3 misses do not count** (`dramnta`, the node's window), only hits.

**What it is** is not identified: a per-core state, driven by the core's own L3-hit traffic, with a time constant of
milliseconds, that sets the rate at which L3 hits are delivered to that core, independent of its clock. Candidates not yet
separated: the L2 prefetcher's adaptive throttle, a per-core request-credit allocation, a power state of the core's L3
interface.

**The prediction before any node change** (`bench-run`, the node's width-3 shape on both sockets, two repeats each):

| hold | gate/up + down wall | own item | stolen item |
|---|---|---|---|
| none | 232 + 129 µs | 15.2 µs, 6.5 % DRAM | 18.8 µs, 15.5 % DRAM |
| a 1 MB ring a core, 256 KB every 50 µs through the window | **258 + 135** | 17.8, **14.6 %** | 19.6, **34.6 %** |

The ring holds the state (one CCD, all warm, fixed: 15.5 → 9.0 µs an item, the wall 182 → 95) but at the node's shape it
costs more than it returns: its lines are L3 hits, so protected, and 8 MB of them a CCD displace the window's
once-referenced planned lines (the bytes model) — the warm items lose twice the DRAM share. The instrument with no
footprint is the planned lines themselves: re-reading slices touched 512 KB earlier is L3 hits (the hold), promotes them
(the protection) and adds no lines — the re-touch of 1g, whose lab gain (warm items 11.3 → 9.7 µs) has this second
mechanism behind it. Measured at the node's shape (two repeats each):

| | wall | own item | stolen item |
|---|---|---|---|
| none | 232 + 131 µs | 15.3 µs, 7.3 % DRAM | 18.8, 16.0 % |
| re-touch | 237–241 + 127 | 16.0–16.2, 7.0 % | 18.3, **5.0 %** |
| a 576 KB ring held at 256 KB / 50 µs | 257 + 132 | 17.9, 10.4 % | 19.1, 26.3 % |

The re-touch protects the stolen lines and returns nothing on the wall, as in the node lab. **So at the node's shape the
warm item's cost is not the decay** (holding the state through the window changes nothing) **but the interference**: in
every phase a CCD's cold items stream DRAM beside its warm ones, and an L3-fed core slows 2–5× while its mates stream
(the interference section). The CCD delivers the sum of its sources, not the maximum (measured above), so that overlap
is worth nothing: so the warm items first with the CCD's DRAM path quiet, then the cold ones (`--ccd-barrier 1`), with and
without the re-touch (two repeats each): own items **15.1–15.2 µs** (15.9 with the re-touch) — unchanged; cold items 34.3
→ 36.5 (less overlap); walls 232 + 129 → 240 + 135. **Not the interference either**, at this shape.

**Where the warm item stands** (L3-fed, 8 cores a CCD, the same kernel): `bench-order` on protected (fixed) lines 4.3 µs;
on fresh lines touched once this round 8.6 (3–7 % of their lines from DRAM — the lost ones); fresh lines touched twice 5.3;
`bench-run` on fixed lines with the hold 9.0; `bench-run` at the node's shape 15–16 whatever the variant; the node 10.6 own
/ 13.3 stolen. Two gaps are unexplained: `bench-run`'s ~2× over `bench-order` on identical lines (its warm numbers are not
a reference for the hardware; its cold and schedule numbers stand), and fresh lines costing ~2× protected ones under eight
cores while one core alone reads them at 4.7 — the node's case. **Resolved** (`bench-order --fresh`, CCD 7, 15 items a core):

| cores touching fresh sets | lines from DRAM an item (of 2,176) | item µs | touched twice: DRAM lines → µs |
|---|---|---|---|
| 1 | 11 | 4.31 | 2 → 4.27 |
| 2 | 32 | 4.47 | 2 → 4.27 |
| 4 | 125 | 5.35 | 7 → 4.31 |
| 8 | 387 (18 %) | 8.41 | 71 → 4.72 |

A warm item costs **~4.3 µs + ~10 ns a DRAM line in its stream**; the DRAM lines are the bytes model's losses (growing with
the CCD's fresh bytes), and one L3 hit promotes them (18 → 3 % lost). So `bench-order`'s numbers are all accounted for:
protected lines 4.3; fresh 8.4 by their lost lines; decay by the gap. The node's own items (6.5 % DRAM) price at ~5.8 µs
on this model against their 10.6; the rest is what `bench-run`'s 2× is made of — its candidate, from today's facts: eight
cores reading 2 MiB-strided regions or queue-claimed slices (~8 apart) in the same rhythm meet in the same L3 sets at the
same moment; a freshly booted box hands out near-contiguous 4 KiB pages too, which is why `--pages 4k` did not differ.
The node's experts sit at a 51/16 MiB stride (`layout::SHARE` = 3,342,336 B), which does not alias. Test: a 4 KB pad
after every item (`--pad-kb 4`): `bringup/box/logs/bench-run-pad.txt`.

**Why it matters here:** every one of the node's expert runs follows a ~440 µs window of DRAM touching and `mwaitx`
waits — no L3 hits for the core — so the node's warm items run in the decayed state on every layer; `bench-run`'s
window-shaped rounds reproduced the node's 15–17 µs warm item for that reason. The node's own 11.7–12 µs (lab-order3) is
this plus its lines' 10–14 % DRAM share. If the state could be held (or the mechanism switched off), the warm items go
~4.5–5 µs: at width 3 that is ~55 core-µs of the ~257 a layer, ~20 % of the run, ~3 ms a verify step — inference from the
budget above, to be predicted by `bench-run` with a hold before it is built in the node.

## The benches since 2026-10-26, and their wrapper
- `bringup/box/box.sh` returns when the command's end marker arrives (it waited its whole limit before: a 10 s bench
  blocked for the 200 s I passed). `bench-run` fills its pools in parallel, 256 MB a CCD. A bench round is ~0.1 s of
  measurement; the rest was these two.
- `bench-run`'s own residual: at window 0 it reads L3-resident items at 8.4 µs where `bench-order` reads 4.3 on the same
  cores and lines; its rounds' barriers and builds are gaps of tens of µs, which the curve above prices at ~10 %, not 2×.
  Open. Its DRAM-fed (cold) numbers and the schedule comparisons stand; its warm numbers carry this.

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
