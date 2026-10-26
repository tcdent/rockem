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
1d. The node's fill counters by claim class (own / CCD / stolen) and its cores' clocks during the run (lab): why its warm item costs 12 µs.
2. The node's warm item at its best: a lab run with the prefetch posted but no cold traffic (the run's misses forced to
   the cards or skipped) to read the node's own floor for a warm item.
3. The differences above, one at a time in the bench, until it reproduces the node's sign.
