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

## Ruled out
- **Token count:** at 1 token (the served ~1.15 a verify-step expert) the bench still favours interleaving (350 vs 432 µs).
- **Item size:** at the served gate/up pair size (~544 KB, 256 rows) still (416 vs 469).
- **Uneven warm sets** (cores reach their cold items at different times): at 9 ± 6 warm items a core still (357 vs 450).
- **A warm set inside L2:** interleaving's gain shrinks (330 vs 350) and the warm items slow (7.65 → 11.25 µs).
- **Eviction of the warm lines:** the fill counters show them L3-resident in every order; `prefetchnta` on the cold
  reads (1 and 4 KB ahead) changes nothing (warm 13.5–13.7 µs in warm-first, as without).

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
1a. The interference matrix (bench): k cores of a CCD on warm items while j others stream cold, j = 0…7 — the warm
   item's cost against j, with the cores' effective clock (APERF/MPERF) beside it: L3 or fabric contention, or power.
2. The node's warm item at its best: a lab run with the prefetch posted but no cold traffic (the run's misses forced to
   the cards or skipped) to read the node's own floor for a warm item.
3. The differences above, one at a time in the bench, until it reproduces the node's sign.
