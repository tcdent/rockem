# 099 — The expert run as a system

**Status:** Open — the system is quantified (2026-10-25/26); one lever is bench-proven in the steady state and waits on a
node lab; the mechanism of the L3-rate state is unidentified
**Area:** decode, the host's expert run, the cache hierarchy, the pool's scheduling, how the benches measure
**Gain:** the re-touch, ~10 % of a layer's run in the steady state (~1.5 ms a verify step at width 3, inference); the
rest of the run is its cold bytes

> yeah hang on don't give up on it yet unless you can explain why it's not working then we still have some more
> discovery to do and modeling the hardware like you're doing at the very edge of our understanding is super valuable
> — tcdent

> youre trying to get the hode to match the bench not the bench to match the node — tcdent

> page sizes and many other things can change on the node if you find gains in performance. we're trying to match best
> case cache hits and dram throughput and the stack can adjust in any way necessary — tcdent

> I still want to try to figure out this unexplained portion. It's very strange to me that we can't definitively
> quantify that — tcdent

This document is the current account. Appendix A is the record of how it was reached, kept because the order of
discovery is itself evidence; Appendix B is the list of candidates refuted on the way, kept so none is tried twice;
Appendix C holds the data tables. Where a number was taken in a regime later found misleading, it says so.

## 1. The question

The host's expert run is, per verify step, ~42 layers × one pool run of ~1,000–1,400 items over 92 cores: each item a
gemv over a 128–136 KB slice of an expert (gate/up or down), claimed from a per-worker deque (slices this core touched in
the prefetch window), from a mate's deque, or from the CCD's shared queue (slices nobody touched: cold). The window before
the run (~440 µs, the cards' phase) touches the predicted experts' slices from DRAM so the run finds them in L3.

090's bench found interleaving a run's cold items with its warm ones 25 % faster than warm-first; the node, 10 % slower.
A bench and the node disagreeing in sign meant the bench was missing part of the system; this document is what the
system turned out to be. Re-measured with warmed-up cores at the end (§3, rule 1), the bench's advantage is 12 %, not
25 % — a third of the premise was a measurement artefact. The sign disagreement stands, smaller, and is open (§5).

## 2. The system as measured

### 2.1 The node's run, by item class (lab-order3 and lab-lossmap, 40–80k layer runs at 18k)

A layer run at width 3, per run (the loss map, 2026-10-26; `bringup/reads/lossmap.sh`, `items.sh`):

| item kind | items | µs an item | lines from DRAM | core-µs a core | DRAM MB |
|---|---|---|---|---|---|
| own deque, touched | 720 | 10.6 | 9.8 % | 82.7 | 8.7 |
| a mate's deque (stolen), touched | 254 | 13.3 | 21.7 % | 36.7 | 8.0 |
| own, touch abandoned mid-item | 111 | 13.3 | 55 % | 16.1 | 5.9 |
| CCD queue (cold) | 335 | 33.5 | 93.9 % | 121.7 | 45.2 |

The cold items are ~half the run's core time at width 3, two thirds at width 4; the touched items carry 36 / 25 / 16 % of
the run's DRAM bytes at widths 2 / 3 / 4. A run's workers end 23–50 µs apart (p10 to max); per verify step the pool's
cores wait ~3.2 ms in these tails (`pool_run` rows). A touched slice's lines come from DRAM at the run in proportion to
the fresh bytes its CCD inserted after it — its age in the window, and the run's own reads before it: 1–2 % young and
before the run's first MB, 23–35 % past 8 MB of either (Appendix C). The oldest slices — the early plan's, the front of
each deque, which thieves take — are the ones lost.

### 2.2 How the L3 behaves (the model; measurements in docs/metatron-cpu.md §3)

- **Victim cache; the toucher must be the reader.** A line a core touched sits in its L2 until evicted; a CCD-mate's
  read of it refetches local-homed memory from DRAM (l3-core-boundary, 2026-10-22).
- **Two classes.** A line that entered L3 once (an L2 victim) is replaced first, oldest first; a line that has hit in L3
  is protected. One L3 hit promotes a line.
- **Expiry is bytes, not time.** At 12 MB a CCD the oldest tenth of a touched set loses ~19 %, the newest ~4 %; 24 MB
  loses 41 %; idle time changes nothing. Fresh lines lose to the previous round's read (protected) lines.
- **A warm item's price:** ~4.3 µs + ~10 ns per line that comes from DRAM instead of L3 (8 cores of a CCD, steady state).
  Fresh lines read once: 8.4 µs (387 of 2,176 lines lost); touched twice: 4.7 (71 lost). One core alone loses nothing.
- Dirty victims are not kept; `prefetcht0` touches keep ~7 %; a non-temporal read does not protect (and the hinted gemv
  reads ~35 % slower); ordering moves the loss, it does not remove it (oldest-first loses the newest instead);
  non-temporal cold reads raise it (the cold lines are the fresh lines' cheapest victims).

### 2.3 The L3-rate state (found 2026-10-26; `bench-order --gap-*`, `bench-run --simple`)

A core's rate of L3-hit delivery depends on its recent L3-hit traffic, with millisecond dynamics:

| what the core did before reading its L3-resident set (8 cores a CCD) | item µs |
|---|---|
| continuous L3 hits (steady state) | 4.35 |
| 100 / 500 µs without hits, unprimed | 5.21 / 6.33 |
| 1 / 3 / 10 ms without hits | 5.94 / 7.5 / 17.4 |
| 3 ms of `pause`, a register-only loop, `mwaitx`, L2-resident reads or DRAM streaming (`prefetchnta`) | 7.1–7.8 (all decay) |
| 3 ms of L3 reads of any data at ≥ ~5 GB/s (256 KB every 50 µs) | 4.3–4.4 (held) |

- Uniform over the run that follows; the core clock is *higher* after a gap (3.72 vs 3.52 GHz) — not the clock.
- Per core: one core's hits do not hold its mates; another CCD's traffic does not hold this one.
- DRAM-fed reads are unaffected. Only L3 hits hold or restore it; L3 misses do not.
- **It costs only under load, and recovers slowly under load:** one core recovers within one 2 MB round; eight cores
  sharing an L3 take ~130 rounds (~10 ms: 263 → 125 → 93 → 81 → 72 µs a 16-item round from cold).
- **The node lives in it:** every run follows a ~440 µs window of DRAM touching and `mwaitx` waits — no hits — so its
  warm items run at the decayed rate on every layer. The node's 10.6 µs own item is this plus its lost lines.
- Mechanism not identified. A per-core structure does not fit the recovery (one core: one round; eight: 130); a shared
  L3 structure whose benefit many cores' hits re-earn slowly, and which idle ages away, fits everything measured.
  Candidates not separated: the L2 prefetcher's adaptive depth, L3 request credits, replacement-state convergence.

### 2.4 The CCD's DRAM path, and the sum of sources

`bench-interfere` (2026-10-25; its warm core ran continuous hits, so it stands): a CCD's DRAM reads saturate at ~40 GB/s
with 3–4 streaming cores alone, ~17–18 GB/s each with the whole socket streaming; every streamer past saturation adds no
bandwidth and slows the CCD's L3-fed cores 2–5× (a warm core 4.6 µs alone, 12.7 beside 7 streamers on its CCD, 23 with
the socket loaded). Not power (3.45 GHz throughout), not the socket (streamers on another CCD change nothing). The node's
run streams cold items on every core of every CCD.

**A CCD's run time is the sum of its sources' times, not their maximum** — re-measured in the steady state
(`bench-run --warmup 100 --pool-mb 2560`, both sockets, ~16 MB a CCD a round): all-warm 176 + 104 = 280 µs, all-cold
470 + 232 = 702, the node's mix 219 + 122 = 340 against a sum of 381 and a maximum of 214. L3-fed and DRAM-fed reads on
a CCD do not overlap; so interleaving, caps, barriers and sibling streaming move time between sources under one path.
(The all-warm configuration's 15.3 µs an item is the decayed rate: its 3 ms touching window has no hits in it.)

### 2.5 What does not matter (measured)

The compiled kernel (121 vs 125 instructions, identical loads, in both benches); the item's code (the node's gate/up
part with SwiGLU and group quantize, the down part with its `sfence`: 4.55–4.76 µs on one core, as a bare gemv);
page size (4 KiB / 2 MiB: no change to the gemv's stream from L3 or DRAM); memory shape (one shared allocation or
private Vecs, 2 MiB-aligned regions or odd strides, item padding, read order, rotation); small-buffer false sharing
(private staggered buffers); the main thread's placement; start synchronization (futex vs spin); the SMT siblings'
C-state (an awake sibling costs the core ~10 % of its compute under the power cap and gives the DRAM stream nothing);
cross-socket interaction (one socket alone runs the same walls as both); the per-CCD share of items by worker count.

## 3. How to measure this (the rules that would have saved a day)

1. **Warm the cores up.** A bench that reads L3-resident sets measures the recovery of §2.3 unless its cores have done
   hundreds of rounds of hits first: `bench-run --warmup`, `bench-order --warmup`. `bench-order`'s orders run in sequence
   (WarmFirst first, WarmOnly fourth): without a warm-up the first order is in the transient and the fourth is not —
   which is how 090's 25 % and `bench-run`'s "unexplained 1.45×" both arose. Every `bench-run` and `bench-window` number
   before the evening of 2026-10-26 was taken in the transient; Appendix A says which were re-measured.
2. **The pool must not wrap.** (warm-up + measured rounds) × ~15 MB a CCD at the node's shape; a 256 MB pool wrapped 17
   times made the "cold" items 35 % L3 hits (the two-class policy keeps twice-hit lines). `--pool-mb 2048` for 130 rounds.
3. **Model the node with fresh lines.** A fixed set is protected by its own reuse and loses nothing; the node touches a
   new layer's experts every window.
4. **One core proves nothing about eight.** One core alone reads fresh lines at 4.7 µs, recovers from idle in one round
   and loses no lines; the node's effects appear with the CCD's eight.
5. **Record homing, the toucher, and what the cores did in the millisecond before.** `/proc/<pid>/numa_maps`, free memory
   a node, toucher = reader (l3-core-boundary); the resident node slows benches run beside it (socket 1 alone: 287 µs a
   phase with the node resident, 197 with its memory free) — use the bench boot `bringup/box/dev.cmdline.bench`.
6. **Counters:** `common::pmc::FILLS` = demand fills from L3 / local DRAM / far, L2-prefetch hits in L2 / in L3 / misses.
   An L3-fed gemv item is ~2,190 prefetch hits in L3 and ~20 demand fills; the L2 prefetcher brings every line. Fill
   counters are not comparable across `prefetchnta` reads or sibling-streamed items. Reading MSRs between phases on many
   cores desynchronizes them and changes the rate (46 cores: 3.06 → 3.72 GB/s); take clocks around whole runs only.
7. **`box.sh` returns at the command's end marker** (before 2026-10-26 it waited its whole limit; a 10 s bench blocked
   200 s). A bench round is ~0.1 s of measurement; `bench-run` fills pools in parallel. A failed build must stop the chain
   (`exit ${PIPESTATUS[0]}`), or the previous binary runs under the new flags and reports a baseline as a result.
8. **The benches and what each answers** (bringup/bench/src; musl binaries in bundles/tools; logs in bringup/box/logs):
   `bench-run` — the node's schedule as the reference (deques, stealing, CCD queues, two phases, window; `--warmup`,
   `--retouch`, `--hold`, `--smt`, `--ccd-barrier`, `--multi`, `--simple`/`--raw` for the bare loop); `bench-order` — a
   warm/cold mix in five orders, fresh sets, gap modes (`--gap-us --gap-mode pause|busy|mwaitx|l2|l3|l3ring|dramnta|
   pause_l3`), the L3-rate state; `bench-interfere` — one warm core against j streamers; `bench-item` — the node's item
   shapes on one core, with cold-line fractions and background streamers; `bench-l3life` — a touched set's survival by
   position, gap, filler; `bench-window` — a private-list model of window then run (superseded by `bench-run`; its numbers
   were taken cold); `bench-power --sib-spin` — power and clock by sibling state; `bench-l3`, `bench-reuse` — the
   2026-10-22 core-boundary work. The lab node code (loss map, re-touch, oldest-first) is
   `bringup/expert-cache/labs/099-lossmap-retouch-fifo.patch`; the reads are `bringup/reads/{lossmap,retouch_ab,items}.sh`.

## 4. Levers, with steady-state numbers

`bench-run` at the node's width-3 shape, both sockets, `--warmup 100 --pool-mb 2048`, cold items 95 % DRAM, two repeats
(`bench-run-warm-clean.txt`, `bench-run-warm-ceilings.txt`):

| lever | wall (gate/up + down) a layer | own item | stolen item | status |
|---|---|---|---|---|
| today's schedule | 223 + 122 = **345 µs** | 13.3 µs, 6 % DRAM | 16.6, 17 % | reference |
| **re-touch** — each worker re-reads the slices it touched 512 KB of touching earlier, through the window | 203–208 + 106 = **310, −10 %** | 11.1 | 12.5, 5 % | **bench-proven; node lab next** |
| warm-then-cold a CCD (a barrier before the cold claims) | unchanged | | | nothing |
| sibling streams the next cold item into the shared L2 | 219 + 125 vs 219 + 122 | 15.1 | 27.5 | zero-sum (cold items 18.5 µs, warm ones slower) |
| a ring hold (a 1 MB L3 buffer a core read at 256 KB / 50 µs through the window) | worse: own DRAM 6 → 15 % (taken cold) | | | negative: footprint |

The re-touch is the hold of §2.3 with no footprint: L3 hits through the window (holds the state), promotion of the
touched lines (protects them: the stolen items' losses 17 → 5 %), no extra lines. Its earlier null results — the node lab
of 2026-10-26 morning, `bench-window`, cold `bench-run` — were taken in the transient; in the node lab the amend also drops
the pending re-touches and a worker's last 512 KB is never re-read.

Measured in the node and found zero-sum: a per-CCD cap on cold streamers (cold items 2× faster, fewer of them, the step
unchanged — the "warm" cores were streamers too, 14–44 % of their lines from DRAM); oldest-first (own items lose more,
stolen less); the morning's re-touch (complete experts' DRAM 14 → 9 %, the partial experts' first touches fewer, the
step unchanged).

### 4.1 What would move the node, from the data

1. **The re-touch, built as the bench ran it** (every slice re-read 512 KB later, kept across the amend, the last 512 KB
   included), with the loss-map measurement: predicted −10 % of the run, ~1.5 ms a verify step at width 3. If the node
   does not show it, the disagreement goes back to the bench first (§5).
2. **Fewer cold bytes:** the cold items are the run — 098's predictor, residency on the cards, formats.
3. **The clock under the 280 W cap** (power per MAC, metatron-cpu §4).
4. **The L3-rate state's mechanism:** if it is a prefetcher or credit setting there may be a knob; if it is replacement
   convergence there is not. Until named, the hold is the lever.

## 5. Open

- **The mechanism of §2.3.** Discriminating experiments: a pointer chase (latency) in the decayed vs steady state — a
  latency rise names the interface, a bandwidth-only loss names concurrency (prefetch depth or credits); the recovery's
  dependence on the number of cores hitting (2 and 4 between 1 and 8).
- **The node vs the bench, two disagreements, by the rule bench-first:** interleaving (+12 % in the bench warm, −10 % in
  the node) and the re-touch (−10 % in the bench, nothing in the node lab). What the node has that `bench-run` lacks:
  the amend mid-window (drops pending re-touches; re-plans), real multi-token items (the bench's 86/12/2 % mix is
  `--multi`), the SwiGLU/quantize between phases, the main thread's route/dispatch on core 1's CCD, the cards' BAR
  traffic and the loader, prow rows. One at a time.
- **The node's residual:** its own item prices at ~5.8 µs by §2.2 plus the decay of its window; it measures 10.6.
- **The 3.2 ms of tails a step** (workers waiting for a run's last core): the two phases' join and the queue's end have
  no bench model yet.

## Appendix A — The record (2026-10-25/26), in the order it happened

Each step's finding, and what later superseded it. Logs: `bringup/box/logs/<name>.txt`.

1. **Per-class item times in the node** (lab-order3): own 11.7 µs, stolen 13.1, cold 34.3; interleaving +10 %. Stands.
2. **The bench's fill counters** (`bench-order`): warm items L3-resident in every order yet 2.3× slower beside
   streamers — "interference, not eviction". Stands as the §2.4 mechanism; it is not the node's warm-item cost.
3. **The interference matrix** (`bench-interfere`, `-bg`): 3–4 streamers saturate a CCD; the socket loaded, 3. Stands.
   Led to the cap, and to an "architecture" sketch of streamers and computers by role that the cap refuted.
4. **The cap in the node** (lab-cap): no gain; the "warm" cores were streamers too. Stands.
5. **`bench-item`**: the item's code is not the cost (4.5–4.8 µs on one core, every shape). Stands. The DRAM lines
   inside prefetched items named as the cost and "removing the leak −16–20 %, ~3 ms a step" estimated — retracted at 7.
6. **The leak reproduced** (`bench-order --fresh`): Milan's L3 drops fresh lines first, one hit protects. Stands (§2.2);
   it was already in docs/metatron-cpu.md §3 from 2026-10-22, unread — the day's first avoidable repetition.
7. **The re-touch in the node** (lab-retouch) and `bench-window`: the leak falls a third, the step does not move; the
   −3 ms estimate retracted. Superseded in part: both were cold measurements; the steady-state bench says −10 % (§4).
8. **The loss map** (lab-lossmap): loss by age and run churn; the layer budget (§2.1). Stands.
9. **Oldest-first** (lab-fifo) and **non-temporal cold reads** (`bench-window`): move or raise the loss. Stands.
10. **`bench-run`** built as the schedule reference; its construction measured mutex starts, futex wakes, shared
    counters, cross-socket steals (a far-socket item 625 µs), and the resident node's drag on benches. Stands as rules.
11. **2 MiB pages, per-CCD shares, token mix, sibling streaming** in `bench-run`: taken cold; sibling streaming re-measured
    warm (zero-sum, §4); the others do not matter (§2.5).
12. **"One delivery path, the sum not the max"**: taken cold; re-measured warm and holds (§2.4).
13. **The L3-rate decay**, found by refusing the two benches' 3× disagreement on identical lines (§2.3). Stands.
14. **The ring hold** at the node's shape: negative (footprint). Stands. The re-touch as the footprint-free hold.
15. **Warm-then-cold barrier**: nothing. Stands.
16. **The 1.45× residual** between `bench-run --simple` and `bench-order` on identical lines: resolved as the recovery
    transient (§2.3, §3.1) after every candidate of Appendix B was refuted; `--warmup` added to both benches.
17. **The premise re-measured warm**: interleaving +12 %, not +25 % (§1).
18. **The steady-state node shape**: the re-touch −10 %; the sum of sources holds; sibling streaming zero-sum (§4).

## Appendix B — Refuted candidates (so none is tried twice)

For the node's warm item and the benches' disagreements, each refuted by a measurement: the item's code (bench-item);
CCD streaming interference as the node's warm-item cost (the cap; the warm-then-cold barrier); eviction of warm lines
in the bench's fixed sets (fill counters); power and clock under these loads (3.45 GHz throughout bench-interfere; 3.45
vs 3.63 GHz between the benches); the socket (other CCDs' streamers); page size; 2 MiB-stride and page-level aliasing
(pad, rotation, 4 KiB pages, odd strides in bench-order); memory shape (shared allocation vs private Vecs, both ways);
small-buffer false sharing (staggered private pages; the page-aligned variant aliased in L1 and read 7 % slower — a
confound to remember); the main thread's placement and spinning; the round machinery, the per-item wrapper, the deque
claims (`--simple`, `--raw`); a big pool as ballast; the compiled gemv; the SMT siblings' C-state as a DRAM lever (an
awake sibling costs ~10 % compute; the scalar touch's +10 % was the access pattern's, not the node's touch); a per-CCD
cap; a ring hold; oldest-first; `prefetchnta` on warm or cold reads; `bench-window`'s sibling-streaming gain (cold,
private lists; not reproduced by the faithful schedule, cold or warm).

## Appendix C — Tables kept from the record

**The node's loss map** (own touched items, re-touch off; lines from DRAM by age — fresh bytes the CCD touched after the
slice — and the CCD's run reads before the item):

| age \ run before | <1 MB | 1–2 | 2–4 | 4–8 | 8+ |
|---|---|---|---|---|---|
| <1 MB | 10.7 % | 1.1 | 1.2 | 2.0 | 6.7 |
| 2–3 MB | 3.8 | 1.0 | 1.5 | 2.3 | 8.0 |
| 4–6 MB | 3.0 | 1.5 | 2.5 | 3.5 | 11.0 |
| 6–8 MB | 3.5 | 2.6 | 4.3 | 5.9 | 15.0 |
| 8–12 MB | 5.5 | 5.8 | 9.3 | 11.4 | 22.9 |
| 12–16 MB | 10.8 | 13.0 | 18.7 | 22.3 | 35.1 |

**The node's lines from DRAM by prefetch status** (`prefetch_use`, verify steps): complete experts 14.0 / 16.4 / 18.3 %
at widths 2 / 3 / 4; partial 43.8 / 32.9 / 27.4 %; missed ~94 %.

**The cap in the node** (24 decodes at 18k, the cap cycling per step): `experts` a step p50 at widths 2 / 3 / 4 — no cap
13.6 / 21.4 / 29.1 ms, cap 3 14.8 / 21.3 / 32.2, cap 2 14.3 / 21.0 / 28.5; cold items 34.8 → 19.6 → 14.9 µs, own items
11.8 → 13.2 → 12.9, a run's wall 150 → 165 → 185 µs.

**The lab re-touch** (alternate steps, 24 decodes): `experts` p50 off → on at widths 2 / 3 / 4: 12.27 → 12.21, 17.97 →
18.35, 26.55 → 24.99 ms; complete experts' DRAM lines 12.2 → 9.0 / 14.5 → 9.3 / 15.4 → 9.9 %; partial experts' 40 → 43 /
32 → 37 / 27 → 31 %.

**Fresh-line loss by the CCD's touched bytes** (8 cores, warm only): 3.3 MB 0.3 %, 6.5 MB 2.6 %, 9.8 MB 7.2 %, 15.2 MB
15.8 %, 21.8 MB 33 %; touched twice at 15.2 MB: 2.0 %; three times: 0.2 %. By core count, 15 items each, lines lost of
2,176 → item µs: 1 core 11 → 4.31; 2 32 → 4.47; 4 125 → 5.35; 8 387 → 8.41; touched twice at 8: 71 → 4.72.

**The gap curve** (8 cores, 15 L3-resident items each, unprimed): 20 µs 4.64, 50 4.86, 100 5.21, 200 5.52, 500 6.33;
primed by a pre-touch: 1 ms 5.94, 3 ms 7.3–7.6, 10 ms 17.4. Hold by a rotating 1 MB ring per 50 µs slot: 1–16 KB nothing,
64 KB 5.94, 128 KB 4.67, 256 KB 4.40, 512 KB 4.37. Recovery after 10 ms idle, by full-rate L3 reads: 100 µs 16.6, 300 µs
10.9, 1 ms 4.35.

**The recovery curve from cold** (`bench-run --simple`, 8 cores, 16 items a worker, every 10th round, µs): 263 125 110 93
95 89 92 90 86 80 81 79 75 73 75 73 72 72 74 75 72 … 72; one core: 142 then 69.

**bench-interfere** (one warm core on CCD 7, j streamers on its CCD / another): j = 0 4.56 µs; 1 4.55 / 4.60; 2 4.59 /
4.56; 3 4.58 / 4.60; 4 5.17 / 4.58; 5 7.11 / 4.57; 6 10.25 / 4.57; 7 12.70 / 4.58; the CCD's streamers 17.6 → 40.0 GB/s,
saturating at 3–4; with the socket loaded (38 streamers elsewhere) the CCD saturates at 3 streamers, 17–18 GB/s, and the
warm item reaches 23 µs at 7.

**The 090 premise, warm** (`bench-order --warmup 300`, 46 cores, 9 warm + 7 cold fresh items a core): WarmFirst 413 µs
(warm item 11.7), ColdFirst 434, Interleave 362 (12.0), WarmOnly 101 (8.3), ColdOnly 311; cold: 440 / 472 / 358 / 112 / 303.
