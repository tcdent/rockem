# The expert path's caches: L2/L3 as a programmed cache

**Status:** Proposed (2026-10-20) — architecture for queue 082 and the decode L3 prefetch; nothing here is built
beyond what §1 lists as today's.
**Scope:** GLM-5.3-Flash on metatron (2 × EPYC 7R13: 12 CCDs of 8 cores, 32 MB L3 a CCD; 2 × 7900 XT), the routed
experts in host DRAM, decode verify steps of 1–4 tokens at 18k context; prefill in §8.

> "I expanded on the zero eight two document because I think this is a compelling use of the tuning and information
> that you're gathering with regards to us interacting with the L two and L three cache as deterministically as
> possible." — tcdent, 2026-10-20

> "Anytime we got an open question like this, we've got to get to the bottom of it. We may be running nondeterminism
> as an application, but this is a deterministic system." — tcdent, 2026-10-20

The design treats each CCD's L3 as a cache the node programs: a predictor says which bytes the next layer needs, a
planner turns that into an explicit per-CCD sequence of slices under a byte budget, a fixed set of cores touches
that sequence into the L3 in a fenced-off set of ways, and the run reads the same sequence in the same order. Every
part is observed by counters the run already carries. Prediction and prefetch change only which bytes are warm; the
arithmetic, and so the output, is untouched (long SCORE bit-identical).

## 1. Where it starts (measured)

| fact | value | source |
|---|---|---|
| verify step at 18k | ~57 ms, of which the CPU experts lap ~25 ms (~46 layer runs, ~580 µs each) | 2026-10-20 |
| host experts a layer run | ~16.6 (resident on the cards: ~4 a layer under the fixed 1M reservation) | 082, 031 |
| an expert's bytes | 13,369,344 B = 48 items of 278,528 B (32 gate/up, 16 down); ~1.11 MB a CCD, ~4 items, slices hashed over the node's 3 CCDs (`slice_slot`) | `layout.rs`, `construct.rs` |
| a run's bytes a CCD | ~18.5 MB (16.6 × 1.11), streamed at ~32 GB/s a CCD | arithmetic |
| time an expert, by prefetch status | 31.0 µs complete, 36.4 partial, 42.4 not predicted (R² 0.67) | 082, build pfuse |
| where a run's lines come from | complete 71.5 % L3, partial 51 %, not predicted 6.9 %; total ~47 % | build fills2 |
| why completed lines are lost | capacity, not sets: 86 % kept at ≤ 10 experts prefetched, 56 % at 23+; 87/81/71/62 % at 1/2/3/4 tokens; 1.2 % of lines past 16 a set | builds fills2, sets |
| L3-served `gemv` on 92 cores | 2.2× DRAM (802 vs 361 GB/s), neighbour-touched as fast as self-touched (~70 GB/s a CCD) | `bench-reuse` |
| `PREFETCH_CAP` | 13: experts lap 25.85 (uncapped) → 25.26 ms; 10 → 26.29; L3-served share stays ~47 % | build 4da52d7 |
| `slot` lap | ~53 µs a layer between a run's end and the next prefetch's start (sub-zones being measured) | 2026-10-20 |
| predictor replay | recall of host experts at cap 13: .584 today, .595 with every reordering of today's signal, .746 for a perfect predictor; per-token top-8 accuracy .51–.62 at layers 4–5, 8–9, 14, 18, 25, 27–28, .75–.81 at 31–41 | run 9593789a, `/tmp/predwork/` |

What these say together: a completed prefetch is worth ~2.2× when its lines survive, and today ~30 % of them do not;
the oracle replay puts better prediction at ~1.15 ms a verify step at cap 13, against ~3.6 ms for completed
prefetches that stay in L3 (082's estimates, inferred from the rates). The cap binds hardest at 3–4 tokens (17–22 host
experts; oracle recall .744 / .586), which gated depth 3 (081) makes common. So delivery and capacity come first,
prediction second — the order 082's "Next" already set.

## 2. The parts and who owns them

| part | runs on | owns | interface |
|---|---|---|---|
| **Predictor** (tcdent's design, §3) | the layer's FFN card, in the host experts' gap | its weights, its online update, the per-token probabilities | `pred[n][E]` in host memory + word `W_PRED` |
| **Planner** (§4) | the main thread, while it waits on the run's end | the per-CCD plan: an ordered slice list a CCD, cut at that CCD's budget | `Plan { per CCD: [slice], budget }`, read by the touch and the run |
| **Touch** (§5) | the keep cores: 4 a CCD (today's `PREFETCH_CORES`) | walking its CCD's plan into L3, in order | a light pool job; abandoned when the run posts |
| **Run** (§5) | all pool cores | the arithmetic (unchanged); its item order follows the plan | two queues a CCD: planned slices, the rest |
| **Ways** (§6) | resctrl, set once at boot | the keep/stream CLOS split of every CCD's 16 ways | static; no switching in the step |
| **Observer** (§9) | every pool item (`rdpmc`), resctrl monitors a step | the rolling rates | `prefetch_use`, `l3_plan`, `l3_ccd` rows |

The host owns the residency table and everything about bytes; the card owns prediction and its learning. The only
thing that crosses is a probability vector a token and one word.

## 3. The predictor (card side)

As tcdent designed it in 082, unchanged in substance:

> "I don't know if I'm necessarily attached to the idea that this needs to be frozen at any point, because I think
> within ten thousand tokens we would either see it settle or not, and that would prove its effectiveness"
> — tcdent, 2026-10-20

- **Where and when.** A launch on the layer's FFN card's queue after the shared expert's down projection
  (`rows8_at` over `ffn_down_shared`) and before the residents' jobs and `ffn_post`'s `hc_post`. Its inputs are
  already there: the next layer's route-ahead rows (`hc_route_ahead`, `scores_v`), this layer's picks (the act
  phase's), the FFN input `x`, and a per-layer ring of the previous step's picks at the next layer.
- **Output contract.** Per token, a probability for each of the next layer's 288 experts: `pred[SPEC_TOKENS][E]` f32
  (4.6 KB) in host-shared memory beside `scores_ffn`, with its own word `W_PRED` in the chunk's word set carrying the
  layer's FFN-half sequence value (`half_seq[l][ch][1]`), as every other handoff. Per token, not a union: the
  planner makes the union, and the replay and training keep the per-token form.
- **Deadline.** The end of the host's run of layer `l`. The planner waits on `W_PRED`; an overrun is visible as
  `plan.wait` time on the main lane. There is no fallback to the route-ahead word: one path.
- **Model.** `logit = route_ahead_score + correction`, the correction zero at start, so its first token is today's
  predictor. Terms: a bias an expert, co-routing from this layer's picks, repetition from the previous step's picks
  at the next layer, a low-rank projection of the FFN input.
- **Learning.** Online, never frozen. Labels are the next layer's picks, produced by that layer's act phase a layer
  later. One launch a MoE layer does both: the update of the (l−1 → l) predictor with layer `l`'s picks, then the
  prediction (l → l+1). The features of the pending prediction (KB a token) stay in the card's VRAM until their
  labels arrive. Loss: per-expert binary cross-entropy, which keeps the probabilities calibrated — the planner's
  objective needs calibrated numbers, not ranks.
- **Exactness.** The predictor's output reaches only the planner. Nothing it computes enters the forward.

**What the replay adds (run 9593789a, 2026-10-20).** Reordering today's signal is exhausted: wider top-k (10/12/16),
the previous step's picks (alone .21 recall at cap 13; as a vote it hurts; as a tie-break ~0), per-layer adaptive
trust (+0.001) and calibrated probabilities together move recall at cap 13 from .584 to .595 (~0.08 ms a verify
step). The previous step's pick outside every token's top-16 recurs 2.3 % a token; inside top-8, 85 % — so repetition
mostly re-states what the route-ahead score already ranks high. The bias and co-routing terms were not replayed as
learned terms; by inference they act on the same candidates and are bounded the same way. The misses are per-token
router accuracy on a stale input (top-8 accuracy .51–.62 on the weak layers), so the gain has to come from a better
estimate of the next layer's FFN input — which is what the low-rank term over `x` can learn (it approximates what the
next layer's attention adds). Two consequences for the design, both decisions for tcdent (§12): the low-rank term
comes first rather than later, and the weakest layers are where its rank is spent.

**Which card.** 082 says card0. The inputs live on the layer's FFN card (`card_of(l)`: card0 for the first half of
the layers, card1 for the second), so the design runs each layer's predictor where its inputs are; the one layer pair
that crosses the cards reads its labels from host memory (the act phase's picks, `XA_P`, already there).

## 4. The planner (host side)

**Objective.** Per CCD, maximize the expected L3-served bytes the run will use:

  maximize Σ_e P(e) · bytes_c(e) · keep_c(B)  subject to  Σ planned bytes_c ≤ B_c

where `P(e) = 1 − Π_i (1 − p_i(e))` over the step's `n` tokens (0 for a resident expert), `bytes_c(e)` is expert
`e`'s slices on CCD `c` (by placement: node 2's CCDs hold less of every expert, 22.2 % against 24.8–26.7 %),
and `keep_c(B)` the fraction a CCD's L3 keeps at plan size `B` (measured, §6). Every byte of an expert has the same
value density `P(e)`, so the greedy walk in descending `P` is optimal: a CCD takes experts in `P` order until its
budget is spent or `P` falls below a floor `θ`. `PREFETCH_CAP` 13 is today's point estimate of this argmax with
ranks standing in for `P`; the replay's calibrated form at θ 0.5 (hard cap 16) gives recall .589 at 2.36 wasted
experts and 12.1 prefetched, against today's .584 / 2.77 / ~12.5; θ 0.4 gives .621 / 2.99 / 13.3.

**The budget a CCD** is the smaller of two measured quantities, never a constant:

| bound | what sets it | today's size (inference) |
|---|---|---|
| capacity `K_c` | the bytes the CCD's L3 keeps until the run reads them: with the way split (§6), the keep ways' bytes; without it, the knee of the retention curve | keep 8 of 16 ways = 16 MB; without the split the knee sits near today's cap (13 experts ≈ 14.5 MB) |
| time `W · r_c` | the window from the post to the next run's start, times the CCD's touch rate (its DRAM share, ~26–31 GB/s with every CCD pulling, §3 of metatron-cpu) | a ~0.5 ms window (arithmetic: 57 − 25 ms over ~46 layers, less the draft and head) holds ~13–15 MB a CCD |

The two bounds land at the same size today, so the cap-13 optimum may be the window as much as the L3 (inferred).
`prefetch_job`'s `done` / `items` at cap 13 says which; §11's phase 1 reads it before anything else is built.

**The plan is one sequence a CCD.** Gate/up slices of the planned experts in `P` order, then their down slices in `P`
order — the order the run consumes them (the gate/up run precedes the down run). The touch walks it from the front and
stops where the budget or the run's post stops it; the run's planned queue is the same sequence. So the oldest lines
in the L3 are the first the run reads, the down slices (read last) are touched last, and a touch the run overtakes
loses its least valuable tail.

**Verify width and residency** enter only through `P`: a wider step has a larger union and a flatter `P`, and the
budget decides how much of it fits; a resident expert has `P` = 0. Gated depth (081) needs nothing special.

**Cost.** The plan is a table walk: `slice_slot` extended at `relearn()` into per-(layer, expert, CCD) slice lists
and byte counts, so planning is a sort of ≤ 32 union entries and a walk — target < 10 µs, done while the main thread
waits for the run's last items (the predictor's word is in by then). The post at the run's end is then a pointer
publish into a preallocated job.

## 5. The touch and the run follow the plan

**Order of events at a MoE layer `l` of a verify step:**

| when | card (the layer's FFN card) | host main thread | pool |
|---|---|---|---|
| act phase done | `hc_pre`: router, act → `W_ACT` | `act.read`, residents' dispatch | touch(l) still running, abandoned by the next post |
| run(l) | shared expert; **predictor: update(l−1→l), predict(l→l+1) → `W_PRED`**; residents' jobs | its own items, then waits `W_PRED`, **plans l+1** | run(l), planned queue first |
| run(l) ends | `hc_post` waits for the host's word | **posts touch(l+1)**, then join, done word, account | keep cores: touch(l+1); others idle (MWAITX) |
| next layer's card phase (~0.5 ms) | attention l+1, then `hc_pre` | waits | touch(l+1) walks each CCD's plan |
| act phase of l+1 | | posts run(l+1) | touch abandoned; run(l+1) |

Today the post comes after the join, the done word, the accounting and the ~53 µs `slot` lap; here it comes first,
so the window starts at the run's end. The touch keeps today's mechanics (`touch`: a 32-byte load a line, yielding
every 4 KB once abandoned).

**Two queues a CCD.** A CCD's slot holds its planned slices (in plan order) and the rest (missed experts, and planned
slices past the budget). The keep cores drain the planned queue first, the stream cores the rest first; each takes
the other queue when its own is empty. The L3-served and DRAM-served streams then run at once on one CCD instead of
one after the other: by arithmetic (not measured), a CCD with 13 MB planned and ~5.5 MB missed finishes in
max(13/70, 5.5/32) ≈ 0.19 ms instead of ~0.58. The split also keeps the run's DRAM fills off the protected ways
(§6), which is what lets the two run together. Every item is the same arithmetic on disjoint output rows whichever
core takes it and in whatever order, so the reorder is exact.

**Bounds this makes visible (arithmetic, an upper bound, not an estimate):** with perfect prediction, a full window
and the measured 2.2×, a layer's run approaches ~0.2 ms against ~0.58 today. Prediction (oracle recall .746 at
today's cap, lower at 3–4 tokens), the window (~13–15 MB a CCD) and the clock under L3 traffic (§10) each take a
share of that; the phases in §11 measure each share separately.

## 6. Making the L3 deterministic

**What the node controls:**

| lever | how | status |
|---|---|---|
| which CCD holds a slice | `slice_slot` (fixed hash of expert and slice within its rows' node): the touch and the compute meet in one L3 | built |
| which cores touch, and how many | the light job's first 4 cores a CCD | built |
| what is touched, in what order, how much | the plan (§4) | proposed |
| when touching starts and stops | the post at the run's end; abandon at the next run's post | start moves earlier |
| the run's item order | the plan's order in the planned queue | proposed |
| which ways a core's L2 victims may enter | resctrl CAT: CLOS `keep` (ways 0–7) for the keep cores, CLOS `stream` (ways 8–15) for the stream cores, the main thread, the residency loader and housekeeping | proposed |
| page and set spread | 2 MiB pages; 1.2 % of a prefetch's lines past 16 a set | measured |

**Why a static split works on Milan.** The L3 is a victim cache of L2: a line enters it only when an L2 evicts it,
and CAT limits which ways that allocation may use; lookups hit in every way. So the CLOS that matters is the one of
the core whose L2 evicts the line. Fixing CLOS by core role gives: the touch (keep cores) installs into the keep
ways; the stream cores' DRAM streams install into the stream ways and can no longer evict a planned line; a stream
core that hits a planned line still hits it. The keep cores' own L2 victims during the run are planned lines already
read (dead) plus activations (small). The residency loader (core 24, on node 1's first CCD) and the main thread
stop evicting their neighbours' plans. No CLOS changes inside a step, so there is no MSR write or resctrl `tasks` move
on the step's path. Way counts are per CLOS per L3 domain, so a slower node's CCDs can take a different split.
The keep/stream boundary (8/8 above) is a measured choice (§11, phase 0), not a guess.

**What the node does not control:**

| | consequence | how it is handled |
|---|---|---|
| Zen 3's replacement within a CLOS's ways (undocumented) | which line a full keep set drops | keep the plan under the keep ways' bytes; then replacement never runs on planned lines |
| whether an L3 hit leaves the line in L3 (Milan's L3 is described as mostly exclusive) | whether consumed lines free keep ways | measured in phase 0; either way the plan's order makes consumed lines the first to go |
| the L2 hardware prefetchers (~98 % of a run's lines; demand fills ~2 %) | over-fetch past a slice's end on the touch | small; the touch's loads already pace them |
| the L3's clock (it follows the CCD's cores) under the socket's L3-traffic limit | L3 delivery falls with the clock | observed per run (§9, §10) |
| the cards' BAR writes | land in DRAM (no DDIO on EPYC): neither help nor hurt | — |

**Non-temporal reads.** On write-back memory `movntdqa` is a plain load on Zen 3, and the hardware prefetcher, not the
kernel's loads, brings the run's lines, so per-load hints cannot keep the run's fills out of L3; the way split is the
lever that can. `prefetchnta` for the touch is the opposite of what the touch wants. Neither is in the design.

**MBA** (1 % steps, per CLOS per CCD) is available to throttle the keep cores' DRAM pull if the touch is ever moved
into the run's tail; with the touch confined to the card phase, when DRAM is idle, the design does not use it.

## 7. Prefill

Prefill's expert run is instruction-bound and power-capped (183 GB/s gate/up against 375 decode, pool 93–94 % busy),
and a weight byte serves 9–64 tokens; warming weights at the dispatch waits measured no gain (078,
`prefetch_lead`, 17.78 → 18.24 ms a layer-chunk). The design does no weight prefetch in prefill: a 512-token chunk
picks nearly every expert, there is nothing to predict, and touches would spend power the gemm needs.

What carries over:
- **The way split stays.** It is static, so in prefill a stream core's victims have half a CCD's ways. The one
  known prefill tenant of the L3 is the activations' traffic (~10 % of the kernel's rate, metatron-cpu §3); whether
  half the ways costs it is unmeasured. Phase 2 measures prefill under the split; if it costs prefill, the CBMs are
  the knob, set once for both phases — one split, measured on both.
- **The predictor learns from prefill.** A chunk's picks are labels for every position; an 18k prompt gives ~18k
  labels a layer before the first decode token — past tcdent's ~10k-token settling horizon. It needs the route-ahead
  rows in prefill (today `hc_route_ahead` runs only for steps ≤ `SPEC_TOKENS`): one matrix-path launch a chunk on the
  FFN card. The cards' prefill time is not the bottleneck (the host's experts are 79 % of a prefill step's main lane,
  078). A candidate, decided by phase 4's settling curve.
- **The plan object is the one 078's whole-experts-a-CCD form would use** (a CCD's sequence of an expert's gate/up
  then down, the hidden vector kept in that L3). That form is 078's, not this design's.

## 8. Timing in numbers (a verify step, 18k)

| quantity | value | kind |
|---|---|---|
| run of a layer | ~580 µs | measured |
| card-side gap the predictor fits in | ~250 µs a layer (`hc_post` starts later) | 082 |
| predictor's work | µs (megabytes of weights) | 082, estimate |
| plan | < 10 µs target, off the critical path (inside the run's wait) | target |
| window for the touch | ~0.5 ms a layer | arithmetic; `prefetch_job.window_us` is the read |
| touch rate a CCD | ~26–31 GB/s (4 cores saturate a CCD's share) | measured (§2–3 of metatron-cpu) |
| L3-served gemv a CCD at full load | ~70 GB/s | `bench-reuse` |

## 9. The feedback loop and what success looks like

**Two loops, one fast and one slow.** The predictor's online update is the fast loop (every layer, on the card).
The budget is the slow one: a constant per CCD derived from phase 0's curve and phase 2's measurement, written like
`DRAFT_KEEP` — a measured number, not a switch. If the rolling rates below drift with the workload (code vs chat,
context length), the budget becomes a function of the step (width, `P` mass) fitted from the rows, never a runtime
toggle. An online budget controller coupled to an online predictor is two learners chasing each other; it is not in
the design unless the rows show drift.

**The rows** (`track-experts`; the items already read `rdpmc` around themselves at no measured cost):

| row | grain | fields |
|---|---|---|
| `prefetch_use` (exists) | layer run | host experts complete / partial / missed, predicted, items done, L3 hits/misses by status |
| `l3_plan` (new) | layer run | per CCD: planned bytes, budget, touched bytes, `P` mass planned, `P` of the cut, θ; the step's width |
| `l3_ccd` (new) | layer run | per CCD (`fill_acc` is already per slot, i.e. per CCD): hits/misses by status and by queue (planned / rest); the CCD's run end |
| `l3_ways` (new) | step | resctrl `llc_occupancy` and `mbm_local_bytes` per CLOS per domain — the keep ways' fill and the bytes each role pulled from DRAM, without touching the run |
| `cpu_clock` (new) | run | APERF/MPERF on a sample of pool cores, HSMP socket W |
| `pred` (new) | token × layer | the per-token probabilities and the labels, for the replay |

**Success, in order of the mechanism:**

| measure | today (2026-10-20) | target | what it isolates |
|---|---|---|---|
| planned lines served from L3 | 71.5 % (complete) | ≥ 90 % | delivery: the way split and the order |
| planned bytes touched before the run | ~69.5 % of items (uncapped, pfuse) | ≥ 95 % | the window: the earlier post, the budget's time bound |
| recall of host experts in the plan | .584 at cap 13 | toward .746 (the oracle at that cap); more where the budget grows | prediction |
| wasted planned experts a layer | ~2.77 | down, at the same recall | precision (each one also evicted a used one before the split) |
| run lines served from L3, all | ~47 % | up | the product of the three |
| experts lap a verify step | 25.26 ms | down | the outcome |
| ms an emitted token at 18k | 26.9 | down | the outcome |

The predictor's own test is tcdent's: the `prefetch_use` rows as a rolling rate against the route-ahead alone; it
settles within ~10,000 tokens or it does not.

## 10. Power

Touches and L3 service are not free on this box (metatron-cpu §4, 2026-10-20):
- **The touch is cheap.** A DRAM load loop costs ~2.3 W a core over idle and holds the clock. It runs in the card
  phase, when the pool otherwise rests in MWAITX; 48 keep cores touching cost ~55 W a socket (inference from the
  per-core figure), under no cap.
- **L3 service is not.** L2/L3-heavy reads hit a per-socket limit near 233 W, below the 280 W cap: 23 cores of L3
  loads on one socket hold 3.18 GHz, 46 hold 2.62. The L3 runs at its cores' clock, so an L3-served run gives some
  of its 2.2× back as clock (inferred; `bench-reuse`'s 2.2× was measured under the same limit, its clock not
  recorded). The two-queue run spreads L3 service across all CCDs of both sockets with DRAM streams beside it, which
  is the measured-best shape (12 + 11 split across sockets held 3.44 GHz where 23 on one socket fell to 3.18).
- **Every wasted planned expert costs twice:** its touch (power and window) and, without the split, an evicted used
  line. Precision is a power lever as well as a capacity one.
- **Prefill gets no touches** (§7): its gemm is already at the 280 W cap.

`cpu_clock` rows make the clock part of every phase's read; a phase that raises the L3 share and lowers the clock is
read on the experts lap, not on the share.

## 11. Build order

Each phase is exact (long SCORE bit-identical, the decode's tokens unchanged), measured at 18k on a quiet host, boot
against boot, against build 4da52d7 (experts lap 25.26 ms, 26.9 ms an emitted token).

| phase | what | the measurement that decides whether to continue |
|---|---|---|
| **0. The L3 as a programmable cache** (bench only; `bench-reuse` extended) | on one CCD and on all 12 at once: touch X MB from 4 cores, stream Y MB of fresh lines from the other 4, then read the touched set; orders (consumed-first or not); with and without the keep/stream CBMs (4/8/10/12 keep ways); whether an L3 hit leaves the line (read twice from different cores); hit and DRAM streams concurrently on one CCD; the touch as a vector load vs `prefetcht0`; clock and socket W throughout | the retention surface `keep(X, Y, ways)`; whether the split holds planned lines against Y ≥ 8 MB; whether concurrent L3 + DRAM sum on a CCD. If the split does not hold, phase 2 is dropped and the budget is the no-split knee |
| **1. The plan, the order, the early post** (host only, today's route-ahead scores as `P`) | the plan as one sequence a CCD with the per-CCD budget; the planned-first order in the run; the post before the join; the replay's calibrated `P` and θ | `prefetch_job` done share at the new post (window vs capacity), complete-L3 share, experts lap. Continue if the experts lap moves or the window is shown to bind (then the time bound is the budget) |
| **2. The way split and the two queues** | CLOS keep/stream fixed at boot; the planned and rest queues a CCD; the loader and main thread in stream | planned lines from L3 ≥ 90 %; experts lap; prefill's ms a prompt token unchanged (else the CBMs move) |
| **3. The card predictor** (082's design) | `W_PRED`, `pred` buffer, the update-then-predict launch; the low-rank term per §12's decision | rolling recall and waste against the route-ahead over ~10k tokens; experts lap. Never frozen; reverted only if it does not settle |
| **4. Prefill labels** (candidate) | route-ahead rows in prefill, the predictor trained on a prompt's positions | decode's recall from the first token after a long prompt, against phase 3's settling curve; prefill's ms a prompt token |

Phase 1 before phase 2 because phase 1 alone separates the window from capacity, and the answer sizes the keep ways.
Phase 3 can start in parallel with phase 2 on the cards' side: they meet only at `P`.

## Phase 0 results (measured 2026-10-20, `bench-l3`, metatron, beside the serving node)
Per CCD four keep cores touch X MB of fresh lines, the CCD's other cores read Y MB of fresh DRAM lines, then each keep
core reads its neighbour's planned slice; fill counters (prefetch hits + demand fills from L3, of all lines) around the
read. resctrl CAT: the keep cores' group fills ways 0..K, every other CPU ways K..16.

**The split protects the planned set.** One CCD, Y = 16 MB (2 runs each agree within a few points):

| X planned | no split | 12 keep ways (24 MB) |
|---|---|---|
| 4 MB | 89 % | 98.5 % |
| 8 MB | 62 % | 89 % |
| 12 MB | 46 % | 80 % |
| 16 MB | 33 % | 66 % |

At production's shape (X = 12 MB, Y = 8 MB), 14 rounds, 3 runs: all 12 CCDs at once **70–72.5 % → 81.7–82.2 %**; one
CCD 69.5–69.9 % → 78.8–79.8 %. A consistent +10–12 points. Even inside its ways the set is not all kept (16 MB in a
24 MB partition: ~65 %), so Zen 3's L3 replacement is not LRU; a smaller set relative to its ways keeps more.

**L3 and DRAM do not add on a CCD.** With the stream cores reading DRAM concurrently, the keep cores' planned read
falls from 17.6 to 9.3–10.0 GB/s a core (its L3 share unchanged at ~80 %), the stream at 6.2–6.5 a core: a CCD
delivers ~60–70 GB/s to its cores whichever source the bytes come from (as the 2026-10-15 probe's "~60 GB/s from L3 a
CCD" said). So §5's two-queues upper bound (a CCD's run 0.58 → 0.19 ms) does not hold; the floor for a decode run is
the CCD's delivery, ~0.29 ms for ~19 MB a CCD.

**The touch must be loads:** a `prefetcht0` touch leaves 8.6–9.2 % of the set in L3 (loads ~80 %).

**A hit leaves the line:** after the keep cores read the planned set, a reread by the stream cores finds 98 % in L3.

**Decision from phase 0:** phase 2's split is worth carrying (+10–12 points at production's shape, est. ~0.5–1 ms a
verify step from today's per-status costs; inferred); its two queues are not worth building as an overlap of L3 and
DRAM (they share the CCD's delivery) — what remains of them is ordering, so the planned items run while their lines
are still in the L3.

## Phase 1 in the node (measured 2026-10-20/21)
- **Ranking and a cap** (the step's predicted experts by votes then rank, `PREFETCH_CAP`): uncapped / 13 / 10 →
  experts lap 25.85 / 25.26 / 26.29 ms a verify step; 13 kept (commit 4da52d7).
- **The earlier post** (the `slot` lap 53 → 29 µs a layer: items from a precomputed queue order, a one-pass top-8):
  the verify step −1.2 ms through the main thread's critical path; the prefetch's completeness unchanged — the window is
  not what binds (commit 63303d2).
- **Gate/up before down** (every planned expert's gate/up slices first, then the down slices): complete / partial /
  missed 7.6 / 2.1 / 6.9 → 4.6 / 5.1 / 6.9 experts a layer, all host lines from L3 47.5 %, experts lap 25.58 ms — the
  same bytes finished, spread differently; reverted.
- The per-CCD byte budget is not built: with the cap binding near the L3's retention and the order neutral, its
  expected effect is within noise (inferred).

## Phase 2 in the node (measured 2026-10-20, build l3split against gemmv2)
The split at boot (`Threads::new`: the prefetch's 47 cores in a `keep` group with ways 0–11, every other CPU ways
12–15): completed experts' lines from L3 75.8 → 79.1 %, partial 74 → 80.2 %, all host lines 47 → 49.3 %; the experts
lap 25.5 → 25.41 ms a verify step, 26.6/27.0 → 26.7/27.2 ms an emitted token, prefill 1.85 → 1.84 ms a prompt token —
not a measurable gain, and short of the phase's gate (planned lines ≥ 90 %). The keep cores also run ordinary DRAM-bound
items, so their own fills land in the keep ways (the bench's keep cores only read the planned set: +10–12 points
there). Reverted; the resctrl mount stays in the image (tcdent: "resctrl stays around because it's valuable"). What
phases 0–2 show together: the L3 can be steered a few points at a time, but decode's time answers weakly at the margin
— capacity caps how many experts can be complete, and a CCD's shared ~65 GB/s delivery caps how fast even L3 bytes
arrive. The term that still moves time is which ~13 experts are prefetched: phase 3.

## Phase 3 (measured 2026-10-21, build pred2 against cap 13)
**Offline** (bringup/expert-cache/phase3: the causal simulation over the decode dump, 4,986 steps at 18k; settled
recall@13 of the next layer's picks): route-ahead alone .585; + bias .608; + co-routing from the picks here .621; + a
rank-64 map of the token's input .628; a full-rank map .632. Online beats a frozen fit; it settles in ~3k positions.

**What ships** (`models/glm-5.3-flash/src/predict.rs`): calibration, bias and co-routing. Updating the rank-64 map
dense (~14M parameters a step, ~330 MB of memory traffic) took 195 ms a step on one core against a ~25 ms step, for
+.007 recall. Only the co-routing rows the step picked have a gradient, so only they are updated. `bench-pred` replays
the dump through the Rust predictor: recall@8/13/16 .4226 / .6126 / .6917 against the simulation's .4231 / .6135 /
.6927. It costs, a layer pair, 6.0 µs to predict, 4.8 µs to rank (Π(1 + eᶻ), the same order as Σ softplus without a
logarithm) and 34 µs to learn.

**Placement:** the prediction runs on the main thread in the slot, from the scores `ffn::picks` already made. Learning
runs on core 120, the loader's SMT sibling, one job a layer through a channel. Learning on the main thread cost
50 µs a layer: the verify step went 56.3 → 59.7 ms (build pred1).

**In the node** (the deploy workload, the same 9,471 decode layer runs as the cap-13 runs):

| | cap 13 (a68fe017 / 2fce7061) | pred2 (b210fa5f) |
|---|---|---|
| Host experts a layer, complete / partial / missed | 7.57 / 2.13 / 6.91, 7.45 / 2.25 / 6.91 | 7.68 / 2.48 / 6.45 |
| Experts lap | 25.87 / 26.2 ms a verify step | 26.13 |
| `slot` lap | 1.31 / 1.35 ms a verify step | 1.67 |
| Decode, 96 tokens | 64.3–65.3 ms a step (four runs) | 63.5, 63.6 |
| Decode, 402 tokens | 63.4–63.9 ms a step | 62.4 (26.1 ms an emitted token, was 26.5–26.7) |

Long SCORE bit-identical. Missed experts fall by 7 %. The experts lap does not move measurably, and the step gains
~1 ms. Kept.

**What this means for phase 4:** the predictor's state lives as long as the node and is not reset per prompt. In the
402-token decode, its first 10 steps against the rest, missed experts a layer: cap 13 6.04 / 7.17, pred2 5.44 / 6.68.
A new prompt does not cost the predictor its fit. The case for prefill labels was therefore the settling after a
boot, but phase 4 measured more than that: training on each prompt lowered misses in every request.

## Phase 4 in the node (measured 2026-10-21, build pred4 against pred2)
Prefill trains the predictor. A chunk's `hc_pre` router takes the next layer's rows too (the matrix path, all
`ROUTE_ROWS`, the same launch). At its dispatch the main thread reads that half (`ffn_ahead`, `[n][E]`) and sends it
with the chunk's picks to the learner (`Lesson::Chunk`). The learner trains pair l − 1 once layer l's picks come
(`Predictor::train`): every prompt position is a label, in batches of `SPEC_TOKENS` the way decode's steps learn.

| | pred2 (b210fa5f) | pred4 (13421e6e) |
|---|---|---|
| Host experts a layer, complete / partial / missed | 7.68 / 2.48 / 6.45 | 7.90 / 2.62 / 6.10 |
| Missed in the boot's first decode (first 10 steps) | 6.58 | 6.17 (pred3) |
| Decode, 402 tokens | 62.4 ms a step, 26.1 an emitted token | 61.6, 25.7 |
| Decode, 96 tokens | 63.6, 26.5 | 62.9, 26.2 |
| Experts lap (a verify step) | 26.13 ms | 25.76 |
| Prefill | 1.84 ms a prompt token, cold 18k 36.5 s | 1.87, 37.0 s |

Long SCORE bit-identical. Every request misses fewer experts, not only the first one after boot. Decode gains
~0.35 ms an emitted token, and prefill pays 0.03 ms a prompt token. That cost is the main thread reading the next
layer's scores out of host memory: the prefill `slot` lap is 0.03 ms a prompt token, 0.04 when they were read and
copied twice (pred3). Queue 088 moves that read off the main thread. Kept.

## 12. Risks, unknowns, and decisions

**Unknowns (each has a phase that answers it):**
- Milan's L3 replacement and hit-exclusivity: phase 0.
- Whether CAT governs victim installs from L2 the way the design assumes (inferred from Milan's L3 being a victim
  cache with allocation enforcement; not measured on this box): phase 0.
- Whether L3-served and DRAM-served streams add on one CCD (the CCD's fabric port carries only the DRAM side; by
  inference they add): phase 0.
- Whether the window or the L3 binds today: phase 1.
- How much of the 2.2× survives the socket's L3-traffic clock limit in a real run: `cpu_clock` rows from phase 1 on.
- Whether a learned term can lift per-token router accuracy on the weak layers: phase 3; the replay found today's
  signal exhausted, so this is the predictor's whole bet.

**Risks:**
- The way split halves the L3 available to everything else; prefill's activations traffic is the one known
  tenant that may notice (phase 2 reads prefill).
- Placement re-weighting (`placement_from`) changes each CCD's bytes an expert; the plan tables rebuild at
  `relearn()`, as `slice_slot` does today.
- A wider verify step (081 at depth 3; any later depth) grows the union past any budget; the plan cuts it by `P`, and
  the residual is DRAM-bound as today. The oracle replay already shows the cap binding at 3–4 tokens.
- The predictor's launch delays the residents' jobs by its duration on the FFN card's queue (µs; they finish well
  inside the host's ~580 µs run).

**Decisions (tcdent, 2026-10-20):**
1. **No per-layer targeting.** The correction is uniform across layers; spending rank on the weak layers waits for
   data that shows a measurable gain.
   > "I don't see why ... we would add complexity by targeting specific layers, unless you see that there's a
   > measurable gain from that ... that is an intellectual application of premature optimization. So by all means,
   > make the case for that later when you collect more data, but that doesn't necessarily feel like the highest
   > priority to me."
2. **Which card: whichever runs the predictor earliest with the best data.** Not fixed to card0 or to the FFN card:
   the choice weighs how early the inputs exist against their quality, and the gaps each card has.
   > "card one, while the FFN is running, comes a little bit earlier and has certainly more time available, but card
   > zero after the FFN has time available as well. So it just comes down to what data we need and how much time we
   > need to be able to process it. Card one also has an HC post and a word wait or two ... gaps to fill. And so I
   > just want us to pick the best candidate in terms of being able to run this as early as possible with the data
   > that we have available, balancing that with the quality of the data that is then available at the time."
3. **resctrl stays in the appliance** ("I don't mind if resctrl stays around because it's valuable for us"); the
   split is carried if phase 0 shows it holds.
4. **Prefill labels: weigh accuracy; time is the only constraint.** A more capable model is taken wherever it fits
   the time available.
   > "the real only constraint that we have is time, and so if we can do a more intelligent model in the time
   > available, there's no reason not to do it."

The plan as a whole (tcdent, 2026-10-20): "I think that this is a pretty novel and significant improvement, and I see
no reason for us to not diligently research this and push through all of the steps that we have presented."
