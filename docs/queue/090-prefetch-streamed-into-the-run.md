# 090 — The prefetch streamed into the run

**Status:** Open — the premise measured on aff83 (2026-10-25, below); the design now a re-targetable stream
**Area:** decode
**Gain:** unquantified. Measured premises: 17.8 % of a layer's posted items are not done when its run overtakes the
job (`prefetch_use`: 429.5 posted, 352.9 done; 454k layer runs, run c3e349); a partial expert's lines hit L3 62.4 %
against a complete one's 85.7 % and an unposted one's 5.9 %; ~2.6 of 13 posted experts never run

## Proposal
tcdent: the decode prefetch is an ordered problem — the order in which bytes arrive matters, and it should be as
sequential as the run that uses them. Today the prefetch is posted at the slot (~600 µs before the layer's run), and
the run, when it starts, overtakes it: whatever is not done is abandoned. Of a layer's 13 posted experts (~156
items), ~76 items are done by then (card13, 2026-10-21); a posted, used expert gets ~72–75 % of its lines from L3
whether complete or partial, an unposted one ~7 %.

The direction: intentionally overlap the prefetch with the run, so the prefetch's span covers the run too and each item
arrives just ahead of the core that will read it — the prefetch a stream timed to the run's consumption order, not a
batch that must finish before it. Prerequisite: arrival order equal to consumption order (the run taking its experts
in the plan's order, built 2026-10-21), so "ahead" is well defined per CCD queue.

## Measured on aff83 (2026-10-25, run c3e349: 907k prefetch jobs, 454k decode layer runs)

`tcdent, 2026-10-25: the core worry — we are potentially hurting ourselves by continuing to prefetch experts that we
won't utilize while we're trying to stream in experts that we're not predicted.` The static job does both: it keeps
touching mispredictions to the end of the window, and it stops — abandoning its tail — the moment the run overtakes it,
which is when streaming would begin to pay.

**The windows are two jobs a layer, not one bimodal job.** Done-at-overtaking by window (the original reading):

| window | jobs | share | done at overtaking | item-start p50 |
|---|---|---|---|---|
| ~59 µs | 314k | 31 % | **22 %** | 10.6 µs |
| ~142 µs | 31k | 3 % | 35 % | 13.9 |
| ~402 µs | 443k | 46 % | 84 % | 53.5 |
| ≥778 µs | 110k | 12 % | **100 %** | ~50 |

Delivery is near-constant-rate across window lengths (~460–630 items/ms, done ÷ window); the run starts whenever it
starts. A third of jobs get windows where the prefetch is mostly still in flight at the run. The same rows by job
(rockem.lead's read, 2026-10-25; a job is ~430 items: 13 experts × 33):

| window | jobs | items | done | cores | core-time ÷ window |
|---|---|---|---|---|---|
| < 100 µs (59) | 314k | 429 | 94 | 88 | 212 % |
| 100–250 µs (129) | 29k | 423 | 140 | 88 | 139 % |
| 250–700 µs (404) | 455k | 358 | 301 | 88 | 86 % |
| ≥ 700 µs (790) | 110k | 138 | 138 | 79 | 20 % |

There are 907k jobs for 454k layer runs: each layer posts twice. The ~59 µs jobs are the early post, superseded by
the rows plan's amend (093) ~100 µs after the slot — the job after one reposts ~338 items (its 335 untaken) and runs
~400 µs to the run; after a ≥ 700 µs job the next is a fresh 430-item post. So the short third is not abandoned
work, it is reposted, and its core-time above 100 % of the window is the taken items' takers working on into the
amended job. The 42.8 % "abandoned at overtaking" counted those reposted items: what is not done when the run starts
is 17.8 % of the posted items (`prefetch_use`), consistent with the amended jobs' 56.7 of 358 (16 %). (Corrected
2026-10-25 by rockem.lead, pairing each job with the next; `overtaken_by` is empty in every row, so the read is by
time.) The long tenth's cores are idle 80 % of a window that holds its whole post.

**The not-done tail is mostly never read.** A not-done item is either untouched or in flight at the overtake. At most
one item a core is in flight (88 cores against ~57 not-done items of an amended job), and the touch reads the abandon
flag every 4 KB (`prefetch_with`), so an in-flight item stops within 4 KB of the overtake. An untouched item costs no
DRAM read; the run reads it once, as it would unposted. The double read is bounded by the in-flight items' touched
part — not the tail's size. (Corrected 2026-10-25: this paragraph called the whole 42.8 % a double read, from ~150
items × 272 KB a job; a job is ~430 items, and that 42.8 % does not match these rows' 17.8 %.)

**The L3 gradient proves arrival-before-use is the knob.** Per decode layer run (15.14 host experts): complete 7.26 /
partial 3.15 / missed 4.73, with L3 hit rates **85.7 % / 62.4 % / 5.9 %**. Partial experts deliver 10× the L3 of
unposted ones; posted experts' aggregate L3 share is ~79 % against the complete experts' 85.7 %. What the gradient does not say: a line fetched
*during* the run is a DRAM read during the run — the same bytes at the same bandwidth. If the `experts` lap is at the
box's DRAM ceiling (074), streaming the tail into the run frees nothing; L3 pays only for lines fetched while DRAM is
otherwise idle (before the run), or where the run is latency-bound rather than bandwidth-bound on its CCD — which is
what phase 0 measures. (Corrected 2026-10-25: an estimate of ~0.5–1 ms a verify step-token stood here, resting on
"every L3-served line is DRAM bandwidth freed on the critical path"; it had no derivation and does not hold for lines
streamed inside a DRAM-bound run.)

**The misprediction tax is at most ~20 % of posted bandwidth.** 13 posted, 10.41 run on the host: ~2.6 posted experts a
layer never run, 20 % of the posted experts. The bandwidth they cost is the part of them touched before the overtake —
all of it in the long windows, little in the short ones — so ~20 % is the bound, not the measure; their lines also
evict the lines of experts that do run (082's note). The static job never cuts them: `prefetch_amend` keeps any expert with a taken item, and
no correction runs after the rows pass.

## The plans measured (aff85, `plan_eval`, 2026-10-25: 50,225 decode layer runs of tcdent's 360k session)
`bringup/reads/plan_eval.py`. P(picked) by rank, early plan / rows plan: rank 1 .999 / .999, 4 .976 / .988, 8 .852 /
.907, 10 .755 / .826, 12 .653 / .722, 13 .603 / .672, 14 .554 / .619, 16 .466 / .512, 20 .329 / .352, 24 .237 /
.237. The 13 posted (non-resident, best first): .798 / .842 picked (10.37 / 10.95 of 13 a layer), covering .603 /
.639 of the 17.2 non-resident picks a layer. Calibration of 1 − 1/key: the rows plan is close (predicted .35 → .37
observed, .65 → .62, .85 → .80, .99 → .97); the early plan is overconfident from ~.3 up (.65 → .49, .85 → .66). So a
probability cut is meaningful on the rows plan today and not on the early one; and the marginal posts (ranks 12–13,
~.65–.72 on the rows plan) are still mostly hits — a cut there pays only if a miss's L3 damage exceeds ~2 hits.

## The cap sweep (lab090, 2026-10-25): the window, not the cap, is the limit
A lab node cycled `PREFETCH_CAP` per decode step (11 / 13 / 15 / 17), the step and use rows carrying it; eight 107-token
decodes at 18k, 120 verify steps a cap (`bringup/reads/cap_sweep.sh`):

| cap | experts, ms a token | total, ms a token | complete | partial | missed | complete L3 | partial L3 | all L3 |
|---|---|---|---|---|---|---|---|---|
| 11 | 7.67 | 22.31 | 7.53 | 2.52 | 4.75 | 86.5 % | 73.0 % | 56.1 % |
| 13 | 7.56 | 21.74 | 7.52 | 3.78 | 3.93 | 84.3 % | 58.4 % | 56.5 % |
| 15 | 7.70 | 21.98 | 7.73 | 4.54 | 3.16 | 82.7 % | 47.7 % | 56.1 % |
| 17 | 7.59 | 21.97 | 7.89 | 5.07 | 2.28 | 81.1 % | 42.7 % | 56.9 % |

The prefetch completes ~7.5–7.9 experts a layer whatever it posts — even at 11 it leaves 2.5 partial: it is bound by
its window (the time from the amend to the run, at the delivery rate), not by its cap. A larger cap turns misses into
partials and dilutes every expert's L3 share; the share of all the run's lines served from L3 stays at 56–57 %, and the
`experts` lap does not move (within ±0.1 ms a token). So trimming low-confidence posts saves nothing (the window never
reached them in full), and posting more gains nothing. What moves the L3-served share is more window — the plan
earlier, as good as the rows plan (098) — or more delivery inside it. `PREFETCH_CAP` stays 13.

## The run's order (bench-order, 2026-10-25): interleave the cold experts with the warm
Each of socket 1's 46 pool cores gets 9 warm items it touches just before the run (1.25 MB, as the prefetch does) and 7
cold items from a pool past the caches (0.97 MB, as the experts nobody prefetched) — a verify step's layer at 18k per
core; all cores start together; the production kernel at 3 tokens (`bringup/bench/src/order.rs`), p50 of 200
iterations: warm first (today's: the plan's order runs the prefetched experts first) **491 µs**; cold first 442;
**interleaved 369**; warm only 118; cold only 327. Warm-first is slower than warm-only plus cold-only: the bus idles
while the cores compute on L3 lines, then the cold tail is bus-bound with nothing to overlap. Interleaving keeps the
bus streaming throughout and lands within ~40 µs of the cold set alone. Applied to the served run (~388 µs a layer),
the same ratio is ~290 µs: ~4 ms a verify step, ~1.5 ms a token — the largest lever measured in this item, and a
change to the run's order, not to the prefetch. To build: `prepare_batch_act`/`prepare_slotted` order each core's
queue so its missed (unposted or not-done) experts' items interleave with its prefetched ones.

## Design: a re-targetable stream with a strength-ordered budget

The job stops being a batch with a deadline and becomes a stream with three correction points, each with better
information than the last:

1. **Post (the slot): the confidence-ordered, window-budgeted fill.** The plan's rank value is a ranking score —
   `pred_rows` ranks the union by Π(1 + eᶻ), its sigmoid gates calibrated online (`M_CAL`/`V_CAL`, `plan_learn` from
   realized outcomes). Each token's sigmoid is trained by BCE to be its P(picked), and the union's key Π(1 + eᶻ) is
   exactly 1 / P(no token picks it) if the tokens' picks are independent — so 1 − 1/key is the score's P(picked) by
   design; whether it is calibrated in fact is what `plan_eval` measures (2026-10-25, aff85). The fill stops when the marginal value goes
   negative: `P(e) × bytes(e) × L3-benefit(e)` against the touch cost (~70 µs a core a 272 KB slice) and the pollution
   cost (091's retention curve: 86 % of lines kept at ≤ 10 experts, 56 % at 23+ — past the knee a prefetched expert
   costs a DRAM read and evicts an earlier one). The window is estimated, not assumed: the running quantile of the
   slot→run gap (the amended jobs' ~400 µs above is the prior), and 090's keep-open extends the budget through the run.
2. **Dispatch start (exists, 093): the rows-pass re-rank.** `prefetch_amend`'s merge — taken items keep their takers,
   the tail re-ranked by the plan with the whole FFN output as the margin's input.
3. **Dispatch, after `ffn_input` (new): the truth-cut.** The real router picks are in (`b_picks`), exact and
   weight-ordered. The job re-targets to (real picks − card residents) — kept experts keep their taken items, **cut
   experts stop their takers immediately** (a skip word on the remaining items, or the amend's retire-and-succeedor).
   The cut is what makes the keep-open safe: without it, a misprediction streams through the run's whole window
   competing with it; with it, the stream is exactly the experts the run will consume, in its order.

The prerequisite stands (built 2026-10-21): the run consumes in the plan's order (`prepare_batch_act`/`prepare_slotted`),
so "ahead" is well defined per CCD queue, and the touch order is the consumption order (`prefetch_with`'s items in
`slice_order`). `settle_pending`'s overtaking close becomes a consumption close: the job retires when its items are
taken or the layer's run ends.

**What each stage needs that does not exist yet:** the per-rank calibration curve (P(picked | rank) from `plan_learn`'s
score-and-outcome pairs — one prow row, no new hooks); the window estimate at post time (the running quantile, against
the table above); the pollution term as a table the stop rule reads (091's retention curve, re-measured per box).

**Stage one is not useful alone.** The real picks arrive with `ffn_input` (the FFN card's `hc_pre`), which the host waits
for immediately before its run; the job is overtaken — stopped within 4 KB — when that run starts. A cut at `ffn_input`
has no window left to save bandwidth in. It is the keep-open's companion: once the job runs on through the run, the cut
is what keeps mispredictions out of it. Both are gated on the phase-0 question below — the run's rate while its CCD
streams. (Corrected 2026-10-25: this paragraph said a cut at dispatch would move a misprediction's bandwidth to the real
experts "before the run arrives"; there is no time between the two.)

## Source
tcdent (2026-10-21); bringup/reads/prefetch.sh (done share, window) and prefetch_use.sh (status, L3 shares).

## Next
0. Done 2026-10-25: the cap sweep (above) — the window binds, not the cap; the per-rank curve is measured (`plan_eval`).
   Next: a longer window (098's earlier plan) or more delivery inside it.
1. The truth-cut with the keep-open (stages one and two together, after phase 0): re-target the open job at
   `ffn_input` to the real picks − residents, cut experts' takers stopped; measured by `prefetch_use`'s
   complete/partial/missed and the `experts` lap against aff83's 7.26 / 3.15 / 4.73 baseline.
2. The calibration curve: a prow row per learning pair (score band → picked), the P(picked | rank) curve the fill
   rule reads; validated against recall@13 .647.
3. Phase 0, per CCD: how far ahead of a run's item its prefetch must be (L3 fill latency at the CCD's delivery rate,
   ~60–70 GB/s shared with the run's own DRAM reads); the run's own rate while its CCD streams (L3 and DRAM share the
   CCD's delivery).
4. The keep-open (stage two): the job's span through the run, items arriving just ahead of their takers, the partials
   converting toward the complete experts' 85.7 % — worth something only where phase 0 finds the run not bound by its
   CCD's DRAM bandwidth.

## History
- 2026-10-21: tcdent's direction; the done-share bound (48.6 %, card13).
- 2026-10-25: the premise measured on aff83 (the tables above): 42.8 % abandoned, the L3 gradient
  85.7/62.4/5.9, the misprediction tax ~20 % of posted bandwidth, the double DRAM read named; the design revised to
  a re-targetable stream — confidence-ordered fill, rows-pass re-rank, truth-cut at dispatch, keep-open through the
  run gated on phase 0.
- 2026-10-25 (rockem.lead's review): corrected against the code and the same run's rows — two jobs a layer (the early
  post and the amend), so the short windows are superseded posts and the 42.8 % double-counts reposted items (17.8 %
  not done at the run); a job is ~430 items, not ~150;
  the not-done tail is mostly untouched (at most an item a core in flight, stopped within 4 KB), so not a double read;
  lines streamed inside a DRAM-bound run free no bandwidth, so the ~0.5–1 ms estimate is withdrawn; the misprediction
  tax is a bound; the score's calibration is unmeasured (it is a probability by design, corrected the same day); the truth-cut has no window without the keep-open. Added: the
  job table and the cap sweep as the first measurement.

- 2026-10-25 (lab090): the cap sweep — 11 to 17 posts complete the same ~7.5–7.9 experts, the L3-served share flat at
  56–57 %, the `experts` lap unchanged: the window is the limit.
