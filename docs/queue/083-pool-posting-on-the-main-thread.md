# 083 — The pool's posting on the main thread

**Status:** Built (aff18–aff29, 2026-10-21/22) — the main thread's overhead outside the pool's runs 3.5 → ~1.0 ms a
verify step; the verify step 55.75 → 53.0 ms at 18k (same 96-token request), long SCORE bit-identical
**Area:** decode
**Gain:** 2.7 ms a verify step, measured

## Proposal
Every pool job's post runs on the main thread, and in decode the main thread's work sits on the step's critical
path one-for-one (2026-10-20: cutting the prefetch's `slot` lap 53 → 29 µs a layer shortened a verify step by 1.2 ms
while the prefetch itself was unchanged). The post's own costs, per decode verify step at 18k (build slotfast, `step`
rows): items sorted into the CCD queues 0.98 ms (a counting sort, but each item's slot through a `dyn` closure call),
the job's allocation plus — with `track-experts` — the prefetch's `prefetch_job` row emitted inside `post` 0.58 ms,
a run's slotting from entry to post 1.13 ms (including the sort), the previous job's drain 0.05 ms. Candidates: hand
`post` a precomputed slot array (the callers have it: `slice_slot`), reuse job allocations, emit rows off the posting
path.

## Source
`post_stats` / `overheads` in `Threads::post` (construct.rs); `step` rows' `post_sort_ms`, `post_free_ms`,
`run_slot_ms`; the slotfast measurement in 081's neighbourhood (commit 63303d2).

## What it was, and what was built (2026-10-22)
The overhead was not the instrumentation (the same 3.1 ms without `track-experts`, aff18lite) and only a third of it
allocation. In order of discovery, per CPU layer (medians):

1. **musl's malloc** (mallocng returns a freed medium block to the kernel and page-faults it back): the node's
   per-layer allocation set — two pool jobs, a run plan, a prefetch's items — cost 77 µs under it on the host against
   1.9 under glibc (`bench-alloc`). `common::alloc` replaces it: power-of-two classes 16 B–1 MiB, a magazine a
   class a thread, 2 MiB slabs mapped populated and never unmapped. The plan 46 → 33 µs, the prefetch's build 30 → 20.
2. **The run's rows on the main thread** (`item_rows`, 15 + 17 µs a layer): 153 slot accumulators × 8 buckets × 3
   atomics swapped, each line dirty in some worker's cache. Now each item writes its time and its L3 fills into its own
   words of the run's data, and the builder core writes the `expert_items`/`prefetch_use` rows and frees the plan
   between plans. `run.post` 20 → 0.1 µs.
3. **The plan's stores.** What remained of the plan (31 µs) was neither its loops (zeroed arrays, no `gitems`, u16
   indices: no change) nor cache coldness of what it reads (a second, warm plan of the same layer: 26.8 vs 27.9 µs). It
   was the ~25 KB it writes: every stage's time tracked its bytes at ~35 ns a line, into lines the 88 workers had
   read (LIFO reuse: an invalidation each) or long evicted (FIFO: a fetch each) alike. The builder now writes its next
   plan's blocks while it idles (`common::alloc::prewarm`): the plan 31 → 17.8 µs, `plan.wait` 20.6 → 8.3 µs a layer.
   The main thread does the same for the prefetch's blocks under the card wait (aff30).
4. Smaller: the plan requested under the dispatch's launches (`xpt_dispatch`'s `marked` callback, ~9 µs more lead), the
   prefetch's items cut once at `relearn` (`slice_runs`), prepared jobs taking their slots as bytes.

Per layer, what is left outside the two runs: `plan.wait` ~8 (the plan's 18 µs against ~10 of lead), the `slot` lap ~18
(the prefetch's build and post), `dispatch`/`route`/`act.read` ~9 each (card launches and reads), `plan_step` +
`ffn_post` 3.4.

## Next
The plan's two job builds (9.7 µs: two counting sorts at ~4 ns an item and two 4.5 KB jobs) and the prefetch's post are
the last main-thread writes of size; the CCD-0 contention of `route`/`dispatch`/`act.read` (§13 Rule 3) is structural.
