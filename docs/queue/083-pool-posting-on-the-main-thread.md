# 083 — The pool's posting on the main thread

**Status:** Open — measured, not built
**Area:** decode
**Gain:** est. a few tenths of a ms to ~1 ms a verify step (of ~2.7 ms measured overhead), exact

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

## Next
Read which of the three is largest per post kind (runs vs prefetches), then the slot array first.
