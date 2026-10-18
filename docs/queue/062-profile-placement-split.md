# 062 — Separate placement training from the profile

**Status:** Open — a refactor candidate, not scheduled
**Area:** decode, prefill (the GLM CPU model and the node's placement retrain)
**Gain:** structural: a zero-category build that does no timing beyond what placement needs, and
one consumer per counter

## Proposal
`Profile` (`models/glm-5.3-flash/src/construct.rs`) carries both reporting and a control input:
- **Training reads a reporting struct.** `Glm::take_training` (`node/src/glm.rs`) takes
  `node_items`, `node_end` and `tokens` out of the profile for `retrain`, the placement trained on
  traffic. Placement needs these in every build.
- **Two consumers, two reset schedules.** The report resets the whole profile every window, so
  `take_training` must run first. `train` and `train_seen` exist only for that ordering.
- **Destructive drains.** `account()` drains the pool's counters (`node_ends`, `overheads`,
  `post_stats`, `last_ends`) with `mem::replace`. Only one reader can see them.
- **Control flow on log text.** `place()` returns a sentence, and `retrain` stops further rounds
  on `r.contains("no room")`.
- **Reporting-only timing in every build.** The `lap()`s, the pool's `entered`/`sorted`/`drained`
  stamps and `freq` (coverage) run whether or not anything reads them.

The shape:
1. Running totals in the pool, never reset. Each consumer keeps its own last reading and takes
   the difference. Placement reads `node_ends()` at retrain time. `train`, `train_seen`,
   `take_training` and the profile's `node_items`/`node_end` go away. The run-end timing stays in
   every build.
2. `Profile` per step and entirely under the `track-step` category: the laps, `account()`'s
   statistics, the pool's overhead stamps and `freq`. The model crate gets a `track-step` feature,
   forwarded from `node`.
3. `place()` returns a value: moved items, bytes and seconds, or no room. The `placement` row and
   `train_stopped` read the value.

Optional, from the same split: the picks as a `track-routing` site (`pos, layer, e0..e7`), which
would replace the coverage summary and `forward-glm --routing-trace`'s file, with
`routing_replay.py` reading DuckDB.

## Source
Found while planning the prow migration, 2026-10-18. tcdent: a strong candidate, not to be done
during the migration.

## History
- 2026-10-18: recorded. The prow migration leaves the structure as it is: the report's `println!`
  becomes rows, placement keeps reading the profile.

## Next
Pick it up after the prow migration, when the `step` rows are in use.
