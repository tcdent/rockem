# 057 — Side queues for independent decode work

**Status:** Open
**Area:** decode
**Gain:** est. sub-millisecond to ~1 ms an emitted token; vLLM measured −23 to −103 µs per moved piece of a 5.4 ms step

## Proposal
Each card has one AQL queue (`Kfd::open` creates one). Work that is independent of the attention
chain until a later reader — the next layer's hyper-connection coefficients, the indexer's query
and pool-key work, table builds — can run on a second queue and join at its first reader (a
barrier-AND or signal wait), overlapping the chain instead of lengthening it.

## Source
vLLM PR #61868 (DeepSeek-V4.1-Flash decode on GB300, 2026-10-15): side streams for the next
layer's mHC coefficients (up to 48 rows), the indexer query, compressor and index-key work,
candidate expansion, the per-group tables and DSpark's aux hidden-state means — each joining at
its first reader. Our launch-fusion notes left "a second queue" open (2026-09-19).

## History
- 2026-10-18: proposed from #61868.

## Next
List the decode step's independent pieces from a timeline (what each launch reads and when its
output is first read), then a second queue on one card for the largest.
