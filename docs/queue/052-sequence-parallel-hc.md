# 052 — Sequence-parallel hyper-connections in prefill

**Status:** Deferred — small gain
**Area:** prefill
**Gain:** est. ~0.1 ms a prompt token at 512

## Proposal
Both cards run `hc_pre`/`hc_post` on every token of a step (the streams are duplicated). Split the
step's rows across the cards; gather each sublayer's normed input; reduce-scatter its partial sums.

## Source
tinygrad `6d7dee84e`, `e340cde11` (Megatron-style sequence parallelism).

## History
- 2026-10-18: listed.

## Next
Only with 037's scheduler, where the duplicated work competes for a busy card.
