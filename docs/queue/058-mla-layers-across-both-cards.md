# 058 — Indexer and attend of an MLA layer across both cards

**Status:** Open
**Area:** long context (decode and prefill)
**Gain:** est. up to half of the context-growing cost (~0.08 ms a prompt token per 1k positions today)

## Proposal
`card_of` puts each MLA layer on one card, so the other card idles through its indexer and attend.
Shard the indexer's pools across both cards (each scores half and selects its top candidates; a
fixed-order merge picks the 2,048) and the attend's heads across both cards. The cost that grows
with context then runs on two cards. An exact merge keeps `ix_select`'s set.

## Source
vLLM PR #61868: dense indexer rows sharded over the TP ranks and all-gathered (−7.6 % a step at
batch 32, enabled from 8 requests). Our inventory 043 (one card idle per MLA layer).

## History
- 2026-10-18: proposed.

## Next
After 019 (WMMA scoring); needs 001's peer writes or a host-memory gather for the candidates.
