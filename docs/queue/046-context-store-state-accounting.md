# 046 — The context store charges kept states to its budget

**Status:** Resolved
**Area:** serving
**Gain:** correctness — the overnight OOM

## Proposal
The store's budget counted positions only; a kept sequence state (~153 MB) at each prompt's end,
before EOS and at a context's close sat outside it.

## Source
The 2026-10-18 00:31 kernel panic during GSM8K ("Out of memory and no killable processes";
913 transcripts × kept states).

## History
- 2026-10-18 05:46 (`1ce8187`): a kept state costs `state_cost` positions (~4,600); `trim` evicts
  LRU contexts on `grew`, `mark` and `close`; a transcript keeps its newest 8 marks. The eval
  client had no timeout and hung five hours on the dead connection.

## Next
A timeout in the eval client.
