# 037 — Prefill chunks in flight as a wavefront

**Status:** Open
**Area:** prefill
**Gain:** est. 5.5 → ~3.3 ms a prompt token (the max of card and CPU instead of their sum)

## Proposal
Chunk B at layer l needs only chunk A's layer-l attention state, not A's experts:
```
cards:  A.attn(l)   B.attn(l)   A.attn(l+1)   B.attn(l+1) ...
CPU:                A.ffn(l)    B.ffn(l)      A.ffn(l+1)  ...
```
Exact (each chunk's arithmetic unchanged). Further: with layer ownership by card, several chunks
keep card 0, card 1 and the CPU busy at once (pipeline across layer groups).

## Source
Our analysis, 2026-10-16 and 2026-10-18; tcdent: "On prefill, we know exactly what's coming next,
so we should be able to push those further on top of each other."

## History
- 2026-10-16: noted as a knob ("pipelining two chunks"); not built.

## Next
042 first (who is busy when), then the scheduler in the prefill path.
