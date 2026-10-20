# 037 — Prefill chunks in flight as a wavefront

**Status:** Built (2026-10-19): prefill at 18k 4.04 → 2.84 ms a prompt token, the cold 18k prompt 76.2 → 54.9 s, long SCORE bit-identical
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

## Built (2026-10-19, runs 17c5dbe3 / b9ee50d2)
042's question answered from the served laps before building (a track-launches boot runs prefill ~5× slow, the
bring-up checks on every token, so its card lanes are not prefill's): the host's laps are in sequence, so the cards'
attention (`kda_card` 1.95 ms a prompt token, the host waiting) and the host's FFN (route 0.30 + experts 1.30) were
summed.

A step of 64 or more tokens runs as `CHUNKS` = 2 chunks (`glm_5_3_flash::chunk_of`: whole launch passes, the last the
remainder), each a run of the step's rows — no buffer is duplicated: every token-indexed launch address is taken from
the chunk's first token (`Cards::tok`), each chunk has its own set of handoff words and (attention, FFN) values, and
`hc_pre`'s act phase writes its table's rows from the chunk's first token (`t0`) with its own word. The host's step
runs a chunk's layer as a card half (the attention and the FFN's card launches, posted) and a host half (the chunk's
experts, sum and handoff) in the order A's card half of l, B's host half of l − 1, B's card half of l, A's host half
of l. Each card's queue is in order, so B's attention follows A's on the KDA state and the latents, and nothing in a
queue spins on host work the host has not already done when the card reaches it. The residency advances and claims
once a step (the first chunk's dispatch, the last chunk's join). After the layers the head and the draft layer address
the whole step again (the first build left chunk B's offset set, and the draft layer's projections over the step's
1023 positions read past the buffers from token 511: page faults on card 1 and a hang in `mla_qb_w`).

| 18k prompt, ms a prompt token | before (fold5) | wavefront |
|---|---|---|
| total | 4.04 | 3.18 |
| cards' wait (`kda_card`) | 1.95 | 0.07 |
| experts | 1.30 | 1.52 |
| residents' join | 0.00 | 0.65 |
| route | 0.30 | 0.37 |
| MLA rows (`mla_proj`) | 0.28 | 0.33 |
| cold 18k prompt | 76.2 s | 60.9 s |

Decode is one chunk and unchanged (a verify step 48.2–49.1 ms by the request's wall). The host is now the critical
path: the residents' join (their launches queued behind the other chunk's attention) and the experts, which stream
each expert's weights once a chunk instead of once a step (the 512 → 1024 step sweep moved experts 2.12 → 1.93).

## The residents dispatched ahead (2026-10-19, run 3a5cbb73)
A chunk's host half is two phases: the dispatch (its FFN input waited for, its residents launched) and the run (the
FFN's output half posted, the host's experts, the join, the sum, the handoff), in the order A card, B run, A dispatch,
B card, A run, B dispatch — a chunk's residents ahead of the other chunk's attention in the cards' queues. The
residents' fence takes a completion line of its own (`Kfd::fence_line`): on the queue's one line, A's residents'
fence was still outstanding when B's MLA projections posted theirs, the signal was armed twice and the counts crossed
(the host's wait never ended; the first boot of this order hung on card 1).

| 18k prompt, ms a prompt token | wavefront | residents ahead |
|---|---|---|
| total | 3.18 | 2.84 |
| residents' join | 0.65 | 0.00 |
| the FFN input's wait (`kda_card`) | 0.07 | 0.35 |
| experts | 1.52 | 1.60 |
| cold 18k prompt | 60.9 s | 54.9 s |

## Next
1. ~~`STEP` 2048~~ — measured: prefill 2.84 → 2.65 against two resident slots a layer and ~+1 ms a verify step;
   its own entry, 075, on the shelf.
2. ~~The MLA layers' bubble by reordering~~ — measured and reverted (2026-10-20, below); the remaining route is the
   MLA rows finished on the card (no host round trip between a chunk's projections and its attend).

## The MLA bubble by reordering (2026-10-20, runs fba2fed9, 40fc015f against ed18fd97)
A decode verify step's trace (lab run 6615b39f) shows the host's own work on an MLA layer's rows is ~20 µs (the
card's projections end at 3087 µs, the host's next lap at 3109): `mla_proj`'s 5.2 ms a verify step is the host
waiting on the card. In prefill the cost is the pool's idle while the host waits on a chunk's projections (the
lap, 0.34 ms a prompt token). Tried: each chunk's card half split at that wait (posting; the rows, selection, attend
and the FFN's launches), the other chunk's experts run between, the output half posted after the other chunk's rest.

| 18k prefill, ms a prompt token | the plain wavefront | split at every layer | split at MLA layers only |
|---|---|---|---|
| total | 2.77 | 3.35–3.61 | 2.91–3.00 |
| the FFN input's wait (`kda_card`) | 0.35 | 1.32–1.37 | 1.02–1.06 |
| `mla_proj` | 0.34 | 0.18 | 0.18 |
| cold 18k prompt | 53.2 s | 64.0 s | 56.7 s |

Long SCORE bit-identical throughout; decode unchanged. Splitting every layer held a KDA layer's FFN launches behind
the other chunk's experts (the card finished the attention and idled). At MLA layers only, the card idles after a
chunk's projections through the other chunk's experts, which cost more than the pool's idle it removed: the wait
moved from the pool to the card. Reverted.
