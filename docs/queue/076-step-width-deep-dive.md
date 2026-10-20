# 076 — The step's width under the wavefront: a deep dive

**Status:** Open — a study
**Area:** prefill, decode (the residency), card memory
**Gain:** up to prefill 2.84 → ~2.65 ms a prompt token at 18k (075's measurement) without 075's decode cost; more
if chunks wider than 1024 keep paying

> "I feel like we have, in the past … tried it and we haven't gotten it to land yet. And so I think it deserves a
> kind of intelligent scientific deep dive." — tcdent, 2026-10-19

## The question
`STEP` sets two things at once: the width of a wavefront chunk (`STEP / CHUNKS`), which sets how many tokens the
host's experts group by expert, and the depth of every step buffer on the cards and the host, which comes out of the
VRAM the residency fills. Prefill wants wide chunks. Decode wants the VRAM. Can the two be decoupled, and what is
the best chunk width once they are?

## What is measured
| when | change | prefill, ms a prompt token | experts | other |
|---|---|---|---|---|
| 2026-10-17 | `STEP` 128 / 256 / 512 (no wavefront) | 9.1 / 8.1 / 7.8 | 4.0 / 2.9 / 2.4 | 512 deployed |
| 2026-10-19 | `STEP` 512 → 1024 (no wavefront) | 5.16 → 4.82 | 2.12 → 1.93 | |
| 2026-10-19 | wavefront, chunks of 512 (037) | 4.04 → 2.84 | 1.30 → 1.60 | the cards' wait 1.95 → 0.35 |
| 2026-10-19 | `STEP` 2048, chunks of 1024 (075) | 2.84 → 2.65 | 1.60 → 1.40 | resident slots 19 → 17 a layer; a verify step ~+1 ms; draft lap 2.13 → 2.97; first steps after a cold prompt 63 → 76 ms |

No earlier `STEP` 2048 attempt is recorded in the queue or the searchable transcripts; if one was made, its result
is lost, which is a reason to keep this entry's record complete.

## Hypotheses to test, each separately
1. **Experts amortize with the chunk's width.** The host streams each expert's weights once a chunk; wider chunks
   read fewer bytes a token. Test: the experts lap and the pool's bytes a token (`track-experts`) at chunk widths
   512, 768, 1024, 1536 at a fixed buffer depth. Expected: falling, flattening where the light experts' bytes stop
   dominating (the 1024 → 2048 pre-wavefront point is unknown).
2. **The decode cost is the residency, not the step.** 075's ~1 ms a verify step comes from two fewer resident
   slots a layer. Test: `STEP` 2048 with the residency's slot count pinned to the 1024 build's (if VRAM allows by
   shrinking something else), or `STEP` 1024 with slots cut to 17: whichever moves decode by ~1 ms names the cause.
3. **The draft lap's growth (2.13 → 2.97) is a separate effect.** Nothing between steps visibly scales with `STEP`.
   Test: a verify step's laps on both builds side by side (`trace.sh`), then the draft's own zones; suspects are
   first-touch of larger host buffers, the cards' allocation layout and the resident slots the draft does not use.
4. **Most card step buffers need only a chunk's depth.** A buffer one chunk writes and consumes inside its own card
   half on one card (MLA attention's intermediates, the KDA projections' rows, the gate/up scratch) can be one chunk
   deep and shared by both chunks, since a card's queue is in order. A buffer that crosses phases or cards stays a
   step deep: the act phase's activations (the resident experts read them after the other chunk's card half), the
   streams, the handoff buffers, the expert outputs. Test: inventory every `Card`, `Peer` and `Shared` buffer by its
   writer, its last reader and the phases between, then size the chunk-local ones by chunk. Expected: most of the
   ~1 GB a card `STEP` 2048 added comes back to the residency.

## Method
- One variable a run; the 18k cold prompt, the 96-token and ≤512-token hits, the warm prompt and long SCORE through
  `deploy15.sh`; prow reads for every number; the hardware watch beside each (PCIe bursts on the c0:03.1 slot distort
  a cold prompt — rerun one that shows them).
- Chunk width becomes its own constant beside `STEP` for the study (`STEP` = the buffer depth, the chunk width ≤
  `STEP / CHUNKS`), so hypotheses 1 and 4 can move independently.
- Record each result here as it lands, including the ones that change nothing.

## What decides it
The served configuration is the one with the lowest verify step at long context; among those, the fastest prefill.
A gain that only trades decode for prefill goes on the shelf with its numbers, as 075 did.

## Source
037 and 075, 2026-10-19; tcdent's request for a deep dive the same day.

## History
- 2026-10-19: opened after 075 was shelved.
