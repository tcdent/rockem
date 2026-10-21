# 089 — The MLA output projection in a verify step

**Status:** Open — measured, not analysed
**Area:** decode
**Gain:** unknown; up to the ~1.6 ms a verify step `mla_o` spends on the critical path

## Proposal
tcdent (2026-10-21), from the card lanes in prow: in the verify step, `mla_o` is the one launch that adds latency to
the cadence the cards keep with the host's experts. It fires only on the MLA layers (a few layers apart, not every
layer), and when it fires it holds the progression: the MLA card's output projection must finish before `hc_post`
hands the block's output to the other card and before the FFN half can start.

Measured (track-launches, run 2026-10-21 card10): `mla_o` (the few-token rows kernel, `rows8` over `O_IN` =
MLA_HEADS × V_HEAD_DIM inputs into H rows, FP8) p50 141.8 µs, p90 144 µs a launch; ~11 launches a verify step across
the two cards (5 on card0, 6 on card1 in one step's window), ~1.6 ms. For scale: `mla_qb` 80 µs, `mla_uv` 35 µs.
The prefill twin `mla_o_w` is 4.55 ms a 512-token chunk.

## Source
tcdent's observation in prow (card lanes); bringup/reads/cards.sh <run> verify; the zones table, kernel `mla_o`.

## Next
Read `mla_o`'s ELF report and its bytes against the card's measured rows rate (are its FP8 bytes streamed at the
rate `ffn_gu`'s are, or is the launch underfilled: how many workgroups, how much of the card it occupies); whether
its work can split across both cards as the attend already is (058), or start on the heads that are done.
