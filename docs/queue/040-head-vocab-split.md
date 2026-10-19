# 040 — Split the head's vocabulary across both cards

**Status:** Resolved (built and measured 2026-10-19)
**Area:** decode
**Gain:** measured — the head lap 2.08 → 1.08 ms a step and the draft 3.7 → 3.0 at 18k (−1.7 ms a verify step, ~−0.9 ms an emitted token); exact (long SCORE bit-identical)

## Proposal
`head` (1.1 ms a step token) and `draft` (1.8) both stream the 1.26 GB lm_head on one card. Split
the vocabulary across the cards; merge the argmax / sampling across the halves.

## Source
Our decode profile, 2026-10-18; tinygrad's argmax per vocab shard exchanged through host memory.

## History
- 2026-10-17: the draft's sync trims measured nothing — the draft lap's floor is the head's 1.26 GB.

## History
- 2026-10-19 00:07 (run of `441f038-dirty`): built — every card holds the final norm and its rows of lm_head
  (`head_rows`: the vocabulary in parts of whole WMMA row groups, 77,440 rows a card), `logits` runs the
  streams' mean, the norm and the part on every card at once (the streams are on every card already) and
  reads the parts back a token into the one vector; `mtp_head` the same over the draft's hidden state, which
  `mtp_ffn` had already put in host memory. Head 1.95 → 1.04 ms a step. The draft read 5.5 ms instead of
  3.7: the other card's rows kernel read the activation from host memory, a workgroup at a time over the BAR.
- 2026-10-19 00:23: the activation staged into that card's head-input buffer first (one copy launch); draft
  3.03 ms a step, head 1.14. Zero PCIe records, long SCORE bit-identical, the same completions. Run 6e1824f5.

## Next
The sampler sees the one logits vector, so nothing changed there; the parts' readback (two 310 KB reads a
token) is the remaining cost — a card-side argmax exchange would remove it only for greedy sampling.
