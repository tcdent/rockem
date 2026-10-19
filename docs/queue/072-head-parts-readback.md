# 072 — The split head's readback

**Status:** Quantified, on the shelf — two 310 KB reads a token from the cards' logits buffers (GTT host memory, write-combining for the CPU)
**Area:** decode
**Gain:** est. ≤ 0.1 ms a step token; the head lap is 1.1 ms a step after 040

## Proposal
Each card's half of the logits lands in its `lg` buffer in host memory and the host copies it into the one
vector a token (two 310 KB reads of write-combining memory: slow for the CPU, see `Kfd::host_owned`). The cards
could write their halves into a host-owned buffer the sampler reads in place, or exchange an argmax for greedy
sampling as tinygrad does.

## Source
040's Next (2026-10-19); the GTT finding in 038.

## History
- 2026-10-19: given its own entry.

## Next
`lg` as a `host_owned` buffer (one allocation, both cards mapped), the sampler over it in place.
