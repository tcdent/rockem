# 038 — Resident experts in prefill

**Status:** Open
**Area:** prefill
**Gain:** est. experts 2.2 → ~1.2 ms a prompt token

## Proposal
`step_batch` dispatches to the resident experts only for steps of up to `SPEC_TOKENS`
(`construct.rs:1568`); a prefill step runs every pick on the CPU while the cards idle. Send the
resident experts' tokens (~14 a step an expert at 512) to the cards in a grouped multi-token iu8
WMMA kernel. Bit-identical: the format's integer group sums are exact in any order; keep the
CPU's f32 flush order.

## Source
tinygrad's grouped int8 experts (`03d069997`, eight waves sharing a tile's activations through LDS).

## History
- 2026-10-18: proposed.

## Next
011 (the iu8 rate), the kernel, then the dispatch for steps past `SPEC_TOKENS`. Check gfx1100's iu8
WMMA signedness flags for unsigned nibbles against signed activations.
