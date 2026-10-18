# 039 — Streaming experts through the cards for large prefill steps

**Status:** Tentative (future)
**Area:** prefill, long context
**Gain:** est. at 4k-token steps ~0.8 ms a prompt token for all experts, ~0.2 at 16k

## Proposal
A per-layer staging region on the cards, double-buffered (~2 × 1.3 GB): while layer l runs, the
copy engines pull layer l+1's heaviest non-resident experts (known a layer early from 005), and the
cards run them with 038's kernel; decode's residency is untouched. All 162 GB over ~50 GB/s is
~3.2 s a step, amortized by the step size. Required for long prompts to hold the hardware ratio
against all-VRAM setups.

## Source
Our analysis, 2026-10-18; llama.cpp's offload of host weights for large batches (from memory).

## History
- 2026-10-18: tcdent: "keep this idea of sliding experts in and out of the GPU as something
  tentative for the future."

## Next
After 038 and 045; larger steps (`STEP`) with it.
