# 035 — Intra-expert activation sparsity

**Status:** Experiment (approximate) — not evaluated
**Area:** decode
**Gain:** est. ~4–6 ms at 60–70 % sparsity, before 003 (bytes ∝ 2/3 · sparsity)

## Proposal
Mask SwiGLU neurons under a calibrated magnitude threshold; read only the active neurons' up rows
and down columns. Saves DRAM bytes only with neuron-bundled storage (an up row and its down column
contiguous), which re-cuts the down projection's quantization groups.

## Source
arXiv 2605.08575 (training-free, tolerance rises with model size); LLM-in-a-flash (bundling); TEAL, CATS.

## History
- 2026-10-17: listed.

## Next
A threshold sweep on the oracle before any layout work.
