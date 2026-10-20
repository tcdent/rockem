# 035 — Intra-expert activation sparsity

**Status:** Experiment (approximate) — reviewed 2026-10-20: no exact form saves bytes; held until accuracy trades open
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

- 2026-10-20: reviewed for an exact form (tcdent: "I'm not yet ready for us to incorporate tradeoffs in terms of
  accuracy yet"). Findings:
  - Which neurons are small is known only after gate (and up) are computed, so at most up and down are skipped:
    bytes saved ≤ 2/3 · sparsity, and only down (1/3) when the threshold needs up's value too.
  - Down is stored as `DOWN_ROWS` rows of width `D_FF`, 4-bit with a scale every 128 along `D_FF`: a neuron is a
    nibble in each of `H` rows. Skipping a neuron saves DRAM bytes only when all 128 neurons of a group are
    skipped; the expert runs are DRAM-bound (four bits is the floor, cpu-expert-bandwidth), so skipped
    arithmetic without skipped bytes gains nothing.
  - Neuron-bundled storage (an up row and its down column contiguous) makes skipping pay, but re-cuts down's
    quantization groups along `H`: a different 4-bit approximation of the checkpoint even at zero sparsity —
    an accuracy change before any threshold.
  - The one exact sparsity: a hidden value that the int8 quantize (`QAct`, scale amax/127) rounds to 0 adds exactly
    nothing in the down pass. Skipping it is bit-identical, but with today's layout it saves bytes only for a
    whole zero group, which a 128-wide group with any live value never is.
  - A verify step needs the union of its tokens' active neurons (two tokens: sparsity s → about s² if they are
    independent), and a prefill step's union is nearly dense: a decode-only lever that MTP already dilutes.
  - Ceiling today: the host's experts are 14.65 ms of a 43.5 ms verify step at 18k; 2/3 · s² of that at s = 0.6 is
    ~3.5 ms, approximate in every form that saves bytes.
  Verdict: no exact form saves bytes. Held as an approximate experiment; nothing to build while accuracy is fixed.

## Next
If accuracy trades open: a measurement only, changing nothing served — the per-layer distribution of |h| per
128-group on the 18k prompt (the int8-zero share, the share under candidate thresholds, the two-token union), from
the CPU forward. Then the threshold sweep on the oracle before any layout work.
