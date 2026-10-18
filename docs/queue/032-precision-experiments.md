# 032 — Attention-state precision below the reference

**Status:** Experiment (off-reference)
**Area:** long context
**Gain:** fp8 latents/pool keys halve the bytes again; bf16 KDA state halves its traffic (~0.1 ms)

## Proposal
Deviations from 030, each gated as an experiment (KL, top-1, selection overlap, evals):
- fp8 latents (DeepSeek-V3.2's deployment; vLLM on SM100);
- fp8 pool keys after a Hadamard rotation (vLLM `glm5next/nvidia/sparse_indexer.py`, SGLang `dsa_indexer_kpool.py`);
- a bf16 KDA state (vLLM #59147: neutral on this model; FlashKDA: no measurable loss in prefill chunks).

## Source
The 2026-10-17 precision research (`/tmp/rs_dt`).

## History
- 2026-10-17: listed when 030 was decided.

## Next
Only when capacity or bandwidth at long context asks for it.
