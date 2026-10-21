# 088 — Prefill's predictor labels off the main thread

**Status:** Done 2026-10-21 (commit 60c51dc)
**Area:** prefill
**Gain:** 0.03 ms a prompt token, prefill back to 1.85 (pred2 1.84)

## Proposal
Phase 4 (docs/expert-cache-architecture.md) trains the prefetch predictor on every prompt position. At each MoE layer's
dispatch the main thread reads the next layer's router scores for the chunk out of host memory (`ffn_ahead`, 512 ×
288 f32 a layer-chunk, strided rows of the `[STEP][ROUTE_ROWS]` buffer): the prefill `slot` lap 0 → 0.03 ms a prompt
token (2026-10-21; 0.04 with the earlier double copy). The buffer is rewritten by the next layer's `hc_pre`, so the
learner cannot read it later. Candidates: the matrix path writes rows E.. of a long step into a per-layer ring of its
own (`[layers][STEP][E]` f32, ~52 MB of host memory), and the learner reads it at its own pace; or the read goes
over the pool while it waits, as the activations' read does.

## Source
docs/expert-cache-architecture.md "Phase 4 in the node"; bringup/reads/steps.sh 13421e6e vs b210fa5f.

## Result
The cost was not the read. Streaming the read over the pool left the `slot` lap at 0.03 ms a prompt token. It was a
fresh 590 KB lesson buffer a layer-chunk (a map, page faults, zeroing). The learner now returns buffers for reuse:
`slot` 0.03 → 0, prefill 1.87 → 1.85 ms a prompt token (pred7, 68ef7287). The ring was not needed.
