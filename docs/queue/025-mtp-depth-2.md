# 025 — Draft depth 2

**Status:** Deferred — waited on the verify step's cost; ready to re-measure
**Area:** decode
**Gain:** est. ~0.5–1 ms an emitted token now; marginal at real-session acceptance

## Proposal
Draft a second token from the draft layer's own output; verify three tokens a step.

## Source
vLLM (0.57 at position 2); tinygrad defaults to two drafts ("as fast as 3 on predictable text").

## History
- 2026-10-17 16:47 (`aa5324d`): second-draft acceptance 0.42 (top-1 0.50); a three-token step
  75–78 ms for 2.45 tokens = 30.6–31.8 ms an emitted token vs 30.2 at depth 1. Kept at
  `DRAFT_DEPTH = 1`; pays once a three-token step is under ~65 ms, which the expert union bounds.
- 2026-10-17: 003 halved the expert phase (10.5 → 6.0 a step token), the condition likely met.

## Next
Re-measure with the cache warm, at real-session acceptance (0.84). A constant change.
