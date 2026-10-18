# 005 — Layer-ahead routing for the prefetch

**Status:** Resolved
**Area:** decode
**Gain:** measured +0.5 ms a step token (29.0 → 28.1–29.0)

## Proposal
Predict layer l+1's experts with layer l+1's router over layer l's FFN input, a layer earlier and
more accurately than the router over the attention block's input.

## Source
Pre-gated MoE (2023) and cross-layer gating; our routing trace.

## History
- 2026-10-17 (`9c2a066`): scored — names 4.88/5.06 of 8 picks (code/chat) against 4.12/4.63.
- 2026-10-17 17:10 (`0ea8ebc`): built — the FFN half's router launch carries both layers' rows
  (`ROUTE_ROWS`), the prefetch issued when the layer's run returns, the attention-half early
  router retired. Served 5.59 named of 8. The shorter window ate part of the better prediction
  (34 % of predicted slices pulled before the run, 60 % before).

## Next
None; feeds 038 and 039 (the step knows layer l+1's picks a layer early).
