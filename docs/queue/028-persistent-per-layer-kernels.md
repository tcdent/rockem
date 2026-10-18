# 028 — Persistent per-layer kernels

**Status:** Deferred — gain judged too small for the effort (≤ ~1.5 ms an emitted token)
**Area:** decode
**Gain:** est. ≤ ~3 ms a step net

## Proposal
One persistent launch a layer with device-wide barriers instead of ~10 launches.

## Source
vLLM #61662 (GLM-5.3-Flash, MI355X: a step 9.36 → 4.82 ms, 1,699 → 220 kernels).

## History
- 2026-10-16 12:40: parked at ~1.5 ms (gfx1100 device barriers 6–8 µs; one kernel takes one VGPR budget).
- 2026-10-17 (`bdc57ec`): re-costed from the timeline — a two-token step posts ~430 launches a
  card, ~2.3 ms of gaps and ~2–3 ms of per-launch dispatch; barriers replacing them ≈ 2 ms; net
  ≤ ~3 ms a step.

## Next
Revisit when the larger levers are spent; read vLLM's structure first.
