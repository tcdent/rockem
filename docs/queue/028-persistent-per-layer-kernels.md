# 028 — Persistent per-layer kernels

**Status:** Quantified, on the shelf — ~1.2 ms a card a step of dispatch (≤ ~1.2 ms an emitted token); the host waits it cannot remove are the larger structure
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

- 2026-10-19 01:08 (lab boot, `track-launches` node, residency off, run cfcf31ed): a two-token verify step at 18k
  posts 422 launches on card 0 and 436 on card 1; a 512-token prefill step 927 / 940. Of a card's 422, 90 are
  `hc_post` (the handoff wait for the host: 27 ms of the 55 ms step, 305 µs mean), 90 `hc_pre`, 81 `rows_h`,
  67 `ffn_gu`, 68 `rows_part`, 68 `kda_step`, 68 `rows_d2`, 42 `ffn_down_shared`, and the MLA layers' 11 each
  of `mla_o`, `mla_attend_w`, `mla_qb`, `ix_select`, `mla_qabs`, `ix_score_w`, `mla_uv`. The stamps are
  completion to completion, so a launch's dispatch gap is inside its zone; the per-launch cost measured on
  2026-09-19 (3.6 µs for a barrier launch: signal, acquire, dispatch) over the ~330 fusible launches a card is
  ~1.2 ms a card a step, the cards overlapping — the earlier "~3 ms a step" counted gaps that are the host's.
  The 90 host waits a card are the dependency structure (the experts on the host), which no persistence
  removes; 001/002/034 are about those.

## Next
On the shelf at ~1.2 ms a card a step. The MLA layers' seven launches and the KDA layers' five are the fusion
candidates if taken; read vLLM #61662's structure first.
