# 067 — The scalar q_abs at a verify step

**Status:** Quantified, on the shelf — 0.4 ms a verify step (39 µs a layer × 11 MLA layers)
**Area:** decode
**Gain:** measured context — the WMMA twin `mla_qabs_w` serves steps past `SPEC_TOKENS`; at two tokens the scalar kernel runs

## Proposal
`mla_qabs` at a verify step takes 39 µs a layer for a 2-token absorb through `W_uk`. It wants the rows shape (a
row a wave over a transposed `W_uk` copy, read once a layer) rather than its per-head loop.

## Source
017's history (2026-10-18): the attend 151 → 36 µs left q_abs the next item.

## History
- 2026-10-19: given its own entry (it lived in 017's Next).

## Next
A transposed `W_uk` copy on the card at load (11 layers × 128 × 512 × 2 bytes) and the rows kernel over it.
