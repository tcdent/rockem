# 058 — Indexer and attend of an MLA layer across both cards

**Status:** Built (2026-10-20): the MLA attend split by positions across both cards — prefill at 18k 2.76 → 2.65 ms a prompt token, decode neutral, numeric (NLL 0.0419 → 0.0468)
**Area:** long context (decode and prefill)
**Gain:** est. up to half of the context-growing cost (~0.08 ms a prompt token per 1k positions today)

## Proposal
`card_of` puts each MLA layer on one card, so the other card idles through its indexer and attend.
Shard the indexer's pools across both cards (each scores half and selects its top candidates; a
fixed-order merge picks the 2,048) and the attend's heads across both cards. The cost that grows
with context then runs on two cards. An exact merge keeps `ix_select`'s set.

## Source
vLLM PR #61868: dense indexer rows sharded over the TP ranks and all-gathered (−7.6 % a step at
batch 32, enabled from 8 requests). Our inventory 043 (one card idle per MLA layer).

## History
- 2026-10-18: proposed.

- 2026-10-20: chosen next for prefill (tcdent). On the full-step card lanes an MLA layer's card chain for a
  512-token chunk is ~24 ms — `mla_attend_w` 17.7 ms, `mla_o_w` 4.6 (079), the projections — against ~17.5 ms of the
  host's experts for the other chunk, so the host stalls on 11 of 42 layers while the other card spins in `hc_post`.

> "One million context is one of my hard constraints." — tcdent, 2026-10-20

## Design (2026-10-20)
The split is by positions, not heads: by heads every card would mirror every MLA layer's latents (~12.9 GB a card at
1M, past the VRAM beside the weights); by positions each card holds half, as today's total.

- **Ownership.** Position `p` belongs to card `(p / G) % 2` (G = 16, a tile), stored compactly at local row
  `(p / 2G) · G + p % G` in that card's half mirror of the layer's latents. The host uploads each finished row to its
  owner (a step's rows, a seed, a restore).
- **Query on both cards.** The MLA card's `q_a` output (`mqa`, `[n][Q_LORA_RANK]`) is copied to the other card, which
  runs `q_b` and `q_abs` itself (its own copy of the layer's `q_b`, `q_a`'s norm and `kv_b`: ~36 MB a layer).
- **Selection.** The indexer stays on the MLA card (`ix_score_w`, `ix_select`: ~0.6 ms a chunk); a split kernel
  turns each token's selected positions into two lists of local rows, its own and the other card's (peer write, then
  its word).
- **Attend on both cards** over their own lists, emitting the unnormalized mix with the softmax's (max, sum)
  (`mla_attend_w` and `mla_attend_merge` in a raw mode); `W_uv` on the unnormalized mix (linear), so what crosses the
  link is value-space (`[n][MLA_HEADS][V_HEAD_DIM]`: 32 MB a 512-token chunk, half the latent-space partials) plus the
  (max, sum) pairs; the other card's land in the MLA card's VRAM with a word.
- **Merge** on the MLA card: `heads = (w_a·h_a + w_b·h_b) / (w_a·l_a + w_b·l_b)`, `w = exp(m − max(m_a, m_b))`, then
  `mla_o` and `hc_post` as today. The other card's `hc_post` (waiting for the layer's output) is posted after its
  attend chain, not before.
- **Numeric, not exact:** the two partials' merge reorders the softmax sums; gated on the long SCORE NLL distance.
- The draft layer (MLA, on card 1) goes through the same functions.

## Built (2026-10-20, runs of the track-launches gate: mla2c/mla2d, then mla2f)
As designed above, with what the decode trace taught: `q_abs` (both cards) and the other card's `q_b` posted with the
projections, behind the fence the host waits on (they need no rows); the other card's W_uv into its own VRAM, then one
copy across (its scattered stores straight over the link ran 79 µs against 36); the verify step's attend chunk 256 →
128 positions (each card's half list filled half the chunks and a workgroup's time stayed the whole list's) — and the
chunk merge's (m, l) load looping, since 16 heads × 17 chunks is past its 256 threads (the first build of it read
head 15's chunks from garbage, NaN logits, a panic in sampling).

| 18k | before | split |
|---|---|---|
| prefill, ms a prompt token | 2.76 | 2.65 |
| `mla_attend_w` a 512-token launch | 17.7 ms on one card | 8.5–9.0 ms on each |
| the host's FFN-input wait, ms a prompt token | ~0.63 | 0.49 |
| cold 18k prompt | 53.3 s | 51.3 s |
| a verify step (`laps.sh`) | 44.05 ms | 44.34 |
| long SCORE NLL | 0.04189 | 0.04678 (mean \|Δ logprob\| 1.4e-2, max 6.3) |

The NLL moved as the earlier accepted numeric changes did (aw3 → act 8.3e-3, act → fold1 8.7e-3, wmma → aw3 9.6e-3
mean |Δ|), at the top of their range. The first run of this build hit a PCIe error storm (11 APEI records, 21 AER
lines on the cold prompt, the host's experts lap doubled); a fresh boot of the same build had none.

## Next
- The draft layer's attend rides the same split; nothing to do.
- The indexer's pools sharded too (058's original second half) when long context makes the indexer matter.
