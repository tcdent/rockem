# 013 — The indexer's scoring and selection on the card

**Status:** Resolved
**Area:** decode, prefill, long context
**Gain:** measured — decode at 18k 48.9 → 41 ms; attention host time 1.8 → 0.0 ms; bit-identical

## Proposal
The host scored every pool (32 heads × 128 dims) and picked the top 512 on one thread for each
MLA layer: 150–190 ms a token at 17–18k positions. Move scoring (`ix_score`) and selection
(`ix_select`, a radix select over the scores' order-preserving bits) to the card.

## Source
tcdent's 17k conversation (2026-10-17); tinygrad's card-side indexer.

## History
- 2026-10-17 (`1270994`): the selection over the thread pool first (24–29 → 0.2–0.3 ms at 3k).
- 2026-10-17 11:28 (`6a9be6f`): scoring on the card; the pool keys mirrored on every step and on
  seed (a first build that mirrored only selecting steps scored garbage at the 2050th position).
- 2026-10-17 12:38 (`bf224d2`): selection on the card, no host round trip.

## Next
WMMA scoring for long context (019).
