# 063 — Residency filled during prefill, from the prompt's own routing

**Status:** Open — the largest exact decode lever at long context
**Area:** decode (at long context), prefill
**Gain:** est. experts 19.1 → ~13 ms a two-token verify step at 18k (−6 ms a step, ~−3 ms an emitted token)

## Proposal
The resident experts (003, 004) fill one slot a decode step and never during prefill. In a world where
every request begins with a long prompt (tcdent, 2026-10-18: "there is never not a time that we're not
prefilling"), decode starts with the cards' slots empty: after the 94 s prefill of an 18k prompt the
`experts` rows read 3 of 1754 slots filled, and a 328-token generation got them to 80. The 30–40 % of
picks resident that gave the 6.7–7.4 ms expert lap was only ever reached after sixteen warm-up rounds.

Prefill knows the prompt's routing for every token (the aged counts already train on prefill picks),
and the cards idle through the CPU's expert phase of every prefill step, ~1.1 s of each 2.5 s step at
512, ~40 s over an 18k prompt: room to pull the 23 GB of slots several times over. Fill the slots
during prefill from the prompt's counts, with loads in flight while the CPU runs experts (decode's
pacing to one load a step came from the loader doubling the card laps it competed with; in prefill the
card has nothing to compete for during the expert phase), so that decode begins at the residency it
otherwise reaches hundreds of tokens later.

## Source
The prow baseline runs of 2026-10-18 (e03735b6: `experts` rows, `steps.sh` at band 10k); 003's
warm-up measurements; tcdent's long-context rule.

## History
- 2026-10-18: proposed from the baseline's `experts` rows.

## Next
1. The loader's schedule in prefill: loads flat out inside the expert phase, paused for the card
   phases; the aged counts' half-life against an 18k prompt's picks.
2. Measure: resident share and the expert lap of the first 100 decode steps after an 18k cold prompt,
   against the baseline's 2 % and 9.5 ms a step token.
3. Then 038 (the residents used in prefill too) and 045 (the copy engine pulls instead of a core).
