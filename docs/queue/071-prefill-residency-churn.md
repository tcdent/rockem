# 071 — The residency's replacement churn through a prompt

**Status:** Quantified, on the shelf — ~3 % of prefill (5.0 → 5.16 ms a prompt token at 512; the BAR writes during the card phases)
**Area:** prefill
**Gain:** measured — 063's loader made 9,174 loads over an 18k prompt at 512 (3,445 at 1024), ~250 a step, after the slots were full in three steps

## Proposal
Once the slots are full, replacements continue at up to four a layer a step; each is a 13 MB BAR write that
competes with the handoffs during the card phases. Fewer replacements once full (the margin in prefill units,
or a budget a step), or loads held through the card phases and released in the CPU's expert phase, where the
link is idle.

## Source
063's history (2026-10-18/19).

## History
- 2026-10-19: given its own entry (it lived in 063's Next). At 1024 the churn is already a third of the 512 case.

## Next
A budget of replacements a step after the fill (one constant), measured on the cold 18k prompt.
