# 086 — The prefetch predictor's input map

**Status:** Quantified, on the shelf — missed −0.10 a layer, but the experts it adds stay partial: no time gained, +0.28 ms a verify step
**Area:** decode
**Gain:** offline +.007 recall@13 on top of phase 3's .621; in the node, by phase 3's ratio (missed 6.91 → 6.45 for
+.036 recall), ~0.1 expert a layer (inferred)

## Proposal
Phase 3's predictor (`models/glm-5.3-flash/src/predict.rs`, docs/expert-cache-architecture.md "Phase 3") has no term
for the token's own FFN input. Adding a rank-64 map a layer pair (x̂·U·V, x̂ the RMS-normed dequantized `b_xq`) lifted
settled recall@13 .621 → .628 in the simulation (rank 16 .626, a shared rank-32 map .626, full rank .632). Updated
dense with Adam, it is ~14M parameters, ~330 MB of memory traffic and 195 ms a step on one core (bench-pred,
2026-10-21), against a ~25 ms step. Cheaper forms to test before it fits: rank 16 (a quarter of the bytes), the map
updated every k steps or by plain SGD (no moments), a shared U across layers (`lrs32`: 0.5M parameters).

## Source
bringup/expert-cache/phase3 (sim.py, summary.csv); the first port of predict.rs with U/V (commit b105800's message).

## Next
Replay the cheaper forms in `bench-pred` (recall and µs a pair); build only if one fits the learner core's step with
room to spare.

## Built on the cards and measured (2026-10-21, commit ddb13a7, reverted)
Rank 64, decode-trained: `pred_h` (x · U, a slice of the input a workgroup over 64), h in `pred_step`, V and dh in
`pred_learn`, U in `pred_map` (64 workgroups). One workgroup reading U whole took ~490 µs a layer at 4 tokens; spread,
the chain was ~220 µs a layer (135 without the map). Offline with the shared-output margin: settled recall@13 .6255 →
.6305; rank 128 the same, full rank worse (.6213). On the long decode run (bringup/box/longdecode.sh, 24 × 450
tokens): missed 5.80 → 5.70 (second half 5.72 → 5.57), complete 7.79 → 8.13, but the verify step 62.03 → 63.27 ms
(`kda_card` +0.9, `mla_proj` +0.3, `slot` +0.2): `hc_post`'s wait is ~180 µs at the median (the card runs the residents
and the shared FFN in the host's 550), so the extra ~85 µs a layer pushes the card's next layer. Takes a shorter
predictor chain first.

## Second try: its learning at the step's end (2026-10-21, card11/card12, reverted)
In the wait only `pred_h` and the map term (chain ~100 → ~116 µs a layer); V, dh and U at the step's end with the other
updates (`pred_learn` ~90 µs, `pred_map` ~160–190 µs: U and its moments in f32, ~125 MB a step a card). Queued ahead of
the head it held the logits (`head` 1.93 → 2.31 ms, verify 61.79 → 62.35); queued behind the logits read it runs while
the host samples (`head` 1.91) — that placement is kept. Long decode run against card10: missed 5.81 → 5.71 (second
half 5.73 → 5.61), but complete 7.75 → 7.74 and partial 2.51 → 2.60: the experts the map adds are posted and not done
by the run. The experts lap 24.62 → 24.67 ms, `slot` 1.07 → 1.23 ms, the verify step 61.79 → 62.07 ms. What bounds the
decode prefetch now is how much of what it posts completes, not which experts it names.
