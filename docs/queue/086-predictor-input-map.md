# 086 — The prefetch predictor's input map

**Status:** Quantified, on the shelf — +.007 settled recall@13 offline (.621 → .628); its cost kept it out
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
