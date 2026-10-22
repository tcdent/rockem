# 092 — Attention-ahead: layer l + 1's own attention as the predictor's margin

**Status:** Ceiling measured (2026-10-22, lab dump pred3, 8 requests, 4,584 tokens) — a kernel to build
**Area:** decode
**Gain:** raw recall@13 at KDA layers .561 → .648 (the card-runnable form measured so far), .690 with the exact FFN
output; the exact-input ceiling is .705. Every expert predicted right is 13.4 MB not streamed for nothing: the
prefetch's bytes are the lever (091)

## Where it came from
tcdent (2026-10-22): "I like the attention head candidate since it actually incorporates the model itself … if we're
able to leverage some of the knowledge that we have around routing, we should be able to predict somewhat. And then the
interesting thing is that the expert routing prediction could potentially eventually inform the actual generation … a
look ahead on expert routing … a head start on the actual real run."

Three Opus research agents worked one brief (bringup/expert-cache/briefs/predictor-research-brief.md); their write-ups
are predictor-research-{literature,architecture,time}.md beside it. They agreed on the diagnosis: the error of
today's predictor is layer l + 1's attention, which the route-ahead on x_l cannot see; it is high-rank (the top 64
directions of the logit delta recover .651 of a .705 ceiling, 128 recover .673), so no map of x_l reaches it (ridge
.613, a 17 M-parameter MLP .620, measured on phase3b); deeper lookahead decays (.592 → .529 → .485 at 1/2/3 layers);
recency and co-routing are at their ceilings; the one-layer MTP drafter carries nothing for it. The architecture agent
proposed computing the missing term with the model's own weights instead of learning it: run layer l + 1's attention
read-out on the stale input against its real recurrent state, already in that card's VRAM.

## Measured (phase3c/score.py over data/phase3c; raw recall@13 of the step's union against the actual picks, no
## learned corrections — today's corrections add ~+.04 on the route-ahead)

| column | all | l + 1 KDA (34 layers) | l + 1 MLA (8) |
|---|---|---|---|
| route-ahead over x_l (today's input) | .569 | .561 | .594 |
| attention-ahead, exact FFN output, state at step start (bound) | — | **.690** | n/a |
| attention-ahead, shared output only, full step, state carried | — | **.648** | n/a |
| attention-ahead, shared output only, read-out only, step-start state (the kernel's form) | pending | pending | n/a |

The lab path (`dump-pred`): the token's residual streams fetched off the card at dispatch, layer l's FFN mix applied with
the shared output (or every routed row, for the bound), layer l + 1's hyper-connection pre, input norm, `kda::step` or
`kda::readout` on the host reference with the state snapshotted at the step's start, its post, the FFN pre, post norm,
router. MLA layers (3, 7, …, 43) have no KDA state; their columns repeat the route-ahead.

## The kernel (to build)
On the layer's FFN card, in the ~180 µs before `hc_post` where `pred_step` runs today, for the step's 2–4 tokens:
hc-pre of l + 1 on the streams mixed with the shared output (two coefficient GEMVs over the flattened streams), the input
norm, the q projection (32 MB bf16, ~40 µs of VRAM), the q convolution against the window, l2-norm, `S·q` a head on the
step-start memory (2 MB), the gated norm (g_a, g_b), o_proj folded into layer l + 1's router (288 × 4096, 2.4 MB),
hc-ffn-pre, post norm → 288 logits a token, added to `pred_step` as a third margin with its own learned weight and
calibration. No state is written. Budget ~36 MB of reads and ~20 MFLOP a token: ~45 µs. The learned corrections
(bias, co-routing, calibration) stay as they are; pretraining the residual from traces is open (tcdent: "I'm also okay
with a concept of pre-training this").

## Next
1. The read-out column's number (running). Below ~.62 at KDA layers, the step-start read-out loses too much to the
   missing update and the carried form (k, v, forget gate: +64 MB of weights) is the kernel instead.
2. The kernel, timed in the window; the plan's precision by rank and completed experts a layer as the production read.
3. The lookahead that informs the run: once the predicted routing is reliable, the host run starts on the predicted
   experts before the real routing confirms (tcdent's direction, not designed).
