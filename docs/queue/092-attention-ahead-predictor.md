# 092 — Attention-ahead: layer l + 1's own attention as the predictor's margin

**Status:** Ceilings measured (2026-10-22, lab dump pred3: 8 requests × 450 tokens, 1,690 steps, 4,603 tokens) — the kernel's form to decide against the window
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
| route-ahead over x_l (today's input) | .576 | .568 | .601 |
| attention-ahead, exact FFN output, full step, state at step start (bound) | .624 | **.694** | n/a |
| attention-ahead, shared output only, full step, state carried | .593 | **.653** | n/a |
| attention-ahead, shared output only, read-out only, step-start state | .559 | **.608** | n/a |

(The first run, 14:04, on 4,584 tokens read .561 / .690 / .648 for the first three rows; the second agrees within .01.)
The x+shared / +rows columns of the old lab code score far *below* the route-ahead (.376–.432 at KDA): adding the FFN
output to x outside the hyper-connection is the wrong arithmetic — the attention-ahead path puts it through the real
mixes, and its exact-FFN variant reaches .694 of a .705 ceiling, so the lab's buffers are right and those three columns
are a bad construction, not bad data. Production's second margin on the same construction has a learned weight to
neutralise it (+.005).

**With the deployed model's learned terms on top** (phase3/sim.py on this dump: the causal online simulation of the
shipped shape — bias, co-routing, calibration, a second margin with its own weight — recall@13 over all 1,690 steps /
settled over the last third):

| shape | all | settled |
|---|---|---|
| today: route-ahead + the x+shared margin | .605 | .598 |
| today + the read-out-only attention-ahead as the margin | .620 | .613 |
| today + the full-step attention-ahead as the margin | .635 | .633 |
| the full-step attention-ahead as the primary, route-ahead as the margin | .643 | .636 |

So the full form is worth **+.038 recall@13 over the deployed model** (.598 → .636 settled; recall@16 .680 → .728),
the read-out +.015. Against ~15.3 host experts a layer that is ~0.6 more of them in the plan and ~0.6 fewer of the 13
planned wasted — a tenth of the 5.2 unplanned. The learned terms add less on top of attention-ahead than on route-ahead
(raw .593 → .643; route-ahead raw .576 → .605): the signal already carries what the bias and co-routing were learning.

**What the update is worth.** The read-out alone (q against the step-start memory: no forget gate, no delta update, no
k/v) recovers +.040 of the route-ahead's gap at KDA layers; the token's own update and the carry across the step's
tokens add another +.045 (.608 → .653); the exact FFN output another +.041 (→ .694). In the delta rule the read-out
follows the update, so a token sees its own k/v in the memory it reads — the read-out form loses exactly that.

**Cost in the window (the card's ~180 µs before `hc_post`; VRAM ~800 GB/s).** The weights are read once a layer for
all the step's tokens, not once a token. Read-out form: q (32 MB bf16) + g_a/g_b (2 MB) + o_proj folded into the router
(2.4 MB) + hc coefficient GEMVs ≈ 37 MB ≈ 45 µs. Full form (the .653 row): q, k, v, o (128 MB) + f_a/f_b, b, g (3 MB) +
the memory sweeps on a 2 MB copy a head set, a token (2–4 tokens: 16 MB r/w) ≈ 150 MB ≈ 170–190 µs — at the window's
edge in bf16; an int8 copy of layer l + 1's four projections for the predictor alone (64 MB, 34 layers → 2.2 GB of the
~12 GB free a card) brings it to ~90 µs, int4 to ~50. The forget gate and delta on a state *copy* (never written back)
keep the forward exact.

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
1. Decide the form against the window (tcdent): the full carried form (.653) at ~180 µs bf16 or ~90 µs with an int8
   predictor copy of the projections, or the read-out (.608) at ~45 µs as the first step. Recommended: the full form
   with the weights read once a layer; the read-out as the fallback if it does not fit when timed.
2. The kernel, timed in the window; the production read is the plan's precision by rank and completed experts a layer.
3. The lookahead that informs the run: once the predicted routing is reliable, the host run starts on the predicted
   experts before the real routing confirms (tcdent's direction, not designed).
