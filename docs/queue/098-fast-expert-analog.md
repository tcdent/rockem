# 098 — A fast analog of the model, for predicting experts

**Status:** Open — a direction (tcdent, 2026-10-25), queued for a later pass; not designed
**Area:** decode, the prefetch's predictor
**Gain:** unknown; the room is measured — the prefetch covers 64 % of the non-resident experts a layer needs (rows
plan, 17.1 a layer, `plan_eval` on aff85), and its last posts are 67–72 % right

## Direction
> i think we should be able to create a reasonable analog of the model that we're serving. that allows us to predict
> experts faster than actually running the model. because we're not trying to accurately predict any tokens. we're just
> trying to accurately predict the layer. and so like we have all of the data available in the model already. the only
> limitation in our ability to use it is that it takes too long. so we should think about alternate ways to think about
> that. — tcdent

> i don't remember if it was the ffn or some other layer, but like we don't have enough time to process that into like
> a one-bit representation that then feeds this model. and maybe we can find slots to be able to do that to where then
> we can process it faster and we can store the training that we've done in potentially a smaller quantization.

And the rule for using any predictor:

> i don't want us to incorporate too much deterministic shuffling in here. that's trying to correct for its
> shortcomings. but i think we can leverage it according to the information it gives us to utilize it as effectively
> as possible. and to make sure that we're not incorporating any waste into the process.

## What is known (2026-10-25)
- Today's predictor (`plan.rs`, 082/093): per token a calibrated sigmoid per expert (BCE), a step's union ranked by
  Π(1 + eᶻ) = 1 / P(no token picks it) under independence. Two passes: the early plan from the route-ahead (layer
  l + 1's router over layer l's FFN input), and the rows plan ~100 µs later over the whole FFN output.
- Measured on tcdent's 360k session (50k layer runs, `bringup/reads/plan_eval.py`): P(picked) by rank, rows plan,
  .999 at 1, .907 at 8, .672 at 13, .512 at 16, .237 at 24; the rows plan calibrated within a few points, the early
  plan overconfident from ~.3 up (.85 predicted, .66 observed). Details in 090.
- The input the better plan needs (the whole FFN output) arrives late; an analog that reaches the same information
  sooner is the lever — which representation (one bit a channel of the FFN output or the residual?), which layers'
  inputs, where it runs in the step's idle time (the cards' waits, the host's), and how its training is stored
  (lower precision) are the open questions.

## Related
005 (layer-ahead routing), 082 (card-side prediction), 092 (attention-ahead predictor, shelved), 093 (the rows plan),
090 (the prefetch's use of the plans).
