# Phase 3 offline experiment: can a learned correction lift the layer-ahead predictor?

Data: bringup/expert-cache/data/pred.bin, appended by the node (feature dump-pred) at every decode layer of ~24 requests x 450
tokens on an 18k-token context (GLM-5.3-Flash, metatron, 2026-10-21). Record layout, little-endian, back to back:
  u16 layer l, u16 n (tokens in the step, 1..4), then per token:
    8 x u16  picks at layer l (the token's actual 8 routed experts at layer l, expert ids 0..287)
    32 x f32 the quantized FFN input's group scales (H = 4096, groups of 128)
    4096 x i8 the quantized FFN input of layer l (x_l ~= q * scale per group)
    288 x f32 the layer-ahead logits: layer l+1's router applied to layer l's normed FFN input (before the sigmoid;
              the predictor today = top by sigmoid(logit) + bias[e]; the bias is not in the file)
Records come in step order; within a step, layers ascend (MoE layers 3..45, plus the draft layer possibly). A step's
record for layer l and its record for layer l+1 are consecutive records with the same n; the label for the
prediction made at l is the picks in the l+1 record (same token index).

The question: does an online-learned correction to the layer-ahead logits (zero-initialized, never frozen, updated
after each labelled step) raise recall of the next layer's picks at a cap of 13 experts per step (the union over the
step's tokens, ranked), over ~10k tokens? Candidates: per-expert bias; low-rank map of x_l (rank 16/32/64) onto the
288 logits; co-routing from layer l's picks; combinations. Train causally (predict, then update with the label),
per layer or shared across layers with a layer embedding, cheap enough to run on a GPU in ~100 us a layer.
Report recall@13 (and @8/@16) over time (rolling) against the uncorrected logits, by layer band, and the settled
gain. Note: the residency on the cards is not in the file; measure recall over all picks (the host experts are
~90 % of them).
