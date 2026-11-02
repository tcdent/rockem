# train — the training side

The code a continual-adaptation round needs and the node never serves with. Its own Cargo
workspace, kept apart from the inference tree on purpose (tcdent, 2026-10-25: training "can be
somewhat isolated"); it depends on `common` and `models/<name>` and changes nothing in them.
The design it implements is `~/Work/localai/continual-adaptation.md` and `sdpo.md`; the
decisions taken for this platform are in the crate docs.

```
train/
  common/          training primitives with no model in them: the transposed products over
                   the inference byte formats (the gradient at a frozen matrix's input), the
                   adapter's low-rank steps
  glm-5.3-flash/   GLM-5.3-Flash's layer backwards over glm_5_3_flash's bytes and kernels, its
                   adapters, and the f32 reference each backward is diffed against
```

Conventions, inherited from the inference tree: a definition (scalar) and one realization a
target, bit-identical where the arithmetic allows it; an f32 reference the kernel is diffed
against at the real shape, and the reference itself checked against finite differences at a
small shape; one op at a time, measured.

## What exists

| op | where | gate |
|---|---|---|
| `dx += Wᵀ·dy` over `q4g128` | `common/src/q4.rs` | AVX2 = scalar bit for bit; scalar = dense `Wᵀ·dy` |
| adapter steps `y += B·(A·x)` and their backward | `common/src/lowrank.rs` | through the expert's diffs |
| routed expert with adapter: forward (keeps what the backward needs) and backward | `glm-5.3-flash/src/expert.rs` | zero adapter = `glm_5_3_flash::expert::forward` value for value; forward and backward within ~2e-6 of the dense f32 reference (gate 1e-4) |
| `dx += Wᵀ·dy` over FP8 128×128-block rows | `common/src/fp8.rs` | AVX2 = scalar bit for bit (both `fma`); scalar = dense `Wᵀ·dy` |
| `dx += Σ dy_r·W[row_r]` over bf16 rows (the router's picks) | `common/src/bf16.rs` | AVX2 = scalar bit for bit; = dense |
| the clamped SwiGLU's and the RMS norm's backward | `common/src/swiglu.rs`, `common/src/rmsnorm.rs` | finite differences |
| the MoE half on its normed input: router (frozen; picked rows), experts with adapters and their weights, shared expert — to `d xn` | `glm-5.3-flash/src/ffn.rs` | zero adapters = `ffn::moe_routed` + `moe_shared` value for value; vs the dense reference at the real shape 1e-4, except the two adapter gradients downstream of the quantized ĥ, gated at a quantizer step (the sides' h differ by summation order; a boundary element flips one INT8 step) |
| the hyper-connection around a block half: `post`'s backward to `d y` and the residual's share; `pre`'s through the sigmoids, the softmax and the Sinkhorn chain (unrolled in reverse over recomputed intermediates), `fn_ᵀ·dc` and the unweighted norm, to `d S` | `glm-5.3-flash/src/hc.rs` | over the inference forward, vs the dense reference 1e-4 at the real shape |
| KDA over a run of positions: the inference step forward; per head the recurrence recomputed from the entering memory and run in reverse, the l2 norms, the convolution over the run's timeline, every projection's transposed product — to `d xn` and the entering state's gradient | `glm-5.3-flash/src/kda.rs` | vs the dense reference 1e-4 at the real shape (y, the state after, `d xn`, `d mem`, `d window`) |
| MLA over a run of positions: the inference step forward; the absorbed attention's backward straight into the latents' gradient (no expanded K/V), the query path, the run's own latents through their norm and `kv_a` — to `d xn`; the indexer gets no gradient | `glm-5.3-flash/src/mla.rs` | two runs vs the dense reference over both, 1e-4 at the real shape (`d xn`, every position's `d latent`) |
| the head: the streams' mean, the final norm, `lm_head`'s transposed product — to `d streams` | `glm-5.3-flash/src/head.rs` | vs dense arithmetic at a 4,096-row head, 1e-4 |
| the loss: forward KL against a top-K teacher (renormalized on its support) and its gradient at the logits | `glm-5.3-flash/src/loss.rs` | finite differences; the floor −ln(student mass on K) when teacher = student |
| the model over a run: the embedding to the logits, the logits' gradient down to the first MoE layer's input; runs chain through the states and a `Carry` (KDA's entering-state gradient, every MLA latent's) | `glm-5.3-flash/src/model.rs` | a 5-layer random model (3 dense, MLA+MoE, KDA+MoE): the directional derivative along the gradient of one layer's adapters within 2 % of ‖g‖ (the f32 forward's noise); see below |
| Adam (torch's form, split into moments and apply) | `common/src/adam.rs` | torch's first steps |
| the optimizer: Adam per expert under a cap on the expert's **effective** movement `‖Δ(B·A)‖_F / ‖W‖_F` (Gram matrices, never the product), one scale by bisection; the distance from the base as the drift instrument; the base norms from the q4 record | `glm-5.3-flash/src/optim.rs` | Gram norms = dense norms; a huge gradient lands on the cap exactly, a small one is untouched |
| a round: sessions fed as ids + targets, forward in runs with KDA snapshots at the boundaries, the backward recomputing each run from its snapshot in reverse (one run's intermediates alive, ~50 MB a position), a step over the touched experts, the report | `glm-5.3-flash/src/round.rs` | a session in runs of 3 = the same session as one run, gradient within 1e-5 and loss within 1e-6; the step moves every touched expert within the cap and consumes the gradients |
| the dense f32 reference of the above | `glm-5.3-flash/src/reference.rs` | finite differences on every parameter class and on `dx`, for the expert, the whole MoE (router path included), the hyper-connection, KDA over a sequence (with the state's gradient) and MLA over a sequence |

## The end-to-end gradient check, and what limits it

`model::tests::run_gradient_matches_finite_differences` perturbs the adapters of one MoE layer along
their own gradient `g` (normalized) and compares the central difference of the loss with `‖g‖`.
Measured 2026-11-01 on the test's random 5-layer model:

| direction | ε | central difference / ‖g‖ − 1 |
|---|---|---|
| layer 4, the `down` adapters' B only (no quantizer on the path) | 0.05 | 0.9 % |
| same | 0.01 | 2.2 % |
| layer 3, `down` B only (through layer 4's expert-input quantizer) | 0.05 | 0.2 % |
| layer 4, every adapter matrix (through the experts' activation quantizer) | 0.01 | 9.4 % |
| same | 0.05 | 3.5 % |
| same | 0.2 | 0.6 % |

Two effects, neither a gradient error: on the quantizer-free direction the discrepancy grows as
1/ε — the f32 forward's noise on the loss (~1e-7) over the interval; across a quantizer it shrinks
with a wider interval — INT8 rounding flips, discontinuities the straight-through gradient does not
see, averaged out over a longer step. A per-parameter check is hopeless here: one adapter entry's
gradient is ~1e-6, below the noise.

## Measured: a round's time, one core

`cargo run --release --example round_rate` — the random five-layer model (`fixtures`), 16 tokens,
targets on half, rank 8, one core of the dev box (Zen 1), 2026-11-02:

| per layer kind, ms a token | forward | backward |
|---|---|---|
| KDA layer | 27 | 37 |
| MLA layer (16 positions) | 71 | 73 |
| MoE half (8 experts + shared) | 30 | 69 |
| dense FFN | 98 | — |
| hc half | 0.07 | 0.26 |
| head, full vocabulary, per target position | 112 | 140 |

Extrapolated to the 45 layers: feed forward ≈ 3.3 s, recompute ≈ 3.3 s, backward ≈ 5.0 s —
**≈ 11.7 s a session token on one core.** Every op here streams its weights once *per token*
(the inference CPU path is a one-token path; the node's multi-token forms are the cards' prefill
kernels), so the round is bound by weight traffic: ≈ 18 GB of weights a token a pass (KDA ≈ 9.4,
experts 4.2, head 1.3, MLA 1.5, shared 1.0, dense FFN 0.5), three passes, ≈ 54 GB a token — at
metatron's ~300 GB/s that is ≈ 180 ms a token however many cores run it. The lever is the one
prefill took: **multi-token forms**, every weight read once a run and applied to the run's tokens,
after which the round is compute-bound (~30 GFLOP a token over the three passes) and threads pay.

## Measured

`cargo run --release --example expert_rate` — one expert, one core, cache-hot (one expert is
~12.6 MB), so the instruction cost of each op rather than the DRAM stream:

| 2026-11-01, EPYC 7551P (Zen 1, the dev box; not metatron) | µs | × inference forward |
|---|---|---|
| inference forward | 1,869 | 1.00 |
| forward with a rank-16 adapter | 2,220 | 1.19 |
| backward, no adapter | 5,882 | 3.15 |
| backward, rank 16 | 6,050 | 3.24 |

The backward reads the same bytes as the forward once more, but each nibble becomes an f32 and
multiplies an f32 coefficient (no integer dot), so it runs ~0.5 instructions a weight against
the forward's ~0.14: instruction-bound where the forward is memory-bound. Whether that shows at
all under the pool on metatron (48 cores sharing the DRAM stream) is the next measurement; the
lever if it does is an integer inner loop over a quantized coefficient. The adapter at rank 16
is +19 % on the forward here — ~0.6 MB of f32 against 12.6 MB of expert, read with plain loops.
