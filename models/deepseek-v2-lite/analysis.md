# DeepSeek-V2-Lite — numerical analysis

Measured behaviour of this model under our implementation, kept next to the model so
that recipe and kernel decisions are made from these tables rather than from best
practice. Every number here was produced by a command in this repository on
2026-09-17; the commands are given so the tables can be regenerated when a recipe,
a kernel, or the goldens change. Present tense describes the model and the current
recipe; dated notes mark what was decided from what.

## 1. What "correct" means here, and against what

Three anchors, each with one job, none reducible to another:

| anchor | question it answers | criterion | tool |
|---|---|---|---|
| the scalar twin of a kernel | is the fast realization implemented right | **equality**, byte for byte | in-package tests, `bringup/gpu/*.py` |
| the oracle (pinned `modeling_deepseek.py`, f32, CPU) | is the model transcribed right | tolerance (f32 order noise) | `forward` vs goldens |
| the oracle, teacher-forced | is a recipe a faithful interpretation of the weights | KL(oracle‖ours) per position, top-1 agreement, oracle-token NLL | `forward --quiet` |

The oracle's own bits are not canonical: torch's reduction orders differ from ours by
the same ~1e-6 that ours differ from each other. The weights were trained in bf16 with
still other orders, so the model is, by construction, robust to noise at the bf16 scale
(~4e-3 relative). What it is not robust to, in *tokens*, is anything at all, because
the router's top-k and the greedy decode are step functions. Token identity is
therefore not a quality measure; teacher-forced divergence is.

## 2. Noise floors by source

| source | magnitude (relative) | measured how | reproducible across tiers? |
|---|---|---|---|
| f32 reduction order (CPU AVX2 vs GPU tree, FMA vs not) | ~1e-7 per layer; does not grow through 27 layers at position 0 | `node rockem.debug=` logits/layer diff, four placements | not yet — see §6 |
| INT8 activation quantization (q4g128-a8 expert input) | ~1e-3 per element (absmax/127) | by construction | yes (integer) |
| INT4 round-to-nearest experts, absmax/7, group 128 | **10–24 % max-norm per expert output** | numpy cross-check on golden inputs, layers 1/4/26 | yes (integer: 0 diff GPU vs CPU at every position) |
| routing flips induced by the above | ~1e-2 in the residual when one happens | per-layer trace, first-flip layer per position | — |

Placement matrix, logits relative error vs all-CPU (lasenora, 2026-09-17):

| placement | pos 0 | pos 3 | pos 5 | pos 8 | pos 9 |
|---|---|---|---|---|---|
| experts on GPU | 0 | 0 | 0 | 0 | 0 |
| attention on GPU | 4e-7 | 6e-3 | 4e-3 | 2e-3 | 5e-3 |
| FFN on GPU | 5e-4 | 6e-3 | 2e-2 | 2e-3 | 7e-3 |
| both | 5e-4 | 4e-3 | 2e-2 | 2e-3 | 3e-2 |

The integer path is bit-identical end to end. The float paths differ at 1e-7 and the
router amplifies that to 1e-2 once a decision within 1e-7 flips (position 1 at layer
21, position 5 at layer 14, position 9 at layer 10 in this run).

## 3. Teacher-forced divergence from the oracle

`forward --quiet --q4-layers …` feeds the oracle's tokens and compares distributions at
every position. Six golden sets: the 10-position "capital of France" set (a hard tail),
and five 82–91-position continuations in different domains (Roman history, Python code,
a dialogue, a science paragraph, a contract clause), 445 positions in all.

The transcription with bf16 experts is the oracle to within f32 on every set (KL ≤ 1e-10,
top-1 agreement 100 %). The INT4 round-to-nearest recipe, and the same recipe with only
**position 0** computed at full precision (see §5):

| set (positions) | all-INT4 KL nats/pos | top-1 | INT4, sink at bf16 | gain |
|---|---|---|---|---|
| Roman history (91) | 0.064 | 81/91 | 0.031 | 2.1× |
| code (91) | 0.092 | 88/91 | 0.057 | 1.6× |
| dialogue (82) | 0.049 | 77/82 | 0.012 | 4.1× |
| science (84) | 0.048 | 79/84 | 0.012 | 3.9× |
| legal (87) | 0.051 | 83/87 | 0.019 | 2.7× |
| "capital of France" (10) | 0.298 | 8/10 | 0.0066 | 45× |

By the usual standard for 4-bit (well-calibrated methods land at 0.01–0.05 nats), plain
RTN is expensive on this model, and between a third and three quarters of its cost on
ordinary text is one token.

**Calibrated INT4 (GPTQ) against RTN, with and without the sink** (2026-09-17). Same
format (q4g128, symmetric, group scales), integers chosen by GPTQ with Hessian error
feedback over the 257 positions of the code, dialogue and science sets; the two
evaluation sets below are not in the calibration set. Two calibrations: one over every
position, one with position 0 dropped from each sequence (3 of 257 rows).

| experts | legal (87), sink computed INT4 | legal, sink shipped f32 | Roman (91), INT4 | Roman, sink f32 |
|---|---|---|---|---|
| RTN | 0.0515 | 0.0196 | 0.0644 | 0.0316 |
| GPTQ, calibration includes position 0 | 0.0158 | 0.0147 | 0.0308 | 0.0290 |
| GPTQ, calibration excludes position 0 | 0.0363 | 0.0172 | 0.0443 | 0.0293 |

Top-1 agreement is 81–86 of ~90 in every cell and does not separate them. Three
readings:

- **Calibration on data that starts with BOS is, in large part, the sink fix.** GPTQ
  calibrated with position 0 present gains almost nothing from the shipped sink (0.0158 →
  0.0147, 0.0308 → 0.0290); calibrated without it, the sink is worth as much as it is
  worth to RTN (0.0363 → 0.0172, 0.0443 → 0.0293). The two gains are *not* independent;
  the hypothesis that calibration would leave the sink alone (position 0 is 1 % of the
  calibration rows, and its post-norm input is ordinary: norm 1.3× the median) was wrong,
  and the mechanism is in §5.
- **The sink is the cheaper half.** RTN with the shipped sink (0.0196 / 0.0316) is within
  ~25 % of GPTQ without it (0.0158 / 0.0308), for 62 KB and no calibration run; the
  calibration run is ~200 s per layer per 8-thread worker in numpy (measured: 191–234 s),
  ~22 min wall for a stream on one 64-core box with four workers, ~12 min across two.
- **Both together is the floor for this format**, 0.0147 / 0.0290, about 25 % under
  RTN + sink. The residual gap between the two calibrations once the sink is shipped
  (0.0147 vs 0.0172 on legal, 0.0290 vs 0.0293 on Roman) is within the difference
  between sets and is not resolved here.

**The quantizer is ours** (2026-09-17, tcdent: "we own our own quant path which has no
frills and max throughput and we understand it"). `bundle --calib models/deepseek-v2-lite/
calib.ids` runs the three calibration sequences through the bf16 transcription, keeps every
routed expert's input per MoE layer, and chooses the integers with `common::q4::gptq`
against `common::linalg::chol_of_inverse` of the damped Hessians: the layer's input Hessian
for gate/up (one factor per layer, shared by its 64 experts), each expert's own SwiGLU
hidden-state Hessian for down. Same scales, same format, no python. Verified against the
numpy reference stream two ways:

| anchor | reference (numpy GPTQ) | `bundle --calib` | 
|---|---|---|
| layer 10 reconstruction error ‖X(W−Q)ᵀ‖²/‖XWᵀ‖², gate/up, mean over 64 experts | 4.570e-4 | 4.570e-4 (each wins ~half the experts) |
| same, down | 9.756e-4 | 9.754e-4 |
| RTN on the same anchor, gate/up / down | 1.11e-2 / 1.39e-2 | — |
| teacher-forced KL, legal / Roman, sink shipped | 0.0147 / 0.0290 | 0.0153 / 0.0275 |

The reconstruction error is the quantity GPTQ minimizes, measured outside it
(`bringup/quant_check.py`), and is an equality-class check: two correct implementations
agree on it to the float path. Over all 26 layers the mean ratio is 1.000 on every layer
and both projections, and the reference wins 1,638 of the 3,328 expert-matrices. KL is the tolerance check behind it. Build time on the
32-core dev box (EPYC 7551P, 64 threads): 83 s of calibration forward (three sequences,
one thread each) + 964 s for the 1,664 experts = 17 min, byte-identical across runs.
The expert phase is ~9× below the idle single-thread rate (1.1 s per gate/up sweep,
2 s per 2048-factor): 64 threads on 512 KB L2 / 8 MB L3-per-4-cores are memory-bound
in the per-expert Cholesky trio (48 % of samples) and the block update (32 %). Next
levers, not taken: a blocked Cholesky, fewer threads per CCX.

## 4. Per-layer sensitivity

INT4 on one layer with bf16 elsewhere, as a share of that set's all-INT4 cost; and the
same sweep on the Roman-history set with position 0 kept at full precision.

| layer | code | dialogue | science | legal | KL only-L, sink bf16 |
|---|---|---|---|---|---|
| 1 | 0% | 1% | 1% | 1% | 9.7e-04 |
| 2 | 1% | 3% | 1% | 1% | 6.2e-04 |
| 3 | 22% | 49% | 45% | 41% | 7.5e-04 |
| 4 | 29% | 3% | 1% | 1% | 4.9e-03 |
| 5 | 1% | 2% | 1% | 1% | 2.5e-03 |
| 6 | 1% | 2% | 1% | 1% | 1.0e-03 |
| 7 | 0% | 2% | 2% | 1% | 9.9e-04 |
| 8 | 0% | 2% | 2% | 1% | 6.6e-04 |
| 9 | 0% | 2% | 1% | 1% | 7.9e-04 |
| 10 | 1% | 1% | 1% | 1% | 1.5e-03 |
| 11 | 1% | 1% | 1% | 1% | 7.6e-04 |
| 12 | 0% | 1% | 2% | 1% | 8.8e-04 |
| 13 | 0% | 1% | 1% | 1% | 2.6e-03 |
| 14 | 0% | 1% | 2% | 1% | 1.2e-03 |
| 15 | 0% | 1% | 1% | 1% | 8.8e-04 |
| 16 | 0% | 1% | 2% | 1% | 1.3e-03 |
| 17 | 0% | 1% | 1% | 1% | 1.3e-03 |
| 18 | 1% | 1% | 2% | 1% | 1.2e-03 |
| 19 | 0% | 1% | 2% | 1% | 1.5e-03 |
| 20 | 0% | 1% | 1% | 1% | 8.6e-04 |
| 21 | 0% | 1% | 1% | 1% | 7.6e-04 |
| 22 | 0% | 1% | 1% | 1% | 8.7e-04 |
| 23 | 0% | 1% | 1% | 1% | 1.1e-03 |
| 24 | 18% | 4% | 5% | 9% | 2.2e-03 |
| 25 | 3% | 6% | 6% | 6% | 8.1e-04 |
| 26 | 0% | 1% | 2% | 1% | 9.3e-04 |

Two different pictures. With the sink token quantized like everything else, layer 3 is
22–49 % of the cost in every domain, layer 24 matters for code (18 %) and layer 25 is
3–6 % everywhere; the other 22 layers are ≤ 3 % each. With the sink at full precision,
layer 3 falls to 7e-4 and the picture flattens: the remaining 0.031 nats is the sum of
26 contributions of 0.6e-3 to 5e-3 (their sum is 0.034 — near-additive), the largest
being layers 4, 5, 13 and 24. Nothing left to target by layer; what remains is the
per-expert rounding error everywhere, which is what calibration addresses.

## 5. Why layer 3: the sink token

The per-layer numbers alone suggested "keep layer 3 wide". Four further measurements
say otherwise.

**Injected error is not larger there.** Per-expert INT4 error on the observed inputs
(all 64 experts, 91 positions): layer 3 has a *lower* mean max-norm relative error
(0.17) than layers 10 (0.19), 13 (0.20) or 24 (0.19), and an absolute error L2 (1.3)
ten times smaller than layer 24's (13.9). Within layer 3 the spread is wide (0.02–0.22)
and the absolute error follows output magnitude (expert 54: 35 L2 output, 0.02 relative).

**Depth gain is real but points the wrong way.** A random perturbation of unit L2 norm
added to every routed expert output at layer L costs, in KL on the 10-position set:
L1 9e-04, L3 1e-04, L5 9e-04, L8 1e-05, L10 1e-05, L13 5e-06, L17 3e-06, L20 1e-06, L24 1e-06, L26 3e-06. Early layers amplify a thousand times more than late ones — and layer 3
is *less* sensitive to random error than layers 1, 4 or 5. Its INT4 error has the norm
of that unit probe and costs 1,300× more; layers 24–25's cost 10,000× more than a
random probe of their norm.

**The error is not in special channels.** The massive-activation channels (1262, 648,
106, 1101, 463 from layer 3 on; 1653 at layer 24 with residual magnitude 112) carry 13 %
of layer 3's FFN signal but under 1 % of its INT4 error. The error is diffuse across
output channels.

**It is one position.** Per-position INT4 error norm at layer 3: position 0 = 175,
every other position ≈ 0.3. The first token's FFN output there has norm 2,467 against a
median of 3.1: this is where the sink token's massive activation is created, and every
later position attends to it. An error of 175 in a state the whole sequence reads is
what 1,300× means. Excluding position 0, the layer-3 error is ordinary noise (mean-vector
ratio 0.18, random ≈ 0.10). Layer 24 has the same structure (position 0 = 162 vs median
3.4). And a probe confirms the class: a perturbation that points the *same* way at every
position costs 10× a random one of equal norm at layers 3 and 10 (1.35e-3 vs 1.3e-4;
9.9e-5 vs 1.0e-5).

**It is the weights, not the activations.** Splitting the recipe at layer 3: INT4
weights alone give the full error (2.27 L2 per position), INT8 activations alone a
tenth of it (0.22). The INT8 activation quantizer is not the problem anywhere measured.

So "layer 3 is sensitive" was "the sink token's state must be exact, and layer 3 is
where it is made". That is a property of the model (attention sinks and massive
activations at the first position are documented across many transformers), and the
right response is not a wider layer: it is to compute position 0 at full precision,
which costs nothing at runtime because the first token's state is a constant of the
model and recipe — BOS has no context. It can be computed once at bundle build and
shipped as the initial cache rows and residual.

**Why calibration finds it too.** GPTQ's Hessian for the down projection is over the
expert's real hidden inputs (the SwiGLU product), and position 0's hidden state at the
sink-creating expert is enormous. Share of trace(H) contributed by the three position-0
rows of the 257-row calibration set, over the 64 experts of a layer:

| layer | gate/up input (post-norm x) | down input (SwiGLU hidden): median expert | max expert |
|---|---|---|---|
| 3 | 0.019 | 0.020 | **0.997** (one expert) |
| 10 | 0.019 | 0.017 | 0.078 |
| 24 | 0.022 | 0.085 | **0.874** (one expert) |

At layer 3 one expert's down-projection Hessian is 99.7 % the sink direction, from
three rows in 257: GPTQ rounds that projection to be exact along it, which is the
sink fix applied to the one matrix that matters, by accident of the calibration data
beginning with BOS. Drop position 0 from calibration and the same expert is rounded
for the other 254 rows, and the recipe pays the full sink cost (§3). The gate/up
Hessians never see it (position 0's post-norm input is ordinary), so the protection
lives entirely in down projections. The shipped sink makes the recipe's quality
independent of whether the calibration set happened to contain the first position.

## 6. Activation ranges per layer (bf16 run, 91 positions, averaged)

`out` is the residual stream after the layer; `hn` the post-attention norm output (the
FFN input); `mlp.out` the FFN contribution; `o` the attention contribution.

| layer | out rms | out absmax | hn rms | hn absmax | mlp.out rms | mlp.out absmax | o rms | o absmax |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.09 | 2 | 0.06 | 0.9 | 0.04 | 0.4 | 0.053 | 1.13 |
| 1 | 0.11 | 3 | 0.29 | 1.5 | 0.04 | 0.4 | 0.032 | 0.21 |
| 2 | 0.16 | 4 | 0.31 | 2.1 | 0.07 | 0.9 | 0.040 | 0.32 |
| 3 | 0.78 | 15 | 0.35 | 3.0 | 0.67 | 11.6 | 0.043 | 0.24 |
| 4 | 1.00 | 21 | 0.39 | 4.1 | 0.27 | 6.1 | 0.049 | 0.39 |
| 5 | 1.07 | 23 | 0.42 | 4.1 | 0.13 | 2.4 | 0.069 | 0.82 |
| 6 | 1.13 | 24 | 0.44 | 4.6 | 0.13 | 2.6 | 0.066 | 0.96 |
| 7 | 1.17 | 25 | 0.48 | 4.8 | 0.13 | 1.8 | 0.070 | 1.38 |
| 8 | 1.22 | 26 | 0.53 | 5.1 | 0.13 | 1.7 | 0.093 | 0.70 |
| 9 | 1.29 | 29 | 0.53 | 5.3 | 0.15 | 4.0 | 0.081 | 0.90 |
| 10 | 1.34 | 29 | 0.57 | 5.9 | 0.17 | 3.5 | 0.117 | 2.61 |
| 11 | 1.47 | 34 | 0.61 | 6.6 | 0.23 | 5.9 | 0.143 | 1.52 |
| 12 | 1.53 | 33 | 0.60 | 5.7 | 0.21 | 2.8 | 0.157 | 3.53 |
| 13 | 1.64 | 33 | 0.65 | 5.9 | 0.31 | 2.0 | 0.146 | 1.64 |
| 14 | 1.74 | 36 | 0.68 | 5.9 | 0.25 | 2.9 | 0.122 | 0.69 |
| 15 | 1.83 | 38 | 0.71 | 6.2 | 0.24 | 3.2 | 0.106 | 0.86 |
| 16 | 1.94 | 39 | 0.74 | 7.0 | 0.31 | 2.5 | 0.117 | 1.36 |
| 17 | 2.07 | 41 | 0.79 | 7.8 | 0.37 | 2.7 | 0.118 | 0.81 |
| 18 | 2.22 | 42 | 0.85 | 8.1 | 0.38 | 4.7 | 0.150 | 1.97 |
| 19 | 2.37 | 44 | 0.92 | 9.5 | 0.44 | 5.3 | 0.107 | 0.59 |
| 20 | 2.58 | 52 | 0.99 | 11.4 | 0.44 | 8.3 | 0.201 | 2.46 |
| 21 | 2.80 | 62 | 1.01 | 11.9 | 0.46 | 7.9 | 0.225 | 3.94 |
| 22 | 3.08 | 75 | 1.07 | 12.3 | 0.54 | 10.3 | 0.254 | 3.17 |
| 23 | 3.58 | 99 | 1.16 | 14.2 | 0.78 | 19.8 | 0.292 | 3.94 |
| 24 | 3.71 | 118 | 1.27 | 19.9 | 1.22 | 35.6 | 0.238 | 2.56 |
| 25 | 4.40 | 166 | 1.13 | 19.5 | 1.59 | 50.0 | 0.356 | 11.03 |
| 26 | 6.04 | 217 | 1.07 | 22.9 | 2.85 | 67.9 | 0.280 | 7.31 |

The residual stream is quiet for three layers, then layer 3's FFN emits a contribution
five to ten times anything before it (absmax 12 vs ≤ 1) and the stream never returns:
its scale is set here. From layer 20 the FFN contributions grow again and layers 24–26
emit the largest activations in the network (absmax 36, 50, 68). These are the layers
the sensitivity sweep found. The mechanism is the obvious one: INT4 round-to-nearest
with a per-group absmax scale has a fixed *relative* error per group, and the layers
whose experts produce the largest values are the ones whose absolute error the
residual stream cannot absorb.


## 7. Routing: fragility and concentration

Relative margin between the sixth and seventh router probability, `(p6 − p7) / p6`,
over 2366 decisions (91 positions × 26 layers, bf16):

| threshold | fraction of decisions below it |
|---|---|
| 1e-2 | 9.0 % |
| 1e-3 | 1.0 % |
| min observed | 1.1e-5 |

One decision in ten within 1 % of flipping, one in a hundred within 0.1 %: any
implementation change that moves activations by 1e-3 changes the token sequence under
greedy decoding within a few positions on essentially every prompt. Hence cross-tier
determinism has to be identical bits (§2), and recipe quality has to be measured
teacher-forced (§3), never by comparing generated text.

Expert usage over the same positions (top-6 share of a layer's traffic; entropy in bits,
6.0 = uniform):

| set | mean top-6 share | mean entropy | most concentrated layers |
|---|---|---|---|
| Roman history | 0.27 | 5.5 | L13 (41 %), L16 (38 %), L7, L12 (35 %) |
| code | 0.24 | 5.66 | L11 (30 %), L14, L10, L12 |
| dialogue | 0.28 | 5.45 | L18 (37 %), L24 (35 %), L14, L9 |
| science | 0.33 | 5.32 | L18 (52 %), L16 (44 %), L20 (43 %) |
| legal | 0.28 | 5.52 | L15 (39 %), L16 (38 %), L13 |

Layer 3 is among the most evenly spread layers in every domain (63 of 64 experts used,
top-6 share 0.22), so its cost was never a few heavy experts. Concentration is
domain-dependent and moderate: no layer routes more than half its traffic to six
experts on any of these texts, and the concentrated layers differ by domain. For a
DRAM/VRAM placement policy that means an LRU over experts will find real locality (a
third of traffic on a tenth of experts in the concentrated layers) but not a fixed hot
set that survives a change of subject.

## 8. What this decides

- **The sink token is computed at full precision, for free.** Position 0's residual
  and cache rows for all 27 layers are constants of the model and recipe; the bundle
  builder computes them once from the bf16 checkpoint and ships them (27 × 576 f32 of
  cache: 62 KB). The node seeds the sequence from them and starts computing at
  position 1. *Implemented 2026-09-17: `sink.f32` in the bundle (`recipe = "… sink:f32"`),
  `Attention::seed` on both tiers, the node and the forward driver start at `FIRST_POS`;
  the CPU recipe with the shipped sink measures 0.0316 nats on Roman history and 0.0121
  on dialogue, matching the position-0-at-bf16 experiment. Ratified as part of the
  implementation, not a recipe option (tcdent, 2026-09-17: "token zero as part of our
  implementation is a go"): the recipe string no longer names it, the forward driver
  seeds by default, and `--no-sink` is the measurement of what it is worth.* This removes between a third and three quarters of
  the INT4 recipe's cost on ordinary text and all of layer 3's apparent sensitivity.
  Prompts always begin with BOS, so this is not a special case in the token loop; it
  is the initial state.
- **No layer is kept wide.** With the sink handled, the remaining cost is diffuse
  (§4): 26 contributions of the same order. The lever for it is calibrated INT4,
  applied everywhere. INT8 activations are not the problem and stay. *Measured
  2026-09-17 (§3): GPTQ over the same format takes RTN + sink from 0.0196 / 0.0316 to
  0.0147 / 0.0290 on the two held-out sets, ~25 %; most of what GPTQ buys over RTN on
  its own is the sink, which the shipped rows already pay for. The sink stays in the
  recipe regardless of calibration: it is free, explicit, and removes the recipe's
  dependence on the calibration set's first position. Decided 2026-09-17 (tcdent, "let's
  wire it up to our model implementation"): the calibrated stream is the shipped recipe,
  `experts:q4g128-a8-gptq`, built by `bundle` by default from the model's `calib.ids`;
  `--rtn` is the twin kept for pricing. The bundle is `bundles/v2lite-q4gptq`.*
- **Precision per expert is not needed yet.** The per-expert error spread within a
  layer (0.02–0.22) is real, but with the sink handled no layer's cost justifies it.
  The bundle's per-record layout keeps the option open at the cost of an offset table.
- **Placement follows usage, precision follows sensitivity, and they are different
  maps.** Usage concentration is what an LRU exploits and it moves with the domain.
  Sensitivity, after the sink, is flat. Neither is the "first layers wide" convention.
- **Per-layer sensitivity is re-measured for every checkpoint, with the sink
  excluded.** With it included the sweep says "layer where the sink is made"; without
  it the sweep says what the recipe actually costs where.

- **The sampler is the checkpoint's recipe as transformers ran it** (2026-09-18).
  `generation_config.json` says `temperature 0.3`, `top_p 0.95` and names transformers
  4.39.3; that version's `GenerationConfig` defaults `top_k` to 50 and applies the warpers
  in the order temperature, top-k, top-p, then a multinomial draw. All three are constants
  of the model (`SAMPLING`); `top_k` carries its provenance in the comment because the
  config does not say it. Semantics, grounded on the warpers' code: top-k keeps every
  logit at or above the k-th largest (ties included), top-p sorts descending, keeps tokens
  while the mass above them is under p of the survivors' mass and keeps the token that
  crosses (llama.cpp's `top_p` keeps the crossing token too; its defaults differ,
  top-k 40, and it applies temperature *after* the truncations, which is a different
  distribution; vLLM applies top-k then top-p like transformers). The CPU op and the card's
  kernel compute one canonical candidate list (value desc, index asc, capped at 256) and
  the same f64 walk. Checked against the warpers themselves on every golden position,
  `bringup/sample_check.py`: 435 of 435 tokens agree (257 calibration + 178 held-out),
  each drawn with that position's own uniform. Nucleus size on those logits: median 1,
  mean 1.5–1.6, p99 5–6, max 17; one token at 61–74 % of positions. The kernel against
  the CPU op on the same 435 rows plus seven synthetic shapes (all equal, 300 tied at the
  top, ties exactly at the k-th, the top at both ends): 442 of 442, `bench-draw`.
- **Token streams on the CPU and on the card are not expected to match** (measured
  2026-09-18, and recorded because it was assumed otherwise): the same prompt drew
  `8913, 11, 254` on the card and `8913, 11, 285` on the CPU, and the sampler was not
  the cause (the CPU op on the card's logits draws 254). The float ops differ at the
  ulp between tiers (reduction order), the residual at the first computed position
  differs by 3e-7, and the INT8 activation quantizer turns an ulp into a whole
  quantum whenever a value sits on a rounding boundary: by position 2 the residual
  differs by 1e-3 from layer 9 on, by position 4 a routing decision has flipped, and
  the logits differ by 0.2–0.5. Each tier is deterministic and each op is gated
  against its twin; the streams are compared only through the ops.

- **The rope tables at long positions** (2026-09-19). Computing YaRN's inverse frequencies
  from the config in f64 and rounding differed from the oracle's f32 arithmetic by an ulp
  in 12 of the 32; invisible at the goldens' positions, 0.0038 in a cosine at position
  100,000, since the angle is the position times the frequency. The 32 values are now a
  constant of the model, the oracle's own, and `construct::tests` checks the tables
  against the oracle at positions up to the checkpoint's 163,840.
- **Attention at long context is the head-merged kernel** (2026-09-19, for the 100k
  target). MLA's absorbed form shares one 576-float latent row across the 16 heads, and
  the first kernel read the cache once per head: 2 GB a token at 2,048. The kernel now
  stages a chunk's rows in LDS four at a time, each row read from memory once per token,
  and every wave takes two heads over them with an online softmax (the running max
  rescales the sum and the output, as the merge does across chunks); the absorbed
  queries are a launch of their own per (token, head); the merge is parallel over waves.
  A lane owns latent dims strided by 32 so the staged reads are conflict-free. Attention
  per token on the 7900 XT: 2,048 → 3.5 ms (was 4.1), 16k → 8.1, 65k → 21.7, 100k →
  25.8 ms at 241 GB/s of cache rows against a 763 GB/s floor of 8 ms. The remaining
  factor is the compute loop, ~2,000 cycles a row against ~800 of issue: LDS-latency
  chains at four workgroups a CU. The reduction order changed, so the decode stream
  changed at position 4; the prompt gate holds (162/162) and the recipe's error against
  the reference is unchanged.

## 9. Not yet measured

- **Sinks that emerge later in a context** (tcdent, 2026-09-17: "will we find sinks
  that emerge throughout longer contexts? … a way that we can proactively determine
  them, or perhaps when we cross a threshold of a token sink we recompute it in a
  higher precision"). The literature reports secondary massive activations at
  delimiter tokens in some models; on our five ≤ 91-position sets the only outlier is
  position 0. Two things to try at 1k–2k context: (a) detect proactively — the
  magnitude of a position's FFN output at the sink-creating layer (2,467 vs 3 here) is
  a threshold the dispatcher can read as the row is written; (b) recompute a flagged
  position's FFN at full precision before its cache row is committed. Both are
  position-level precision, the per-position switch in the forward driver is the
  measurement tool, and the cost is one token's bf16 expert reads per occurrence.
  *Future project.*
- Calibrated INT4 on the remaining three sets (code, dialogue, science are the
  calibration data; only legal and Roman are held out so far) and the 10-position tail.
- **Activation group width.** The INT8 activation is quantized in groups of 128 to match
  the weight groups; a finer width (32 or 64) would need its own flush cadence in the
  expert kernels and its own quality price. tcdent, 2026-09-18: not below 128 for now;
  pinned as a later exploration. Its quality side, measured 2026-09-18 on the calibration
  activations (6,682 position-layer rows, relative INT8 error ‖x−sq‖²/‖x‖², mean / max):
  group 32 4.4e-5 / 1.7e-4; 64 6.8e-5 / 3.5e-4; 128 1.1e-4 / 6.3e-4; whole row 7.2e-4 /
  4.3e-3. Halving the group takes a third off an error that is already two orders under
  the INT4 weights' (§2), so 64 buys nothing that shows in the KL, and it costs the expert
  kernels a flush every 64 instead of 128. The latency side is not an argument either
  way: the router's quantize is a dependent chain per wave whose depth does not change
  with the group, only the number of groups. The weight group's own trade-off, measured
  2026-09-18 on layer 10's 64 experts (INT4 RTN, ‖W−sQ‖²/‖W‖² over all 192 matrices;
  f32 scale bytes per weight byte): 32 9.6e-3 / 0.25; 64 1.18e-2 / 0.125; 128 1.41e-2 /
  0.062; 256 1.65e-2 / 0.031; 512 1.89e-2 / 0.016; whole row 2.40e-2 / 0.004. Each
  doubling costs ~17 % more weight error and halves the scale traffic; 128 is where the
  scales are 6 % of the stream and the error is within 1.5× of the finest group. The
  quantize's placement is fixed by the same width:
  the first workgroup that holds 128 consecutive activations is the router's, so it
  cannot move earlier (the residual is written 8 rows per workgroup), and moving it into
  the shared experts' rows kernel trades ~7 µs of quantize for ~10 µs of lost overlap.
  Two reformulations were priced (2026-09-18) and neither is taken:
  - *Quantize before the norm's scalar.* The integers are invariant to the rmsnorm scale
    r (it multiplies every value and the group's absmax alike), so the quantize could run
    from `ln·h` before the workgroup reduction that produces r, on both paths identically.
    The router with the norm stamped apart: norm 4.8 µs, rows 10.0, softmax+pick 4.8,
    quantize 7.6 (stats build, 27 µs total). The quantize does not wait on the norm today;
    it waits on rows, softmax and pick because it is last. So the reorder alone hides
    only the reduction it stops depending on, under 2 µs a layer, ~0.5 % of the token;
    the ceiling of any placement is the whole phase hidden under the rows' memory
    latency, 26 × 7.6 = 196 µs, 3.2 % of a 6.12 ms token, and that needs waves
    partitioned between rows and quantize in a kernel whose rows phase already uses all
    of them. Not worth its restructuring.
  - *Multiplies for the two IEEE divisions* (`amax/127`, `1/s`). Run both ways over every
    expert input in the calibration sets (6,682 position-layer activations, 13.7 M
    integers, `post_attention_layernorm.out` of layers 1–26): `127/amax` changes 13
    integers (1 in 10⁶), one per affected activation, all by ±1, in 0.2 % of the
    activations; a hardware reciprocal one ulp off changes 45 (3 in 10⁶) in 0.7 %. Over
    a 2,048-position context that is roughly one position in twenty with one INT8 off
    by one in one layer, none of them a routing input (the route picks from f32 and the
    quantize follows), but each a seed for the CPU and card streams to part later. The
    divisions are a couple hundred of the ~11,000 cycles a group costs, about 4 µs a
    token. Bit-parity between the paths is the check everything else stands on; it is
    not sold for 0.06 %.
- On the radar (tcdent, 2026-09-17): Intel's AutoRound (learn each weight's up/down
  rounding by signed gradient descent over a block on calibration data — the next method
  above GPTQ; needs a block forward, which the transcription has) and llm-compressor's
  MoECalibrationModule (route every calibration token to every expert during
  calibration so each expert's statistics see the full set — what `bundle --calib`
  already does by construction, since each expert is calibrated on the layer's whole
  input matrix rather than on its routed tokens).
- Attention precision: KV cache width (f32 today), q/kv projections at INT8 — and
  whether the sink token's cache rows need to stay f32 when the rest narrows (likely,
  by the same mechanism).
- Longer contexts: all sets are ≤ 91 positions; the sink mechanism should be
  re-checked at 1k–2k.
- Per-expert error over many more positions, to see whether the within-layer spread
  is stable enough to price per-expert precision if a future recipe wants it.
- The float-kernel order contract (§2), and the placement matrix at 0 in every cell
  once it lands. Until then the CPU and card token streams part within a few
  positions (§8), which is the INT8 quantizer amplifying ulps, not a bug in either.
- **The `sample` kernel's time.** Its four stamped phases sum to ~117k shader cycles on
  the slowest wave (max pass 47k, θ₀ 22k, count/scan/gather 31k, order and walk 17k),
  about 45 µs at the clock the other kernels run at, yet the timeline brackets it at
  88 µs inside the token (unchanged from the previous kernel; head + draw 0.68 ms either
  way). Either a lone single-workgroup kernel runs at a lower clock or the dispatch
  bracket carries ~40 µs of its own; not yet separated. The max pass is VRAM latency
  with one workgroup (twenty loads in flight per thread bought 55 → 18 µs); the rest is
  LDS latency chains.

## Regenerating

```
# baselines, sweeps, probes (host, bringup/forward); G = a goldens dir, B = the bundle
./target/release/forward --snapshot $S --goldens $G --bundle $B --q4-layers none --quiet --stats stats.tsv --usage usage.tsv
./target/release/forward --snapshot $S --goldens $G --bundle $B --q4-layers all  --quiet             # the shipped path: sink seeded, from FIRST_POS
./target/release/forward --snapshot $S --goldens $G --bundle $B --q4-layers all  --no-sink --quiet   # position 0 under the recipe (the sink's worth)
for L in $(seq 1 26); do ./target/release/forward … --q4-layers $L --quiet; done                      # xargs -P for parallel
./target/release/forward … --q4-layers none --noise L:1.0[:fixed] --quiet                            # gain / bias probes
./target/release/forward … --q4-layers 3 --q4-experts 3:5,17 --quiet                                 # per-expert masks
# per-expert error, channel analysis, bias ratio: numpy over the goldens + checkpoint (see the story log)
# placement matrix (lasenora): node rockem.debug=<dir> per placement, diff pos<N>.f32 / pos<N>.layers.f32
# goldens: bringup/goldens.py --decode 60 --prompt "…"   (five domain sets exist under bringup/goldens/)
# calibrated stream (shipped path): bundle --snapshot $S --out bundles/<name> --calib models/deepseek-v2-lite/calib.ids [--experts-only]
#   reference: bringup/quant.py --snapshot $S --out bundles/<name> --calib code,dialogue,science [--exclude-pos0] --layers a-b
#   check one layer against it: bringup/quant_check.py --snapshot $S --layer 10 --streams rtn=… ref=… rs=…
# then: forward … --bundle bundles/<name> --q4-layers all [--no-sink] --quiet   on the held-out sets
# sink share of the GPTQ Hessian per expert: numpy over goldens + checkpoint (see the story log)
```
