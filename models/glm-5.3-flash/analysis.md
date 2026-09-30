# GLM-5.3-Flash — numerical analysis

Measured behaviour of this model under our implementation, kept next to the model so
that recipe and kernel decisions are made from these tables rather than from best
practice. The implementation does not exist yet, so the tables do not either: each
section says what it will hold and which command produces it, and holds what is known
today from the checkpoint's `config.json` (zai-org/GLM-5.3-Flash, read 2026-09-30) and
from published measurements (kindlingai/glm-5.3-flash-gx10, vLLM on GB10, read
2026-09-30), each marked. Present tense describes the model; dated notes will mark what
was decided from what.

> "This is just a standalone GLM 5.3 Flash analysis document in the same format as
> V2-Lite that helps support our implementation" — tcdent, 2026-09-30

## 0. The model

`Glm5NextForConditionalGeneration`, text model `glm5_next_text`, bf16, about 320B
parameters of which about 18B are active per token (repo).

**Layers.** 45. Thirty-four are KDA linear attention (64 heads of 128, a short
convolution of kernel 4, a gate with lower bound −5), keeping one recurrent state per
sequence and no cache. Eleven are DeepSeek sparse attention, at layers 3, 7, 11, …, 43:
MLA with `kv_lora_rank` 512, `q_lora_rank` 1536, 64 heads, 256-wide keys and values,
no rotary half (`qk_rope_head_dim` 0, `mla_use_nope`), and an indexer (32 heads of 128,
`index_topk` 2048, k-pool 4 with compression, the tail always selected) that chooses
which keys each query attends. The residual stream is mHC (`hc_mult` 4, Sinkhorn 20
iterations). Context 1,048,576 positions. Vocabulary 154,880, untied embeddings.

**FFN.** The first three layers are dense SwiGLU, hidden 4096 to 12,288. The other 42
are MoE: 288 routed experts of 2048 plus one shared expert, 8 routed per token. The
router runs in f32 with a sigmoid score and a learned bias per expert (`noaux_tc`): the
bias is added for the selection, the weights are the sigmoid without it, normalized
over the eight (`norm_topk_prob`) and scaled by 2.5. No group limiting. The expert
activation carries a clamp (`swiglu_limit` 10.0).

**Extras.** One MTP head (`num_nextn_predict_layers` 1). A vision tower
(`glm5_next_vision`, hidden 1024) the text path does not need.

**Checkpoints** (sizes from the safetensors indexes, repo): bf16 598.5 GiB; FP8 305.8;
NVFP4 181.3 (routed experts at 4 bits, everything else bf16).

**Bytes, from the shards' headers** (`bringup/glm/bytes.py --md`, all 62 shards of the
official FP8 checkpoint, 2026-09-30; the 4-bit column is parameters at 4 bits plus a
scale per 128):

| class | tensors | parameters | at the checkpoint's dtypes | at 4 bits |
|---|---|---|---|---|
| routed experts (42 layers + the MTP layer's, 288 each) | 37,152 | 311.65B | 311.7 GB, FP8 | 160.7 GB |
| KDA projections (34 layers) | 476 | 3.54B | 7.1 GB, bf16 (f32 gates) | 1.8 GB |
| attention `o_proj`, KDA and MLA | 46 | 1.95B | 3.1 GB | 1.0 GB |
| `lm_head` | 1 | 0.63B | 1.3 GB, bf16 | 0.3 GB |
| embeddings | 1 | 0.63B | 1.3 GB, bf16 | 0.3 GB |
| vision tower (not run) | 347 | 0.56B | 1.1 GB, bf16 | — |
| shared experts | 129 | 1.08B | 1.1 GB, FP8 | 0.6 GB |
| MLA projections (11 layers) | 72 | 0.60B | 0.8 GB | 0.3 GB |
| dense FFNs (layers 0–2) | 9 | 0.45B | 0.5 GB, FP8 | 0.2 GB |
| MLA indexer | 84 | 0.09B | 0.2 GB, bf16 | 0.0 GB |
| router, mHC, MTP head, norms | 453 | 0.12B | 0.3 GB | — |
| **total** | | **321.3B** | **328.3 GB** | |

What a token reads: 8 routed experts in 42 layers = 336 experts, 25.2M parameters each,
**8.5B parameters, 4.4 GB at 4 bits with scales**, plus the shared experts and every
non-expert tensor (≈9B parameters without the vision tower, **≈18 GB at bf16**, 16 GB at
the checkpoint's mix). The routed experts are 97 % of the parameters and none of the
per-token latency except through the bus that streams them.

## 1. What "correct" means here, and against what

Three anchors, each with one job, none reducible to another:

| anchor | question it answers | criterion | tool |
|---|---|---|---|
| the scalar twin of a kernel | is the fast realization implemented right | equality, byte for byte | in-package tests, to write with each kernel |
| the oracle | is the model transcribed right | tolerance (f32 order noise) | `forward` vs goldens |
| the oracle, teacher-forced | is a recipe a faithful interpretation of the weights | KL(oracle‖ours) per position, top-1 agreement, oracle-token NLL | `forward --quiet` |

**The oracle is the first thing to decide.** A pinned transformers modeling file for
`glm5_next`, run in f32 on the CPU, is the natural one; at 320B parameters that is
about 1.2 TB as f32, so it has to stream layers from the bf16 checkpoint on disk rather
than hold them, or run a prefix of layers only. The published per-layer references in
the gx10 repo's `dev/kernel-tests` (they match vLLM's decode path) are a second anchor
for the layer kinds that are new. Token identity is not a quality measure here either:
the router's top-8 and the greedy decode are step functions, and teacher-forced
divergence is what is measured.

## 2. Noise floors by source

To measure once the oracle runs. Prior from published measurements, marked:

| source | magnitude | measured how |
|---|---|---|
| f32 reduction order | expected ~1e-7 a layer; 45 layers and mHC mixing the stream — to measure | `node rockem.debug=` logits/layer diff across placements |
| INT8 activation, group 128 | ~1e-3 per element (absmax/127) | by construction |
| INT4 routed experts, calibrated | to measure; the NVFP4 checkpoint is the public point of comparison | teacher-forced KL |
| FP8 dense linears | +0.5 % NLL on prose (repo: per-channel weight scales, per-token activation scales) | their `quality/` |
| NVFP4 on `in_proj`, `o_proj`, shared experts | about as much again (repo) | — |
| NVFP4 on every dense layer | 2–3× the FP8 cost (repo) | — |
| routing flips induced by the above | to measure | per-layer trace, first-flip layer per position |

## 3. Teacher-forced divergence from the oracle

Measured 2026-09-30, every layer's routed experts INT4 round-to-nearest, absmax/7 in
groups of 128 along the input, no calibration, the shared experts and everything
dense at the checkpoint's precision; the passages without the model's prefix, so the
sink is the first word (§5) — `oracle.py --q4 all` against the plain run,
`measure.py kl`:

| passage | positions | KL(oracle‖INT4) mean / max, nats | top-1 agreement | NLL of the text, oracle → INT4 | layers where some pick changed |
|---|---|---|---|---|---|
| code | 75 | 0.040 / 0.98 | 0.960 | 0.465 → 0.460 | 41 of 42 |
| dialogue | 77 | 0.055 / 0.98 | 0.935 | 1.171 → 1.292 | 41 |
| science | 76 | 0.051 / 0.38 | 0.950 | 1.213 → 1.178 | 41 |
| legal | 80 | 0.081 / 3.66 | 0.892 | 1.057 → 1.015 | 41 |
| long | 88 | 0.064 / 3.66 | 0.966 | 0.981 → 0.960 | 41 |

Plain INT4 costs 0.04–0.08 nats of KL per position on these passages, with a tail
(legal and long share their first position, "The", at 3.7 nats), and moves the greedy
token at 3–11 % of positions. The text's NLL moves both ways by less than the passages can resolve.
Every MoE layer sees at least one pick change on 75 positions, as §7's margins
predicted. For reference, V2-Lite's plain INT4 measured 0.03–0.05 nats before its sink
was handled and 0.015–0.03 after, on the same passages; what the sink is worth here
is the measurement running now (§5).

## 4. Per-layer sensitivity

To measure with the per-layer switch in the forward driver. Three dense layers first,
then alternating blocks of three KDA layers and one MLA layer; only the eleven MLA
layers carry a cache, the rest a recurrent state.

## 5. Sinks

Measured 2026-09-30 on the five passages (`measure.py ranges`, and a probe with the
model's own prefix).

**Without the prefix** (the passage's first word at position 0): position 0 is a sink
of moderate size from layer 3 to the mid-thirties — its residual is the largest of any
position's by 4–6× at layers 3–9, 15–30× at layers 10–25, 3–6× from 34 on, with an
absolute value of 10–28 over a stream whose rms is 0.05–1.4 (mHC keeps the stream
small). The last layer is a different thing: its residual reaches 70–185 on every
position, and it is the shared expert, whose output there is 115–167 against ≈1 in
every other layer. That is the model shaping the final norm's input, not a sink.

**With the prefix `[gMASK]<sop>`**, which every prompt to this model carries (the chat
template opens with it), **the `<sop>` token is a massive sink and the first word is
not**: its residual is 260 at layer 18, 1,050 at layer 25, 2,600 at layer 34, **5,114 at
layer 44**, against 0.2–83 for the passage's positions; `[gMASK]` beside it reaches 196.
The passage itself scores better with the prefix in place (NLL 0.465 → 0.437 on code).

| layer | first word as position 0, plain | `[gMASK]` | `<sop>` | first word, with the prefix | the rest |
|---|---|---|---|---|---|
| 10 | 1.8 | 10.1 | 12.1 | 0.4 | 0.2 |
| 18 | 5.2 | 9.7 | 260 | 0.5 | 0.4 |
| 25 | 5.5 | 17.4 | 1,050 | 0.7 | 0.7 |
| 34 | 6.2 | 35.3 | 2,592 | 1.8 | 3.4 |
| 44 | 19.0 | 196 | 5,114 | 8.2 | 83 |

So the sink is exactly V2-Lite's shape, two positions long and a constant of the model:
the prefix is always the same two tokens, so their state after position 1 — the 34 KDA
layers' recurrent states and the 11 MLA layers' two cache rows — is computed once at
full precision and shipped, and every sequence starts at position 2.

**What that buys** (code passage with the prefix, every layer's routed experts INT4
round-to-nearest, the two prefix positions' experts at full precision — `--q4 all --sink
2` — against the prefixed oracle run): **KL 0.0070 nats a position, max 0.097, top-1
agreement 0.974, NLL +0.004**, against 0.040 / 0.98 / 0.960 for the same passage and
recipe without the prefix (§3). The same recipe with the prefix but the sink positions
quantized too (`--sink 0`) is running, to separate the prefix's own effect from the
protection's.

## 6. Activation ranges per layer

Measured 2026-09-30, max over the five passages (`measure.py ranges`; the full
per-passage tables are in `bringup/goldens/glm/<set>/`):

| layers | residual max | residual rms | FFN input max (post-norm) | routed experts' output max | shared expert's output max |
|---|---|---|---|---|---|
| 0–2 (dense) | 0.0–0.1 | 0.00–0.02 | 0.4–1.5 | — | — |
| 3–9 | 1.2–1.5 | 0.05 | 1.3–2.6 | 0.7–3.4 (13.9 at layer 3) | 0.3–0.7 (9.5 at 3, 12.6 at 6, 4.4 at 7) |
| 10–17 | 2.9–5.2 | 0.09–0.16 | 1.3–1.8 | 0.7–7.9 | 0.3–1.0 (9.3 at 10) |
| 18–28 | 9.9–10.9 | 0.31–0.38 | 1.3–1.8 | 0.5–2.9, **37 at 18**, 16 at 21, 13 at 26 | 0.3–1.4 |
| 29–33 | 16.2–17.2 | 0.41–0.48 | 1.9–2.2 | 1.2–12 (12 at 29) | 0.4–0.8 |
| 34–43 | 20.6–28.5 | 0.62–1.04 | 2.0–4.2 | 1.1–7.2 | 0.6–3.8 (13.5 at 43) |
| 44 | 185 | 1.41 | 5.6 | 34.5 | **167** |

The residual grows in steps at layers 10, 16/18, 29 and 34, each a layer whose routed
experts emit an outlier (7.7, 37, 12, 4.2) — the routed experts write the big values, the
stream keeps them. The INT8 activation (absmax / 127 per group of 128) sees FFN inputs
of 1.3–5.6: an ordinary range, no outlier channel at the input of the experts.

## 7. Routing: fragility and concentration

Measured 2026-09-30 over the five passages together, 395 positions, 8 picks each
(`measure.py routing`, and the combined histogram saved as
`bringup/goldens/glm/routing_hist.pt`):

| layers | experts with any pick, of 288 | experts holding 50 % of picks | 90 % | largest single expert's share | positions whose 8th and 9th scores differ by < 1e-3 |
|---|---|---|---|---|---|
| 3–11 | 279–287 | 60–90 | 186–219 | 1–4 % | 5–12 % |
| 12–17 | 255–268 | 37–56 | 145–175 | 3–5 % | 6–8 % |
| 18–28 | 221–255 | 25–35 | 110–141 | 3–9 % | 4–8 % |
| 29–44 | 248–265 | 30–45 | 142–158 | 3–6 % | 3–10 % |

Three things to carry into the design:

- **Concentration is real but layer-dependent.** Early layers spread picks over most of
  the 288; the middle of the network routes half its traffic through 25–35 experts a
  layer. A card-resident set of the 50 %-of-picks experts would be roughly 2,000 experts
  across the 42 layers, about 25 GB at 4 bits — the first number for the placement table's
  frequency axis. Five passages are too few to fix the set; the histogram wants a corpus.
- **Picks are fragile.** The router's sigmoid scores sit close: 3–12 % of positions per
  layer have their eighth and ninth candidates within 1e-3, and the median margin is
  4e-3 to 1.6e-2. Noise of INT4's size on the residual will flip the last pick on a few
  percent of positions per layer; §3 measures what that costs.
- **The pick is a rank over sigmoid + bias, the weights the sigmoid alone, normalized.**
  Our route kernel's rank-based selection carries over; the score and the weights are
  new arithmetic.

## 8. What this decides

Nothing yet. The decisions the model forces, with what is known:

- **Where the dense weights live.** ≈18 GB bf16 (§0) fits two 7900 XTs only split by
  layer, with the residual crossing between the cards once per token; FP8 dense (≈9 GB)
  fits one card at the published +0.5 % NLL. The rule that dense weights stay bf16 was
  set on another model's numbers; here it has a measured price and is reopened.
- **The routed experts stream from DRAM.** ≈4.2 GB a token at 4 bits. At the bandwidth
  the production box measures, the expert phase is on the order of 10 ms a token before
  speculation — arithmetic, unmeasured.
- **Speculation is the multiplier.** A step's cost is a fixed part, a part per expert
  touched and a part per token (repo's scheduler); verifying k drafts costs the union of
  their experts, not k times one. On a DRAM-streamed MoE that is the whole argument for
  speculative decoding. The checkpoint's own MTP head is the licensed path; the DFlash2
  drafter is non-commercial.
- **Recurrent state and the context store.** A stored context is 34 layers' states at
  checkpoints plus the 11 MLA layers' rows, and speculation needs one state per request
  with accepted tokens replayed through the stock kernel, bit-exact (the repo's
  RecoverSSM shape). The store is designed around rows; this is a design change.
- **Kernels.** MLA with a 2048-key sparse selection and no rotary half; the sigmoid-bias
  router; KDA (a chunked recurrent kernel with a short convolution and a gated delta
  rule); the indexer; mHC; the MTP head; a 154,880-row head. The expert kernels' INT4
  group / INT8 activation shape applies with wider rows, and prefill on the CPU experts
  must read each expert once per step across the tokens that picked it.
- **Thinking off is outside the training** (repo, every checkpoint): long outputs
  corrupt; low reasoning effort is the substitute, and the chat template carries it.

## 9. Not yet measured

In order:

1. ~~The safetensors index read into a byte table by tensor class~~ — done, §0.
2. ~~An oracle that runs on hardware we have~~ — done, all 45 layers: `bringup/glm/oracle.py`,
   the reference modeling code in f32 on the CPU, a layer at a time from the FP8
   shards, about 18 minutes a passage on elsenor.
3. ~~Goldens: the five domain sets at this model's tokenizer, with per-layer edges~~ — done,
   `bringup/goldens/glm/<set>/`.
4. ~~The routing histogram per layer (§7)~~ — done on five passages; a corpus next.
5. ~~Position-0 magnitudes per layer (§5)~~ — done, and the prefix probe: the sink is `<sop>`.
6. ~~Activation ranges (§6)~~ — done; the recipe sweeps of §2–§4 are running.

## Regenerating

Nothing yet. The tools this document needs, in the order they are needed: a byte table
from the index (`bringup/`), the oracle (`bringup/forward` for `glm5_next`, or the
published per-layer references), goldens (`bringup/goldens.py` for this architecture),
then the sweeps.
