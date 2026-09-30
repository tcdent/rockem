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

To measure on the golden sets. Published on the NVFP4-expert, FP8-dense recipe (repo,
greedy, thinking off): GSM8K (250) 97.2 %, HumanEval 156/164, needle 12/12 to 507k,
unchanged from their stock. Their speculative acceptance runs 88–97 % on structured
text, 58–75 % on code, 24–38 % on prose: the speculation multiplier is a property of
the domain, and the recipe's effect on it is a measurement of its own.

## 4. Per-layer sensitivity

To measure with the per-layer switch in the forward driver. Three dense layers first,
then alternating blocks of three KDA layers and one MLA layer; only the eleven MLA
layers carry a cache, the rest a recurrent state.

## 5. Sinks

Unknown. Nothing published on this model mentions massive activations. The first probe
is position 0's residual and FFN-output magnitude per layer on the golden sets, and
whether a shipped position-0 state is even meaningful when 34 layers carry a recurrent
state rather than cache rows.

## 6. Activation ranges per layer

To measure on a bf16 run over the golden sets: residual, normed input, expert input,
expert output, the KDA state's range, and the indexer's scores.

## 7. Routing: fragility and concentration

To measure: the routing histogram per layer over the golden sets and a longer corpus,
the margin of the eighth pick against the ninth, and how often INT4 experts flip a pick.
The histogram is also the first input to expert placement between the cards and the
sockets' DRAM.

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
2. An oracle that runs on hardware we have, for at least the first eight layers (three
   dense, four KDA, one MLA): the transcription gate for the new kernels.
3. Goldens: the five domain sets at this model's tokenizer, with per-layer edges.
4. The routing histogram per layer (§7).
5. Position-0 magnitudes per layer (§5).
6. Activation ranges (§6), then the recipe sweeps of §2–§4.

## Regenerating

Nothing yet. The tools this document needs, in the order they are needed: a byte table
from the index (`bringup/`), the oracle (`bringup/forward` for `glm5_next`, or the
published per-layer references), goldens (`bringup/goldens.py` for this architecture),
then the sweeps.
