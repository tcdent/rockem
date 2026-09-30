# GLM-5.3-Flash — candidate analysis

The candidate for the next implementation, kept in the shape of
[`../deepseek-v2-lite/analysis.md`](../deepseek-v2-lite/analysis.md) so that the same
questions get the same kind of answer before a kernel is written. Nothing here is
measured by this repository yet: the numbers come from the checkpoint's `config.json`
(zai-org/GLM-5.3-Flash, read 2026-09-30) and from kindlingai/glm-5.3-flash-gx10 (vLLM on
GB10 boxes; their `NOTES.md` and `experimental/README.md`, read 2026-09-30), each marked.
Present tense describes the model; the sections that the V2-Lite document fills with
tables say here what the table will be and which tool produces it.

> "Do you want to start stubbing out the same kind of analysis file for this model, so
> that we can explore it as a candidate for our next implementation?" — tcdent, 2026-09-30

## 0. The model, against V2-Lite

| | DeepSeek-V2-Lite | GLM-5.3-Flash (`glm5_next_text`) |
|---|---|---|
| parameters, total / active | 15.7B / 2.4B | ~320B / 18B (repo) |
| layers | 27, all MLA | 45: **34 KDA linear attention + 11 sparse MLA** (`layer_types`; MLA at 3, 7, 11, …, 43) |
| hidden / dense FFN | 2048 / 10944 | 4096 / 12288 (`first_k_dense_replace` 3) |
| routed experts / active / shared | 64 / 6 / 2 | **288 / 8 / 1**, `moe_intermediate_size` 2048 |
| router | softmax, top-k | **sigmoid, `noaux_tc`** (a bias per expert), `norm_topk_prob` true, scale 2.5, router in f32 |
| MLA | `kv_lora_rank` 512, rope 64 | `kv_lora_rank` 512, `q_lora_rank` 1536, 64 heads, nope 256 / rope **0** (`mla_use_nope`), v 256; a **NoPE indexer** (32 heads, dim 128, `index_topk` 2048, k-pool 4) picks the keys |
| linear attention | — | KDA: 64 heads × 128, short conv 4, one recurrent state per sequence, gate lower bound −5 |
| residual stream | plain | **mHC** (`hc_mult` 4, Sinkhorn 20 iterations) |
| context | 163,840 | 1,048,576 |
| vocabulary | 102,400 | 154,880 |
| speculation | — | MTP head (`num_nextn_predict_layers` 1); a separate DFlash2 drafter exists (CC BY-NC-ND) |
| modality | text | text + vision tower (`glm5_next_vision`, hidden 1024) |
| checkpoints | bf16 | bf16 598.5 GiB; FP8 305.8; NVFP4 181.3 (routed experts 4-bit, rest bf16) — sizes from the safetensors indexes (repo) |

Arithmetic from the config, to be replaced by the index's byte counts:

- one routed expert = 3 × 2048 × 4096 = 25.2M parameters; 42 MoE layers × 288 = 12,096
  experts ≈ 305B; a token touches 8 × 42 = 336 of them ≈ 8.5B parameters ≈ **4.2 GB at
  4 bits** plus the shared expert (≈1B, 0.5 GB at 4 bits or 2 GB bf16)
- everything that is not a routed expert (KDA and MLA projections, the indexer, the three
  dense FFNs, embeddings, `lm_head`, mHC) ≈ 9B parameters ≈ **18 GB bf16** — *estimate;
  the repo reports ~4 GB bf16 dense per rank at TP=4, which agrees*

## 1. What "correct" means here, and against what

The same three anchors as V2-Lite, each still to be built:

| anchor | question it answers | criterion | tool |
|---|---|---|---|
| the scalar twin of a kernel | is the fast realization implemented right | equality, byte for byte | in-package tests (to write) |
| the oracle | is the model transcribed right | tolerance (f32 order noise) | `forward` vs goldens — **the oracle does not exist yet**: transformers' `glm5_next` modeling file, pinned, f32, CPU, if one loads without vLLM; else the repo's `dev/kernel-tests/*_ref.py` per-layer references (they match vLLM's decode path) |
| the oracle, teacher-forced | is a recipe a faithful interpretation of the weights | KL per position, top-1, oracle-token NLL | `forward --quiet` |

Open before anything else: **what runs the oracle.** A 320B bf16 forward at f32 on a CPU
needs ~1.2 TB of weights as f32 or a layer-at-a-time streaming oracle; the production
box has 512 GB. Either the oracle streams layers from the bf16 checkpoint (598 GiB on
disk) or it runs on a subset of layers against the repo's references. *Decide first.*

## 2. Noise floors by source

To measure, as V2-Lite §2. Prior from the repo (their NLL, prose, vs bf16 dense):

| source | magnitude | measured how |
|---|---|---|
| f32 reduction order | ~1e-7 per layer expected; 45 layers, mHC mixes the stream — to measure | placement matrix |
| INT8 activation, group 128 | ~1e-3 per element by construction | — |
| INT4 experts, our GPTQ | to measure; the NVFP4 checkpoint is the comparable public point | teacher-forced KL |
| FP8 dense linears | **+0.5 % NLL** (repo, per-channel weight, per-token activation scales) | their `quality/` |
| NVFP4 on selected dense (`in_proj`, `o_proj`, shared experts) | about as much again (repo) | — |
| NVFP4 on every dense layer | 2–3× the FP8 cost (repo); not their default | — |
| routing flips | to measure; sigmoid + bias routing, 8 of 288 | per-layer trace |

## 3. Teacher-forced divergence from the oracle

To measure on the five golden sets once an oracle exists. The repo's quality gates on
their recipe (NVFP4 experts + FP8/NVFP4 dense, greedy, thinking off): GSM8K (250) 97.2 %,
HumanEval 156/164, needle 12/12 to 507k, unchanged from stock. Their published
DFlash2 acceptance is 88–97 % on structured text, 58–75 % on code, 24–38 % on prose:
the speculation multiplier depends on the domain (§8).

## 4. Per-layer sensitivity

To measure. What is different from V2-Lite: three dense layers first; the eleven MLA
layers are the only ones with a cache; 34 layers carry a recurrent state instead. The
per-layer switch in the forward driver is the tool, as before.

## 5. Sinks

Unknown. The repo does not mention massive activations; the V2-Lite result (the sink at
position 0 costs most of INT4's error) is the first probe to run on this model: position
0's residual and FFN output magnitude per layer, and whether the shipped-sink trick
applies to a stream that has recurrent state rather than cache rows at position 0.

## 6. Activation ranges per layer

To measure (bf16 run, the five sets). The `swiglu_limit` 10.0 in the config is a
clamp inside the expert activation; note it when transcribing.

## 7. Routing: fragility and concentration

To measure, and this section feeds the placement table of the hybrid design
(`docs/`, memory `hybrid-experts`): the routing histogram per layer over the golden sets
and a longer corpus decides which of the 288 experts sit on the cards and which stream
from each socket. The router is sigmoid with a learned bias (`noaux_tc`), top-8 of 288
with normalized weights scaled by 2.5; group limiting is off (`n_group` 1). Our route
kernel's pick is softmax-then-rank; this one is sigmoid, bias-added for selection,
sigmoid-without-bias for the weights, normalized — a new route kernel.

## 8. What this decides

Nothing is decided; these are the decisions the model forces, with what is known:

- **Where the dense weights live.** ≈18 GB bf16 (§0) does not fit one 7900 XT beside a
  cache and fits two only split by layer; FP8 dense (≈9 GB) fits one card with the repo's
  +0.5 % NLL. Options: bf16 split across the two cards by layer, with the residual crossing
  once per token (the data-wait pattern between two queues); or FP8 on one card. *Our rule
  that dense weights stay bf16 was set on V2-Lite's numbers; this model reopens it with a
  measured price.*
- **The experts stream from DRAM.** 4.2 GB a token at 4 bits; across two sockets at the
  bandwidth the box measures (the 7R13 memory names the ceilings) the expert phase is on
  the order of 10 ms a token before speculation — *arithmetic, unmeasured.*
- **Speculation is the multiplier, and its cost model is the experts touched.** The repo's
  scheduler prices a step as fixed + per-expert-touched + per-token; verifying k drafts
  costs the union of their experts, not k×. On a DRAM-streamed MoE that is the whole
  argument for speculative decoding. The checkpoint's own MTP head is the licensed
  path; DFlash2 is non-commercial.
- **Recurrent state and the context store.** 34 KDA layers keep one state per sequence;
  a stored context is then states at checkpoints plus the 11 MLA layers' rows, and
  speculation needs the repo's RecoverSSM shape (one state per request, accepted tokens
  replayed through the stock kernel, bit-exact). *A design change to
  `context-cache-design`, not an extension.*
- **What maps from V2-Lite.** The head-merged MLA kernel (same `kv_lora_rank`; no rope
  half, a 2048-key sparse selection from the indexer instead of the whole context); the
  expert format and the CPU expert kernel (the same INT4 group / INT8 activation shape,
  wider rows); the bundle, wire, store and evals. **New:** KDA (a chunked recurrent
  kernel with a short conv and a gated delta rule), the indexer, mHC, the sigmoid-bias
  router, the MTP head, a 154,880-row head.
- **Prefill on the CPU experts must read each expert once per step** (the per-token loop
  in `host_experts` is a decode shape); the repo's fused prefill folds the gather,
  SwiGLU and requantization into the first GEMM.
- **Thinking off is outside the training** (repo, measured on every checkpoint): long
  outputs corrupt; low reasoning effort is the substitute. The chat template matters.

## 9. Not yet measured

Everything above §8 that says "to measure", and first:

- the safetensors index read into a byte table by tensor class (routed experts, shared,
  KDA, MLA, indexer, dense FFN, embeddings, head, mHC, vision) — replaces the arithmetic
  in §0
- an oracle that runs on hardware we have, for at least the first eight layers (three
  dense, four KDA, one MLA) — the transcription gate for the new kernels
- the routing histogram over a corpus, per layer (§7), from the oracle's picks
- position-0 magnitudes (§5)
- the box's DRAM ceilings and the hybrid's phase ratio (memory `hybrid-experts`)

## Regenerating

Nothing yet. The tools this document needs, in the order they are needed: a byte table
from the index (`bringup/`), an oracle or references (`bringup/forward` for `glm5_next`,
or the repo's `dev/kernel-tests`), goldens (`bringup/goldens.py` extended to the new
architecture), then the sweeps V2-Lite's document lists.
