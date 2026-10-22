# Expert prediction for offloaded MoE inference: the literature, 2024–2026 (question A)

Research for predictor-research-brief.md, question A. Written 2026-10-22. Sources are primary: arXiv abstracts, HTML
and PDFs read for this note. Where a number comes from a citing paper and not the original, the note says so. Text
marked **(inference)** is my reading, not a reported result.

## How to read the numbers against ours

Papers report one of three metrics, and they are not interchangeable:
- **Per-token recall@k with exactly k predicted** ("hit rate", "accuracy", ExpertRecall@K). The closest match to our
  route-ahead's per-token top-8 accuracy (.51–.62 on the weak layers, brief §"What the predictor is today").
- **Per-label accuracy** (each expert counted as a binary label, as in MoE-Beyond's 97.5 %). This is inflated by the
  true negatives and should be ignored.
- **Cache hit rate** (residency plus prefetch, as in MoE-CORE, SeqMoE and SP-MoE). This is a system number, not a
  predictor number.

No paper measures our figure, which is recall@13 of a step's union against the ~15 host experts left after the
residents are removed. The comparable number is per-token top-k recall of the next layer's picks.

Our model sits at an extreme. It has E = 288 and top-8 (models/glm-5.3-flash/config.json), sigmoid noaux_tc routing
and 4 hyper-connection streams. Every paper with recall numbers evaluates 8–128 experts. "Patterns behind Chaos"
(below) finds DeepSeek-V3's 256-expert routing the least correlated across layers and across tokens of the large
models it profiled. **(inference)** That explains why our route-ahead (.51–.62) sits well below the 79–90 % that the
same baseline reaches on Qwen3 or DeepSeekMoE. It also means the published absolute numbers will not transfer; the
relative gains are the transferable part.

## The table

"Route-ahead" below means our baseline: layer l + 1's router applied to layer l's hidden state. Fate, AdapMoE, HOBBIT,
DAOP, MoE-CORE, HybriMoE, MoE-Infinity's related work and ProMoE's "skip-based" all use this baseline under other
names.

| Method (venue, date) | Predicts from | Target | Reported accuracy (model, k) | Predictor size / cost | Training | URL |
|---|---|---|---|---|---|---|
| **Pre-gated MoE** (ISCA'24) | layer N's gate input, through a *modified* gate | layer N+1's experts | quality preserved, but only after fine-tuning the model (Switch-style models) | gate-sized | fine-tunes the model, which changes the forward | https://arxiv.org/abs/2308.12066 |
| **SiDA-MoE** (MLSys'24) | the batch's token embeddings, through an LSTM with sparse attention ("hash") | every layer, every token | up to 99 % (Switch-base, top-1) | LSTM | offline distillation; *replaces* the router | https://arxiv.org/abs/2310.18859 |
| **ExFlow** (2401.08383) | layer l's picks, through a conditional-probability table | layer l + 1 (for placement) | "strong inter-layer affinity" in GPT-MoE 8–64 E; no recall numbers | table E×E | offline traces | https://arxiv.org/abs/2401.08383 |
| **MoE-Infinity** (2401.14361) | the request's expert-activation matrix so far, matched by cosine to historical matrices | all later layers | no predictor accuracy in the abstract; latency 3.1–16.7× | trace store | online trace collection, no training | https://arxiv.org/abs/2401.14361 |
| **AdapMoE** (ICCAD'24) | route-ahead; layer 1 from the previous token's last-layer activation through a trained layer | l + 1, and l + 2/l + 3 when l + 1 is cached | no exact number in the text read; −25 % activated experts, 1.35× | none, plus one small layer | offline for layer 1 only | https://arxiv.org/abs/2408.10284 |
| **HOBBIT** (2411.01433) | route-ahead | l + 1 … l + 3 | top-1 96 % at l + 1, ~90 % at l + 2/l + 3 (Mixtral-8x7B, 8 E) | none | none | https://arxiv.org/abs/2411.01433 |
| **ProMoE** (2410.22134, v3 2025-09) | layer l's gate input, through a per-layer 2-layer MLP | l + 1, or l + 2 with "stride" | **84.7 % learned vs 66.9 % route-ahead ("skip-based") vs 58.3 % token-based**, averaged over DeepSeek-MoE-16B (64 E, top-6) and Qwen2-57B-A14B (64 E, top-8); stride costs ~5 % | ~2 M params a layer, ~200 µs on CPU | offline traces, 9:1 split, 1–2 GPU-hours | https://arxiv.org/abs/2410.22134 |
| **ExpertFlow v1** (2410.17954, v2 2026-04) | input token ids, through a transformer | every layer in one pass | "accurate"; no number in the abstract | small transformer | offline | https://arxiv.org/abs/2410.17954 |
| **Fate** (2502.12224) | route-ahead (Gate_in_i) | l + 1 | decode **78.79 %** with Gate_in_i vs **83.99 %** with Attn_in_{i+1} (layer l + 1's pre-attention input) vs 76.94 % with Attn_in_i; 97.15 % fetching everything above the 75th-percentile score (DeepSeekMoE-16B 64 E top-6 / Qwen1.5-MoE 60 E top-4) | none (CPU) | none | https://arxiv.org/abs/2502.12224 |
| **DAOP** (DATE'25) | route-ahead for layers ≥ 4 | l + 1 | 84.11 % (Mixtral-8x7B) | none | none | https://arxiv.org/abs/2501.10375 |
| **fMoE / FineMoE** (2502.05370) | the prompt embedding's semantic match plus the trajectory of earlier layers' gate distributions, matched against stored "expert maps" | layers up to the prefetch distance d | +36–39 % expert hit rate vs state of the art | map store | online history, no gradient training | https://arxiv.org/abs/2502.05370 |
| **Klotski** (2502.06888) | the gate's bias toward hot experts | l + 1 | system paper (multi-batch pipeline) | — | — | https://arxiv.org/abs/2502.06888 |
| **HybriMoE** (DAC'25, on kTransformers) | route-ahead ("impact-driven inter-layer prefetch") | l + 1 | system numbers (decode 1.70×) | none | none | https://arxiv.org/abs/2504.05897 |
| **MoE-Beyond** (2508.17137) | the sequence of past activations, through a 4-layer transformer | next layer | 97.5 % per-label (inflated), F1 86.6 %; GPU cache hit 17 → 72 % at 10 % cache (DeepSeek-V2-Lite) | small transformer | offline, 66 M activation traces | https://arxiv.org/abs/2508.17137 |
| **DuoServe-MoE** (2509.07379) | a layer-level predictor from activation traces | decode, next layer | 54–67 % top-2 (Mixtral; figure cited by 2511.10676) | light | offline | https://arxiv.org/abs/2509.07379 |
| **PreScope / LLaPor** (2509.23638) | layer l's hidden state (PCA 4096 → 256/512) **plus layer l's picks and their weights**, per-layer MLP, deeper for middle layers | l + 1 | ≥ 80 % every layer on DeepSeek-V2-Lite, Mixtral, Qwen3-30B and Moonlight; "15–68.4 % better than gate-based"; hot-top-4 ≥ 94 % | 0.5–2.8 MB a layer, 0.12–0.48 ms | offline 30 epochs plus periodic online fine-tuning on runtime samples | https://arxiv.org/abs/2509.23638 |
| **SP-MoE** (2510.10302) | the *draft model's* attention output at layer l, through the *target's* layer-l gate | the verify step's experts, every layer up to a cutoff | top-1 88.94 % (DeepSeek-Lite with an AWQ draft, draft/target attention cosine 94.6 %); ~88 % for Mixtral/Phi with cosine ~57–60 % | none | none (needs a full-depth draft with the same layer structure) | https://arxiv.org/abs/2510.10302 |
| **ExpertFlow (UConn)** (2510.26730) | mean-pooled token embedding, step size, layer index, earlier layers' activation bits, through a random forest that corrects the pre-gate | l + S, S adaptive | long-distance asymptote 60–65 % vs pre-gate 26–33 % (DeepSeek-V2-Lite, Qwen1.5-MoE, Qwen2-57B) | CPU random forest | offline | https://arxiv.org/abs/2510.26730 |
| **Pre-attention prediction** (ETH, 2511.10676) | layer l's *own* pre-attention normed input, through a per-layer MLP (d → 2048 → E) | layer l (same layer, across its attention) | layer 1 exact top-k set **93.03 %** (DeepSeek-V2-Lite top-6), 94.69 % (Qwen3-30B top-8), over-provisioned 98.7 %; "other layers similar or better"; vs Fate's 78.79 % | ~4 M params a layer **(inference: d·2048 + 2048·E)**; 0.075–0.129 ms | offline, 10 M MMLU samples, 30 epochs; eval split and phase not stated | https://arxiv.org/abs/2511.10676 |
| **MoE-SpeQ / Specter** (2511.14102) | an INT4 draft *copy* of the target | future tokens' experts, several tokens ahead | > 90 % of the target's top-k (Phi-MoE, Qwen1.5-MoE) | a quantized full model | none | https://arxiv.org/abs/2511.14102 |
| **OD-MoE** (2512.03927) | a quantized "shadow" copy of the whole model running ahead, its KV aligned every step | several layers ahead (up to l + 4) | recall 99.94 % FP16 / 97.34 % INT8 / 95.67 % NF4 (Mixtral top-2); without alignment ~30 % by token 256 | 45 GB shadow | none | https://arxiv.org/abs/2512.03927 |
| **SpecMD** (2602.03921) | benchmark of cache policies | — | expert access is *not* temporally local (LRU/LFU); a "Least-Stale" eviction policy gives 85× fewer collision misses | — | — | https://arxiv.org/abs/2602.03921 |
| **Speculating Experts** (2603.19289, 2026-03) | "quasi-hidden state" q_l = LN_{l+1}(r_l + d_l): the post-attention residual plus **d_l = Σ g_e · d_{l,e}, each picked expert's average output** (offline); optional estimator (proj → 2×FFN SiLU + residual → LN → E) | l + 1 | router on q_l: recall@k ~90 % beyond layer 2 (Qwen3-30B-A3B); the estimator gives up to +25 points on early layers, 83–90 % overall | estimator 4 M (GPT-OSS-20B), 17 M (Qwen3-30B), 45 M (GPT-OSS-120B) | offline KL distillation, plateau at 4–5 M tokens | https://arxiv.org/abs/2603.19289 |
| **ST-MoE** (2606.15453) | cross-layer and cross-token ("spatio-temporal") correlation | next layers | abstract only, no numbers | light | — | https://arxiv.org/abs/2606.15453 |
| **SpecPrefetch** (2607.24787, 2026-07) | layer l's router input x^l, through a **per-pair low-rank adapter ẑ = B A x** | l + 1 | **R@8 85.60 vs Fate (route-ahead) 81.40, R@10 92.31 vs 88.38** (Qwen3-VL-30B-A3B, 128 E top-8, GSM8K); vs ProMoE 82.97/89.94; a *single shared* MLP is worse than route-ahead (79.36 R@8); DeepSeek-VL2-Tiny (64 E top-6) ties Fate | 6.48 M over 47 pairs on Qwen3 (**rank ≈ 63, inference** from (L−1)·r·(D+n)); 0.82 M on DS-VL2-Tiny | offline, one epoch over an instruction corpus, KL to the next router's full distribution | https://arxiv.org/abs/2607.24787 |
| **DraftExpert** (2607.24434) | a trained per-layer *draft expert* inside a self-speculative drafter, with a router-agreement loss | the verifier's experts for each kept draft token | prefetch hit 88 % (DeepSeek-V2-Lite), 86 % (Moonlight), both 64 E top-6 | one small expert a layer | offline distillation | https://arxiv.org/abs/2607.24434 |
| **SPICE** (2608.21240) | a draft hidden state rolled forward layer by layer through the target's attention and its predicted experts; halts when the routing mass on top-K drops below τ | l + 1 … l + H, adaptive | no recall numbers; ablation: deep prediction with CPU fallback is best | — | — | https://arxiv.org/abs/2608.21240 |
| **SeqMoE** (2609.12978) | sequence-to-sequence model over the activation history | multi-step, multi-layer | 96.97 % cache hit at 45 % residency (system number) | — | offline | https://arxiv.org/abs/2609.12978 |
| **MoE-CORE** (2610.01950, 2026-10) | route-ahead plus a routing-history-aware cache | l + 1 … l + 3 | on **DeepSeek-V4-Flash** and **GLM-5.2**: exact hit 89.05 % with no prefetch, 92.43 % at depth 1, 93.81 % at depth 2, 94.05 % at depth 3; the cache is worth more than the prefetch (TPOT +30 % without it vs −15.5 % from prefetching) | none | none | https://arxiv.org/abs/2610.01950 |
| **Patterns behind Chaos** (2510.05497, v5 2026-05) | trace study: DeepSeek-V3, Qwen3-235B, Llama4-Maverick, Kimi K2; 24 k requests | — | the top 20 % of cross-layer expert pairs carry ~45 % of activations (DeepSeek-V3, the weakest), 55 % (Kimi K2), 68 % (Qwen3), 80 % (Llama4); same-expert reuse on consecutive tokens only in higher layers; public traces at huggingface.co/datasets/core12345/MoE_expert_selection_trace | — | — | https://arxiv.org/abs/2510.05497 |

Speculative-decoding work on the cost of the expert union, which is not prediction but changes what has to be
fetched: SpecFit (https://arxiv.org/abs/2610.15417) picks the verify subtree and draft depth by draft probability per
unit of expert-union cost on Qwen3-30B-A3B. AcceptMoE (https://arxiv.org/abs/2608.02989) and "The Limits of
Speculation" (https://arxiv.org/abs/2609.22156) bound or trim the verify step's union.

Systems with no predictor, listed in the brief: MoE-Lightning (CPU attention, a throughput pipeline), MoE-Gen
(module batching), Fiddler (CPU computes the experts that are not resident, static popularity), kTransformers and Lina
(popularity estimates for scheduling).

## What the literature says, in five findings

1. **The learned gains over route-ahead are real, and largest where route-ahead is weakest.**
   - ProMoE: 66.9 → 84.7 %. Its model set includes Qwen2-57B, the case closest to ours.
   - ExpertFlow (UConn): 26–33 → 60–65 % at long distances.
   - Where route-ahead is already strong (Qwen3, ~81 % R@8), SpecPrefetch's adapter adds ~4 points at R@8 and ~4 at
     R@10.
   - **(inference)** At our .51–.62 we sit in ProMoE's regime, not SpecPrefetch's.
2. **Every predictor that beats route-ahead is trained offline, on millions of tokens, and only runs inference at
   serve time.**
   - Speculating Experts' estimators plateau at 4–5 M tokens.
   - SpecPrefetch trains one epoch over a corpus. Pre-attention uses 10 M samples for 30 epochs. ProMoE uses 1–2
     GPU-hours of traces. PreScope trains 30 epochs offline, then fine-tunes online.
   - Ours is the one case trained online from zero. **(inference)** That is the most likely reason our rank-64 map
     (queue 086) gave only +.005–.007, while the same object (SpecPrefetch's rank-~63 B·A·x) gives +4 points at
     R@8. Two more signs point the same way: our full rank did *worse* (.6213), and rank 128 equalled rank 64. Both
     look like an optimisation and data limit, not a capacity limit. Our pair sees 4 tokens a step, ~10 k samples in
     a 10 k-token run, against the literature's 10⁶–10⁷.
   - The training target also differs. SpecPrefetch and Speculating Experts distil with KL to the next router's
     *full* distribution; ours is BCE on the top-8 labels. The full distribution gives a denser signal per token.
3. **Layer l + 1's attention is the larger missing piece, and a nonlinear map across one attention block recovers most
   of it.**
   - Fate's input ladder (decode, DeepSeekMoE/Qwen1.5): Attn_in_i 76.9 % → Gate_in_i (our x_l) 78.8 % →
     Attn_in_{i+1} (x_l plus layer l's FFN output) 84.0 % → 100 % at the true input. Layer l's FFN output is worth
     ~5 points; layer l + 1's attention is worth ~16.
   - Our own bound table (expert-cache-architecture.md, "More data") has the same shape: the full FFN output adds
     only +.022–.027.
   - The ETH pre-attention paper predicts the *same* layer's experts from its pre-attention input. That is exactly the
     "attention surrogate" object question B asks about, across one attention block. A 2-layer MLP (d → 2048 → E)
     reaches 93 % exact top-6-set match on DeepSeek-V2-Lite's layer 1, against Fate's 78.8 %.
   - Caveats: layer 1 only in the text, MMLU-trained, phase unstated. Still the strongest published evidence that
     depth 2 with a nonlinearity, trained offline, recovers what attention adds.
4. **Two layers ahead costs little, at least on coarse models.**
   - HOBBIT: 96 → ~90 % top-1 (Mixtral, 8 E). ProMoE: stride costs ~5 %. AdapMoE uses the same activation 2–3 layers
     ahead.
   - MoE-CORE, on DeepSeek-V4-Flash and GLM-5.2, the closest models to ours: depth 2 adds +1.4 hit points over
     depth 1, and depth 3 adds +0.2.
   - **(inference)** On E = 288 the loss per extra layer is larger than on Mixtral. None of these papers measures
     recall at depth 2 on a 256+-expert model.
5. **History and co-routing tables are weak for fine-grained DeepSeek-style routing.**
   - Patterns behind Chaos: DeepSeek-V3 has the weakest cross-layer pair concentration and the weakest token-to-token
     reuse of the models profiled. SpecMD finds no temporal locality.
   - This matches our replay: the previous step's picks give .21 alone, and a tie-break gives ~0.
   - The history-based systems (MoE-Infinity, fMoE, MoE-Beyond, SeqMoE) report cache hit rates, not next-layer
     recall, and target settings where the request's history is the only signal.

## Which inputs we already have (brief, "The information available")

| Literature input | Ours | Note |
|---|---|---|
| Layer l's gate input → layer l + 1's router (route-ahead) | yes (1, 3) | shipped |
| A learned map over layer l's gate input (ProMoE, SpecPrefetch, PreScope) | yes (1), x_l on the FFN card | learned online today; the literature trains offline |
| Layer l's picks and gate weights as features (PreScope, ExpertFlow-UConn) | yes (2) | our co-routing table uses the picks but not the weights, and not jointly with x |
| Each picked expert's average output d_{l,e} (Speculating Experts) | **no, but cheap to make** | needs E × H a layer; we have the exact picks and weights before `hc_post` |
| Shared-expert output added to the input | yes (our second margin, d2) | no paper uses this; Speculating Experts' d_l is the routed analogue |
| Layer l + 1's pre-attention input (Fate's best, pre-attention MLP) | no (7): it needs the host's routed rows | arrives at `hc_post`, which is the window's start |
| A full-depth draft model with the same layer structure (SP-MoE, MoE-SpeQ, OD-MoE) | no | our MTP drafter is one layer; it has no per-layer hidden states for the target's layers |
| Token ids / embeddings (SiDA, ExpertFlow, fMoE semantic) | yes (5) | weak in middle layers; PreScope's Fig. 6 shows token-based predictors win only in the first and last layer groups |
| Multi-stream (hyper-connection) residuals | yes, 4 streams | **no paper studies prediction on HC/mHC models** (searched) |

**(inference)** On the hyper-connection point: route-ahead applies layer l + 1's router to layer l's `hc_pre` collapse.
Layer l + 1 collapses the four streams with its own, input-dependent mixing. A predictor that reads the four streams,
or applies l + 1's mixing, sees an input that no single-stream paper had. Whether that matters is for question B and
for replay to decide.

## What transfers to our shape

Our shape: 2–4-token steps, a ~500 µs window, recall@13 against ~15 host experts, one layer ahead, about 180 µs of
card idle time of which ~100 µs is used.

- **Per-token predictors transfer directly.** Every paper predicts per token. Our union over 2–4 tokens is a
  post-processing step the planner already does.
- **The cost transfers.** A rank-64 per-pair adapter is ~0.28 M parameters (4096·64 + 64·288), 0.56 MB in bf16 a pair.
  An MLP of 4096 → 1024 → 288 is ~4.5 M, 9 MB a pair, ~10–15 µs to read at the card's bandwidth **(inference)**.
  Both are inference only; our earlier cost came from online Adam on U in f32 (~125 MB a step a card).
- **The absolute recalls do not transfer** (fine-grained routing; see above). The gain ratios are what to carry over.
- **Speculative-decoding prefetch transfers poorly.**
  - SP-MoE, DraftExpert, MoE-SpeQ, Specter and OD-MoE all need a draft with per-layer states that correspond to the
    target's layers, or a full quantized copy.
  - The usable part is DraftExpert's timing rule: prefetch a token's experts as soon as that draft token is kept.
    **(inference)** For us that means starting the *next* step's first layers from the draft's token ids and the
    MTP layer's state. The literature gives no number for that case.
- **The binding constraint is not prediction.** expert-cache-architecture.md ("Prediction is no longer the bound",
  2026-10-21) records that more right experts ranked low in the 13 stayed partial. MoE-CORE reports the same shape on
  DeepSeek-V4-Flash and GLM-5.2: residency outweighs prefetch. A better predictor pays when it moves experts *into*
  the completed set, which argues for calibrated probabilities (KL/BCE) and for predicting fewer experts more
  confidently, not only higher recall@13.

## Ranked short list

1. **An offline-pretrained per-pair predictor over x_l, distilled to layer l + 1's full router distribution, run
   inference-only on the card. Online learning stays on the cheap terms (bias, calibration) as today.**
   - The numbers behind it:
     - ProMoE: 66.9 → 84.7 % against route-ahead, in the weak-route-ahead regime we are in.
     - SpecPrefetch: rank-~63 B·A·x, R@8 81.4 → 85.6 and R@10 88.4 → 92.3 on a 128-expert top-8 model, with 6.5 M
       parameters for 47 pairs.
     - PreScope: per-layer MLP with the picks and weights as extra features, "+15–68 % over gate-based".
   - The literature's form: a per-pair low-rank adapter first, then a 2-layer MLP if low rank saturates. SpecPrefetch's
     single *shared* MLP lost to route-ahead, so the predictors must be per pair.
   - Prerequisite: the dump. x_l and the l + 1 router's full logits (not only the picks), at 10⁶-token scale. Prefill
     tokens count, since every prefill token has labels at every layer.
   - **(inference)** This is the step most likely to turn queue 086's +.005 into a material gain.
2. **Default vectors (Speculating Experts): add Σ g_e · d_{l,e} over layer l's exact picks to the second margin's
   input.**
   - It approximates the "every routed row" bound (.6255 → .6471 deployed, +.022) from data the FFN card has before
     `hc_post`.
   - Cost: E × H bf16 = 2.4 MB a layer, plus an 8-row weighted sum a token.
   - The d_{l,e} can be running means of each expert's output, which the host already computes.
   - Speculating Experts reports "substantial" gains on GPT-OSS and negligible ones on Qwen3, with no ablation
     numbers. The ceiling is our own +.022.
3. **Nonlinear attention surrogate / two layers ahead.**
   - The ETH pre-attention MLP (d → 2048 → E) recovers what one attention block adds: 93 % exact-set match vs Fate's
     78.8 %, on layer 1 of DeepSeek-V2-Lite.
   - The same object trained on (x_l → l + 1's router) is candidate 1's nonlinear form.
   - Trained on (x_l → l + 2), it gives the second window. The literature's depth penalty is ~5 % (ProMoE stride) and
     +1.4 points of hit at depth 2 on DeepSeek-V4-Flash/GLM-5.2 (MoE-CORE).
   - Ranked third because the ETH result is one layer, MMLU-trained and with an unstated phase, and because no paper
     measures depth 2 on a 256+-expert model.

Not recommended, from the literature:
- **Methods that change the forward:** Pre-gated MoE, SiDA, SpecPrefetch's "execute the prediction" variants and
  Speculating Experts' speculative execution (which cost 5–17 points on Qwen3 tasks).
- **Shadow or draft copies of the model:** OD-MoE, MoE-SpeQ, SP-MoE. They need a full-depth second model, and OD-MoE
  needs a 45 GB shadow.
- **History matching:** MoE-Infinity, fMoE, SeqMoE. The routing they would exploit is the kind fine-grained
  DeepSeek-style models show least.
