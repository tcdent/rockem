# The predictor's architecture for our signal shape, and how to validate it (questions B and D)

**Date:** 2026-10-22. **Brief:** predictor-research-brief.md. **Scope:** questions B (architecture) and D (offline
validation). Question A (the literature survey) and C (the drafter as a signal) are covered only where they bear on B.

**Model constants** (models/glm-5.3-flash/config.json, through build.rs into `config_consts.rs`): E = 288 routed
experts, TOP_K = 8, one shared expert, H = 4096, 45 layers, of which 3 are dense (0–2) and 42 are MoE (3–44). The
attention is KDA (gated delta rule, 64 heads × 128) in 34 layers. MLA/DSA (sparse) is in layers 3, 7, …, 43 (l ≡ 3 mod
4). mHC: 4 residual streams.

**Measured vs inferred.** Every number in the "measured" column and in §1 comes from scripts run today against the
existing phase3b dump. The scripts are in bringup/expert-cache/scratch/arch/ and the logs in data/arch/. The dump has
4,986 decode steps and 13,729 tokens (2.75 tokens a step) at 18k context, with x_l for layers 3–43 and the picks. The
router weights come from the checkpoint. Expected-recall figures for things not built are **inference** and are marked
as such.

**Metric.** Unless stated otherwise: union recall@13 over all of the next layer's picks (phase3's metric), raw logits
through the deployed margin (`10·(σ + bias − 8th best)`), **without** the learned bias/co-routing/calibration. On this
scale the route-ahead scores .581. The shipped predictor's corrections take it to .62 (phase3b, settled). Numbers in
this document are therefore comparable with each other and with "route-ahead alone", not directly with .62.

## Ranked candidates

| # | Architecture | Inputs at prediction time | Parameters / bytes read a layer | FLOPs a token a layer | Recall@13 (raw → with today's corrections) | Learns | Serving cost | Main failure mode |
|---|---|---|---|---|---|---|---|---|
| 1 | **Attention-ahead (KDA targets):** layer l+1's own KDA read-out run on the stale input. Input x̂ = x_l + shared output. q = conv(W_q·n(x̂)) with the l2-norm per head, o = S_{l+1}ᵀq over the step-start state, o_norm and the output gate, then the router folded through o_proj (M = R_{l+1}·diag(g)·W_o, 288 × 8192). This gives a third margin with its own learned weight. | x_l, the shared output, layer l+1's KDA state and conv cache (in VRAM on the same card), weights | No new learned weights beyond 1–2 scalars. Reads: W_q at data-aware rank 512 12.6 MB (FP8 full 34 MB, bf16 full 67 MB), state 4.2 MB f32, M 4.7 MB, gates 2.1 MB: **~24 MB** | ~20 MFLOP (q 12.6, read-out 2.1, M 4.7) | **Inference:** ceiling .70 on the 30 KDA-target pairs. If it recovers 50 % of the logit delta there: raw ≈ .64 → **~.65–.67** with the corrections. At 25 %: ≈ .61 → ~.63. The measured oracle curve is in §1.2. | Zero-shot from the model's weights. Online calibration (the existing Adam). W_q's low-rank basis fit once from prefill activations. | ~30–40 µs a layer (bytes-bound on a ~800 GB/s card); fits next to today's ~100 µs chain in the ~180 µs wait | (a) The step's own k/v updates are ignored for tokens 1–3, and KDA's self term (q·k) may be large. (b) FFN_l's routed output is missing (stale input). (c) The mHC pre/post scaling is reduced to a learned scalar. (d) Nothing for the 10 MLA-target pairs. (e) The pair at the card boundary needs the other card's state. |
| 2 | **Distilled surrogate from x_l, dense target:** a per-pair map x_l → the 288-logit delta, trained on the next router's full logits (MSE/KL on margins), not the 8-hot picks. Ridge/linear, or a small shared MLP. | x_l (normed), optionally the shared output | Ridge: 4096 × 288 a pair = 1.18 M (2.4 MB bf16). MLP shared trunk 4096→1024→1024 + 40 heads 1024 × 288 = 17 M (34 MB) | Ridge 2.4 M. MLP ~11 M | **Measured offline** (60/40 split, raw; test-split route-ahead .584): ridge **.613**, shared MLP **.620**, a per-pair constant delta .595. **Inference:** about +.01–.02 over today's .62 once the corrections overlap. | Pretrained from traces, where prefill gives every prompt token a dense target for free, plus a slow online refresh. | Prediction ~5 µs. The cost is the learning: 086's lesson. Keep the update off the wait (closed form at the step's end or from prefill only). | Context-dependent attention is not in x_l. The gain is small and lands in plan slots 9–13, where 086 showed extra right experts end up partial, not complete. |
| 3 | **Two-horizon plan:** today's l → l+1 plan plus the confident head of an l → l+2 plan (top 4, precision .94), posted a layer early | As today, plus layer l+2's router rows | +E × H router rows a pair (already in VRAM) | +2.4 M | **Measured raw:** l+2 from x_l .521 vs l+1 .581 (−.06). Precision of the top 2/4/6/8/13: .972/.937/.892/.840/.708 vs l+1's .992/.976/.949/.911/.793. | As today | One more route-ahead and rank (~20 µs) | Adds no DRAM-idle time while the host run saturates DRAM (§2). It only pays if the run leaves DRAM idle or the prefetch overruns its window. Two plans also compete for L3 capacity. |

Recommendation: validate #1 before building anything (§3, steps 1–3). It is the only candidate that addresses what
the measurements say is missing: layer l+1's context-dependent attention, which no function of x_l can supply. #2's
offline numbers say a map from x_l stays near +.03–.04 raw. #3 does not pay on this box's timeline as measured today.

## 1. What the data says (measured today)

### 1.1 The ceiling and where the gap is
- With the exact next-layer input (x_{l+1}, reproducing the router's picks: per-token top-8 agreement .96–.995 through
  the int8 dump), union recall@13 is **.705**. This is the cap ceiling: a step's union is 13.9 / 19.2 / 24.2 experts at
  2 / 3 / 4 tokens. The route-ahead scores .581. Total headroom: **.124**. The shipped corrections recover about .04 of
  it.
- The gap by the type of the target layer's attention: MLA target .105 (10 pairs), KDA target after an MLA layer .127
  (10), KDA → KDA .135 (20). **Most of the gap is at KDA targets.** That is where a cheap attention-ahead exists.
- Weak pairs: 17 → 18 (.51 vs .76) and 24 → 25 (.48 vs .76), both KDA targets.

### 1.2 How recall grows with the share of the delta recovered (oracle)
The logit delta is δ = R_{l+1}(x_{l+1}) − R_{l+1}(x_l) (288-d).
- Recovering a fraction α of δ exactly: α = .25 → .624, .5 → .663, .75 → .692, 1 → .705. Recall is concave in α. **The
  first half of the delta is worth two-thirds of the headroom.**
- Recovering δ's top-r principal components exactly (fit on the same data, so optimistic): r = 8 → .610, 32 → .633,
  64 → .651, 128 → .673. Rank 64 explains 63–89 % of δ's variance (65–70 % typical). **δ is high-rank in logit space.**
  A rank-64 correction, even an oracle one, gets about half the headroom. This is consistent with 086's rank 16/64/128
  all landing within .005 of each other: the learnable part of δ from x_l is low-rank, and the rest is not a function
  of x_l.

### 1.3 What a map from x_l can learn (offline, train on the first 60 % of steps, test on the rest)
Test-split references: route-ahead .584, exact .709.
- Ridge x_l → δ (dense target, λ ∝ N): **.613** (+.029). A per-pair constant δ (what a bias can learn): .595.
- The previous step's last token's actual δ at this pair: ·0.5 → .5855, ·1.0 → .543. Ridge on [x_l, that δ] → .611.
  The previous position's delta carries nothing usable beyond x_l. This matches run 9593789a's finding on the previous
  step's picks.
- Shared MLP (4096 → 1024 → 1024, pair embedding, per-pair heads, MSE on δ, 3 epochs): **.620** (+.036; §1.6).
- Phase3's online numbers (BCE on picks) agree: a full-rank linear term reached .632 settled against .621 without it.
  **A function of x_l alone is worth about +.03–.04 raw (+.01 on top of the corrections in phase3). The remaining ~.09 is context.**

### 1.4 Two layers ahead
- Raw recall@13 of layer l+2's picks from x_l through layer l+2's router: **.521**, against .581 for l+1. The loss is
  .06, about 10 % relative. Consistent with ProMoE's "about 5 % less accuracy" for stride prefetching on smaller models,
  and with SeqMoE's observation that cross-layer predictors degrade quickly with distance.
- Plan precision (share of the union's top-k that is used), 1-ahead vs 2-ahead: top-2 .992 / .972, top-4 .976 / .937,
  top-8 .911 / .840, top-13 .793 / .708. The marginal precision of plan slots 9–13 is .60 at 1-ahead and .50 at
  2-ahead. **The head of the plan survives two layers. The tail does not.**

### 1.5 Sequence structure (the step's consecutive positions)
- Picks shared at the same layer (of 8): adjacent positions in a step 2.06, the previous step's last token 1.32, a
  random token 0.47.
- Cosine of δ: adjacent in a step .548, previous step's last .483, random .403. Most of this is the shared mean δ.
- Positions are sticky, but only between the step's own tokens, which are computed together and so are unknown at
  prediction. The previous step's information is already used through the route-ahead. The Π(1 + eᶻ) union ranking
  already rewards experts shared across the step's tokens. **The sequence information that matters is the one the
  model itself keeps: the KDA state of layer l+1, which carries the whole context in 4.2 MB. Candidate #1 reads it.**
  (Inference: the previous step's last token is not always the step's predecessor, because rejected drafts exist. The
  dump lacks positions and acceptance; see §3.)

### 1.6 The shared MLP
One MLP for the 40 pairs: GELU 4096 → 1024 → 1024 plus a pair embedding, per-pair heads 1024 → 288, ~17 M parameters,
AdamW, MSE on δ/σ. Trained 3 epochs on the first 60 % (330k token-pairs, ~14 min on 48 cores). On the test split:
route-ahead .584, **MLP .620**, ridge .613, exact .709. Test MSE on pair 20: 0.223 against a mean square of 1.08, but
most of that is δ's mean, which a constant already captures (.595). Training loss was still falling (0.40 → 0.26)
while test loss flattened (0.240 → 0.223): this is limited by data and by the function, not by width. The nonlinearity
buys +.007 over ridge. **A function of x_l saturates at about +.03–.04 raw at this data size.** (Inference: prefill
traces, ~18k positions a prompt, would give 100× the data. Phase3's online-vs-offline comparison suggests the
x_l-only ceiling, full-rank .632 settled, is close to what more data finds.)

### 1.7 Is a low-rank W_q enough for the attention-ahead?
- Weight-energy rank of q_proj (8192 × 4096): r512 keeps .50–.63, r1024 .68–.78. Not low-rank as a matrix.
- Data-aware (q's variance over the dump's x_{l}, layers 10/17/25): r128 .71, r256 .80–.82, **r512 .89–.90**,
  r1024 .95–.96. Fit the basis from activations (prefill gives plenty), not from the weight's SVD.
- o_proj needs no compression: folded with the router it is M = 288 × 8192 (4.7 MB). The per-head l2-norm of q makes
  each head's read-out insensitive to a uniform error in q's scale.

## 2. The reasoning behind the ranking

**(1) Recovering layer l+1's attention contribution.** The route-ahead misses three things. FFN_l's output: phase3b
puts the whole FFN output's worth at about +.027, and the shared part on the card at +.005. Layer l+1's attention.
And mHC's remixing and norm. An attention output depends on the context. A surrogate that sees only x_l can learn
E[attn | x_l], and §1.3/§1.6 measure that at +.03–.04 raw (ridge, MLP). So a distilled per-pair MLP/low-rank map is **not** the right object
for the bulk of the gap, at any width. The right object is the attention itself. For softmax/MLA layers that is the
forward (indexer, 18k latents), which is why the literature does not do it. For KDA it is cheap: given the recurrent
state S at the step start, the read-out is linear in q, o = Sᵀq, plus small nonlinear per-head norms and gates.

The state already exists on the card, as the next layer's own memory of the whole sequence. The only stale part is
the input: x_l instead of layer l+1's attention input, which is the hc collapse of the streams after FFN_l. This is
the same staleness the route-ahead already lives with, so the attention-ahead adds the context term without adding a
new error source. "Distilled attention surrogate" then reduces to a compression of W_q (data-aware rank 512, §1.7),
fit once, plus a learned calibration scalar. This candidate rests on the model's own weights, not on learned capacity.
Its value is unmeasured and is step 2 of §3.

Expected range (**inference**): the attention-ahead cannot recover FFN_l's routed part, and it approximates the
in-step k/v updates and the mHC scalars. A plausible share of the KDA-target delta is 25–60 %. On §1.2's curve that is
raw .61–.65, and about .63–.67 with today's corrections, against the .705 / .746 (host set) cap. The weak pairs (17,
24) are KDA targets and should move most.

Related work for the components:
- Next-layer gate on the current hidden state: Eliseev & Mazur https://arxiv.org/abs/2312.17238, AdapMoE
  https://arxiv.org/abs/2408.10284, Fate https://arxiv.org/abs/2502.12224.
- The "quasi-hidden state" (normed residual plus a default vector), which is the closest to adding a missing-component
  estimate: https://arxiv.org/abs/2603.19289.
- Per-layer MLP predictors (~2M parameters, offline-trained, "84.7 %" accuracy cited by MoE-SpeQ):
  https://arxiv.org/abs/2410.22134.
- Pre-attention linear routers with a ranking loss (93–98 % on DeepSeek-V2-Lite/Qwen3-30B, predicting the *same*
  layer from before its attention, i.e. with the attention still missing): https://arxiv.org/abs/2511.10676.

None of these runs the next layer's attention approximately. Their reported accuracies come from smaller expert
counts and same-layer or near-identical inputs, so they do not transfer to E = 288 / top-8 one layer ahead. Our own
route-ahead per-token top-8 is .51–.81.

**(2) Two layers ahead.** Accuracy: −.06 raw (§1.4). Bytes-in-time: the prefetch's useful window is the host DRAM's
idle time, which is the card's attention phase (~500 µs). During the host run DRAM is saturated by the run's own
misses (~316–322 GB/s against STREAM ~293). A plan posted a layer earlier finds no extra idle DRAM to fill. It moves
the deadline, and the deadline is not binding today (`slot` 0.36 ms a verify step). Two resident plans also compete
for the L3 (2 × 174 MB against 384 MB, with the keep ways partitioned).

So a two-layer plan loses on bytes-in-time. It wins only if a measurement shows DRAM idle during the run (a run served
mostly from L3) or a prefetch overrunning its window. The l+2 head at .94 precision for the top 4 would then fill that
slack. (Inference, from the timeline in docs/expert-cache-architecture.md §8 and the measured rates. The DRAM-idle
share during the run is the number that decides it, and prow's pool rows can give it.) SeqMoE
(https://arxiv.org/abs/2609.12978) argues for deeper lookahead because PCIe moves only ~1.4 experts a layer there.
Our bound is DRAM bandwidth shared with the run, not link latency, so their argument does not transfer.

**(3) Consecutive positions.** Measured weak as a feature (§1.5). The useful form of "the step's tokens are a
sequence" is that the KDA recurrence over them is cheap to run. Candidate #1's refinement is to include tokens 0..i−1
of the step in token i's state, via the step's k/v at low rank: about another ~25 MB a layer. That is an ablation in
§3 step 2, not a default. Stickiness from the previous step is already exhausted (run 9593789a, §1.3).

**Cross-cutting: the learning target.** Today every pair learns from 8 picks a token (BCE). The next router's 288
logits are computed on the same card a layer later, so a dense distillation target (MSE/KL on the margins) costs
nothing to obtain. It carries far more signal a token. Ridge on the dense target (§1.3, offline) matched phase3's best
online BCE linear terms with closed-form fitting. (Inference: dense targets should speed the online settling, today
~3k positions, and help any learned term in #1 or #2.)

## 3. Validation before building (order of work)

Everything below is lab-only (`dump-pred` and Python). Nothing touches the serving path until step 4.

0. **Done today:** ceilings, oracle curve, ridge, two-ahead, positions and W_q rank from the existing dump.
   Scripts: bringup/expert-cache/scratch/arch/{weights,common,check,expA,expB,expC,expD,wq_rank,wq_data}.py. Logs:
   data/arch/. Gotcha: rank the union by the *margin* (σ + bias), not by softplus of raw logits. The router's bias
   (~6.5) dominates the selection, and ranking by raw logits gives nonsense (exact input → .45).
1. **Decompose the gap** (dump-pred extension a; cheap, existing buffers). Per decode layer record, per token, add:
   - the position id, the draft index and whether the token was accepted;
   - the shared expert output [H] (bf16);
   - layer l+1's attention output as it enters the streams [H] (bf16), with the hc coefficients that scale it (attn
     post [4] and the FFN pre [4] of l+1, or their product);
   - a residency bit per pick.

   In Python: router(x_l + c·a_{l+1}) is the **ceiling of any attention-ahead**. router(x_l + FFN part) is FFN_l's
   share. The remainder is mixing and norm. If the attention ceiling is below ~.64 raw, stop: #1 cannot pay. The
   residency bits make host-set recall and wasted bytes (13.4 MB an unused planned expert) computable from one file,
   with no join against trace.json. Size: ~18 KB a token-layer, ~10 GB for a 24-request run (data/ is git-ignored).
2. **Score the attention-ahead itself** (dump-pred extension b). In the lab node, compute layer l+1's KDA read-out
   over x_l (+ shared) with the step-start state on the host in f32 (the reference math, slow is fine). Dump its 288
   logits in three variants: (i) step-start state only; (ii) with the step's tokens 0..i−1 folded in; (iii) (i) with
   W_q at data-aware rank 512. Each is a further LG array, so phase3/sim.py `--lg2 LGattn` scores it with the deployed
   corrections (settled recall@13) unchanged, and the variants give the in-step-update and the low-rank costs directly.
3. **Score on bytes, not only recall.** For each candidate, report at cap 13 over host experts:
   - recall;
   - the marginal precision of slots 9–13 (today .60);
   - wasted experts a layer;
   - the change in experts complete if the plan's order is kept: the 086 lesson that extra right experts ranked low
     stay partial.

   Report each by target type (KDA/MLA) and by step size (2/3/4 tokens). Re-run #3 here with a measured DRAM-idle
   share from the run (prow pool rows) before reconsidering it.
4. **Only then a kernel:** `pred_attn` in the FFN card's queue before `pred_step`, timed against the ~180 µs wait in a
   lab build with track-launches. Gate: long SCORE bit-identical (prediction never enters the forward) and missed
   experts down on the long decode run.

Parallel and cheap: #2 with the dense target in phase3/sim.py online (a `--target dense` option), to settle whether
the learning-target change alone is worth porting to `pred_learn`.
