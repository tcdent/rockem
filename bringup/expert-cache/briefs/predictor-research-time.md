# Predictor research — question C widened: time (2026-10-22)

Brief: [predictor-research-brief.md](predictor-research-brief.md). The angle here is *when* the information about
the coming layer's experts exists in a step, not how well one signal ranks. Measured numbers are from this
session's scripts on the existing data (bringup/expert-cache/scratch/time/: `time_signals.py`, `controls.py`,
`combine.py` on data/trace.json; `ahead_d.py`, `rank_precision.py` on data/phase3b, the pred2 dump). Literature numbers
are the papers' own, quoted from the arXiv text (copies under scratch/lit/). Anything else is marked *inference*.

## 0. The frame: where the bytes can come from

The architecture doc and 091 already established that the window is at the box's DRAM ceiling: ~316 GB/s, a
13-expert plan needs ~550 µs against a ~500–534 µs window, and 97.8 % of the plan arrives. 082's "prediction is no
longer the bound" result says extra right experts pay only if they are also *complete*. So "time" means one of:

1. **DRAM idle time that is not in the window today**, into which a prediction could post bytes;
2. **a later, better signal that costs no window**, because the bytes it would change are touched late anyway;
3. **a first layer that has no window at all today.**

Per-expert run cost by status (architecture doc §1): 31.0 µs complete, 36.4 partial, 42.4 not planned. Today, per
layer of a 4-token step: 7.7 complete, 2.4 partial, 5.2 not planned. The ceiling of the whole problem: all 15.3
complete saves 2.4 × 5.4 + 5.2 × 11.4 ≈ 72 µs a layer, ~3.0 ms a verify step over 42 MoE layers. The 5.2 not planned
are ~59 µs a layer, **~2.5 ms a verify step**. Every estimate below is measured against that 2.5.

Where the step's DRAM sits idle (*inference*, from the code and the measured laps): the post for layer l + 1 happens
in layer l's done path, after its run (`construct.rs` ~2830: "prefetched now that this layer's run is done (a run
abandons the prefetches in flight)"). So DRAM idles (a) inside each run's L3-served share (~55 % of the host lines
come from L3 at ~2.2× the DRAM rate, so roughly a third of a ~550 µs run, ~150–200 µs, reads little from DRAM), and
(b) across the step's non-MoE time: head 1.9 ms, the draft lap 3.1 ms at depth 1 / 6.0 ms at depth 2 (081), the three
dense layers. L3 capacity (~14.5 MB a CCD kept, about one layer's plan) is what keeps (b) from serving more than one
layer.

## 1. The drafter (question C proper)

### What the literature did

| Work | Signal and when | Reported number | Inputs it needs | Do we have them? |
|---|---|---|---|---|
| **SP-MoE** (Chen et al., 2025) [arXiv 2510.10302](https://arxiv.org/html/2510.10302v2) | During drafting, the *draft model's* attention output at layer l through the *target's* layer-l gate; prefetch layers 0..cutoff, the cutoff set by GPU memory and drafting time | Draft/target attention-output cosine up to 94.6 % (DeepSeek-V2-Lite with an AWQ draft); top-1 expert accuracy 88.9 % on V2-Lite, ~88 % Mixtral/Phi; TPOT −6.9 to −71 % vs baselines (vs AdapMoE: 8–17 %; on DeepSeek-Lite only 6.9–12.6 %); drafting is 16.2 % of decode time on their setups | A full-depth draft with the target's layer count and shapes (Mistral-7B for Mixtral, Phi-mini-MoE, a quantized V2-Lite) | **No.** Our draft is one MTP layer: no per-layer hidden states for layers 3–44 exist before the verify step |
| **MoE-SpeQ** (Wang et al., 2025) [arXiv 2511.14102](https://arxiv.org/html/2511.14102v1) | An INT4 copy of the target as the draft; its routing for k tokens × all layers (the "Expert Lookahead Buffer"), prefetched in phases: confident entries early, the rest once the draft completes | INT4 draft reproduces the FP16 top-4 set 90.9 % of the time on Qwen-MoE, against 84.7 % for a one-layer-ahead learned predictor; up to 2.34× over the best offloading baseline | The whole model a second time at 4 bits, run as the draft | **No** (that is a different speculation design; our experts are already int4 and host-bound) |
| **MoE-SpAc** (2026) [arXiv 2603.09983](https://arxiv.org/html/2603.09983v1) | The verify step's per-expert activation counts as a utility signal; prefetch during the draft phase | +42 % TPS over the best SD baseline (edge, Qwen3-30B-A3B, draft γ = 8, 17 % of experts cached) | A dense draft model and a GPU cache | Partly: the verify step's counts exist (`freq`, `last_picks`) |
| **Cascade** (Saxena et al., MICRO 2025) [arXiv 2506.20675](https://arxiv.org/abs/2506.20675) | Not prefetch: speculation utility = token gain ÷ verify cost; drafts touch 2–3× the weights | Slowdown limited to 5 % (from 1.5×), +7–14 % throughput over a fixed K | Per-step cost and acceptance | Yes (081 does the analogous gate on draft probability) |
| **EcoSpec** (Xie et al., 2026) [arXiv 2607.12696](https://arxiv.org/abs/2607.12696) | Draft selection that charges each draft path its marginal expert cost ("expert scattering") | Up to 1.62× on DeepSeek-V3.1 / Qwen3-235B / GPT-OSS-120B | A light expert predictor at draft time | Partly; a lever on bytes (fewer experts), not on prediction |

What carries over: SP-MoE's "cutoff layer" is the right shape for us — prefetch during the draft only as deep as the
memory that holds it allows. Our holder is the L3, which keeps about one layer's plan, so the cutoff is **the first
MoE layer**. The draft-model-as-predictor results (88–91 %) are not reachable here: they need a draft with the target's
depth.

### What our drafter actually offers before the verify step
- **The step's token ids** (the sampled token and up to three drafts): exact, known from the end of the draft lap. The
  verify step routes *these* tokens whether they are accepted or not, so acceptance (~0.9) does not dilute the signal.
- **The MTP layer's hidden state and its own routing** for each draft: a final-layer-like representation of the
  very position the verify step will process (the MTP input is the previous position's final hidden state plus the
  token's embedding). By *inference* it relates to the late layers' routing (33–44), where the route-ahead is
  already best (top-8 accuracy .76, recall@13 .63), and least to the early layers where it is weakest. Its own 288
  experts are a different expert space; a learned map would be needed. **Unmeasured** — the dump has no MTP state.
- **Time:** the draft lap plus head (~5–8 ms a step, the pool idle) — usable for the first MoE layer only (§0).

So the drafter's real value is for **layer 3**, the first MoE layer, which today gets **no prefetch at all**: the
post sits in a MoE layer's done path and predict.rs makes a pair only where l and l + 1 are both MoE (the trace's rows
are layers 4–44). Layer 3 has the longest window in the step (head + draft + dense layers 0–2) and no plan.
Candidate signals for it: layer 3's router over layer 2's dense-FFN input (the same route-ahead as every other layer,
on the card, ~35 µs at 4 tokens by analogy with the second router), the step's token ids, and the previous step's
layer-3 picks. Value (*inference*): at the early band's measured route-ahead recall (~.50 at cap 13; cap ~16 since
its window is long) ~8–9 of layer 3's ~16 host experts go from not planned to complete: ~95–110 µs a verify step
(~0.2 %).

## 2. Further ahead: l → l + 2, l + 3, whole-step prediction

### Measured on our model (pred2 dump, last 1,500 steps, 4,132 tokens; recall over all picks, no residency)
Layer l + d's router over layer l's normed FFN input (d = 1 reproduces the dump's route-ahead to int8 rounding,
max |Δ| 0.04). Per-token top-8 accuracy / step-union recall@13, by target-layer band:

| d | 4–9 | 10–20 | 21–32 | 33–44 | targets ≥ 9 (same set for every d) |
|---|---|---|---|---|---|
| 1 (today) | .645 / .503 | .661 / .563 | .695 / .590 | .758 / .631 | **.703 / .592** |
| 2 | .549 / .450 | .575 / .507 | .591 / .516 | .665 / .570 | **.608 / .529** |
| 3 | .503 / .423 | .529 / .475 | .513 / .461 | .600 / .524 | **.546 / .485** |
| 4 | .447 / .389 | .492 / .450 | .461 / .423 | .550 / .487 | **.500 / .451** |
| 6 | .475 / .412 | .412 / .388 | .394 / .372 | .468 / .426 | **.425 / .395** |

Each layer of lead costs ~.04–.06 recall@13 (−11 % for the second layer). Learned corrections (+~.03 at d = 1) would
by *inference* add about the same at d = 2.

### The literature on decay
- **HOBBIT** [arXiv 2411.01433](https://arxiv.org/html/2411.01433v2): Mixtral-8×7B (8 experts, top-2), current gate
  input through later gates: top-1 accuracy 96 % next layer, "around 90 %" for the next two or three. Eight experts
  hide the decay our 288 show.
- **ProMoE** [arXiv 2410.22134](https://arxiv.org/html/2410.22134v3): learned 2-layer MLP per layer (~2 M params)
  over the layer input; stride prefetching (prediction distance +1) costs ~5 % accuracy. Their token-ID,
  whole-iteration predictor (all layers known before the iteration) averages **58.3 %**: maximal lead, low accuracy.
- **Fate** [arXiv 2502.12224](https://arxiv.org/html/2502.12224v1): adjacent-layer gate inputs; 97.15 % prefetch
  accuracy reported (with over-fetch above a confidence threshold); 78.8 % exact by prediction as cited by 2511.10676.
- **PreScope/LayerScope** [arXiv 2509.23638](https://arxiv.org/html/2509.23638v1): layer-group-specific predictor
  (LLaPor, top-4 > 90 %, online learning) plus a cross-layer scheduler that plans prefetches over all layers instead
  of one; motivated by PCIe contention between prefetch and on-demand loads.
- **FineMoE/fMoE** [arXiv 2502.05370](https://arxiv.org/abs/2502.05370): prefetch distance d; layers 1..d from a
  semantic match of the prompt to past "expert maps", later layers from trajectory matching; +39 % hit rate, −47 %
  latency.

### Whole-step prediction from the step's own early picks (measured, trace.json, causal online tables)
Co-routing tables from this step's actual picks at layer l − d to layer l, learned online: recall@13 alone **.325
(d = 1), .327 (d = 10), .312 (d = 20), .307 (d = 30)** — flat with distance. With the source lagged by one step it
falls to .239, the recency level. So it is a "these tokens" signal, not a layer-pair one. Added to the route-ahead:
.588 → .593 (d = 1) / .594 (from layer 4), i.e. +.005–.006 — the same bound the architecture doc found for
reordering. Data-limited (231 steps for a 288 × 288 table a pair), so a longer run could show more; it would not
reach the route-ahead.

**Conclusion on lead:** predicting further ahead buys nothing here, because no layer can use more lead than the L3
holds (~one plan) and the extra lead is already available at d = 1 (§4). l → l + 2 costs .063 recall for a window
the L3 cannot fill.

## 3. Sequence-level signals

Measured adjacent-position overlap of the same layer's picks (pred2, consecutive tokens within a step; share of the 8
picks; random = 8/288 = .028): layers 3–9 **.09**, 10–20 **.25**, 21–32 **.30**, 33–43 **.31**; all .25 (min .05,
max .35). Same-layer history as a ranking (trace, recall@13 of the host set): last 1 step .210, 2 .225, 4 .234, 8
.239, 16 .243; recency-weighted EMA .223–.236; whole-run frequency .205.

Published counterpart: Mixtral's routing analysis (Table 5, [arXiv 2401.04088](https://arxiv.org/abs/2401.04088)):
consecutive tokens share the first-choice expert 14.0–14.1 % at layer 0 (random 12.5 %), 27.9–28.4 % at layer 15,
19.7–22.7 % at layer 31; first-or-second 44.9–46.5 % / 62.3–67.0 % / 44.5–52.9 % (random 46 %). The same shape as
ours: near random at the first layers, strongest in the middle. Relative to chance ours is stronger (9× at the middle
and late layers against Mixtral's ~2.2×), but in absolute terms one position shares ~2 of 8 experts with the next, so
recency's ceiling is ~.25 recall. That is where the .21 sits: at its ceiling, not an underused signal. The early
layers, where the route-ahead is weakest, are also where recency is weakest (.09) — the early layers' routing follows
the token (Mixtral: "more aligned with the syntax … especially at the initial and final layers"), which points at
token ids for layers 3–9 (unmeasured: the dump has no token ids).

## 4. Making the window longer (question 4)

### 4a. The plan for l + 1 exists a run earlier than it is posted (no new prediction)
The route-ahead for l + 1 comes from the same launch as layer l's router (`hc_route_ahead`), and `pred_step` takes
~52 µs. So the d = 1 plan could exist ~60–100 µs after run(l) *starts*, not at its end. Only the second margin
(router over x + shared output, +.005 offline) needs the shared FFN first. Measured precision of the d = 1 plan by
rank (raw route-ahead + bias, votes of per-token top-8, pred2 last 1,500 steps):

| rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P(used) | .96 | .94 | .92 | .91 | .89 | .87 | .84 | .80 | .75 | .70 | .64 | .61 | .56 | .54 | .52 | .52 | .48 | .44 |

The first ~8 of a plan are nearly certain from x_l, so they can be committed early. The tail (9–16) is a coin flip.
The DRAM idle inside run(l)'s L3-served share (~150–200 µs, §0) could carry ~3–5 more experts' bytes a layer
(*inference*). Spent on ranks 14–18 (precision ~.5) that is ~1.5–2.4 more complete hits: ~15–27 µs a layer,
**~0.6–1.1 ms a verify step**. The costs, which decide it: two layers' lines in one L3 at once (l's plan being read,
l + 1's arriving: the CAT keep/stream split of §6, perhaps alternating by layer parity), touch cores taken from a run
that is computing at the L3 rate, and today's rule that a run abandons a prefetch in flight. This is queue 090's
direction ("the prefetch streamed into the run") with one addition: 090 needs no new predictor — d = 1 is already
early enough.

### 4b. A later, better signal for the tail costs no window
The window touches a plan in rank order, so the tail is touched in the window's last ~40 %. A signal that arrives at
the window's start (`hc_post`, when the host's routed rows reach the card) can re-rank the tail before the touch gets
there, without delaying a byte. Candidates for that late signal:
- **The routed rows** (layer l's whole FFN output). Measured in phase3b as a second margin: +.027 recall@13 (crude
  sum, not the hyper-connection collapse). It is the largest single gain the dumps show.
- **Pre-attention prediction** ([arXiv 2511.10676](https://arxiv.org/abs/2511.10676)): predict layer l + 1's picks
  from its own pre-attention state with two linear layers and a ranking-aware BCE: exact-match **93.0 % on
  DeepSeek-V2-Lite**, 94.7 % Qwen3-30B-A3B, 97.6 % Phi-mini-MoE — +15 points over Fate's cross-layer 78.8 %.
  Here that state exists only after `hc_post`, i.e. at the window's start. Applied to the whole plan it would cost
  its latency out of the window: break-even Δrecall ≈ (latency / 520 µs) × 13 / 16.6, i.e. **+.075 at 50 µs**.
  Applied to the tail only, it costs nothing as long as it lands before the touch reaches rank ~9 (~300 µs in).
- **Default vectors** ("Speculating Experts", [arXiv 2603.19289](https://arxiv.org/html/2603.19289v1)): the
  routed output approximated at pick time as Σ g_e · d_{l,e}, with d_{l,e} each expert's mean output. That moves the
  routed rows' information *earlier* — to before `hc_post`, on the card — for 288 × 4096 values a layer. Their
  recall@k on Qwen3-30B-A3B is ~90 % past the first two layers. A small trained estimator (4–45 M parameters,
  converging in 4–5 M tokens) adds up to +25 points on the early layers.

Value of a refined tail (*inference*): if ranks 9–16 rise from ~.6 to ~.75 precision, that is ~1.2 more hits a layer,
~14 µs, **~0.6 ms a verify step**, and it stacks with 4a: the head is posted early, the tail late. This is the
"hierarchical, predicted-but-uncertain" shape MoE-SpeQ uses on its lookahead buffer (confident entries first, the rest
when the draft completes), applied within a layer.

### 4c. The step-level idle time
Head + draft + dense layers (~5–8 ms a step) is the step's longest idle DRAM stretch, but the L3 holds one plan. So it
serves layer 3 (§1) and nothing else on the CPU path. The only other place it could go is the cards' residency (BAR
writes ~21.6 GB/s from the cards' socket: ~100–150 MB, 8–11 experts in that stretch). That belongs to the
residency loader, not the prefetch, and is not pursued here.

## 5. What to add to the dump to settle these (question D, the time part)
1. **Token ids** per row (trace and pred dump): measures the token-id prior for layers 3–9 and for layer 3's plan.
2. **Layer 3's picks plus a layer-2 route-ahead** (layer 3's router over layer 2's FFN input): today's pred dump starts
   at layer 3's *picks* as a source, never as a target.
3. **The MTP layer's FFN input and its own picks** per draft, aligned to the verify step that processes those tokens:
   needed for any statement about the drafter's state as a feature.
4. **The pre-attention state of l + 1** (`hc_post`'s output collapsed as the next layer's `hc_pre` sees it, or
   simply layer l + 1's FFN input, which the dump already has one record later): it lets the tail re-ranking of 4b
   be scored in replay. Note that X at record l + 1 is the *post-attention* input, so a pre-attention predictor
   needs this state written explicitly.
5. Per layer, from prow: the run's L3-served span and the DRAM rate inside it. This is the measured size of 4a's
   window, which today rests on the 55 % / 2.2× arithmetic.

## 6. Ranked conclusion

What would most increase bytes in time for the host experts, best first:

1. **Post the d = 1 plan's head at run(l)'s start, into the DRAM idle inside the run** (4a). The signal (route-ahead
   + layer l's picks) already exists ~60–100 µs into run(l); the top 8 of a plan are 80–96 % precise. Estimated
   **~0.6–1.1 ms a verify step**: ~1.5–2.4 of the 5.2 not-planned experts a layer become complete, by extending the
   cap where time allows. This is *inference*; L3 co-residency of two layers' lines and core sharing decide it.
   Measure first: the run's L3-served span (dump item 5). It is 090 with its predictor question answered (d = 1 is
   enough; no l + 2).
2. **Re-rank the plan's tail at `hc_post` with a later, better signal** (4b): routed rows (+.027 measured, crude),
   default vectors (moving that information earlier), or a pre-attention predictor (93 % exact-match on V2-Lite in
   the literature). It is free in window time because the tail is touched last. Estimated **~0.4–0.8 ms a verify
   step** (*inference*); it stacks with 1. It attacks the not-planned 5.2 directly.
3. **Give layer 3 a plan** (§1): the one MoE layer with no prefetch, and the longest idle window in the step (head,
   draft, dense layers). A layer-2 route-ahead plus the step's token ids. **~0.1 ms a verify step**: small but
   certain and cheap. This is the only place the drafter's output (the step's tokens, known early) buys time on this
   box.
4. **Not recommended:** l → l + 2 or deeper (measured −.063 / −.107 / −.141 recall@13 at d = 2 / 3 / 4 for lead the
   L3 cannot hold); whole-step priors from co-routing or recency (measured +.005–.01; recency is at its ~.25 ceiling);
   the drafter's hidden state as a per-layer predictor (the published 88–91 % results need a full-depth draft, and
   our MTP layer has no per-layer states).

Against the ~5.2 not-planned host experts a layer (~2.5 ms a verify step at most): 1 and 2 together are estimated at
**~1–1.9 ms**, i.e. ~2–4 of the 5.2 moved to complete. That is an upper-leaning estimate: it assumes the L3 keeps both
layers' lines, and it is unmeasured until dump items 4–5 exist. The remainder is per-token router accuracy on the
early layers. Only a better estimate of l + 1's input (question B) reaches it, not more time.
