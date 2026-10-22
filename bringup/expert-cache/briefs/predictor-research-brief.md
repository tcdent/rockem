# Brief — the expert predictor's architecture (research, 2026-10-22)

tcdent: "research modeling architectures that we want to use given the shape of the information that we have
available for prediction, so that we can determine what the optimal architecture is … I'm sure that there's some
research that's come out in the last year or perhaps two that pushes that forward."

## The system in one paragraph
GLM-5.3-Flash (MoE: `E` routed experts a layer, `TOP_K` picks a token, one shared expert; ~46 MoE layers; constants in
models/glm-5.3-flash/src/lib.rs, generated from the bundle's config — read them, do not assume) served on one box:
attention and the shared expert on two consumer GPUs, the routed experts' weights (int4, 13.4 MB an expert) in host
DRAM, run by a 92-core CPU pool. Decode is speculative (MTP: a draft of up to 3 tokens, verified in a step of 2–4
tokens; acceptance ~0.9). A step walks the layers; at each MoE layer the host runs the experts the step's tokens picked
(~15 distinct host experts a layer for a 4-token step), reading their bytes from DRAM. **The prefetch**: during the
card's attention of layer l (~500 µs), the pool pulls the experts *predicted* for layer l + 1 into the CCDs' L3s
(14.5 MB a CCD, 174 MB box-wide for a 13-expert plan). The box streams ~316–322 GB/s (its STREAM Triad is ~293; see
docs/metatron-cpu.md §2), so 13 experts take ~550 µs against a ~500 µs window: **the rate is the hardware's; the lever
is bytes — predict the right experts and fewer of them.**

## What the predictor is today (docs/expert-cache-architecture.md §3, models/glm-5.3-flash/src/plan.rs, predict.rs)
Per layer pair (l → l + 1), online, on the layer's GPU, in idle time (~180 µs a layer available):
- `logit[e] = route_ahead[e] + corrections`, where `route_ahead` = layer l + 1's *router* applied to layer l's FFN
  input x_l (the next layer's router on a stale hidden state: before layer l's FFN and before layer l + 1's attention).
  A second margin applies the router to x_l + the shared expert's output.
- Corrections, learned online with per-expert BCE + Adam, labels = layer l + 1's actual picks (arrive one layer later):
  a bias an expert, a co-routing table [E][E] from layer l's actual picks (row: an expert picked here → experts at
  l + 1), calibration scalars on the margins. State f32 ~2·E² + small, per pair.
- Plan: the union over the step's tokens ranked by Π(1 + e^z), top `PLAN` = 24 → host drops the experts resident on
  the cards and prefetches up to 13.

Measured: settled recall@13 ≈ 0.62 offline (bringup/expert-cache/phase3b). In production, per layer of a 4-token
step: ~15.3 host experts needed; 7.7 arrive complete, 2.4 partial, 5.2 not planned; of the 13 planned ~5 go unused.
The route-ahead's own top-8 accuracy on the weak layers is .51–.62: the error is per-token router accuracy on a stale
input. Tried and exhausted on the ranking side (replay run 9593789a): wider top-k, previous step's picks at l + 1
(alone recall .21; as a vote hurts; as tie-break ~0), adaptive per-layer trust (+.001), calibration (+.01).
Shelved: a rank-64 low-rank map over x_l (queue 086: +.007 recall measured offline, learned online from zero — the research agents trace the small gain to online learning and to the delta being high-rank).

## The information available at prediction time (layer l's FFN phase, on the GPU)
1. x_l — layer l's FFN input (post-attention residual) for each of the step's 2–4 tokens, f32/bf16, H wide.
2. Layer l's router logits (all E) and its TOP_K picks per token (exact).
3. route_ahead: layer l + 1's router rows · x_l (and · (x_l + shared output)) — E logits a token.
4. The previous step's picks at layer l + 1 (per layer ring), and in principle the full history of picks per layer per
   position of the sequence (not kept today beyond one step).
5. Token ids and positions of the step's tokens; which tokens are the draft's (speculative) and the acceptance rate.
6. The residency table: which experts of l + 1 are already on the cards (those need no prefetch).
7. NOT available: layer l + 1's attention output (that is exactly what the next router sees), the next layer's weights
   beyond the router rows (the full model is there in VRAM/DRAM but running it is the forward itself).
8. Anything computable from the above with a small model: the cards have ~12 GB VRAM free each and ~180 µs idle a
   layer; kernels are written in Rust for gfx1100 (WMMA bf16 85 TFLOP/s measured) — a few-MFLOP model a token is
   free, a few-GFLOP one is not. Online learning is preferred (no offline training pipeline exists today, but routing
   traces exist and could pretrain).

## What we have to evaluate offline
bringup/expert-cache/data/trace.json + use.json: a decode run's rows (step time, layer, n tokens, per-token candidate
lists (route-ahead ranked), actual picks, residency flags, previous step's picks). bringup/expert-cache/replay/*.py:
`load.py` reads it, `replay.py` scores a ranking function by recall at a cap against the host set (`run(scorefn, cap,
…)`). The hidden states x_l are NOT in the trace today (they could be dumped: bringup/box/dumppred.sh dumps per-layer
pred inputs for a run — `dump-pred` feature). bench-pred checks predict.rs against plan.rs.

## Questions for the research
A. **Literature, 2024–2026.** MoE expert prediction / prefetching for offloaded or memory-constrained inference:
   pre-gating and lookahead routers (Pre-gated MoE, SiDA, ExpertFlow, ProMoE, AdapMoE, MoE-Infinity, Lina, HOBBIT,
   Klotski, DAOP, fMoE, MoE-Lightning, "expert routing speculation", any 2026 work), cross-layer routing
   correlation, learned predictors on hidden states, token-level expert affinity, speculative-decoding-aware expert
   prefetch (drafts know the next tokens). For each: inputs used, reported top-k recall/accuracy, model size and
   cost, whether online or trained offline, URL.
B. **The architecture for our signal shape.** Given 1–8 above and the budget, 2–3 candidate architectures ranked,
   each with: inputs, parameters, FLOPs a token, expected recall@13 (with the reasoning or the paper it rests on), how
   it learns (online / pretrained from traces / both), the failure mode. In particular: what does a predictor need to
   recover layer l + 1's attention contribution from x_l (depth, rank, nonlinearity; is a distilled "attention
   surrogate" per layer pair the right object), and does predicting *two* layers ahead (l → l + 2, doubling the
   window) survive the accuracy loss.
C. **The drafter as a signal.** The step's tokens are known ahead (the draft) with ~0.9 acceptance; the drafter's own
   routing or hidden states exist before the verify step. Is there work using speculative tokens to prefetch experts
   for the *next* step's layers, and what would it buy here?
D. **How to validate before building**: what to add to the trace dump (x_l? the shared output? the previous layers'
   picks?) so the candidate can be scored in replay.py-style Python against recall@13 and bytes, and a rough order of
   the work.

Write findings as a document with sources (URLs), numbers, and a ranked recommendation; mark inference as inference.
Do not change code. Do not delegate to other agents.
