# GLM-5.3-Flash: next steps

The overview of the work on the serving node for GLM-5.3-Flash on metatron (dual EPYC 7R13,
2× Radeon 7900 XT on socket 1, NUMA nodes 2 and 3). Each concept lives in its own numbered entry under
[`queue/`](queue/), in one format — status, area, gain, proposal, source, dated history, next —
and new concepts take the next number. This page holds the state, the rules, the vocabulary and
the index. Reorganized 2026-10-18 from the single-page version (in git history before then).

## Where the node stands (2026-10-18)

- **Decode: ~24–25 ms an emitted token** (~41 tok/s) at short context — speculative decoding at
  depth 1 (024) and experts resident on the cards (003); from 36 two days earlier. A two-token
  verify step 46–48 ms; a step token: kda card 8.6, experts 6.0, mla 2.7, router 2.6, draft 1.8,
  head 1.1, slot 0.7. Target: **20 ms** (tcdent).
- **Prefill: 5.5 ms a prompt token** (~180 tok/s) at short positions in 512-token steps,
  6.7–6.8 at 18k — the WMMA rows pair (009) and the bf16 attention state (030). Experts 2.2 of it
  on the CPU. The wait that grows with position adds ~0.08 ms a prompt token per 1k positions
  (~80 ms at 1M as the attend and scorer stand: 018, 019).
- **Reference point:** tinygrad PR #18863 on six R9700s — the PR title 300 / 53 (prefill / decode
  tok/s), geohot's tweet of 2026-10-18 3,000 / 150 at a 256k context. The prefill gap is
  analysed in 043 and 037–039.

## Rules

- Serial first: the fastest single request is the priority.
- Decode kernels keep their bit-identity rules. Prefill may have its own kernels where the gain
  is real (tcdent, 2026-10-17):

  > it might be time for us to start thinking about prefill as its own set of auxiliary kernels
  > … in the interest of keeping our project clean and getting prefill as fast as possible, I
  > think that we can maintain parallel kernels where appropriate.

  A kernel pair is gated numerically against the single-token path plus the NLL and eval gates;
  bit-identical is still preferred where it is free.
- Prefill is its own subsystem that shares conventions; what it inherits from generation is
  inventoried and quantified (043).
- A step change gates on steps longer than one 32-token pass and one hc_pre round, every
  position's logits compared.
- The attention state follows the reference; deviations are experiments (030, 032).
- Approximate methods enter only with a measured cost (KL and top-1 against the oracle, the
  evals; a paired comparison on ~400 problems decides a difference the size of 033's).
- Measure before building where a cheap measurement can decide. Compare builds in ABBA blocks
  on one boot where drift matters (vLLM PR #61868's protocol: every on-block against every
  off-block, paired by request).
- Once a measurement picks a shape, the switch that compared the shapes goes.
- The timeline stays on while the work needs it; turning it off is a late saving (044).

- **Nothing with a gain is pruned** (tcdent, 2026-10-19: "we are not arbitrarily pruning techniques which have
  theoretical or measured gains no matter what the number actually is"). A technique with a theoretical or measured
  gain keeps an entry of its own with the number; a shelf note inside another entry's text is promoted when that
  entry closes.

## Status vocabulary

| Status | Meaning |
|---|---|
| **Resolved** | built and measured, or measured and decided |
| **Rejected (measured)** | built or measured; lost, with the number |
| **Deferred — small gain** | worth something, judged too small for now (mostly sub-millisecond); kept so the technique is not lost when small gains are picked up |
| **Deferred — waits on …** | blocked by another entry or a measurement |
| **Open** | proposed, not built; may need a measurement first |
| **Tentative (future)** | an idea held for later by decision |
| **Experiment (off-reference / approximate)** | a deviation from the reference; enters only with its eval cost |
| **Decision (tcdent)** | the analysis is done; the choice is tcdent's |
| **Set aside** | ruled out by reasoning, not measurement; reopen if the reason changes |

## Index

| # | Concept | Status | Area |
|---|---|---|---|
| [001](queue/001-peer-vram-writes.md) | Peer VRAM writes for the handoffs | Quantified, on the shelf — ≤ 0.1 ms a step token (decode), ~0.04 ms a prompt token (prefill); the helper and the benches exist, the handoff refactor is the work left | decode, prefill |
| [002](queue/002-card-controlled-expert-phase.md) | The expert phase controlled by a card | Built: the FFN card leads the routed experts, a verify step 45.9 → 43.5 ms | decode |
| [003](queue/003-experts-resident-on-cards.md) | Routed experts resident on the cards | Resolved (decode) | decode |
| [004](queue/004-expert-cache-policy.md) | The expert cache's replacement policy (ARC) | Done (aged counts + a layer turn for the one load a step; measured 2026-10-18 10:29) | decode |
| [005](queue/005-layer-ahead-routing.md) | Layer-ahead routing for the prefetch | Resolved | decode |
| [006](queue/006-routing-boundary.md) | The 8th/9th routing boundary: resolve exactly or exploit | Decision (tcdent) — the speed half estimated 2026-10-18 at ~0.02 ms a step token; the determinism half is not a speed item | decode, determinism |
| [007](queue/007-rows-kernels-r-rows-a-wave.md) | Rows kernels: R rows a wave on the multi-token width | Resolved | prefill |
| [008](queue/008-few-token-pass-width.md) | A pass width for 2–4 tokens (verify steps) | Open, one step taken 2026-10-18 — the f16-transplant decode deployed (−15–18 % on `ffn_gu`, bit-identical); `ffn_down_shared`'s limiter still to find | decode (verify) |
| [009](queue/009-wmma-rows-prefill.md) | WMMA rows kernels for prefill | Resolved; the efficiency round measured 2026-10-18 (small ceiling: the pair is ~8 % of the step's card time) | prefill |
| [010](queue/010-ozaki-exact-products.md) | Ozaki scheme: exact products from int8 slices | Rejected (for the rows pair) — kept for any future order-independence need | prefill, determinism |
| [011](queue/011-iu8-wmma-rate-recheck.md) | iu8 WMMA rate recheck | Resolved (measured 2026-10-18) — eight chains 87.4 TOP/s, four 87.1, bf16 84.8: iu8 WMMA runs at the bf16 rate on gfx1100 | prefill |
| [012](queue/012-attend-one-launch-a-step.md) | The MLA attend in one launch a step | Resolved | prefill |
| [013](queue/013-indexer-on-card.md) | The indexer's scoring and selection on the card | Resolved | decode, prefill, long context |
| [014](queue/014-attend-tiled-across-queries.md) | Attend tiled across consecutive queries | Rejected (measured) | prefill |
| [015](queue/015-attend-register-tiled.md) | Attend scores register-tiled (four positions a thread) | Rejected (measured) | prefill, decode |
| [016](queue/016-attend-heads-per-workgroup.md) | Attend: four heads a workgroup | Rejected (measured) for decode; Quantified, on the shelf for prefill — 0.1 ms a prompt token (the kernel alone: 151 → 142 µs a token a layer at a 512-step, −6 %); superseded by 018 if it is built | decode, prefill |
| [017](queue/017-attend-split-positions.md) | Attend: positions split across workgroups (decode at long context) | Quantified (bench, 2026-10-18) — the attend is 0.38–0.44 ms a layer a verify step from 2,050 positions on (~4.4 ms a step over 11 layers): the 24 → 31 ms step at long context; ready to build with 018's shape | decode, long context |
| [018](queue/018-attend-wmma-prefill.md) | Sparse attend on WMMA for prefill (heads as rows) | Quantified (bench, 2026-10-18) — the attend is 151–160 µs a token a layer at a 512-step from 2k positions on: ~1.7 ms a prompt token over 11 layers, a quarter of the 6.7; ready to build | prefill, long context |
| [019](queue/019-indexer-wmma-scoring.md) | Indexer scoring on WMMA | Quantified (bench, 2026-10-18) — `ix_score` is 1.6 µs a token per 1k positions a layer, linear: 0.3 ms a prompt token at 18k, 4.7 at 262k, ~18 at 1M; a decode step at 262k 14 ms of scoring, at 1M ~56; `ix_select` at decode 0.07 ms a layer at 65k, ~0.7 at 1M | long context (prefill and decode) |
| [020](queue/020-cpu-expert-gemm.md) | A multi-token INT4×INT8 expert kernel on the CPU | Resolved | prefill |
| [021](queue/021-cpu-expert-phase-bound.md) | What bounds the CPU expert phase in prefill | Open — a measurement first | prefill |
| [022](queue/022-hc-pre-prefill.md) | hc_pre for prefill | Deferred — small gain | prefill |
| [023](queue/023-chunked-kda.md) | Chunked KDA (WY form, FlashKDA) | Deferred — small gain for prefill; returns with longer verify steps | prefill, decode (verify) |
| [024](queue/024-mtp-depth-1.md) | Speculative decoding with the draft layer (MTP), depth 1 | Resolved | decode |
| [025](queue/025-mtp-depth-2.md) | Draft depth 2 | Deferred — waited on the verify step's cost; ready to re-measure | decode |
| [026](queue/026-draft-shares-target-indexer.md) | The draft layer reuses the target's indexer picks | Deferred — small gain | decode |
| [027](queue/027-model-free-drafts.md) | Model-free drafts from the prompt's own n-grams (SuffixDecoding) | Open | decode |
| [028](queue/028-persistent-per-layer-kernels.md) | Persistent per-layer kernels | Deferred — gain judged too small for the effort (≤ ~1.5 ms an emitted token) | decode |
| [029](queue/029-kda-projection-bytes.md) | The KDA projections' bytes (DFloat11 or FP8) | Decision (tcdent) for FP8; Open for DFloat11 | decode |
| [030](queue/030-attention-state-reference-precision.md) | The attention state at the reference's precision | Resolved (decided; both phases built and deployed) | long context, decode, prefill |
| [031](queue/031-latent-mirror-by-context.md) | Latent mirror and residency sized by the context in use | Open | long context, decode |
| [032](queue/032-precision-experiments.md) | Attention-state precision below the reference | Experiment (off-reference) | long context |
| [033](queue/033-halving-k.md) | Halving k with a fixed reference mass | Rejected (measured) | decode |
| [034](queue/034-expert-deferral.md) | Expert deferral | Experiment (approximate) — not evaluated | decode |
| [035](queue/035-intra-expert-sparsity.md) | Intra-expert activation sparsity | Experiment (approximate) — not evaluated | decode |
| [036](queue/036-bit-plane-expert-storage.md) | Bit-plane expert storage | Experiment (approximate) — not evaluated | decode |
| [037](queue/037-prefill-chunk-wavefront.md) | Prefill chunks in flight as a wavefront | Built: 4.04 → 2.84 ms a prompt token | prefill |
| [038](queue/038-resident-experts-in-prefill.md) | Resident experts in prefill | Open | prefill |
| [039](queue/039-expert-staging-for-prefill.md) | Streaming experts through the cards for large prefill steps | Tentative (future) | prefill, long context |
| [040](queue/040-head-vocab-split.md) | Split the head's vocabulary across both cards | Open | decode |
| [041](queue/041-router-lap-trace.md) | Trace the router lap | Resolved: the host's lap is gone (064, 001, 002); the router's cost is on the FFN card, `hc_pre` (078 for prefill) | decode |
| [042](queue/042-prefill-utilization-report.md) | A per-step utilization report for prefill | Open — an instrument | prefill |
| [043](queue/043-prefill-inherited-patterns.md) | What prefill inherits from generation | Open — a living inventory | prefill |
| [044](queue/044-timeline-offset.md) | The timeline instrument's own cost | Deferred — small gain (a known late saving) | decode, prefill |
| [045](queue/045-residency-loader-sdma.md) | Fill residency slots with the card's copy engine | Open | decode, prefill (039) |
| [046](queue/046-context-store-state-accounting.md) | The context store charges kept states to its budget | Resolved | serving |
| [047](queue/047-kda-decode-narrow-tiles.md) | KDA decode: narrow value tiles to widen the grid | Deferred — small gain | decode |
| [048](queue/048-rms-fold-into-quantizer.md) | Fold the RMS norm into the activation quantizer | Deferred — small gain | decode |
| [049](queue/049-split-k-decode-linears.md) | Split-K decode linears reduced through LDS | Deferred — small gain | decode |
| [050](queue/050-deltalog-kda-state.md) | DeltaLog for the KDA state | Deferred — small gain | decode |
| [051](queue/051-indexer-block-max-pruning.md) | Exact top-2048 indexer with block-max pruning | Deferred — speculative gain; matters at long context | long context |
| [052](queue/052-sequence-parallel-hc.md) | Sequence-parallel hyper-connections in prefill | Deferred — small gain | prefill |
| [053](queue/053-draft-in-prefill-wavefront.md) | The draft layer as one more layer of the prefill wavefront | Deferred — small gain | prefill |
| [054](queue/054-lut-gemm-prefill-recheck.md) | LUT GEMM for the CPU experts in prefill | Deferred — needs 021's answer first | prefill |
| [055](queue/055-int4-expert-entropy.md) | Entropy of the INT4 expert nibbles | Deferred — likely a dead end; a cheap check | decode |
| [056](queue/056-considered-and-set-aside.md) | Considered and set aside (by reasoning, not measurement) | Set aside — reopen if the stated reason changes | various |
| [057](queue/057-side-queues-decode.md) | Side queues for independent decode work | Open | decode |
| [058](queue/058-mla-layers-across-both-cards.md) | Indexer and attend of an MLA layer across both cards | Built (the attend, by positions): prefill 2.76 → 2.65 ms a prompt token, numeric | long context (decode and prefill) |
| [059](queue/059-host-loop-occupancy.md) | How much of the decode step the host loop holds | Open — a measurement | decode |
| [060](queue/060-decode-occupancy-audit.md) | Decode occupancy audit: what to fuse, what to split | Open — an audit, then our own experiments | decode (and the verify step's 2–4 tokens) |
| [061](queue/061-gpu-architecture-quantified.md) | The cards' architecture, quantified | Open — topology read and probes first | decode, prefill (every card kernel) |
| [062](queue/062-profile-placement-split.md) | Separate placement training from the profile | Open — a refactor candidate, not scheduled | decode, prefill (CPU model, placement retrain) |
| [063](queue/063-residency-filled-in-prefill.md) | Residency filled during prefill, from the prompt's own routing | Open — the largest exact decode lever at long context | decode (long context), prefill |
| [064](queue/064-host-laps-off-the-critical-path.md) | The host's between-phase laps off the critical path | Done: 001 (input side) and 002 (output side) | decode |
| [065](queue/065-expert-activation-reread.md) | The expert kernel's activation re-read a part | Quantified, on the shelf — ≤ 0.13 ms a prompt token | prefill |
| [066](queue/066-expert-phase-quantize-and-posting.md) | The expert phase's quantize run and posting | Quantified, on the shelf — ~0.13 ms a prompt token | prefill |
| [067](queue/067-scalar-qabs-at-decode.md) | The scalar q_abs at a verify step | Quantified, on the shelf — 0.4 ms a verify step | decode |
| [068](queue/068-attend-tile-loop.md) | The WMMA attend's tile loop | Quantified, on the shelf — ~5 µs a tile, ~1.3 ms a verify step at 18k | decode, prefill (long context) |
| [069](queue/069-select-round-trips.md) | The indexer selection's round trips at long context | Quantified, on the shelf — ~0.7 ms a verify step past 32k | long context |
| [070](queue/070-scorer-short-context-twin.md) | The indexer scorer's short-context form | Quantified, on the shelf — 0.05 ms a verify step | decode |
| [071](queue/071-prefill-residency-churn.md) | The residency's replacement churn through a prompt | Quantified, on the shelf — ~3 % of prefill at 512 | prefill |
| [072](queue/072-head-parts-readback.md) | The split head's readback | Quantified, on the shelf — ≤ 0.1 ms a step token | decode |
| [073](queue/073-grouped-expert-kernel-rate.md) | The grouped expert kernel's own rate | Quantified, on the shelf — none today; needed before more picks move to the cards | prefill |
| [074](queue/074-decode-expert-runs-idle.md) | The decode expert runs' idle | Open — up to ~4 ms a verify step measured | decode |
| [075](queue/075-step-2048-under-the-wavefront.md) | `STEP` 2048 under the wavefront | Quantified, on the shelf — prefill −0.19 ms a prompt token against two resident slots a layer and ~+1 ms a verify step | prefill, decode |
| [076](queue/076-step-width-deep-dive.md) | The step's width under the wavefront: a deep dive | Open — a study (chunk width vs buffer depth, the residency, the draft lap) | prefill, decode |
| [077](queue/077-warm-next-layer-weights.md) | Warm the next layer's weights while the cards wait for the host | Open — an experiment | decode |
| [078](queue/078-hc-pre-prefill-twin.md) | `hc_pre`'s matrix-path twin for prefill | Built — the exact fixes and the router's matrix path: prefill 2.65 → 2.14–2.21 ms a prompt token; prefill now CPU-expert bound | prefill |
| [079](queue/079-mla-o-rate.md) | The MLA output projection's rate in prefill | Open — 4.6 ms a 512-token launch, est. well under 1 ms | prefill |
| [080](queue/080-launch-granularity.md) | Launch and timeline granularity: steps cut too fine | Open — noted (`rows_d2`'s 32-token passes, ~530 launches a card a prefill step) | prefill, profiling |
| [081](queue/081-confidence-gated-draft-depth.md) | Draft depth chosen a step by the draft's confidence | Deployed — gated depth 3: 18k 26.6–27.2 → 24.8–25.0 ms an emitted token; the draft instrumented | decode |
| [082](queue/082-card-side-expert-prediction.md) | Expert prediction on the cards' idle time | Open — designed: on card0 in the host experts' gap, per-token probabilities, learned online; prefetch yield measured: 31 / 36 / 42 µs an expert complete / partial / missed | decode, prefill |
| [083](queue/083-pool-posting-on-the-main-thread.md) | The pool's posting on the main thread | Built — the main thread's overhead outside the pool's runs 3.5 → ~1.0 ms a verify step (common::alloc, rows off the main thread, the plan's stores prewarmed); step 55.75 → 53.0 | decode |
| [084](queue/084-one-token-gemv-from-l3.md) | The one-token gemv from L3 under full load | Open — ~9 GB/s a core on 92 cores vs a load loop's ~55 | decode |
| [085](queue/085-pool-core-quiet.md) | The pool's cores kept quiet | Open — 1 kHz idle wakes, ticks on busy cores, the NMI watchdog | decode, prefill |
| [086](queue/086-predictor-input-map.md) | The prefetch predictor's input map | Quantified, on the shelf — −0.10 missed a layer, all of them partial: no time gained | decode |
| [087](queue/087-predictor-on-the-main-thread.md) | The prefetch predictor's main-thread time | Done — the predictor on the cards, slot 1.67 → 0.93 ms, 25.4 ms an emitted token | decode |
| [088](queue/088-prefill-labels-off-the-main-thread.md) | Prefill's predictor labels off the main thread | Done — lesson buffers reused, prefill 1.87 → 1.85 | prefill |
| [089](queue/089-mla-o-in-the-verify-step.md) | The MLA output projection in a verify step | Open — ~142 µs a launch, ~1.6 ms a verify step on the critical path | decode |
| [090](queue/090-prefetch-streamed-into-the-run.md) | The prefetch streamed into the run | Open — a direction: arrival timed to the run's consumption order | decode |
| [091](queue/091-prefetch-inside-the-window.md) | The prefetch inside the window | Measured to the box's ceiling (316 GB/s on fresh bytes against a STREAM Triad of ~293): the rate is the box's, the lever is bytes | decode |
| [092](queue/092-attention-ahead-predictor.md) | Attention-ahead: layer l+1's own attention as the predictor's margin | Quantified, on the shelf — +.038 settled recall@13 for ~150 MB a layer; the small model is within .02 of any learned model on these inputs | decode |
| [093](queue/093-routed-rows-as-the-signal.md) | The routed rows as the predictor's signal | Built (aff36): pred_step behind hc_post with the whole FFN output as the margin's input; offline +.027 settled | decode |

## Order (2026-10-18, evening — the 20 ms path)

tcdent's target is 20 ms an emitted token at long context (50 tok/s); the baseline is run e03735b6
(a 54 ms two-token verify step at 18k, 30 ms an emitted token from the steps). Measured and tuned at
≥ 10k positions only.

1. 063 (residency filled during prefill): ~6 ms a step, exact.
2. 040 (the head split across the cards): ~2 ms a step; 064 and 041 (the host laps): 2–3 ms.
3. 028 (persistent per-layer kernels): ~3 ms a step, after a track-launches count.
4. The shelf items in reach (047, the scalar q_abs, 001): ~1 ms.
5. Then tcdent's decision between 029 (FP8 KDA projections, 5–6 ms a step) and 025 + 026 (depth 2,
   a multiplier once the step is cheaper) — either reaches 20 on paper.
6. Prefill as its own track: 042, 037, 038, 018/019 at longer positions.
7. Prefill next (tcdent, 2026-10-20): 058 (the MLA attend across both cards: the MLA layers' card chain is ~7 ms longer
   than the host's experts a chunk), then 078 (`hc_pre`'s prefill twin), then 079 (`mla_o_w`'s rate).

## Decisions open for tcdent

- 006: resolve routing ties exactly, or exploit them for placement.
- 029: FP8 for the KDA projections.
- 034–036: which approximate methods get an eval run.
