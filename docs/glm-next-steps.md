# GLM-5.3-Flash: next steps

Draft, 2026-10-17. Sets up the next round of work on the serving node for GLM-5.3-Flash on
metatron (dual EPYC 7R13, 2× Radeon 7900 XT on socket 0). Started from the build at `de37280`;
statuses updated at `aa5324d`. Sources for the outside work cited here:
`/tmp/glm-prefill-vs-tinygrad.md`, `/tmp/glm-perf-research-2026-10-17.md`,
`/tmp/glm-fresh-lens-2026-10-17.md`.

Every item carries a status: **Resolved** (built and measured, or measured and decided),
**Tried, not needed** (built or measured, and set aside with the reason), or **Unimplemented**
(with what it waits on).

## Where the node stands

**Decode: ~30.2 ms an emitted token** at short context with speculative decoding (draft depth 1,
acceptance 0.92 on the gate text, 0.84–0.90 in real sessions; 29 ms a step token), from 36
without it. At 18k positions 34 ms (41 before speculation, 48.9 before the indexer moved to the
card). Target for this round: 29 ms (tcdent).

**Prefill: 6.2 ms a prompt token** in 512-token steps at short context, 8.2 at 3k positions
(from 36 one token at a time); an 18k prompt 8.3 ms a token cold. At 512: KDA card chain 2.8,
MLA projections 0.9, experts 2.2 against a 1.0 ms DRAM floor; the indexer's selection ~0 at 3k.

**What each phase is held by (measured):**

- Prefill card kernels (timeline of a 32-token step, card 0, before R rows a wave): rows_h
  1.01 ms a prompt token, hc_pre 0.62, ffn_gu 0.50, the down projections 0.44, rows_part 0.30,
  mla_attend 0.26, mla_o 0.20, kda_step 0.13. Rows launches over 32 tokens still sit well above
  the 125 µs weight-streaming floor after R rows a wave (not re-timelined).
- Prefill experts: not the dot kernel. Items are already placed per node by each node's
  measured stream rate (`layout::placement_from`, moving only the difference; node 2's smaller
  share, 25.8 / 25.3 / 23.1 / 25.8 %, is that placement). A 512-step's per-node runs still end
  7.7–8.6 ms apart; what holds the expert phase at 2.2 against the 1.0 floor is not yet traced.
- Long-context attend at decode: limited by too few workgroups (one token is 32 workgroups on
  84 CUs), not by L2.
- Decode: the token alternates between the cards (attention, hyper-connections, router) and the
  CPU (routed experts); each idles while the other works. A verify step's expert phase is the
  union of its tokens' picks (23 ms for two tokens, 31.5 for three).

**Status 2026-10-17 evening:** ~29.7 ms an emitted token at short context and ~32 at 18k, by
speculative decoding (§6): a two-token verify step of 57 ms at 0.92 acceptance (28.1–29.0 ms a
step token: kda card 9.3, mla 2.8, experts 10.5, router 1.8, head 1.1, draft 1.9, slot 1.0). The
indexer's selection is on the card (attention lap 6.3 → 0.0 at 18k). New target: **24 ms**
(tcdent). The arithmetic puts it in §3: the expert phase (21 ms a step) is at the DRAM floor and
the card phase has ~3 ms left in it.

## Rules this round keeps

- Serial first: the fastest single request is the priority.
- Decode kernels keep their bit-identity rules. Prefill may have its own kernels where the gain
  is real (tcdent, 2026-10-17):

  > it might be time for us to start thinking about prefill as its own set of auxiliary kernels
  > … in the interest of keeping our project clean and getting prefill as fast as possible, I
  > think that we can maintain parallel kernels where appropriate.

  A kernel pair is gated numerically against the single-token path plus the NLL and eval gates;
  bit-identical is still preferred where it is free.
- A step change gates on steps longer than one 32-token pass and one hc_pre round, every
  position's logits compared.
- Approximate methods enter only with a measured cost (KL and top-1 against the oracle, the
  evals in `evals.toml`).
- Measure before building where a cheap measurement can decide.
- Once a measurement picks a shape, the switch that compared the shapes goes (tcdent,
  2026-10-17).
- The timeline instrument stays on while the work needs it; turning it off is a known late
  saving (tcdent, 2026-10-17). It adds a completion signal to every launch, so per-phase
  attribution with it on can mislead; build-to-build comparisons with it on stay valid.
  **Unimplemented:** one measurement of the current build with the timeline on and off, to
  know the offset.

## 1. Direct writes between the producers and the cards

Both cards sit on socket 0 with peer access available. Every card-to-card handoff today goes
through uncached host memory: two PCIe trips (one card writes host memory, the other reads it),
about 90 or more a token. A peer write into the receiver's VRAM is one trip, and the receiver
polls its own memory.

**Status: Unimplemented.** No configuration change is needed: the kernel config already has
`CONFIG_HSA_AMD_P2P=y` and `CONFIG_PCI_P2PDMA=y`, the IOMMU is in translation mode, and
resizable BAR is the node's one memory path. The KFD layer has no peer VRAM mapping yet. The
work is an allocation helper on the pattern of `Kfd::host_shared` (allocate VRAM on the
receiver, `IOC_MAP` it to the writer's `gpu_id` too); the map call failing is itself the test
of peer access. Before writing it: each card's `p2p_links` in the KFD topology and
`/sys/module/amdgpu/parameters/pcie_p2p` on the box.

**Measure first** (bench-sized, beside the serving node, as `bench-hc`):

1. Card-to-card ping-pong: a peer write plus a local poll, against today's host-memory handoff.
   Per-handoff latency decides this section's gain.
2. CPU-to-VRAM BAR writes: rate and flag latency from socket 0 and from socket 1
   (write-combining stores, `sfence`, then the flag).
3. Visibility on the card: data written over the BAR or by the peer, read after its flag with
   the L2 bypassed or invalidated. A correctness test, run many times.

**Then:** move the existing attention-phase handoffs (the KDA head split, the duplicated
streams, `hc_post`) to peer writes if (1) shows a real difference. This pays before any expert
moves.

**Status 2026-10-17: not measured.** The KFD layer maps host memory on every card but has no
VRAM peer mapping, so the ping-pong bench is new code before a number exists. Lower priority than
it was: the layer-ahead router (§4) retired the attention-half early router and its handoff, and
§3's design below needs no card-to-card handoff.

## 2. The expert phase controlled by a card

tcdent, 2026-10-17:

> it seems like we might want to run the control of the expert phase on the card, and have it
> gather information or wait for information to become available to it. And then so, a GPU,
> another socket, could write directly to the output that it would be able to know it's
> available, and then a CPU expert could also write to it

> That's actually a really cool architecture, and I think we should try it.

**Shape.** The controlling card computes the FFN input and the router's picks and publishes
them (input, picks, weights) to host memory on socket 0's nodes, where the CPU workers read
them. It starts its own resident experts at once. Every producer writes its expert outputs
straight into a slot in the controller's VRAM and then its own sequence word: CPU workers on
either socket through the BAR, the other card by peer write. One workgroup on the controller
watches the words locally and releases the combine. The host main thread is no longer a relay
inside the layer; the CPU becomes a pool of workers reacting to the card's publications. This
extends the card-driven token and the data-wait protocol of the hybrid experts, and it is the
same write-only, ordered-per-region protocol the fabric_ntb cards fit.

**Constraints.**

- A word per producer, never a counter: a CPU cannot do a reliable read-modify-write over a BAR
  mapping. A node's last finishing worker writes the node's word after a host-side count, so
  the card watches 4–8 words, not hundreds.
- The combine runs in a fixed order, not by arrival: a slot per pick, `w_k · y_k` summed in pick
  order — today's host order, so decode's bits can survive. No atomic adds into one buffer.
- Socket-1 nodes' writes cross the socket link: in order per core, later on average. They are the
  flags to time.
- Partials are written to VRAM and copied out in wide stores (`hc_handoff`'s pattern): GEMVs
  emitting straight into uncached host memory cost 200–300 µs a launch (measured).

**What it buys.** One protocol for any placement of an expert: either card, either socket.
The direct latency gain (the host's routing and summing, polling across PCIe) is probably about
a millisecond (estimate); the larger gain is what §3 can place behind it.

**Status: Unimplemented.** Waits on §1's peer mapping and benches.

**Status 2026-10-17: deferred, not rejected.** §3 turned out not to depend on it: the card
realization of a routed expert is bit-identical to the CPU's (same integer group sums, flush and
f32 order), so the host can keep the join — summing the picks in pick order from either tier — and
a refreshed residency stays reproducible. What §2 could remove is the host's wait for the cards'
expert outputs after its own run: from the next deploy the profile prints it (`join`, ms a step
token); the cards' jobs (~7 a layer at two tokens, ~20 µs each) finish well inside the host's
half, so the expectation is a few tenths of a millisecond a step token, below the device barriers
§2 would add. If `join` reads higher, §2 is back on.

## 3. Routed experts on the cards

The cards idle through the expert phase.

**Measurements: Resolved** (routing trace and replay, `forward-glm --routing-trace`,
`bringup/gates/routing_replay.py`, code 654 / chat 1998 positions):

- Free VRAM: ~12 GB a card on the current build → ~41 experts a layer across both cards
  (43 MoE layers with the draft's).
- Hit share of picks, code / chat:

  | experts a layer | LRU | LFU | Belady (ceiling) | static list from the text itself | code's list on chat |
  |---|---|---|---|---|---|
  | 32 | 0.39 / 0.43 | 0.51 / 0.37 | 0.65 / 0.66 | 0.54 / 0.38 | 0.13 |
  | 48 | 0.49 / 0.53 | 0.62 / 0.47 | 0.74 / 0.74 | 0.65 / 0.48 | 0.19 |
  | 64 | 0.57 / 0.60 | 0.70 / 0.55 | 0.81 / 0.80 | 0.74 / 0.56 | 0.25 |

- A static hotlist does not travel (0.19); a cache refreshed within the conversation serves
  about half the picks at ~41–48 experts a layer.
- Worth: about half the routed bytes from VRAM concurrently with the CPU's half, ~5 ms a step
  token. It also lowers the verify step's expert union, which is what holds draft depth 2 back
  (§6).

**Status of the cache itself: Unimplemented.** Waits on §2 for the join. Refreshes cost
~0.5 ms per 13.4 MB expert over PCIe and run off the critical path. Policy: LFU wins on code,
LRU on chat; replay an adaptive policy (ARC) on both traces before choosing — **Unimplemented.**

**Status 2026-10-17: measured and the kernels built; the step integration pending.**

- Free VRAM: ~12 GB a card at the 32k context (15.2 GB used across both cards plus the draft
  layer's 0.09) — ~41 experts a layer across both cards at 13.4 MB each.
- Routing-trace replay (`forward-glm --routing-trace`, `bringup/gates/routing_replay.py`; code
  654 positions / chat 1998): at 48 experts a layer LRU serves 0.49 / 0.53 of picks, LFU
  0.62 / 0.47, Belady 0.74 / 0.74; a static list from the code text applied to chat 0.19 — a
  cache refreshed within the conversation, never a hotlist (the 2026-09-30 finding holds).
- `xpt_gu` / `xpt_down` (`expert::gfx11`): a (token, pick) job over an expert in a card slot,
  **bit-identical to the CPU** (32768/32768 outputs over 8 experts, `bench-xpt`), 19 µs an expert
  at 4 jobs (713 GB/s), 58 µs alone (launch + sync).
- Expected once integrated: experts 21 → ~11 ms a step → ~24–25 ms an emitted token.
- Integration: resident picks' jobs to their card with the token's quantized activation (4.3 KB a
  token over the BAR), outputs copied to host memory with a word, `run_batch` on the rest
  meanwhile, the host's pick-order sum unchanged; a per-layer slot table a card filled from the
  node's frequency counts after a warm-up, refreshed later. Sized for the 32k context unless
  tcdent trades VRAM between experts and context.

## 4. Routing: the predictor and the boundary

**Layer-ahead prediction.** Layer l+1's router applied to layer l's FFN input.
**Measured: Resolved** — it names 4.88 / 5.06 of 8 picks (code / chat) against today's
predictor's 4.12 / 4.63, a layer earlier. **Status: Unimplemented.** Expected gain ~1 ms a step
token (the idle-DRAM window is the attention phase either way; the gain is accuracy and an
earlier start). Open: where layer l+1's router runs (on the host pool it costs ~4 ms a step
token at dispatch latency, too much; on the card it means every layer's router on both cards,
99 MB a card), and the prefetch must be issued after `run_batch(l)` returns (a run abandons
in-flight prefetches).

**The 8th/9th boundary.** 4–13 % of positions a layer have margins under 1e-3; flips there are
arithmetic noise, and an unforced flip gives a different path of equal quality (corpus NLL
2.3156 vs 2.3162). Two uses of that fact, mutually exclusive:

- Resolve exactly: compute the router with an error bound and recompute the boundary scores in
  compensated arithmetic only inside it (Shewchuk-style adaptive predicates). Removes
  arithmetic-caused flips between realizations; the same applies to the indexer's 2048th key.
- Exploit: inside the noise band, pick whichever candidate is resident (VRAM, or already in L3).

**Status: Unimplemented; tcdent's decision** (exploiting needs §3's cache to have anything to
prefer).

**Status 2026-10-17: layer-ahead built and deployed (0ea8ebc); the boundary question is still
open.** Scored first on the routing trace: layer l+1's router over layer l's FFN input names
4.88 / 5.06 of 8 picks (code / chat) against 4.12 / 4.63 for the attention-input predictor;
served 5.59 vs 5.2–5.5. Built as the FFN half's one router launch carrying both layers' rows
(`ROUTE_ROWS`), the prefetch issued when the layer's run returns (a run abandons prefetches in
flight), the attention-half early router and the `early` lap retired. Gain **0.5 ms a step
token** (29.0 → 28.1–29.0): the better prediction is partly eaten by the shorter window (34 % of
predicted slices pulled before the run, 60 % before). Resolve-vs-exploit for the 8th/9th
boundary: unchanged, tcdent's call.

## 5. Prefill kernels

- **Rows kernels, R rows a wave on the multi-token width: Resolved** (`de37280`): each staged
  activation read feeds R FMAs, bit-identical; a prompt token 7.8 → 6.2 ms.
- **A few-token pass width (2..=4 tokens, four wide): Resolved** (`bc45af0`): a verify step
  60 → 58 ms. One or two rows a wave there measured worse than four. FP8 few-token launches still
  cost 1.7–2× a token's — **Unimplemented.**
- **WMMA rows (plain or Ozaki): the rate measurement Resolved** (`bench-wmma`, register-only
  probes): iu8 WMMA 87 TOP/s, bf16 WMMA 85 TFLOP/s, scalar f32 FMA 17.5 TFLOP/s, against the rows
  kernels' ~1.4 TFLOPS — over 10× headroom for an Ozaki-form kernel (bf16 weights and f32
  activations sliced into int8, exact int32 accumulation, order-independent), whose bound will be
  staging operands into fragments. iu8 measured no faster than bf16 (RDNA3 lists int8 at twice
  fp16): rerun with eight chains a wave before trusting the ratio — **Unimplemented.** The kernel
  itself — **Unimplemented.** RDNA3 notes: wave32 WMMA needs operands duplicated across
  half-waves; FP8 decodes to f16 in registers (Marlin's bit-shift), the exponent bias folded
  into the scale.
- **The MLA attend:**
  - One launch over a step's tokens: **Resolved** (`4e220bf`): 5,632 launches a step → 1;
    a 512-step 10.4 → 9.4 ms a token at 3k; bit-identical.
  - Indexer scoring and selection on the card: **Resolved** (`6a9be6f`, `bf224d2`): decode at
    18k 48.9 → 41 ms; the attention's host time 1.8 → 0.0 ms; bit-identical.
  - Tiles of consecutive queries over the union of their selections: **Tried, not needed** —
    a wash to slightly worse (little overlap between consecutive queries' top-2048 at long
    context).
  - Register-tiled scores (four positions a thread): **Tried, not needed** — 2× slower.
  - Four heads a workgroup (`ATTEND_H` = 4): **Tried, not needed** — worse at decode (kda card
    14.5 → 20.3 at 18k: 32 workgroups on 84 CUs), 4.4 → 4.3 at 3k prefill. Kept at 1.
  - Positions split across workgroups with a fixed-order merge (fills the card at decode; not
    bit-identical to today): **Unimplemented**; a timeline at 18k first.
- **Experts, the multi-token kernel: Resolved** (`5b713ab`, `q4::gemm` in ik_llama.cpp's
  rows-in-lanes shape, bit-identical): experts 2.4 → 2.2 ms a prompt token. The remaining gap to
  the 1.0 floor — **Unimplemented (untraced)**: the per-node run ends' spread at the 512 step,
  the pool's per-run overheads (slotting, wake, tail), and whether the rates measured at start
  still hold under a prefill step's load.
- **hc_pre** (0.62 ms a prompt token at a 32-token step, ~0.2 at 512, a per-launch cost):
  **Unimplemented.**
- **Chunked KDA (FlashKDA v1, CHUNK 16): Unimplemented, low priority** — `kda_step` is 0.13 ms a
  prompt token. Its value returns with longer verify steps (§6).

**Status 2026-10-17.**

- WMMA rate measured (`bench-wmma`, `bringup/kernels`): iu8 WMMA **87 TOP/s**, bf16 WMMA 85
  TFLOP/s, scalar f32 FMA 17.5 TFLOP/s at four chains — against the rows kernels' ~1.4 TFLOPS,
  so an Ozaki-form pair has 10× or more rate headroom after the slice products; its bound is
  staging operands, not the matrix unit. iu8 measured no faster than bf16 (RDNA3 lists 2×): try
  eight chains before trusting the ratio. The pair itself is not built.
- The rows kernels at a few tokens: a third pass width `FEW = 4` with four-step rounds (for the
  verify step's 2–4 tokens) — the card's two-token step 21.7 → 17.6 ms of kernels; bf16 launches
  1.18× one token's, FP8 still 1.7–2×. **Rejected:** one or two rows a wave for that width (a
  wider grid): FP8 worse (ffn_down_shared 1277 / 1208 µs a step vs 994 at four rows), bf16 the
  same — the FP8 gap is not occupancy. Open.
- **The prefill experts' gap, traced from the 512-step profile (2026-10-17):** experts 2.2 ms a
  prompt token against the 1.0 DRAM floor. The nodes' run ends are 1.72–1.78 ms a token with a
  0.02 ms slotting, 0.00 waking and 0.00 tail — no scheduling gap, the four nodes end together. The
  streaming itself is the time: 316 MB a token in 1.75 ms is ~180 GB/s against the 323 GB/s
  ceiling, because at 512 tokens a weight read feeds 512 dots and the q4 `gemm` is compute-bound
  (17 GOP a token: 8.7 TOP a step in ~1.1 s ≈ 7.7 TOPS of the ~12.5 the 96 cores' `vpmaddubsw`
  pipes give). So the floor at a 512 step is compute, ~1.4 ms, and the lever is the `gemm`
  kernel's efficiency (the ik_llama r8 shape is already in; next would be a wider tile or int8
  activations in `vpmaddwd` directly), not placement. The attend dot, hc_pre, chunked KDA: unchanged.

## 6. Decode levers beyond placement

- **Speculative decoding with the draft layer (MTP), depth 1: Resolved** (`1855eb7`,
  `bc45af0`). The checkpoint's layer 45 is in the bundle and runs on the head's card after every
  step; a two-token verify step with rejection sampling (`common::sample::verify`, accept with
  min(1, p/q), else the residual, both under the sampler's truncation); the KDA state rewinds on
  rejection (a state a token for steps of 2..=SPEC_TOKENS, `kda_rewind`); the L3 prefetch covers
  the union of a verify step's predicted picks (expert phase 32 → 23 ms). Acceptance 0.92 on the
  gate text (draft top-1 0.86), 0.84–0.90 in real sessions; 36 → ~30.2 ms an emitted token, 34 at
  18k. Gated bit-identical (steps of 2, 3, 4, 512; the long SCORE).
- **Depth 2: Tried, not needed for now** (`aa5324d`). Second-draft acceptance 0.42; a
  three-token verify step 75–78 ms for 2.45 tokens = 30.6–31.8 ms an emitted token against 30.2
  at depth 1. The loop stays generic in `DRAFT_DEPTH` (= 1, the measurement in its comment).
  Depth 2 pays once a three-token step is under ~65 ms, which the expert union bounds: revisit
  after §3.
- **The target's indexer picks shared with the draft** (`index_share_for_mtp_iteration`;
  SGLang carries the target's top-k into the draft layer): **Unimplemented** — the draft uses
  its own indexer today.
- **Model-free drafts from the prompt's own n-grams (SuffixDecoding)**, for code and agentic
  output: **Unimplemented.** The DFlash2 drafter is non-commercial (CC BY-NC-ND) and out.
- **Persistent per-layer kernels, re-costed:** **Unimplemented.** Parked on 2026-10-16 at
  ~1.5 ms; vLLM's GLM-5.3-Flash build took a step 9.36 → 4.82 ms (1,699 → 220 kernels) on MI355X.
  Read their structure before re-costing on gfx1100's 6–8 µs device barriers.
- **The KDA projections' bytes (4.6 ms a token a card at bf16):** lossless DFloat11 (Huffman
  coding of the bf16 exponent) or FP8 (tcdent's dense-precision decision): **Unimplemented.**

**Status 2026-10-17: MTP built, deployed, the lever that paid.** One draft after every step
(the draft layer on the head's card), a two-token verify step with the KDA ring and a rejection
sampler; acceptance 0.86 greedy on the gate text, 0.92 served (0.84–0.90 on a real coding
session); 36 → ~30 ms an emitted token, 41 → ~32 at 18k. What made it pay: the verify step's
expert prefetch (the early router for n ≤ SPEC_TOKENS, then layer-ahead) and the FEW pass width.
**Rejected: depth 2.** The second draft's top-1 is the main model's at 0.50 (0.42 under the
sampler); served, a three-token step cost 75–78 ms for 2.45 tokens — 30.6–31.8 ms an emitted
token against 30.2, and a loss at 0.84 acceptance. The loop stays generic (`DRAFT_DEPTH = 1`);
depth 2 pays once a three-token step is cheaper than ~65 ms, which the expert union bounds (§3).
**Persistent per-layer kernels, re-costed from our own timeline (2026-10-17):** a two-token verify
step posts ~430 launches a card with ~2.3 ms of gaps between launches (card 1, `rockem.timeline`),
and the CP's per-launch dispatch inside the busy time is another ~5–8 µs each (~2–3 ms). A
persistent per-layer kernel can recover at most that ~5 ms a step, and pays gfx1100's 6–8 µs
device-wide barriers at every dependency it replaces (several a layer × 45 layers ≈ 2 ms), so the
net is ≤ ~3 ms a step, ~1.5 ms an emitted token, for a rewrite of the layer's launch structure.
Behind the card-phase kernels (§5) in gain per effort; not built. DFloat11 for the KDA projections
(lossless, ~11 bits a weight): ~30 % of the KDA bytes (4.6 ms a token a card at bf16) if its decode
rides free under the stream — a kernel to write and measure; not built. FP8 KDA projections: tcdent's.

## 7. Precision decisions, with their measurements

**Decided (tcdent, 2026-10-17): the attention state follows the reference.**

> Lets follow the reference and consider any deviations to be additional experiments.
>
> bf16 latents, f32 kda, bf16 indexer scored in 32

The reference is `modeling_glm5_next.py` (transformers v5) at the config's default dtype:

- **MLA latents in bf16:** the `kv_a_layernorm` output stored as bf16 (l.1189–1196).
- **KDA recurrent state in f32:** forced to f32 whatever the model dtype is (l.762).
- **Indexer in bf16, scored in f32:** the per-token keys stored in bf16 (l.846). The pools are
  formed in bf16, the softmax weights cast to the keys' dtype (l.856–866). The scores are
  `q.float() @ pool_keys.float()`, ReLU, then the f32 head weights (l.999–1003).

To match it exactly, the pool's arithmetic follows the reference too, not only the storage: the
weighted sum of a pool's four keys runs at bf16, as the reference does it.

Anything off this line is an experiment, gated like the approximate menu (§8): KL and top-1
against the oracle, the indexer's selection overlap at long positions, the evals. Candidates: fp8
latents (DeepSeek-V3.2's deployment, vLLM on SM100); fp8 pool keys after a Hadamard rotation (vLLM,
SGLang); a bf16 KDA state (vLLM PR #59147 reports it neutral on this model).

The gate's reference point: our oracle and goldens run the reference code in f32 throughout,
while the reference at its default dtype runs everything in bf16, weights and activations alike.
We follow it for the stored attention state, not for activations. So the new state is measured
against the f32 goldens as a numeric distance, and the goldens stay what they are.

- **bf16 MLA latents: decided (tcdent, 2026-10-17)** — the latent cache takes the checkpoint's
  default dtype (`config.json`: `"dtype": "bfloat16"`), and the KDA state stays f32 as the more
  sensitive state. Half the latent bytes (64 MB a layer at 32k; ~13 GB of VRAM at 1M). Our oracle
  and gates are f32, so the change is gated numerically, not bit-identically. Every
  production stack serves this model from bf16 or fp8 latents, fp8 measures at noise on
  GSM8K and LongBench-v2 for GLM-5.3-Flash, and GLM's MLA has no RoPE dimensions (the part of an
  MLA cache found most sensitive). The documented failures at long context are selection and
  accumulation bugs, not storage. Measurement: KL and top-1 against the f32 oracle, and the
  overlap of the indexer's top-2048 selection at 16k–100k positions — a number nobody has
  published. **Status: Unimplemented.**

  **Grounding (read 2026-10-17):**
  - *The reference runs the latents in bf16.* transformers v5 made `dtype="auto"` the
    `from_pretrained` default (v5.0.0 release notes; `modeling_utils.py` ~l.4106, 829), which
    takes `config.dtype` (bfloat16); `modeling_glm5_next.py` l.1189–1196 writes the normed latent
    into the cache, whose dtype is the incoming tensor's (`cache_utils.py` l.113). The indexer's
    per-token keys go in the same way (l.846). Upstream deployments go lower still: DeepSeek-V3.2's
    reference uses an fp8 latent cache ("we use fp8 kv cache in actual deployment",
    `inference/model.py` l.569–571), and vLLM switches GLM-5.3-Flash to fp8 KV on SM100
    (`Glm5NextModelConfig`, `config.py` l.118–134). bf16 is the conservative end.
  - *The reference keeps the KDA state in f32 regardless of the model dtype.*
    `modeling_glm5_next.py` l.762 (`last_recurrent_state.to(torch.float32)`); FLA's KDA kernels
    accumulate and return an f32 state (`fla/ops/kda/fused_recurrent.py` l.125–150, 279–281);
    vLLM's `kda_state_dtype` maps `auto` to float32 (`mamba_utils.py` l.133–148); SGLang defaults
    the SSM state to float32. "More sensitive" is the field's default, not a measured GLM loss:
    vLLM PR #59147 reports bf16 state on GLM-5.3-Flash with long-context NLL unchanged within
    noise, and Kimi-Linear GSM8K is at parity (#61245, 0.8961 f32 vs 0.8976 bf16). Keeping f32
    costs nothing in capacity — the state is per sequence (~136 MB), not per position, so it
    does not grow with the 1M context.
  - *The 1M arithmetic:* 12 MLA layers on the cards (the draft layer's included; 5 on card 0,
    7 on card 1 by `card_of`) × 1,048,576 positions × 512 × 2 bytes = 12.9 GB at bf16, 25.8 GB at
    f32 — f32 does not fit beside the rest of the cards' state.
- **The indexer's pool keys: decided above (bf16, scored in f32).** Today `[max_ctx/4][128]` f32 a layer (`glm_kfd.rs:567`),
  1.6 GB across 12 layers at 1M, and every decode token reads all of them (~2 ms of streaming at
  1M, plus ~13 GMAC of scoring). The reference pools in bf16 and scores in f32
  (`modeling_glm5_next.py` l.856–866); vLLM and SGLang store the pooled keys in fp8 with an f32
  scale after a Hadamard rotation (`glm5next/nvidia/sparse_indexer.py` l.61–101,
  `dsa_indexer_kpool.py` l.284–322); tinygrad in half. Every stack accumulates scores in f32. No
  published measurement ties key precision to selection flips at long context; the selection
  overlap above is the measurement that would.
- **FP8 KDA projections:** **Unimplemented; tcdent's decision** (§6).

**Status 2026-10-17 21:40: the first phase built and deployed; the numbers below.** The latent
and the indexer's key are stored at bf16 precision (rounded to nearest even at `finish_row` and on
the bundle's seeded rows; the buffers, kernels and mirrors unchanged, so the arithmetic on them is
still f32), the pool keys formed as the reference forms them (weights cast to bf16, the sum at
bf16), the KDA state f32. Measured against the f32 build on the long SCORE text (3632 ids,
`bringup/gates/score_diff.py`): mean |Δ logprob| 9.1e-3 nats, 99.4 % of positions moved, one
position by 5.8 nats (a near-tie), **NLL 0.0414 → 0.0446**. **How much that gate resolves
(2026-10-18):** an ulp-level change elsewhere (the router's weight sum taken in ranked instead of
id order, caught and reverted) moved the same text's NLL by 0.003 the other way; the text's
near-ties make ±0.003 the gate's noise, so the bf16 state's cost is below what this text can
show. A longer, higher-perplexity text (or the evals) is the gate for costs this small. Decode step cost unchanged; acceptance
on the short text 0.88–1.00 across runs (0.92 before). The second phase — bf16 buffers for the
latents and pools (half the bytes the attend and scorer read, the 1M arithmetic) — is layout work on
the host cache, the cards' mirrors, `mla_attend`, `ix_score` and the host's `heads`, gated by the same
comparison. The selection-overlap measurement at 18k is still to be made.

**Status 2026-10-17 evening (before the build):** What the decision changes in the
tree: the MLA cache row keeps its latent in bf16 (host `cache: [max_ctx][MLA_ROW]`, the cards'
`lat` mirror, `finish_row` writes it, `mla_attend` and the host's `heads` read it); the indexer's
per-token keys in bf16 with the pool's four-key sum at bf16 (`pool_key`, the cards' `pools`
mirror) and the scores in f32 as now (`ix_score`, `score_pools`); the KDA state untouched. The
gates for it are numeric: KL and top-1 of the gate text's logits against the f32 build (the
step gate's bit-identity is kept within the build: one token vs steps), and the indexer's
top-2048 selection overlap at 18k (the Cody transcript) between the two builds. What it buys
now: half the latent and pool bytes the attend and the scorer read (at 18k the attend is ~2.3 ms
a two-token step on the head's card, the scorer ~0.3), and the 1M context's VRAM arithmetic; it
does not move the 24 ms figure, which §3 owns. FP8 KDA projections: unchanged, tcdent's.

## 8. Approximate menu (each needs its eval cost)

**Status of every item: Unimplemented** (none tried, as intended).

- Expert deferral (KTransformers): the lowest-scored experts' outputs added two layers later, so
  the CPU's experts overlap the card's next layer. DeepSeek-V3: decode +33 %, LiveBench −0.5 %.
  With mHC, where the deferred add lands in the four streams needs working out.
- Halving k with a fixed reference mass: the top 4 normalized by the top 16's mass (−0.55 MMLU
  points on Qwen3.5-397B, 10 → 5).
- Intra-expert activation sparsity: needs neuron-bundled storage (up row and down column
  contiguous) to save DRAM bytes, and the down projection's quantization groups re-cut.
- Bit-plane expert storage (high plane first): precision per pick by gate weight, progressive
  prefetch, a bit-serial kernel's input, from one layout.

**Status 2026-10-17 22:15: one measured — halving k.** `ROUTED_K` (lib.rs): the top 4 of the
router's 8 run, their weights normalized by all 8 scores (the reference mass); the dropped slots are
the same sentinel the resident experts use, so nothing else changes. Measured on the resident-experts
build: the expert phase 6.0 → 3.9 ms a step token warmed, **a verify step ~40 ms, ~21 ms an emitted
token** (24–25 at k = 8); the long SCORE text's NLL **0.0426** against 0.0446 at k = 8 (the f32 build
0.0414) — on that low-perplexity text the cost is not visible; acceptance on the short decode text
0.78–0.85 (0.88–0.92). **Rejected (2026-10-18):** GSM8K under our harness on the same 408
problems, k = 4 **83.3 %** against k = 8 **91.2 %** — 64 problems came out differently, 48 of them
k = 8's (z ≈ 4), and k = 4 ran out of the 320-token budget on 22 against 8 (it reasons longer
before an answer). Eight points for 3 ms an emitted token; `TOP_K` stays the model's 8. The
runs' absolute numbers are ours only (the 320-token cap, our prompt format), not a leaderboard's;
the pair needed 400 paired problems, not 1319 — the two full runs were stopped. The harness's
cap is wrong for a thinking model and should rise before any absolute GSM8K is read. Expert deferral, intra-expert sparsity, bit-plane storage: not evaluated — each
is a storage or scheduling change before any run.

## Order

1. ~~The cheap measurements~~ done 2026-10-17 but §1's benches: free VRAM, the routing-trace
   replay, the iu8 WMMA rate.
2. ~~MTP with KDA commit-after-verify (§6)~~ deployed 2026-10-17.
3. ~~Layer-ahead routing (§4)~~ deployed 2026-10-17.
4. **Experts on the cards (§3), host join** — the 24 ms lever; kernels built and gated, the step
   integration next.
5. Then §2 only if the host join's wait shows; §1's benches if a handoff is on the path.
6. Prefill: WMMA rows (Ozaki or plain), the attend dot, the traced expert-phase gap (§5).
7. Depth-2 drafting again once the step is cheap (§6); the precision measurements (§7).

## Decisions open for tcdent

- §4: resolve routing ties exactly, or exploit them for placement.
- §6: the KDA projections' precision (DFloat11 or FP8). §7's attention state is decided: it
  follows the reference (bf16 latents, f32 KDA state, bf16 indexer scored in f32), and any
  deviation is an experiment.
- §8: which approximate methods are worth an eval run.

## Considered and set aside

Winograd/Goertzel for the 4-tap conv (negligible work); Newton or Knight–Ruiz balancing instead
of Sinkhorn (trained on 20 truncated iterations); Strassen/AlphaTensor (numerics and bookkeeping
exceed the savings at these shapes); product quantization for the indexer (breaks the oracle);
LayerSkip/SWIFT (need training); Sherman–Morrison rollback of KDA state (ill-conditioned as
β‖k‖² → 1); LUT GEMM at 4 bits for decode (DRAM-bound already).
