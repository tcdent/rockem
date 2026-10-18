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

## 7. Precision decisions, with their measurements

- **bf16 MLA latents.** Half the latent bytes (64 MB a layer at 32k). Not "native" to the
  checkpoint — the latents are activations, and our oracle and gates are f32 — but every
  production stack serves this model from bf16 or fp8 latents, fp8 measures at noise on
  GSM8K and LongBench-v2 for GLM-5.3-Flash, and GLM's MLA has no RoPE dimensions (the part of an
  MLA cache found most sensitive). The documented failures at long context are selection and
  accumulation bugs, not storage. Measurement: KL and top-1 against the f32 oracle, and the
  overlap of the indexer's top-2048 selection at 16k–100k positions — a number nobody has
  published. **Status: Unimplemented.**
- **FP8 KDA projections:** **Unimplemented; tcdent's decision** (§6).

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

## Order (what remains)

1. §1: check `p2p_links` and `pcie_p2p` on the box, the peer VRAM helper, the three benches.
2. §1 then §2: peer-write handoffs if the ping-pong says so, then the card-controlled expert
   phase.
3. §3: the expert cache on the cards (ARC replayed first), then depth-2 drafting revisited.
4. §5: the iu8 eight-chain rerun and the WMMA/Ozaki rows kernel; the attend's split positions
   (after a timeline at 18k); the traced expert-phase gap; the FP8 few-token cost.
5. §6: the shared indexer picks for the draft; persistent kernels re-costed.
6. The timeline on/off offset, whenever a measurement boot is convenient.

## Decisions open for tcdent

- §4: resolve routing ties exactly, or exploit them for placement.
- §6–§7: the KDA projections' precision (DFloat11 or FP8); bf16 latents (after the measurement).
- §8: which approximate methods are worth an eval run.

## Considered and set aside

Winograd/Goertzel for the 4-tap conv (negligible work); Newton or Knight–Ruiz balancing instead
of Sinkhorn (trained on 20 truncated iterations); Strassen/AlphaTensor (numerics and bookkeeping
exceed the savings at these shapes); product quantization for the indexer (breaks the oracle);
LayerSkip/SWIFT (need training); Sherman–Morrison rollback of KDA state (ill-conditioned as
β‖k‖² → 1); LUT GEMM at 4 bits for decode (DRAM-bound already).
