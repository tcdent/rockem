# 001 — Peer VRAM writes for the handoffs

**Status:** Both halves built (2026-10-19): the card-to-card handoffs in the receiving card's VRAM, and the FFN input's norm, quantize and router pick on the card, folded into the FFN card's `hc_pre` — prefill 4.42 → 4.02 ms a prompt token, a verify step at 18k 46.47 → 45.68 ms; the act phase's ~20 µs a layer still on the FFN card's critical path
**Area:** decode, prefill
**Gain:** measured piecemeal — a flag round trip 2.67 µs through peer VRAM against 3.47 through host memory; and on the host side of the same design: the join's 0.86 ms a step (038), the prefill expert handoff's 0.2 ms a prompt token, the route and sum laps' 3.5 ms a step (064), every one a CPU touch of a buffer that exists only because the cards hand off through the host

## Proposal
Every card-to-card handoff goes through uncached host memory: two PCIe trips (one card writes,
the other reads). A peer write into the receiver's VRAM is one trip, and the receiver polls its
own memory. Applies to the attention phase's handoffs (the KDA head split, the duplicated
streams, `hc_post`), which sit inside the `kda card` lap.

## Source
Our own design discussion, 2026-10-17. The KFD topology (read 2026-10-18) puts the cards on NUMA nodes 2 and 3 (one a quadrant of socket 1, root complexes a0 and c0), with a peer link between them: `p2p_links` type 2, weight 52, flags 3 (enabled, no 32-bit atomics, peer DMA allowed).

## History
- 2026-10-17: no configuration change needed — `CONFIG_HSA_AMD_P2P=y`, `CONFIG_PCI_P2PDMA=y`,
  the IOMMU in translation mode, resizable BAR. The work is an allocation helper on the
  pattern of `Kfd::host_shared` (allocate VRAM on the receiver, `IOC_MAP` it to the writer's
  `gpu_id`); the map call failing is the test of peer access.
- 2026-10-17: deprioritized — layer-ahead routing (005) retired one handoff, and the resident
  experts (003) need no card-to-card handoff.
- 2026-10-17: evidence the link matters — the residency loader's BAR writes, flat out,
  doubled every card lap (003).

- 2026-10-18: vLLM PR #61868 stores a GEMM epilogue straight into the next boundary kernel's
  mailbox (−41 µs, a copy kernel removed). With peer mapping our producing kernels could do the
  same into the receiver's VRAM, instead of writing VRAM and copying out with `hc_handoff`.

## Next
1. ~~On the box: each card's `p2p_links`~~ — read 2026-10-18 over the console shell: gpu 50288 (node 4, io link to CPU node 2) ↔ gpu 60504 (node 5, io link to CPU node 3), p2p weight 52 flags 3; the link exists and crosses the socket's fabric between quadrants, not a shared switch. Note the c0 root port's card throws corrected PCIe replay errors under load (memory `metatron-pcie-link-errors`): the ping-pong bench should count them.
2. The helper, then three benches beside the serving node: card-to-card ping-pong against the
   host-memory handoff; CPU-to-VRAM BAR write rate and flag latency from each socket;
   visibility (data read after its flag with L2 bypassed or invalidated), many times.
3. ~~Move the attention-phase handoffs if the ping-pong shows a real difference.~~ Measured, below.

## Reopened (2026-10-19, tcdent)
> I mentioned something like this way back with zero zero one, and it got pushed off because we didn't think
> there was enough game. But I think that this is an architectural decision that is just fundamental and
> informs the way that the system is built.

What the day showed, each measured in its own entry: KFD's host memory is write-combining for the CPU, so the
host's bulk reads and writes of any handoff buffer are slow (the old join's 160 KB copy 0.86 ms a step; the
expert handoff's 16 MB copy 4.8 ms a layer at prefill; 038, 064). The fix that worked — the host's own pages
registered for the cards (`Kfd::host_owned`), the cards writing into them — is one half of the rule; the other
half is the cards handing to each other through the receiver's VRAM (this entry's measurement), never through
the host. And the host's laps between card phases (064: route.pick, sum.y, dispatch, the handoff — 3.5 ms a
verify step) exist because the data comes down to the host and goes back up twice a layer. The rule that
falls out:

- **host-touched memory is the host's**: anonymous pages, cached, registered for the cards (`host_owned`);
  nothing the CPU reads or writes in bulk lives in a GTT allocation;
- **card-to-card memory is the receiver's VRAM**, peer-mapped (`vram_shared`): one trip, the receiver polls its own;
- **the handoff words stay coherent and tiny**;
- **work on data the cards hold happens on the cards**: the router's top-k, the quantize, the weighted sum
  over the expert rows (064's pieces, 002's shape).

## Built (2026-10-19 09:40, run a9f5139c)
Each card owns a `Peer`: its receiving buffers for the KDA shares, the FFN rows and the MLA projection by
layer parity, and its own copy of the handoff words, all fine-grained VRAM mapped for the peer
(`vram_shared`). A producer's `hc_post` copies its share to the peer's buffer (a step of many tokens through
the copy kernel's grid first, a few tokens inside `hc_post`) and writes its word into its own copy and the
peer's; every card waits on its own words and reads its own share where it made it and the peer's where it
landed. The host memory keeps what the host touches: the FFN input and router scores it reads, the routed
experts' outputs it writes, the one word it waits on and the one it sets (into every card's copy). The long
SCORE bit-identical, zero PCIe records.

| at 18k, a 1024 step | host-memory handoffs | peer VRAM |
|---|---|---|
| prefill, ms a prompt token | 4.42–4.44 | 4.19–4.21 |
| `kda_card` lap, a 1024 step | 2,136 ms | 1,958 |
| `mla_proj` lap | 321 | 288 |
| the cold 18k prompt | 83.7 s | 79.4 |
| verify step at 18k | 46.5 ms | 46.5 |

Five times the 0.04 ms a prompt token the entry estimated at 512: at 1024 a handoff is 16 MB, and the old
path crossed the link three times a handoff (the producer's write to host memory, both cards' reads of it,
each card reading its own share back over the link too). Decode's handoffs are a few tens of KB and
latency-bound, as estimated: unchanged.

## Built, the host-side half (2026-10-19 10:00–12:00, runs prow21–26)
`ffn_act` on the FFN card, launched after `hc_pre`: for each token the post-norm of the FFN input, its
quantization (`QAct::quantize`'s arithmetic, exact), and the router's picks and weights (`route_scores`'s:
the sigmoid, the top-k with the bias as a wave butterfly keeping the lower expert between equal keys, the
mass in id order) — into the card's own activation buffers for its resident jobs and into host memory for the
host's experts (`Shared::xact`, read streamed after the kernel's word). The host's `route.pick` and
`route.quant` laps are gone and the dispatch uploads to the other card only. Five deploys to get there:
the first missed experts 256–287 (288 routed experts, 256-thread loops — found by the host-versus-card check
in a lab boot: the quantized bytes matched, the picks did not); the second recovered the layer-ahead
prefetch's predictor (the router logits' second half), without which a verify step's items ran at DRAM speed
(12 → 7 GMAC/s a core); the norm's sum became `rmsnorm8` on both tiers (eight strided chains added in order:
one chain of 4096 was the kernel's long pole) — a numeric change, gated: long SCORE NLL 0.04362 → 0.03931,
mean |Δ logprob| 8.3e-3, the same class as the bf16 attention state's.

| at 18k | peer VRAM only | + `ffn_act` |
|---|---|---|
| prefill, ms a prompt token | 4.19–4.21 | 4.02–4.10 |
| route lap, a 1024 step | 94 ms | 7 ms |
| the cold 18k prompt | 79.4 s | 77.2 s |
| verify step | 46.5 ms | 50.2–50.7 ms |
| of which the host's route lap | 1.5 ms | 0.3 |
| of which the cards' wait (`kda_card`) | 18.9 ms | 21.3 |

Decode lost 0.8 ms a step: the kernel runs on the FFN card's queue between `hc_pre` and the shared expert's
rows, and at a verify step that card's FFN phase is the longer side, so its ~55 µs a layer lands on the step
(the host's 26 µs a layer of routing and quantizing moved off the host, onto the card's path, and grew). Its
own work is ~15 µs; the rest is the launch and the word. Folding it into `hc_pre`'s last phase (the kernel
already ends with an arrival and the router's word) removes the launch and the barrier; a side queue (057)
would overlap it with the shared expert's rows.

## The act phase folded into `hc_pre` (2026-10-19 13:10, run b1cd…)
The decode regression measured to the piece first (a track-launches boot and in-kernel phase stamps, `kstats.sh`):
the FFN card's router `hc_pre` was always followed by 22 µs of idle before the next kernel (31 before `ffn_gu`
before 001, invisible then because the host woke on the router's own word); `ffn_act` was 54 µs of serial chains
in two workgroups (loads 5, the norm's eight chains 20, quantize 13, the wave-0 top-k and its tail 19). A theory
that the idle was the link draining the router's host-memory writes was tested against a 1024-token prefill step
(16 MB of such writes, 4 µs of idle) and is wrong; the idle's cause is not known. The fold removes the kernel
boundary whatever it is: after the scores' arrival every workgroup with a token waits on a VRAM ready word and runs
the token's act phase (`act_token`: the norm as a 256-thread tree, `common::rmsnorm::sumsq_tree` on the host; the
four groups a wave with their loads first; the top-k butterfly with the id-order positions from the lanes), the last
workgroup out writing the host's word. Long SCORE NLL 0.0419 (the tree's order; the act build 0.0393, aw3 0.0436).

| a verify step at 18k | peer VRAM only | separate `ffn_act` | folded |
|---|---|---|---|
| verify step | 46.47 ms | 47.42 | 46.32 |
| cards' wait (`kda_card`) | 18.70 | 21.43 | 20.19 |
| host route lap | 1.62 | 0.35 | 0.35 |
| prefill, ms a prompt token | 4.19–4.21 | 4.02–4.10 | 4.00–4.07 |

The fold took back 1.1 ms a step (the 22 µs idle and the 4 µs launch, 42 layers); the act phase still holds the
host's wake ~35 µs a layer past the old router word against the 10–15 µs expected — being stamped (the act phase in
`hc_pre`'s four stats slots).

## The act phase to the piece (2026-10-19 13:40–15:30, runs a8bdb6a6, 6615b39f, b6c5a3ab)
Stamped (wave 0 of the first workgroup, the stats object on a track-launches boot, `kstats.sh`): the scores' wait
4.1 µs, the norm 3.6, the quantize 8.6, and the picks unstamped but the rest of a launch that had grown from ~66 to
118 µs. Two causes, both serial chains the compiler built:

- **The wave reductions.** `core::arch::amdgpu::wave_reduce_umax` / `_add` compile to a loop over the 32 lanes, one
  at a time. Replaced everywhere by xor butterflies over `ds_bpermute` (`common::gfx11::wave_max_u32`,
  `wave_sum_i32`), exact (max and integer sum have no order) — fold2.
- **The picks.** A rank a thread an expert (the count of experts above its key) loaded each key and waited on it,
  288 times, twice over for threads 0–31 (288 experts on 256 threads); batching the loads left a float compare, an
  equality compare, a tie test and an add a key with mask dependencies between them (18.9 µs). The key and the
  expert now form one order-preserving u64 (the key's bits mapped to unsigned order above, the expert's complement
  below: the host's greater key first, the lower expert between equals; −0 would sort below +0, and a sigmoid plus
  a bias cannot be −0), one 64-bit compare and a carry-add a key, both of a thread's experts in one pass: 5.0 µs.

| a verify step at 18k (`laps.sh`, the 96-token hit) | peer VRAM only | separate `ffn_act` | fold1 | fold2 | fold4 |
|---|---|---|---|---|---|
| run | a9f5139c | 550345f6 | b1cd9bfb | a8bdb6a6 | b6c5a3ab |
| verify step, ms | 46.47 | 47.42 | 46.32 | 46.71 | **45.68** |
| cards' wait (`kda_card`) | 18.70 | 21.43 | 20.19 | 20.56 | 19.55 |
| host route lap | 1.62 | 0.35 | 0.35 | 0.36 | 0.36 |
| the router `hc_pre`, µs (stamped lab) | | | | 118 | 85.5 |

Long SCORE bit-identical to fold1 for fold2 and fold4; prefill 4.02–4.05 ms a prompt token; the cold 18k prompt
76.7 s. (The separate-kernel column was read from runs d6b0296d/550345f6, whose route lap of 0.35 says they already
carried the kernel though their commit is 2898ff0.)

fold4's act phase on wave 0 is ~20 µs a layer (wait 2.6, norm 3.7, quantize 8.6, picks 5.0), which is what the
cards' wait still holds over the peer-only build (0.85 ms a step, 42 layers) while the route lap gave back 1.26.
The quantize ran its four groups one after another, five dependent permutes for the max and five for the sum a
group: one `common::q4::gfx11::quantize_groups::<N>` now runs N groups' butterflies in lockstep, used at every
quantize site (both models' expert kernels, `quantize_to_lds`, the act phase with N = 4) — fold5: no gain (run
7746cdae: a verify step 45.76 ms, `kda_card` 19.74, within the noise of fold4; the quantize stamp 8.6 → 8.2 µs).
The butterflies were not the quantize's cost: its loads issue together and its permutes interleave, so what is left
is the latency of reading the token's row back from L2 after other workgroups wrote it (a stats object adds a
coherent read a stamp, so its phases read high). Kept for the one quantize instead of six copies, bit-identical.

## Next
1. The act phase's ~20 µs a layer (fold4's stamps): the token's row is read twice (the norm's strided order, then
   the groups'); one load in the groups' layout for both needs the norm's tree order redefined on both tiers (a
   numeric change). The scores' wait (2.6–3.6) is the grid's arrival.
1. ~~The handoff buffers inventoried~~ — done above; the card-to-card ones moved.
2. The host-side pieces of 064 onto the cards (route.pick, sum.y, the quantize, the dispatch's uploads).
3. The handoff buffers inventoried by who writes and who reads them (`Shared::{ys, fy, my, yexp, xh, xout,
   scores_ffn, lg}`), each assigned by the rule; the card-to-card ones (`ys`, `fy`, `my`) to peer VRAM with
   the words in it — the refactor this entry always named.
2. The host-side pieces of 064 moved onto the cards in the same pass, since they are the same decision.
3. The coherence rule for `host_owned` data a kernel publishes with its own word (not a packet completion):
   the kernel's release must write the card's L2 back; verify on gfx11 before relying on it (038's rows are
   read after the packet's completion signal, which does).

## Measured (2026-10-18 09:40, `bench-p2p` beside the serving node)
- The helper: `Kfd::vram_shared(size, others)` — fine-grained VRAM (`MEM_COHERENT`, gfx11 maps it
  MTYPE UC so the owner's polls bypass its L2), host-visible, `IOC_MAP`ped into the peers. The peer
  map succeeds on both cards; 10,000 messages of 64 KB checked after their word, **0 mismatches**.
- `bringup/kernels` `pingpong` (one wave; payload then the word, Release; the peer polls its own
  word, Acquire, checks the payload) and `bringup/bench/src/p2p.rs`:

  | measure | peer VRAM | host memory (today's handoffs) |
  |---|---|---|
  | card↔card round trip, no payload | **2.67 µs** | 3.47 µs |
  | card↔card round trip, 64 KB a message, one wave | 336 µs | 917 µs |
  | host→card→host word round trip, from the cards' socket (nodes 2/3) | 2.8 µs | — |
  | the same from socket 0 (nodes 0/1) | 3.0–3.6 µs | — |
  | BAR write, 13 MB, from nodes 2/3 | 21.6 GB/s | — |
  | BAR write, 13 MB, from nodes 0/1 | 10–17 GB/s | — |

- What it would buy the handoffs: a decode handoff is a token's streams or partials (`STEP·H·4`
  buffers, a few tens of KB live at a verify step), latency-bound: 0.4 µs a one-way message, a few
  per layer, **≤ 0.1 ms a step token** of ~26. At prefill (512 tokens) the handoffs are bandwidth-bound
  (~12 MB a layer); one trip instead of two saves ~0.04 ms a prompt token of 6.7. Both under the
  gate noise, so the refactor (regions by table, the receiver's VRAM as the handoff target, the words
  in it) stays parked; the helper and the bench stay for 002/039 if the cards come to exchange more.
- Fell out, and measured the other way: the residency loader runs on core 24 (node 1, socket 0),
  writing slots at ~16 GB/s where nodes 2/3 write at 21.6. Moved to core 48 (node 2) and deployed
  (09:45): the expert lap 9.2–9.8 → 9.9–10.5 ms a step token and the route lap 1.3 → 2.0. The pool
  takes every physical core, so the loader shares a worker's core wherever it sits, and on socket 1
  (the slower memory, [[epyc-7r13-topology]]) its 13 MB reads slow the node that bounds the expert
  phase. Reverted to 24 (09:55). The BAR rate is not the loader's bottleneck; one load a step is.
