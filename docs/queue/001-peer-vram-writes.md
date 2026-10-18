# 001 — Peer VRAM writes for the handoffs

**Status:** Deferred (measured 2026-10-18) — the helper and the benches exist; the handoffs' ceiling is ≤ 0.1 ms a step token
**Area:** decode, prefill
**Gain:** measured — a flag round trip 2.67 µs through peer VRAM against 3.47 through host memory (0.4 µs a one-way message); a single wave's 64 KB payload 2.7× faster

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

## Next
1. ~~On the box: each card's `p2p_links`~~ — read 2026-10-18 over the console shell: gpu 50288 (node 4, io link to CPU node 2) ↔ gpu 60504 (node 5, io link to CPU node 3), p2p weight 52 flags 3; the link exists and crosses the socket's fabric between quadrants, not a shared switch. Note the c0 root port's card throws corrected PCIe replay errors under load (memory `metatron-pcie-link-errors`): the ping-pong bench should count them.
2. The helper, then three benches beside the serving node: card-to-card ping-pong against the
   host-memory handoff; CPU-to-VRAM BAR write rate and flag latency from each socket;
   visibility (data read after its flag with L2 bypassed or invalidated), many times.
3. ~~Move the attention-phase handoffs if the ping-pong shows a real difference.~~ Measured, below.

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
