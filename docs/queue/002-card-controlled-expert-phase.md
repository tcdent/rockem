# 002 — The expert phase controlled by a card

**Status:** Built (2026-10-20): the FFN card leads the routed experts — a verify step at 18k 45.9 → 43.5 ms, long SCORE bit-identical
**Area:** decode
**Gain:** measured — a verify step at 18k 45.9 → 43.5 ms (the host's sum and handoff 1.37 → 0.05 ms, experts 15.7 → 14.65)

## Proposal
tcdent, 2026-10-17:

> it seems like we might want to run the control of the expert phase on the card, and have it
> gather information or wait for information to become available to it. And then so, a GPU,
> another socket, could write directly to the output that it would be able to know it's
> available, and then a CPU expert could also write to it

> That's actually a really cool architecture, and I think we should try it.

The controlling card publishes the FFN input, picks and weights; every producer (CPU workers
on either socket through the BAR, the other card by peer write) writes its outputs into a slot
in the controller's VRAM and then its own sequence word; one workgroup watches the words and
releases a fixed-order combine. Constraints: a word per producer (no CPU atomics over a BAR),
pick-order summation (no atomic adds), partials staged in VRAM and copied out in wide stores.

## Source
Our design discussion, 2026-10-17; it extends the hybrid experts' data-wait protocol and is the
protocol the fabric_ntb cards fit.

## History
- 2026-10-17: deferred — the card's expert realization is bit-identical to the CPU's, so the host
  keeps the join and 003 did not need this.
- 2026-10-18: rejected on measurement — the `join` lap (the host's wait for the cards' experts)
  reads 0.01–0.02 ms a step token; the device barriers this would add cost more.

- 2026-10-20: reopened as the rest of 064 and built on 001's tooling. The 2026-10-18 measurement priced only
  the join; the host's weighted sum (1.04 ms a verify step) and the handoff (0.28) were the cost.

> "we're thinking about the system theoretically as the GPU zero kind of like leading the run and that the CPUs
> are treated as an auxiliary which is providing data into it and so that's why we have this ability to drop
> output directly into VRAM and then flag the work as complete. Which is the same technique that we use for GPU
> one and could use with additional GPUs as well." — tcdent, 2026-10-20

## Built (2026-10-20, run 34f5563b against 00f516ae)
A MoE layer's FFN card is the leader of its routed experts. Every producer writes its (token, pick) rows into the
leader's VRAM (`Peer::rows`, `[STEP · TOP_K][H]` f32, fine-grained, mapped for the host and the peer): the host's
workers through the BAR (each down item ends with an `sfence`: write-combining stores are per core), the leader's
resident jobs locally, the other card's by peer write. The host waits for its pool and the residents' fences, then
writes one word into the leader's copy (`ffn_done`) — no sum, no handoff. The leader's `hc_post` waits for it,
sums each token's rows in pick order with the act phase's VRAM copy of the weights (the host's arithmetic: a
multiply and an add a pick from zero), folds `fy + sum` into its streams, writes the sums into the peer's
receiving buffer (`Peer::yr`) and, once every workgroup's are out, the peer's copy of the experts' word; the peer's
`hc_post` folds them in. `Shared::xout` (host memory) and `Kfd::host_owned` are gone.

| 18k | before | the leader sums |
|---|---|---|
| a verify step (`laps.sh`) | 45.94 ms | 43.51 |
| the host's sum + handoff (`sum` → `done`) | 1.37 | 0.05 |
| experts | 15.70 | 14.65 |
| the cards' wait (`kda_card`) | 19.89 | 19.98 |
| a verify step by the request's wall | 48.0–49.0 | 46.4–47.0 |
| an emitted token | 26.6–27.0 | 25.7–25.9 |
| prefill, the warm 18k prompt | 2.83 ms a prompt token | 2.77 |

Long SCORE bit-identical. The cold prompt's prefill (3.06) ran through a PCIe error burst (14 APEI records, 26 AER
lines) and is not a number. Why the experts lap fell ~1 ms is not traced (the sum's DRAM reads no longer follow it;
the rows now leave the cores as write-combining stores instead of through the caches).

## Next
- The experts lap's fall explained (`experts.sh` / `pool.sh` on both runs).
- Prefill's share measured on a clean cold prompt: 128 MB of rows a 1024-token layer-step cross the BAR now.
- More producers on the same protocol: another card, another socket's workers (fabric_ntb).
