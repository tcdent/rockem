# metatron's CPUs and memory, as measured

How metatron's two EPYC 7R13s and their DRAM behave, collected from the measurements of 2026-09-20 to
2026-10-20, so CPU kernel and placement work can start from what is already known. Every number has a date
and a source. Numbers from before the NPS2 flip (2026-10-15 19:30) are marked NPS1. Inference is marked as
inference.

> "we've done these kind of ad hoc experiments on the CPU to figure out and learn how it operates, and I was
> hoping that more of that was documented." — tcdent, 2026-10-20

The routed experts run on these CPUs, and they are the system's bottleneck. Decode's expert runs are bound by
DRAM bandwidth. Prefill's are bound by instruction throughput, and since 2026-10-20 also by the sockets' power
cap.

> "I don't want us to keep thinking about this idea that we're at the edge of RAM, so there's nothing else that
> we could do. Because we've already seen concepts like prefill get us up to L3 cache rates ... So I'm not
> comfortable with us reaching a point where we're saying that we don't have anything else to shave time with."
> — tcdent, 2026-10-20

## 1. The machine

| | |
|---|---|
| CPUs | 2 × AMD EPYC 7R13 (Zen 3 "Milan", an AWS/OEM SKU). Stepping 1, microcode 0xa0011d5 on both sockets |
| Cores | 48 a socket, 96 physical cores, 192 threads (SMT on) |
| CCDs | 6 a socket × 8 cores. Each CCD has 32 MB of L3 (16-way), 192 MB a socket. Each core has 512 KB of L2 |
| Clock | 2.65 GHz base. Reseller listings give ≈3.6 GHz all-core. Measured: 3.44 GHz on all cores, falling to ~3.04 under the power cap (§4) |
| Power | the HSMP `power1_cap` = `cap_max` = **280 W a socket** (2026-10-20). Reseller listings say ~225 W TDP |
| DRAM | 16 × 32 GB DDR4-3200 dual-rank, configured at 3200 (DMI type 17): one DIMM a channel, all 8 channels a socket populated. 512 GB in all; 204.8 GB/s a socket in theory |
| Fabric | fclk 1600 / mclk 1600 MHz on both sockets (HSMP, 2026-10-16) |
| NPS | **NPS2** (set by tcdent in UEFI on 2026-10-15). A 6-CCD part cannot run NPS4 (Cisco's 7003 tuning paper) |
| NUMA nodes | 4 nodes of 24 cores, 3 CCDs and 4 channels each. Node 0 = cpus 0–23, node 1 = 24–47 (socket 0), node 2 = 48–71, node 3 = 72–95 (socket 1). SMT siblings are 96–191 |
| Node distances | 10 to itself, 12 to the other node on its socket, 32 to the other socket |
| Hugepages | `hugepagesz=2M hugepages=106000` on the cmdline. Expert memory lives on 2 MiB pages |

**Why this SKU.** On 2026-09-20 tcdent chose the 48-core 7R13 over 64-core parts because the work is
bandwidth-bound. The extra cores would only have added power. How many CCDs the 7R13 has was first inferred
from its cache total (216 MB = 192 L3 + 24 L2). Two later measurements confirm 6 × 8. Under NPS2 each node has 3
CCDs. resctrl reports 12 L3 domains: 0–5 on socket 0 and 8–13 on socket 1.

**Package layout (AMD pub. 57075).** There are 8 CCD sites around the IO die. Each quadrant pairs two sites with
two memory controllers and one IO hub, and the pairing is fixed. NPS only decides how this is interleaved and
exposed. Which 6 sites the 7R13 populates is not documented. CCD-as-NUMA would give 6 nodes with uneven channel
shares. The SKU whose layout gives one CCD a channel is the 32-core 7543/75F3 (8 CCDs × 4 cores).

**Devices.** Both 7900 XTs and the 10G NIC (82599) are on **socket 1**: one card a quadrant, with KFD io links to
CPU nodes 2 and 3 (root complexes `a0` and `c0`), and the NIC on node 2. Linux socket 1 is DMI "P1" and the
board's silkscreen "CPU2", the socket beside the rear I/O panel.

**Socket asymmetry.** Socket 1 runs hotter: Tctl 72 vs 59 °C on 2026-10-16, and every CCD was 12–16 °C above its
counterpart. Socket 1 also streams less (§2). After CPU2's cooler was turned on 2026-10-16, the idle gap at probe
time was 7–10 °C.

### Firmware knobs

The knob table lives in `os/firmware/knobs.toml`. `knobs.efi` runs before every boot and reads the knobs back.
It cannot write them: AMI's setup-store protection returns SECURITY_VIOLATION, and setting an admin password did
not change that (2026-10-16). Knobs are set by hand in the setup UI.

| knob | state | measured effect |
|---|---|---|
| NPS (AmdSetup 0xD1) | 2 | the one knob that paid: all-local reads went 306–314 → 326 GB/s. Once the expert layout matched it, the expert phase realized the gain (§2) |
| APBDIS (0x1D0) = 1, Fixed SoC P-state (0x1D2) = P0 | set 2026-10-16 | socket 0's nodes went 84 → 91–94 GB/s. Socket 1 did not move |
| DF C-states (0x1D1) | Disabled 2026-10-16 | nothing (321–323 total vs 324 before) |
| ASPM | Disabled in shipped firmware | `pcie_aspm=off` on the cmdline was a null result for that reason |
| HSMP Support (AmdSetup 0x1D4) | Enabled | readouts only: `node hsmp` shows fclk/mclk/power, and `node hsmp df 1 0` pins socket 1's DF P-state at runtime |
| Memory interleave, interleave size, DRAM power-down, Determinism, Global C-state | Auto | not measured |

The hardware agent's research (2026-10-16) found that DF C-states, power-down and determinism do not affect
local streaming. Doubled refresh above 85 °C DIMM temperature costs ~7 %. Milan's HSMP (protocol 2, SMU 45.103.0)
reports neither socket temperature nor DIMM temperature, and no jc42 sensors appear.

### The kernel's share of the cores

The cmdline keeps the kernel off the pool's cores: `isolcpus=domain,managed_irq,<92 cores> nohz_full=<same>
rcu_nocbs=<same> irqaffinity=0,24,48,72,96,120,144,168`. Housekeeping runs on the first physical core of each
node and its SMT sibling. The pool takes the 92 isolated physical cores (`construct::physical_cores` reads
`/sys/devices/system/cpu/nohz_full`) and uses the first SMT sibling of each. On 2026-10-16 this took the token
from 37.1–37.3 to 36.4–37.0 ms (experts 13.0–13.6).

> "especially as we're considering this box an appliance and less of a Linux system … we can keep them off of our
> cores because they're not really doing anything to benefit us." — tcdent, 2026-10-16

Before isolation, every pool core took the 250 Hz tick, and the NIC's 64 queues landed on node 2's pool cores,
the slow node.

## 2. Memory

### Read ceilings (boot probe, `rockem.probe=1`)

| when | mode | node-local, each node | all nodes, local pages | pages interleaved over all nodes |
|---|---|---|---|---|
| 2026-10-15 | NPS1 | 154 / 148 | 297 | 165 |
| 2026-10-15 (later boots) | NPS1 | 158 / 152 | 306–314 | 166 |
| 2026-10-15 19:35 | NPS2 | 84 / 84 / 80 / 86 | **326** | 166 |
| 2026-10-16 12:35 | NPS2 + APBDIS + P0 | 93 / 93 / 78 / 86 | 334 | 164 |
| 2026-10-16 runs | same | 88–94 / 89–94 / 78–81 / 86–89 | 333–334 | |
| 2026-10-16 16:20 | + DF C-states off | 92/91/79/86, then 84/83/79/86 | 321–323 | |
| 2026-10-16 16:40 | + DIMM swap, cooler turned | **93 / 94 / 83 / 89** | 330 | |

All values are GB/s of read streams. Each thread reads its own buffer, first-touched by itself, and all threads
start together. The table reports the best of 3 runs.

- **A node's ceiling is ~82–91 % of its four channels' 102.4 GB/s.** Socket 0's nodes reach 91 % (Dell's
  figure). Socket 1's nodes stay at 78–89.
- **Interleaving costs ~half.** At 164–166 GB/s it is the same in both NPS modes, because the socket links cap
  cross-node traffic.
- **Socket 1 is slower, and node 2 is the slowest in every run.** Excluded so far:
  - the fabric clock (both sockets run 1600, pinned at P0)
  - a dropped channel (the nodes hold equal memory, and all modules train at 3200)
  - the one SK Hynix module among fifteen Samsungs. It was swapped to socket 0 on 2026-10-16, and its new node
    read its best ever (93).

  Turning CPU2's cooler raised socket 1's nodes by 3–4 GB/s each. That fits the thermal explanation (inference:
  the swap happened in the same boot, so the two changes are confounded).
- **The target.** Dell's Milan BIOS characterization (March 2021, the same DIMM population) reached ~378 GB/s
  STREAM Triad (90–92 % of 409.6). metatron's 323–334 is ~79–82 %.

  > "real world data which supports a new target for our practical limit" — tcdent, 2026-10-16

  Dell's NPS4 and CCD-as-NUMA settings do not transfer to a 6-CCD part. The fabric pin did transfer, but only
  on socket 0.

### Threads a CCD

| when | scope | 1 a CCD | 2 | 4 | 8 | every SMT sibling | 2 MiB pages |
|---|---|---|---|---|---|---|---|
| 2026-10-15 15:47 (NPS1) | node 0 = socket 0, 6 CCDs | 118 | 143 | **156** | 149 | 145 | 148 |
| 2026-10-15 19:35 (NPS2) | node 0, 3 CCDs | 66 | 79 | **85** | 84 | 82 | 82 |
| 2026-10-16 12:35 (NPS2 + APBDIS) | node 0 | 64 | 78 | 83 | 83 | | |

Four cores a CCD saturate one node's channels, and eight cost nothing. Dell's one-core-a-CCD result is the same
2–4 % here: STREAM Triad below reads 290–296 at 12 or 24 threads against 283 at 92.

### The box's STREAM figure (measured 2026-10-22, probe boot, `bench-stream`)

STREAM as Dell ran it for the Milan 378 GB/s Triad figure: three arrays a thread, far larger than any cache (64 MB each,
192 MB a thread), re-read every iteration, non-temporal stores for the result (the vendor compilers' codegen; a plain
store costs a read for ownership and prints a quarter less), bytes counted STREAM's way, best of 10. Threads pinned
one a CCD and up; the probe's idle pool resident.

| threads | Copy | Scale | Add | Triad |
|---|---|---|---|---|
| 12 (one a CCD) | 267 | 268 | 295 | **290–293** |
| 24 | 276 | 278 | 296 | **296** |
| 48 | 282 | 283 | 290 | 290 |
| 92 | 277 | 280 | 282 | 283 |
| 12, 256 MB arrays | 265 | 267 | 278 | 280 |
| 12, 16 MB arrays (48 MB a CCD: partly L3) | 263 | 265 | 348 | 358 |
| 12, plain stores | 215 | 204 | 236 | 228 |
| socket 0 alone / socket 1 alone (6 threads each) | 152 / 133 | 156 / 139 | 165 / 140 | **169 / 147** |
| node 0 alone / node 2 alone (3 threads) | 75 / 56 | 79 / 71 | 81 / 62 | 83 / 75 |

**The box's Triad is ~293 GB/s, 72 % of 409.6; pure reads run a few % above it (the decode prefetch reads 316 on
fresh bytes, `prefetch_job` rows).** The 23 % to Dell's 378 is the box, not software: socket 1 reads 13 % less than
socket 0 alone (node 2 is the weak node in every measurement since 2026-10-16), and socket 0 alone is 11 % under
Dell's per-socket 189, the NPS2-against-NPS4 band (Rome: NPS2 = 0.92 of NPS4) plus interleave. There is no
cross-socket loss: "both sockets together" reads 290 because every thread gets the same bytes and the round ends with
the slowest — 2 × 147 — not because anything crosses the fabric (nothing does, in the bench or in production, where
every expert byte is node-local). The probe's "nodes alone sum 365, together 337" is the same arithmetic. What is
real is that the slowest node ends every layer, which is what the run-rate placement (`calibrate`/`retrain`) exists
to absorb. The 7R13 cannot do NPS4 ([[epyc-7r13-topology]]); the socket-1 deficit and the fabric pin
are firmware or hardware (§1). A STREAM run with default-sized arrays (15 MB a thread at 16 threads) fits this box's
L3s and would print ~358 here — any 378 quoted for a Milan box needs its array size.

**How to read a streaming bench on this box (learned the hard way, 2026-10-22):** (1) a buffer re-read every round
is partly served from L3 whenever the per-CCD footprint is within a few × 32 MB — the L3 keeps a protected share of
what hit before (§3) — and the fewer cores a CCD, the smaller the footprint: `bench-touch touch` at two cores a CCD
read 383 GB/s with 40 MB a core, 608 with 20 MB, 318 with 72 MB; at eight cores a CCD (320 MB a CCD) it was honest
within 1 %. (2) The kernel zeroes a buffer as it is first touched and the last ~32 MB a CCD of that sit in its L3 —
so even memory "no round has read" reads from cache in the last rounds at one or two cores a CCD (Triad read 496 at
one a CCD). Stream a scratch after allocating, and report the median round, never the best. (3) A rule derived at a
small footprint — "two cores a CCD read the ceiling" was written here this morning from (1) — is an artifact; the
fresh-memory rows read 247 / 259 / 286 / 290 GB/s at 1 / 2 / 4 / 8 cores a CCD. (4) `bench-touch` on a probe boot
reads 5–10 % under a serving boot's production rows (the halted pool stays resident); compare modes within one run.

### What places memory, and why it matters

**Never stream across nodes** (NPS2, 2026-10-15). With every core reading its own node's pages the box reads
326 GB/s. With the pages interleaved it reads 165. The pool's `slot_order` lets a core take items from its own
CCD, then its own node, then its node's other CCDs, and never from another node. Removing cross-node work
stealing took the expert phase from 30.5 to 23.1 ms a token. Before that, NPS1 cross-socket steals were "poison
too, but rarer".

**Layout follows the nodes** (2026-10-16, `layout.rs`). The expert stream is cut into blocks of 32 experts. A
block is 4 runs of 51 2 MiB pages, and run `r` holds every expert's share `r`: 8 gate/up items of 278,528 B,
then 4 down items. Each expert record is 13,369,344 B with no padding. The loader first-touches run `r` from a
thread pinned to node `r mod nodes`, so every node holds exactly its share of every expert. This took the expert
phase from 24.6 to 13.7–14.0 ms a token, i.e. 4.5 GB in 13.9 ms = **323 GB/s**, the measured all-local ceiling.

> "expert slices adjacent to the dies that access them" — tcdent, 2026-10-16

**Placement is trained on serving** (2026-10-16). The pool's own run ends reweight each node's share of items:
`layout::placement_from` moves only the difference, and `Stream::migrate` copies at ~8 GB/s. The shares converge
to 26.7 / 24.8 / 22.2 / 26.4 %, and the run ends fall within 0.1 ms of each other. The idle probe does not
predict the pool. Its ranking (84/83/77/86) put node 3 third, but under the pool node 3 is the fastest. Node
rates in place were 400 / 372 / 333 / 395 items/ms. Seeding each node with a contiguous block of every expert's
rows matters: an every-fourth-item split cost ~1.5 ms.

**Item shape matters more than page size.** In the 2026-10-15 probe (bf16, K=5120, 2 MiB pages alternating
nodes, every item on its rows' node):

| item shape | bf16 GB/s |
|---|---|
| 16-row items claimed in order (160 KB each) | **264** (ceiling 312) |
| 64-row items | 106 |
| 256-row items | 90 |
| a static partition: each thread its own node-local pages | **84** |
| unplaced | 89 |

At K=4096 the same probe read 151, which was power-of-two stride camping. When 48 cores start their streams at
the same offset of 2 MiB-aligned pages, they hit the same channels and banks in lockstep. Keep items small and
consecutive, and never give a thread one long private range.

**One core** (2026-10-15 probe, NPS1) streams bf16 at 16–22 GB/s, FP8 at 4.1 (the e4m3 unpack is
compute-bound), and INT4 at 13–28. With 33.6 MB of rows streaming from DRAM, a core feeding the expert kernel
pulls 21 GB/s (2026-10-19). One core's hardware prefetchers hide its own stream.

**Decode's expert run streams 375 GB/s** across both sockets (queue 074, 2026-10-19): 467 items of 128 KB in
160 µs. Every node ends alike. A quarter of the cores finish early, which is what a bandwidth-bound run looks
like. Finer items (`SUB_FEW` 4) cut the tail without moving the wall. This 375 is above the stream probe's
323–334 (see §8).

## 3. Caches

**The L3 model in one place (read this before any cache experiment; every line is measured below).**

1. **Victim cache.** A DRAM fill lands in the reading core's L1/L2; it reaches the CCD's L3 only when that L2 evicts it.
   A line still in a core's L2 is not in L3: a CCD-mate reading it gets local-homed memory from DRAM, not forwarded
   (remote-homed is forwarded). So the core that touches a line must be the core that reads it ("A touched line is
   not yet an L3 line", 2026-10-21/22).
2. **Two classes of line.** A line that entered L3 once, as an L2 victim, is in the low class: replacement takes those
   first, oldest first. A line that has *hit* in L3 is protected (95 % of a 24 MB read set kept). Re-referencing a line
   while it is still in L1/L2 does not promote it; a dirty victim is not kept at all; a `prefetcht0` touch keeps ~7–9 %
   (touch with loads) (2026-10-22).
3. **Expiry is bytes, not time.** A low-class line's risk is the fresh bytes its CCD inserts after it: at 12 MB a CCD
   the oldest tenth of a touched set loses ~19 %, the newest ~4 %; 24 MB loses 41 %; idle time inside a layer changes
   nothing (2026-10-26, `bench-l3life`).
4. **What fresh lines lose to:** the previous round's lines the run *read* (protected, useless). Flushing them restores
   ~100 % but costs 12.4 ns a line (~350 µs a core a layer); a second touch pass (an L3 hit a line) costs ~200 µs a layer
   on the touching cores, which the decode prefetch window does not have; re-touching pays only where a line's loss
   probability exceeds ~25 % (an L3 re-touch ~4× cheaper than a DRAM first touch) (2026-10-22, 2026-10-26).
5. **Ordering moves loss, it does not remove it:** reading the oldest touched lines first loses the newest instead (the
   node's own deques, 2026-10-26, queue 099). **Once-read cold lines are useful victims:** reading them `prefetchnta`
   raises the next window's loss (4.2 → 8.4 %, `bench-window`). Loss is set by the fresh bytes a CCD inserts.
6. **A non-temporal read does not protect:** the run's warm reads with `prefetchnta` cut the next round's loss 15 → 3 %,
   but the hinted gemv reads ~35 % slower (2026-10-26).
7. **Interference, separate from loss:** a CCD's DRAM path saturates at ~40 GB/s with 3–4 streaming cores alone, ~17–18
   GB/s each with the whole socket streaming; streamers past saturation slow the CCD's L3-fed cores 2–5× (queue 099).
8. **Benches:** `bench-l3` (keep/stream, CAT), `bench-reuse` (self/peer), `bench-l3life` (survival by position),
   `bench-order --fresh` (a run's warm/cold mix), `bench-window` (the decode window then the run), `bench-interfere`,
   `bench-item`. Fixed warm sets are L3-protected by their own reuse: model the node with **fresh** lines.

**How Milan's L3 fills** (architecture facts, from `l3-intentional-expert-path`, 2026-10-02):

- The L3 is a victim cache of L2. A DRAM fill lands in L1/L2 and reaches L3 only when L2 evicts it.
- To put data in a CCD's L3, stream it through a core of that CCD, with plain loads or PREFETCHT0/T1/T2 (not
  NTA).
- No instruction places a line in L3 on AMD.
- Another CCD's L3 is reachable only by a fabric coherence transfer.
- GPU writes over the BAR go to DRAM; EPYC has no DDIO-like allocation.

**What one CCD delivers** (probe 2026-10-15 16:57–17:22, NPS1, node 0, CCD 0):

| measurement | GB/s |
|---|---|
| CCD 0's 8 cores stream a slice from DRAM | 45 |
| the same slice again, 4 MB (L2: 512 KB a core) | 231 |
| again at 8–16 MB | 152–163 |
| again at 24 / 32 / 48 MB | 99 / 137 / 76 |
| another CCD of the socket reads what CCD 0 streamed | 75 (over the fabric: better than DRAM, half of local L3) |
| a CCD from its L3 to its cores, beyond their L2s, at 4–8 cores | **~60, whatever the kernel** (int4 / bf16 / plain stream) |
| one core from L3: int4 / bf16 / plain stream | 30 / 47 / 62 |
| a CCD's share of DRAM when all six CCDs pull | ~26 |
| all six CCDs re-reading their own 11 MB at once | **351** (DRAM: 148). A later boot: 408 vs 146 |
| NPS2, three CCDs of a node re-reading their own 11 MB | 209 (DRAM: 79) |

So the L3 cleanly keeps up to ~16 MB a CCD of streamed data, and an L3 hit is worth ~2.3× a DRAM read when every
CCD pulls. The INT4 kernel is not the limiter from L3: one core reaches 30 GB/s from L3 against 12.5 when fed
from DRAM.

**resctrl** (2026-10-15). The dev kernel has CONFIG_X86_CPU_RESCTRL, and the 7R13 reports cat_l3, cqm_llc,
cqm_mbm_total/local and mba. After `mount -t resctrl resctrl /sys/fs/resctrl`:

- L3 cbm_mask ffff (16 ways), 16 CLOSIDs, min_cbm_bits 0
- monitoring of llc_occupancy, mbm_total_bytes and mbm_local_bytes, with 256 RMIDs
- MBA at 1 % granularity
- 12 L3 domains

While serving, every L3 reads 32–33.5 MB occupied. A monitoring group's occupancy per domain works as a readout:
after CCD 0 streamed an 11 MB slice twice, the group showed 32.8 MB on domain 0 and ~0 elsewhere. Way
reservation and MBA caps have not been tried.

**The L3 prefetch experiments:**

| when | what | result |
|---|---|---|
| 2026-10-15 (first form) | prefetch predicted experts into L3 | zero gain. The slices were not in the computing CCD's L3 in time (mechanical) |
| 2026-10-15 18:40 | decode: the next layer's predicted experts' slices pulled under the attention phase, slotted to the CCD that computes them | experts 24.5 → 19.5–19.8 ms a token. 85–89 % of slices land in the window, 100 % once the block layout shortened the phase |
| 2026-10-20 (queue 078, `Routed::prefetch_lead`) | prefill: each slot's first 32 MB of non-resident gate/up slices at each dispatch wait | **no gain**: experts lap 17.78 → 18.24 ms a layer-chunk. 98.6 % of slices were pulled before the run overtook them. Reverted |

The prefill result means one of two things: either the run is not DRAM-bound at its start, or the warmed lines are
not served from L3. No L3 hit count (the Zen 3 demand-fill-source counters) has been taken yet. Taking one is the
step before trying again.

**The activations' L3 traffic** (queue 021/065, 2026-10-19). Each of an expert's 64 gate/up parts runs on a
different core and pulls the expert's tokens' activations (4 KB a token) into its own L2. With all 92 cores doing
this at once it costs ~10 % of the kernel's rate: 58.6 vs 65.7 GMAC/s a core at 16 tokens. One core alone loses
3–4 %. On the shelf at ≤ 0.13 ms a prompt token.

### A touched line is not yet an L3 line: the core boundary inside a CCD (measured 2026-10-21/22)

**The question.** Decode's prefetch touches the next layer's expert slices on four cores of each CCD so the run finds
them in that CCD's L3. The run measured only 83 % of a *completed* prefetched expert's lines from L3 (`prefetch_use`
rows, the fill counters around every item), and raising that share (CAT, more prefetch cores, a better predictor)
never moved the run's time in proportion. Phase 0 of docs/expert-cache-architecture.md had recorded a planned set
losing 30–54 % to a competing stream. tcdent: "I don't think that we've accurately mapped and understood the L3 cache
eviction, and so we're fighting against evictions when it's not that we're running out of megabytes of headroom …
we're probably stepping on our own toes." This section is the map. Every number below is from one sitting on a quiet
box (the serving node resident and idle, the cards idle, THP `always`, resctrl mounted with every domain at `ffff`
and MB unthrottled), read back from a file on the box so the console could not truncate it.

**Method.** `bench-l3` (bringup/bench/src/l3.rs, the musl binary in `bundles/tools/`, run from the box over
`bringup/box/box.sh`): per CCD, four "keep" cores touch X MB of fresh lines (AVX loads, a line each); the CCD's other
cores stream Y MB of fresh DRAM lines through the same L3; then each keep core reads its *neighbouring* keep core's
slice with the Zen 3 fill counters programmed around the read (`common::pmc`: event 0x43 demand fills by source,
umask 0x02 local CCX / 0x08 local DRAM / 0x50 far; 0x70/0x71/0x72 L2 prefetches that hit L2 / hit L3 / missed L3).
"From L3" is the share of fills, prefetched or demanded, that came from the CCX rather than memory. `--reread` has
the stream cores read the set a second time. `bench-reuse` (src/reuse.rs) runs the decode `gemv` over a region in
three modes — *self* (the toucher reads it), *peer* (its CCD neighbour reads it), *peer-flush* (the toucher streams
1 MB of scratch through its L2 first) — with the same counters. Where a buffer is homed: `/proc/<pid>/numa_maps`
of the running bench. The expert record is ~13 MB (`layout::RECORD`); a CCD's share of a layer's 13-expert plan is
~14 MB; a CCD's L3 is 32 MB, 16-way, a victim cache of the eight 512 KB L2s.

**1. Alone, a set stays.** One CCD, nothing competing (Y = 0), neighbour read:

| X touched | from L3 |
|---|---|
| 4 MB | 97.1 % |
| 8 MB | 96.7 % |
| 13 MB | 98.5 % |
| 16 MB | 97.8 % |
| 24 MB | 95.0 % |
| 13 MB, with the run's 8 MB stream after it | 95.4 % (95.2 % on CCD 0) |
| 12 MB, 16 MB stream (phase 0 recorded 46 %) | 91.6 % |
| 13 MB, 8 MB stream concurrent with the read | 93.6 % |
| 13 MB, then reread by the stream cores | 99.9 % |

(CCD 1; its buffer, as it turned out, 89 % remote-homed — see 4.) Capacity is not the problem at 13 MB a CCD, nor
at 24; a `prefetcht0` touch leaves 7 % (phase 0: 9 %), so the touch must be loads.

**2. The loss appears with the other CCDs, and is not DRAM-channel sharing.** 13 MB + 8 MB stream each:

| CCDs touching at once | from L3 |
|---|---|
| 3 (one node: its four channels shared) | 94.8 % |
| 6 (socket 0) | 92.6 % |
| 6 (socket 1 alone) | 66.0 % |
| 6 (CCDs 3–8, both sockets) | 71.1 % |
| 12 | 77.9 %, 76.5 %, 78.9 % (12 MB), 78.0 % (THP `never`) |
| 12, no stream (the touch alone) | 87.6 % |
| 12, 6 MB each | 87.2 % |
| 12, CAT 12 keep ways | 83.1 % |

**3. It is per node, not per socket, and a constant share of the lines.** Single CCDs alone, 13 MB, Y = 0:
CCD 0 95.2 (with Y = 8), CCD 1 98.5 / 97.8 / 98.0, CCD 2 97.9 — all node 0; CCD 3 81.1, CCD 4 80.6 (node 1); CCD 6
80.4 / 82.3 (40 rounds) / 82.8 (THP `never`), CCD 9 82.1 (nodes 2, 3). On CCD 6 a 4 MB set reads 76.8 %: the share
lost does not depend on the set's size, nor on page size, nor on how long the set sits. On CCD 6 with `--reread`:
the keep cores' first read 82.0 %, the stream cores' reread right after **99.8 %**, zero DRAM fills. Nothing was
evicted. The lines missed on the first read were fetched from DRAM *then*, and were in L3 from then on.

**4. The variable is where the memory is homed.** Node free memory at the time: node 0 680 MB, node 1 356 MB,
node 2 65.4 GB, node 3 30.0 GB (the node reserves 212 GB of hugepages and the context cache holds 157 GB). The bench
on CCD 1 (node 0) got its 1.3 GB buffer 89 % on node 2: *remote*. The bench on CCD 6 got it 100 % on node 2:
*local*. So the table in 3 reads: remote-homed lines, touched on one core and read by a neighbour, ~98 % found;
local-homed lines, ~82 % on the first read. Production's expert slices are node-local by design (`layout::placement_runs`),
so production is always the local case — and its 83 % for a completed expert is this number.

**5. Where the lost lines are: the toucher's L2.** `bench-reuse`, 1.5 MB a core, 8 cores, 30 rounds:

| the region read by… | CCD 1 (remote-homed) | CCD 6 (local-homed) |
|---|---|---|
| the core that touched it (self) | 0.97 hit / 0.11 miss (~90 %) | 0.97 / 0.11 (~90 %) |
| its neighbour (peer) | 0.96 / 0.12 (~89 %) | **0.73 / 0.35 (~68 %)** |
| its neighbour after the toucher flushed its L2 with a 1 MB stream (peer-flush) | 0.74 / 0.34 (~68 %) | 0.71 / 0.37 (~66 %) |

(L2 prefetch fills that hit / missed L3, a line read; the demand fills were ~0.01 a line: the L2 prefetcher brings
nearly every line.) A core finds its own touched lines equally well everywhere. A neighbour finds them for
remote-homed memory and loses a third of them for local-homed memory — and a third of a 1.5 MB region is exactly
what a 512 KB L2 holds. In bench-l3, four touching L2s hold 2 MB of a 13 MB set, 15 %: the measured first-read loss
is 16–18 %.

**The mechanism (inferred; fits 1–5, with two open points).** The L3 is a victim cache: a line a core has just
loaded is in that core's L1/L2 and reaches L3 only when the L2 evicts it. The last 512 KB each toucher streamed is
therefore not in L3 when the run starts. When another core of the same CCX asks for such a line, the CCX must either
forward it core-to-core or fetch it from memory. The measurements say the fabric *forwards it for remote-homed
memory and refetches it from DRAM for local-homed memory* (DRAM being the cheaper source when it is on this node),
and only that refetch installs the line in L3 — which is why a second read finds 99.8 %, why the loss is a fixed
share (the touchers' L2 capacity over the set), why it depends on nothing else we varied, and why neither CAT nor
more prefetch cores could help (nothing was evicted; more touching cores means more L2s holding tails). It is also
why phase 0's "L3 and DRAM share the CCD's delivery" came out as it did under concurrent streams: a share of the
"L3" reads were DRAM reads.

Open points: (a) a 4 MB set on CCD 6 loses 23 %, where four 512 KB L2s would predict up to 50 % — an L2 streaming
a set may keep less than its capacity of it; (b) peer-flush loses on both nodes, where pushing the lines into L3
first should have restored them. Both are measurable with the same two benches; neither changes the conclusion that
the lines a neighbour misses are the ones that never left the toucher's L2.

**What phase 0 got wrong.** Its one-CCD retention numbers (69.5 % at 12 MB + 8 MB; 46 % at 12 + 16) were taken
without recording where the buffers were homed or what else was running; today's same cells read 95.4 % and 91.6 %
(remote-homed) — so those numbers do not describe the L3's capacity or replacement, which both benches now show to be
nearly ideal at 13–24 MB a CCD. Its CAT result (+10–12 points at production shape) stands as measured but is now
explained as partly an artefact of the same first-read refetch. Lesson, for every cache measurement from here: record
the buffer's homing (`numa_maps`), the free memory per node, and whether the reader is the toucher.

**The design consequence.** The placement principle was right and incomplete: node-local placement puts the bytes on
the chips next to the CCDs that use them; inside the CCD the touch and the read must also be *core*-local, because a
core-to-core hand-off of local-homed lines is a DRAM read. For decode's prefetch: the run should read each planned
item on the core that touched it (bench-reuse's self row, ~90 % everywhere, the L2-resident part hitting L2), which
means the prefetch job recording each item's taker and the run's per-CCD queue handing planned items to their takers
first — core affinity for planned items, a few bytes a job. Predicted: completed experts' lines from L3 83 → ~95 %
on the existing `prefetch_use` counters, no bench. The same rule reads back onto the expert run itself, whose eight
cores a CCD today share nothing deliberately: any line one core pulls that another core of the CCD later needs is a
second DRAM read.

### What the L3 keeps of a touch (measured 2026-10-22, `bench-reuse`, CCD 6, local memory)

The decode prefetch fills a CCD's L3 by *touching* lines (one load a line); the run then reads them. What the L3 keeps
of such a set, for the core that touched it (self) and for its neighbour (peer), and of a set the gemv has *read*
before (warm: the same region every round); L2-miss lines served from L3 (`pf_hit_l3 / (pf_hit_l3 + pf_miss_l3)`):

| set a CCD | touched, self | touched, peer | touched twice in L1 (self2) | re-referenced in L2 (self3) | stored back unchanged (selfw / peerw) | read before (warm) |
|---|---|---|---|---|---|---|
| ≤ 4 MB | 99 % (the L2s hold it) | 99 % | | | | |
| 6–8 MB | 91–92 % | 89 % | | | | |
| 12 MB | 91 % | 72 % | 91 % | 91 % | **7 % / 2 %** | 96 % |
| 24 MB | 66 % | 48 % | 66 % | 66 % | 3 % / 1 % | **95 %** |

(The L3 is 32 MB, 16-way, 32768 sets, cores 48–55; `bench-touch`'s items/slices modes confirm the touch kernel reaches
the DRAM ceiling on contiguous streams, 366–378 GB/s box-wide, and 272 → 322 on the production item shape as the lines
in flight go 4 → 16.)

Reading: a line that enters the L3 as an L2 victim after one reference is the first the replacement chooses, and it
is chosen well below capacity (9 % of a 12 MB set, a third of 24 MB, with nothing else running). A line that has *hit*
in the L3 is kept (95 % of 24 MB). Re-referencing a line while it is still in L1 or L2 does not change its class, and a
dirty victim is not kept at all (the stored-back set read from DRAM). So the only promotion a touch can buy is a
second pass over the set after it has left the L2s — an L3 hit a line, ~200 µs a layer of L3 time at the plan's size,
which the prefetch window (fully used by the first pass at ~23 GB/s a CCD) does not have. The node's 85 % for a
completed prefetched expert against the policy's ~88–91 % at its 14 MB plan is that floor plus the run's own DRAM
installs; the one lever left on that gap is CAT around the affinity, bounded at ~0.7 ms a verify step (inferred from
the serial delivery model).

**Why the lines are lost, and the price of keeping them** (2026-10-22). The cache is not short of room; it is full
of the previous round's lines, *protected* because they were read: with the previous round's region flushed
(`clflushopt`) before the touch, the toucher found 100 % of 12 MB and 98 % of 24 MB (against 91 % and 66 %), the
neighbour 77 % / 75 % (against 72 / 48; its remainder is the L2 tail). In the node that is the previous layer's planned
experts, read by its run and useless since. The flush itself costs 12.4 ns a line a core on this box (5 GB/s of lines,
one CCD or all twelve alike): a core's share of a layer's plan, 1.75 MB, takes ~350 µs, most of a run or half the
prefetch window. The other promotion, a second touch pass that hits in the L3, costs ~200 µs a layer of L3 time the
window does not have. So a touch-based prefetch into this L3 keeps ~90 % of its lines at the plan's size, the node's
85 % is that plus its run's own installs, and the last five points cost more than the ~0.3–0.4 ms a verify step they
would return (CAT with cores split by item kind, or a partial second pass).

### A touched line's lifetime is bytes, not time, and the oldest go first (measured 2026-10-26, `bench-l3life`, CCD 7)

Each core touches a fresh set, then the set is reread in touch order with the fill counters read every tenth of it;
the share of each tenth that came back from DRAM (one load a line; the reader is the toucher):

| shape | between touch and reread | lost | oldest tenth → newest tenth |
|---|---|---|---|
| 1 core × 12 MB | nothing | 16.5 % | 32.6 29.5 26.7 21.1 16.9 11.3 9.4 6.8 5.0 5.6 |
| 8 cores × 1.5 MB (12 MB a CCD) | nothing | 10.7 % | 18.7 16.3 15.5 14.1 11.3 8.4 6.9 5.9 6.2 4.2 |
| 8 × 1.5 MB | 2 ms idle | 11.0 % | 23.1 … 2.9 |
| 8 × 1.5 MB | 50 ms idle | 19.0 % | 30.3 … 10.9 |
| 8 × 1.5 MB | 1 MB a core of other fresh lines | 26.1 % | 40.7 … 12.8 |
| 8 × 3 MB (24 MB a CCD) | nothing | 41.2 % | 72.5 … 15.8 |

- Within the once-referenced class the replacement is age-ordered: a line's risk is the fresh bytes the CCD has inserted
  after it. Time alone does not matter on the scale of a layer (2 ms idle: no change); 50 ms of idle raises it, the
  box's background traffic (not separated).
- A non-temporal read keeps a line out of the protected class (`bench-order --warm-hint nta`, one CCD, 15 MB fresh a
  round): the next round's warm-only loss 15.2 → 2.8 %, cold-first 34 → 10 %; but the hinted gemv reads ~35 % slower
  (8.7 → 11.7 µs an item), so no run got faster.
- **The re-touch's break-even:** a re-touch reads from L3 at ~14 GB/s a core, a first touch from DRAM at ~3.3 GB/s a core
  in the window (1.5 MB in 440 µs), ~4× apart. Re-touching a line saves its loss probability p of a DRAM line and costs
  ~1/4 of a first touch, so it pays where p > ~25 % — at the node's 12 MB a CCD, only the oldest tenth or two. Measured
  at the node's window (`bench-window`, 46 cores): the loss is 6.4 % of the touched lines either way the run is ordered
  (oldest or newest first), and re-touching all of it (1.4 %) leaves the run's DRAM lines and time unchanged.

### What a store costs, and musl's malloc (measured 2026-10-22, the node's planner on core 7)

A store into a line this core does not own costs ~35 ns, and it costs the same whether the line was last *read* by
the pool's 88 workers (shared in their caches: an invalidation across the box) or evicted long ago (a fetch). The
node's run plan — ~25 KB of fresh data a layer that every worker then reads — took 31 µs on the builder core, every
stage's time proportional to the bytes it wrote, and a second, warm plan of the same layer the same 27 µs (the reads
were warm; the stores were not). Reuse order did not matter: last-in-first-out blocks (hot but shared) and
first-in-first-out blocks a megabyte deep (cold) both cost ~35 ns a line. What does: writing the blocks ahead of
time, while the core idles, so the plan's stores find the lines modified in its own L2 (`common::alloc::prewarm`) —
the plan 31 → 17.8 µs, the remainder its loops (two counting sorts at ~4 ns an item) and its reads of lines the
workers wrote (the prefetch's takers).

musl's malloc (mallocng, the static node's libc) hands a freed medium block (1–8 KB) back to the kernel and page-faults
it in at the next allocation. The node's per-layer allocation set — two pool jobs, a run plan, a prefetch's items,
~20 blocks — cost 77 µs an iteration on the host under musl against 1.9 under glibc and 1.5 under `common::alloc`
(`bench-alloc`, bringup/bench; build it for both targets). On the box that was a third of the main thread's 3.5 ms a
verify step. `common::alloc` (common/src/alloc.rs) is the node's allocator now: power-of-two classes from 16 B to 1 MiB,
a magazine a class a thread, a central list a class, 2 MiB slabs mapped populated and never unmapped.

How it was found: stage stamps inside the plan (`plan_build` rows: cumulative µs after each stage), then the same plan
built twice with the second timed — if warm is not faster, the cost is not what the code reads.

## 4. Clocks and power

**The power cap sets the all-core clock** (2026-10-20, APERF/MPERF through `/dev/cpu/N/msr`, power through the
`amd_hsmp_hwmon` hwmon):

- Each socket is capped at 280 W, which is also `cap_max`, so the cap cannot be raised.
- When the expert gemm runs on ~24 or more cores of a socket, the socket reaches 280 W and the clock falls from
  **3.44 to ~3.04 GHz**. That is the whole 13–17 % "all-core drop" per core that queue 078 left unexplained.
- Work per GHz is unchanged at every load (31.1 GMAC a GHz), so the drop is not cache or fabric contention.
- A register-only `vpmaddubsw` loop holds 3.44 GHz on all 92 cores at ~250 W. The gemm's loads and stores add
  ~2.3 W a core.
- 23 + 23 cores split across the sockets keep 3.44 GHz.
- **Power per MAC is a speed lever on this box.**

**No cpufreq driver.** The firmware sets the clock.

**Idle power** (HSMP, 2026-10-16): 75 / 81 W a socket before the node starts, and 92 / 102 W with the pool
resident in MWAITX.

**Ticks and wake-ups** (2026-10-20):

- Idle pool threads look for work once a millisecond (the idle state's 1 ms sleep).
- Busy pool cores still take ~500 timer ticks a second, because `nohz_full` is not effective there. That costs
  ~1.5 % on a bench.
- Busy workers between jobs wait on the job pointer's cache line with MWAITX. On 2026-10-15, 95 cores spinning
  through the card phases had heated the package for no gain.

### Power per kind of work

Measured 2026-10-20 with `bench-power` (5 s windows; each core its own first-touched buffers; clocks from
APERF/MPERF, socket watts from HSMP `power1_input`). Socket 0 idles at ~87 W with the node resident. "Extra W a core"
is (socket W − 87) / 23 from the 23-core run on socket 0, where the clock still holds unless noted.

| work, every core of the set | rate a core, 1 core | extra W a core (23 cores) | clock / socket W at 46 cores (socket 0 full) | rate a core at 46 |
|---|---|---|---|---|
| dependent scalar adds (`chain`) | 3.46 Gop/s | 2.1 | 3.44 GHz / 187 W | 3.42 Gop/s |
| AVX2 loads from DRAM, 256 MB (`dram`) | 27.1 GB/s | 2.3 | 3.44 / 163 | 2.5 GB/s (median) |
| `vpmaddubsw` on registers only (`vmadd`) | 126 GMAC/s | 3.7 | 3.44 / 254 | 124.9 |
| AVX2 loads over 16 KB, L1 (`l1`) | 198 GB/s | 6.0 | **3.04 / 280 (cap)** | 98.7 (median; see below) |
| the expert `gemm`, 32 tokens, cache-resident (`gemm`) | 107.0 GMAC/s | 6.0 | **3.06 / 280 (cap)** | 93.7 |
| the expert `gemm` from DRAM, 33.6 MB (`gemm-dram`) | 104.8 | 6.1 | **3.00 / 280 (cap)** | 86.2 |
| AVX2 loads over 2 MB, L3 (`l3`) | 82.6 GB/s | 6.4 (already 3.19 GHz at 23 cores) | **2.62 / 253** | 43.6 |
| AVX2 loads over 256 KB, L2 (`l2`) | 109.5 GB/s | 6.8 | **2.87 / 272** | 90.0 |

All 92 cores read the same as 46 a socket (each socket independently capped at 280 W).

What it says:
- **Moving data between a core and its caches costs the most power, not multiplying.** An L1 load loop draws as
  much as the whole gemm, ~60 % more than the multiplies alone; the gemm is in the load loop's power class, so its
  loads (among them the scratch weights it stores and reloads every group) are what takes a full socket over 280 W
  and the clock from 3.44 to ~3.05 GHz. Power per MAC is a speed lever; fewer L1/L2 loads per MAC is the way to it.
- **DRAM streaming is cheap in core power** (the cores mostly wait): 2.3 W a core, clock untouched.
- **L2- and L3-heavy loads lower the clock below the power cap** — 2.62 GHz at 253 W for L3 reads on 46 cores, 3.19
  GHz at 235 W on 23 — so a second limit binds there (inferred: a current limit, EDC/TDC; not measured). A design that
  serves weights from L3 to a whole socket pays clock for it.
- Anomaly, not explained: the L1 loop's per-core rate on 23 cores spread 110–197 GB/s at full clock (one core alone
  198).

The L2/L3 clock limit is per socket (measured 2026-10-20, `bench-power --mode l3`, 2 MB a core):

| L3 loads on | clock | socket W | GB/s a core (median) |
|---|---|---|---|
| 23 cores, all on socket 0 | 3.18 GHz | 233 / 95 | 55.5 |
| the same 23 split 12 + 11 across the sockets | 3.44 | 203 / 198 | 72.3 |
| 46 cores, 23 a socket | 3.18 | 234 / 238 | 55.4 |
| 8 cores, mostly one CCD (cores 1–8) | 3.44 | 163 | 71.4 |
| 8 cores, 4 + 4 on two CCDs | 3.44 | 173 | 82.5 |

So the limit binds on how much L3 traffic one socket carries, near 233 W — below the 280 W power cap (inferred: a
socket current limit; no telemetry names it). Separately, one CCD's L3 has a delivery ceiling: eight cores sharing it
get 71 GB/s each, 82.5 when the same eight are split over two CCDs.

## 5. The expert kernels on this CPU

**The format** (`common/src/q4.rs`, `q4g128-a8`):

- INT4 weights with one f32 scale per 128 columns, stored biased by +8, in tiles of four rows walked group-major
  (256 B a group).
- The activations are INT8, quantized per group of 128.
- `vpmaddubsw` takes the unsigned nibbles directly. The bias costs one `8·Σa` scalar per group per token.
- `gemv` takes one token. `gemm` takes up to `GEMM_TOKENS` = 64 and is bit-identical to `gemv` for every token.
- Four bits is the floor (tcdent, 2026-09-20). LUT kernels (T-MAC style) only pay below four bits, so they were
  declined for decode (queues 054, 056).

**Zen 3 facts that bound the kernel** (the 2026-10-20 llvm-mca study, `llvm-mca-14 -mcpu=znver3`):

- `vpmaddubsw` issues only on FP0 and FP3.
- The multiply pipes' floor is 18 cycles per 1024 MACs.
- `vpbroadcastd` takes an FP uop.

In the old gemm, a group of 8 rows × 128 k was ~71 % of an item at 32 tokens. That loop was 100 µops per 1024 MACs:
32 were the token quads' `vpbroadcastd` (one per `vpmaddubsw`) and 36 were accumulation. The rest of the item was
the flush (~13 %), activation slicing (~8 %) and setup (four `transpose8`, ~7 %).

**The rewrite** (2026-10-20, commit 7bd0e9e, the layout sub-agent's shape):

- A weight vector holds two rows' 16 k in its 128-bit halves.
- A token's 16 k is one `vbroadcasti128` that feeds four multiplies.
- Each i16 accumulator spans the whole group, exactly, reduced by unpacks. There are no transposes.
- The kernel takes two tokens at a time, and resolves the token pointers once a call.

It is bit-identical to `gemv` for every n from 1 to 64 at both expert widths.

| one metatron core, tokens | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| new gemm ÷ old gemm | 1.77× | 1.53× | 1.40× | 1.35× | 1.25× | 1.30× |

The new gemm runs at 75–106 GMAC/s.

**Rates on one core, before the rewrite** (`gemm-rate`, 2026-10-19, either socket the same):

| | cache-resident | 33.6 MB streaming from DRAM |
|---|---|---|
| a token at a time (`gemv`) | 58 GMAC/s | 40–42 GMAC/s (21 GB/s) |
| `gemm`, 16 tokens | 76 | 71–78 at 16–64 |
| `gemm`, 64 tokens | 85 | |

**All 92 cores at once, before the rewrite** (`gemm-rate`, 2026-10-19 03:30, served boot idle, 33.6 MB slab):

| tokens | GMAC/s total | a core | GB/s of weights | bound |
|---|---|---|---|---|
| 1 | 624 | 6.8 | 313 | DRAM |
| 4 | 2,369 | 25.7 | 295 | DRAM |
| 16 | 5,468 | 59.4 | 171 | compute |
| 64 | 6,931 | 75.3 | | compute at all-core clocks |

The prefill shape (2,048 slabs of 32 rows from a 134 MB pool) reaches 6,327 GMAC/s at 16 tokens (68.8 a core) and
7,100 at 64. A fresh stream for each item costs nothing.

**In real runs** (`track-experts`, 2026-10-19, an 18k prompt at `STEP` 1024, before the rewrite):

| tokens an item | 1 | 2 | 3–4 | 5–8 | 9–16 | 17–32 | 33–64 | 65+ |
|---|---|---|---|---|---|---|---|---|
| GMAC/s a core | 18 | 27 | 39 | 49 | 56 | 60 | 60 | 61 |
| share of gate/up time | 0.4 % | 0.5 % | 1.2 % | 3.6 % | 11.8 % | 29.8 % | 34.1 % | 18.7 % |

After the rewrite (2026-10-20), gate/up runs at 68.8 / 76.4 / 74.6 GMAC/s a core at 9–16 / 17–32 / 33–64 tokens.

**What each phase is bound by:**

- **Decode** (one or two tokens an expert): items run at 12–24 GMAC/s a core, which is the bandwidth floor. The
  run streams 375 GB/s (§2). The only levers are fewer bytes (more experts resident on the cards, a tighter
  format) or a higher ceiling. On the 1M build (c71c97b) resident experts fell from ~19 to ~4 a layer, and the
  experts lap of a verify step at 18k rose from 20.4 to 25.8 ms.
- **Prefill** is instruction-bound and power-capped. The cold 18k prompt's 1,513 runs stream gate/up at
  183 GB/s and down at 169, with the pool 93–94 % busy and tails 5.4 % of a run (queue 078, 2026-10-20).
  - An item's rate is set by its token count. Since 2026-10-20, the clock drop under the cap also costs a share
    of it.
  - A kernel alone on a core runs 73–85 GMAC/s at 16–64 tokens (before the rewrite); in a run it got 55–62. By
    inference, the power cap (13–17 %) and the activations' L3 traffic (~10 %) account for that gap; the two
    were measured separately, not together.
  - The rewrite moved prefill from 2.25 to 2.02 ms a prompt token at 18k, and the cold 18k prompt from 44.7 to
    39.7 s.

## 6. Experiments that did not pay

| what | when | result | where |
|---|---|---|---|
| Pages interleaved over all nodes (the old loader) | 2026-10-15 | 165 GB/s vs 297–334 node-local. Any one tensor sat on one node | §2 |
| Cross-node work stealing | 2026-10-15 | removing it: experts 30.5 → 23.1 ms a token (NPS2) | `slot_order` |
| Static per-thread partition of node-local pages | 2026-10-15 | 84 GB/s vs 264 with small dynamic items | probe |
| Larger pool items (64–1024 rows) | 2026-10-15 | slower than 16-row items (106 / 90 vs 264) | probe |
| First L3 prefetch form | 2026-10-15 | zero gain (timing) | §3 |
| Placement by the idle probe's node rates | 2026-10-16 | experts 14.3–14.7 vs 13.6–14.0 | glm memory |
| Every-fourth-item split across nodes | 2026-10-16 | ~1.5 ms worse than contiguous blocks | glm memory |
| Half-item compute parts | 2026-10-16 | no change (13.5–13.9) | glm memory |
| Prefetch yield 64 KB → 4 KB | 2026-10-16 | no change | glm memory |
| DF C-states off | 2026-10-16 | nothing | §1 |
| APBDIS + SoC P0 on socket 1 | 2026-10-16 | nothing (socket 0 gained) | §1 |
| `pcie_aspm=off` | 2026-10-15 | nothing; firmware ships ASPM disabled | §1 |
| Mixed-vendor DIMM as node 2's cause | 2026-10-16 | cleared by a swap | §2 |
| Residency loader on core 48 (cards' socket) | 2026-10-18 | expert lap 9.9–10.5 vs 9.2–9.8 ms on core 24: its reads slow the slowest node | `glm_kfd.rs` `LOADER_CORE` |
| First `gemm` in `gemv`'s tile order | 2026-10-17 | 2.4 → 3.1 ms a prompt token (slower) | 020 |
| `SUB_FEW` 4: finer decode items | 2026-10-19 | tail 20.4 → 15.6 %, wall unchanged, lap 15.6 → 16.3 ms | 074 |
| Down items' BAR stores staged as whole lines | 2026-10-20 | per-item rate level with gate/up, run wall unmoved (6.82 → 6.74 ms) | 078 |
| Software prefetch in the gemm, 4 groups ahead | 2026-10-20 | experts lap 21.35 → 21.62 ms a layer-chunk | 078 |
| Prefill L3 warm-up at dispatch waits | 2026-10-20 | 17.78 → 18.24 ms a layer-chunk | 078 |
| Weights pre-arranged in RAM (`rearrange_r8`) | 2026-10-20 | 1.20× at 2 tokens → 1.00× at 64 cache-resident, 0.95–0.97× streaming at 16+; not built | 078 |
| Per-node activation copies | 2026-10-20 | moved item rates or nothing, never the wall (no number recorded) | `cpu-expert-bandwidth` memory |
| `gemm2` (two 16-row passes of the old shape) | 2026-10-20 | 1.11–1.21× from 16 tokens, slower below; dropped for the rewrite | 078 |
| LUT GEMM at 4 bits | 2026-09-20 | set aside by reasoning (DRAM-bound decode; prefill recheck deferred) | 054, 056 |
| the new gemm's flush/setup reworked (4-token blocks, flush from registers, tables) | 2026-10-20 | llvm-mca −11 % a group; on a metatron core 5–10 % slower at 16–64 tokens | 078 |
| Zeroed arrays, u16 indices, no per-item closures in the planner | 2026-10-22 | the stage stamps did not move: the plan's time was its bytes written, not its instructions | §3, what a store costs |
| First-in-first-out block reuse (`common::alloc`, 1 MiB deep) against last-in-first-out | 2026-10-22 | the same ~35 ns a written line: cold lines and shared lines cost alike; kept (no worse, and what `prewarm` walks) | §3 |

## 7. How to measure

- **Read the code before theorizing.** For a kernel question, model the release disassembly before reaching for
  clocks, thermals or contention (tcdent, 2026-10-20). Build unstripped with the same codegen. The command below
  is assembled from the Makefile's node build plus the recorded flags:
  `os/run.sh cargo build --release --target x86_64-unknown-linux-musl -p node --target-dir /work/target-elf
  --config profile.release.strip=false`. Then feed the loop to `llvm-mca-14 -mcpu=znver3`.
- **The boot probe** (`rockem.probe=1`, `node/src/probe.rs`) measures:
  - DRAM per node, all-local and interleaved
  - the threads-a-CCD sweep, SMT siblings and 2 MiB pages
  - the DIMMs (SMBIOS type 17)
  - the L3 passes and resctrl occupancy
  - pooled bf16/FP8/INT4 GEMVs
  - each card's VRAM, link and BAR rates

  It prints through netconsole, then idles. **Compare boot probes only.** `node probe dram` from the shell beside
  a resident serving node reads ~5 % low (308 vs 324–334), because the pool's 96 workers share the cores. The
  halted probe's own pool stays resident, so a shell reading on that boot is also low (socket 0 read 83–85 vs
  93). Node rates vary ±4 GB/s from boot to boot.
- **Run a static binary on the box.** Build with `--target x86_64-unknown-linux-musl` and copy the executable to
  `bundles/tools/`. The box mounts `bundles/` over NFS as `/bundle`, and the binary runs from the telnet console
  shell (`rockem.telnet=1`) under `taskset`. Example:
  - build: `cargo test --release -p common --target x86_64-unknown-linux-musl --no-run`
  - copy the executable to `bundles/tools/gemm-rate`
  - run: `taskset -c <core> /bundle/tools/gemm-rate gemm_rate --ignored --nocapture`

  `gemm-rate2` and `gemm-rate3` are later variants: the slab bench, and the spread-activation slab bench.
- **`bench-power`** (`bringup/bench/src/power.rs`, new 2026-10-20) runs one kind of work on every listed core:
  `bench-power --mode M --cpus 1-23,25-47 [--secs 5] [--n 32]`. Modes: idle, chain, vmadd, l1, l2, l3, dram,
  gemm, gemm-dram. It reports the rate a core, the clock actually held (APERF 0xE8 / MPERF 0xE7), socket watts
  (`amd_hsmp_hwmon` `power1_input`, sampled every 100 ms) and k10temp. `bundles/tools/loadprobe(.sh)` is the
  earlier load probe (`q4::tests::load_probe`). Its source is not in main.
- **`bench-dram`** (`bringup/bench/src/dram.rs`): private first-touched buffers, pinned threads (`--stride`
  spreads them), barrier-fenced passes. Keep a thread's working set well over ~2× L3, or the "DRAM" number is
  inflated. `bench-box --no-gpu` gave the 2026-10-15 NPS1 numbers.
- **prow reads** (`bringup/reads/`):
  - `experts.sh [run] [since] [until]`: items by token-count bucket (`track-experts`)
  - `pool.sh`: per-core ends of each run from `pool_run` rows
  - `cards.sh` and `kstats.sh`

  Bound every interval join to a time window. An unbounded join across a run's zones and launches took
  prow-server down on 2026-10-20.
- **Quiet host.** Before trusting a regression, `ps` the box for stray processes (a runaway client once corrupted
  an hour of numbers). Never read numbers from a timeline/stamps boot: stamps change what they measure.
- **Live readouts on the box:**
  - `node hsmp` (fclk/mclk/power, per socket)
  - `/sys/class/hwmon/*` (`amd_hsmp_hwmon` power and cap, `k10temp` Tctl)
  - `/dev/cpu/N/msr` (the kernel has X86_MSR)
  - resctrl at `/sys/fs/resctrl`
  - the `amd_df` PMC `dram_channel_data_controller_N` counters, which read per-channel skew without a reboot
    (not yet used)

### Decode's L3 prefetch, what it is worth (measured 2026-10-20, `prefetch_use` rows, build pfuse)
A decode layer run has ~16.6 host experts: 7.2 fully prefetched, 4.5 partly, 4.9 not predicted (30 %); the prefetch
finishes 69.5 % of its items (the pool's `prefetch_job.done` under-counts: workers report only when they leave) and
~31 % of the experts it predicts go unused. Each layer's expert time fitted on those counts (6,560 runs, R² 0.67):
**31.0 µs an expert fully prefetched, 36.4 partly, 42.4 not predicted.** A complete prefetch saves ~27 %, where the
L3 probes' ~2.3× over DRAM predicts ~18 µs: the prefetched bytes are not delivered at L3 speed when the run reaches
them (candidates: eviction by the run's own lines, the wrong CCD, the per-socket L3 clock limit — not yet told apart).

**Delivery is not the problem (measured 2026-10-20, `bench-reuse`, 1.5 MB a core, 30 rounds of fresh regions):** the
decode `gemv` over a region its CCD neighbour just touched runs as fast as over one it touched itself — on 92 cores at
once 8.7 vs 8.7 GB/s a core (802 vs 805 summed), warm 9.1 (861), cold from DRAM 3.8 (361): L3-served is 2.2× DRAM at
full load. So the production 31-vs-42 µs (1.37×) says the prefetched lines are mostly not in L3 when the run reads
them (eviction or timing), not that L3 is slow to deliver. At full load an L3-served `gemv` gets ~9 GB/s a core (~70 a
CCD), far under a bare load loop's L3 rate — the one-token kernel from L3 is its own question.

**Where a decode run's bytes come from (measured 2026-10-20, build fills2, `prefetch_use` rows: each pool item reads its
core's fill counters with `rdpmc` and adds them to its expert's prefetch status; 9,471 decode layer runs at 18k).** Of
the L2 prefetcher's line requests:

| expert's prefetch | experts a run | from L3 | from DRAM |
|---|---|---|---|
| complete | 7.2 of 16.6 | 71.5 % | 28.5 % |
| partial | 4.5 | 51 % | 49 % |
| not predicted | 4.9 | 6.9 % | 93 % |

So the decode prefetch's shortfall is three measured parts: a completed prefetch still loses 28.5 % of its lines to DRAM
by the time the run reads them; 30 % of the host experts are not predicted; 27 % are partly touched when the run
overtakes the prefetch. Estimated worth a verify step, from these rates (inferred): ~3.6 ms if completed prefetches
stayed in L3, ~2.5 ms if the missed experts were predicted and completed, ~1 ms for the partial ones. Demand fills are
~2 % of the lines in every class: the hardware L2 prefetcher brings nearly everything. Instrument cost: none measured
(27.3 ms an emitted token, as without it). The counters need the NMI watchdog off (the node writes
`/proc/sys/kernel/nmi_watchdog` 0) and `rdpmc` allowed (`/sys/devices/cpu/rdpmc` 2); reading them through
`/dev/cpu/N/msr` from one thread costs ~8.5 ms for 96 cores and is unusable inside a run.

**Why a completed prefetch loses lines: L3 capacity (measured 2026-10-20, builds fills2 and sets).** Set conflicts are
not it: histogramming sampled decode prefetches by L3 set (the weights sit on 2 MiB hugetlbfs pages, so an address
modulo 2 MiB is physical; Zen 3's slice hash not modeled), a CCD's prefetch spreads over 32,745 of 32,768 sets, 9.1 lines
a used set, and only 1.2 % of its lines are past 16 in their set. The volume is the problem: a CCD prefetches ~19 MB a
layer (297k lines) into its 32 MB L3. Completed experts' L3 share falls with the layer's prefetch size:

| experts the layer prefetched | completed experts' lines from L3 |
|---|---|
| ≤ 10 (avg 8.9) | 85.9 % |
| 11–14 | 80.0 % |
| 15–18 | 73.2 % |
| 19–22 | 65.2 % |
| 23+ (avg 25.5) | 56.3 % |

and with the verify step's width (the prefetch is the union of the tokens' picks): 87 / 81 / 71 / 62 % at 1 / 2 / 3 / 4
tokens. Completed-and-retained experts are roughly flat beyond ~13 prefetched (6.6 × 80 % at 12.7, 7.9 × 65 % at 20.5):
past that a prefetched expert costs a DRAM read and evicts an earlier one. Levers: rank the prefetch by confidence and
cap it near what an L3 retains; better prediction (fewer unused experts, queue 082); the union ranked by agreement
across tokens. Gated depth 3 (081) widens the union, a cost its measurements did not separate.

**The L3 as a programmable cache (phase 0 of docs/expert-cache-architecture.md, measured 2026-10-20, `bench-l3`).**
resctrl CAT (16 L3 ways, 12 domains, mountable on the appliance) protects a touched set from other cores' DRAM fills:
12 MB touched while 8 MB of fresh lines pass through the CCD keeps 70–72.5 % in L3 unsplit, 81.7–82.2 % with 12 ways
reserved (all 12 CCDs at once); 33 → 66 % for 16 MB under 16 MB of traffic on one CCD. Replacement is not LRU (a set
smaller than its ways keeps more). A CCD delivers ~60–70 GB/s to its cores from L3 and DRAM together — concurrent
DRAM traffic halves an L3 read's rate. A `prefetcht0` touch keeps ~9 % (loads ~80 %). A read line stays in L3.

## 8. Open questions

- **The missing ~10–13 % of DRAM.** The box reaches 79–82 % of theoretical against Dell's 90–92 %. Socket 1 and
  node 2 trail despite pinned fabric clocks; the thermal explanation is the strongest candidate but unproven.
- **Decode's 375 GB/s against the probe's 323–334.** The decode expert run measures above the stream ceiling.
  Possible causes (unmeasured): L3 hits from the decode prefetch, or the probe's access pattern.
- **Whether prefill's expert run would hit warmed L3 at all.** The fill-source counters and an L3 hit count are
  the next instrument (§3). `/proc/sys/kernel/numa_balancing` has not been read: page-fault sampling was
  suspected in the 2026-10-15 INT4-from-L3 line on metatron (2 → 15 GB/s on one CCD, against elsenor's 11 → 27).
- **Power per MAC.** Which instruction mix keeps a socket under 280 W at the gemm's rate, and whether splitting
  load across sockets or capping active cores a socket beats running all 92 (§4's table).
- **Whole experts a CCD** (queue 078, the preferred form): a CCD's cores split an expert's gate/up rows, then its
  down rows, so the hidden vector stays in that L3. This would remove the global barriers and the 92-core run
  tails. Not built.
- **resctrl way reservation and MBA** for a prefetched set: available, untried.
- **The probe's pooled INT4** (64-row items) sat at 50–52 GB/s while serving streamed the same kernel at
  236 GB/s effective (2026-10-15): a probe artifact not yet understood. The pooled FP8 (60 GB/s) is far under
  96 × 4.1 one-core.
- **NPS1 with today's block layout** is unmeasured. It would place runs 0,2 / 1,3 a socket.
- **Timer ticks on busy pool cores** (~500 a second despite `nohz_full`, ~1.5 %).
