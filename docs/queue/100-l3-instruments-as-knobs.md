# 100 — The L3's documented instruments and knobs, used

**Status:** Pitched (2026-10-26). The counters are in `common/pmc.rs` (FILLS2) and `bench-order` (`--pmc fills2`,
`--pf-off`, `--chase`) as of rockem.lead's same-day build; nothing measured yet (the box serves production; next bench
boot)
**Area:** the host's expert run and its prefetch window; the benches; the cache hierarchy
**Gain:** unknown until the first boot; the candidates are a per-core prefetcher configuration for the touch and run
cores, a bandwidth cap on cold-item cores to protect the warm ones (bench-interfere's 2–5×), way reservation for the
window, and a cheap protection for prefetched slices (a mate's single read)

## Direction

> i would love for you to provide a pitch perhaps as a new q-u-e-u-e document around the instrumentation in terms of
> a way that we can utilize and experiment with those as knobs that we did not have available beforehand — tcdent,
> 2026-10-26

> where the public record has things that we did not for example the hit rule and the clock domain we should
> encourage the lead to spend more time on those based on a shape that we think it might follow because of the
> published documentation we can potentially prove that that's what we observe in our reverse engineering because
> that is a region that we still have open questions on — tcdent, 2026-10-26

The literature search (docs/l3-literature.md, 2026-10-26) found no public measurement of the two L3 behaviours we
live with — once-touched lines dropped early (A) and the L3-hit rate state (B) — but it did find, in AMD's own
register documentation, instruments and controls this box has had all along and we have never used. Every one is
verified in the Milan PPR, the QoS spec or the Linux Zen 3 event tables (docs/l3-literature.md §7, with page numbers
and encodings). This document is the pitch for using them: first as instruments that settle open questions, then as
knobs in the node.

## 1. What we did not have, and what each lets us do

### Observation

| instrument | what it reads | what it settles that nothing else could |
|---|---|---|
| **PMC 0x62 / 0x6D** (L2 fill latency, fills pending) | mean latency of each L2 fill and mean fills in flight, per core | B's first fork in one run: "the core issues less" (same latency, fewer in flight) vs "the L3 answers slower" (latency up). Every B candidate falls on one side. |
| **PMC 0x5F + MergeEvent** (exact MAB occupancy) and **0x41/40h** | the L1 side's outstanding misses and the prefetchers' issue rate | whether a prefetch-depth collapse is what the decayed state is |
| **0x70–0x72 at umask 1Fh vs E0h** | L2 prefetchers' hits/misses separately from the L1 prefetchers' | which prefetcher carries each line of a warm item; `FILLS` has mixed them under 0xFF (a value the PPR marks reserved) since 2026-10-21 |
| **L3 PMCs per core and per slice** (MSRC001_0230+, CoreId / SliceId) | lookups and misses from the L3's own side, for one core or one slice | closes the fill counter's blind spot — 0x43's "From L3 **or different L2** in same CCX" cannot tell an L3 hit from a mate's L2; `prefetch_use`'s 83–95 % has carried that ambiguity. Also a per-slice view of whether B is a slice-local effect. |
| **L3 0x90 / 0x9A** | the CCX's mean miss latency | whether B touches the miss path at all (our cold items say no; this says it in cycles) |
| **resctrl occupancy per RMID** (already mounted) | bytes of L3 held by a group of cores | the window's actual footprint against the plan's 14 MB a CCD, live, without a bench |

### Control

| knob | granularity | what it does | the experiment it enables | the production use, if it pays |
|---|---|---|---|---|
| **Prefetch Control MSR C000_0108** | per physical core, five bits: L1 stream / stride / region, L2 stream, L2 up-down | disables any hardware prefetcher on that core; plain WRMSR at CPL0; present on Milan (CPUID Fn8000_0021_EAX[13] = 1) | the gap curve with the L2 stream prefetcher off (B4): if B vanishes it lives in prefetcher training; if it persists, in the L3/XI path. Also: the touch kernel with prefetchers off vs on (does the hardware prefetcher help or fight a one-load-a-line touch?), and the gemv's L3 rate with only software prefetch at a chosen distance | a per-role prefetcher configuration: touch cores, run cores and the main thread each with the set that measures fastest. We have tuned access patterns around prefetchers we could not configure. |
| **CAT (`L3_MASK_n`, 16 ways; CLOS per core via `PQR_ASSOC`)** | ways per class of service, per CCD | reserves or partitions L3 ways; overlapping bits are competitively shared; a 0 mask forbids allocation | B6: private 2 ways per core during recovery — replacement metadata vs CCX queue. A6: fresh-line loss inside a 4-way partition at the same inserted bytes — per-set priority vs capacity (the "running out of megabytes" reading, closed for good) | the window's lines in reserved ways the run's cold stream cannot touch (bounded at ~0.7 ms a verify step by the serial model, metatron-cpu §3) — measurable now with the per-core L3 PMC rather than inferred |
| **MBA (`L3QOS_BW_CONTROL_n`, 1/8 GB/s units)** | external bandwidth cap per class of service, per CCD | throttles the L2-miss path of the cores in that class once their external bandwidth exceeds the cap | B7: can the throttle hardware produce B's signature (uniform, hit items only)? And: cap the cold-item cores at the CCD's saturation point (bench-interfere: 3–4 streamers saturate ~40 GB/s and every streamer past that slows the CCD's warm cores 2–5×) and read the warm cores' item time | the run's cold items in a capped class so they cannot push the CCD past saturation while warm items run; the cap *is* the "sum of sources" lever (099 §2.4) in hardware, no scheduling |
| **The "read by more than one core" rule** (SOG §2.6.4; US 7,640,399 schemes 3/4) | per line | a line read by a second core stays in L3 on later hits and is "preferred"/"sticky" in replacement | A2: a mate's single L3-rate read of a prefetched slice (~4× cheaper than the toucher's DRAM touch) — does it survive fresh insertions like a twice-touched line? | the window's touchers touch, a mate reads once: protection the re-touch pass buys today at ~200 µs a layer, bought for a quarter of that — if the documented rule holds for our lines |
| **PREFETCHNTA's documented semantics** ("marked for quicker eviction from the L2 … not inserted into the L3") | per instruction | a read that never becomes an L3 line | A4: count L3 fills from an NTA stream while a fresh set sits in L3 — the SOG says zero; our bench-window says the loss rises | the run's cold items read NTA so they never displace the window's lines — measured once at −35 % gemv speed (metatron-cpu §3 item 7); worth one more look with the documented L2 marking understood |

## 2. The first boot's worth of runs, in order

> **Run 2026-10-27 (bench boot, rockem.lead; bringup/box/logs/window-l3-an5.txt):** 1–3 done. B3: decayed (10 ms gap)
> l2_lat_q 143,423 vs 13,495 steady (×10.6) over 1,647 vs 2,149 fills, fills_pend 65,124 vs 14,342 (×4.5 for a ×4 item) —
> each fill waits ~14× longer at the same occupancy. B2: chase 26.8 µs an item steady (12.3 ns a hit), 27.2 / 27.6 /
> 28.5 after 100 µs / 1 / 10 ms — latency flat. B4: L2 stream off (0x08) steady 4.70, after 3 ms 14.05; L1 off (0x07)
> 6.10 / 14.0 — worse without prefetchers, not training. Together: a per-core cap on outstanding requests to the L3
> (metatron-cpu §3.8). A2: a mate's one read, loss 11.7 → 3.0 % (12 MB), 15.3 → 3.9 % (15 MB) — the lever holds. Next: 4.

1. `bench-order --pmc fills2` steady vs after 10 ms (B3): the latency/occupancy split. One number decides which half of
   the B candidates survives.
2. `bench-order --chase` after gaps (B2) with `--pf-off` variants (B4): L3-side vs request path; prefetcher or not.
3. A1′ and A2 with `bench-reuse` (docs/l3-literature.md §7.5): the hit rule's two predictions — and A2 is the one
   with a lever behind it.
4. MBA cap on streamers beside a warm core (`bench-interfere` under a CLOS): the sum-of-sources lever in hardware.
5. CAT private ways during recovery (B6) and the 4-way partition for fresh-line loss (A6).

Each is one boot-gap run with an existing bench and a knob it did not have.

## 3. What would make this a production change

- A prefetcher configuration per core role that measures faster for the touch or the run: ~0 code, a WRMSR at pool
  start (`appliance-not-linux-system`: the pool owns its cores; the kernel never reconfigures them).
- An MBA cap on the cold-item class if it protects warm items more than it slows cold ones — the run's two phases
  already separate the classes by core.
- A2's cheap protection, if the documented rule holds, replaces or shrinks the re-touch (099 §4.2, deployed).
- CAT for the window only if the per-core L3 PMC shows the window's lines actually lost to the run's stream rather
  than to the previous round's protected lines (the 2026-10-22 flush result says the latter; CAT does nothing for it).

## 4. Rules that carry over

- Warm the cores up (099 §3.1) before any L3-resident measurement; the counters above measure the recovery otherwise.
- Record homing, the toucher and the CLOS/RMID of every core in the bench log; a CLOS left set on a pool core
  outlives the bench (`PQR_ASSOC` is per logical processor and resctrl rewrites it only for its own groups).
- A Prefetch Control bit set in a bench boot must be cleared or the next bench measures the wrong machine; put the
  reset in the bench harness, not the operator's memory.
- Everything here is documented for Milan in the PPR; the lasenora VM cannot try the prefetch MSR (KVM hides the
  CPUID bit) — box only.
