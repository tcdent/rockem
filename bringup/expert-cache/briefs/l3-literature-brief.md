# Brief for rockem.l3 — what others have measured about Milan's L3 (tcdent, via rockem.lead, 2026-10-26)

**The ask (tcdent):** research as extensively as possible, with sub-agents fanned out by track, what anyone has actually
*quantified* about the behaviours below on AMD Zen 3 (EPYC 7003 "Milan", our 7R13) or its neighbours (Zen 2/Zen 4, Ryzen
5000/7000, Milan-X), and whether anyone over the years has encountered the two things we observe and cannot name. Research
only: no runs on metatron (it is serving production; the benches exist and rockem.lead runs them on request).

**Read first, in this order:** `docs/metatron-cpu.md` §3 (the L3 model as we measured it, with its summary at the top),
`docs/queue/099-expert-run-as-a-system.md` §2–§3 and Appendix C (the measurements, the rules, the tables),
`bringup/expert-cache/briefs/predictor-analog-brief.md` only for the box's shape. The hardware: 2 × EPYC 7R13, 6 CCDs a
socket, 8 cores a CCD sharing a 32 MB 16-way L3 (a victim cache of the eight 512 KB L2s), SMT siblings idle, kernel
`isolcpus`/`nohz_full` on the pool cores, THP `madvise`, hugetlb 2 MiB pages for the experts.

## The two unanswered behaviours (our measured signatures — match these, not the words)

### A. The replacement drops freshly inserted lines well below capacity ("the unexpected eviction")
- A set of lines touched once (one load a line, so they enter L3 as clean L2 victims) and read back by the same cores:
  8 cores of one CCD touching 12 MB in total lose 7 % of it to DRAM at the read-back, 15 MB → 16 %, 22 MB → 33 %, 24 MB
  → 41 % — in a 32 MB L3. One core touching 12 MB alone loses 0.1 %.
- The loss is by age within the once-referenced lines: the oldest tenth of the set loses ~19 %, the newest ~4 %.
- One extra L3 hit protects a line: touched twice, 15 MB → 2 %; three times 0.2 %. A set read every round keeps 100 %.
- What the fresh lines lose to is the *previous* round's read lines (protected, useless now): flushing them first lets
  the toucher keep 100 % of 12 MB and 98 % of 24 MB.
- Expiry is bytes inserted after the line, not time (2 ms idle changes nothing; 50 ms changes little).
- A line stored to (dirty victim) is not kept at all (stored-back lines read from DRAM: 7 % / 2 % kept).
- A `prefetcht0` touch keeps ~7–9 %; loads keep ~90 %.
- A non-temporal read (`prefetchnta` ahead of the loads) does not protect the line; non-temporal reads of *other* lines
  make the fresh lines' loss worse (the cold lines were the cheapest victims).
- Toucher ≠ reader: a CCD-mate reading a line still in the toucher's L2 refetches it from DRAM for local-homed memory
  but is forwarded it for remote-homed memory.
What we want to know: AMD's actual L3 replacement policy on Zen 3 (RRIP-like insertion at distant re-reference? a
reuse bit? dead-block prediction?), whether "first-insertion lines are replaced first" is documented or reverse-engineered
anywhere, whether anyone has measured the retained fraction of a once-touched set vs its size on a Zen L3, and whether
the dirty-victim and local-vs-remote forwarding behaviours are known.

### B. A core's L3 read rate decays when its L3 hits stop, and recovers slowly under load (the thing we cannot name)
- Eight cores of a CCD reading their own L3-resident 2 MB sets (gemv, 128 KB items): 4.35 µs an item in steady state.
  After a gap without L3 hits: 100 µs → 5.2, 500 µs → 6.3, 1 ms → 5.9 (primed), 3 ms → 7.5, 10 ms → 17.4 (4×).
- Uniform over the run that follows (first third = last third); the core clock is *higher* after the gap.
- How the gap is spent does not matter: `pause`, a register-only loop, `mwaitx` (C1-ish), L2-resident reads, DRAM
  streaming with `prefetchnta` — all decay. Only L3 hits hold it: ~5 GB/s of hits a core (256 KB every 50 µs) suffices;
  1.3 GB/s partly; 0.3 GB/s not. Hits on any lines, not the set's.
- Per core: one core's hits do not hold its seven mates; another CCD's traffic does not hold this one.
- DRAM-fed reads are unaffected (cold items 24.3 vs 24.4 µs). Fill counters show the same sources and counts in both
  states (~2,190 L2-prefetch hits in L3, ~20 demand fills an item).
- Recovery: one core alone recovers in one 2 MB round (~70 µs); eight cores sharing the L3 take ~130 rounds (~10 ms,
  ~260 MB of hits a core): 263 → 125 → 93 → 81 → 72 µs a 16-item round from cold. So it costs only when several cores
  load the L3, and a per-core structure does not fit the recovery — a shared L3 structure whose benefit many cores'
  hits re-earn slowly, and which idle ages away, fits everything.
- Not the clock (APERF/MPERF), not power (3.45 GHz throughout; cap effects separately known), not the kernel (identical
  code), not memory placement (pages, strides, allocation), not false sharing, not SMT state.
What we want to know: has anyone measured L3 bandwidth or hit latency on Zen as a function of recent activity? Candidate
mechanisms to look for in documentation, patents, errata, forum reports and papers: the L2 prefetcher's adaptive
throttle/depth (training on L3-hit latency or bandwidth), per-core request credits toward the L3/CCM, L3 "way
power-down"/data-retention low-power states, DF C-states or P-states and anything that scales with *L3 hit* traffic,
RRIP/aging logic that runs on idle cycles, replacement-state convergence under multi-core sharing. The one-core-vs-eight
recovery asymmetry is the sharpest discriminator we have.

## Tracks to fan out (one sub-agent each, then a synthesis)
1. AMD's own documents: PPR for Family 19h Model 01h (Milan), the Software Optimization Guide for Zen 3 (and Zen 2/4 for
   deltas), the "AMD64 Architecture Programmer's Manual" cache sections, the L3 "shadow tags" and "mostly exclusive"
   statements, performance-counter definitions for L3 (events 0x43, 0x70–0x72 we use; any L3-side counters), power
   management (CC6/PC6, DF C-states, "L3 power gating", "DLWM"), and anything on replacement.
2. Patents: AMD's cache replacement, insertion, reuse-prediction, dead-block, and "victim cache with shadow tags"
   patents (2015–2023) — look for insertion-at-low-priority schemes and anything with an idle-time aging clock or
   per-core L3 bandwidth allocation.
3. Reverse-engineering and microbenchmarks: chipsandcheese (Zen 3 L3 deep dives, bandwidth vs threads), Agner Fog,
   uops.info, 7-cpu.com, the "Zen 3 L3 replacement policy" discussions, Travis Downs / Daniel Lemire posts, "L3 cache
   eviction policy reverse engineering" papers (Abel & Reineke's nanoBench and cache-replacement work covers Intel; find
   any AMD coverage), RWTH/ETH memory-hierarchy papers on EPYC, the `cachefuzz`/`cachequery` literature.
4. Field reports: HPC and database engineers on Milan (Dell/HPE/Lenovo performance briefs, Oracle/SAP/ClickHouse/DuckDB
   blog posts), game-engine and HFT engineers on Ryzen 5000, kernel mailing lists (Linux `perf`, scheduler, `nohz_full`
   and C-state interactions), AMD community forums, Reddit r/hardware threads with measurements — anything describing
   "L3 slow after idle", "cache warm-up takes milliseconds", "bandwidth ramps", "first touch not retained".
5. Prefetcher behaviour: documented and measured behaviour of Zen 3's L1/L2 prefetchers (stream, stride, region),
   throttling under contention, training windows, page-boundary rules, and any adaptive depth that depends on observed
   latency — the mechanism that would make B a prefetcher effect.

## Deliverable
`docs/l3-literature.md` in the repo (not /tmp): for each of A and B, what is *quantified* in the sources (with the
measurement, the hardware, the date and the link), what merely asserts, how each finding maps onto our signatures (fits /
contradicts / silent), the candidate mechanisms ranked by the evidence, and the discriminating experiments the sources
suggest that we have not run (we can run anything on the box: metatron-cpu §7 has the method; `bench-order --gap-mode`
and `bench-run --simple` are the instruments). Quote sparingly and exactly; mark inference as inference. Send rockem.lead
a message when the synthesis is in the tree; partial notes are welcome earlier if a track finds a direct hit on B.
