# 084 — The one-token gemv from L3 under full load

**Status:** Open — measured, not explained
**Area:** decode
**Gain:** unknown; bounds what an L3-served decode expert can be worth (expert-cache-architecture)

## Proposal
On 92 cores at once, decode's one-token `gemv` over L3-resident weights runs ~9 GB/s a core (~70 GB/s a CCD; warm
861 GB/s summed, `bench-reuse`, 2026-10-20), while a bare AVX2 load loop over 2 MB a core pulls ~55 GB/s a core from L3
on 23 cores (`bench-power --mode l3`). One core alone runs the gemv at ~26 GB/s warm. Find what limits the gemv from
L3 under load (its access pattern, the per-socket L3 clock limit, L2 prefetcher behaviour, the kernel's instruction
mix) — the fill counters and llvm-mca apply as for the prefill gemm.

## Source
`docs/metatron-cpu.md` §3–§5; `bringup/bench/src/reuse.rs`, `power.rs`.

## Next
`bench-reuse --mode warm` at 1, 8 (one CCD), 23, 46, 92 cores with clocks (APERF) beside the rate; then the gemv's
disassembly through llvm-mca.
