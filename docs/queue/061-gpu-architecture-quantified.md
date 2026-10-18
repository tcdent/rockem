# 061 — The cards' architecture, quantified

**Status:** Open — topology read and probes first
**Area:** decode, prefill (every card kernel)
**Gain:** unknown; the CPU side's equivalent work (topology, per-node rates, placement) gave the
expert phase most of its gains

## Proposal
tcdent, 2026-10-18: "we've quantified the NUMA regions and the dies available on our CPUs, but we
haven't necessarily mapped out the direct architecture of our GPUs … we should probably quantify
the GPU architecture in the same way so that we understand exactly how we're using the available
physical hardware to the same extent."

The 7900 XT (Navi 31, cut down) as we believe it is built — every line to verify on the box:

| Level | Believed | Our kernels |
|---|---|---|
| Dies | 1 GCD + 5 MCDs | cache and memory in 5 address-interleaved slices |
| Shader engines | 6 SEs, 84 CUs = 42 WGPs | how a small grid spreads (`kda_step`: 32 workgroups) |
| WGP | 2 CUs sharing LDS (128 KB WGP mode / 64 KB CU mode) | our LDS mode and residency a WGP |
| CU | 2 SIMD32, dual-issue, ≤16 waves a SIMD | waves from VGPRs (`kda_step` 153, `ffn_gu` 175, `mla_qb` 181) |
| L0 / L1 / L2 | 32 KB a CU / 256 KB a shader array / 6 MB | GEMV activations, WMMA operand tiles |
| Infinity Cache (MALL) | 80 MB (16 MB per MCD) | a KDA layer's weights ~86 MB a card — just over it; `bench-hc`'s 1.7 TB/s was MALL |
| VRAM | 320-bit GDDR6, 20 Gbps, ~800 GB/s | measured 640–770 GB/s |
| Front end | CP, AQL dispatch | measured: ~10 µs launch floor, ~5 µs gaps, device barrier 6–8 µs, sync 14–16 µs |
| Link | PCIe 4.0 x16, socket 0 | BAR rates, peer access (001) |

## Source
Our CPU work (the 7R13 topology, per-node rates, `layout::placement_from`); the occupancy research
of 060 (Fleet: fused register footprint → one wave a SIMD).

## History
- 2026-10-18: proposed. Known so far: the WMMA rates (iu8 87 TOP/s, bf16 85 TFLOP/s), the VRAM
  stream, the launch/sync/barrier floors, the MALL showing in benches. The boot log reads
  "4 cards held at their highest clocks" while KFD lists two gfx1100 nodes — check what the four are.

## Next
1. Read the driver's topology: `/sys/class/kfd/kfd/topology/nodes/N/properties` and `caches/` on
   each card.
2. Probe mode (`rockem.probe=1`) extensions: bandwidth and latency at working sets stepping
   through L0 → L1 → L2 → MALL → VRAM; the MALL's interleave; the VGPR allocation granule and LDS
   residency per WGP; sustained clocks under a WMMA load.
3. Map the kernels: each phase's binding level; whether decode's weights stream from VRAM or hit
   the MALL (the KDA layer's 86 MB against 80); how small grids spread over shader engines. Feeds 060.
