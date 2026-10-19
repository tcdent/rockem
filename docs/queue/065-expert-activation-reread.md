# 065 — The expert kernel's activation re-read a part

**Status:** Quantified, on the shelf — ≤ 0.13 ms a prompt token (10 % of the host's expert items' time)
**Area:** prefill (the CPU expert phase)
**Gain:** measured bound — the phase's items run 56–61 GMAC/s a core against the kernel's 65–72 with one activation set (2026-10-19, 92 cores)

## Proposal
Each of an expert's 64 gate/up parts (32 rows, `expert::SUB` = 2 a 64-row item) runs on a different core and
pulls the expert's tokens' activations (4 KB a token) out of the core complex's L3 into its own L2 again. Fewer,
larger parts (`SUB` 2 → 1: 64-row parts, no bundle change, half the re-reads, ~0.06 ms est.), or a
token-block-stationary order that keeps a block's activations in L2 across an expert's parts on one core, or a
wider `GU_CHUNK` (the bundle's item layout: a repack of the 162 GB expert file).

## Source
021's spread-activation slab bench (`gemm-rate3`): 58.6 / 58.3 / 65.9 GMAC/s a core at 16 / 32 / 64 tokens on 92
cores with each slab's tokens drawn from a 4 MB pool, against 65.7 / — / 71.8 with one set reused; one core alone
loses 3–4 %.

## History
- 2026-10-19: split out of 021 at close.

## Next
`SUB` 1 is a one-constant measure (a prefill step at 18k, the `expert_items` rows by bucket).
