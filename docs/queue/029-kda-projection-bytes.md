# 029 — The KDA projections' bytes (DFloat11 or FP8)

**Status:** Decision (tcdent) for FP8; Open for DFloat11. **Exp4 built and bit-checked at the kernel (2026-10-25),
not in the node**: lossless, 77.3 % of the bytes, the rows kernels' outputs equal the bf16 kernels' bit for bit on the
bundle's own KDA matrices at every width, 1.08–1.17× at the verify widths.
**Area:** decode
**Gain:** est. — DFloat11 ~30 % of 4.6 ms a token a card at bf16 (if its decode overlaps the stream); FP8 ~50 %

## Proposal
- DFloat11: lossless Huffman coding of the bf16 exponent, ~11 bits a weight; keeps "dense native".
- FP8: half the bytes; a deviation from the reference, so an experiment.

## Source
DFloat11 (arXiv 2504.11651); SGLang #44629 (KDA projections as one W8A16 launch); vLLM's FP8 rows.

## Exp4 (2026-10-25): a fixed-length lossless form, measured
> tcdent, 2026-10-25: "there's a way that we can implement this without any effect on the quality of the generation";
> then "let's see if we can do the bit check on generation so that we can understand if there is first of all any loss
> that we're experiencing and second of all any performance gain."

**The weights' exponents** (`bench-exponents`, all 34 KDA layers, 9.36 GB of bf16): 29–125 distinct exponents a
matrix, entropy 2.56–2.70 bits; a Huffman code of the exponent (DFloat11) would store 10.58 bits a weight (66.1 %).
99.99 % of weights lie within 15 binades of their group-of-128's largest exponent; exponent-0 weights are 14,716 of
4.68 billion.

**The format** (`common::exp4`): per group of 128 a base exponent (the largest less 15, or 0) and per weight its sign
and mantissa byte and a 4-bit code (exponent = base + code), no code set aside for zeros; a group holding an exponent
below its base (0.68 % of groups) is stored whole as bf16. A lane's eight weights are one 12-byte chunk (eight bytes,
four bytes of nibbles). **Round trip: 306 of 306 matrices identical to the bundle's bf16; 77.3 % of the bytes.**

**The kernels** (`rows_h_e4`, `rows_h1_e4`, `rows_d2_e4`, `rows_part_e4`, `rows_part1_e4`, kda.rs): the bf16 kernels'
arguments, grids and LDS over the exp4 buffer, decode and verify widths only (`rows_fmt_few`); a lane rebuilds its
weights' bf16 in registers, two at a time (one byte permute, one add for both exponents), then the bf16 step's products
in its order. `bench-exp4` on three layers' part-0 matrices (the node's slicing), chains of 43 rotating over the layers so
the weights stream from VRAM, card 0 beside the idle node (µs a launch, bf16 / exp4):

| matrix | 1 token | 2 tokens | 4 tokens | bits |
|---|---|---|---|---|
| `sq` [12608][4096] | 142.6 / 128.2 (1.11×) | 152.6 / 141.0 (1.08×) | 154.0 / 142.5 (1.08×) | equal |
| `o` [4096][4096] | cache-resident at 1 token (1.35×) | 48.4 / 41.5 (1.17×) | 48.7 / 41.8 (1.17×) | equal |
| `fg` [8192][128] | 11.1 / 9.6 (1.15×) | 12.9 / 12.9 (1.00×) | 13.0 / 13.2 (0.99×) | equal |

The path there, each step measured: the head loaded before the bytes (two trips to VRAM a step) 0.70×; the loads
issued together 0.94×; a branch-free decode two weights a register 0.97×; the bytes and codes interleaved into one
96-bit load a lane 1.08–1.13×. The heads are still vector loads (the row pointer is not provably wave-uniform; a
`readfirstlane` did not make them scalar); the ceiling from the bytes alone is 1.29×.

**What is not done:** the node. Prefill steps use the WMMA kernels on bf16, so the node would hold both forms (+3.65 GB
of VRAM a card, out of the residency's budget) unless the WMMA rows kernels read exp4 too; and the generation check on
a boot (`bringup/box/gen_ref.sh`: the bf16 reference `bf16-aff81` is recorded and reproduces on the same boot).

## History
- 2026-10-17 (`bdc57ec`): DFloat11 scoped; not built.

## Next
DFloat11 kernel prototype measured against the bf16 stream; FP8 only if tcdent opens it.
