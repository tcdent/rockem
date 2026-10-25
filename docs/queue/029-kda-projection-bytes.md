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

**The node (aff83, 2026-10-25): exp4 only.** The cards hold the KDA projections as exp4 and nothing else. Decode and
verify steps run the exp4 rows kernels. A prefill or SCORE step unpacks each matrix (`exp4_unpack`) into one bf16 scratch
a card (the largest part matrix, 103 MB) and runs the bf16 WMMA kernels on it: an exp4 WMMA kernel measured 0.55× at
512 tokens (the decode repeated for every token block), the unpack-then-WMMA path 0.92–0.94× at 1,024 tokens
(`sq` unpack 293 µs, `o` 66 µs).

> yeah let's go ahead and integrate it with the wmmma kernels because freeing up some of that vram is going to be
> essential and i don't think that adding any additional vram is something we can do at this point anyway — tcdent

Measured on aff83 against aff81 (bf16):

| check | aff81 bf16 | aff83 exp4 |
|---|---|---|
| long SCORE (3,632 ids) | reference | bit-identical, three runs |
| `gen_ref.sh` 18k cold (24 tokens) / decode (126 tokens) | `bf16-aff81` | identical, both |
| the cards' state (both cards) | 20.57 GB | 18.42 GB |
| routed experts resident | 4 a layer (201 on card 1) | 6 a layer (275 / 266) |
| verify step at 18k, ms a token: kda / total | 7.99 / 22.7 | 7.89 / 22.4 |
| prefill at 18k, ms a prompt token | 1.97 / 2.00 | 1.99 / 2.03 |
| SCORE, warm, ms a token | 2.23 | 2.22 |
| the cards' boot (`cards` row) | 10 s | 36.6 s |

The freed VRAM goes to the expert slots (the slots fill the budget), so a step's cost moves from the CPU experts to the
cards: on SCORE `experts` 1.24 → 1.10 and `kda` 0.25 → 0.44, the total the same. Prefill is +0.02–0.03 ms a prompt
token (the unpack, at 1,024-wide steps). The boot's encode is single-threaded on the host: +26.6 s a boot.
`Quantified, on the shelf — 26 s a boot`: the encode split over the loader's threads.

## History
- 2026-10-25 (aff83): exp4 deployed — the node holds only exp4, generation bit-identical, ~1.07 GB a card freed.
- 2026-10-17 (`bdc57ec`): DFloat11 scoped; not built.

## Next
The boot encode across threads (above). DFloat11 is not needed: exp4 is lossless at 77 % of the bytes with faster decode kernels. FP8 only if tcdent opens it.
