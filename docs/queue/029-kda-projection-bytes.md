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

## Why exp4 is not faster in the node (2026-10-25)

The rows kernels' 1.08–1.17× in the isolated bench did not reach the node (aff83 verify steps: `sq` 159.9 → 150.9 µs,
`o` 59.9 → 60.8, from the card lanes). `bench-exp4 --node` runs each layer's `sq`, `fg`, `o` back to back as the node
does, every launch timed from its own stamps; it reproduces the node (`o` 1.02×). The isolated `o` gain was the
Infinity Cache: three layers of `o` alone (100 MB) partly fit its 80 MB, and in the node `o` follows an 80–103 MB `sq`.

Three limiters, in the order found:

1. **The compiler sank the chunk loads into the decode branch.** The chunk is used only when the group is not stored
   whole, so LLVM moved its load under the head's test and waited for it at once (`s_waitcnt vmcnt(0)` a row a step):
   a round trip a row. Decoding on every lane and selecting the whole group's bf16 (loaded on its rare branch only)
   keeps the loads together. With two steps' loads issued ahead (`EXP4_AHEAD`), node order, bits equal:

   | node order, × against bf16 | 1 token | 2–4 tokens |
   |---|---|---|
   | aff83 (shipped) | sq 1.11, o 1.01 | sq 1.08–1.09, o 1.02 |
   | select + two ahead | sq 1.21, fg 1.21, o 1.20 | sq 1.08–1.10, fg 1.06–1.08, o 1.04–1.05 |
   | select, one ahead | sq 1.22, o 1.10 | sq 1.13, o 1.03 (the run hung the card at 4 tokens, below) |

2. **`sq` at verify widths: workgroup rounds.** Wave stamps (a throwaway stats build; each wave's start and end):
   an exp4 wave takes 46 µs against bf16's 66, but at 152 registers the card holds two workgroups a CU, so the 394
   workgroups run in three rounds, the last a third full; bf16 (140 registers) holds two and a half and runs in two.
   The launch is 133 µs against 148. Fewer registers (≤ 128: three a CU) or a grid sized to the card's slots is the lever.

3. **`o` at verify widths: rounds in lockstep.** `o` is 128 workgroups: every wave starts at once and they move
   through each round together — stage the activations, barrier, load the weights, compute, barrier — so DRAM idles
   while everyone stages and computes, and fewer bytes shorten only the loading. Waves take ~40 µs either way. The
   lever: fetch the next round's weights before the barrier (`workgroup_sync` waits on LDS only, so loads can stay in
   flight across it).

Neither lever is built. The select form is in the tree, not deployed (verify steps ~0.2 ms a step at 18k by the
node-order layer times, 216 → 211 µs a layer a card).

**Sizing (2026-10-25, tcdent: "we need to reevaluate the register and workgroup and wave sizing … optimize the places
that interact with it to utilize the resources as thoroughly as possible").** A sweep (`bench-exp4 --sweep`, since
removed with its variants; results in `bringup/box/logs/exp4-sweep-1.txt`) over rows a wave (1, 2, 4) and steps loaded
ahead at a few tokens (1, 2, 4) and at one (2, 4, 8), each variant in place of one matrix's kernel in node order, bits
checked against the shipped kernel. One row a wave lost everywhere at 4 tokens (0.72–0.93×). Two rows a wave won for all
three matrices, and eight steps ahead at one token. Then `rows_h_e4` came out at 129 registers, one past the three
workgroups a CU its 20.8 KB of LDS allows: the row pointers marked wave-uniform (scalar registers) took it to 125.

| kernel | rows a wave | ahead (few / one) | registers | workgroups a CU |
|---|---|---|---|---|
| `rows_h_e4` | 2 | 2 / 8 | 125 | 3 |
| `rows_part_e4` | 2 | 4 / 8 | 164 | 2 |
| `rows_d2_e4` | 2 | 2 / 8 | 53 | 3 (LDS) |

The narrow one-row twins (`rows_h1_e4`, `rows_part1_e4`) are gone: at two rows a wave every grid is past the CUs.
Node order, idle node, µs a layer (sq + fg + o):

| | 4 tokens | × bf16 | 1 token | × bf16 |
|---|---|---|---|---|
| bf16 | 222.2 | 1.00 | 203.6 | 1.00 |
| aff83 (shipped exp4) | 216.3 | 1.03 | 208.0 | 0.98 |
| sized exp4 | 193.6 | 1.15 | 165.7 | 1.23 |

At 4 tokens `sq` streams 633 GB/s and `o` 575 against ~730 at one token: what is left is the lockstep rounds (stage,
barrier, load, compute, barrier) — the structure, not the sizing.

**No staging (2026-10-25).** tcdent: "i'm curious if some of those lockstep rounds could be fused, which kind of
supports your idea of dropping the staging entirely in that we just move to the next step when we can." Past one token
the exp4 kernels stage nothing (`exp4_rows_direct`): each wave reads its lanes' activations from L2 (the launch before
wrote them) and runs its rows' steps with no barrier. Bits: the products are the staged pass's `wl · x`, and the norm's
sum of squares is replayed exactly (lane `l` holds the staged threads `8l..8l+8`'s sums; the 32-thread butterfly as
masks 1, 2, 4 inside a lane and 8, 16 across lanes, then the eight waves in order). Each step runs at its own width:
half of the verify steps are two tokens and a quarter three, and the staged pass computed all of them at four.
Node order, idle node, µs a layer, bits equal at every width:

| | 1 token | 2 tokens | 3 tokens | 4 tokens |
|---|---|---|---|---|
| bf16 | 204.1 | 220.6 | 221.7 | 223.3 |
| aff83 (shipped exp4) | 208.0 | 208.2 | 210.1 | 211.1 |
| sized, staged | 165.7 | — | — | 193.6 |
| no staging, own width | 166.4 | 176.4 | 182.5 | 189.8 |

Over the verify-width mix (2: 50 %, 3: 25 %, 4: 25 %) a layer is ~181 µs a card against ~210 on aff83: ~1 ms a verify
step. `sq` at two tokens streams 710 GB/s, bf16's rate per byte. What was tried and kept out, all measured in node
order: a pipeline loading the next steps' weights before each step's arithmetic (`o` unchanged, `sq` 125–130 µs: its
registers); waves a workgroup 8 / 4 / 2 / 1 (no difference: the grid's quantization was not the limit); four rows a wave
(4 tokens 205.8). `o` stays at ~550–580 GB/s at every width and shape: one round of waves, 26 MB — the launch's own ramp,
not its loop. Gotcha found on the way: a pointer rebuilt from `readfirstlane` loses the global address space, so its
loads compile as flat loads, which also count on the scalar counter; the chunks' loads waited on every head's wait.

**A card hang (2026-10-25):** the one-ahead build hung card 0 in `bench-exp4 --node` at 4 tokens (exp4 sequence; the
same ELF had passed 1–3 tokens and the bit checks at every width). The node on that card panicked and the box reset.
Not reproduced or explained; the two-ahead build ran every width twice without it. Benches beside the serving node
share its card: a hang there takes production down.

## History
- 2026-10-25 (aff83): exp4 deployed — the node holds only exp4, generation bit-identical, ~1.07 GB a card freed.
- 2026-10-17 (`bdc57ec`): DFloat11 scoped; not built.

## Next
The boot encode across threads (above). DFloat11 is not needed: exp4 is lossless at 77 % of the bytes with faster decode kernels. FP8 only if tcdent opens it.
