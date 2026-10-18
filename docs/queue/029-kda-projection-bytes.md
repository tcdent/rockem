# 029 — The KDA projections' bytes (DFloat11 or FP8)

**Status:** Decision (tcdent) for FP8; Open for DFloat11
**Area:** decode
**Gain:** est. — DFloat11 ~30 % of 4.6 ms a token a card at bf16 (if its decode overlaps the stream); FP8 ~50 %

## Proposal
- DFloat11: lossless Huffman coding of the bf16 exponent, ~11 bits a weight; keeps "dense native".
- FP8: half the bytes; a deviation from the reference, so an experiment.

## Source
DFloat11 (arXiv 2504.11651); SGLang #44629 (KDA projections as one W8A16 launch); vLLM's FP8 rows.

## History
- 2026-10-17 (`bdc57ec`): DFloat11 scoped; not built.

## Next
DFloat11 kernel prototype measured against the bf16 stream; FP8 only if tcdent opens it.
