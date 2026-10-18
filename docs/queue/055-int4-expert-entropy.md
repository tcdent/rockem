# 055 — Entropy of the INT4 expert nibbles

**Status:** Deferred — likely a dead end; a cheap check
**Area:** decode
**Gain:** est. ≤ ~10 % of expert bytes, decode-rate-bound on the CPU

## Proposal
Entropy-code the expert weights (Huffman 1952; ANS, Duda 2013) if the nibble histogram is skewed.

## Source
The old-results research, 2026-10-17 (est. ~3.6 bits of entropy for group-absmax RTN; AVX2 rANS
decodes a few G symbols/s a core, short of the DRAM's ~650 G nibbles/s).

## History
- 2026-10-17: listed with the caveat.

## Next
Measure the bundle's nibble histogram entropy; drop if near 4 bits.
