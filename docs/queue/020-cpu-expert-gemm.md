# 020 — A multi-token INT4×INT8 expert kernel on the CPU

**Status:** Resolved
**Area:** prefill
**Gain:** measured — experts 2.4 → 2.2 ms a prompt token at 512; bit-identical to `gemv`

## Proposal
`run_batch` grouped tokens by expert but ran the one-token `gemv` per (token, slot). A gemm that
decodes a weight block once for several tokens.

## Source
ik_llama.cpp's `mul_mat_q4_0_r8_q8_2_avx2` (rows in lanes, nibbles unpacked once for up to 8
tokens), from the 2026-10-17 kernel research; our q4 convention (nibble = w + 8, `gsum`) matched.

## History
- 2026-10-17 06:45: a first version in gemv's tile order measured slower (2.4 → 3.1).
- 2026-10-17 (`5b713ab`): `q4::gemm` in ik's shape — an 8×8 dword transpose a block, no
  horizontal sums, four tokens a pass.

## Next
021 (what bounds the phase now).
