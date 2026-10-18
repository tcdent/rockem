# 007 — Rows kernels: R rows a wave on the multi-token width

**Status:** Resolved
**Area:** prefill
**Gain:** measured — a prompt token 7.8 → 6.2 ms (kda card 3.8 → 2.8, MLA projections 1.2 → 0.9)

## Proposal
A wave owned one row and every FMA read an activation from LDS: LDS-bound at ~0.5 TFLOPS over
32 tokens. Give a wave R rows so each staged activation read feeds R FMAs, keeping each output's
accumulation order (bit-identical).

## Source
The prefill timeline, 2026-10-17 07:15 (a rows_h launch over 32 tokens 830 µs against 125 for one).

## History
- 2026-10-17 09:15 (`de37280`): R = 4 (`common::gemv::ROWS`), 2 for the FP8 gate/up (four spilled).
  Superseded for prefill steps by the WMMA pair (009).

## Next
None.
