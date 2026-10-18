# 009 — WMMA rows kernels for prefill

**Status:** Resolved; the efficiency round measured 2026-10-18 (small ceiling: the pair is ~8 % of the step's card time)
**Area:** prefill
**Gain:** measured — 6.2 → 5.5 ms a prompt token (short), 8.2 → 7.0 at 18k; the 18k prompt 153 → 129 s

## Proposal
Prefill's own rows kernels on `v_wmma_f32_16x16x16_bf16`: activations rounded to bf16 as staged
(the reference's activation dtype), FP8 weights as fp8 × block scale → bf16 (the reference's
dequantization), f32 accumulation.

## Source
tinygrad PR #18863's WMMA prefill; our `bench-wmma` (bf16 WMMA 85 TFLOP/s on a 7900 XT).
tcdent's license for prefill kernel pairs (2026-10-17).

## History
- 2026-10-18 07:40 (`fe11f0d`): `rows_wmma`: a workgroup 128 rows × 128 tokens, a wave 16 rows
  with eight token tiles; the fragment layout measured first; the staged rows padded off the
  128-byte LDS bank width (a 16-way conflict made it 3.5× slower). KDA sq rows 2.8 ms a launch at
  512, ~10 TFLOPS.
- 2026-10-18 07:51 (`72df137`): the FP8 twins (`rows_wmma_with`).
- Gate: step vs token path top-1 0.9954, mean KL 2.3e-3; the long SCORE NLL 0.0446 → 0.0407.
- 2026-10-18 09:00 (`971f5df`, item 030's bf16 state): at 18k positions 7.0 → 6.7–6.8 ms a prompt token; the card wait that grows with position (attend + scorer) is 2.3 → 3.7 ms across the 18k prompt, now the larger share of a long step's growth.

- 2026-10-18 10:10, the efficiency round measured (`bench-rows-w`, the kernels alone at the served
  shapes, 512 tokens, K = 4096): `rows_part_w` 26.5 TFLOP/s at m = 12608 (2.0 ms), 22.4 at m = 4096;
  `rows_h_w` (the norm folded in) 18.1 and 14.4 — 1.5× slower than the plain kernel, because every
  row block recomputes the 128 tokens' rms from a full read of their activations (99 row blocks at
  m = 12608 read the same 2 MB each). A wave owning 32 rows × 64 tokens (each LDS A tile feeding two
  WMMAs, half the LDS traffic) measured 4 % *slower*: LDS traffic is not the limiter; reverted. The
  pair's place in the 512-step: ~2.9 + 2.0 ms × 34 KDA layers ≈ 170 ms of the step's ~2,150 ms of
  card time (0.3 of 6.7 ms a prompt token) — the KDA scan (023) holds the rest.

## Next
Small: accumulate the rms in the staging loop (the sum of squares falls out of the values staged;
apply 1/rms at the end) to bring `rows_h_w` to `rows_part_w`'s rate — ~30 ms a 512-step, 0.06 ms a
prompt token. The card lap at 512 is the scan's: 023.
