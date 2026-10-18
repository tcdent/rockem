# 009 — WMMA rows kernels for prefill

**Status:** Resolved
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

## Next
Efficiency: ~10 TFLOPS is ~12 % of peak. tinygrad's tiling (four output tiles a wave, four or
eight waves sharing activations through LDS, 2×2 waves with the weight tile dequantized in LDS)
is the reference for the next round.
