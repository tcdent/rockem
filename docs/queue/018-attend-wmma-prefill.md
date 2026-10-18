# 018 — Sparse attend on WMMA for prefill (heads as rows)

**Status:** Deployed 2026-10-18 13:06 — prefill at 18k positions 6.7 → 5.0–5.4 ms a prompt token, the 18k prompt 127 → 100 s; SCORE NLL +0.003 (the text's noise floor); the step gate at long positions pending a lab boot
**Area:** prefill, long context
**Gain:** measured ceiling ~1.7 ms a prompt token at any context past 2,050 positions (the kernel runs at ~1.8 TFLOP/s; WMMA takes the same work ~10× down)

## Proposal
All 64 heads share a token's latents and selection, so scoring is
`[64 heads × 512] · [selected × 512]ᵀ` — WMMA-shaped even for one query — followed by softmax and
the latent mix. Each latent loaded once for all heads. A prefill kernel pair.

## Source
DeepSeek FlashMLA's decode on tensor cores; tinygrad's `flash_attention_mla_sparse`; the attend
experiments 014–016 (every scalar shape tried; this one untried).

## History
- 2026-10-18 (`971f5df`): measured — the card wait that grows with position (attend and scorer)
  grows ~0.08 ms a prompt token per 1k positions: ~80 ms a prompt token at 1M as the kernels stand.
- 2026-10-17: proposed; not built.

## History (cont.)
- 2026-10-18 11:15 (`bench-mla-attend`): `mla_attend` over a 512-token step 151 µs a token a layer
  at 2k positions, 160 at 18k, 161 at 65k, 155 at 262k — constant once 2,048 positions are selected.
  Per token a layer the work is 64 heads × 2,048 positions × 512 for the scores and again for the
  mix, 0.27 GFLOP, so the kernel runs at ~1.8 TFLOP/s against the rows pair's 18–26 on WMMA.
  Scoring is `[64 × 512] · [512 × 2048]` a token: 4 head tiles × 128 position tiles × 32 k-steps,
  the latents staged once for all heads. The scalar kernel at four heads a workgroup (016) gains 6 %,
  so the latent reads are not where the time goes; the per-thread dots are.

- 2026-10-18 13:30, built (`mla_qabs_w` → `mla_attend_w` → `mla_uv_w`; `bench-mla-attend` diffs the
  heads against the scalar kernel): at a 512-step from 2k positions on, **36 µs a token a layer**
  (q_abs 0.6, attend 34.5, W_uv 0.9) against the scalar's 151 — 4.2×; |Δ|max 0.3–0.4 on values of ~35
  (bf16 q_abs and probabilities, the reference's dtypes). The kernel: a workgroup per (token, 16
  heads), eight waves; a tile of 16 positions an iteration, the latents staged into LDS double-buffered
  with the next tile's rows loaded into registers under the current work; each wave 64 channels of the
  scores' dot (4 WMMAs, partials summed through LDS) and 64 channels of the mix; the running softmax in
  registers with the probabilities as the A fragment. What it took, in order of what each was worth:
  the eight-way redundant scores (every wave the full dot) were 57 % of the time; an 8-way LDS bank
  conflict on the reduction reads (a 64-byte lane stride) was the next half; W_uv in every workgroup
  (512 dependent row reads a wave) came out into its own kernel; a conditional tuple of arrays and a
  prefetch tuple live across a barrier (an `asm!` memory clobber) each went to scratch until held as
  scalars; an out-of-bounds const array length compiled to an abort that deleted the barriers after it
  (a deadlock, the card reset, the node lost). Expected served: ~1.3 ms a prompt token at any context
  past 2,050 positions.

- 2026-10-18 13:06, deployed (`971f5df`…): the node's `mla_attend` is q_abs → attend → W_uv by the
  step. Served: the 512-step at 18k positions **6.7–6.8 → 5.0–5.4 ms a prompt token**, and the card
  wait no longer grows with position (1.9 → 2.1 ms across the prompt against 2.3 → 3.7); the 18k cold
  prompt **127 → 100 s** (no hardware errors in either boot). Long SCORE: NLL 0.0407 → 0.0437, mean |Δ
  logprob| 9.6e-3, 99 % moved, max 4.9 at a near-tie — the same size as the bf16-state step and the
  text's ±0.003 floor. Decode acceptance on the short text 0.85 (0.88).

## Next
The step gate (top-1/KL) at long positions needs the gate binary beside the node: a `rockem.resident=0`
lab boot, batched with the next lab run. Then: `mla_attend` → the three launches by the step (q_abs and W_uv twins by `SPEC_TOKENS`, the
chunked form with its merge below it); the gate (step gate top-1/KL at long positions, long SCORE);
deploy; the scalar `mla_attend` kernel stays as the bench's reference. On the shelf after: the attend
at 34 µs a token is ~5 µs a tile where its WMMA, LDS and VALU add to ~1.5 — two tiles an iteration
and the exp/bpermute chains are the next round.
