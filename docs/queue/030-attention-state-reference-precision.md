# 030 — The attention state at the reference's precision

**Status:** Resolved (decided; both phases built and deployed)
**Area:** long context, decode, prefill
**Gain:** capacity — 1M latents 12.9 GB at bf16 vs 25.8 at f32 (f32 does not fit); half the latent and pool bytes read

## Proposal
tcdent, 2026-10-17:

> Lets follow the reference and consider any deviations to be additional experiments.
>
> bf16 latents, f32 kda, bf16 indexer scored in 32

The reference is `modeling_glm5_next.py` (transformers v5) at the config's default dtype: the
`kv_a_layernorm` latent stored bf16 (l.1189–1196); the KDA state forced to f32 (l.762); the
indexer's per-token keys bf16 (l.846), pools formed in bf16 with the softmax weights cast to the
keys' dtype (l.856–866), scores `q.float() @ pool_keys.float()` then ReLU and the f32 head weights
(l.999–1003). The oracle and goldens stay f32; the change is gated as a numeric distance.

## Source
transformers v5.0.0 release notes (`dtype="auto"` default; `modeling_utils.py` ~l.4106, 829);
`cache_utils.py` l.113; DeepSeek-V3.2 (fp8 latents in deployment); vLLM `kda_state_dtype` →
float32; vLLM #59147 (bf16 state neutral on GLM-5.3-Flash). Sources fetched to `/tmp/rs_dt`.

## History
- 2026-10-17: f32 latents questioned; "native" corrected (the checkpoint holds weights, the
  latents are activations); the reference's behaviour grounded.
- 2026-10-17 21:42 (`979fca8`): phase 1 — values rounded to bf16 where written, pools formed the
  reference's way, buffers unchanged. Long SCORE: mean |Δ logprob| 9.1e-3, NLL 0.0414 → 0.0446,
  within that text's ±0.003 noise (`891baeb`).
- 2026-10-18 09:00 (`971f5df`): phase 2 — packed rows (`layout::MLA_ROW_U16`: latent and indexer
  key as bf16 bits, the gate f32; 1792 bytes a layer against 3072) on the host, in the cards' `lat`
  and `pools` mirrors and in the context store (19 KB a position against 33). Long SCORE
  bit-identical to phase 1; decode unchanged; prefill at 18k 7.0 → 6.7–6.8 ms a prompt token. The
  latents' mirror at 32k: 360 → 180 MB. Lessons: a `u16` inner loop compiled to halfword loads on
  gfx1100 (fixed with `gemv::gfx11::widen8_at`, two u64 loads for eight bf16); the store's prefix
  seeding had to hand out the packed form too (an f32 seed went NaN on a cache resume).

## Next
The indexer's top-2048 selection overlap at 18k and longer between the f32 and bf16 builds
(unpublished anywhere); 031. The card wait that grows with position (attend and scorer) grows
~0.08 ms a prompt token per 1k positions — ~80 ms a prompt token at 1M if left as is (018, 019).
