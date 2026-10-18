# 060 — Decode occupancy audit: what to fuse, what to split

**Status:** Open — an audit, then our own experiments
**Area:** decode (and the verify step's 2–4 tokens)
**Gain:** unknown until the audit; the literature's wins are 1.3–1.9× at batch 1 on other hardware

## Proposal
Fuse along the dependency chain; parallelize across it. A kernel can be *fused too wide* (it
absorbs independent work and serializes it, or its register budget lowers waves per SIMD) or
*parallelized too thin* (too few workgroups for 84 CUs). For every launch of a decode step and of
a two-token verify step, record:
- its workgroups against 84 CUs, and its waves per SIMD from the kernel report's VGPRs and LDS;
- whether it is on the chain or beside it (what it reads, when its output is first read);
- its time from the timeline.

Then: under-filled launches on the chain → split (split-K, narrower tiles, more workgroups) with a
cost model, not to fill the CUs; independent launches on the chain → free launches in the same
queue (barrier bit clear), or 057; device-wide barriers inside fused kernels → per-dependency
counters.

## Source
tcdent, 2026-10-18: "a fused kernel might be taking on a little bit too much work, and … it's
possible to have some additional parallelism to take advantage of more of the compute." Grounding
(read 2026-10-18):
- vLLM PR #61868 (GB300, DeepSeek-V4.1-Flash): fused the chain's boundaries (all-reduce + mHC +
  quantize, −11.4 % at batch 8; an epilogue into the mailbox, −41 µs) and moved independent work to
  side streams (−23 to −103 µs a piece); split-K for 1–8-row GEMMs with "few output tiles" (−60 µs).
- Fleet (AMD, arXiv 2604.15379, MI350 megakernel): the fused register footprint "limits occupancy
  to a single wave per SIMD", every L2 miss stalls — "an inherent tradeoff of the megakernel
  approach"; still 1.3–1.5× at batch 1–8. Split by output columns wins at batch 1–16, split-K from 32.
- ForgeMegakernel (arXiv 2609.12379): one launch that keeps grid-wide barriers lost (0.59–0.95× of
  SGLang's bandwidth); a launch boundary measured at 4.63 µs; the fix is counter dependencies.
- Cohere's megakernel (2026-09-08, H100, 1.58× at batch 1): the gain is wave quantization (idle SMs
  in a kernel's last wave) and boundary barriers, filled with the independent branch; task order
  inside it moved throughput up to 19 %.
- vLLM PR #61662 (GLM-5.3-Flash MonoKernel, MI355X): 9.36 → 4.82 ms a step; the gain shrinks with
  width (−43 % at width 2, −24 % at 8); fragile cross-CTA waits under GPU sharing.
- vLLM PR #60731 (MI300X): a 5-row lm_head on split-K 295 µs vs 96 on the plain GEMM — split-K is
  not automatically better. AITER PR #6126: split-KV counts chosen to maximize occupancy added merge
  work "without adding parallelism"; replaced by a fitted cost model.
- AMD consumer GPUs and multiple queues: ROCm #10836 (gfx1100) — HIP streams overlap poorly;
  Vulkan's win includes overlap within one barrier-free command buffer. TheRock #9241 (Redline on
  a 7900 XTX): two queues best, three or four time out, address-zero VM faults and GPU resets from
  queue churn. llama.cpp #16991: concurrent streams −3 % on an RX 6800 XT, +1–9 % on NVIDIA.

## Our facts so far (kernel report, 2026-10-18; 32 kernels, no scratch)
- On the decode chain: `kda_step` 153 VGPRs, 32 workgroups a card a layer; `hc_pre` 135, 192
  workgroups, all resident for its last-workgroup pass; `mla_attend` 36, a workgroup a head (016:
  four heads a workgroup cut it to 32 and lost); **`ix_select` 37, one workgroup a token — at decode
  a single workgroup scans every pool of a layer**, a serial scan that grows with context;
  `ffn_gu` 175, `mla_qb` 181, `rows_qkv` 172, `rows_part` 172 (one-token widths).
- Free launches (barrier bit clear) already exist in the launcher; the one queue per card has
  never had a second (057).

## History
- 2026-10-18: proposed; grounded by the literature above.

## Next
1. The audit table from a timeline of a decode step and a verify step (both widths; the fused vs
   split choice can differ between them).
2. First experiments: `ix_select` across workgroups with a fixed-order merge at long context;
   `kda_step` narrow value tiles (047); the decode attend split over positions (017).
3. 057 re-weighed: free launches within one queue before a second queue (instability reported on
   gfx1100).
