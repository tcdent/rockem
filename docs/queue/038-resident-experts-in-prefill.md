# 038 — Resident experts in prefill

**Status:** Resolved (built and measured 2026-10-19)
**Area:** prefill
**Gain:** measured — prefill at 18k 4.82 → 4.53–4.62 ms a prompt token (experts 1.93 → 1.50, 27 % of picks on the cards), the cold 18k prompt 90.6 → 86.3 s; decode unchanged with the join down to 0.1 ms a step; exact

## Proposal
`step_batch` dispatches to the resident experts only for steps of up to `SPEC_TOKENS`
(`construct.rs:1568`); a prefill step runs every pick on the CPU while the cards idle. Send the
resident experts' tokens (~14 a step an expert at 512) to the cards in a grouped multi-token iu8
WMMA kernel. Bit-identical: the format's integer group sums are exact in any order; keep the
CPU's f32 flush order.

## Source
tinygrad's grouped int8 experts (`03d069997`, eight waves sharing a tile's activations through LDS).

## History
- 2026-10-18: proposed.

## History
- 2026-10-19 04:30 (run 3aa84e93, `STEP` 1024, kernels `kernels-xg`): three grouped kernels — `xpt_gemm_gu`
  (a workgroup is 128 rows of one resident expert, eight waves a 16-row iu8 WMMA tile each, over every token
  that picked it sixteen a tile: the q4 nibbles widened to unsigned bytes against the int8 activation, the
  i32 group sum exact, the flush the CPU's expression in the CPU's group order), `xpt_quant` (the clamped
  SwiGLU and `QAct::quantize` a wave a group), `xpt_gemm_down` (likewise into the output rows by (token,
  pick)); the dispatch groups a step's resident picks by slot for any width (one path, decode included) and
  the join is a wait. Long SCORE bit-identical with 25 % of its picks on the cards: the kernels' bits are the
  CPU's. Prefill at 18k 4.82 → 4.56 ms a prompt token (experts 1.93 → 1.53–1.60, 22–26 % of picks resident),
  the cold 18k prompt 90.6 → 86.0 s. Decode read 1.3 ms a step worse: the outputs had moved into GTT host
  memory, which the CPU maps write-combining — the host's experts wrote and the sum read it uncached (the old
  join's 0.86 ms a step for 160 KB was the same thing). Fixed by a KFD userptr registration of the model's own
  pages (`Kfd::host_owned`): the cards write there, the CPU keeps its caches.

- 2026-10-19 05:00–05:55 (runs prow10–prow13): the owned buffer registered `MEM_COHERENT` was uncached on
  the card side — fine for the grouped kernel's writes, but the per-token kernel's scattered 4-byte stores
  then each crossed the link alone (join 12 ms a step); registered cached (as `host()` is), the fence
  flushes whole lines. Prefill's claims moved to each layer's join, so a resident in use this step can still
  be replaced: the residency went back to 33 % of decode picks (28 % when the claims ran in the dispatch
  under `in_use`). And the two-token verify step keeps the per-token kernels: the grouped kernel at two
  tokens ran its sixteen-token tiles over mostly empty columns with one memory latency a k-step and left the
  host waiting 0.7–1.3 ms a step on the join; the twins split at `SPEC_TOKENS` as the rows kernels do. Run
  36d2… (05:52, zero PCIe records): prefill 4.53–4.62 ms a prompt token, experts 1.50, 27 % of the prefill
  picks on the cards; the verify step 48.3–48.9 ms as before (join 0.86 → 0.08, the host's experts +0.5 from
  reading the cards' rows out of DRAM instead of the join's copy); long SCORE bit-identical.

## Next
The light experts: at 1024 the residents take the heavy ones, so what is left on the host is the
bandwidth-bound tail (021). The grouped kernel's own rate (one memory latency a k-step, no LDS) matters only
if the cards ever become the prefill phase's longer side; they finish inside the host's phase today.
