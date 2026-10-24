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
- **The KDA projections' stream rate (2026-10-24, run d7c400a0, live serving at 94k–107k):** the
  rows kernels of a verify step run at ~590 GB/s a card against the DRAM ceiling — the first
  measured fill of this audit's table. The shape: 43 launches a card a verify step, 0.18 ms a
  launch average, ~4.7 GB of projection weights a card a step (34 KDA layers × ~139 MB: the
  [M_SMALL = 12,608][H] gate/qkv launch, the [H][PART_QKV] output projection, f_b, taps).
  Compute is not the constraint — a gemv is ~2 FLOPs a byte, ~9.4 GFLOP in ~7.9 ms is ~1.2 TFLOPS
  against ~90 — the cores wait on memory. The remaining ~30 % to the ceiling has three named
  suspects, all of the kernel's own construction (`common::gemv::gfx11`, tuned in the one-token
  era when verify steps were the exception; gated depth 3 made them the majority):
  1. the verify width's round is 4 steps (vs FLIGHT 8 at one token) — fewer bytes in flight per
     wave, a drain between rounds every round;
  2. occupancy: `rows_per_wg(2..4)` is 32, so the big launch is 128 workgroups across 84 CUs
     (~1.5 a CU against the ~4 that fit); at one token it is 512 — over-queued instead;
  3. launch tails: the small launches (f_b at 2 MB) ramp and drain the pipe for their whole run.
  The audit's first experiment is built: `bench-rows` (bringup/bench/src/rows.rs) benches the
  decode path's `rows_h` at the served shapes and widths, with the launch's geometry on every
  row. The runs for the next bench (the card idle, or a boot; the same boot for the ceiling):

  ```
  bench-rows --elf <kernels.gfx1100.elf>                  # the sweep: m 12608,4096 × n 1,2,3,4, single-launch
  bench-rows --elf <kernels.gfx1100.elf> --chain 43       # the serving cadence (43 launches a card a verify step)
  bench-dram   # the card's streaming ceiling, the same boot, to print the sweep against
  ```

  Reads: the verify widths' (n 2–4) GB/s against n 1's at the same m — a half-rate confirms the
  4-step round / the 128-workgroup grid as the limiter; `--chain 43`'s per-launch time against
  the single-launch's — a drop confirms the launch tails are a real cost (merge f_b's launch;
  independent rows, no dependency). The round depth and the rows a wave are compile-time
  constants (`common::gemv::{FEW, FLIGHT, ROWS}`): the sweep judges the shape as built, and a
  variant of them (round 4 → 8 at the verify widths) is a one-constant rebuild, bit-identical
  arithmetic, judged by re-running the same two commands. 029 (FP8 projections) would halve the
  bytes instead; this audit's fixes get the same milliseconds at full precision.
- **`bench-rows` measured (2026-10-24, metatron aff78, card 0 beside the idle serving node):** µs a launch (GB/s):

  | m (MB) | n 1 | n 2 | n 3 | n 4 | grid n 1 / n ≥ 2 |
  |---|---|---|---|---|---|
  | 12608 (103), single | 157 (656) | 161 (643) | 161 (643) | 162 (638) | 1576 / 394 |
  | 12608, chain 43 | 142 (726) | 145 (711) | 146 (707) | 147 (705) | |
  | 4096 (34), single | 35 (962) | 52 (641) | 53 (637) | 53 (637) | 512 / 128 |
  | 4096, chain 43 | 19.6 (1712) | 37.2 (902) | 37.3 (900) | 37.5 (895) | |
  | 65536 (537), chain 43 | 707 (759) | 723 (743) | | 729 (736) | 8192 / 2048 |
  | 1024 (8), chain 43 | 10.8 (776) | | | 27.2 (308) | 128 / 32 |
  | 256 (2), chain 43 | 8.7 (241) | | | 23.7 (89) | 32 / 8 |

  Reads:
  1. **From VRAM the kernel is at the card's ceiling at every width.** m 65536 (5× the 80 MB Infinity Cache) streams
     759 GB/s at one token and 736 at four, against the 640–770 measured in 061. The round depth (suspect 1) and the
     grid at the big shapes (suspect 2 at m 12608) cost 2–3 %.
  2. **m 4096 and below are cache rates in this bench** (1712 GB/s is the Infinity Cache): a re-run reads the same
     34 MB. Served, the 34 layers' 4.7 GB a card never sits in the cache, so these rows bound the issue rate, not the
     served stream. To bench them from VRAM the weights have to rotate through > 80 MB across the chain.
  3. **Small launches at the verify widths are occupancy-bound** (suspect 2, where it bites): at n ≥ 2
     `rows_per_wg` is 32 rows, so m 256 is 8 workgroups and m 1024 is 32 on 84 CUs. They take 24–27 µs against
     9–11 at one token, even cache-resident: ~15 µs a small launch a verify step. The candidate is the rows a wave
     past one token chosen by the grid (`r` 1 when `m / (8·ROWS)` is under ~2 workgroups a CU; the LDS-bound reason
     for `ROWS` 4 was 32-token passes, not widths 2–4), bit-identical arithmetic. Its size is the count of small
     launches in a verify step × ~15 µs; the timeline names them.
  4. **Chain vs single:** the single-launch rows carry the ~15 µs sync of each round (061's 14–16 µs), so the chain
     rows are the serving figure; they do not separate launch tails inside a chain.
  5. The served 590 GB/s against these 705–759: the gap is not the big launches' streaming. Read 3 is the measured
     candidate; the rest is between launches (barriers, dependent launches) and needs the timeline.

  Two notes on the procedure: `bench-dram` is the host's DRAM bench (CPU threads), not the card's ceiling — 061's
  640–770 is the card's, and `bench-box`'s VRAM probe (now `rockem.probe=1`) needs the cards free (beside the node it
  ran out of VRAM). And `bench-rows`' header says "4 waves" for 256 threads: gfx11 runs wave32, so it is 8 (the
  `/ 64`), cosmetic.
- **The narrow twin built (2026-10-24, after the sweep):** `rows_h1` and `rows_part1` — the same kernels at one row
  a wave (`rows_over::<_, _, 1>`, kda.rs) — and the host picks them in `rows_at` when the wide grid would be under
  ~2 workgroups a CU past one token (`m / rows_per_wg(n, ROWS) < 2 · CUs`): the o projection's m H (128 workgroups
  on 84 CUs), the router rows, anything smaller; the M_SMALL launch (394 workgroups) and the draft's head stay wide.
  The rows' arithmetic is unchanged, only their grouping — the outputs are the same bits, no gate needed beyond the
  bit-identity check. The A/B, when the card is free:

  ```
  bench-rows --elf <kernels.gfx1100.elf> --m 4096,256,1024 --chain 43                # rows_h, as built
  bench-rows --elf <kernels.gfx1100.elf> --m 4096,256,1024 --chain 43 --kernel rows_h1   # the narrow twin
  ```

  The n 2–4 rows should fall from 24–27 µs to ~9–11; then a served ABBA on one boot (the wavefront's own step rows
  and the timeline's `rows_part`/small-launch zones are the measure). Note the elf must be rebuilt (`make` in the
  gpu crate) before the node or the bench see the new kernels; the host picks them by name at boot.
- **The narrow twin measured and deployed (2026-10-24, aff80):** bits equal to `rows_h` at every shape and width
  (`bench-rows --check`). µs a launch at n 4, chain 43, one boot (wide `rows_h` / narrow `rows_h1`):

  | m | wide grid | wide | narrow |
  |---|---|---|---|
  | 256 | 8 | 23.2 | 14.8 |
  | 1024 | 32 | 27.0 | 19.7 |
  | 1536 | 48 | 34.9 | 22.4 |
  | 2048 | 64 | 35.2 | 33.1 |
  | 3072 | 96 | 34.1 | 37.6 |
  | 4096 | 128 | 36.8 | 50.1 |

  The twin wins below about one workgroup a CU and loses past it: one row a wave reads the staged activations four
  times as often, and past ~84 workgroups that costs more than the occupancy buys. So the host's switch is
  `wide grid < CUs`, not `< 2 · CUs` as first written — which had put the output projection (m H, 128 workgroups)
  on the twin, 13 µs slower a launch. Served, the twin takes the router rows (576 rows, 18 workgroups wide); the
  output projection and the M_SMALL launch stay wide. **Verify step at 10–20k: 22.68 → 22.51 ms on aff80** (aff76–79
  read 22.67–22.69 there), long SCORE bit-identical — **but aff81, the same ELF and no step change, read 22.74**: the
  0.17 ms is within the boots' spread, so the twin's served gain is unestablished (the bench's ~7–8 µs a small
  launch at verify widths stands; how many small launches a verify step has is the timeline's). The m 4096 rows above are Infinity Cache rates (34 MB re-read);
  the crossover from VRAM could sit elsewhere, and a bench that rotates its weights through > 80 MB would place it.
  Notes: `bench-rows` sized the grid at `ROWS` for every kernel, so `--kernel rows_h1` launched a quarter of its
  workgroups (it now takes one row a wave for the `*1` kernels); the production ELF before the twin is kept at
  `bundles/tools/kernels-aff79.gfx1100.elf`.

- **`ix_select_d` built, unmeasured (2026-10-24):** the few-token step's selection without the
  wide kernel's cross-slice round trips. At `n ≤ SPEC_TOKENS` past 32k positions — decode and the
  2–4-token verify both — the wide kernel runs one workgroup a token (182 µs a layer at 262k, the
  kernel report's serial scan), and 019's slices can't help: they spin on each other every radix
  pass, and with few tokens on the chain those round trips are pure latency. `d` (mla.rs, beside
  `ix_select`) splits the token's pools into `decode_slices(pools)` spans of ~4096 pools, a
  workgroup each; each picks its own top-`KEEP` candidates by a radix select wholly in its LDS and
  posts them (key, pool) to a new scratch (`mselc`, [SPEC_TOKENS][64][512 × 2] u32, ~1 MB) in
  index order; every workgroup arrives once (the wide kernel's fetch-add pattern) and the last to
  arrive merges alone — a radix select over ≤ 32k candidates, above-threshold taken outright,
  equals by index — then writes `sel`/`counts` and resets the state. The set is the one-workgroup
  scan's exactly: a pool in the global top `KEEP` has fewer than `KEEP` pools before it in its own
  slice under the same (key descending, index ascending) order. The host takes `d` in
  `glm_kfd.rs`'s `ix_select` whenever `n ≤ SPEC_TOKENS && pools > SELECT_SINGLE`; the state words
  past the wide kernel's (`SS_DCNT`, `SS_DARRIVE`) grow `SELECT_STATE` by 66. Kernel report: 43
  VGPRs, no scratch. The A/B and the bit check (bench-mla-attend now times both and diffs the
  sets, the wide kernel's output the reference it already checks against the host's `pick`):

  Built on the dev box 2026-10-24 (the elf at
  `models/glm-5.3-flash/gpu/target/amdgcn-amd-amdhsa/release/glm_5_3_flash_gpu.elf`, kernel report
  43 VGPRs no scratch; the bench binary `bringup/bench/target/release/bench-mla-attend`); where
  the elf is rebuilt by hand it is `cargo +nightly build --release` in
  `models/glm-5.3-flash/gpu` (the pinned nightly; plain `cargo` has no amdgcn target) and the bench
  is `cargo build --release --bin bench-mla-attend` in `bringup/bench`. The A/B:

  ```
  bench-mla-attend --elf <kernels.gfx1100.elf> --pos 65536,262144,1048576 --tokens 1,2,4
  ```

  (`--tokens 512` in the bench's default never takes `d` — the host's few-token gate is
  `n ≤ SPEC_TOKENS && pools > SELECT_SINGLE`.) Read: `ix_select_d`'s µs against the wide kernel's at each position (expect the scan divided by
  the slice count and the merge a few µs — 182 → ~15–30 at 262k), identical set: true. Then a
  served ABBA at long context (the timeline's `ix_select` zone the measure).
- **`ix_select_d` measured (2026-10-24, metatron aff80, card 0 beside the idle serving node): slower, and its set is
  wrong — not deployable as built.** µs a launch (wide `ix_select` / `ix_select_d`):

  | positions | slices wide / d | 1 token | 2 tokens | 4 tokens |
  |---|---|---|---|---|
  | 65536 | 4 / 4 | 89.4 / 126.5 | 90.4 / 124.8 | 79.9 / 125.0 |
  | 262144 | 16 / 16 | 89.9 / 174.4 | 91.1 / 172.9 | 98.2 / 181.5 |

  1. **The premise was the serial scan, which serving no longer runs.** The wide kernel at 262k is 90 µs, not 182: it
     already splits the token's pools over 16 slices (019), so the few-token step is not one workgroup's scan.
  2. **The set: the first slice's candidates are missing.** At 65,536 positions (4 slices of ~4,096 pools) `d`'s
     first selected position is 16,388 (pool 4,097, slice 1) where the wide kernel's is 296; ~1,500 positions a token
     differ, about one slice's share of the 512 pools, the counts equal (2049). The merge drops slice 0's candidates (or
     their count word) — a bug to find in the merge or the posting, which would also have to be fixed before any timing
     counts, though the timing above is already 1.4–1.9× the wide kernel's.
  3. 1M did not fit beside the serving node (its latents); it needs the cards free.

  **The host gate in `glm_kfd.rs` routes every few-token step past `SELECT_SINGLE` pools to `d`**, so the next node
  deploy from this tree would select the wrong positions at every decode and verify step past ~32k context (the long
  SCORE gate runs prefill steps, which never take `d`, so it would pass). The gate stays undeployed until the set is
  right and the timing wins.

- **`ix_select_d` concluded — the posting bug found and fixed, the idea set aside (2026-10-24).**
  1. *The bug:* the posting wrote each slice's candidates at `2·(rank + r)` — no slice stride — so
     every slice posted over the block's start and the merge read slice 0's slots holding whichever
     slice posted last; slice 0's candidates were gone, its count word intact. Fixed to
     `2·(sl·KEEP + rank + r)`, both posting paths; the merge's reads were already correct. Static
     profile unchanged (43 VGPRs, no scratch).
  2. *The premise was stale:* the wide kernel at 262k is 90 µs, not 182 — `select_slices` already
     fans the few-token step out (that is what its `SPEC_TOKENS` gate is for; a step of many tokens
     is the one that takes a slice a token). The audit's "one workgroup a token at decode" read the
     kernel report's single-slice era.
  3. *The timing stands even against the fixed kernel:* `d`'s shape — a local radix plus a posting
     scan per slice, then a single workgroup's four passes over ≤ slices·KEEP candidates — costs
     more than the wide kernel's shared-histogram spins save at these spans. Nothing here to
     deploy.
  The host gate is back to the wide kernel for every step; the kernel, the state words and the
  bench's A/B remain in the tree for a retiming if the timeline ever names `ix_select` as a zone
  worth the merge's cost (the bench compares the sets and the order now, so a rerun is checkable).
  The experiment's value was the correction: `ix_select` is not the audit's serial outlier — the
  kernel report's 37-VGPR one-workgroup read was. The remaining decode-side candidates from the
  audit table are `kda_step`'s narrow value tiles (047) and the attend split (017).

  The bench needed three fixes to run (bringup/bench/src/mla_attend.rs): `mla_attend_w` and `mla_attend_merge` gained
  the `ml` output in 058 (2026-10-19) and the bench still passed the old arguments (the kernarg-size assert);
  `ix_select_d` was compared with `sel_w`, the selection over `ix_score_w`'s scores, while it ran over `ix_score`'s
  (the scalar scores overwrite the buffer first; the two differ by 4–20 positions) — it is compared with the wide
  kernel's selection over the same scores now; and it prints the order, not only the set (the attend sums in the
  list's order, so a reordered set changes decode's last bits). The `|Δ|max` of the attend line compares the
  normalized `mla_attend` with `mla_attend_w`'s raw mix (058's raw mode), so it is no longer a check.

- **`ix_select_d` removed (2026-10-24, aff81):** the kernel, its state words (`SELECT_STATE` back to the wide
  kernel's), the host's field and its 1 MB candidate buffer, and the bench's A/B — measured slower than 019's slices
  at every point and dropping a slice's candidates. This entry keeps the design and the numbers; the code is in git
  history if the timeline ever names `ix_select` again.

## History
- 2026-10-18: proposed; grounded by the literature above.
- 2026-10-24: first measurement landed (the KDA projections' stream rate, above) from the live
  run's zones; `bench-rows` built (the sweep and the chain cadence at the served shapes), the
  runs above are ready for the next bench.
- 2026-10-24: the sweep measured (above): the big shapes at the ceiling, the small launches
  occupancy-bound at the verify widths; the narrow twin (`rows_h1`/`rows_part1`) built, the host
  selection in `rows_at`, the A/B commands above.
- 2026-10-24: `ix_select_d` built (the few-token step's fixed-order merge, above), unmeasured —
  the bench box's run is next.
- 2026-10-24: `ix_select_d` measured (above): slower than the wide kernel and its set wrong; the
  posting bug found and fixed, the premise corrected (019's slices already cover the few-token
  step), the idea set aside, the gate back to the wide kernel.

## Next
1. The audit table from a timeline of a decode step and a verify step (both widths; the fused vs
   split choice can differ between them).
2. First experiments: `ix_select` across workgroups with a fixed-order merge at long context;
   `kda_step` narrow value tiles (047); the decode attend split over positions (017).
3. 057 re-weighed: free launches within one queue before a second queue (instability reported on
   gfx1100).
