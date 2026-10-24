# 095 — Voxtral-Small-24B in the node

**Retired from the manifest 2026-10-24** (tcdent: "We do not need to continue deploying Voxtral"): Granite Speech 4.1 2B
(docs/queue/096) serves transcription — better WER at a twelfth of the decoder bytes, suited to the CPU. The engine,
builder and gates stay in the tree; the bundle's files are in `bundles/voxtral-staging` (`bringup/gates/bundle_merge.py`
puts them back).

**Status:** LIVE beside GLM on metatron (aff68, 2026-10-23 15:24: `models glm-5.3-flash voxtral-small-24b`; GLM's
18k workload and long SCORE bit-identical beside it; `/rockem/v1/transcribe` answered three LibriSpeech clips word for
word in ~18 s each — one 30 s window: encoder + prefill ~15 s, then ~90 ms a token). **aff75 (2026-10-23 evening): a
30 s clip in 6.0 s** — encoder 1.0 s, prefill 2.5 s, 54 ms a token; the speed pass below, read through prow
(`bringup/reads/voxtral.sh`). aff69 fixes a refusal: a
connection closed with audio rows still queued left them in the engine and the next BOS refused every later
transcription; a new connection drops them now. In the gate: the decoder's top-1 against the reference's own
transcript 37/38; the three clips transcribed exactly (casing and punctuation aside), ending at EOS on their own.
**Area:** node, bundle, bring-up
**Gain:** transcription as a separate API step (docs/queue/094), audio in and reviewable text out, on the CPU beside
GLM — both resident, the wire selecting (tcdent 2026-10-23: "not parallel but in ram", "same node process").

## What was built
- `models/voxtral-small-24b`: the pinned checkpoint (`da5b4240`), `build.rs` over both configs, `layout` (the decoder's
  fused q|k|v, o, gate|up, down and the lm_head as a `q4g128` item stream in 2 MiB page runs, run `r` on node `r mod
  nodes`: 12.44 GB; the encoder, projector, embedding and norms native bf16/f32: 2.72 GB), `weights`, `decoder`
  (`q4::gemm` an item over the pool, INT8 activations a token, f32 rotary θ 1e8, GQA attention over an f32 K/V cache,
  SiLU; a placeholder's input is its row), `encoder` (conv frontend, sinusoidal positions, 32 bidirectional pre-norm
  layers, final norm, the projector over four positions; a tiled f32 GEMM over the pool).
- `bundle --model voxtral-small-24b` from the HF shards, `bringup/voxtral/edge_files.py` for `tokenizer.json`
  (tekken converted; ids checked against mistral-common) and `mel_filters.f32`; the `[[model]]` entry with
  `[model.audio]` appended to the GLM bundle's manifest.
- `bringup/voxtral/oracle.py`: transformers' reference in f32 per layer on the bf16 values over a transcription
  request, every edge saved; `--q4` / `--w8` / `--fp8` emulate a weight recipe for recipe-matched goldens.
  `bringup/forward-voxtral`: the decoder from the oracle's embedding rows, the encoder from its windows, and
  `--transcribe` end to end from `windows.py`'s windows (greedy, through vocab.bin).
- The node: `Engine` gains the encoder hooks; AUDIO on the wire (the window's bytes folded into every placeholder's
  store key; refusals as 094 states); `loader::load_bound`; `node/src/voxtral.rs` (greedy).

## The gate's numbers (2026-10-23, the dev host)
| edge | vs the f32 reference |
|---|---|
| encoder output (tone clip) | rel 3.2e-4; projector rows 1.9e-4 |
| encoder output (speech) | rel 1.9e-3; rows 2.2e-2 — to be understood (f32 both sides; the error enters somewhere in the 32 layers) |
| decoder layer 0, audio positions | rel 7e-4 – 5e-3 |
| decoder layers 0–38 vs the INT4-matched reference | rel 1e-3 → 5e-2 (the INT8 activations' and summation's share) |
| decoder layer 39 vs the INT4-matched reference | rel 0.18: the last layer's activations are huge and INT8 per 128 loses them |

**The recipe (tone clip, the reference's own 48-token hallucinated transcript; top-1 teacher-forced):** f32 0.979,
INT8 per 128 0.979, e4m3 per 128×128 block 0.917, INT4 per 128 0.833. On speech, the INT4 decoder transcribes the
three clips exactly: the tone transcript is a flat distribution any noise flips, speech is confident. INT4 ships
(12.4 GB a token); INT8 (24 GB, lossless on this measure) is the fallback if a larger speech set shows damage; FP8
is not it.

## Lessons
- `common::q4::GROUP` (the quantization group, 128) imported into the decoder shadowed the model's `GROUP` (query
  heads a kv head, 4): every query head read kv head 0, and the gate read it as "the recipe is coarse" for an
  afternoon until a single-position f32 recomputation inside the gate disagreed with itself. The constant is
  `KV_GROUP` now; a kernel file that imports a quantization constant must not share a name with a model constant.
- The oracle's transcript (minutes of bf16 generation on the CPU) is cached beside the goldens (`transcript.json`).
- A Vec's `as_ptr().add(len)` is not the next Vec: the gate|up GEMM wrote past `gate` into nothing until `gu` became one
  allocation (a release-mode segfault on the first step).

## The two-model hang (aff65 → lab-fixed, 2026-10-23) — FOUND
`loader::map_memory` named each hugetlbfs file by the bundle file's basename (`/dev/hugepages/dense.bin`); both
models have a `dense.bin`, so Voxtral's load reopened the file GLM's 16 GB mapping was backed by and `set_len` it to
3 GB — GLM's next read past that was a SIGBUS, which killed the node (pid 1), which panicked the kernel (the text
never reached the console) and reset the box. The file is named by directory and file now
(`glm-5.3-flash.dense.bin`). The hunt below stands as the record of what it looked like from outside.
With both models resident the node answers its hello, Voxtral transcribes (libri2 over the wire on metatron: word for
word, prompt + first token 14.9 s, 92 ms a token), and GLM's first step hangs: no token, the box's telnet dead within
seconds, a silent reset minutes later. Reproduced on every boot with both; GLM alone and Voxtral alone are fine. Ruled
out so far, one lab boot each (`bringup/box/lab_boot.sh`, the experimental `bundles/both` of symlinks, production
untouched): two pools pinned to the same cores (one host pool shared by both engines now: `main` builds it, the
engines take `&'static Threads` — still hangs), the cards (GLM on the CPU only — still hangs), the node's allocator
(the system allocator — still hangs). The pool's call pattern (Voxtral's plain and node-placed runs, then GLM's
prepared slotted jobs) completes on the dev host (`bench-pool`). Console prints
(2026-10-23 13:35): GLM's two prefix steps at open complete (before Voxtral loads); the first request's step prints
"layer 0" and nothing more, and a watchdog thread sleeping 5 s between prints of the pool's state never printed — the
process (or the box) is frozen, not spinning in the pool. The box's telnet shell dies with it. Next: a GLM step at
open after both models have loaded (the serve loop out of the picture), the watchdog on the housekeeping core.
(The host's RAID array is reshaping, 2026-10-23 14:00: NFS reads at 120–170 MB/s, a lab boot ~25 min.)

## The speed pass (2026-10-23, aff72–74) — libri2 (30 s, one window, 382-id prompt, 40 ids out) through the service
Read back with `bringup/reads/voxtral.sh` (rows `audio_window` and `vox_step` on `track-step`, a row a window and a
step, the phases as columns; the `request` row carries the model name since aff74). tcdent: "take a pass at seeing if
we can get this faster … try a little bit on tuning and then explore other options".

| build | wall | encoder a window | prefill a 32-token step | decode a token |
|---|---|---|---|---|
| aff71 (before) | 18.8 s | ~8–10 s | ~800 ms | ~90–100 ms |
| aff72: `common::sgemm` | 15.7 s | 1.34 s | 800 ms | 100 ms |
| aff73: per-token work over the pool, down split four ways | 8.6 s | 1.30 s | ~255 ms | 72.8 ms |
| aff74: K/V cache a kv head's positions contiguous, attention a (kv head, 8 tokens) an item, vectorized SiLU, rope a token an item | 6.1 s | 0.93 s (1.31 the first after boot) | 236 ms | 52.9 ms |
| aff75: the small runs spread over the complexes, attention blocks of 2 tokens | 6.0 s | 0.99 s | 210 ms | 53.8 ms |

aff75's step (prow, `voxtral.sh`): prefill a step: norm 8.5, qkv 12.8, rope 2.2, attn 13, o 17.7, gu 64, silu 18, down 75, head 1.3
(the weights 170 of 210); decode a token: attn 4.9, o 2.6, gu 25.3, silu 1.7, down 13.8, head 1.3 (the weights 45 against a
41 ms DRAM floor). The encoder a window: norm+pack 86, qkv 155, attention 103, o 124, ffn 490, projector 13. libri2 end
to end 18.8 → 6.0 s; libri0 (shorter) 5.0, libri1 4.4.

- **The encoder's GEMM** (`common::sgemm`): the dot form (`linalg::dot` a row and position, a horizontal reduction an
  output, the activation row re-read for every weight row) ran at a few FLOP a cycle. Now the weights are packed once
  into 16-row k-major panels and a 6×16 tile holds twelve accumulators: two loads and six broadcasts feed twelve FMAs a
  k; k walks in blocks of 256 (a panel's 16 KB in L1) through a C scratch; the epilogue folds the bias, the GELU and
  the residual add. The dev host (32 Zen 1 cores, loaded) 43 → 7.3 s a window from the kernel alone, 7.3 → 3.8 s from
  placement: the items (48 positions × 128 columns) are ordered by column block and `Pool::complexes` hands a
  contiguous run of them to one core complex, whose L3 holds the weight block while its cores stream the positions
  (before that, every item pulled its 640 KB block from DRAM: the traffic, not the FMAs, was the bound). The gate's
  error shrank (speech rows rel 2.2e-2 → 7.5e-3, tone 3.2e-4 → 2.6e-4): the dot form's summation order was the worse
  one, which closes most of the "2e-2 on speech rows" item. q|k|v fused with the 1/8 folded into q (a power of two,
  the reference's bits), the convolutions as GEMMs over the unrolled input, attention a (head, 48 positions) item
  against the head's keys and transposed values packed as panels, `softmax_row` vectorized (`exp8`).
- **On the box the encoder is 1.3 s**, 2.5 TFLOP/s over 96 cores for the GEMMs (27 % of peak; the dev host's Zen 1
  reaches 45 % of its) and attention at a third of that rate: every activation lives in one node's DRAM and three
  nodes read it across the fabric (~45 GB a window). Quantified, on the shelf — ~0.7 s a window: node-local
  activations (position blocks bound per node) with the packed weights replicated a node (4 × 2.7 GB).
- **The decoder's prefill** (aff73): the norm + quantize, the o-input quantize and SiLU·up + quantize ran on the main
  thread — a million SiLUs a layer a step, 28 ms a step for the SiLU alone — now a token an item; down's 20 items of
  256 rows (20 of 96 cores streaming a third of the layer's bytes) split four ways at runtime (whole tiles). A step
  800 → 255 ms; decode 100 → 73 ms a token.
- **Attention** read K and V at `[position][KV_DIM]`: a head's rows a page apart, past the prefetcher — 18 ms of the
  73 a decode token at 400 positions, and 11 → 47 ms a prefill step as the position grew. aff74 lays the cache out
  `[layer][kv head][position][HEAD_DIM]` and makes an item a kv head × a block of 8 tokens (the four query heads that
  share it): the head's keys and values are one sequential stream read once a position for 32 (token, head) rows.
  `rows()`/`seed()` keep the store's POSITION layout.

## Open
- Speed, what the pass left (see the table above): the encoder's node placement (on the shelf, ~0.7 s), down at K =
  32768 (72–114 ms a prefill step against ~40 expected: the 1 MB of INT8 activations a 32-token step is re-read a tile
  pair from beyond L2), the pool's wake cost on the per-token runs (~140 µs a run, 80 norm runs a step = 11 ms), the
  decode's DRAM floor at 12.4 GB a token (~41 ms of the 73).
- The encoder's 7.5e-3 on speech rows (was 2.2e-2; f32 both sides).
- A larger speech set with ground truth for WER, the recipe's real test.
