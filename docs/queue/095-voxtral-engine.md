# 095 — Voxtral-Small-24B in the node

**Status:** LIVE beside GLM on metatron (aff68, 2026-10-23 15:24: `models glm-5.3-flash voxtral-small-24b`; GLM's
18k workload and long SCORE bit-identical beside it; `/rockem/v1/transcribe` answered three LibriSpeech clips word for
word in ~18 s each — one 30 s window: encoder + prefill ~15 s, then ~90 ms a token). aff69 fixes a refusal: a
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

## Open
- Speed: the encoder's GEMM is a naive tile (43 s a window on the dev host, idle); the prefill and decode on the box to
  be measured (aff65). Expected: decode ~40 ms a token at the DRAM ceiling, prefill a 30 s window in ~1 s.
- The encoder's 2e-2 on speech rows.
- A larger speech set with ground truth for WER, the recipe's real test.
