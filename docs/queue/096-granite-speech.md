# 096 — Granite Speech 4.1 2B: transcription suited to the CPU

**Status:** BUILT and gated on the dev host (2026-10-24), not booted — awaiting tcdent's go-ahead (lab boot:
`bringup/box/lab_boot.sh <label> bringup/box/dev.cmdline.lab-granite`, the bundle `bundles/lab-granite` beside
production). The edge's two changes for rockem.api are listed below.
**Area:** models/granite-speech-4.1-2b, bundle, node, bring-up
**Gain:** the leaderboard's best open transcription WER (5.33 on the Open ASR English set, against Voxtral Small's 6.62,
Mini 3B's 7.05, Whisper large-v3's 7.44) with 2B parameters instead of 24B: the decoder streams 0.9 GB a token
instead of 12.4, the encoder is 0.44B instead of 0.64B. The cards stay GLM's.

> tcdent, 2026-10-24: "we should think about choosing something that is somewhere in the same realm of performance and
> is more suitable to running on the CPU so that we can keep the GPUs free for our LLM decode ops … as long as we have
> accuracy in terms of words, that we don't need quite as much fancy punctuation." Then: "I'm leaning heavily toward
> granite … pull together an early implementation with as much pre-optimization as you can include."

## The model (ibm-granite/granite-speech-4.1-2b, Apache 2.0, pinned in `models/granite-speech-4.1-2b/checkpoint.toml`)
- **Features:** 16 kHz → 80 log-mels (n_fft 512, Hann 400, hop 160; log10, the clip's max − 8 floor, /4 + 1 — the
  same normalization as Whisper's), two 10 ms frames stacked: `[T][160]` at 20 ms.
- **Encoder (0.44B):** `input_linear` 160 → 1024; 16 Conformer blocks — `h += ½·FF₁`, `h += Attn`, `h += Conv`,
  `h += ½·FF₂`, `h = LN(h)` — the attention over blocks of 200 frames (4 s) with Shaw relative positions
  (`s·q·k + s·q·e_{c−r}`, a 1025-row table over distances clipped to ±200), the convolution module `LN → 1024→4096 →
  GLU → depthwise k15 → BatchNorm → SiLU → 2048→1024`; after block 8 the self-conditioned CTC `h += out_mid(softmax(out(h)))`
  over 348 characters.
- **Projector (0.16B):** every 15 frames a window, 3 learned queries through a 2-layer BLIP-2 Q-Former (post-LN at
  1e-12; self-attention, cross-attention to the window, GELU FFN), linear 1024 → 2048: 3 rows a window, 10 a second.
- **Decoder (1.4B + 0.4B of embeddings and head):** Granite 4.0 1B — 40 layers, 2048 wide, 16 query / 4 kv heads of
  128, RoPE θ 1e4, SiLU gate·up (4096) down, RMSNorm 1e-5 — with Granite's multipliers: the embedding ×12 (a placeholder's
  row too), attention scaled by 1/128 in place of 1/√d, every sublayer's output ×0.22 into the residual. The config's
  `logits_scaling` 8 the served composite model does **not** apply (its forward is the bare lm_head; transformers 5.13),
  so neither do we. `has_lora_adapter: false`: one weight set.
- **Prompt:** the chat template is plain text, `USER: <|audio|>can you transcribe the speech into a written format?\n
  ASSISTANT:` — ids `[6584, 25, 220, <100352 × rows>, 4919, 499, 1380, 3191, 279, 8982, 1139, 264, 5439, 3645, 30, 198,
  36660, 3931, 2891, 25]`; the transcript ends at `<|end_of_text|>` 100257. The prefix the node ships is the one id
  `USER` (6584). The "punctuated" prompt (`transcribe the speech with proper punctuation and capitalization.`) is the
  same model with casing and punctuation; the raw one is what tcdent asked for.
- **Rows per audio:** `ceil(⌊(⌊N/160⌋ + 1)/2⌋ / 15) · 3` for `N` samples. On the wire a window is 3 s of 10 ms frames
  (`[80][300]`, 96 KB) = 150 encoder frames = 10 projector windows = 30 rows; the encoder runs over all of a request's
  windows at once (its attention blocks and convolutions span them), so the node queues the windows and encodes them
  when the first placeholder needs a row. A request's audio is padded to whole 3 s windows at the edge.

## What was built
- `models/granite-speech-4.1-2b`: config constants via build.rs; `layout` (the decoder's fused q|k|v, o, gate|up, down
  and the lm_head padded to 100608 rows as a `q4g128` item stream in **four regions** — region r holds share r of every
  matrix of every layer, padded to the 2 MiB page once; Voxtral's per-layer padding would have wasted a quarter of this
  stream — and the dense bf16 tables for the encoder's 32 tensors a block, the Q-Former's 26 a layer); `weights`;
  `decoder` (Voxtral's with the multipliers and the padded vocabulary: padded logits −∞); `encoder` (everything through
  `common::sgemm` with the ½ and the 1/√d folded into the weights, the batch norm folded into the depthwise taps, the
  relative-position table packed once per block; a (block, head) attention item; the Q-Former's projections over
  every window at once and its attention a window an item).
- `common::sgemm`: SiLU epilogue, `silu`, sigmoid `glu`, `mul_add` (the depthwise taps).
- `bundle --model granite-speech-4.1-2b` (0.87 GB stream, 1.36 GB dense, 35 s), `bringup/granite/` (oracle.py with
  `--q4/--w8` recipe emulation and the reference's own mel filterbank dumped as `mel_filters.f32`; lab_bundle.py),
  `bringup/forward-granite` (encoder gate, decoder gate, `--transcribe` end to end), `node/src/granite.rs` (the
  engine; `granite_window` and `granite_step` rows on track-step; `bringup/reads/granite.sh`).

## The gate (libri2, 12.5 s, 2026-10-24, dev host)
| edge | vs the f32 reference on bf16 values |
|---|---|
| every Conformer block's output (16) | rel ≤ 5e-6 |
| encoder output | rel 7.1e-5; projector rows 7.1e-6 |
| decoder layers vs the INT4-matched reference | rel 1.5e-2 (l1) → 7.5e-2 (l39); logits rel 0.15 (the INT8 activations' and summation's share) |
| decoder vs the f32 reference | position 0 (the `USER` token) explodes at layer 20+ (rel 0.63, |Δ| 5e3: the sink position under INT4 — the recipe's own goldens reproduce it); every other position in the 1e-2 range |
| top-1 over the transcript | 34/34 against both; the transcript word for word |

End to end on the dev host (32 Zen 1 cores, loaded): encoder 1.8 s for 12.5 s of audio (ff 0.9, attention 0.4,
conv 0.4), prefill of 145 ids 1.6 s, decode 80 ms a token — the dev host's one-node numbers; the box's are the
measurement (Voxtral's dev-host prefill was 40 s against the box's 2.5).

## For rockem.api (the edge), before the boot
1. `[model.audio]` gains `fft = 512` (the transform's size; `fft_window` 400 stays the Hann window's) and `mel_scale =
   "htk"` (informational: the bank ships in `mel_filters.f32`, `[80][257]`). When `fft > fft_window`: center-pad the
   signal by `fft/2`, zero-pad each windowed frame to `fft` (centered), `rfft(fft)`, filters `[bins][fft/2 + 1]`. The
   log, floor, /4 + 1 and the windowing into `[bins][300]` blocks are Whisper's, already there.
2. `FORMATS["granite-speech-4.1-2b"]`: the prompt above, the `<|audio|>` expanded to `rows × windows` placeholders
   (100352), no BOS before `USER`; language is not a prompt parameter (the model detects it; non-English needs the
   English prompt).

## Open
- The box's numbers (encoder, prefill, decode a token) and the WER on more than three clips; the INT4 decoder's
  position-0 explosion (a sink-token effect, harmless to the transcript so far) — the V2-Lite lesson (FIRST_POS) if it
  ever shows.
- Padding at the edge: a clip's last window is zero audio the model hears as silence (≤ 3 s); Voxtral had the same.
