# 094 — AUDIO: an encoder's input on the wire

**Status:** Landed in part (aff61, 2026-10-23) — wire v5 is deployed: the hello names the model, the
manifest is a `[[model]]` list with each model's files under `<name>/`, a session per model on the
node, the API speaking v5 with `ROCKEM_MODEL` (the 18k workload served, long SCORE bit-identical).
AUDIO is reserved and refused by a model without an encoder. Still to build: the Voxtral bundle and
engine with AUDIO's receive (rockem.lead); the spectrogram, the transcription prompt and
`POST /rockem/v1/transcribe` (rockem.api).
**Area:** wire, node, bundle, API
**Gain:** a transcription model behind the same protocol as every other node (Voxtral Small 24B,
tentatively); the same verb carries any model's encoder input later (images)

## Why

Transcription is a separate step at the edge: audio in, text out, which the user reviews and
then sends as an ordinary turn. The chat path never carries audio.

> tcdent (2026-10-23): "thinking of it as a separate transcribe step so codey just sees parsed
> text which gives me an opportunity to review." And: "offload any preprocessing to the api."

The model reads audio as embeddings in its token sequence. A placeholder id marks each position
an audio row fills, and an encoder plus projector computes the rows from a spectrogram window.
The wire carries ids only today; this adds the one input it lacks.

One node serves both models.

> tcdent (2026-10-23, with rockem.lead): Voxtral runs in the same node process as GLM — both
> resident, not run in parallel, no dynamic weight loading, one queue; the wire says which model
> a connection uses.

So version 5 carries two changes at once: the hello names the model, and AUDIO.

## The model, as the spec assumes it

Voxtral-Small-24B-2507 (checkpoint config, transformers' reference, mistral-common 1.12.0 run on
the model's own `tekken.json`; details in the api's memory `voxtral-transcription`):

| what | value |
|---|---|
| encoder input | one 30 s window: log-mel, 128 bins × 3,000 frames (16 kHz, hop 160, window 400) |
| encoder | Whisper-large-v3 shape: 32 layers, 1,280 wide, 1,500 positions per window |
| to rows | 4 consecutive positions concatenated (5,120), then linear → GELU → linear (5,120 → 5,120, no bias) |
| rows per window | 375 (12.5 a second) |
| placeholder | `[AUDIO]`, id 24; each takes the next row in place of its token embedding |
| decoder | Mistral Small 3.1: 40 layers, GQA 8, vocab 131,072, 32k context |

Windows are encoded independently. The spectrogram is not: it is computed over the whole clip,
padded to a multiple of 30 s and normalized by the clip's maximum, then cut into windows. That
is the API's work; the node receives finished windows. Because of that normalization a window's
bytes depend on the whole clip, which is why a digest of each window is enough for the store and
why two different clips never share a window.

The transcription prompt, as the API sends it:

```
<s> [INST] [BEGIN_AUDIO] [AUDIO]×375·N [/INST] (lang:xx) [TRANSCRIBE]   then GENERATE
```

## The hello names the model

The client's hello grows from 16 bytes to 48:

| offset | size | field |
|---|---|---|
| 0 | 4 | magic, the bytes `rock` |
| 4 | 4 | version, u32 LE, 5 |
| 8 | 8 | identity, u64 LE: the bundle's (one bundle, one identity, every model in it), or 0 for any |
| 16 | 32 | model: the `name` of one of the manifest's models, UTF-8, zero-padded |

The node answers with its 20-byte header as today, its positions being that model's budget, and
the connection speaks that model until it closes: its ids, its verbs, its context store. A name
the bundle does not have is refused like a wrong identity, by closing without a header. There is
no default model; the client always names one. Each model keeps its own store and its own state,
and the node serves one connection at a time across all of them, so a request for either model
waits in the same line.

## The verb

| value | meaning |
|---|---|
| `AUDIO` (`0xFFFFFFFD`) | client → node: one encoder window follows, exactly the size the manifest states, as raw little-endian f32 (`128 × 3000 × 4` = 1,536,000 bytes for Voxtral), mel-bin-major. No length field: the size is a constant of the model, like everything else that frames the wire. |

- **Order.** AUDIO is valid inside a sequence (after BOS) and before the placeholders it fills.
  The node encodes the window and queues its rows. Each placeholder id fed after that takes the
  next queued row as its input in place of the token embedding. Several AUDIO words may precede
  a run of placeholders; rows are consumed in the order the windows arrived.
- **Refusals** (the connection closes, as for any protocol error): AUDIO before BOS, or for a
  model with no encoder; a placeholder id with no row queued; rows still queued at GENERATE,
  SCORE or BOS.
- **The counts must agree.** Every row queued is consumed by exactly one placeholder before the
  sequence generates: as many placeholders as rows, no more and no fewer.
- **In the node's step,** a placeholder position's input is its row instead of a token
  embedding; the ids still frame the step (up to 32 a step, placeholders among them).
- **Positions.** A placeholder is a position like any id: it counts against the header's budget,
  steps through prefill with its neighbours (up to 32 a step), and is in the cache after.
- **SCORE** is not defined for placeholder positions (there is no id to score); a SCORE before
  all rows are consumed is a refusal.

A transcription session:

```
client → node   header
node → client   header
client → node   BOS, [INST], [BEGIN_AUDIO],
                AUDIO, <window 1>, AUDIO, <window 2>,
                [AUDIO] × 750, [/INST], [TRANSCRIBE], GENERATE, 0
node → client   id, id, …, </s>
```

## The node's store

Audio is not cached (tcdent, 2026-10-23). Each connection's placeholder positions are keyed
from a salt no other connection has, so they never match the store, and every window's rows are
always taken by its placeholders. Text-only prompts cache as before. A transcription is a single
request, not a conversation that returns, so nothing is lost: the encoder and the prompt run in
full each time.

The rows a connection queued but never consumed belong to that connection. A new connection
starts with none, and the refusals for leftover rows (at BOS, GENERATE or SCORE) apply within
one sequence.

### How this went wrong before

The store first matched audio positions by the ids plus a 64-bit digest of each window, folded
into the running key at AUDIO, so a repeated clip would hit. On a hit the node skipped the
placeholder positions, but the rows the windows queued stayed, which is the case the refusals
exist to catch. Separately, a connection that closed with rows still queued left them in the
engine, and the node refused every later Voxtral connection until a restart (found on aff68,
2026-10-23). Both were fixed in aff70: the salt, and leftover rows dropped at a new connection.

## The bundle

The manifest's `[model]` table becomes a list, one entry per model the bundle carries; the
identity stays the hash of the one manifest. A model with an audio encoder carries a
`[model.audio]` table on its entry:

```toml
[[model]]
name = "glm-5.3-flash"
bos = 154822
# …

[[model]]
name = "voxtral-small-24b"
bos = 1
eos = 2
vocab_size = 131072

[model.audio]
placeholder = 24       # the id each row fills
window = [128, 3000]   # mel bins × frames, f32: the AUDIO payload
rows = 375             # rows one window becomes
sample_rate = 16000
hop = 160
fft_window = 400
```

The bundle also carries the mel filterbank (`mel_filters.f32`, 128 × 201) so the API computes
the spectrogram without a library and both ends name one definition, as `vocab.bin` does for
decoding. Each model's edge files (tokenizer, template, vocabulary, filterbank) live under a
directory named for the model, so two models' files never collide.

## Versioning

Wire version 5: the model field in the hello, and the AUDIO verb. Text-only clients change only
their hello. A model without an encoder refuses AUDIO.

## Decided (rockem.lead's review, 2026-10-23)

1. **One verb per medium.** AUDIO now; IMAGE later gets its own value, payload size and manifest
   table. Verbs are reserved downward from `MAX − 2`, since a model may carry several encoders.
2. **f32.** 1.5 MB a window is about 1 ms on the LAN. bf16 would save nothing needed and would
   perturb the input that bring-up diffs against the reference.
3. **The node encodes on receipt,** on the host's CPU first (Voxtral is CPU-only for now); the
   encode overlaps the rest of the prompt arriving. Whether the encoder moves to a card is a
   measurement, not the protocol's concern.
4. **Greedy, only.** Voxtral here is the transcription model: temperature 0. Sampling stays per
   model and never exposed; chat stays GLM. Voxtral as chat would be a second model entry over
   the same weights, if ever.

## Order of work

1. The hello's model field and the manifest list first, where the API and node work meet.
2. Node (rockem.lead): a session per model, the hello, the store key, AUDIO's receive, the
   engine's embedding input for placeholders; then the Voxtral engine and its bundle recipe.
3. API (rockem.api): per-model bundle reading, the hello's model field, the spectrogram from the
   bundle's filterbank, the transcription prompt, `POST /rockem/v1/transcribe`.
