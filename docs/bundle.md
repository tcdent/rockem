# The bundle

Everything a node loads, and everything the edge needs to speak to it, in one directory.
The layout is `models/deepseek-v2-lite/src/layout.rs`, the builder is `bundle/`, and this
document and that module say the same thing; if they ever disagree, the code is wrong.

A bundle is built once from a pinned checkpoint and never edited. Its contents are a
function of the checkpoint revision, the recipe and the calibration text, and the
manifest names all three plus the hash of every file, so the manifest's own bytes
identify the bundle. That identity is what the wire checks at connect (`docs/wire.md`).

## Contents

| file | what | size (V2-Lite) |
|---|---|---|
| `manifest.toml` | what this bundle is: recipe, calibration, geometry, the model's name and the wire's constants, every file's sha256, the pinned checkpoint | ~1 KB |
| `experts.q4g128.bin` | the routed experts, one fixed-size INT4 record per expert in (layer, expert) order | 7,647,264,768 B |
| `dense.bf16.bin` | every non-expert tensor in bf16, one file, in `layout::dense` order | 2,623,265,792 B |
| `vocab.bin` | token bytes by id: a decode table | 1,081,491 B |
| `sink.f32` | the BOS token's attention-cache row for every layer, computed once at full precision | 62,208 B |
| `kernels.gfx1100.elf` | the GPU code object, carried as data and dispatched through the node's own queue | ~64 KB |
| `tokenizer.json` | the checkpoint's tokenizer, verbatim: the edge encodes with it | 4,607,451 B |
| `tokenizer_config.json` | the checkpoint's tokenizer config, verbatim: the chat template and the added tokens | 1,279 B |

Sizes are stated so a reader can check a file by `ls`; each one is a closed-form function
of the model constants and is asserted where the file is read.

Bundles on the dev host live under `bundles/`. Variants of one checkpoint that differ only
in recipe (`v2lite-q4rtn`, `v2lite-q4gptq`, the calibration experiments) have identical
dense weights, vocabulary, sink, tokenizer and kernels; only the experts file and the
manifest differ. Each directory holds its own copy; a bundle is whatever its manifest
lists.

## The manifest

```toml
recipe = "experts:q4g128-a8-gptq"
calibration = "calib.ids" # sha256:e23b5cd1…
record_bytes = 4595712
experts = 1664
layers = [1, 27]

[model]
name = "deepseek-v2-lite"
bos = 100000
eos = 100001
vocab_size = 102400

[files]
"experts.q4g128.bin" = "sha256:cb20c75f…"
"dense.bf16.bin" = "sha256:7cc79780…"
"vocab.bin" = "sha256:03a3e2c4…"
"kernels.gfx1100.elf" = "sha256:27d72f91…"
"sink.f32" = "sha256:758c6a55…"
"tokenizer.json" = "sha256:41f3bf64…"
"tokenizer_config.json" = "sha256:31181eaf…"

# checkpoint.toml
# repo = "deepseek-ai/DeepSeek-V2-Lite"
# revision = "604d5664dddd88a0433dbae533b7fe9472482de0"
# pinned = 2026-09-16
```

| field | meaning |
|---|---|
| `recipe` | the expert record format and how its integers were chosen: `experts:q4g128-a8-gptq` (calibrated) or `experts:q4g128-a8-rtn` (round-to-nearest, the twin kept for pricing what calibration buys). The dense weights, the vocabulary and the sink are not recipe choices and are not named |
| `calibration` | present for GPTQ: the calibration file's name and sha256. Absent for RTN |
| `record_bytes` | bytes per expert record |
| `experts` | number of records in the experts file |
| `layers` | `[first MoE layer, layer count]`: records cover layers `first..count` |
| `[model]` | `name`, the model crate this bundle was built by and is read with, and the constants the wire assumes, from that crate: the BOS and EOS ids and the vocabulary size. A client speaks the wire from these without reading the Rust. The context length is not here: it is the node's memory budget, chosen per card, and the node reports it in its wire header (`docs/wire.md`) |
| `[files]` | sha256 of every file in the bundle. The kernels entry is present with an empty hash when the bundle was built without `--kernels` |
| `# checkpoint.toml` | the model's `checkpoint.toml` verbatim, commented: repository, revision, pin date |

**Identity.** FNV-1a, 64-bit, over the bytes of `manifest.toml` exactly as written
(offset basis `0xcbf29ce484222325`, prime `0x100000001b3`). Because the manifest carries
every file's hash, the tokenizer's included, and the checkpoint revision, two bundles with
equal identities have the same weights, the same vocabulary, the same tokenizer and the
same constants. The node computes it at start and the client sends the one it expects in
the wire header. The node reads the manifest for nothing else; it opens each weight file
by its constant name.

## The files

**`experts.q4g128.bin`.** A stream of `experts` records of `record_bytes` each, one per
routed expert, in (layer, expert) order for the MoE layers; the byte range of expert `e`
in layer `l` is `layout::record(l, e)`. A record is four streams, each a multiple of 64
bytes so the f32 scales stay aligned and records tile:

| stream | contents | bytes (V2-Lite) |
|---|---|---|
| gate+up weights | `[2·D_FF][H]` INT4, gate rows then up rows fused | 2,883,584 |
| gate+up scales | one f32 per row per 128-column group | 180,224 |
| down weights | `[H][D_FF]` INT4 | 1,441,792 |
| down scales | one f32 per row per group | 90,112 |

The INT4 format is `common::q4`: symmetric, scale per group of 128 columns, nibbles
stored biased by +8 and tiled four rows at a time in the order the AVX2 kernel loads them.
The GPU kernels read the same bytes. Whether the integers came from RTN or GPTQ is the
recipe; the bytes have one layout.

**`dense.bf16.bin`.** Every tensor that is not a routed expert, bf16, back to back in a
fixed order: the embedding, the final norm, the LM head, then each layer's norms,
attention projections and FFN (dense gate/up/down for the first `layers[0]` layers; the
router and the shared experts after). Offsets are the constants in `layout::dense`, which
the builder writes by and the loader reads by, so the file has one definition and no
header.

**`vocab.bin`.** `[u32; VOCAB_SIZE + 1]` little-endian byte offsets, then the concatenated
UTF-8 bytes of every token in id order; token `i` is `bytes[off[i]..off[i+1]]`. The BPE
vocabulary is byte-level, so entries are raw bytes and a single token can end inside a
multi-byte character; a reader streaming text accumulates bytes and emits the complete
prefix. The two added tokens (BOS, EOS) are their literal strings. Ids past the
tokenizer's last (100,002 through 102,399 for V2-Lite) are empty: the model can in
principle sample them, and they decode to nothing. This is a decode table only; it holds
no merges and cannot encode.

**`sink.f32`.** `[NUM_HIDDEN_LAYERS][KV_ROW]` f32: one attention-cache position, every
layer's row, for the BOS token. BOS has no context, so its forward pass is a constant of
the model, computed at bundle build through the bf16 model at full precision. Every
sequence is seeded from it and computed from `FIRST_POS`; position 0 is never computed at
runtime. Not a recipe choice (`analysis.md` §5).

**`kernels.gfx1100.elf`.** The code object built from `models/deepseek-v2-lite/gpu`, with
no relocations and no scratch. The node copies it into executable memory and dispatches
its kernels through its own AQL queue. Named for the ISA it needs.

**`tokenizer.json`, `tokenizer_config.json`.** The checkpoint's files, byte for byte,
`layout::TOKENIZER_FILES`. The first is the byte-level BPE the edge encodes with, and the
source `vocab.bin` is derived from. The second holds the chat template and the added
tokens' definitions; the edge reads the `chat_template` string from it and renders it with
a Jinja engine, the way the transformers library does, so nothing is extracted or
rewritten at build. What the edge does not need in it (`add_bos_token`,
`model_max_length`) it ignores: BOS is the wire's verb, and the context is the node's.

## Who reads what

The node reads the weight files, the sink, the vocabulary, the kernels, and the manifest
for its identity; it never opens the tokenizer files. The console, the node's text twin,
decodes with `vocab.bin`. The bring-up tools read the weight files against the reference.

The API service at the edge is the bundle's second reader, in a second language
(`api/rockem/bundle.py`). It needs
the manifest (identity, computed the same way; `[model]` for the wire's constants),
`tokenizer.json` to encode, `tokenizer_config.json` for the chat template, and `vocab.bin`
or `tokenizer.json` to decode. Its package is the bundle minus its weights: those five
files, about six megabytes, with the weight files still listed in the manifest but absent
from the directory. A reader verifies the hash of whatever it opens. Because the wire's
BOS is the token id 100000, encoding the rendered chat template with the tokenizer yields
BOS as the first id and the client sends what the tokenizer produced.

> "Being able to contain everything we need in a bundle, and then parse that with both
> ends of the system is a good idea."

## Building

```
bundle --snapshot <hf snapshot> --out bundles/v2lite-q4gptq --kernels <kernels.gfx1100.elf>
```

GPTQ against `models/deepseek-v2-lite/calib.ids` (`--calib <file>` names another),
`--rtn` for the round-to-nearest twin, `--experts-only` and `--dense-only` to rebuild one
half (the dense half includes the tokenizer files and skips calibration). About twenty
minutes on 32 cores for the calibrated build. The manifest is written last, after every
file's hash.
