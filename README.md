# rockem

An inference stack built from the checkpoint up, one model at a time, on hardware we
own. Model zero is DeepSeek-V2-Lite. Everything between its weights and a token on a
socket is in this tree: the transcribed forward, the kernels for the CPU and the card,
the quantizer, the bundle format, the sampler, a purpose-built Linux image whose whole
userland is the node, and the wire the node speaks.

## Intentions

**Understand every number.** Each op has one shipped realization and a scalar twin
written in the same form, and the two are compared byte for byte. Above that, the
transcribed model is checked against the reference implementation's activations edge by
edge, and every recipe is priced against the reference teacher-forced, in nats per
position, on held-out text. Nothing ships on the strength of looking right; generated
text is never the measurement, because a routing decision in a mixture of experts flips
on a 1e-7 difference and the token stream diverges from there.

**Hard-code the model.** `config.json` is transcribed into constants at build time, each
model's fused per-layer kernels live next to the model rather than in a toolkit, and what
we learn about a model becomes a constant of it: the first token's state is computed once
at bundle build and shipped (`analysis.md` §5), the sampling recipe is the checkpoint's
own (temperature, top-k, top-p, as transformers ran it), the calibration text is a file
in the model's directory. There are no flags for these.

> "This is a great example of us hard coding optimizations into our own flat
> implementation."

**No frills.** The node has no runtime, no framework and no dependencies. Kernel launches
happen only at synchronization points ("the GPU boundaries are wave level"). Every kernel
takes a token count: a prompt goes through in steps of up to 32 and a generated token is
a step of one, through the same code. The wire is a stream of 32-bit words with no
framing of its own. The console is the same stream in text. When something is slow, the
profile says where and the fix is usually loop order.

**Text stays at the edge.** The node reads and writes token ids and nothing else, plus a
log-probability per id when asked to score. The tokenizer, HTTP, chat templates and,
later, grammar compilation for structured output live in an API service on the cluster;
the node never learns a character.

## Shape

| path | what |
|---|---|
| `common/` | primitives with no model in them: gemv, INT4 q4g128 format, rmsnorm, rope, softmax, sampling, the linear algebra the quantizer needs. `no_std`, inlined into kernels. The crate doc is the coding convention for the tree. |
| `models/deepseek-v2-lite/` | the model: `config.json` → constants, the transcribed forward, fused kernels for x86_64 and gfx1100, the bundle layout, `calib.ids`, `analysis.md`, the numerical analysis every recipe decision is justified in, and `evals.toml`, the published benchmark numbers and every run of ours against them |
| `bundle/` | checkpoint → bundle: GPTQ-calibrated INT4 experts, bf16 dense weights, vocabulary, the sink state, the GPU code object, the checkpoint's tokenizer for the edge, a manifest with every hash and the wire's constants (`docs/bundle.md`) |
| `node/` | the node: loads a bundle, runs the model on the CPU or on the card through KFD directly (its own AQL queue, no GPU runtime, the router and the draw on the card, one wait per token), keeps every computed context in host memory and reuses prefixes transparently, serves the wire or the console |
| `os/` | the image: a pinned kernel, an initramfs that is the node, QEMU for the first rung and a VM with the card passed through for the second |
| `bringup/` | the bring-up loop: reference goldens, the forward driver that diffs every edge and prices recipes, per-op benches against a measured ceiling, the reference quantizer, and `evals/`, published benchmarks run against a served node through the edge's own client |
| `api/` | the edge, Python: reads a bundle's manifest, vocabulary and tokenizer (`rockem.bundle`), speaks the wire with the node's position budget mirrored (`rockem.wire`). The API service grows here; never on the node |
| `docs/` | `wire.md`, the protocol; `bundle.md`, the bundle; `kernel-tuning-feedback.md`, how kernels get measured |
| `stories/` | the log of how it was built, in order, with the decisions and the dead ends |

## Where things run

The dev host builds everything and runs the bring-up tools. QEMU boots the image with a
bundle shared in, the proxy for a production node. lasenora holds the card, a 7900 XT,
and runs the node with the whole model in VRAM through its own queue, one wait per token:
115 tokens a second generating on V2-Lite at a short context, prompts at 480 a second in
steps of 32, and a 100,000-position context (6.2 GB of cache beside the weights) that
retrieves a fact planted 60,000 tokens back.

## Build and run

```
make release PROFILE=dev                       # kernel + initramfs (the node) for a profile
BUNDLE=bundles/v2lite-q4gptq make qemu         # boot it with a bundle shared in

bundle --snapshot <hf snapshot> --out bundles/v2lite-q4gptq --kernels <kernels.gfx1100.elf>
                                               # 20 minutes on 32 cores; --rtn for the round-to-nearest twin

node rockem.bundle=bundles/v2lite-q4gptq rockem.mem=anon                       # console: ids in, text out
node rockem.bundle=... rockem.listen=0.0.0.0:7332                              # the wire (docs/wire.md)
cd api && ROCKEM_BUNDLE=../bundles/v2lite-q4gptq ROCKEM_NODES=<node>:7332 uv run uvicorn rockem.app:app
                                               # the edge: HTTP in front of the nodes (api/rockem/app.py)
node rockem.bundle=... rockem.gpu=1 rockem.ctx=100000                          # everything on the card; ctx = positions of cache

forward --snapshot <hf snapshot> --goldens bringup/goldens/legal --bundle bundles/v2lite-q4gptq --q4-layers all --quiet
                                               # teacher-forced KL against the reference
cd bringup/evals && uv run python run.py humaneval --node <node>:7332 --bundle ../../bundles/v2lite-q4gptq
                                               # a published benchmark against a served node; the number lands in models/<name>/evals.toml
```

## Where it stands

INT4 experts, calibrated by our own GPTQ against 257 positions of our own text, with the
first token's state shipped at full precision: 0.015 nats per position from the f32
reference on a held-out legal clause, 0.028 on Roman history. The expert kernels on the
card run at 0.86 of the measured memory ceiling and are bit-identical to the CPU path.
The numbers, how they were measured and what they decided are in
`models/deepseek-v2-lite/analysis.md`.
