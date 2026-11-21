# rockem

An inference stack built from the checkpoint up, one model at a time, on hardware we
own. Everything between a model's weights and a token on a socket is in this tree: the
transcribed forward, the fused kernels for the CPU and the card, the quantizer, the
bundle format, the sampler, a purpose-built Linux image whose whole userland is the node,
the wire the node speaks, the HTTP edge in front of it, and the instruments that measured
every decision.

Model zero was DeepSeek-V2-Lite, on one card. The model in production is GLM-5.3-Flash
(45 layers, a mixture of 288 experts with 8 routed a token, linear attention on most
layers, a 1M-token context), served from a dual-socket EPYC box with two consumer cards,
with Granite Speech 4.1 2B beside it for transcription.

## Intentions

**Understand every number.** Each op has one shipped realization and a scalar twin
written in the same form, and the two are compared byte for byte. Above that, the
transcribed model is checked against the reference implementation's activations edge by
edge, and every recipe is priced against the reference teacher-forced, in nats per
position, on held-out text. Nothing ships on the strength of looking right; generated
text is never the measurement, because a routing decision in a mixture of experts flips
on a 1e-7 difference and the token stream diverges from there. A change to a kernel is
gated the same way: a long SCORE of a fixed prompt must be bit-identical to the served
build's, and a decode's speculative path must follow the same fingerprint.

**Hard-code the model.** `config.json` is transcribed into constants at build time, each
model's fused per-layer kernels live next to the model rather than in a toolkit, and what
we learn about a model becomes a constant of it: the first token's state is computed once
at bundle build and shipped, the sampling recipe is the checkpoint's own, the calibration
text is a file in the model's directory. There are no flags for these.

> "This is a great example of us hard coding optimizations into our own flat
> implementation."

**No frills.** The node has no runtime, no framework and no dependencies. It talks to the
cards through the kernel's KFD interface directly, with its own AQL queue per card: no
ROCm runtime, no HIP. Kernel launches happen only at synchronization points. Every
kernel takes a token count: a prompt goes through in steps of up to 1024 and a generated
token is a step of one to four (the token and its speculative drafts), through the same
code. The wire is a stream of 32-bit words with no framing of its own. The console is the
same stream in text. When something is slow, the profile says where.

**Text stays at the edge.** The node reads and writes token ids and nothing else, plus a
log-probability per id when asked to score, plus audio windows for a model with an
encoder. The tokenizer, HTTP, chat templates and tool-call parsing live in an API service
on the cluster; the node never learns a character.

**Measure before and after, and write it down.** Every change that was meant to be
faster was measured on the production box against the step it replaced, and the record
of what paid, what gave nothing and why is kept with the code: `docs/queue/` holds one
numbered entry per idea, with its status, from the first to the hundred-and-second.

## How GLM-5.3-Flash is served

The box is two EPYC 7R13 (96 cores, 512 GB of DRAM) and two RX 7900 XT. The dense half of
every layer — the linear-attention (KDA) recurrence, the MLA attention with its sparse
indexer, the hyper-connection mixing, the shared expert, the head — runs on the cards,
split by layer between them. The 288 routed experts a layer, INT4 at a group of 128, live
in host DRAM and are streamed by a thread pool pinned a thread per core and placed by
NUMA node; the card waits on a word at each layer for the host's rows and sums them with
the rows of the experts it holds resident in spare VRAM (the most-picked, loaded a few per
step, with a learned predictor prefetching the next layer's picks into the CPUs' L3). A
token is generated with the checkpoint's own multi-token-prediction layer as the draft:
up to three drafts verified in one step, accepted or rejected by rejection sampling, so
the output distribution is the model's. Every computed context stays in host memory and
a new request reuses the longest cached prefix transparently, resuming a conversation's
next turn without recomputing it; a prompt is prefilled in chunked steps with two chunks
in flight across the cards and the host.

Where it stood when this was written: a long conversation's turn (an 18k-token prompt
cached) generates at ~25 ms an emitted token, about 40 tokens a second a stream; a cold
18k prompt prefills at ~2 ms a token. The whole node, the thermal daemon that holds the
box's fans to slow setpoints, and the kernel boot as one image from an OCI registry over
PXE, with the firmware's knobs applied by a UEFI stage ahead of the kernel and the one
secret the box needs (its BMC credential) reaching it over an NFS export served to its
address alone.

## Shape

| path | what |
|---|---|
| `common/` | primitives with no model in them: GEMV inner loops, the INT4 q4g128 format, norms, rope, softmax, sampling, the byte copies and the allocator. `no_std`, inlined into kernels. The crate doc is the coding convention for the tree. |
| `models/glm-5.3-flash/` | the production model: `config.json` → constants, the transcribed forward and its CPU realization, the fused gfx1100 kernels (`gpu/`), the expert path, the speculative draft, the bundle layout, and `analysis.md`, the numerical analysis each recipe decision is justified in |
| `models/deepseek-v2-lite/` | model zero, kept whole: the single-card path, its GPTQ calibration, `analysis.md`, and `evals.toml`, published benchmarks run against it |
| `models/granite-speech-4.1-2b/` | the transcription model: a CPU-only encoder and decoder served beside GLM in the same node, fed audio windows over the wire |
| `models/voxtral-small-24b/` | the first transcription model, superseded by Granite for speed; kept as the record |
| `bundle/` | checkpoint → bundle: INT4 experts (our own GPTQ or round-to-nearest), the dense weights, the prefix state, the GPU code object, the tokenizer for the edge, a manifest with every hash and the wire's identity (`docs/bundle.md`) |
| `node/` | the node: PID 1 of the image; loads a bundle into hugepages, constructs the models, drives the cards through KFD, keeps the context store, serves the wire or the console; the probe that measures a box's ceilings at boot |
| `thermal/` | the thermal daemon: the CPU coolers through the Super I/O, the cards' fan tables through amdgpu, the chassis fans through the BMC, each zone held to a setpoint that moves slowly (`docs/queue/101-fan-control.md`) |
| `train/` | the training side, its own workspace: every GLM layer's backward over the inference byte formats, the adapter optimizer and the round, for continual adaptation on the served weights (`train/README.md`) |
| `os/` | the image: a pinned kernel with our Super I/O driver patch, an initramfs that is the node, the firmware-knobs UEFI stage, the release and publish scripts, QEMU for the first rung |
| `bringup/` | the bring-up loop: reference goldens, the forward drivers that diff every edge and price recipes, the gates (`gates/`: bit-identity of SCORE and the decode fingerprint, the kernel report that refuses an object whose grid cannot be resident), the box scripts (`box/`: deploy, restore, the saturation bench), per-op benches, the expert-cache and L3 experiments, `evals/` |
| `api/` | the edge, Python, never on the node: reads a bundle, speaks the wire asynchronously, schedules requests over the nodes, parses tool calls per model, transcribes, and serves HTTP: rockem's own `/rockem/v1/generate` (`docs/api.md`) beside OpenAI- and Anthropic-shaped surfaces. `api/client/` is the Rust client for our own surface |
| `docs/` | `wire.md`, the protocol; `bundle.md`; `api.md`; `kernel-tuning-feedback.md`, how kernels get measured and what the instruments mean; `expert-cache-architecture.md`, the host expert path and the CPUs' caches as a programmed cache; `metatron-cpu.md`, the box's CPUs, memory, power and firmware knobs as measured; `metatron-thermal-map.md`; `queue/`, the numbered decision log |
| `stories/` | the long-form log of a day's work, where one was written |

## Where things run

The dev host builds everything and runs the bring-up tools. QEMU boots the image with a
bundle shared in, the first rung. The second rung is a machine with one 7900 XT passed
into a VM, where V2-Lite runs whole in VRAM: 115 tokens a second at a short context, prompts
at 480 a second in steps of 32, a 100,000-position context that retrieves a fact planted
60,000 tokens back. The production box runs GLM-5.3-Flash and Granite Speech, boots from
the registry, and is driven from the dev host: `bringup/box/deploy.sh` builds, publishes,
reboots it and runs the gates; `restore_production.sh` puts the served build back after a
bench boot.

## Build and run

```
make release PROFILE=dev                       # kernel + initramfs (the node and the thermal daemon) + knobs.efi
make kernels-glm                               # GLM's gfx1100 code object and its static report
BUNDLE=bundles/<name> make qemu                # boot the image under QEMU with a bundle shared in
make publish                                   # the staged release to the registry the box boots from

bundle --snapshot <hf snapshot> --out bundles/<name> --kernels <kernels.gfx1100.elf>
                                               # checkpoint → bundle (INT4 experts; --rtn for round-to-nearest)

node rockem.bundle=bundles/<name> rockem.mem=anon                              # console: ids in, text out
node rockem.bundle=... rockem.listen=0.0.0.0:7332 rockem.gpu=1 rockem.ctx=1048576
                                               # the wire (docs/wire.md), the cards, a 1M-position context
cd api && ROCKEM_BUNDLE=../bundles/<name> ROCKEM_NODES=<node>:7332 ROCKEM_MODEL=glm-5.3-flash uv run uvicorn rockem.app:app
                                               # the edge: HTTP in front of the nodes

bringup/box/deploy.sh <label>                  # production: build, publish, reboot the box, run the gates
cd bringup/evals && uv run python run.py humaneval --node <node>:7332 --bundle ../../bundles/<name>
                                               # a published benchmark against a served node
```

The node's command-line keys are documented at the top of `node/src/main.rs`; the image's
at the top of `os/initramfs/init`.

## Where it stands

GLM-5.3-Flash's dense weights at the checkpoint's precision and its experts at INT4,
round-to-nearest, serving the 1M-position context the design is held to. ~25 ms an emitted
token on a long conversation and ~2 ms a prompt token, both measured on the production
box with the instruments in `docs/kernel-tuning-feedback.md`. V2-Lite: 0.015 nats per
position from the f32 reference on held-out legal text with our GPTQ, expert kernels at
0.86 of the measured memory ceiling, bit-identical to the CPU path. The numbers, how they
were measured and what they decided are in each model's `analysis.md` and in `docs/queue/`.
