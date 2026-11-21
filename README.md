# rockem

An inference stack built from the checkpoint up for one architecture: a mixture of
experts whose routed experts live in the DRAM of a Zen 3 (EPYC Milan) host and whose dense
half runs on AMD gfx11 (RDNA 3) cards, the two joined at every layer. Everything between a
model's weights and a token on a socket is in this tree, written for that shape rather
than ported to it: the transcribed forward, the fused kernels for the CPU and the card, the
quantizer, the bundle format, the sampler, the diskless image whose whole userland is the
node, the wire the node speaks, the HTTP edge in front of it, and the instruments that
measured every decision. The models it serves today are GLM-5.3-Flash for chat and Granite
Speech 4.1 2B for transcription, from one node; the method is what the tree is about.

## The method

**One realization, checked byte for byte.** Each op has one shipped realization and a
scalar twin written in the same form, and the two are compared byte for byte. The
transcribed model is checked against the reference implementation's activations edge by
edge, and every recipe is priced against the reference teacher-forced, in nats per
position, on held-out text. Generated text is never the measurement: a routing decision in
a mixture of experts flips on a 1e-7 difference and the token stream diverges from there.
A change to a kernel or to the node is gated the same way before it serves: a long SCORE of
a fixed prompt must be bit-identical to the served build's, a decode's speculative path
must follow the served build's fingerprint step for step, and a code object is refused if
any kernel asks for scratch or for more registers than its grid can hold resident.

**Hard-code the model.** `config.json` is transcribed into constants at build time, each
model's fused per-layer kernels live next to the model rather than in a toolkit, and what
is learned about a model becomes a constant of it: the first token's state is computed once
at bundle build and shipped, the sampling recipe is the checkpoint's own, the calibration
text is a file in the model's directory. There are no flags for these, and there is one
path through the code, not a fast path with a fallback beside it.

**No runtime between the node and the hardware.** The node talks to the cards through the
kernel's KFD interface directly, with its own AQL queue per card, its own code-object
loader and its own fences: no ROCm runtime, no HIP. The kernels are Rust, compiled by
rustc's LLVM for `gfx1100`, and are built, measured and disassembled with the same
toolchain as the host code. Launches happen only at synchronization points; a card waits on
a word in host memory for the host's expert rows rather than being told by a packet, and
the host waits on the card's completion the same way. Every kernel takes a token count: a
prompt goes through in steps of up to 1024 and a generated token is a step of one to four,
through the same code.

**The experts are the host's.** A mixture of experts streams its routed experts from DRAM
once a token, INT4 at a group of 128, through a thread pool pinned a thread per core and
placed by NUMA node, at the measured bandwidth of the box's memory channels; the dense half
of each layer and the attention state are the cards'. What the two halves hand each other
each layer is a few kilobytes of activations and a word. The most-picked experts are held
resident in the cards' spare VRAM and summed with the host's rows, and a learned predictor
prefetches the next layer's likely experts into the CPUs' L3 while the cards run the layer
before. Context is the host's too: every computed position's attention rows stay in DRAM,
and a request reuses the longest cached prefix transparently, so a conversation's next turn
resumes where the last one ended.

**Text stays at the edge.** The node reads and writes token ids and nothing else, plus a
log-probability per id when asked to score, plus audio windows for a model with an encoder.
The wire is a stream of 32-bit words with no framing of its own; the console is the same
stream in text. The tokenizer, HTTP, chat templates, tool-call parsing and the request
queue live in an API service on the cluster; the node never learns a character.

**Measure on the box, before and after, and write it down.** Every change meant to be
faster was measured on the production machine against the step it replaced, through the
node's own instruments: a row per step with every phase's time, a row per kernel launch on
the cards' timelines, and a statistics channel the kernels stamp from inside. The record of
what paid, what gave nothing and why lives with the code: `docs/queue/` holds one numbered
entry per idea with its status, and `docs/metatron-cpu.md` and
`docs/expert-cache-architecture.md` hold the box as measured, down to the firmware knobs
and the L3's replacement behaviour. A gain not taken keeps its measured size on the shelf.

## The operating system

The node is an appliance, diskless and stateless. The machine has no operating system of
its own: it PXE-boots, and the image it receives is one OCI artifact pulled from a registry
under a single tag — the kernel, an initramfs, the kernel command line, and a UEFI stage.
Nothing is installed and nothing persists; a reboot is a clean machine, and a deploy is a
build, a publish and a reboot.

- **The initramfs is the node.** `os/initramfs/init` is a short shell script: it mounts
  devtmpfs, proc, sysfs, resctrl, hugetlbfs and a tmpfs, brings the LAN up, mounts the
  weights read-only over NFS, starts the thermal daemon, and `exec`s the node as PID 1.
  There is no init system, no service manager, no login, no package manager, no writable
  filesystem; the whole userland is the node, the thermal daemon, a metrics agent for the
  cluster's monitoring, and busybox. Everything the
  node knows at runtime — the weights in hugepages, the context store, the resident experts'
  counts — is in DRAM and VRAM and is rebuilt at boot.
- **The firmware is part of the image.** `knobs.efi` (`os/firmware/`) is chained ahead of the
  kernel on every boot: it reads the firmware settings the tree names in `knobs.toml` out of
  the UEFI setup store, writes the ones that differ, and resets the box if it wrote
  anything. NUMA mode, the fabric's power states and the rest are a line in a file, a commit
  and a publish, with the running board's identity checked before anything is written.
- **Nothing secret is in the image.** The one credential the box needs, the thermal daemon's
  BMC login, reaches it over a second NFS export served to that host's address alone and
  mounted read-only at `/etc/rockem`. The image can be published anywhere.
- **The kernel is pinned, current stable, and ours where it has to be.** `os/pins.toml` pins
  every external input to a release — the kernel tarball and its hash, the configuration
  baseline, the GPU firmware blobs, the builder image — and a release's manifest records
  the hashes actually used. The configuration comes from a working machine's `lsmod`
  folded into `localyesconfig`; the one patch is the Super I/O driver for the board's fan
  and thermistor chip.
- **The box is an appliance, not a Linux system.** The kernel's ticks, threads and
  interrupts are kept off the cores the expert pool owns; the node's own allocator never
  unmaps; the thermal daemon holds each zone to a setpoint that moves slowly, with the fans'
  hand-back on its exit. The console is the kernel log over netconsole to the dev host, and
  it is what the deploy script reads for the gates and the hardware watch.
- **One path to and from production.** `bringup/box/deploy.sh` builds the image from the
  tree, publishes it, reboots the box through its console, waits for the node with a bound,
  runs the gates and reads the instruments back; `restore_production.sh` puts the served
  build back after a bench boot. A node that panics is a kernel panic (it is PID 1), and the
  box reboots into the last published image.

## Shape

| path | what |
|---|---|
| `common/` | primitives with no model in them: GEMV inner loops, the INT4 q4g128 format, norms, rope, softmax, sampling, the byte copies and the allocator. `no_std`, inlined into kernels. The crate doc is the coding convention for the tree. |
| `models/glm-5.3-flash/` | the chat model: `config.json` → constants, the transcribed forward and its CPU realization, the fused gfx1100 kernels (`gpu/`), the expert path, the speculative draft, the bundle layout, and `analysis.md`, the numerical analysis each recipe decision is justified in |
| `models/granite-speech-4.1-2b/` | the transcription model: a CPU-only encoder and decoder served beside the chat model in the same node, fed audio windows over the wire |
| `models/<other>/` | the models the stack was brought up and prototyped on, kept whole as the record and the regression surface |
| `bundle/` | checkpoint → bundle: INT4 experts (round-to-nearest, or our own GPTQ calibration), the dense weights, the prefix state, the GPU code object, the tokenizer for the edge, a manifest with every hash and the wire's identity (`docs/bundle.md`) |
| `node/` | the node: PID 1 of the image; loads a bundle into hugepages, constructs the models, drives the cards through KFD, keeps the context store, serves the wire or the console; the probe that measures a box's ceilings at boot |
| `thermal/` | the thermal daemon: the CPU coolers through the Super I/O, the cards' fan tables through amdgpu, the chassis fans through the BMC, each zone held to a setpoint that moves slowly (`docs/queue/101-fan-control.md`) |
| `train/` | the training side, its own workspace: the layer backwards over the inference byte formats, the adapter optimizer and the round, for continual adaptation on the served weights (`train/README.md`) |
| `os/` | the image: the pinned kernel and its patch, the initramfs, the firmware-knobs UEFI stage, the pins, the release and publish scripts, QEMU for the first rung |
| `bringup/` | the bring-up loop: reference goldens, the forward drivers that diff every edge and price recipes, the gates (`gates/`), the box scripts (`box/`), per-op benches against measured ceilings, the expert-cache and L3 experiments, `evals/` |
| `api/` | the edge, Python, never on the node: reads a bundle, speaks the wire asynchronously, schedules requests over the nodes, parses tool calls per model, transcribes, and serves HTTP: rockem's own `/rockem/v1/generate` (`docs/api.md`) beside OpenAI- and Anthropic-shaped surfaces. `api/client/` is the Rust client for our own surface |
| `docs/` | `wire.md`, the protocol; `bundle.md`; `api.md`; `kernel-tuning-feedback.md`, how kernels get measured and what the instruments mean; `expert-cache-architecture.md`; `metatron-cpu.md`, the box as measured; `metatron-thermal-map.md`; `queue/`, the numbered decision log |
| `stories/` | the long-form log of a day's work, where one was written |

## Build and run

The operating system is distributed the way a container image is, and the machine netboots
it. `make release` stages a release — the kernel, the initramfs, the kernel command line,
the kernel's configuration and the firmware-knobs stage — and `make publish` pushes those
five files to a container registry as one OCI artifact (`application/vnd.rockem.os.v1`,
the build's commit as its annotation) under a single tag, `latest`. The production box has
no disk to install to: its firmware PXE-boots iPXE from the site's netboot server, iPXE
pulls the artifact's layers from the registry by that tag, runs the knobs stage, and boots
the kernel with the published command line and the initramfs. There is no other tag: the
registry keeps the most recently pulled tags only, and a tag the box does not follow is a
tag it can stop booting from. Rolling forward is a publish and a reboot; rolling back is a
publish of an older tree and a reboot; the machine's state is in the artifact and the
weights share, never on the machine.

```
make release PROFILE=dev                       # stage a release: kernel + initramfs (the node and the thermal daemon) + knobs.efi
make kernels-glm                               # the chat model's gfx1100 code object and its static report
BUNDLE=bundles/<name> make qemu                # boot the staged release under QEMU with a bundle shared in
make publish                                   # push the staged release to the registry as the one artifact the box boots

bundle --snapshot <hf snapshot> --out bundles/<name> --kernels <kernels.gfx1100.elf>
                                               # checkpoint → bundle (INT4 experts; --rtn for round-to-nearest)

node rockem.bundle=bundles/<name> rockem.mem=anon                              # console: ids in, text out
node rockem.bundle=... rockem.listen=0.0.0.0:7332 rockem.gpu=1 rockem.ctx=1048576
                                               # the wire (docs/wire.md), the cards, a 1M-position context
cd api && ROCKEM_BUNDLE=../bundles/<name> ROCKEM_NODES=<node>:7332 ROCKEM_MODEL=<model> uv run uvicorn rockem.app:app
                                               # the edge: HTTP in front of the nodes

bringup/box/deploy.sh <label>                  # production: build, publish, reboot the box, run the gates
cd bringup/evals && uv run python run.py humaneval --node <node>:7332 --bundle ../../bundles/<name>
                                               # a published benchmark against a served node
```

The node's command-line keys are documented at the top of `node/src/main.rs`; the image's
at the top of `os/initramfs/init`.
