# 2026-09-16 — from an empty repo to a bit-identical GPU kernel

A running log, appended as things happened. Not polished on purpose.

## Morning: picking up where the other session left off

Restored from the localai session's transcript. That session had built the OS side of
`~/rockem` (kernel 7.2.6, initramfs, node skeleton, QEMU boot in 11 s) and had ended
mid-sentence: I was about to write a memory when tcdent interrupted to have me look up
my own session id so the work could move into this directory as its own thing. Did that,
carried the five prior memories over, wrote the one that got interrupted (the rule that
the lasenora VM rung ends the moment the repo would need a virtualization-only workaround).

The plan from last session said GPT-2 first as a plumbing fixture, then DeepSeek-V2-Lite.
tcdent: "I probably trust you more on the DeepSeek V2 Light case." So V2-Lite is model
zero and GPT-2 never happens. Good — fewer throwaway ops.

## The contract question

I had sketched a TOML op inventory with a Python generator. When asked to explain it I
realized I couldn't defend it: the generator would emit const tables and a manifest, nothing
else, and Rust already has a place for constants. Recommended dropping it. tcdent agreed
and added the better idea: parse the model's own `config.json` instead of retyping numbers.
So `build.rs` does a dumb transcription of config.json into consts and anything derived is
written next to a comment. "Don't go overboard on the JSON parsing" — it's a 60-line
string search, no serde.

## Goldens and the forward

Forked an agent to write the HF golden-capture script while I wrote the ops. It came back
with two transformers-4.56 shims (the pinned 4.33-era modeling file calls a cache method
that no longer exists, and eager attention asserts a mask that 4.56 stops passing at
q_len=1) and a self-test on a tiny random model. The 31 GB checkpoint downloaded in two
minutes, which I did not expect. The real golden capture ran in f32 on this box in about
five minutes and decoded " Paris.\nThe capital" — the oracle works.

Wrote the transcribed forward in Rust: rmsnorm, bf16 linear, rope, softmax, top-k,
swiglu, absorbed MLA decode. bf16 weights straight out of the mmap'd checkpoint, because
the oracle was using 63 GB of RAM at the same time and a second f32 copy wouldn't fit.

First diff run: embedding exact, every position-0 edge within 1e-6 relative through all 27
layers, three of five argmax matched — and everything at position > 0 was wrong. That
pattern says rope. Read the oracle's `apply_rotary_pos_emb` in full instead of the docstring
and there it was: DeepSeek de-interleaves the 64-wide vector into even/odd halves before
the llama-style rotation and never permutes back. One new op, `deinterleave_rotate_half`,
and the forward matched the oracle to 5e-5 on every edge with five of five tokens.

Lesson I'd already been told and had to relearn: read the whole function, not the part
that looks standard.

## The packed recipe

Wrote the bundle builder: round-to-nearest INT4, group 128, absmax/7, packed through the
same `q4::pack` the kernel reads. 1,664 experts in 60 s, 7.6 GB, manifest with sha256 and
the recipe string. Then the forward with packed experts: five of five tokens still, but
per-expert errors up to 46 % in max-norm. Bug or quantization? Cross-checked in numpy on
the golden inputs with an independent RTN: 10–24 % per expert on this model. Same range.
Not a bug. INT4 RTN is just that lossy per expert here and the argmax survives anyway.
This is exactly the A/B the recipe-string idea was for.

Nice moment: the first "faster" step. The scalar bf16 GEMV was 2.7 s/token; a 40-line
AVX2/FMA realization dropped it to 0.6 s with the diff table unchanged. That's the ladder
the whole design rests on, walked once.

## "You might want to abstract slightly"

tcdent noticed the diff dumps were going to be the daily loop and suggested pulling the
repeated pattern out. He was right sooner than he knew: the bundle builder had already
duplicated the checkpoint loader. Made `bringup/harness` — checkpoint lookup, goldens,
the worst-per-edge report, and a bench that reports GB/s against a ceiling measured in
the same run. The expert kernel bench came out of that in 120 lines: 0.71 of streaming on
one thread, 0.94 at sixteen, unpinned on a Naples box.

## Where does a kernel live?

Then the conversation that changed the tree. tcdent: are we building a general toolkit?
I had put every op in one `ops/` crate with const generics, which is exactly how a toolkit
starts. The honest reading: numbers-as-const-generics is fine, but `P` for the rope width
was a structure difference dressed as a number, and a flat shared crate invites the next
model to add an option instead of a file. Proposed per-model directories; he pushed it
further — "each layer of the model can have what is essentially a kernel for it" — and
asked why model ops sat in `ops/` rather than next to the model. No good reason: it was a
split by toolchain, which we'd rejected earlier. And GPU inlining needs no macro, the
`amdgcn` target is ordinary rustc.

Restructured: `common/` for primitives with no model in them, all inline; the model
directory is the crate, holding config.json, the oracle file, the consts, and fused
per-layer kernels — `attn::step`, `ffn::dense`, `ffn::moe`, `expert::forward`. The `ops`
and `model` crates are gone. Diff and bench unchanged after the move. `attn.rs` is 119
lines and is the whole attention half-layer.

## The first GPU kernel

"Do you want to play around with some kernels now?" Yes. Picked the expert GEMV: same q4
bytes as the CPU, integer accumulation, no WMMA — the placement-invariance claim made
concrete.

Wrote a wave-per-tile version: lane owns 8 bytes of a row per group, three `ds_bpermute`
steps sum the row's eight lanes per group. Compiled to a 6.7 KB code object on the first
try (46 VGPRs, wave32). Ran on lasenora through the same ctypes-over-HIP path the last
session proved: right magnitude, wrong values.

Suspected the bpermute addressing (byte address vs lane index). Probe kernel: byte address
was right. Probe two dumped per-lane partials, the 8-lane sum, and the raw loads for tile
0 group 0 against a host recomputation: all exact. Which left the flush. And there it was:
I'd subtracted the +8 nibble bias per element *and* subtracted the 8·Σa term in the flush.
The scalar path does the former, the AVX2 path the latter; I'd written a kernel that did
both. Fixed to the AVX2 form, and while there made the scalar reference use the identical
flush expression so all three could be compared for equality rather than tolerance.
Tests tightened to `assert_eq!`. Result: **scalar, AVX2 and GPU produce the same bytes
from the same record.** That was the point.

Then speed. 17.7 µs per expert, 0.34 of the 760 GB/s ceiling: latency-bound, 16
sequential group steps per wave and two waves per SIMD. Made the kernels grouped over
experts (one launch for all six hits) and prefetched all 16 group words per lane. The
prefetch array spilled to scratch — 264 bytes of private segment, and the single-expert
case got *slower*. The eleven-group down kernel had unrolled fine; sixteen was past the
threshold. Rather than fight the unroller, changed the decomposition: each lane owns a
(row, group) cell and reads its 64-byte line whole, so there's no per-group cross-lane
sum at all, and the in-order group sum happens once per tile with one bpermute per group.

No scratch, 71 VGPRs, still bit-identical. 64 experts stream at 666 GB/s, 0.88 of the
ceiling; six experts — what a layer actually issues — at 0.67, with the two 2.5 µs launch
floors a visible slice of that. Single expert is 39 µs and latency-bound; not the design's
case, but the q loads per word are the likely reason and LDS staging would fix it.

Time from "let's play" to bit-identical-and-0.88: about ninety minutes, two probes, one
real bug that was mine.

## Evening: the optimization ladder

tcdent: "keep pushing through the optimization … collect some numbers of our baseline
versus our improvements." So: one launcher, one vectors set, a number after every rung.

First made the real case real. `dump-expert --goldens … --layer 5 --step 0 --token 3`
takes an actual token's post-norm hidden state and its six routed experts from the
oracle, runs them through the scalar path, and the launcher runs the same six on the GPU
with per-expert activations for the down projection (the bench had been using stride 0).
Six real experts, both GEMVs, bit-identical. And the recipe's cost on real activations:
the weighted routed sum lands 12 % off the oracle's f32 experts in max-norm, per expert
11–25 %. Same as the numpy estimate said it would be.

| rung | what | 1 expert | 6 experts (a layer's hits) | 64 experts | of 760 GB/s @64 |
|---|---|---|---|---|---|
| v1 | per-group 8 B/lane, 3 bpermute per group | 17.7 µs | — | — | 0.34 (single) |
| v2 | + prefetch all 16 group words | 41.7 µs (scratch spill) | 16.4 µs/expert | 11.9 µs/expert | 0.51 |
| v3 | lane owns a (row, group) 64 B cell; in-order sum once per tile | 43.5 µs | 10.8 µs/expert | 6.8 µs/expert | 0.89 |

v3's single-expert time is the tell: a wave's lane does 16 activation loads per 8-byte
weight word, 128 per cell, every one a global load. That's the latency chain. Rung four
stages the expert's quantized activations in LDS once per workgroup.

**v4 — LDS staging of the activations.** Two lessons, neither the one I was after. First,
`s_barrier` is `IntrNoMem` to LLVM: it is not a fence, and one tile in 64 read LDS before
another wave's store had landed. The fix is what HIP's `__syncthreads` emits — `s_waitcnt
lgkmcnt(0)` then `s_barrier`, in one `asm!` block so it is a compiler barrier too. Second,
it was *slower*: 0.72 of the ceiling at 64 experts (from 0.89), and the single-expert time
did not move at all. So the activation loads were never the latency chain. Wrong theory,
measured, discarded.

What the single-expert number is actually saying, on re-reading the access pattern: a lane
reads its 64-byte cell as eight 8-byte loads. Each load *instruction* therefore touches 32
different cache lines across the wave, and there are eight of them per cell. The memory
pipeline pays per line request. v5 remaps so one load instruction reads 512 contiguous
bytes across the wave.

**v5 — wave-contiguous loads.** Each load instruction now reads 512 contiguous bytes
across the wave (`global_load_b128` per lane), quads of lanes share a cell, two cross-lane
adds finish the group sum. Correct, coalesced… and the single-expert number was *still*
40 µs. Three variants with wildly different inner loops and one flat number means the
number isn't the kernel. It was the launcher: a device synchronize around every pair, and
the first launch into an idle queue costs ~25 µs of wake-up. Batched 32 pairs back to back
inside one sync window — which is how the token loop will issue them anyway — and the
single expert dropped to 14.9 µs. The 16-entry term array had also spilled (68 B); summing
in group order as steps complete removed it.

Then the 6-expert row read **1.09 of the ceiling**. Six records are 27.6 MB and the 7900
XT has an 80 MB Infinity Cache, so repeated pairs over the same six never touched VRAM.
The launcher now rotates through every disjoint set of six in the layer. The 64-expert row
(294 MB) was honest all along: 0.93.

**v6 — fuse.** Gate and up tiles in one wave with SwiGLU in the epilogue, and the down
kernel quantizing `h` in its prologue with wave reductions (same `round`, same formula).
Bit-identical on both halves, `h` never leaves the card — and slower, because pairing the
tiles halved the waves in flight and spilled again. Moved the SwiGLU across the launch
boundary into the down prologue instead: up kernel back to one tile per wave, no scratch.
Same fusion principle (SwiGLU never gets a launch), twice the parallelism.

**v7 — `v_dot4_i32_iu8`.** The biased-nibble layout was chosen for `vpmaddubsw` on the
CPU: one unsigned operand, one signed. gfx11's `sudot4` is the same pairing, four bytes at
a time. Four dot4s per 8-byte word instead of ~40 scalar ops. Integer, so still exact.

| rung | 1 expert | 6 experts, VRAM | 64 experts | of 760 @64 | note |
|---|---|---|---|---|---|
| v3 (sync-bounded timing) | 43.5 | 10.8 µs/expert | 6.8 | 0.89 | launcher was measuring queue wake-up |
| v5 batched timing | 14.9 | (1.09 — cache-warm) | 6.5 | 0.93 | |
| v6 fused, SwiGLU in down prologue | 17.6 | 9.6 | 7.4 | 0.82 | h stays on the card |
| **v7 dot4** | **12.8** | **8.8** | **7.1** | **0.86** | 11 µs is bandwidth + two launch floors |

Where it ends tonight: one layer's six routed experts cost ~53 µs on the card, 26 layers
≈ 1.4 ms per token if every hit is VRAM-resident, output bytes identical to the CPU path.
The remaining gap is the drain-and-ramp between the two launches, ~0.5 % of a token; a
grid barrier would remove it and isn't worth its complexity yet.

Things I got wrong today, for the record: the activation-load latency theory (v4), the
per-launch sync in the timing method, the cache-warm working set, and the double bias
subtraction. Each one was found by measuring something narrower than the number I cared
about — a probe, a batch, a rotation.

## Night: putting the forward inside the appliance

tcdent: "how do we get it so that the container is part of the forward pass?" — i.e. stop
running the forward as a dev tool on the host and run it as PID 1 in the booted image.
Which is also the answer to "can we stream output into tmux 1.3": the thing in that pane
should be the appliance's serial console.

Pieces, in order: the bundle grew a dense file (every non-expert tensor, bf16, at offsets
both the builder and the loader compute from the same constants — V2-Lite's non-expert
weights are only 2.6 GB, the experts are the other 28) and a vocab table so the console
prints text. The forward driver got `--dense` and reproduced the goldens from the bundle
file alone. The node grew a loader (hugetlbfs file, mapped, filled by the calling thread,
locked — anonymous memory only when asked for), the model construction, and a console
loop: a line of token ids in, the continuation out as bytes as they come. First run on
the host: "The capital of France is Paris.\nThe capital of France is Paris.\n…" at 3.6
tok/s single-threaded from INT4 experts. Greedy on a small model loops; the tokens are the
oracle's. He laughed at the inlined SHA-256 in the bundle builder. Thirty lines beat a
dependency the container would refetch on every build, and the manifest hash checks out
against sha256sum.

The QEMU rung gets the bundle over virtio-9p, mounted by one line in `/init` behind a
`rockem.bundle9p=` token, with six dev-profile kernel config lines. That is the whole
QEMU-specific surface so far, and it stays in the dev profile. The one host-side change
is `make kvm` (a `usermod`, with its comment) so the guest doesn't emulate AVX2 an
instruction at a time.

**It ran.** Two consoles, one prompt: `100000 549 6077 280 7239 317`.

Pane 1.3, the appliance: kernel 7.2.6, `/init`, node as PID 1, 10.3 GB pulled over the 9p
share into twelve 1 GB hugepages in 698 s under TCG (about 15 MB/s — the emulator
interpreting the guest's copy loop, and the moment tcdent drew the virtualization line:
"we don't necessarily want to build an OS that is tuned for virtualization"). Then, one
emulated AVX2 instruction at a time, "<｜begin▁of▁sentence｜>The capital of France is Paris".
Slow is fine. The image boots, mounts, loads, constructs, serves.

Pane 1.4, lasenora: the same node built for glibc, bundle on local disk (the 10 GbE copy
took 31 s), `rockem.experts=gpu`. The routed experts went into VRAM in 1.5 s — all of
them, V2-Lite's 7.6 GB fits in the 7900 XT — and the two grouped kernels from earlier
tonight ran under the transcribed forward through a 130-line HIP shim: "The capital of
France is Paris." ×6, **48 tokens at 8.7 tok/s**, prefill 0.66 s. Same bytes as the CPU
path, so the tokens are the oracle's. The attention, norms, router and shared experts are
still single-threaded bf16 AVX2 on the host, which is now the whole cost per token; the
experts that were half of it are gone into the card.

From "resume that session" to tokens out of the GPU inside the design's own layout: one
day. The `Routed` seam is the one new idea it took — the model file hands six indices and
an input to a realization chosen at construction, and the kernel never asks where the
bytes are.

## 2026-09-17, morning: the retrospective pass

tcdent asked for a look back before attention: "certain patterns that were perhaps not as
apparent in the beginning become apparent … candidates for standardization … from mostly
a stylistic and lexical perspective." A survey of the tree found three things already
consistent by accident (every kernel file has `Weights`, `Scratch` and one verb; `cfg` only
ever on declarations; the header comment is the spec), three inconsistencies (three
vocabularies for reference-vs-shipped, expert scratch built by hand in four places, the
test RNG copied three times), and two hazards (the Python launcher retyping Rust
constants; trace edge names agreeing by convention only).

Applied as one pass, no behavior change: the conventions are written in `common`'s crate
doc; `gpu` modules became `gfx11` (named for the ISA they need, like `avx2`); the bare
name is the shipped realization and `*_scalar` its twin, everywhere; `expert::ScratchBuf`
is the one owner of expert scratch per binary; `common::testing::Rng`; `dump-expert`
emits every launch constant into `meta.txt` and the launcher reads them; the forward
driver fails on a traced edge the goldens don't have. The code object came out byte-
identical after the rename (same sha as the bundle's copy), the forward still matches
the oracle, and the benches read the same numbers. Then: attention.

## Attention on the card

Three launches per layer, split only where the grid has to agree: `attn_proj` (input norm
as prologue, q and kv projections one wave per row), `attn_mla` (a workgroup per head:
latent norm, cache write and the de-interleaved rope folded in, absorbed scores over the
cache, softmax, latent output, value expansion), `attn_out` (o_proj with the residual add
as epilogue). One new common piece: a bf16 wave-per-row GEMV, plus `common::gfx11` for the
LDS, barrier and wave-reduction primitives every kernel was about to reinvent.

Validation was a dump of layer 5's real state at position 3 from the CPU forward — input,
cache, rope rows, and the five traced edges — and a launcher diffing each edge. First run,
all five within ~1e-6 relative. f32 summation order differs so this is tolerance, not
bytes; the integer kernels get bytes, the float ones get 1e-4 and land three orders under.
77 µs per layer at short context; 0.85 of the bandwidth ceiling at 2048 (where the cache
is read once per head — the all-heads-per-row version is the next rung on this kernel).

Then the same seam as the experts: an `Attention` realization chosen at construction. The
CPU one wraps the existing `step` and owns the cache; the GPU one holds the dense weights
and the cache in VRAM and moves the residual across PCIe twice per layer. Node on lasenora,
both realizations on the card:

    [6 prompt tokens in 0.41s, 48 generated at 14.78 tok/s]

3.6 → 8.7 → 14.8. What's left on the host is exactly what the number says: the shared
experts, the dense layer, the LM head and the router — about a gigabyte of bf16 per token
through one AVX2 thread. All of it is the bf16 GEMV piece that now exists on the card.

## The whole token on the card

"Pick your favorite next optimization." The one that finishes the move: the FFN half and
the LM head as launches, so a token leaves the GPU only as six routing indices. Three
kernel shapes, each a prologue + the bf16 row piece + an epilogue: `bf16_rows_norm`
(input norm → rows; serves the dense gate/up, the shared gate/up and the LM head),
`dense_down` (SwiGLU → rows → residual add), `shared_down` (the same, with the six routed
outputs folded into the residual add). Two more seams of the same shape, `FeedForward`
and `Head`, and the router stays on the host because sixty-four rows is nothing and the
indices are the thing every tier needs.

First run, everything on the card:

    [6 prompt tokens in 0.07s, 48 generated at 103.21 tok/s]

3.6 → 8.7 → 14.8 → 103. And the text changed: "The official language of France is
French. The currency of France is Euro…" where the CPU path repeats "The capital of
France is Paris." Both start with the oracle's first four generated tokens and split at
the fifth.

Suspected the new kernels. The launchers said no: the MoE half matches the CPU path to
9e-8, the dense layer to 2e-8, the head to 6e-7. Then the node itself, in four placements,
logits dumped per position and diffed against all-CPU:

    experts=gpu           0        0        0        0        0      ← bit-identical, every position
    attn=gpu           4e-7     6e-3     4e-3     2e-3     5e-3
    ffn=gpu            5e-4     6e-3     2e-2     2e-3     7e-3
    attn=gpu ffn=gpu   5e-4     4e-3     2e-2     2e-3     3e-2      (pos 0, 3, 5, 8, 9)

The integer expert path is exactly the same bytes end to end — the placement-invariance
claim, holding through 48 tokens on the real model. The float paths differ at 1e-6 per
layer, and a MoE router is a discontinuity: somewhere in 26 layers a 1e-6 nudge flips a
top-6 choice, different experts run, and the logits move by 1e-2. Each placement alone
still picked the oracle's token at position 9 (margin 0.16 under the recipe); both
together moved it by 0.75 and the runner-up won. Not wrong, not the same either.

Which is the design's own argument, now measured: bit-identity across tiers is the only
thing that makes "same input, same output" independent of cache state, and for the float
kernels it isn't there yet. Getting it means the CPU and GPU realizations of the bf16 GEMV
and of attention committing to one reduction order — the GPU's tree, which the CPU can
follow with an 8×8 transpose per 64 elements and no FMA contraction — the way the integer
path already commits to one flush expression. That's the next rung on the float kernels,
and it's a decision about cost, not a bug hunt.

## The analysis document

The seam conversation kept producing questions that had cheap measurements behind
them, so they became a document next to the model: `models/deepseek-v2-lite/analysis.md`.
What it took: a bf16 expert realization from the checkpoint, a per-layer recipe switch
in the forward driver, teacher-forced KL / top-1 / NLL in its report, per-layer
activation ranges, and a 91-position golden set (the 10-position one turned out to be a
hard tail: 0.30 nats where ordinary text costs 0.064).

What it found, in one afternoon of runs: the transcription is the oracle to 5e-11 nats;
the INT4 round-to-nearest recipe costs 0.064 nats per position on ordinary text; layer 3
alone carries most of that and layers 24–25 most of the rest, while the 23 other MoE
layers cost under a thousandth each; and the activation table says why — layer 3's FFN
is where the residual stream's scale gets set (absmax 12 after ≤ 1 before it), and 24–26
are where it grows fastest. The "keep the first layers wide" convention points at the
least sensitive layers of this model. One decision in ten in the router sits within 1 %
of flipping, one in a hundred within 0.1 %, which is the number behind every token
divergence we've seen and the reason quality is measured teacher-forced and
determinism is a contract.

tcdent: "at least it gives us a very interesting table that tells us practically what
areas of the model we are changing with our own knowledge … rather than just driving
blind." The next recipe writes itself from it: 3, 24, 25 wider than INT4.

## Chasing layer 3

tcdent: "be careful about targeting layers, because an MoE by definition is going to
activate expert layers depending on the context." So: four more golden sets in four
domains (code, dialogue, science, a contract), the per-layer sweep on all of them, expert
usage histograms, per-expert INT4 error for four layers, a noise probe that injects a
unit random perturbation at any layer, and a per-position switch in the recipe.

Layer 3 held across every domain — 22–49 % of the cost — and every simpler explanation
died in turn. Its experts have *lower* error than layer 10's. It is *less* sensitive to
random perturbation than layers 1, 4, 5. The error is not in the massive channels. Then
the mean-error-vector ratio came out at 0.85 where noise gives 0.10, and yet
consecutive positions' errors were nearly orthogonal, which is only possible if one
position dominates the mean. Position 0. Its INT4 error at layer 3 is 175; every other
position's is 0.3. The first token's FFN output at layer 3 has norm 2,467 against a
median of 3.1: this is where the attention sink's massive activation is made, and every
later token reads it. Keeping position 0 at full precision takes the recipe from 0.298
to 0.0066 on the hard prompt and halves-to-quarters it on every domain. With the sink
handled, the per-layer sweep goes flat: 26 contributions of the same order.

And the fix is free: BOS has no context, so position 0's state is a constant of the
model and recipe, computed once at bundle build and shipped as the initial cache rows.
A day of "which layer should be wide" ended with "no layer; one token; zero cost".
What it cost to learn: about forty forward runs and four numpy scripts, all of them the
kind of local, cheap, verifiable measurement the methodology note is about.

**Sink, shipped.** `sink.f32`: BOS through the bf16 model once at bundle build, 27 cache
rows, 62 KB, `recipe = "experts:q4g128-a8-rtn sink:f32"`. `Attention::seed` on both
tiers, the node and the forward driver start every sequence at position 1. Position 0
is no longer computed anywhere at runtime, and the driver reads the same 0.0316 nats
with the shipped rows as the experiment did with position 0 at bf16. tcdent's next
question — do more sinks emerge in long contexts, can we detect them by magnitude as the
row is written, and recompute that position at full precision — is filed in analysis.md
§9 as the future project; the per-position switch in the forward driver is already the
instrument for it.

## Does calibration already know about the sink?

tcdent: "go ahead and run another run to validate our theory … with and without the
sync precomputed … I do think this is deserving of some more quantification from us."
The theory going in (mine, after tcdent's first guess that calibration partly fits BOS):
GPTQ weights by the inputs, the expert input is post-norm, position 0 is 3 rows of 257,
so calibration should leave the sink alone and the two gains should stack.

Two GPTQ streams, one calibrated on everything, one with position 0 dropped; each
evaluated with and without the shipped sink on the two held-out sets. The forward driver
ran twelve times; the quantizer ran for most of a day, first oversubscribed on one box
(64 numpy threads per expert, 20 minutes a layer), then as four 8-thread workers per box
across elsenor and elluzbel (tcdent: "split two jobs between two essentially identical
machines"), about 4 s an expert.

The theory was wrong, and the way it was wrong is the interesting part. GPTQ calibrated
with position 0 present gets almost nothing from the sink (0.0158 → 0.0147 on legal);
calibrated without it, it needs the sink exactly as much as RTN does (0.0363 → 0.0172).
The mechanism took one numpy probe: the Hessian for an expert's *down* projection is
over its SwiGLU hidden states, and at the sink-creating expert of layer 3, three rows out
of 257 are 99.7 % of that Hessian's trace. GPTQ rounds that one matrix to be exact
along the sink direction, which is the sink fix, found by accident of every calibration
sequence beginning with BOS. The gate/up side never sees it — position 0's post-norm
input is ordinary — so the whole protection lives in one projection of one expert per
sink-making layer (layer 24 has its own, at 87 %).

So: the shipped sink is the explicit, free version of something calibration was doing
implicitly, and it makes the recipe's quality independent of what the calibration set
starts with. Calibration on top of it is worth about 25 % (0.0196 → 0.0147, 0.0316 →
0.0290), which is the diffuse rounding error the per-layer sweep said was left. Both
tables are in analysis.md §3 and §5; whether the calibrated stream becomes the recipe is
tcdent's call.

**Token zero, ratified.** tcdent: "This is a great example of us hard coding optimizations
into our own flat implementation … for now token zero as part of our implementation is a
go." The node already had it; what was still experiment-shaped was the edges: the recipe
string advertised `sink:f32` as if it were a choice, the forward driver computed position 0
unless asked not to, and "1" was retyped wherever a sequence began. Now `FIRST_POS` lives
in the model crate next to `BOS_TOKEN`, the recipe string names only the experts, the
driver seeds by default and `--no-sink` is the instrument that measures what the sink is
worth (0.298 vs 0.0073 on the hard prompt, reproduced after the change).

## Owning the quantizer

tcdent: "Verify a layer against the golden and we own our own quant path which has no
frills and max throughput and we understand it." Also: "we will want to be able to run
it on the cluster anyway so having it close to our existing deploy strat is excellent
prep." So GPTQ went where RTN already lived: the bundle builder. `common::linalg` got a
dot, an axpy and the three triangular passes from a Hessian to the upper factor of its
inverse; `common::q4::gptq` got the sweep next to `pack`, with a test that the identity
factor reduces it to RTN; the bundle builder got `--calib`, which runs `calib.ids`
through the bf16 transcription with a `Routed` wrapper that keeps every expert input, one
Hessian factor per layer for gate/up and one per expert for down. About 300 lines, no
python on the path.

Verification was the point. The reconstruction error GPTQ minimizes is a deterministic
number measurable outside it, so `quant_check.py` computes it for every expert of a
layer from the goldens' inputs: the Rust stream and the numpy reference agree to four
digits with the per-expert wins split down the middle, and both sit 24× (gate/up) and
14× (down) under RTN. Then the tolerance anchor: teacher-forced KL 0.0153 / 0.0275 on
legal / Roman against the reference's 0.0147 / 0.0290.

Speed took two passes. The first build ran 32 minutes; perf put 61 % in the block
update's axpy, and a padded-stride experiment killed the 4K-aliasing theory (25 %, not
25×). The real cause was loop order: every weight row re-read the 1 MB factor block from
memory. Sweeping each row's 128-column segment to completion in L1 and tiling the block
update so a slab of factor rows serves a tile of weight rows took the idle single-thread
sweep from 13.5 s to 1.1 s and the build to 17 minutes, byte-identical. It is still 9×
off the idle rate under 64 threads on a Zen 1 box with 512 KB of L2 per core, now in the
per-expert Cholesky; that is the next lever and it was not pulled tonight.

tcdent, mid-build: "Do you need a 10 min benchmark?" No — 70 seconds; the cap was a cap.
Two things on the radar from tcdent: Intel's AutoRound and llm-compressor's
MoECalibrationModule. The second turns out to be what `--calib` already does — every
expert sees every calibration position, not just the ones routed to it.

**Wired in.** tcdent: "I call that a major win my guy! Let's wire it up to our model
implementation." The builder now quantizes with GPTQ by default against the model's own
`calib.ids`, `--rtn` is the twin kept for pricing, and the shipped bundle is
`bundles/v2lite-q4gptq` (experts byte-identical to the verified stream; dense, vocab, sink
and kernels hash-identical to the RTN bundle's). On lasenora, all on the card, the hard
prompt that RTN + sink still answers with "The capital of France is Paris." six times over
now continues into the Seine and the museums, at 95 tok/s. Same kernels, same format,
different integers.

**Sampling.** The node was greedy. tcdent: "we do need to implement sampling… seed could be
baked into the build" and then "maybe we bake seed and temp with the model config. If we do
ever change that it would likely be per-model." The checkpoint had already answered the
temperature question: its `generation_config.json` says 0.3 (and top-p 0.95, not applied yet),
the same file BOS and EOS were transcribed from. So `TEMPERATURE` and `SEED` sit next to them
in the model crate, `common::sample` is a xoshiro generator and an in-place logits → token draw
(temperature 0 is greedy; the cumulative walk is f64 so a 102,400-wide tail is not lost), and
the node resets the generator per prompt. Two runs on lasenora, two prompts each, identical
tokens, 97 tok/s.

## The wire

tcdent, on I/O into the cluster: the tokenizer and HTTP live in a container on the Nomad
cluster, "which leaves inference as a super minimal byte stream perhaps binary with a very
light protocol definition." Then, on framing: "perhaps we rely on the network protocol TCP
to just handle that for us … building a protocol on a byte stream that is oriented
specifically towards streaming lets us keep this super simple." And on hearing there would be
no length fields: "You don't know how much you're speaking my language."

So `node/src/wire.rs` is a TCP stream of little-endian u32 words and nothing else. Ids
below the vocabulary are tokens; the model already owns two of the verbs (BOS starts a
sequence and re-seeds the sink, EOS ends a generation and is fed back like any token) and
the stream needs exactly one more, GENERATE, one reserved value. Abort is closing the
connection. A sixteen-byte header each way at connect, framed by position, carries the
version and an FNV of the manifest so a client built against another bundle is refused
before a word is exchanged. Nagle is off. The node's loop became a `Session` with three
moves — start, feed, generate — that the wire and the console both drive, the console being
the text twin: decimal ids in, a newline for GENERATE, token bytes out.

First run: netcat piped through xxd showed the header come back and the ids follow; a
thirty-line Python client saw the first token 55 ms after GENERATE and the rest at 97
tok/s, one four-byte word at a time; a wrong identity was refused. Structured output is
parked as a future verb: a grammar compiled to a token automaton at the edge, executed on
the node as a mask before the draw.

## KFD-direct

tcdent: "let's build like it's there already and let me know when you're blocked by it" — "it"
being the resizable BAR, the one memory path. So `node/src/kfd.rs`: `/dev/kfd` and the render
node, the process bound to the GPU's VM, memory allocated through KFD at the same address on
both sides, one AQL queue with its ring, index page, end-of-pipe buffer and a context-save
area sized by the numbers the kernel itself publishes in sysfs (and validates: the index words'
page must be exactly one mapping, the save area exactly the kernel's size plus its debug
area), the doorbell page mapped from the kfd fd, the code object copied into executable host
memory — the ELF has no relocations, the kernels no scratch — and a dispatch packet per
launch. Completion is a barrier packet naming a signal word the command processor decrements
in host memory; the host arms MONITORX on that line, re-checks, and MWAITXs with a timeout
that doubles as the hang watchdog. The structures came from the ROCm runtime's headers, the
ioctls from the kernel tree we build, and the validation rules from the kernel's own
`kfd_queue.c`, which is how three EINVALs became three one-line fixes.

First dispatch through our own ring, on lasenora: the expert pair ran, the barrier stamped
its timestamps, the host woke. Byte-identical to the HIP launcher on all 12,288 outputs (two
processes: a process that holds its own KFD context cannot also start the HIP runtime). A
sync round trip — doorbell to woken host — is 14.3 µs, of which 3.8 µs is the barrier packet
itself on the card (its system-scope release flushes L2); dispatches carry agent-scope fences.
Under HIP the same round trip measured ~25 µs.

Blocked, as asked: the kernel refuses host-visible VRAM outright on a small BAR ("Alloc host
visible vram on small bar is not allowed"), so the node's KFD path stops at its first weight
upload with "2501 MiB host-visible VRAM requested, 256 MiB visible: enable resizable BAR".
The launcher was verified with the six records in host memory instead, which is the bench's
placement, not the node's. The three model realizations on the launcher are written and
compiled; KFD is the node's GPU path whenever no HIP library is named, and HIP stays as the
twin until the first bit-identical token through KFD, after the toggle.

**BAR flipped, HIP retired.** tcdent enabled above-4G decoding and resizable BAR and rebooted
lasenora; the card came back with a 32 GB BAR 0 and bound to vfio-pci, because a modprobe
config claims it for a Windows VM at boot (the previous binding to amdgpu had been by hand).
Rebound, all 20,464 MiB host-visible. First run of the node through its own queue: the same
tokens as HIP, 73 tok/s. The gap was placement: the residual and the other crossing buffers
were coherent host memory, which the card reads uncached over PCIe — the LM head alone
re-read the 8 KB residual 12,800 times a token. Made them host memory the card caches, with a
system-scope acquire on every dispatch and the system-scope release on the barrier doing the
visibility: 115 tok/s, against 97 under HIP. Then the gate: 51 of 51 debug files identical
across the two launchers — every position's logits, every layer's residual, every routing
decision. `hip.rs` is gone (a copy in the session scratchpad until the tree is committed),
the node has one GPU path, `bench-kfd` times the launcher alone, and the static image node
builds with it. The 7.6 GB expert stream loads through the BAR in 1.5 s.

**The CPU side, against DRAM.** tcdent: "offload all of the experts of this model to CPU and
then we can get an idea for how close we're getting to theoretical peak." First attempt lied:
the expert bench's ceiling fell as threads were added, because threads drifted, memory was
wherever the page cache put it, and a 73 MB working set on a 32 MB L3 is half cache. So
`bench-dram` (private first-touched buffers, pinned threads, barrier-fenced passes) and the
same discipline in `bench-experts`. Lasenora's single-CCD 5700X streams 27–29 GB/s from one
thread and no more from eight, half the DIMMs' 57.6, the fabric's limit; the expert kernel
sits at 0.97–0.99 of that. Elsenor's four-die 7551P does ~16 GB/s per die and scales by die;
the kernel 0.96–0.99 of whatever the dies give. The node with everything but attention on one
CPU thread: 11 tok/s, which is 2.2 GB read per token at 24 GB/s — 0.85 of the box. The CPU
path is at the wall already, single-threaded; the wall moves on the 7R13, where one thread is
a tenth of a socket and the experts will want a thread per CCD.

## The context store

tcdent: "we leverage our large pool of DRAM to hold the caches … way faster than
recomputing … I don't need any kind of persistence." And, on naming them: "something
simple like a hash of the actual content." The design fell out of the sink: a cached context
is the cache rows of a transcript, the sink is the transcript `[BOS]`, and `Attention::seed`
just needed a length. DRAM is the truth and VRAM a cache of it: the attention kernel writes
each row to the card and to a host staging row (one extra store; reading VRAM back through
the BAR is the slow direction), the session copies it into the active context, and a swap-in
is one write through the BAR. Contexts are position-major runs of 62 KB, so a prefix is a
contiguous range, a fork is a memcpy, and a future NVMe demotion is a sequential write.
Identity is a running FNV over the id words, so every prefix has a name; the index maps a
key to (context, positions) and a hit compares the stored ids before trusting it. Matching
happens as ids arrive on the wire, which the node already fed one at a time; a fully cached
prompt recomputes its last position for the logits. Eviction is LRU by context under a
budget of positions, half the free memory by default. Nothing on the wire changed.

The gate: fresh runs against cache-hit runs, every per-position dump compared. On lasenora,
all on the card: a repeated prompt (51 of 51 files identical, prefill 0.00 s, "4 cached") and
an extended one (57 of 57, "5 cached"); on the CPU path the same. Two hundred lines in the
node, one argument on one kernel.

**Segments, transfers as ops, and a bug the gate caught.** tcdent: "make the segment we copy
from pluggable", "the NVMe drives don't ever need to be formatted and mounted … they can just
be memory we access over PCI", and "can we also think about this like we do inference? Bare
AVX ops or kernel ops for transfers." So a context's closed rows became a `Segment` trait with
one realization, `Dram`, the active context staying the store's one live region because the
kernel mirrors into it; an NVMe realization is a second one, and striping eight drives is
arithmetic over the same read. Transfers became ops: `common::copy::stream`, non-temporal
256-bit stores with a scalar twin and an alignment test, used by the fork and by every write
through the BAR, measured by `bench-copy` against DRAM and the link.

The first BAR numbers were 5 GB/s, and the cause was a fault per 4 KB page on first touch of
the mapping; `MAP_POPULATE` on the KFD mapping took it to 22 GB/s for 128 MB, near the PCIe
4.0 x16 ceiling, with `memcpy` and `stream` within a percent of each other there (the link is
the wall, not the store form). DRAM-to-DRAM is 14.5 GB/s on lasenora's single channel and 7.6
on elsenor.

Then the gate earned its keep. After the refactor the fresh runs and the cache-hit runs
disagreed from the first computed position on — and the *hit* runs were the correct ones.
`start()` had stopped seeding the sink, because seeding had moved into the prefix path, so a
sequence that matched nothing ran with whatever the card's cache held. One line put it back,
and the invariant is now stated where it lives: a sequence that matches nothing must be
identical to one that matches everything. 21 of 21 dumps on the CPU, 51 of 51 for a repeat on
the card, 57 of 57 for a prompt served from a cached prefix.

**The uploads.** tcdent: "can you look into the dense and expert uploads? This is one place
that our optimizations should generalize." Splitting the timer said the mapping took 0.00 s
and the copy ran at 5.9 GB/s — so `MAP_POPULATE` had been silently ignored on the driver's
mapping and every fault was landing inside the copy, one per 16 pages. `bench-copy` then
measured populating a BAR window on its own: 7.7 GB/s from one thread, 14 from two, no more
from eight; `madvise(MADV_POPULATE_WRITE)` refused. So every host-visible window is
populated at allocation by a few threads, once, and every copy after that is at the link:
2.6 GB mapped in 0.19 s and written at 22.1 GB/s, 7.6 GB in 0.54 s at 22.5. What generalizes
is the rule, not the number: a window is faulted in before it is timed or used, and the copy
op never pays for the mapping. The next lever is in the kernel we ship — the 16-page prefault
is a constant in ttm — and is noted, not pulled.

**The layout experiment, and what it exposed.** tcdent: "Are there any tricks we can use for
blocking and sequential access that push this to the edge?" and the rule: "whenever we have an
approach that maxes a particular op we use it everywhere that op is needed." The candidate
trick was one layout everywhere — the card's cache position-major like the store's, so a
swap-in is one copy. `bench-attn` timed per-token attention and the seed at 256 and 2,048
positions in both layouts, one constant, two ELFs. At 256 the layouts tie on the token and
position-major seeds 2.8× faster; at 2,048 position-major costs 11 % per token (rows at a
62 KB stride) and seeds 2.3× faster (6.7 vs 15.5 ms). A swap-in is paid once per context
switch and the token every token, so the break-even is about ten generated tokens, and the
card stays layer-major; the store stays position-major for the prefix. The swap-in was then
rewritten as gather-then-one-run per layer — the maxed BAR op in its long-run form — which
on lasenora is DRAM-gather-bound at 8.4 GB/s, the single channel again, and link-bound on a
box with channels. Gate: 51 of 51 dumps identical through the new seed.

What the experiment actually taught was in the other column: attention at a 2,048 context
costs 7.8 ms a token to read 127 MB of rows, 16 GB/s on a card that streams 760. The MLA
kernel is one workgroup per head, 16 workgroups on 84 CUs, a wave per position: at short
contexts that is the right shape and at long ones it is a fifth of the card idling. That is
the next kernel — positions split across workgroups with a reduce, the flash-decoding form —
and it moves a long-context token from ~13.5 ms toward ~9, which also changes the layout
question, so the layout is revisited after it, not before.

## Splitting attention

tcdent: "let's see what you can do about getting the attention reads faster because there's
got to be some room in there … we've yet to disprove our abilities." And, on the tuning
choices: note where a number was picked around an assumed mean and leave it open. The MLA
kernel became a workgroup per (head, chunk of positions): each chunk scores its positions,
keeps its own softmax max and sum, and leaves an unnormalized latent output in a partials
buffer; a new `attn_merge` kernel, one workgroup per head, rescales the chunks by
exp(m − M), divides by the total and does the value expansion. Inside a chunk the output
accumulation became eight independent streams, a wave taking every eighth position with
each lane owning sixteen dims, summed through LDS — the serial 2,048-iteration loop was
where the old kernel spent its time. The split policy is per launch: as many splits as fill
the card without a workgroup dropping below 64 positions, capped at 256 by the LDS; the
three numbers are labeled assumed means in `gfx11_consts`, with the measurements that
picked them (four splits beat one by 10 % at 256 positions; eight beat thirty-two by 9 % at
2,048). Every edge is within tolerance of the CPU realization; the store gate holds at 135
of 135; the mirror row is now diffed too.

| context | attention per token, before | after |
|---|---|---|
| 64 | ~2.9 ms | 2.61 ms |
| 256 | 2.97 ms | 2.75 ms |
| 1,024 | — | 3.34 ms |
| 2,048 | 7.81 ms | 4.08 ms |

The short-context token pays one more launch (112 tok/s against 115) and gains it back in
the split at 256. What the numbers also say: batched, the four launches cost 44 µs a layer
at position 3 and 111 at 2,047, but through the node — a host sync per layer to read the
residual back for the router — a layer costs ~100 and ~150. The host round trip is now a
third of a short-context token. The next lever is one sync per token: the router on the
card, the whole layer stack posted as one run of packets.

## One wait per token

tcdent: "Push on that next lead." The lead was the host round trip after every layer, and
removing it meant nothing of a token could need the host between the embedding and the
draw. Three things moved onto the card. The router: one workgroup that norms the residual,
runs the 64 router rows, does the softmax and the greedy top-k in one thread in exactly the
CPU's tie order, writes the experts' record offsets for the expert kernels, and quantizes
the activation to INT8 op for op as the CPU does, so the integers match bit for bit. The
residual: it never comes back; layer 0's step carries the embedding in, and the down
kernels mirror each layer's output into host memory for the trace, read once after the
wait. The draw: a `sample` kernel, one workgroup over the vocabulary, the CPU sampler's
twin — same max, same exponentials, 256 partials joined in f64 — so the host reads four
bytes. `State::decode` now takes a model and a draw and returns the token; `CpuModel` and
`KfdModel` are the two tiers, and the node has `rockem.gpu=1` and nothing else, the mixed
placements gone.

Two bugs the gates caught. Every packet had been acquiring at system scope, an L2
invalidate per kernel; now only the first launch after the host has written does, and the
launcher tracks that. And the sampler: making every position draw broke the store, because
a position served from the store is not computed — the fresh run and the hit run drew
different streams and the gate said 15 of 195. The fix is the right definition: the draw at
position p is a function of the seed and p alone, so how the prefix arrived cannot matter.
195 of 195 on the card, 24 of 24 on the CPU.

| | before | after |
|---|---|---|
| the node, all on the card, short context | 114 tok/s | 132 tok/s |
| a token on the card, position 8 | — | 7.5 ms: attention 1.9 and FFN 4.2 (6.8 interleaved as a token runs them), head and draw 0.7 |
| posting a launch, host side | — | 0.2 µs |

The layers cost 0.7 ms more interleaved than the two halves measured apart — attention and
FFN alternating their weight streams every layer — which is where the next measurement
starts. Cross-tier: the card's routing matches the CPU's on 22–27 of 27 layers per position and
the residuals agree to 1e-3, the known float-order flips. The FFN half is now the largest
term, 4.2 ms for 1.75 GB — 0.57 of the link's ceiling, five bandwidth-bound kernels each
paying its own tail and gap — and the bf16 dense weights are 2.2 of the 2.9 GB a token
reads: the shared experts alone 900 MB. That is a recipe question, not a kernel one.

## The timeline

tcdent: "I don't think quant below bf16 is suitable for the dense weights, but look at your
other measures." So the instrument first: every dispatch now carries a signal from a ring,
the command processor stamps its start and end, and after the token's wait the launcher
hands back the timeline — per kernel, the time inside it and the gap before it — with no
kernel touched. `bench-token` accounts a token with it, interleaved as decode runs the
layers and per half alone.

It named two things at once. The router, one workgroup, was the largest kernel in the
token: 49.5 µs a layer, 1.29 ms a token, to stream 256 KB — one wave per row is
latency-bound when it is the only workgroup on the card. And the gaps between packets, 3.8
µs each, 240 of them, 0.9 ms. The router lost every serial part: its rows are streamed by
all 256 threads at once, the softmax runs across 64 threads, and the greedy top-k became a
rank — each expert counts the ones that beat it, ties to the lowest index, exactly the
CPU's order — so the chosen six write their own slots. 49.5 → 22 µs. The shared experts'
rows kernel needs only the residual, so it launches without the barrier bit and overlaps
the routed experts' up: a negative gap in the timeline. An eight-deep load unroll in the
router measured worse and went back out.

| | before | after |
|---|---|---|
| a token on the card, position 8 | 7,197 µs | 6,150 µs |
| the node, short context | 132 tok/s | 144 tok/s |
| position 2,047 | 109 tok/s | 119 tok/s |

Left on the table, all measured: the gaps, 0.6 ms a token, 3.8 µs per barrier launch; the
router at 22 µs for what should be 5, cause not found; the FFN kernels at 0.5–0.8 of the
link, tails of small grids; and 0.3 ms from interleaving the halves, unexplained. The
cross-tier routing check still agrees on 22–27 of 27 layers per position through the
rank-based pick, and the store gate holds at 195 of 195.

## Statistics from inside

tcdent: "if we had a protocol where you had an address space that you could write
statistics into, and then inside kernels they could drop numbers into those registers …
it's just interesting to think how much we could extract from the system by just having
some naming conventions for registers that we write to." And on my worry about one
representative wave: keep a vector, average after the fact — "an average of each wave
throughout a full turn".

The instruction that made it near-free is the card's own real-time clock, readable from a
wave; on gfx11 that is a returning send-message, since `s_memrealtime` is gone, which the
first build found the hard way. The other is `dispatch_id`: a kernel knows the index of
its own packet in our queue, so a slab `[launch][slot][wave]` needs no argument and no
per-launch pointer — the launcher writes the slab's address once into a static in the
loaded image (the stats object's one relocation, which the loader now applies; the release
object still has none). `stat!(slot, value)` is a store by every wave into its own entry,
`stat_clock!()` a stamp, both expanding to nothing without the feature, and the release
ELF is byte-identical either way. Slot names live next to each kernel.

The first readout, a token at position 8: the router's 22 µs is rows 12.7, quantize 6.8,
softmax 3.9, pick 2.4 — the one-workgroup rows latency-bound, the quantize larger than it
should be. Attention's chunk kernel: prologue 9.4, scores 3.9, output 0.9 µs a wave. And
for free, the sink detector from the analysis doc's future list: the largest residual any
wave saw in each layer's down kernel, 1.4 at layer 0 rising to 154 at layer 26 for an
ordinary position — the number that read 2,467 at layer 3 for the sink. The overhead is
measured by the same timeline: about a microsecond per clock read.

**The next optimization, measured and declined.** The statistics said the router's quantize
was 6.8 of its 22 µs, and the obvious move was into the expert kernels' prologue, where the
down kernel already quantizes its own input. Built, and the expert kernels stayed bit-identical
to the scalar realization through it — the gate did its job — but the timeline said no: the
router lost 4 µs and every expert-up workgroup gained 6.5, the same work done 528 times
instead of once, and the token was 65 µs slower. Back out, with the reason in the kernel's
comment. What stayed: the router's softmax in one wave, 22 → 20.5 µs. Node 144.5 tok/s; the
dump tool now also writes the expert input as f32, for whatever wants it next.

**Why the quantize is a third of the router.** tcdent: "I was trying to figure out why the
quantization would be like 30% of the process." The channel's first clock could not say:
its stamps cost a microsecond each, and sub-stamping a 7 µs phase into three parts with
eight stamps read as 10 µs. So the clock became the shader cycle counter, one scalar
instruction, and the report infers each kernel's rate from the timeline. Then the ledger,
each entry built, measured through the same token bench, and kept only if the token
improved: the four byte stores per lane packed into one word (no change); the wave
reductions on the nightly's hardware reduce instructions (slightly slower); the two groups
each wave handles interleaved so one chain's latency hides the other's (no change); and the
same code timed on its first and second pass (the second slower, so not cold code). What
the numbers say is that a one-workgroup kernel pays every dependent step's latency
serially and nothing hides it; the quantize is about 11,000 cycles a group of 128 values
for work that is a few hundred instructions, and none of the four obvious causes is it.
The remaining candidate is where the stores drain, and the honest position is that a
2.5 % item has had its day. The router keeps the simplest form, 20 µs, and the ledger
keeps the next person from re-walking these four.

**Two reformulations, priced.** tcdent asked for the numbers behind the two ideas declined
above: what the parity-preserving one would save, and how often the parity-breaking one
would actually drift. Both were measurable in an hour, so neither was estimated. The
router's norm was stamped apart from its rows for one run (norm 4.8 µs, rows 10.0,
softmax and pick 4.8, quantize 7.6): the quantize is not waiting on the norm's scalar at
all, it is waiting on everything else because it comes last, so quantizing before the
scalar hides under 2 µs a layer, half a percent of the token, and the ceiling of any
placement, the whole phase hidden under the rows' loads, is 3 %. The other question was
answered by running the quantizer both ways over every real expert input in the
calibration sets, 13.7 million integers: replacing the two IEEE divisions with one changes
13 of them, always by one, in 0.2 % of position-layer activations; an ulp-off hardware
reciprocal changes 45. Over a full context that is a position in twenty with one INT8 off
by one somewhere, never in a routing input, but each one a seed for the two paths to part
later, for about 4 µs a token. The ledger closes with both numbers in `analysis.md` §9.

## The bundle's second reader

A second developer agent, building the API service, wrote `docs/bundle.md` and asked for
the edge's half of the bundle: the tokenizer, the chat template, and the constants the wire
assumes. Three of the four asks went in as proposed, one was moved. `tokenizer.json` and
`tokenizer_config.json` are copied from the checkpoint verbatim, hashed and listed; the
whole config rather than an extracted `chat_template.jinja`, because the template refers to
`bos_token` and `eos_token` defined in the same file, the tokenizers library reads it as it
stands, and a bundle is built from the checkpoint, never edited. A `[model]` table names
`bos`, `eos` and `vocab_size`, the model crate's constants, so a client speaks the wire
without reading the Rust. The context length stayed out: it is a memory budget chosen per
card, not a property of the checkpoint, and a bundle is a function of checkpoint, recipe
and calibration. So it became the node's, `rockem.ctx` at boot, and the wire's header
version 2 carries it back to the client as one u32, the positions a sequence may hold after
BOS, which also answers whether the edge needs to know about `FIRST_POS` (it does not).
Version 1 had zero clients, so the bump cost nothing.

Because the wire's BOS is the token id itself, encoding the rendered chat template with the
bundle's tokenizer yields BOS as the first id and the client sends what the tokenizer gave
it; nothing is stripped. The manifests of the shipped bundle and its RTN twin were
regenerated with `--dense-only`, which now skips calibration and copies the tokenizer
files, twelve seconds plus the hashing.

The gate found a bug that was not new. A second connection to a node panicked in the store:
`serve` reset `started` at accept without closing the active context, so the next BOS
opened a new context over the old one's region while the old one still stood in the index
with no rows, and the first prefix match read from an empty segment. Every earlier wire
run had been one connection per node. The region now decides: `start` closes the active
context whenever the active region is allocated, and the flag only gates ids before BOS.
Gate on lasenora, all on the card: fresh prompt, the same prompt again (full hit), the
prompt extended by its first generated id (prefix hit) — three identical streams, header
`v2`, positions 2,047; `rockem.ctx=64` reports 63, fills, closes the connection, and the
node stays up for the next. `docs/wire.md` and `docs/bundle.md` describe the implemented
state; the proposal section is gone.

## The sampler is the recipe

tcdent: "We didn't do a top_k gather yet did we?" We had not: the draw was temperature
and the full softmax walk, and the crate said so. The research came first, on the
version the checkpoint names. transformers 4.39.3 defaults `top_k` to 50 when a config
does not mention it, so the recipe DeepSeek shipped is temperature 0.3, then the 50
largest, then the nucleus at 0.95, then the draw, in that order, with the crossing token
kept; llama.cpp keeps the crossing token too but applies the temperature last, a
different distribution, and vLLM matches transformers. `SAMPLING` is now three constants
of the model, the one that is not in the config carrying its provenance in a comment.

The op is one candidate list and one walk on both tiers: the top-k and its ties by value
descending and index ascending, capped at 256, the masses at the top in the shared
exponential, the f64 sums in one order. The CPU op was checked against the warpers
themselves on every golden position, 435 of 435 tokens, and the nucleus turned out to
be one token at two positions in three, median 1, at most 17. The card's kernel finds a
lower bound of the k-th largest from the 256 thread maxima, gathers everything above it,
and orders the list by rank rather than by sort: each thread counts the pairs before its
own and that count is its slot, two barriers where a bitonic sort had 72. Hundreds tied
near the top, which real logits never are, fall to a bisection over the float's
order-preserving bits.

The gate for it, `bench-draw`, runs the kernel against the op on the 435 golden rows and
seven shapes real logits never take, and the first release build failed 28 of them with
float bits where token ids should be. A build with a few extra stores at the end passed
every row. The disassembly had one more barrier than the source: LLVM had duplicated an
`asm!("s_barrier")` into both arms of a divergent branch, because inline asm is not
`convergent`, and a wave with lanes on both sides barriered twice. Every kernel in the
tree had been one optimization away from this. `workgroup_sync` is now the intrinsic
between two compiler barriers; 442 of 442 since.

Two more things the gate turned up. The card's and the CPU's token streams part within
a few positions, and the sampler is not why: the CPU op on the card's logits draws the
card's token. The tiers' float ops differ at the ulp, and the INT8 activation quantizer
turns an ulp on a rounding boundary into a whole quantum, so the residual differs by
1e-3 from layer 9 of position 2 and a route flips at position 4. Each tier is
deterministic; the streams were never a gate, and the analysis now says so. And the
kernel's four phases sum to 45 µs of shader cycles while the timeline brackets it at
88 µs, the same bracket the previous kernel had; the difference is an open item. The
token stands where it stood, 140 tok/s in the bench, 0.68 ms for head and draw, and the
endpoint on lasenora serves the recipe.

## A published number

tcdent: "I want some way so that you are in the loop with the eval scores ... we can also
optimize for gut-checked benchmark scores." The design question was where benchmarks
live: they are about a model but they are not the bundle's, and they never go to the
node. They went where the other anchors are. The tasks are in `bringup/evals/`, next to
the goldens and the forward driver, and run against a served node over the wire through
the API's own client library (`rockem.wire`, `rockem.edge`), so what a benchmark sees is
what a client of the service sees, under the model's recipe, stop sequences and all. The
numbers live with the model, `models/deepseek-v2-lite/evals.toml`: the published score
with its source and protocol, and every run of ours under it with the bundle's identity
and recipe, so a kernel or recipe change shows against the anchor the paper used.

HumanEval first, because its prompts are short and the node has no batched prefill: a
prompt token costs what a generated one costs, 140 a second either way, so the 164
problems take five and a half minutes and GSM8K's eight-shot prompts would take an
afternoon. The card says 29.9 for the base model. Ours:

| bundle | pass@1 | capped at 512 tokens |
|---|---|---|
| q4 GPTQ, shipped | 25.6 (42/164) | 26 |
| q4 RTN, the twin | 27.4 (45/164) | 23 |

Three problems apart, inside a noise of six; the calibration that buys 0.004 nats
does not show at this resolution, and neither does anything else, which is the point of
the first run. A sixth of the completions ran to the cap, the model enumerating cases in
a loop, which tcdent had seen by hand and asked to have grounded: Holtzman et al. on
likelihood decoding, Xu et al. on the self-reinforcement of a repeated sentence, Fu et
al. on why the language itself invites it, and the Codex paper choosing T = 0.2 for
pass@1 anyway. The bf16 reference on the same prompts under the same recipe is the
check that it is the model and not us. The endpoint served the twin for the six minutes
of its run and is back on the shipped bundle.

## Scoring, and steps of thirty-two

Two features the benchmarks needed, taken in the order MMLU needed them. First a verb:
SCORE, wire version 3. After it, every id fed is answered with one word holding an f32,
ln p of that id given everything before, under the model's own distribution and no
recipe, before the id is fed. A `score` kernel on the card (the old sample kernel's first
half: max, sum of exponentials, a log) and `common::softmax::log_prob` on the CPU, the
sink's logits computed once at start for the first position, and a `settle` that leaves
the store's matching state with logits current, which also fixed GENERATE on an empty
prompt. Gated against the node's own logits dumps (1.9e-6) and against the reference
(0.02–0.07 nats, the recipe's error). The API's client library got `Client.score`. MMLU
runs through it, four choices per question with the prompt served from the store on the
last three: abstract algebra scored 33.0, a subject the card's 58.3 averages over.

Then batched prefill, the large one. tcdent: "I like both of those ideas. Let's
incorporate them." The design: every kernel takes a token count and a generated token is
a step of one through the same code, so there is one path. Activations are staged in LDS
a 256-element step at a time and each token's reduction keeps the order the one-token
kernels had, so a position's bits do not depend on how many tokens stepped with it. The
experts are grouped by expert on the card (`route` per token, `group` builds each
expert's token list) so a record is read once a step. Attention got a `kv` kernel that
writes every row of the step before `mla` reads any, and `mla` splits each token by its
own length. The norms became a kernel of their own where a rows kernel could not hold
thirty-two of them, and the router does its own. The wire needed no change: ids already
in the socket go through together, up to 32 a step.

The first build reproduced the decode stream bit for bit and was 40 % slower, and the
afternoon went to why, one lesson at a time, each found with `bench-step` and the
timeline rather than by reading code: the token loop's runtime bound had every block
executing under masks; the unrolled rounds hoisted forty-three steps of loads into 190
registers and scratch; lanes read their staged activations eight floats apart and hit
the same banks four at a time; and the code object in host memory, fine when the kernels
were small, now cost every launch 8–14 µs of instruction fetch over the link — the
router, whose code had not changed, went from 20 to 33 µs, which was the clue. Along the
way a runaway client I had left spinning on lasenora corrupted an hour of measurements,
and the DPM toggles I tried in the confusion changed nothing; both are in the ledger so
the next person does not. Where it landed: the prompt gate byte-identical at 162 of 162
positions, 480 prompt tokens a second in steps of 32 (from 123), and decode at 123 (from
144.5), the difference the three launches a layer the design adds and the expert kernels'
remaining inefficiency at one token, both on the ledger.

The gates after the dust settled: HumanEval on the batched build reproduced every one of
the 164 completions of the earlier run byte for byte, in 237 seconds instead of 328; and
the score gate caught the one bug batching had introduced, the first id of a step scored
from row 0 of the previous step's logits instead of its last row, two positions in 86
off by fourteen nats. Fixed by remembering the last row; the four sets back to
0.02–0.07 nats from the reference.

What the two features were for, the same afternoon: GSM8K in full, 1,319 eight-shot
chain-of-thought problems, 38.1 against the card's 41.1, in 25 minutes at 776 ids a
second where the sequential node would have taken two and a half hours; and MMLU through
the SCORE verb at 1,718 ids a second, college computer science 50.0, the full 14,042
questions running as this is written. Both are in `evals.toml` next to the published
numbers.

MMLU in full landed while the next kernel was being written: 57.5 against the card's 58.3
over 14,042 questions in 92 minutes, 376 of them at fewer shots because their passages did
not fit the context. Marketing 90, global facts 28; the subjects the model knows and the
ones it does not are the ones you would guess.

## A hundred thousand

tcdent: "let's aim for 100k context since it fits and will teach us more about
optimizations we can carry forward to the production box and the next model." The
context was never a memory limit — 62 KB a position, 6.2 GB at 100k beside the 10.3 GB
of weights — it was the attention kernel reading the cache once per head, sixteen times
a token, and a default of 2,048 chosen for the first card.

Before touching the kernel, a gate: the rope tables against the oracle's own at
positions up to the checkpoint's 163,840. It failed at once. The inverse frequencies
computed from the config in f64 and rounded differed from torch's f32 arithmetic by an
ulp in twelve of thirty-two, nothing at the goldens' positions and 0.0038 in a cosine at
100,000, the angle being the position times the frequency. The thirty-two values are the
model's constants now, the oracle's bits, and the test holds to 2e-6.

Then the kernel, three times. Absorbed queries for all sixteen heads in LDS: 52 KB, one
workgroup a CU, nothing to hide the row loads behind; slower than the original. Heads in
registers four at a time with every wave reading every row: twelve reads of the cache
from L2 per token, 40 ms at 100k. Rows staged in LDS four at a time and read once, an
online softmax so a single pass does scores and outputs, lanes owning latent dims strided
by 32 after sixteen consecutive ones turned out a 16-way bank conflict, the absorbed
queries in a launch of their own after one workgroup per token spent 108 µs a layer on
them, and the merge spread over waves after one thread walked 782 chunk partials in turn.
Each step was chosen from the statistics channel's phase stamps, not from reading the
code; the tuning doc has the sequence. Where it stands, attention per token on the
7900 XT: 3.5 ms at 2,048 where the original kernel took 4.1, 8.1 at 16k, 21.7 at 65k,
25.8 at 100k — 241 GB/s of cache rows against a floor of 763, so a third of the way to
the memory bound, with the compute loop's LDS-latency chains the ledger's next line.
The decode stream changed at position 4 with the reduction order and the prompt gate
holds at 162 of 162.

The gates after: HumanEval 25.6 again, the recipe's error against the reference unchanged
to the third decimal, and the node at `rockem.ctx=100000` handed a pass phrase in its first
sentence and asked for it after 8,000, 30,000 and 60,000 tokens of filler answered
"copper heron sixteen" all three times, verbatim, and went on to finish the sentence it
had read it in. Prompts went in at 359, 245 and 193 ids a second as the context grew,
attention's share growing with it. The endpoint serves at 100k now.

