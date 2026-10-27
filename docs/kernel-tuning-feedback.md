# Kernel tuning as a feedback problem

A topic note, not the project's root documentation. It sets down how this project
tunes kernels — what we measure, what each measurement can and cannot decide, and the
gates a change has to pass — next to the methodology Z.ai describes for the
GLM-5.3-Flash inference stack ("Toward Recursive Self-Improvement: How GLM Built Its
Own Inference Infrastructure", 2026-09-17,
https://z.ai/blog/glm-built-its-inference-infrastructure). The two were arrived at
independently on very different systems (one user, one node, a handful of models,
versus 100,000 accelerators serving 62 trillion tokens a week) and land on the same
shape. Where they name something we do without naming, we take the name. Where they
have an instrument we lack, it is listed at the end with a note on when we get to it.

## The shape

Their phrase is *dense feedback*, and "dense" does not mean more logs. It means every
observation is

- **local** — tied to a kernel, an input condition, a code path, a launch parameter,
  so it narrows the problem instead of describing it;
- **cheap and timely** — answerable by a kernel test or a microbenchmark, not a full
  deployment, so a wrong hypothesis costs minutes;
- **objectively verifiable** — decided by a reference implementation, a test, or a
  controlled experiment, never by a correlation in a log.

And it comes in three kinds, each answering one question:

| kind | question | their instrument | ours |
|---|---|---|---|
| correctness | is the computation right? | kernel paths compared against unpartitioned implementations, under the deployment's real partitioning | the scalar twin at **equality**; launchers against the CPU's real layer state (`bringup/gpu/*.py`); the placement matrix (`node rockem.debug=`) |
| system behaviour | where is the time going? | execution traces, runtime events, timelines across compute / waiting / communication | the launch timeline (see Observation), read by `bench-token`, and recorded by the node and `forward-glm --timeline` as prow zones; the node's `step` rows (see *Rows*) |
| performance | which approach, under which conditions? | microbenchmarks under specific input shapes, then end-to-end confirmation | `bringup/bench` with the bandwidth ceiling measured in the same run; the rotating working set; end-to-end tok/s on the node |

The correspondence is close enough that their three example cases each have a
counterpart in the story log. Their KDA context-parallel bug — `tl.dot` silently
computing FP32 inputs in TF32, an error that only showed after accumulation over long
context — is the class our equality rule exists to catch, and the per-position logits
diff would have shown it growing with position. Their GIL-across-DeepEP finding is a
timeline observation we currently cannot make. Their KDA decode kernel, redundant
per-tile normalization merged into one block with a warp reduction, is the same
transformation as the expert kernel's cell-per-lane rewrite.

**What the 100k attention kernel taught (2026-09-19).** Three rewrites in an afternoon,
each measured with the statistics channel before the next: absorbed queries for all
heads in LDS (52 KB) left one workgroup per CU and nothing to hide row-load latency;
heads in registers with every wave re-reading every row made the kernel L2-bound at
twelve reads of the cache per token; staging rows in LDS with lanes owning sixteen
consecutive floats was a 16-way bank conflict. The shape that works: rows staged once,
lanes owning strided dims, two heads a wave with an online softmax, and the per-token
work (absorbed queries) in its own launch rather than repeated per chunk or serialized
in one workgroup. The phase stamps, not the timeline, distinguished each of these.

**Four lessons the n-token kernels taught (2026-09-19).** A runtime token count in an
unrolled loop became every block executed under masks, twice the one-token time: the
width is a `const` and the kernel dispatches on it. Forty-three unrolled steps of a
10,944-wide row put 190 registers and scratch in a kernel that needs 23: a round is eight
steps at most. Lanes reading their eight activations eight floats apart hit the same
banks four lanes at a time, an 8-way conflict that cost the multi-token rows kernels 20×
their weight-streaming time: staged at a stride of ten. And the code object in host
memory, fine at 48 KB of small kernels, cost every launch 8–14 µs of instruction fetch
over the link once the kernels doubled and a token's weights had streamed through L2
between launches: it lives in VRAM. Each was found with `bench-step` (an op at a time
with a wait after each) and the timeline, not by reading the code.

**A lesson the sampler gate taught (2026-09-18).** A workgroup barrier written as
`asm!("s_barrier")` is not `convergent` to LLVM, and the compiler duplicated one into both
arms of a divergent branch, so a wave with lanes on both sides barriered twice and the
kernel read LDS slots before other waves had written them. The release build failed 28 of
442 rows; a build with a few extra stores at the end passed all of them, which is how it
was found. `common::gfx11::workgroup_sync` is now the LLVM intrinsic between two `asm!`
compiler barriers. The gate that caught it is the shape every op gate should have: the
realization against its twin on real inputs and on the inputs real data never produces.

## Correctness: against what

Three anchors, each with one job, none reducible to another. This is the part that
took the longest to get straight and the part most worth keeping straight.

| anchor | decides | criterion |
|---|---|---|
| the scalar twin of a kernel | the fast realization is implemented right | equality, byte for byte (integer kernels today; float kernels once the reduction order is a contract) |
| the oracle (pinned modeling file, f32) | the model is transcribed right | tolerance at the f32 order-noise level |
| the oracle, teacher-forced | a recipe is a faithful interpretation of the weights | KL(oracle‖ours) per position, top-1 agreement, oracle-token NLL, on a wide golden set |

> "our reference implementation is not the reference implementation; it's ours" — tcdent

The oracle's bits are not canonical: its reduction orders differ from every order of
ours by the same 1e-6 that ours differ from each other. The weights were trained under
bf16 noise, so the model is robust to perturbations at that scale in *quality* and
robust to nothing at all in *tokens*, because the router's top-k and the greedy decode
are step functions. Token identity is therefore never a correctness measure; it is a
determinism measure, and determinism is a contract about reduction order, not an
accuracy property. Quality lives in the recipe and is measured teacher-forced.

> "being able to reconcile what's happening in the transformation between those two
> seams … we retain parity between CPU and GPU kernels, and then we can, in theory,
> use them somewhat interchangeably" — tcdent

Parity is bit-identity, and it is achievable: the integer expert path has it end to
end today (zero difference at every position across tiers). The float kernels get it
when each op's reduction order is written into its contract — tree-shaped, so it is
parallel-friendly on any tier and more accurate than sequential against exact
arithmetic — and every realization commits to it with no FMA contraction. That is a
cost on the slow-path tier only.

## Performance: against what

Every kernel number is a ratio to a ceiling measured in the same run on the same
bytes: a plain streaming read for bandwidth-bound kernels. "Fast" has a denominator.
Three instrument failures we made and fixed, each of which produced a number that was
wrong in a way the number itself could not reveal:

- a device synchronize around every launch pair measured the idle queue's wake-up,
  not the kernel (three different kernels read a flat 40 µs) — batched launches inside
  one sync window, which is also how the token loop issues them;
- a six-record working set sat in the 80 MB infinity cache and read 1.09 of the VRAM
  ceiling — rotate the working set past the cache;
- a per-lane array the compiler would not fully unroll spilled to scratch and made a
  "faster" version slower — check `.private_segment_fixed_size` on every build.

The pattern in all three: measure something narrower than the number you care about
(a probe, a batch, a rotation, a note in the ELF) until the number is explained.

## Gates

Stated before the experiment, so a miss is a defined discrepancy rather than a
feeling. Theirs: "Prefill + KV Transfer within 5 % of Prefill alone." Ours:

| gate | criterion | status |
|---|---|---|
| kernel correctness | integer realizations equal the scalar twin; float realizations within 1e-4 of the CPU step on a real layer state (equality once the order contract lands) | in force |
| sampler | the card's `sample` equals `common::sample::draw` on every golden logits row and the degenerate shapes (`bench-draw`); the CPU op reproduces transformers' warpers on every golden position (`bringup/sample_check.py`) | in force, 442/442 and 435/435 |
| scoring | the SCORE verb's ln p equals log-softmax of the node's own logits dumps within 1e-4 on a golden sequence (`bringup/score_check.py`); against the reference it reports the recipe's error in nats | in force, max 1.9e-6 |
| transcription | forward matches the oracle at every traced edge and reproduces its greedy tokens on the golden set | in force |
| placement | the placement matrix reads 0 in every cell for every realization on every tier | integer: in force; float: pending the order contract |
| recipe quality | teacher-forced KL below a stated bound on the wide golden set (≥ 400 positions, ≥ 4 domains) before a recipe gets a name | measured, bound not yet chosen |
| kernel performance | ≥ 0.85 of the bandwidth ceiling on a working set larger than the last cache, batched launches | in force for the expert kernels |
| end-to-end | tok/s on the node, real prompt, each placement (`bringup/gates/wire.py` prints the decode rate over the wire) | measured, no bound; 132 tok/s at ctx 100000 (2026-09-19) |
| steps | a prompt fed in steps of 32 produces every position's dumps byte-identical to the same prompt fed one id at a time (`make gate-dump`: `bringup/gates/dump_diff.sh` on the GPU box, two `rockem.debug` runs, `cmp` per position; refuses to run while a stray wire client is on the box) | in force, 162/162 — with a known rare miss: ~1 run in 15 differs at one token and layer by ~1e-3 with the picks unchanged, either side; predates the launch fusions; the bare forward (`bench-repeat`) never reproduces it, the node does; open |
| benchmark | a published benchmark, run against a served node through the edge's own client under the model's recipe, within its noise of the published number (`bringup/evals`, kept in `models/<name>/evals.toml`) | HumanEval in force |
| build hygiene | no scratch, no runtime `cfg`, no retyped constants across the Python boundary; the code object's static report (`make kernels` → `bringup/kernel_report.py`: registers, scratch, LDS, instruction/barrier/load counts per kernel, diffed against the last build's report) prints only what moved and fails on scratch, on more than 192 VGPRs, or on a barrier count that changed | in force |

**Decode identity (2026-10-27).** The long SCORE gate exercises only the kernels of steps past `SPEC_TOKENS`
(the WMMA rows twins and the step-shaped passes); every decode-width kernel — the `rows_*_e4` passes, the
expert-prediction analog — is invisible to it. Build an16 changed `rows_d2_e4`, `rows_h_e4` and `rows_part_e4`
through a reduction helper in gemv.rs that the analog's rows variant shared with them, with SCORE bit-identical,
and the decode moved from its first speculative step. Two gates close the gap:
*static, before any boot* — `bringup/gates/kernel_diff.sh new.elf served.elf`: every kernel symbol's instruction
stream diffed against the served object; a change outside the allowed prefixes (`an_`) fails the object (an16
fails on the three rows kernels, an15 passes); *on the box* — `bringup/box/decode_identical.sh`: the speculation
path of the cody-decode hits (`spec_step`'s (pos, made, verified, accepted) sequence) against the reference
`bringup/gates/cody-decode.fingerprint` taken from an13; deterministic across builds, boots and PCIe error bursts
(an12, an13, an13s, an13f, an15, two restores — 54 steps, 96/54/41 a request), so a build that moves it has changed
the computation. deploy.sh runs it after its prow reads (three extra hits give the steady-state path); the
`an*_gates.sh` workloads run it over 24. The step-width histogram is not the fingerprint (its count moves with the
run boundary). Residency cannot move the path by rounding — the card expert path is bit-identical to the host's
and the join is in pick order (002, 003), and an12 → an13 → an13f (rounding-level changes to the analog's margins)
left it untouched — yet in every run the cold request, the first hit and the second hit each follow their own path
and hits 3+ another, same prompt; unexplained, consistent across builds.

## Observation: four channels, none inside a kernel

The question tcdent put on 2026-09-17 — "is there a way we can do that without
interleaving anything into the kernels themselves … perhaps we incorporate some
conventions or maps for memory that lets us pull from the card versus write anything in
the loop? Or we make them all macros so debug/perf is a one-liner and gets interleaved
and compiled out in release targets?" — has one structural answer: everything a GPU
kernel produces is in a device buffer at a launch boundary, and launches happen only at
synchronization points, so every observation can be made from outside the kernel. The
kernel that ships is the kernel that was measured. Four channels, with where each stands:

| channel | what it observes | mechanism | status |
|---|---|---|---|
| **timeline** | where the time goes: every launch's start and end on the card's 100 MHz clock, and the gap before it | every dispatch carries a signal from a ring in the launcher; the command processor stamps it; `Kfd::timeline()` after a wait (`node/src/kfd.rs`) | **in force**; `bench-token` accounts a token by kernel with it; the GLM node built with `track-launches`, and `forward-glm --timeline`, record every launch as a prow zone (see *Rows*) |
| **edge mirrors** | the values at a kernel's edges: this position's cache rows, each layer's residual, the routing, the logits | a kernel writes its edge to a host-memory mirror as well as its device buffer — one extra store, no branch — read once after the token's wait (`KfdModel::token_trace`) | **in force** for the four edges the store and the debug dumps need; a named edge table for the rest is the extension |
| **static** | registers, LDS, scratch, occupancy per kernel | the kernel descriptors in `kernels.gfx1100.elf` (`*.kd`: rsrc1/rsrc2, private and group segment sizes) | scratch is asserted zero at load; a report tool is not written |
| **statistics** | what a kernel knows about itself: the time of each phase inside it, a value at a point in its epilogue — one entry per wave, so the host averages or ranks after the fact and no wave is a representative | `common::stats`: `stat!(slot, value)` and `stat_clock!()`, expanding to nothing without the `stats` feature; a slab `[launch][slot][wave]` of f32 bits the kernel finds by its own dispatch id (the timeline's index), the launcher fills the slab's address into `ROCKEM_STATS` in the loaded image; slot names declared next to each kernel (`stat_names`); the stats code object is a bring-up build (`--features stats`, `bundles/kernels-stats.gfx1100.elf`), the release object is byte-identical to one without the feature | **in force** for `route` (four phases), `attn_mla` (three), and the down kernels (the residual's magnitude, the sink detector); read by `bench-token` |
| **hardware counters** | cache hit rates, memory transactions, wave occupancy per launch | three PM4 command blobs (start / stop / read) in vendor-specific AQL packets around a dispatch, exactly what ROCm's profiler emits; the gfx11 register offsets and PM4 opcodes are in the kernel tree we build (`drivers/gpu/drm/amd/include/asic_reg/gc/gc_11_0_0_*.h`, `amdgpu/nvd.h`); the packet sequence is transcribed in the memory note *rocm-perfcounter-mechanism* and belongs here when built | researched 2026-09-17; **buildable now** that the node owns its queue; not built — nothing so far has needed a counter to explain |

**On probes inside kernels.** tcdent's proposal (2026-09-18): "a protocol where you had
an address space that you could write statistics into, and then inside kernels they could
drop numbers into those registers … a statistic that exists inside of our own stack … just
having some naming conventions for registers that we write to" — and, on sampling, that
one wave is not representative, so keep a vector and average afterwards. That is the
statistics channel above, and it replaced the earlier objection (a probe shapes register
allocation) with a measurement: the stats build's timeline is compared with the release
build's, and the cost is visible per kernel. The first clock was the real-time one the timeline
uses, a returning send-message on gfx11 at about a microsecond a read, which made
phases under a few microseconds unreadable and inflated any phase that held several
stamps; `stat_clock!` now reads the shader cycle counter (`s_getreg`, one scalar
instruction, 20 bits), and the report infers each kernel's clock rate from the timeline
so cycles print as time with the assumption shown. The rule that keeps it honest: `stat!`
is the one thing allowed to differ inside a kernel body between the two builds, and its
body is a store; it never branches on what it wrote. What it answered first: the router's
22 µs is rows 12.7, quantize 6.8, softmax 3.9, pick 2.4 — the rows latency-bound on one
workgroup, the quantize a candidate to move into the expert kernels' prologue.

**The stamps cut the object too.** A stats build's clock reads are instructions
(`s_sendmsg_rtn_b64 … MSG_RTN_GET_REALTIME` for the real-time clock, `s_getreg` for cycles),
so the disassembly of a kernel splits at them into exactly the phases the slab times:
`llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx1100` from the rustup toolchain, cut
the kernel's listing at each clock read, and count instruction classes a region — VALU,
`v_rcp`/`v_div_scale`, `ds_bpermute`, `s_waitcnt vmcnt(0)` (a full drain of the memory
queue, one dependent round trip), `flat_` against `global_` (a pointer that came out of a
device-address table is generic, and a generic load is also charged to `lgkmcnt`, so every
LDS wait drains it too), `scratch_` and `.vgpr_spill_count` from `llvm-readobj --notes`.
Read before the next build, not after: a measured phase that a rewrite did not move is a
class the rewrite did not touch, and the counts say which — and a fit of phase time to any
one class holds only for the code shape it was fitted on. What it answered in 098
(2026-10-27): the analog chain's phase B, one wave a token with every lane computing alike,
held at 20–31 µs through four rewrites of its Sinkhorn and its loads. A two-table fit to the
region's `vmcnt(0)` drains (14 → 19.9 µs, 19 → 26.6) failed when a build cut the drains to 6
and measured 31.4; a four-table fit to the region's instruction count (1,885 → 19.9, 3,806 →
26.6, 4,858 → 31.4, ≈ 12.6 µs + 3.9 ns an instruction, nine cycles each — a single wave's
dependent issue with nothing on the SIMD to hide its latency) priced those builds and then
failed the other way when the lanes were given different work: 2,051 instructions of DPP
butterflies, lockstep sums and `v_readlane` measured 11.4 µs, under the old intercept. What
the tables actually priced was the dependent chain of a given shape; the general quantity is
one wave's critical path, which the counts bound from above but do not give. The usable rule:
for a few tokens on one wave, straight-line dependent instructions are the cost, lanes doing
the same work buy nothing, and a cut that changes the shape must be measured, not read off
the previous shape's line. The register pressure that moved loads into loops, the ALU chain
that grew when they came back out, and the shape change that halved B left no trace in the
slab; all three were in the object.

**The rule that makes observation a property rather than a session.** Every launch site
declares the bytes its kernel moves, the ceiling is measured on the box at startup, and
the timeline reports each kernel as a fraction of that ceiling — the bench harness's rule
("every number is a ratio to the bytes it had to move at a stated bandwidth") enforced
by the launcher. With that, the node's own per-token account, printed on request from the
shipped binary, is the system-behaviour feedback the table above calls missing. *Not yet
done; the next item.*

## Rows: what the node and the bring-up tools record

Measurements leave the node as prow rows (`~/Work/prow`), not console lines; the console is the
server's log (what it runs, what went wrong). A hook names a category, a Cargo feature of the
crate that calls it, so what is recorded is decided at build time and a build records it from
the first step on; `node --no-default-features` records nothing and is the baseline for what
recording costs. Every run's manifest carries `build.commit`, `build.dirty`, `build.features`
and the node's `model`, `bundle`, `ctx`, `gpu`, `cmdline` and `image`; `forward-glm` runs carry
`tool` and `args`. The node sends to `192.168.88.3:7380` (`rockem.prow=` overrides), from core 168.

| category | crate | sites |
|---|---|---|
| `track-step` (default) | node | `step`: a row a GLM step, the CPU model's `Profile` summed over the step's tokens (`*_ms` phases, the pick and item counts); `step_node`: the step's expert runs by NUMA node; `experts`: every 16 positions, the coverage since start and the cards' resident slots |
| `track-store` (default) | node | `request`: a generated continuation (prompt, hits, generated, ms, verify steps, drafts accepted); `store_resume`, `store_keep`; `messages`: the placement's re-trainings |
| `track-boot` (default) | node | `boot`: each loaded file, the bundle, the prefix, the cards' upload (`what`, `s`, `gb`) |
| `track-launches` | node, forward-glm | `zones`: every launch on lanes `card0`, `card1` (name, start, `dur_ns`), from the completion stamps through the card's clock. Stamping costs ~1.7 µs a launch and turns on the resident experts' host check, so the dev image leaves it out: `make node NODE_FEATURES="--features track-launches"`. A step of more launches than the ring holds keeps its last 512 |

Queried with `~/Work/prow/bin/prow-sql rockem "<SQL>"` (DuckDB; a view per site and `runs`,
joined on `run`). The time by phase a token over a run's single-token steps:

```sql
SELECT count(*) steps, round(avg(experts_ms), 2) experts, round(avg(mla_attn_ms), 2) attend,
       round(avg(hc_ms + slot_ms + kda_card_ms + kda_cpu_ms + mla_proj_ms + mla_attn_ms + mla_out_ms
             + dense_ffn_ms + route_ms + shared_ms + experts_ms + head_ms + draft_ms), 2) total
FROM step WHERE run = '<run>' AND width = 1;
```

A kernel variant against another (what `tl_agg.awk` summed from the old timeline lines), µs a
launch per card, one `forward-glm --elf <variant> --timeline` run each:

```sql
SELECT r.args, z.lane, z.name, count(*) launches, round(avg(z.dur_ns) / 1e3, 1) us
FROM zones z JOIN runs r USING (run) WHERE r.tool = 'forward-glm' AND z.category = 'track-launches'
GROUP BY ALL ORDER BY z.lane, us DESC;
```

## Skeletons

Their name for a validated transformation kept with its applicability conditions,
resource constraints and evidence, so the next kernel starts from it. Ours live in
the story log as narrative; distilled, the expert kernel's ladder reads:

| rung | transformation | applies when | evidence |
|---|---|---|---|
| grouped launch | one launch over every expert a layer hit, experts named by record offset | any per-expert kernel with a 2.5 µs floor | 39 ms → 53 µs per layer |
| cell per lane | a lane owns one (row, group) cell and reads its cache line whole; no per-group cross-lane sum | group-quantized formats with a per-group flush | no scratch; 0.34 → 0.89 of ceiling |
| wave-contiguous steps | each load instruction reads 512 contiguous bytes across the wave; quads share a cell | any row-major tile read | `global_load_b128`, 0.93 |
| sum in group order as you go | no term array; one running f32 per row | whenever a term array would exceed the unroller's patience | spill removed |
| SwiGLU across the launch boundary | the epilogue becomes the consumer's prologue | when fusing into the producer halves the waves in flight | 0.82 → fused at no cost |
| `v_dot4_i32_iu8` on biased nibbles | the unsigned×signed pairing `vpmaddubsw` uses on the CPU | any u4·i8 dot on gfx11 | 4 dot4 per 8-byte word vs ~40 VALU ops |
| the whole token on the card | the FFN half and the head as launches; the residual crosses only at the seam | when every weight the layer needs fits in VRAM | 14.8 → 103 tok/s |

Attention's ladder has one rung so far (three launches split at grid-wide
dependencies; 0.85 of the ceiling at 2048 context) and one known next rung (read each
cache row once for all heads).

## Where the two systems differ

They serve; we don't. Their system-behaviour feedback is mostly about concurrency
between requests, transfers and compute across processes and threads. At batch 1 on
one node the same question has a smaller shape: launch-to-launch bubbles on one queue,
host round trips at the seams, and whether a tier's kernels are waiting on bytes or on
each other. That is still a timeline question, and the answer is still "observe it,
don't infer it."

They trade compute for bandwidth (W8A8, ReplaySSM, mixed INT8/FP8/BF16 KV cache) and
price it end to end. We have the pricing instrument (teacher-forced KL) and have used it
to find that the INT4 recipe's cost is concentrated at the sink token and at three
layers, which is the kind of result that decides where the bytes go. KV cache precision
is on our unmeasured list; theirs is in production.

## To incorporate, and when

- **Launch accounting.** *Measured 2026-09-19:* a barrier launch's gap is 3.6 µs —
  the completion signal 1.7 (now an instrument the node runs without), the agent
  acquire 1.1 (required: the step gate fails on stale reads without it), the rest the
  processor's turnaround. 348 → 267 launches a token by folding the norm into the
  rows kernels' staging, the cache rows into the absorb grid, and the grouping into
  the router's last workgroup (an arrival counter). Token 7.64 → 6.97 ms.
- **The hybrid handshake.** *Built 2026-09-20:* the host runs its share of the
  experts behind the card, layer by layer, and the card waits for data rather than
  for the host: the router's last workgroup writes a done word the host spins on,
  the host writes a valid word per slot, and the combine kernel polls the valid
  words in-kernel after its own rows. No packets. The packet form (a mark and a
  gate per layer) was built first and measured: 0.2 ms a token with nothing to wait
  for, a 10 µs wake on release; a late-written packet wakes the processor in 40 µs,
  and the ring in VRAM hangs it. The combine's span on the timeline is the wait.
- **Timelines.** *Done 2026-09-18* (see Observation): per-launch device timestamps
  from the launcher's own signals. Still to do: the node reporting its own token, bytes
  declared per launch, the ceiling measured at boot.
- **Configuration → kernel mapping.** Their mapping from a parallelism strategy to the
  kernels it involves. Ours is the placement matrix generalized: for each seam, which
  realization runs on which tier, and the test that compares them on the same bytes.
  *When: as the placement table in the node's construction report, once a third
  tier (the fabric) exists.*
- **A skeleton library** as a file per kernel rather than narrative in the story log:
  rung, transformation, applicability, evidence. *When: on the second model, when a
  kernel is first copied.*
- **Bounds on the recipe gate.** A KL bound and a golden set of ≥ 400 positions across
  ≥ 4 domains as the precondition for a recipe string. *When: with the first
  calibrated recipe.*
- **One command that runs every gate** in the table above and prints pass/fail per
  row — the twins at equality, the launchers within tolerance on a real layer state,
  the store gate (fresh-run dumps against cache-hit dumps, today a shell loop retyped
  each time), the forward against the goldens, no scratch in the code object, and the
  performance rows against declared bounds. *When: before anything is committed.*
  *Started 2026-09-19:* the Makefile holds `kernels` (build + static report) and
  `gate-dump`; the remaining rows are still separate commands.
- **Kept numbers.** A file per box — measured ceilings, per-kernel fractions, tok/s per
  context length — that the gate command reads and compares against, so a fall past a
  stated margin is a failure and not a feeling. *When: with the gate command.*
- **Controlled experiments as first-class tools**: the noise probe (random and fixed),
  the per-layer and per-position recipe switches, and the per-expert masks in the
  forward driver are the start of a set that should be documented with what each can
  and cannot decide. *When: with the analysis document's next revision.*
