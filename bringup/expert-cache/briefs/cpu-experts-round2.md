# The CPU expert path, round 2 (2026-10-20) — shared brief

## Where we are
GLM-5.3-Flash on metatron: 2 sockets of EPYC 7R13 (Zen 3, 48 cores / 6 CCDs a socket, 32 MB L3 a CCD, NPS2 => 4 NUMA
nodes, 8 DDR4-3200 channels a socket), 512 GB, 2x 7900 XT. The routed experts (4-bit q4, groups of 128) mostly run on
the CPU: a pool of 92 pinned cores (cores 0, 24, 48, 72 are housekeeping), items slotted to the CCD whose node holds the
expert slice (`run_slotted`, block layout: weights are node-local). The 1M-token context reservation leaves only ~4
experts a layer resident on the cards, so nearly every pick runs on the CPU.

- Prefill at 18k: 2.02 ms a prompt token, a cold 18k prompt 39.7 s. A prefill step's host time is ~80 % the CPU
  experts. Per 512-token layer-chunk: experts lap 18.24 ms (gate/up run 10.57, down run 5.78, quantize ~2).
- Decode at 18k: ~27.5 ms an emitted token, a verify step ~59 ms of which the CPU experts lap ~26 ms. Decode's
  one-token items stream from DRAM at ~375 GB/s across both sockets (queue 074) — "the DRAM ceiling" — but tcdent does
  not accept that as the end: bytes from L3 instead of DRAM, fewer bytes, overlap, are all on the table.

## The kernel (just rewritten, commit 7bd0e9e; `common/src/q4.rs`, module `avx2`)
`gemm` (2..64 tokens): per pair of 4-row tiles and group of 128 k, unpack the stored nibbles by masks + `vperm2i128`
into 32 vectors (two rows' 16 k a vector: row v | row v+4); per token pair `dots` broadcasts a token's 16 k with
`vbroadcasti128` (4 multiplies per broadcast), `vpmaddubsw` into i16 sums spanning the whole group (<= 30480, exact),
then `vpmaddwd` + unpack/add reduction to the 8 row sums; flush per token per group:
`(acc - WBIAS*gsum) -> cvt f32 * (wscale * ascale)` added to `y[token]` (stack). `gemv` is the one-token path.
Hard constraint: bit-identical to `gemv` for every token (`tests::gemm_is_gemv_for_every_token`). AVX2+FMA only.
Measured on one metatron core, cache-resident GMAC/s: 2 tok 75.6, 4: 92, 8: 98, 16: 104, 32: 106, 64: 103; one-token
`gemv` 58. In real prefill runs gate/up items do 69–76 GMAC/s a core at 9–64 tokens. With all 92 cores running the
cache-resident bench at once (nothing shared), the OLD kernel dropped 13–17 % a core (85 -> 71–77 at 32 tokens):
unexplained.

## Tools
- llvm-mca: `llvm-mca-14 -mtriple=x86_64 -mcpu=znver3 -iterations=200 f.s` (Intel syntax with `.intel_syntax noprefix`).
  Zen 3 facts (uops.info): vpmaddubsw/vpmaddwd only on FP0/FP3 (floor 18 cycles per 1024 MACs); vpbroadcastd m32 takes
  an FP1/FP2 uop; 2x256-bit loads, 1x256-bit store a cycle.
- Unstripped release node for disassembly: `os/run.sh cargo build --release --target x86_64-unknown-linux-musl -p node
  --features track-launches --target-dir /work/target-elf --config profile.release.strip=false` (run from the repo
  root; in a worktree `os/run.sh` mounts the worktree, and `../prow` is symlinked for you).
- Bench: `cargo test --release -p common gemm_rate -- --ignored --nocapture`; for the box build the static binary with
  `os/run.sh cargo test --release -p common --target x86_64-unknown-linux-musl --no-run`, copy the
  `target/x86_64-unknown-linux-musl/release/deps/common-<hash>` executable into `bundles/tools/<name>` (the box sees
  `bundles/` at `/bundle`), then run on the box. The dev host is a Zen 1 7551P: correctness only, never speed.
- Profiling data of real runs: `bringup/reads/*.sh` over `prow-sql` (DuckDB on a shared server). `experts.sh` (item
  buckets and per-core rates), `pool.sh` (run tails), `laps.sh`, `requests.sh`, `zones.sh`. NEVER run an unbounded
  join across a whole run's zones (it crashed the server once today): bound queries to a time window.
- History: `docs/queue/*.md` (index `docs/glm-next-steps.md`); CPU expert entries 020, 021, 054, 056, 065, 066, 074,
  076, 078 (today's measurements, read its last sections). Memory notes (read-only for you):
  `~/.claude/projects/-home-slomp-Work-rockem/memory/` — cpu-expert-bandwidth.md, l3-intentional-expert-path.md,
  epyc-7r13-topology.md, milan-bandwidth-targets.md, hybrid-experts.md, experts-on-cards.md, batched-prefill.md.

## Rules
No deploys, no reboots, no commits, no pushes. Do not edit files outside your worktree (except where told). Report
measured vs inferred separately.
