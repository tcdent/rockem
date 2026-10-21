# The CPU expert kernel: q4 gemm on Zen 3 (EPYC 7R13), brief for kernel exploration

## What it is
`common/src/q4.rs`, module `avx2`: `gemm<K>` (multi-token, the prefill path) and `dots<NT>`. Weights are 4-bit
(q4, group of 128, per-row-per-group f32 scale, WBIAS 8), activations int8 per token with per-group scale and
gsum (`Act`). `gemm` takes a pair of 4-row tiles (8 rows), per group loads the packed nibbles, masks lo/hi, does
four `transpose8` (dword transpose: 8 rows in the lanes, one 4-k quad per row per 32-byte vector, the q4_0_r8 shape)
into a 1 KB `blk` scratch, then for the tokens 4 at a time `dots<4>`: per sub-block of 32 k, for each of 8 weight
vectors and each token, `vpbroadcastd` the token's quad and `vpmaddubsw` (u8 weights x i8 acts -> i16 pairs),
`vpaddw` into an i16 sum, then per sub-block `vpmaddwd` with ones and `vpaddd` into i32. Flush per token per group:
`(acc - WBIAS*gsum) -> f32 * (wscale*ascale)`, added into `y[token]` (stack). i16 sums over a sub-block are exact.
Callers: gate/up (K = 4096 = H, m = GU_ROWS rows an item) and down (K = 2048 = D_FF, m = DOWN_ROWS = 128),
n = 1..64 tokens an item (n = 1 goes to `gemv`). In prefill most items have 9–64 tokens.

## The hard constraint
Bit-identical output to today's `gemm` for every n and every row (same integer sums per group, same f32 flush
expression and the same order of f32 additions across groups). AVX2 + FMA only: Zen 3 has no AVX-512 and no
AVX-VNNI. 16 ymm registers.

## Measured (metatron, one Zen 3 core, `gemm_rate` cache-resident 32 rows x 4096)
GMAC/s by tokens: gemv(1) 58; gemm 2: 43, 4: 61, 8: 69, 16: 77, 32: 85.5, 64: 82.6. 33.6 MB streaming from DRAM:
16: 73, 32: 80, 64: 80. All 92 cores at once cache-resident: ~71–77 at 32 (a 13–17 % drop, not yet explained).
In a real prefill run items do 55–62 GMAC/s at 9–64 tokens.
A pre-arranged (transposes done offline) layout measured 1.02–1.07x at 8–32 tokens, ~1.0 at 64: not the cost.

## Static analysis (llvm-mca-14 -mcpu=znver3 on the release build's disassembly, files here)
- `gemm_gate.s`: objdump -M intel of the gate projection's gemm (unstripped build of the real node, LTO).
- `loop4.s`: the hot NT=4 sub-block loop: ~26 cycles/iteration for 32 vpmaddubsw (1024 MACs), bound by the 4 FP
  pipes (~25 each); 100 FP uops/iter: 32 vpmaddubsw, 32 vpbroadcastd, 28 vpaddw, 4 vpmaddwd, 4 vpaddd, plus 8
  weight loads from the blk scratch. Loads 20.5.
- `setup.s` group setup ~83 cycles/group; `bsetup.s` per-4-token activation slicing (bounds checks) ~11.5;
  `flush.s` one token's flush ~4.9.
- Per group at 32 tokens ~1160 cycles: dot loop 71 %, flush 13 %, slicing 8 %, setup 7 %. At 16 tokens: 67/13/7/13.
Tool: `llvm-mca-14 -mtriple=x86_64 -mcpu=znver3 -iterations=200 file.s` (a file starts `.intel_syntax noprefix`).
Build the release node unstripped to disassemble your variant:
`os/run.sh cargo build --release --target x86_64-unknown-linux-musl -p node --features track-launches --target-dir /work/target-elf --config profile.release.strip=false`
then `objdump -d --no-show-raw-insn -M intel` and `nm -C --size-sort -S` to find `common::q4::avx2::gemm::<4096, ...>`.
Or faster: write a standalone `#[inline(never)]` instantiation in a test and disassemble the test binary
(`cargo test --release -p common --no-run`, host is a Zen 1 EPYC 7551P: use it for bit-identity, never for speed).
Existing tests and the rate bench: `common/src/q4.rs` tests module (`gemm_rate`, bit-identity tests vs gemv).

## Already in progress elsewhere (do not duplicate)
Two tile pairs (16 rows) per pass so each broadcast feeds two vpmaddubsw (est. 100 -> 84 FP uops per 1024 MACs).

## Background
`docs/queue/` holds every past experiment (one file each, with measured results); `docs/glm-next-steps.md` indexes
them. Relevant: 021 (CPU expert items, buckets), 074 (decode at the DRAM ceiling), 078 (today's prefill findings),
066, 011. Memory notes on the CPU kernel: four bits is the floor (no sub-4-bit formats), LUT kernels declined.
