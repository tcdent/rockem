#!/usr/bin/env bash
# One node process, the gate's prompt as `lines` console lines in a row: `node_lines.sh <bundle dir> [lines] [node args...]`.
# Under rockem.debug the node prints every position's argmax and top-2 margin; each line's block is
# compared with the first's. In-process repeats of the node's own path, against fresh-process ones.
# LIMIT (2026-09-19): the store serves the repeated prompt from its cache (even at rockem.cache=0 it keeps
# 2048 positions), so lines after the first recompute only the last position and the generated ones;
# the comparison then covers those alone. Distinct prompts per line would defeat the cache.
set -u
cd "$(dirname "$0")/../.."
B=${1:?bundle dir}; L=${2:-50}; NODE=${NODE:-target/release/node}; IDS=$(cat bringup/gates/prompt.ids); W=/tmp/rockem-gate/node-lines
ELF=${ELF:-models/deepseek-v2-lite/gpu/target/amdgcn-amd-amdhsa/release/deepseek_v2_lite_gpu.elf}
M=${M:-deepseek-v2-lite}
bringup/gates/install_elf.sh $ELF $B $M
systemctl --user stop rockem-node 2>/dev/null; pkill -x node 2>/dev/null; sleep 1
rm -rf $W; mkdir -p $W/dump
(for i in $(seq 1 $L); do echo "$IDS"; done; sleep 3) | timeout 3000 $NODE rockem.bundle=$B rockem.mem=anon rockem.gpu=1 rockem.max=8 rockem.debug=$W/dump "${@:3}" > $W/out.log 2> $W/err.log
python3 - "$W/err.log" "$L" <<'PY'
import sys
# a line's block starts at pos 1 (a store hit would start later: run with rockem.cache=0 so every line recomputes)
lines = [l for l in open(sys.argv[1]) if l.startswith("  pos ")]
blocks, cur = [], []
for l in lines:
    if l.startswith("  pos 1:") and cur:
        blocks.append(cur); cur = []
    cur.append(l)
blocks.append(cur)
n = len(blocks[0])
if any(len(b) != n for b in blocks):
    print(f"node lines: blocks of unequal size {[len(b) for b in blocks[:6]]}... — a store hit? run with rockem.cache=0"); sys.exit(1)
bad = 0
for i, b in enumerate(blocks[1:], 1):
    if b != blocks[0]:
        bad += 1
        first = next(j for j in range(n) if b[j] != blocks[0][j])
        print(f"line {i + 1} differs: first at {blocks[0][first].strip()} vs {b[first].strip()}")
print(f"node lines: {len(blocks) - 1} lines against the first ({n} positions each), {bad} differ")
PY
