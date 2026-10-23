#!/usr/bin/env bash
# The node itself, again and again, on the console with the gate's prompt: `node_repeat.sh <bundle dir> [runs] [node args...]`.
# Every run's dumps (rockem.debug) against the first run's; prints each run that differs and where it
# first parts. Isolates the node's own path (store, sink, trace, draw) from the wire and the batching.
set -u
cd "$(dirname "$0")/../.."
B=${1:?bundle dir}; R=${2:-20}; NODE=${NODE:-target/release/node}; IDS=$(cat bringup/gates/prompt.ids); W=/tmp/rockem-gate/node-repeat
ELF=${ELF:-models/deepseek-v2-lite/gpu/target/amdgcn-amd-amdhsa/release/deepseek_v2_lite_gpu.elf}
M=${M:-deepseek-v2-lite}
bringup/gates/install_elf.sh $ELF $B $M
systemctl --user stop rockem-node 2>/dev/null; pkill -x node 2>/dev/null; sleep 1
rm -rf $W; mkdir -p $W
bad=0
for i in $(seq 1 $R); do
  mkdir -p $W/run$i
  (echo "$IDS"; sleep 2) | timeout 300 $NODE rockem.bundle=$B rockem.mem=anon rockem.gpu=1 rockem.max=8 rockem.debug=$W/run$i "${@:3}" > $W/run$i.log 2>&1 < /dev/stdin
  [ $i = 1 ] && continue
  off=0; for f in $W/run1/pos*; do cmp -s $f $W/run$i/$(basename $f) || off=$((off+1)); done
  if [ $off != 0 ]; then bad=$((bad+1)); echo "run $i: $off files differ"; for p in $(ls $W/run1 | grep -o "pos[0-9]*" | sort -u | sed s/pos//); do python3 bringup/gates/dump_where.py $W/run1 $W/run$i $p | grep -v "every layer the same; logits same" | head -2; done | head -3; fi
done
echo "node repeat: $((R - 1)) runs against the first, $bad differ"
