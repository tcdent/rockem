#!/usr/bin/env bash
# The dump-diff gate, run on the GPU box: the same prompt prefilled in one batch and one
# id at a time must leave byte-identical dumps (logits, per-layer traces, router top-k) at
# every position. `dump_diff.sh <bundle dir> [node args...]`; exit 1 on the first differing position.
# The node under test is target/release/node with the built code object; the serving unit is stopped
# for the run and not restarted — the caller decides what serves afterwards.
set -u
cd "$(dirname "$0")/../.."
B=${1:?bundle dir}; IDS=$(cat bringup/gates/prompt.ids); W=/tmp/rockem-gate
# NODE and ELF: the node binary and code object under test (defaults: the tree's builds; set both to gate another build)
NODE=${NODE:-target/release/node}
ELF=${ELF:-models/deepseek-v2-lite/gpu/target/amdgcn-amd-amdhsa/release/deepseek_v2_lite_gpu.elf}
bringup/gates/install_elf.sh $ELF $B
systemctl --user stop rockem-node 2>/dev/null; pkill -x node 2>/dev/null; sleep 1
pgrep -x python3 -a | grep -E "wire.py|client.*\.py" && { echo "stray wire clients above: the host is not quiet"; exit 1; }
for mode in batch seq; do
  rm -rf $W/$mode; mkdir -p $W/$mode
  $NODE rockem.bundle=$B rockem.mem=anon rockem.gpu=1 rockem.debug=$W/$mode rockem.listen=127.0.0.1:7332 "${@:2}" > $W/$mode.log 2>&1 < /dev/null & PID=$!
  timeout 180 bash -c 'until ss -ltn | grep -q :7332; do sleep 1; done' || { echo "node did not come up"; tail -5 $W/$mode.log; exit 1; }
  # SEQ_SLOW: the pause between ids in the sequential run; 0.05 is one id a step, less makes the node batch whatever arrived
  if [ $mode = seq ]; then SLOW=${SEQ_SLOW:-0.05} CAP=8 python3 bringup/gates/wire.py 127.0.0.1 $IDS; else CAP=8 python3 bringup/gates/wire.py 127.0.0.1 $IDS; fi
  kill $PID; wait 2>/dev/null
done
same=0; bad=0
for f in $W/batch/pos*.f32; do b=$(basename $f .f32); case $b in *.layers) continue;; esac; p=${b#pos}; if cmp -s $f $W/seq/$b.f32 && cmp -s $W/batch/$b.layers.f32 $W/seq/$b.layers.f32 && cmp -s $W/batch/$b.topk.u32 $W/seq/$b.topk.u32; then same=$((same+3)); else bad=$((bad+1)); python3 bringup/gates/dump_where.py $W/batch $W/seq $p; fi; done
echo "dump diff: $same files identical, $bad positions differ"
# a reference from the first identical run; on a failure keep both dumps and say which side left the reference
if [ $bad = 0 ] && [ $same -gt 0 ] && [ ! -d $W/ref ]; then cp -r $W/batch $W/ref; fi
if [ $bad != 0 ]; then
  K=$W/fail-$(date +%H%M%S); mkdir -p $K; cp -r $W/batch $W/seq $K/
  if [ -d $W/ref ]; then
    for side in batch seq; do
      off=0; for f in $W/ref/pos*; do cmp -s $f $W/$side/$(basename $f) || off=$((off+1)); done
      echo "  $side vs the reference: $off files differ"
    done
  fi
  echo "  dumps kept in $K"
fi
[ $bad = 0 ] && [ $same -gt 0 ]
