#!/usr/bin/env bash
# The node over the wire without rockem.debug, again and again: `score_repeat.sh <bundle dir> [runs] [node args...]`.
# Each run SCOREs the gate's prompt on a fresh node and prints the ln p bits; runs are compared with the
# first. Observes the forward through the logits alone — no dumps, no trace reads.
set -u
cd "$(dirname "$0")/../.."
B=${1:?bundle dir}; R=${2:-20}; NODE=${NODE:-target/release/node}; IDS=$(cat bringup/gates/prompt.ids); W=/tmp/rockem-gate/score-repeat
ELF=${ELF:-models/deepseek-v2-lite/gpu/target/amdgcn-amd-amdhsa/release/deepseek_v2_lite_gpu.elf}
bringup/gates/install_elf.sh $ELF $B
systemctl --user stop rockem-node 2>/dev/null; pkill -x node 2>/dev/null; sleep 1
rm -rf $W; mkdir -p $W
bad=0
for i in $(seq 1 $R); do
  $NODE rockem.bundle=$B rockem.mem=anon rockem.gpu=1 rockem.listen=127.0.0.1:7332 "${@:3}" > $W/run$i.log 2>&1 < /dev/null & PID=$!
  timeout 180 bash -c 'until ss -ltn | grep -q :7332; do sleep 1; done' || { echo "run $i: node did not come up"; tail -3 $W/run$i.log; kill $PID; wait 2>/dev/null; continue; }
  python3 bringup/gates/score_client.py 127.0.0.1 $IDS > $W/run$i.lp
  kill $PID; wait 2>/dev/null
  [ $i = 1 ] && continue
  if ! cmp -s $W/run1.lp $W/run$i.lp; then bad=$((bad+1)); echo "run $i differs: $(python3 -c "
a=open('$W/run1.lp').read().split(); b=open('$W/run$i.lp').read().split()
d=[k for k in range(len(a)) if a[k]!=b[k]]; print(f'{len(d)} of {len(a)} scores, first at id index {d[0] if d else -1}')")"; fi
done
echo "score repeat: $((R - 1)) runs against the first, $bad differ"
