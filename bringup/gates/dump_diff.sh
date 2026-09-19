#!/usr/bin/env bash
# The dump-diff gate, run on the GPU box: the same prompt prefilled in one batch and one
# id at a time must leave byte-identical dumps (logits, per-layer traces, router top-k) at
# every position. `dump_diff.sh <bundle dir>`; exit 1 on the first differing position.
# The node under test is target/release/node (the musl build); the serving unit is stopped
# for the run and not restarted — the caller decides what serves afterwards.
set -u
cd "$(dirname "$0")/../.."
B=${1:?bundle dir}; NODE=target/release/node; IDS=$(cat bringup/gates/prompt.ids); W=/tmp/rockem-gate
systemctl --user stop rockem-node 2>/dev/null; pkill -x node 2>/dev/null; sleep 1
pgrep -af "gates/wire.py|client.*\.py" && { echo "stray wire clients above: the host is not quiet"; exit 1; }
for mode in batch seq; do
  rm -rf $W/$mode; mkdir -p $W/$mode
  $NODE rockem.bundle=$B rockem.mem=anon rockem.gpu=1 rockem.debug=$W/$mode rockem.listen=127.0.0.1:7332 > $W/$mode.log 2>&1 < /dev/null & PID=$!
  timeout 180 bash -c 'until ss -ltn | grep -q :7332; do sleep 1; done' || { echo "node did not come up"; tail -5 $W/$mode.log; exit 1; }
  if [ $mode = seq ]; then SLOW=0.05 CAP=8 python3 bringup/gates/wire.py 127.0.0.1 $IDS; else CAP=8 python3 bringup/gates/wire.py 127.0.0.1 $IDS; fi
  kill $PID; wait 2>/dev/null
done
same=0; bad=0
for f in $W/batch/pos*; do b=$(basename $f); if cmp -s $f $W/seq/$b; then same=$((same+1)); else bad=$((bad+1)); echo "DIFF $b"; fi; done
echo "dump diff: $same files identical, $bad differ"
[ $bad = 0 ] && [ $same -gt 0 ]
