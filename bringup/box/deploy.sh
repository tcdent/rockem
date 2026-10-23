#!/bin/bash
# deploy.sh <label> [short]: the prow baseline. Build the initramfs from the tree, publish, reboot metatron, then a
# long-context workload (18k cold prefill, decode at 18k on the cache hit, a warm-box 18k prefill, the long
# SCORE), read back through bringup/reads (prow-sql) with the console kept only for the hardware watch.
set -u; V=$1; ROOT=/home/slomp/Work/rockem; BOX=$ROOT/bringup/box; LOG=$BOX/logs; REQ=$BOX/requests; R=$ROOT/bringup/reads
# a second argument "short": the cold 18k prompt, the 96-token decode and long SCORE only (a CPU-side change)
SHORT=${2:-}
cd /home/slomp/Work/rockem
# the production command line, always (a lab boot leaves its own in os/release/dev.cmdline)
cp bringup/box/dev.cmdline.serving os/release/dev.cmdline
/usr/bin/grep -E "kernels" bundles/glm-5.3-flash-q4rtn/manifest.toml | cut -c1-100
timeout 500 make initramfs NODE_FEATURES="${NODE_FEATURES---features track-launches}" 2>&1 | tail -n 2 | cut -c1-100
timeout 180 make publish 2>&1 | /usr/bin/grep -E 'pushed|error|missing'
: > $LOG/metatron-console.log
(sleep 1; printf 'echo b > /proc/sysrq-trigger\r\n'; sleep 2) | timeout 8 nc 192.168.88.6 23 >/dev/null 2>&1
echo "rebooted $(date +%H:%M:%S)"
sleep 60
until timeout 2 bash -c 'echo > /dev/tcp/192.168.88.6/7332' 2>/dev/null; do sleep 5; done; echo "node up $(date +%H:%M:%S)"
/usr/bin/grep -E "spec     :|panicked|cmdline|prow" $LOG/metatron-console.log | sed -E 's/.*\] //' | cut -c1-200
tmux send-keys -t rockem:2.10 C-c; sleep 3; tmux send-keys -t rockem:2.10 C-c; sleep 2
tmux send-keys -t rockem:2.10 "cd /home/slomp/Work/rockem/api && ROCKEM_BUNDLE=/home/slomp/Work/rockem/bundles/glm-5.3-flash-q4rtn ROCKEM_MODEL=glm-5.3-flash ROCKEM_TRANSCRIBE_MODEL=voxtral-small-24b ROCKEM_NODES=192.168.88.6:7332 uv run uvicorn rockem.app:app --host 192.168.88.2 --port 8765" Enter; sleep 10
T0=$(date -u +'%Y-%m-%d %H:%M:%S')
req() { # req <label> <json>: one request, wall time, the hardware watch over the console
  L=$LOG/metatron-console.log; n0=$(wc -l < $L); t0=$(date +%s.%N)
  out=$(curl -s -m 300 http://192.168.88.2:8765/v1/chat/completions -H 'content-type: application/json' -d @$2); t1=$(date +%s.%N)
  echo "$1: wall $(echo "$t1 - $t0" | bc) s; $(echo "$out" | python3 -c 'import sys,json; d=json.load(sys.stdin); u=d.get("usage",{}); print("prompt", u.get("prompt_tokens"), "completion", u.get("completion_tokens"))' 2>&1 | head -1)"
  w=$(( n0 > 40 ? n0 - 40 : 1 )); sleep 2
  apei=$(tail -n +$w $L | /usr/bin/grep -c "Hardware error from APEI"); aer=$(tail -n +$w $L | /usr/bin/grep -cE "pcieport .*aer_status|AER: .*error received|Corrected error|Uncorrected")
  echo "hw watch : $apei APEI records, $aer AER lines"
}
echo "--- 18k cold ---";   req "cody 18k cold" $REQ/cody-user.json
echo "--- 18k decode ---"; req "cody 18k hit, 96 tokens" $REQ/cody-decode.json
[ "$SHORT" = short ] || { echo "--- 18k decode 512 ---"; req "cody 18k hit, up to 512 tokens" $REQ/cody-decode512.json; }
[ "$SHORT" = short ] || { echo "--- 18k warm ---";   req "cody2 18k warm box" $REQ/cody-user2.json; }
echo "--- long SCORE ---"
cd /home/slomp/Work/rockem/api && t0=$(date +%s); timeout 600 uv run python ../bringup/gates/score_client.py 192.168.88.6 /home/slomp/Work/rockem/bundles/glm-5.3-flash-q4rtn glm-5.3-flash $(cat $REQ/score-long.ids) > $LOG/score-long-${V}-ld.lp 2>/dev/null; echo "SCORE 3632 ids: $(( $(date +%s) - t0 )) s"; cmp $LOG/score-long-rw1-ld.lp $LOG/score-long-${V}-ld.lp && echo "long SCORE bit-identical"
sleep 8
echo "--- prow reads ---"; $R/runs.sh 2; $R/requests.sh "" 6; $R/steps.sh "" "$T0"
echo done
