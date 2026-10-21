#!/bin/bash
# dumppred.sh: the dump-pred lab boot (expert-cache phase 3 data), 24 decode requests on the 18k context, the box's
# /tmp/pred.bin fetched to bringup/expert-cache/data/pred.bin
set -u; cd /home/slomp/Work/rockem; BOX=/home/slomp/Work/rockem/bringup/box; LOG=$BOX/logs; REQ=$BOX/requests; DATA=/home/slomp/Work/rockem/bringup/expert-cache/data
timeout 600 make initramfs NODE_FEATURES="--features track-launches,dump-pred" 2>&1 | tail -n 1 | cut -c1-80
timeout 180 make publish 2>&1 | /usr/bin/grep -E 'pushed|error'
: > $LOG/metatron-console.log
(sleep 1; printf 'echo b > /proc/sysrq-trigger\r\n'; sleep 2) | timeout 8 nc 192.168.88.6 23 >/dev/null 2>&1
echo "rebooted $(date +%H:%M:%S)"; sleep 60
until timeout 2 bash -c 'echo > /dev/tcp/192.168.88.6/7332' 2>/dev/null; do sleep 5; done; echo "node up $(date +%H:%M:%S)"
for p in $(pgrep -f "[u]vicorn rockem.app"); do kill $p; done; sleep 2
tmux send-keys -t rockem:2.10 "cd /home/slomp/Work/rockem/api && ROCKEM_BUNDLE=/home/slomp/Work/rockem/bundles/glm-5.3-flash-q4rtn ROCKEM_NODES=192.168.88.6:7332 uv run uvicorn rockem.app:app --host 192.168.88.2 --port 8765" Enter; sleep 12
curl -s -m 300 http://192.168.88.2:8765/v1/chat/completions -H 'content-type: application/json' -d @$REQ/cody-user.json > /dev/null; echo "prompt cached $(date +%H:%M:%S)"
for f in $REQ/predreq/q*.json; do out=$(curl -s -m 300 http://192.168.88.2:8765/v1/chat/completions -H 'content-type: application/json' -d @$f); echo "$(basename $f): $(echo "$out" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("usage",{}).get("completion_tokens"))' 2>&1)"; done
(timeout 120 nc -l -p 9977 > $DATA/pred.bin &) ; sleep 1
$BOX/box.sh 'ls -la /tmp/pred.bin; cat /tmp/pred.bin | nc 192.168.88.2 9977' 100
sleep 2; ls -la $DATA/pred.bin; echo done
