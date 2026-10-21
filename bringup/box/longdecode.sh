#!/bin/bash
# longdecode.sh <label>: on the box as booted (no rebuild), the 18k coding prompt cached, then the 24 decode requests of
# requests/predreq (450 tokens each, ~10.8k decode tokens: past the prefetch predictor's settling), then the prefetch
# yield over the run's decode (bringup/reads/prefetch_use.sh): an accuracy A/B between builds on the same tokens.
set -u; V=$1; cd /home/slomp/Work/rockem; REQ=bringup/box/requests; R=bringup/reads
T0=$(date -u +'%Y-%m-%d %H:%M:%S')
curl -s -m 300 http://192.168.88.2:8765/v1/chat/completions -H 'content-type: application/json' -d @$REQ/cody-user.json > /dev/null; echo "prompt cached $(date +%H:%M:%S)"
for f in $REQ/predreq/q*.json; do out=$(curl -s -m 300 http://192.168.88.2:8765/v1/chat/completions -H 'content-type: application/json' -d @$f); echo "$(basename $f): $(echo "$out" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("usage",{}).get("completion_tokens"))' 2>&1)"; done
echo "--- $V: prefetch yield over the run's decode ---"; $R/prefetch_use.sh "" "$T0"
echo "--- second half ---"; T1=$(date -u -d "@$(( ($(date -u -d "$T0" +%s) + $(date -u +%s)) / 2 ))" +'%Y-%m-%d %H:%M:%S'); $R/prefetch_use.sh "" "$T1"
$R/requests.sh "" 26 | tail -n 8
