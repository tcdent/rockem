#!/bin/bash
# transcribe_https.sh <wav> [language]: one transcription through the service (https://rockem.a10k.co), the wall time
# and the text, the way the edge sends it (docs/queue/094).
f=${1:?wav}; lang=${2:-en}
body=$(mktemp); trap 'rm -f $body' EXIT
python3 -c "import base64,json,sys; print(json.dumps({'audio': base64.b64encode(open(sys.argv[1],'rb').read()).decode(), 'language': sys.argv[2]}))" "$f" "$lang" > $body
t0=$(date +%s.%N)
out=$(curl -s -m 300 -N -H 'content-type: application/json' --data-binary @$body https://rockem.a10k.co/rockem/v1/transcribe)
t1=$(date +%s.%N)
echo "$out" | python3 -c '
import sys, json
text = ""; usage = None
for line in sys.stdin:
    if not line.startswith("data:"): continue
    d = line[5:].strip()
    if d in ("", "[DONE]"): continue
    e = json.loads(d)
    text += e.get("text", "") if isinstance(e.get("text"), str) else ""
    usage = e.get("usage", usage)
print("text  :", text.strip())
print("usage :", usage)'
echo "wall  : $(echo "$t1 - $t0" | bc) s"
