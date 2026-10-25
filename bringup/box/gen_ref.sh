#!/bin/bash
# gen_ref.sh <label>: a generation to compare boots by — the deploy's cold 18k prompt, then its 512-token decode on the
# cache hit — through the API's chat endpoint, the whole responses kept as bringup/box/logs/gen-<label>-{cold,decode}.json
# with the wall times. Sampling is seeded by position, so two boots whose decode path computes the same bits write the
# same text (029's exp4 check against the bf16 path). `gen_ref.sh cmp <a> <b>` compares two labels.
set -u
cd "$(dirname "$0")"
REQ=requests; LOG=logs
if [ "$1" = cmp ]; then
  for part in cold decode; do
    python3 - "$LOG/gen-$2-$part.json" "$LOG/gen-$3-$part.json" "$part" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
ma, mb = a["choices"][0]["message"], b["choices"][0]["message"]
same = all(ma.get(k) == mb.get(k) for k in ("content", "reasoning_content", "tool_calls"))
ta = (ma.get("reasoning_content") or "") + (ma.get("content") or "")
tb = (mb.get("reasoning_content") or "") + (mb.get("content") or "")
at = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
print(f"{sys.argv[3]}: identical {same}; {a['usage']['completion_tokens']} vs {b['usage']['completion_tokens']} tokens; first differing character {at if not same else '-'} of {len(ta)}")
PY
  done
  exit 0
fi
for part in cold:cody-user decode:cody-decode512; do
  name=${part%%:*}; file=${part#*:}
  t0=$(date +%s.%N)
  curl -s -m 600 http://192.168.88.3:8765/v1/chat/completions -H 'content-type: application/json' -d @$REQ/$file.json > $LOG/gen-$1-$name.json
  t1=$(date +%s.%N)
  echo "$1 $name: wall $(echo "$t1 - $t0" | bc) s; $(python3 -c 'import sys,json; d=json.load(open(sys.argv[1])); u=d["usage"]; print("prompt", u["prompt_tokens"], "completion", u["completion_tokens"])' $LOG/gen-$1-$name.json)"
done
