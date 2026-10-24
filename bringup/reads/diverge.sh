#!/usr/bin/env bash
# Where a request stopped matching the node's cached transcript (`store_diverge`, track-store): `diverge.sh [run] [n]`.
# For the last `n` divergences: the position, the transcript's length, then the cached text (from 8 ids before the
# divergence; `|` marks it) and the request's text from the divergence — decoded through the GLM bundle's vocab.bin.
set -u; . "$(dirname "$0")/_run.sh"
RUN=$(pick_run "${1:-}"); [ -n "$RUN" ] || { echo "no run" >&2; exit 1; }
VOCAB=${VOCAB:-$(dirname "$0")/../../bundles/glm-5.3-flash-q4rtn/glm-5.3-flash/vocab.bin}
q "SELECT strftime(ts, '%H:%M:%S') AS t, pos, transcript, cached, new FROM store_diverge WHERE run = '$RUN' ORDER BY ts DESC LIMIT ${2:-5}" | python3 -c '
import sys, struct
vb = open(sys.argv[1], "rb").read()
nv = int(sys.argv[2]) + 1  # the table: vocab_size + 1 u32 offsets, then the bytes
offs = struct.unpack_from(f"<{nv}I", vb, 0)
base = 4 * nv
def tok(t):
    t = int(t)
    return vb[base + offs[t]: base + offs[t + 1]].decode("utf-8", "replace") if t + 1 < len(offs) else f"<{t}>"
for line in sys.stdin:
    parts = line.rstrip("\n").split("  ")
    cols = [p.strip() for p in parts if p.strip()]
    if len(cols) < 5 or not cols[1].isdigit():
        print(line.rstrip()); continue
    t, pos, tr, cached, new = cols[0], cols[1], cols[2], cols[3], cols[4]
    c = cached.split(); w = new.split()
    k = min(8, int(pos))
    print(f"{t}  pos {pos} of {tr}")
    print("  cached: " + repr("".join(tok(x) for x in c[:k])) + " | " + repr("".join(tok(x) for x in c[k:])))
    print("  new   :                     | " + repr("".join(tok(x) for x in w)))
    print("  ids   : cached " + " ".join(c[k:k+8]) + "  /  new " + " ".join(w[:8]))
' "$VOCAB" "${VOCAB_N:-154880}"
