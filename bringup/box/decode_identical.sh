#!/bin/bash
# decode_identical.sh [run]: the decode-identity gate on the cody-decode requests (the same 18k prompt, a cache hit after
# the first; deploy.sh sends four, the an*_gates workloads 24). PASS = at least one request on the reference path. The fingerprint is the speculation path — spec_step's (pos, made, verified, accepted) sequence of the
# steady-state requests (hits 2–24; the cold request and the first hit follow their own paths, consistently) — compared
# with the reference taken from an13 (run 4a6ba8f4, 2026-10-27; identical on an12, an13s and the an13 restore across
# boots and PCIe error bursts). SCORE cannot see a kernel that runs only at decode widths (the analog); this can: an16
# departed at the FIRST speculative step (made 2 / verified 1 vs 3 / 2), i.e. a changed computation, not a near-tie.
set -u; ROOT=/home/slomp/Work/rockem; B=$ROOT/bringup; cd $ROOT; . $B/reads/_run.sh
RUN=${1:-$(pick_run "")}; REF=$B/gates/cody-decode.fingerprint
q "SELECT epoch_ms(ts) t, pos, made, verified, accepted FROM spec_step WHERE run LIKE '$RUN%' ORDER BY ts" | grep -E "^ *[0-9]" > /tmp/claude-1000/spec-gate.txt
python3 - "$REF" "${2:-}" <<'PY'
import sys, collections
ref_path, make = sys.argv[1], sys.argv[2]
rows=[tuple(int(x) for x in l.split()) for l in open('/tmp/claude-1000/spec-gate.txt')]
reqs=[]; cur=[]; last=None
for t,pos,m,v,a in rows:
    if last is not None and pos<last: reqs.append(cur); cur=[]
    cur.append((pos,m,v,a)); last=pos
if cur: reqs.append(cur)
fps=[' '.join(f"{p}:{m}{v}{a}" for p,m,v,a in r) for r in reqs]
c=collections.Counter(fps); common,n=c.most_common(1)[0]
if make=="--make-reference":
    open(ref_path,'w').write(common+"\n"); print(f"reference written: {n} of {len(reqs)} requests share the path ({len(reqs[fps.index(common)])} steps)"); sys.exit(0)
ref=open(ref_path).read().strip()
on=sum(1 for f in fps if f==ref)
if on>=1: print(f"decode-identical: {on} of {len(reqs)} requests on the reference path ({ref.count(' ')+1} steps; the cold request and the first two hits follow their own paths)")
else:
    a=common.split(); b=ref.split(); i=next((k for k in range(min(len(a),len(b))) if a[k]!=b[k]), min(len(a),len(b)))
    print(f"DECODE DIFFERS: 0 of {len(reqs)} requests on the reference path; {n} share a path of {len(a)} steps vs the reference's {len(b)}; first difference at step {i}: {a[i] if i<len(a) else '-'} vs {b[i] if i<len(b) else '-'}"); sys.exit(1)
PY
