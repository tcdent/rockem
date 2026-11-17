#!/bin/bash
# decode_identical.sh [run]: the decode-identity gate on the cody-decode requests (the same 18k prompt, a cache hit after
# the first; deploy.sh sends four, the an*_gates workloads 24). PASS = at least one request on the reference path. The fingerprint is the speculation path — spec_step's (pos, made, verified, accepted) sequence of the
# steady-state requests (hits 2–24; the cold request and the first hit follow their own paths, consistently) — compared
# with the reference taken from an13 (run 4a6ba8f4, 2026-10-27; identical on an12, an13s and the an13 restore across
# boots and PCIe error bursts). SCORE cannot see a kernel that runs only at decode widths (the analog); this can: an16
# departed at the FIRST speculative step (made 2 / verified 1 vs 3 / 2), i.e. a changed computation, not a near-tie.
set -u; ROOT=/home/slomp/Work/rockem; B=$ROOT/bringup; cd $ROOT; . $B/reads/_run.sh
RUN=${1:-$(pick_run "")}; REF=$B/gates/cody-decode.fingerprint; COLD=$B/gates/cody-decode-cold.fingerprint
# MIN_POS (default 18000): only steps at or past it — the cody requests' (the streams gate's prompts sit near 3.6k)
MIN_POS=${MIN_POS:-18000}
q "SELECT epoch_ms(ts) t, coalesce(slot, 0) slot, pos, made, verified, accepted FROM spec_step WHERE run LIKE '$RUN%' AND pos >= $MIN_POS ORDER BY slot, ts" 2>/dev/null | grep -E "^ *[0-9]" > /tmp/claude-1000/spec-gate.txt
# a run without the slot column (a build before 102's slot node): slot 0 throughout
[ -s /tmp/claude-1000/spec-gate.txt ] || q "SELECT epoch_ms(ts) t, 0 slot, pos, made, verified, accepted FROM spec_step WHERE run LIKE '$RUN%' AND pos >= $MIN_POS ORDER BY ts" | grep -E "^ *[0-9]" > /tmp/claude-1000/spec-gate.txt
python3 - "$REF" "${2:-}" "$COLD" <<'PY'
import sys, collections
ref_path, make, cold_path = sys.argv[1], sys.argv[2], sys.argv[3]
rows=[tuple(int(x) for x in l.split()) for l in open('/tmp/claude-1000/spec-gate.txt')]
# a request is a run of rising positions on one slot (102: the rows are ordered by slot, then time)
reqs=[]; cur=[]; last=None; lslot=None
for t,slot,pos,m,v,a in rows:
    if last is not None and (pos<last or slot!=lslot): reqs.append(cur); cur=[]
    cur.append((pos,m,v,a)); last=pos; lslot=slot
if cur: reqs.append(cur)
fps=[' '.join(f"{p}:{m}{v}{a}" for p,m,v,a in r) for r in reqs]
c=collections.Counter(fps); common,n=c.most_common(1)[0]
# the cold request (the first of the run: the prompt computed, its drafts made from its last step) is deterministic on
# its own: its path is a second reference, independent of what a resumed sequence verifies first
if make=="--make-cold-reference":
    open(cold_path,'w').write(fps[0]+"\n"); print(f"cold reference written: {len(reqs[0])} steps"); sys.exit(0)
import os
if os.path.exists(cold_path):
    cold=open(cold_path).read().strip()
    if fps and fps[0]==cold: print(f"cold-identical: the first request's {len(reqs[0])} steps are the reference's")
    else:
        a=fps[0].split() if fps else []; b=cold.split(); i=next((k for k in range(min(len(a),len(b))) if a[k]!=b[k]), min(len(a),len(b)))
        print(f"COLD DIFFERS: {len(a)} steps vs the reference's {len(b)}; first difference at step {i}: {a[i] if i<len(a) else '-'} vs {b[i] if i<len(b) else '-'}")
if make=="--make-reference":
    open(ref_path,'w').write(common+"\n"); print(f"reference written: {n} of {len(reqs)} requests share the path ({len(reqs[fps.index(common)])} steps)"); sys.exit(0)
ref=open(ref_path).read().strip()
on=sum(1 for f in fps if f==ref)
if on>=1: print(f"decode-identical: {on} of {len(reqs)} requests on the reference path ({ref.count(' ')+1} steps; the cold request and the first two hits follow their own paths)")
else:
    a=common.split(); b=ref.split(); i=next((k for k in range(min(len(a),len(b))) if a[k]!=b[k]), min(len(a),len(b)))
    print(f"DECODE DIFFERS: 0 of {len(reqs)} requests on the reference path; {n} share a path of {len(a)} steps vs the reference's {len(b)}; first difference at step {i}: {a[i] if i<len(a) else '-'} vs {b[i] if i<len(b) else '-'}"); sys.exit(1)
PY
