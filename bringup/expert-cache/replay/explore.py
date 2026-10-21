exec(open('load.py').read())
print(collections.Counter(r['l'] for r in R).most_common(3), min(r['l'] for r in R), max(r['l'] for r in R))
print(collections.Counter(r['n'] for r in R))
print(R[100]['picks'],R[100]['cand'][0][:8])
# score-order? check if sorted
print(sum(all(p==sorted(p) for p in r['picks']) for r in R)/len(R))
print(sum(len(r['cand'])==r['n'] and len(r['picks'])==r['n'] and len(r['res'])==r['n'] for r in R))
# request starts: prev empty
starts=[i for i,r in enumerate(R) if not r['prev']]; print(len(starts), [R[i]['l'] for i in starts[:20]])
# gaps
ts=[r['t'] for r in R]; import numpy as np; g=np.diff(ts); print(np.percentile(g,[50,90,99,99.9]), (g>1e6).sum())
# resident count per layer-run
print(np.mean([sum(c=='1' for s in r['res'] for c in s) for r in R]))
