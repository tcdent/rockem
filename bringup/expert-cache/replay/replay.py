import numpy as np, sys
FLOOR=0.05
exec(open('load.py').read())
BANDS=[(4,5),(6,15),(16,30),(31,44)]
def band(l): return next(i for i,(a,b) in enumerate(BANDS) if a<=l<=b)
# per-row host set and residency knowledge
lastres=collections.defaultdict(dict)  # layer -> expert -> resident flag (last observed, incl. current row)
for r in R:
    host=set(); 
    for p,s in zip(r['picks'],r['res']):
        for e,c in zip(p,s):
            lastres[r['l']][e]=(c=='1')
            if c=='0': host.add(e)
    r['host']=host
    r['resk']={e for e,v in lastres[r['l']].items() if v}
# temporal state: history counts per layer (decayed), built causally
def run(scorefn, cap, state_init=None, update=None):
    st=collections.defaultdict(lambda: None)
    out=[]
    for r in R:
        sc=scorefn(r, st)
        ranked=[e for e,v in sorted(sc.items(), key=lambda kv:-kv[1]) if v>FLOOR and e not in r["resk"]][:cap]
        S=set(ranked); hit=len(S&r['host'])
        out.append((len(r['host']),hit,len(S)-hit,len(S)))
        if update: update(r,st,S)
    return np.array(out)
def topk_votes(k, prevw=0.0, histw=0.0, rankw=1.0, lw=None):
    def f(r,st):
        sc=collections.defaultdict(float)
        for c in r['cand']:
            for i,e in enumerate(c[:k]): sc[e]+=1000-rankw*i if lw is None else lw(r['l'])*(1000-i)
        # votes-then-rank: votes*1000 - sum of ranks  (tie-break by rank)
        for e in r['prev']: sc[e]+=prevw*1000
        if histw:
            h=HIST[r['l']]
            for e,v in h.items(): sc[e]+=histw*1000*v
        return sc
    return f
