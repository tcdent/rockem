exec(open('exp.py').read().split("variants={}")[0])
N=len(R); half=N//2
def calib(rows, perlayer=False):
    # table[key][rank 0..16][prev 0/1] -> (hits, n); rank 16 = not in token's cand16
    T=collections.defaultdict(lambda: np.zeros((17,2,2)))
    for r in rows:
        pv=set(r['prev']); key=r['l'] if perlayer else 0
        for p,c in zip(r['picks'],r['cand']):
            ps=set(p); pos={e:i for i,e in enumerate(c)}
            for i,e in enumerate(c): T[key][i,int(e in pv)]+=[e in ps,1]
            for e in pv:
                if e not in pos: T[key][16,1]+=[e in ps,1]
    return T
def ptab(T,key=0,prior=None):
    t=T[key]; 
    if prior is not None: t=t+prior  # shrink
    return (t[:,:,0]+0.5)/(t[:,:,1]+1)
Tg=calib(R[:half]); P=ptab(Tg)
print('P(pick|rank,prev=0)',np.round(P[:,0],3)); print('P(pick|rank,prev=1)',np.round(P[:,1],3))
def probscore(Pfn, useprev=True, gamma=1.0):
    def f(r,st):
        P=Pfn(r); pv=set(r['prev']) if useprev else set()
        q=collections.defaultdict(lambda:1.0)
        for c in r['cand']:
            seen=set(c)
            for i,e in enumerate(c): q[e]*=(1-P[i,int(e in pv)])**gamma
            for e in pv:
                if e not in seen: q[e]*=(1-P[16,1])**gamma
        return {e:1-v for e,v in q.items()}
    return f
def ev(name,f,rows_mask=None):
    out={c:run(f,c) for c in CAPS}; res[name]=out
    sec=np.arange(N)>=half
    print(f'{name:38s}'+''.join('  %5.3f %5.2f'%summ(out[c])[:2] for c in CAPS)+'   2nd-half cap13 rec %.3f'%summ(out[13],sec)[0])
res={}
print(f"{'variant':38s}"+''.join(f'  cap{c} rec waste' for c in CAPS))
ev('today (top8 votes,rank)',topk_votes(8))
ev('prob, global calib, no prev',probscore(lambda r:P,useprev=False))
ev('prob, global calib, +prev',probscore(lambda r:P))
ev('prob, +prev, gamma .7',probscore(lambda r:P,gamma=0.7))
# per-layer calibration from first half (static)
Tl=calib(R[:half],perlayer=True)
PL={l:ptab(Tl,l,prior=Tg[0]*0.05) for l in range(4,45)}
ev('prob, per-layer calib (1st half), +prev',probscore(lambda r:PL[r['l']]))
# causal adaptive per-layer calibration: running counts, decayed
class Ad:
    def __init__(s,dec): s.dec=dec; s.T=collections.defaultdict(lambda: Tg[0]*0.02)
adapt_res={}
for dec in (0.98,0.995):
    A=collections.defaultdict(lambda: Tg[0]*0.02*1.0)
    def f(r,st,A=A): return probscore(lambda r:(A[r['l']][:,:,0]+0.5)/(A[r['l']][:,:,1]+1))(r,st)
    def upd(r,st,S,A=A,dec=dec):
        one=calib([r])[0]; A[r['l']]=A[r['l']]*dec+one
    out={}
    for c in CAPS:
        A.clear(); out[c]=run(f,c,update=upd)
    res[f'prob, causal per-layer calib d{dec}']=out
    print(f'{"prob, causal per-layer d%s"%dec:38s}'+''.join('  %5.3f %5.2f'%summ(out[c])[:2] for c in CAPS)+'   2nd-half cap13 rec %.3f'%summ(out[13],np.arange(N)>=half)[0])
import pickle; pickle.dump(res,open('res3.pkl','wb'))
