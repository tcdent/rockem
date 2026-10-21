exec(open('replay.py').read())
use=np.array([r['use'] for r in R])
NT=np.array([r['n'] for r in R]); BD=np.array([band(r['l']) for r in R]); L=np.array([r['l'] for r in R])
def summ(o,mask=None):
    if mask is not None: o=o[mask]
    return o[:,1].sum()/o[:,0].sum(), (o[:,2]).mean(), o[:,3].mean()
CAPS=[8,10,13,16]
# history: decayed per-layer counts of host-picked experts, causal
def hist_score(decay):
    H=collections.defaultdict(lambda: collections.defaultdict(float))
    def make(k, cw, pw, hw):
        def f(r,st):
            sc=collections.defaultdict(float)
            for c in r['cand']:
                for i,e in enumerate(c[:k]): sc[e]+=cw*(1000-i)
            for e in r['prev']: sc[e]+=pw*1000
            for e,v in H[r['l']].items(): sc[e]+=hw*1000*v
            return sc
        def upd(r,st,S):
            h=H[r['l']]
            for e in list(h): h[e]*=decay
            for p in r['picks']:
                for e in p: h[e]+=1.0/len(r['picks'])
        return f,upd,H
    return make
variants={}
for k in (8,10,12,16): variants[f'cand top{k}']=(topk_votes(k),None)
# prev alone: score = multiplicity in prev
def prevonly(r,st):
    sc=collections.defaultdict(float)
    for e in r['prev']: sc[e]+=1
    return sc
variants['prev alone']=(prevonly,None)
for k in (8,12,16):
  for pw in (0.005,0.02,0.5,1.0):
    variants[f'top{k} + prev x{pw}']=(topk_votes(k,prevw=pw),None)
res={}
def go(name,f,upd=None, reset=None):
    if reset: reset()
    res[name]={c:(reset() if reset else None, run(f,c,update=upd))[1] for c in CAPS}
for n,(f,u) in variants.items(): go(n,f)
for dec in (0.5,0.8,0.95):
    for k,pw,hw in ((12,1.0,0.5),(12,1.0,1.0),(16,1.0,1.0),(12,0.0,1.0)):
        mk=hist_score(dec); f,u,H=mk(k,1.0,pw,hw)
        res[f'top{k}+prev x{pw}+hist{dec} x{hw}']={c:(H.clear(), run(f,c,update=u))[1] for c in CAPS}
mk=hist_score(0.8); f,u,H=mk(16,0.0,0.0,1.0)
res['hist0.8 alone']={c:(H.clear(), run(f,c,update=u))[1] for c in CAPS}
import pickle; pickle.dump(res,open('res.pkl','wb'))
# oracle
for c in CAPS:
    o=np.array([(len(r['host']),min(c,len(r['host']))) for r in R]); print('oracle cap',c,o[:,1].sum()/o[:,0].sum())
print(f"{'variant':40s}"+''.join(f'  cap{c}: rec  waste' for c in CAPS))
for n,d in res.items():
    print(f'{n:40s}'+''.join('  %5.3f  %5.2f    '%summ(d[c])[:2] for c in CAPS))
