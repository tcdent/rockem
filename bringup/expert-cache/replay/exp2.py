exec(open('exp.py').read().split("variants={}")[0])
# per-layer: unlimited-cap recall of cand top8 union, prev, and per-token pick-level recall
pl=collections.defaultdict(lambda: np.zeros(6))
for r in R:
    c8=set(e for c in r['cand'] for e in c[:8]); c16=set(e for c in r['cand'] for e in c[:16]); pv=set(r['prev'])
    # token-level: fraction of a token's picks in its own top8
    tok=sum(len(set(p)&set(c[:8])) for p,c in zip(r['picks'],r['cand'])); ntok=8*r['n']
    h=r['host']; pl[r['l']]+=[len(h), len(h&c8), len(h&c16), len(h&pv), tok, ntok]
print('layer host  c8-union c16-union prev  token-top8-acc')
for l in sorted(pl):
    a=pl[l]; print(l, '%5.1f'%(a[0]/231), '%.3f %.3f %.3f %.3f'%(a[1]/a[0],a[2]/a[0],a[3]/a[0],a[4]/a[5]))
# within-layer: candidate rank position hit rate (token level)
hits=np.zeros(16); 
for r in R:
    for p,c in zip(r['picks'],r['cand']):
        for i,e in enumerate(c): hits[i]+= e in p
print('P(pick | cand rank i)', np.round(hits/sum(r['n'] for r in R),3))
# prev given in cand vs not
a=b=cc=dd=0
for r in R:
    c8=set(e for c in r['cand'] for e in c[:8]); pv=set(r['prev']); allp=set(e for p in r['picks'] for e in p)
    for e in pv:
        if e in c8: a+=1; b+= e in allp
        else: cc+=1; dd+= e in allp
print('P(picked | prev & in c8)=%.3f n=%d ; P(picked | prev & not c8)=%.3f n=%d'%(b/a,a,dd/cc,cc))
