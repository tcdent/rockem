exec(open('exp3.py').read().split("res={}")[0])
best=probscore(lambda r:P)
o_t=run(topk_votes(8),13); o_b=run(best,13)
print('cap13 by tokens: n  host  today rec/waste  best rec/waste  oracle')
for n in (1,2,3,4):
    m=NT==n; orc=np.minimum(o_t[m,0],13).sum()/o_t[m,0].sum()
    print(n, '%5.1f'%o_t[m,0].mean(), '%.3f %.2f'%summ(o_t,m)[:2], '%.3f %.2f'%summ(o_b,m)[:2], '%.3f'%orc, 'pred %.1f'%o_t[m,3].mean())
print('cap13 by band')
for b,(lo,hi) in enumerate(BANDS):
    m=BD==b; orc=np.minimum(o_t[m,0],13).sum()/o_t[m,0].sum()
    print((lo,hi), '%5.1f'%o_t[m,0].mean(), '%.3f %.2f'%summ(o_t,m)[:2], '%.3f %.2f'%summ(o_b,m)[:2], '%.3f'%orc)
# all caps by tokens for today & best
for nm,f in (('today',topk_votes(8)),('best',best)):
    for c in CAPS:
        o=run(f,c); print(nm,c,' '.join('n%d %.3f/%.2f'%(n,*summ(o,NT==n)[:2]) for n in (1,2,3,4)), ' | ', ' '.join('%.3f/%.2f'%summ(o,BD==b)[:2] for b in range(4)))
# threshold rule: prefetch experts with p>=th, cap 16
print('threshold rule (prob+prev), hard cap 16')
for th in (0.2,0.3,0.4,0.5,0.6):
    def f(r,st,th=th):
        s=best(r,st); return {e:v for e,v in s.items() if v>=th}
    o=run(f,16); print(th,'rec %.3f waste %.2f size %.2f'%summ(o), ' by n:',' '.join('%.3f/%.1f'%(summ(o,NT==n)[0],summ(o,NT==n)[2]) for n in (1,2,3,4)))
