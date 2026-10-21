# Tables and rolling curves from res_*.npz: recall@8/13/16 overall, settled (positions >= 9000), by band; CSV out.
import numpy as np, glob, sys, os
D='/home/slomp/Work/rockem/bringup/expert-cache/data/phase3/'
names=sys.argv[1:] or sorted(os.path.basename(f)[4:-4] for f in glob.glob(D+'res_*.npz'))
bands={'3-15':slice(0,13),'16-29':slice(13,27),'30-42':slice(27,40)}
ref=np.load(D+'res_base0.npz') if os.path.exists(D+'res_base0.npz') else None
rows=[];curves={}
for nm in names:
    r=np.load(D+f'res_{nm}.npz'); h,t,n=r['hits'],r['tot'],r['n']
    cum=np.cumsum(n); S=len(n)
    sel=cum>=9000
    if h[sel].sum()==0: continue
    tot=lambda m,b=slice(None): h[m][:,b].sum((0,1))/t[m][:,b].sum()
    allr=tot(slice(None)); setr=tot(sel)
    w13=(13-h[sel][:,:,1]).mean()
    row=[nm]+list(allr.round(4))+list(setr.round(4))+[round(float(tot(sel,b)[1]),4) for b in bands.values()]+[round(float(w13),2)]
    rows.append(row)
    win=np.minimum(cum//1000,13)
    curves[nm]=[float(h[win==k][:,:,1].sum()/t[win==k].sum()) for k in range(14)]
print('%-12s %6s %6s %6s | %6s %6s %6s | %6s %6s %6s | %s'%('name','all@8','@13','@16','set@8','@13','@16','b3-15','b16-29','b30-42','waste13'))
for r in rows: print('%-12s %6.4f %6.4f %6.4f | %6.4f %6.4f %6.4f | %6.4f %6.4f %6.4f | %.2f'%tuple(r))
print('\nrolling recall@13 by 1k positions')
for nm,c in curves.items(): print('%-12s'%nm,' '.join('%.3f'%v for v in c))
if 'basemax' in curves:
    print('\nrolling gain@13 vs basemax')
    for nm,c in curves.items(): print('%-12s'%nm,' '.join('%+.3f'%(v-b) for v,b in zip(c,curves['basemax'])))
with open(D+'summary.csv','w') as f:
    f.write('name,all8,all13,all16,settled8,settled13,settled16,set13_l3_15,set13_l16_29,set13_l30_42,wasted13\n')
    for r in rows: f.write(','.join(map(str,r))+'\n')
with open(D+'rolling13.csv','w') as f:
    f.write('name,'+','.join(f'k{k}' for k in range(14))+'\n')
    for nm,c in curves.items(): f.write(nm+','+','.join('%.4f'%v for v in c)+'\n')
