# score.py: recall@13 (the union over a step's tokens against the actual picks) of the dump's columns, by attention kind of
# layer l+1, with today's margin on each; the exact-next-input ceiling from the next layer's own column 0 input is in
# scratch/arch (expD). Run after build.py.
import numpy as np, torch, sys
sys.path.insert(0, '/home/slomp/Work/rockem/bringup/expert-cache/scratch/arch')
D='/home/slomp/Work/rockem/bringup/expert-cache/data/phase3c/'
PK=np.load(D+'PK.npy').astype(np.int64); n=np.load(D+'n.npy'); E=288
LG=[np.load(D+f'LG{k}.npy',mmap_mode='r') for k in ['',1,2,3,4,5,6]]
B=np.load('/home/slomp/Work/rockem/bringup/expert-cache/data/arch/B.npy') if __import__('os').path.exists('/home/slomp/Work/rockem/bringup/expert-cache/data/arch/B.npy') else None
st=np.concatenate([[0],np.cumsum(n)]); S=len(n); L=PK.shape[0]
MLA_NEXT={j for j in range(L) if (j+4) in (7,11,15,19,23,27,31,35,39,43)}   # j = layer l-3; l+1 MLA
def margin(j,lg):
    g=torch.sigmoid(torch.from_numpy(np.ascontiguousarray(lg)))
    if B is not None: g=g+torch.from_numpy(B[j+4])  # B by model layer: row j is layer j+3, the logits are layer j+4's router
    thr=g.topk(8,dim=-1).values[...,7:8]; return 3.0*(g-thr)*10
def recall(k,layers,cap=13):
    h=tot=0
    for j in layers:
        if j+1>=L: continue
        sc=margin(j,LG[k][j]); lab=PK[j+1]
        sp=torch.nn.functional.softplus(sc)
        for s in range(S):
            a,b=st[s],st[s+1]; u=sp[a:b].sum(0); y=torch.zeros(E); y[lab[a:b].reshape(-1)]=1
            h+=y[u.topk(cap).indices].sum().item(); tot+=y.sum().item()
    return h/tot
kda=[j for j in range(L-1) if j not in MLA_NEXT]; mla=[j for j in range(L-1) if j in MLA_NEXT]
names=['route-ahead over x_l (today)','x+shared (second margin)','+resident rows','+all rows (FFN bound)','attn-ahead: exact FFN, step-start state (bound)','attn-ahead: shared, full step, state carried','attn-ahead: shared, READ-OUT ONLY, step-start (the kernel)']
# sanity: each column's logit scale and its per-token top-8 overlap with the route-ahead, on a KDA-next layer
j0=kda[len(kda)//2]
for k in range(7):
    a=np.asarray(LG[k][j0][:400]); b=np.asarray(LG[0][j0][:400])
    ta=np.argsort(-a,axis=1)[:,:8]; tb=np.argsort(-b,axis=1)[:,:8]
    ov=np.mean([len(set(x)&set(y)) for x,y in zip(ta,tb)])
    print(f"col {k} layer {j0+3}: mean {a.mean():7.3f} std {a.std():6.3f} top8-overlap-with-col0 {ov:4.2f}")
print(f"{'column':52} {'all':>6} {'l+1 KDA':>8} {'l+1 MLA':>8}")
for k in range(7):
    print(f"{names[k]:52} {recall(k,range(L-1)):6.3f} {recall(k,kda):8.3f} {recall(k,mla):8.3f}")
