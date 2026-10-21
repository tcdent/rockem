# Causal online simulation of the layer-ahead predictor with learned corrections.
# Per step: predict every pair (l -> l+1) for the step's tokens, score recall of l+1's picks (union, cap K),
# then update with the labels (BCE per expert). Batched over the 40 layer pairs within a step.
import numpy as np, torch, json, sys, time, argparse
ap=argparse.ArgumentParser()
ap.add_argument('--name',required=True)
ap.add_argument('--terms',default='')       # comma list: cal,bias,rep,co,lr<r>,lrs<r>,lrn<r>,full
ap.add_argument('--lr',type=float,default=1e-3)  # lr for x-dependent terms (low-rank / full) and co-routing
ap.add_argument('--lrb',type=float,default=1e-2) # lr for calibration, bias, repetition
ap.add_argument('--lrc',type=float,default=3e-3) # lr for co-routing
ap.add_argument('--threads',type=int,default=4)
ap.add_argument('--steps',type=int,default=0)
ap.add_argument('--offline',type=float,default=0) # >0: non-causal ceiling: train on the first frac (epochs), test on the rest
ap.add_argument('--epochs',type=int,default=3)
ap.add_argument('--maxrank',type=int,default=0)
ap.add_argument('--data',default='phase3')   # data/<dir>/: phase3 (the first dump) or phase3b (the extended one)
ap.add_argument('--lg',default='LG')         # the route-ahead logits the margin d comes from: LG, LG1, LG2, LG3 (phase3b)
ap.add_argument('--lg2',default='')          # a second margin from these logits, with its own calibration
ap.add_argument('--lg2prev',type=int,default=0) # 1: the second margin from the previous step's last token's logits (known before this step)
a=ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0)
D='/home/slomp/Work/rockem/bringup/expert-cache/data/'+a.data+'/'
X=np.load(D+'X.npy',mmap_mode='r'); SC=np.load(D+'SC.npy'); LG=np.load(D+a.lg+'.npy',mmap_mode='r'); LG2=np.load(D+a.lg2+'.npy',mmap_mode='r') if a.lg2 else None
PK=np.load(D+'PK.npy').astype(np.int64); n=np.load(D+'n.npy'); B=np.load(D+'bias.npy')
P=40; E=288; H=4096
S=len(n) if not a.steps else a.steps
st=np.concatenate([[0],np.cumsum(n)])
bias_next=torch.from_numpy(B[4:44].copy())            # bias of layer l+1 for pairs l=3..42
terms=set(a.terms.split(',')) if a.terms else set()
prm=[];prmb=[]
ca=torch.full((P,1,1),3.0,requires_grad=True); cc=torch.zeros((P,1,1),requires_grad=True); prmb+=[ca,cc]
if a.lg2: ca2=torch.zeros((P,1,1),requires_grad=True); prmb.append(ca2)
learn_cal='cal' in terms or len(terms)>0
if 'bias' in terms: bb=torch.zeros((P,1,E),requires_grad=True); prmb.append(bb)
if 'rep' in terms: rho=torch.zeros((P,1,E),requires_grad=True); prmb.append(rho)
if 'co' in terms: C=torch.zeros((P,E,E),requires_grad=True)
lrk=[t for t in terms if t.startswith('lr')]
if lrk:
    t=lrk[0]; kind=t[:3] if t[2] in 'sn' else 'lr'; r=int(t[len(kind):])
    if kind=='lrs': U=(torch.randn(1,H,r)/H**.5).requires_grad_()       # shared down-projection across layers
    else: U=(torch.randn(P,H,r)/H**.5).requires_grad_()
    V=torch.zeros((P,r,E),requires_grad=True); vb=None
    prm+=[U,V]; nonlin=(kind=='lrn')
if 'full' in terms: W=torch.zeros((P,H,E),requires_grad=True); prm.append(W)
opts=[]
if prm: opts.append(torch.optim.Adam(prm,lr=a.lr))
if learn_cal: opts.append(torch.optim.Adam(prmb,lr=a.lrb))
if 'co' in terms: opts.append(torch.optim.Adam([C],lr=a.lrc))
Ks=[8,13,16]
def feats(s):
    t0,t1=st[s],st[s+1]; m=t1-t0
    lg=torch.from_numpy(np.ascontiguousarray(LG[0:40,t0:t1]))
    g=torch.sigmoid(lg)+bias_next[:,None,:]
    thr=g.topk(8,dim=2).values[:,:,7:8]; d=(g-thr)*10
    y=torch.zeros(P,m,E); y.scatter_(2,torch.from_numpy(PK[1:41,t0:t1]),1.0)
    f=dict(d=d,y=y,m=m)
    if a.lg2:
        src=np.ascontiguousarray(LG2[0:40,t0:t1]) if not a.lg2prev else np.repeat(np.ascontiguousarray(LG2[0:40,max(t0-1,0):max(t0,1)]),t1-t0,axis=1)
        g2=torch.sigmoid(torch.from_numpy(src))+bias_next[:,None,:]
        f['d2']=(g2-g2.topk(8,dim=2).values[:,:,7:8])*10
    if 'co' in terms:
        pl=torch.zeros(P,m,E); pl.scatter_(2,torch.from_numpy(PK[0:40,t0:t1]),1.0); f['pl']=pl
    if lrk or 'full' in terms:
        q=torch.from_numpy(np.ascontiguousarray(X[0:40,t0:t1])).float().view(P,m,32,128)
        x=(q*torch.from_numpy(SC[0:40,t0:t1])[...,None]).view(P,m,H)
        x=x/(x.pow(2).mean(2,keepdim=True).sqrt()+1e-6); f['x']=x
    return f
def logits(f,prev):
    z=ca*f['d']+cc
    if a.lg2: z=z+ca2*f['d2']
    if 'bias' in terms: z=z+bb
    if 'rep' in terms: z=z+rho*prev[:,None,:]
    if 'co' in terms: z=z+torch.bmm(f['pl'],C)
    if lrk:
        h=torch.matmul(f['x'],U)
        if nonlin: h=torch.nn.functional.gelu(h)
        z=z+torch.bmm(h,V)
    if 'full' in terms: z=z+torch.bmm(f['x'],W)
    return z
def score(z,y):
    zu=z.max(1).values if a.maxrank else torch.nn.functional.softplus(z).sum(1)  # -log(1-q), q=1-prod(1-p)
    yu=(y.sum(1)>0).float()
    out=[]
    for K in Ks:
        idx=zu.topk(K,dim=1).indices
        out.append(yu.gather(1,idx).sum(1))
    return torch.stack(out,1), yu.sum(1), yu
def step_update(z,y,m):
    loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='sum')/m/P
    for o in opts: o.zero_grad()
    loss.backward()
    for o in opts: o.step()
    return loss.item()
hits=np.zeros((S,P,len(Ks)),np.float32); tot=np.zeros((S,P),np.float32); bce=np.zeros(S,np.float32)
prev=torch.zeros(P,E); t0=time.time()
if a.offline>0:
    Str=int(S*a.offline)
    for ep in range(a.epochs):
        prev=torch.zeros(P,E)
        for s in np.random.RandomState(ep).permutation(Str) if ep>0 else range(Str):
            f=feats(s); z=logits(f,prev); step_update(z,f['y'],f['m'])
    rng=range(Str,S)
else: rng=range(S)
for s in rng:
    f=feats(s)
    z=logits(f,prev)
    with torch.no_grad():
        h,tt,yu=score(z,f['y']); hits[s]=h.numpy(); tot[s]=tt.numpy()
    if a.offline>0:
        prev=yu; continue
    bce[s]=step_update(z,f['y'],f['m'])
    prev=yu
    if s%1000==0: print(a.name,s,'%.0fs'%(time.time()-t0),flush=True)
np.savez(D+'res_'+a.name+'.npz',hits=hits,tot=tot,bce=bce,n=n[:S])
r=hits[rng].sum((0,1))/tot[rng].sum()
print(a.name,'recall@8/13/16',r.round(4),'time %.0fs'%(time.time()-t0))
