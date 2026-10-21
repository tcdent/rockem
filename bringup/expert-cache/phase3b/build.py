# The extended dump (dumppred.sh pred2.bin, 2026-10-21): per decode layer record, per token, the picks, the FFN input
# (scales, int8), layer l+1's route-ahead logits over the FFN input (LG), then over the FFN input plus the shared
# expert's output (LG1), plus the resident experts' rows too (LG2), plus every routed row: the whole FFN output (LG3).
# Writes dense arrays [layer j (0..40 = layers 3..43), token t] to data/phase3b/; steps in order.
import numpy as np, sys
SRC='/home/slomp/Work/rockem/bringup/expert-cache/data/pred2.bin'
D='/home/slomp/Work/rockem/bringup/expert-cache/data/phase3b/'
H=4096; E=288; MORE=3
TOK=16+128+H+E*4+MORE*E*4
m=np.memmap(SRC,np.uint8,'r'); N=len(m)
offs=[];lay=[];ns=[]
o=0
while o<N:
    l,n=int(m[o])|int(m[o+1])<<8, int(m[o+2])|int(m[o+3])<<8
    offs.append(o);lay.append(l);ns.append(n); o+=4+n*TOK
assert o==N,(o,N)
offs=np.array(offs);lay=np.array(lay);ns=np.array(ns)
L=41; S=len(offs)//L
assert len(offs)==S*L
lay2=lay.reshape(S,L); ns2=ns.reshape(S,L)
assert (lay2==np.arange(3,44)[None]).all() and (ns2==ns2[:,:1]).all()
n=ns2[:,0]; T=int(n.sum()); start=np.concatenate([[0],np.cumsum(n)])
X=np.lib.format.open_memmap(D+'X.npy','w+',np.int8,(L,T,H))
SC=np.zeros((L,T,32),np.float32); PK=np.zeros((L,T,8),np.int16)
LGS=[np.lib.format.open_memmap(D+f'LG{k}.npy','w+',np.float32,(L,T,E)) for k in ['',1,2,3]]
for s in range(S):
    for j in range(L):
        o=offs[s*L+j]+4
        blk=np.asarray(m[o:o+n[s]*TOK]).reshape(n[s],TOK)
        t0=start[s]; t1=start[s+1]
        PK[j,t0:t1]=blk[:,:16].copy().view(np.int16)
        SC[j,t0:t1]=blk[:,16:144].copy().view(np.float32)
        X[j,t0:t1]=blk[:,144:144+H].view(np.int8)
        for k in range(4):
            a=144+H+k*E*4
            LGS[k][j,t0:t1]=blk[:,a:a+E*4].copy().view(np.float32)
X.flush(); [g.flush() for g in LGS]
np.save(D+'SC.npy',SC);np.save(D+'PK.npy',PK);np.save(D+'n.npy',n)
import shutil; shutil.copy('/home/slomp/Work/rockem/bringup/expert-cache/data/phase3/bias.npy',D+'bias.npy')
print('steps',S,'tokens',T,'finite',[bool(np.isfinite(g[:,:1000]).all()) for g in LGS])
