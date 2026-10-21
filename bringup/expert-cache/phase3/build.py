# Dense arrays [layer j (0..40 = layers 3..43), token t] from pred.bin; steps in order.
import numpy as np
D='/home/slomp/Work/rockem/bringup/expert-cache/data/phase3/'
I=np.load(D+'index.npz'); offs,lay,ns=I['offs'],I['lay'],I['ns']
L=41; S=len(offs)//L
assert len(offs)==S*L
lay2=lay.reshape(S,L); ns2=ns.reshape(S,L)
assert (lay2==np.arange(3,44)[None]).all() and (ns2==ns2[:,:1]).all()
n=ns2[:,0]; T=n.sum(); start=np.concatenate([[0],np.cumsum(n)])
m=np.memmap('/home/slomp/Work/rockem/bringup/expert-cache/data/pred.bin',np.uint8,'r')
TOK=16+128+4096+1152
X=np.lib.format.open_memmap(D+'X.npy','w+',np.int8,(L,int(T),4096))
SC=np.zeros((L,T,32),np.float32); LG=np.zeros((L,T,288),np.float32); PK=np.zeros((L,T,8),np.int16)
for s in range(S):
    for j in range(L):
        o=offs[s*L+j]+4
        blk=np.asarray(m[o:o+n[s]*TOK]).reshape(n[s],TOK)
        t0=start[s]; t1=start[s+1]
        PK[j,t0:t1]=blk[:,:16].copy().view(np.int16)
        SC[j,t0:t1]=blk[:,16:144].copy().view(np.float32)
        X[j,t0:t1]=blk[:,144:144+4096].view(np.int8)
        LG[j,t0:t1]=blk[:,4240:].copy().view(np.float32)
X.flush()
np.save(D+'SC.npy',SC);np.save(D+'LG.npy',LG);np.save(D+'PK.npy',PK);np.save(D+'n.npy',n)
print(S,T,PK.max(),PK.min(),np.isfinite(LG).all(), SC.mean(), LG.std())
