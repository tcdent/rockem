# Walk pred.bin's records; write per-token arrays (layer, step id, tok idx, picks, logits, x int8, scales).
import numpy as np, sys
P='/home/slomp/Work/rockem/bringup/expert-cache/data/pred.bin'
m=np.memmap(P,np.uint8,'r'); N=len(m)
TOK=16+128+4096+1152
offs=[];lay=[];ns=[]
o=0
while o<N:
    l,n=int(m[o])|int(m[o+1])<<8, int(m[o+2])|int(m[o+3])<<8
    offs.append(o);lay.append(l);ns.append(n); o+=4+n*TOK
assert o==N,(o,N)
offs=np.array(offs);lay=np.array(lay);ns=np.array(ns)
print('records',len(offs),'tokens-recs',ns.sum())
print('layers',np.unique(lay,return_counts=True))
print('n',np.unique(ns,return_counts=True))
# sequence pattern sample
print(list(zip(lay[:100],ns[:100])))
np.savez('/home/slomp/Work/rockem/bringup/expert-cache/data/phase3/index.npz',offs=offs,lay=lay,ns=ns)
