import numpy as np
D='/home/slomp/Work/rockem/bringup/expert-cache/data/phase3/'
LG=np.load(D+'LG.npy',mmap_mode='r'); PK=np.load(D+'PK.npy'); n=np.load(D+'n.npy'); B=np.load(D+'bias.npy')
for j in [0,10,20,30,39]:
    l=3+j; g=1/(1+np.exp(-LG[j]))+B[l+1]
    srt=np.sort(g,1)[:,::-1]; thr=srt[:,7:8]; d=g-thr
    lab=np.zeros_like(g,bool); np.put_along_axis(lab,PK[j+1].astype(int),True,1)
    top8=d>=0; acc=(top8&lab).sum()/lab.sum()
    # raw logit ranking without bias
    srt2=np.sort(LG[j],1)[:,::-1]; acc2=((LG[j]>=srt2[:,7:8])&lab).sum()/lab.sum()
    print(l,'top8 acc w/bias %.3f raw %.3f'%(acc,acc2),'d pct',np.percentile(d,[1,50,90,99]).round(3),'d|pos',np.percentile(d[lab],[10,50,90]).round(3))
