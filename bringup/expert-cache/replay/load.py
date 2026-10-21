import json, collections
d=json.load(open('/home/slomp/Work/rockem/bringup/expert-cache/data/trace.json')); u=json.load(open('/home/slomp/Work/rockem/bringup/expert-cache/data/use.json'))
R=[]
for (t,l,n,c,p,r,pv),ur in zip(d['rows'],u['rows']):
    assert ur[1]==l and ur[2]==n and 0<ur[0]-t<100000, (t,l,ur)
    cand=[list(map(int,x.split())) for x in c.split(';')]
    picks=[list(map(int,x.split())) for x in p.split(';')]
    res=r.split(';')
    prev=list(map(int,pv.split())) if pv else []
    R.append(dict(t=t,l=l,n=n,cand=cand,picks=picks,res=res,prev=prev,use=ur[3:]))
