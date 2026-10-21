import json, struct, numpy as np
d='/home/slomp/hf-cache/glm-5.3-flash/'
m=json.load(open(d+'model.safetensors.index.json'))['weight_map']
B=np.zeros((64,288),np.float32)
for l in range(3,46):
    k=f'model.language_model.layers.{l}.mlp.gate.e_score_correction_bias'
    f=open(d+m[k],'rb'); n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n))
    t=h[k]; a,b=t['data_offsets']; f.seek(8+n+a); assert t['dtype']=='F32', t['dtype']
    B[l]=np.frombuffer(f.read(b-a),np.float32)
np.save('/home/slomp/Work/rockem/bringup/expert-cache/data/phase3/bias.npy',B)
print(B[3,:5],B[45,:5], B[3:46].std(1).round(3))
