# build.py <forward dir> <columns dir> <out dir> <positions> [step]: the analog dump as the simulation's arrays. Index j
# = layer j + 3 over the layers the forward saved (3 .. 3 + L - 1): PK [L][T][8] (picks), every column [L][T][288] f32
# (pair j's logits for layer j + 4; rows without a pair zero), n (steps of `step` tokens), bias (by model layer).
import sys, os, numpy as np, shutil
fwd, cols, out, T = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); step = int(sys.argv[5]) if len(sys.argv) > 5 else 3
E, K, H, HC = 288, 8, 4096, 4
ROW = (HC * H + 2 * H) * 2 + K * 2
L = 0
while os.path.exists(f'{fwd}/l{L + 3}.bin'): L += 1
os.makedirs(out, exist_ok=True)
PK = np.zeros((L, T, K), np.int16)
for j in range(L):
    PK[j] = np.memmap(f'{fwd}/l{j + 3}.bin', np.uint8, 'r').reshape(T, ROW)[:, -16:].copy().view(np.int16)
np.save(f'{out}/PK.npy', PK)
for c in sorted(os.listdir(cols)):
    if not os.path.isdir(f'{cols}/{c}'): continue
    A = np.lib.format.open_memmap(f'{out}/{c}.npy', 'w+', np.float32, (L, T, E))
    have = []
    for j in range(L):
        f = f'{cols}/{c}/l{j + 3}.f32'
        if os.path.exists(f):
            A[j] = np.fromfile(f, np.float32).reshape(T, E); have.append(j + 3)
    A.flush(); print(c, 'pairs', have[0] if have else '-', '..', have[-1] if have else '-', len(have))
n = np.full(T // step, step); n = np.append(n, T - n.sum()) if T % step else n
np.save(f'{out}/n.npy', n)
shutil.copy('/home/slomp/Work/rockem/bringup/expert-cache/data/phase3/bias.npy', f'{out}/bias.npy')
print('layers', L, 'tokens', T, 'steps', len(n))
