# check_cols.py <cols dir> <dump dir> <n> <pair l>: each column's per-token top-8 (sigmoid + bias) against layer l+1's picks.
import sys, numpy as np, os
E, K, H, HC = 288, 8, 4096, 4
ROW = (HC * H + 2 * H) * 2 + K * 2
c, d, n, l = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
B = np.load('/home/slomp/Work/rockem/bringup/expert-cache/data/phase3/bias.npy')
pk = np.fromfile(f'{d}/l{l + 1}.bin', np.uint8).reshape(n, ROW)[:, -16:].copy().view(np.uint16)
for col in sorted(x for x in os.listdir(c) if os.path.exists(f'{c}/{x}/l{l}.f32')):
    z = np.fromfile(f'{c}/{col}/l{l}.f32', np.float32).reshape(n, E)
    g = 1 / (1 + np.exp(-z)) + B[l + 1]
    top = np.argsort(-g, 1)[:, :K]
    print(f'{col:12} top-8 overlap with the picks {np.mean([len(set(a) & set(b)) for a, b in zip(top, pk)]):.3f}')
