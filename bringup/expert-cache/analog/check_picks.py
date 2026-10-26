# check_picks.py <dump dir> <goldens dir>: the dump's picks per MoE layer against the oracle's (sets per position).
import sys, numpy as np, json
from safetensors.numpy import load_file
H, HC, K = 4096, 4, 8
ROW = (HC * H + 2 * H) * 2 + K * 2
d, g = sys.argv[1], sys.argv[2]
R = load_file(g + '/routing.safetensors')
n = len(json.load(open(g + '/meta.json'))['ids'])
tot = same = 0; worst = []
for l in range(3, 45):
    b = np.fromfile(f'{d}/l{l}.bin', np.uint8).reshape(n, ROW)
    pk = b[:, -16:].copy().view(np.uint16)
    gk = R[f'l{l}.topk']
    s = sum(len(set(pk[t]) & set(gk[t])) for t in range(n)); print(l, n * K - s, end="; ")
    same += s; tot += n * K
    worst.append((n * K - s, l))
print(f'picks shared with the oracle: {same}/{tot} ({same / tot:.4f}); layers with most differing: {sorted(worst)[-5:]}')
