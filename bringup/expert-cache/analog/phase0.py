# phase0.py: today's two plans on the lab dump pred3 (data/phase3c), measured as plan_eval measures the served ones.
# The shipped shape (plan.rs): per pair l -> l+1, z = ca*d + ca2*d2 + cc + bias + co, d from the route-ahead (LG),
# d2 from the second margin; the early plan's d2 from x + the shared output (LG1), the rows plan's from x + the whole
# FFN output (LG3); the learning (pred_learn, a step's end) sees the rows pass's d2, which overwrote the early's.
# A step's key Π(1 + e^z) over its tokens; P(picked) by rank of the union; calibration of 1 - 1/key in bins.
# Usage: phase0.py [--d2 LG3] [--early LG1] [--plan LG,LG1,...] (extra columns ranked by the same learned terms).
import numpy as np, torch, argparse, time
ap = argparse.ArgumentParser()
ap.add_argument('--data', default='phase3c')
ap.add_argument('--learn', default='LG3')        # the d2 the learning sees
ap.add_argument('--plans', default='LG1,LG3')    # d2 columns to plan with (each a plan): early, rows, others
ap.add_argument('--first', default='LG')      # the first margin's column (today: the route-ahead)
ap.add_argument('--from-token', type=int, default=0)  # score only steps starting at or after this position
ap.add_argument('--fallback', default='')   # where a column has no logits (an all-zero row: l + 1 not KDA, or before its --from), this column instead
ap.add_argument('--score', default='all')   # pairs scored: all, kda (l + 1 a KDA layer) or mla
ap.add_argument('--resident', default='0,3,6')  # resident-set sizes a layer (causal LFU over every earlier step's picks)
ap.add_argument('--window', default='8,10')     # experts' worth of bytes the prefetch window delivers, in post order
ap.add_argument('--stitch', default='')         # A:B:k — the posts' first k from plan A, the rest from plan B (today: early:rows:3)
ap.add_argument('--pairs', default='')        # a-b: score only pairs l = a..b (model layers)
ap.add_argument('--third', default='')        # a third margin with its own learned weight (from zero)
ap.add_argument('--ca-init', type=float, default=3.0)  # the first margin's initial weight (plan.rs: 3)
ap.add_argument('--lrb', type=float, default=1e-2)     # Adam rate of the calibration and bias (plan.rs LR_B)
ap.add_argument('--settle', type=float, default=0.0)  # score only steps after this fraction
ap.add_argument('--warm', type=int, default=0)  # passes over the data before the scored one (a longer-learned state)
ap.add_argument('--threads', type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0)
D = '/home/slomp/Work/rockem/bringup/expert-cache/data/' + a.data + '/'
PK = np.load(D + 'PK.npy').astype(np.int64); n = np.load(D + 'n.npy'); B = np.load(D + 'bias.npy')
cols = sorted(set([a.first, a.learn] + a.plans.split(',') + ([a.fallback] if a.fallback else []) + ([a.third] if a.third else [])))
LG = {c: np.load(D + c + '.npy', mmap_mode='r') for c in cols}
L = PK.shape[0]; P = L - 1; E = 288; PLAN = 24
S = len(n); st = np.concatenate([[0], np.cumsum(n)])
bias_next = torch.from_numpy(B[4:4 + P].copy()).float()
ca = torch.full((P, 1, 1), a.ca_init, requires_grad=True); ca3 = torch.zeros((P, 1, 1), requires_grad=True); cc = torch.zeros((P, 1, 1), requires_grad=True)
ca2 = torch.zeros((P, 1, 1), requires_grad=True); bb = torch.zeros((P, 1, E), requires_grad=True)
C = torch.zeros((P, E, E), requires_grad=True)
ob = torch.optim.Adam([ca, cc, ca2, bb] + ([ca3] if a.third else []), lr=a.lrb); oc = torch.optim.Adam([C], lr=3e-3)
MLA_NEXT = np.array([(j + 4) % 4 == 3 for j in range(P)])   # l + 1 an MLA layer (3, 7, 11, ...)
SCORED = {'all': np.ones(P, bool), 'kda': ~MLA_NEXT, 'mla': MLA_NEXT}[a.score]
if a.pairs:
    lo, hi = map(int, a.pairs.split('-')); SCORED = SCORED & np.array([lo <= j + 3 <= hi for j in range(P)])
def margin(c, t0, t1):
    z = np.array(LG[c][0:P, t0:t1])
    empty = ~np.any(z, axis=-1)   # [P][m]: no logits for this pair and token (an analog's MLA pairs, its positions before --from)
    if a.fallback and c != a.fallback and empty.any():
        z[empty] = np.asarray(LG[a.fallback][0:P, t0:t1])[empty]
    g = torch.sigmoid(torch.from_numpy(z).float()) + bias_next[:, None, :]
    return (g - g.topk(8, dim=2).values[:, :, 7:8]) * 10
plans = a.plans.split(',')
hit = {c: np.zeros((PLAN, 2)) for c in plans}; cal = {c: np.zeros((10, 3)) for c in plans}; rec13 = {c: np.zeros(2) for c in plans}
RES = [int(x) for x in a.resident.split(',')]; WIN = [int(x) for x in a.window.split(',')]
cnt = np.zeros((P, E))
names = plans + ([a.stitch] if a.stitch else [])
# per plan, per resident size: [resident hits, window hits per WIN..., non-resident picks, layer-steps]
win = {c: np.zeros((len(RES), len(WIN) + 3)) for c in names}
wpair = {c: np.zeros((P, 2)) for c in names}   # per pair at the last R and W: picked among the delivered, layer-steps
s0 = int(S * a.settle); t = time.time()
for ps, s in [(w, x) for w in range(a.warm + 1) for x in range(S)]:
    t0, t1 = st[s], st[s + 1]; m = t1 - t0
    d = margin(a.first, t0, t1)
    pl = torch.zeros(P, m, E); pl.scatter_(2, torch.from_numpy(PK[0:P, t0:t1]), 1.0)
    y = torch.zeros(P, m, E); y.scatter_(2, torch.from_numpy(PK[1:P + 1, t0:t1]), 1.0)
    yu = (y.sum(1) > 0).numpy()
    base = ca * d + cc + bb + torch.bmm(pl, C)
    if a.third: base = base + ca3 * margin(a.third, t0, t1)
    if ps == a.warm and s >= s0 and t0 >= a.from_token:
        with torch.no_grad():
          orders = {}
          for c in plans:
            z = base + ca2 * margin(c, t0, t1)
            lk = torch.nn.functional.softplus(z.clamp(max=20)).sum(1)          # log key
            orders[c] = torch.argsort(lk, dim=1, descending=True).numpy()
            idx = orders[c][:, :PLAN]
            p = (1 - torch.exp(-lk)).numpy()
            got = np.take_along_axis(yu, idx, 1)[SCORED]; idx = idx[SCORED]; p = p[SCORED]; yus = yu[SCORED]
            hit[c][:, 0] += got.sum(0); hit[c][:, 1] += SCORED.sum()
            pp = np.take_along_axis(p, idx, 1); b = np.minimum((pp * 10).astype(int), 9)
            np.add.at(cal[c], (b.ravel(), 0), got.ravel()); np.add.at(cal[c], (b.ravel(), 1), 1); np.add.at(cal[c], (b.ravel(), 2), pp.ravel())
            rec13[c] += [got[:, :13].sum(), yus.sum()]
          # the window's view: posts = the plan's order without the resident experts; what the first W of them pick
          for j in np.nonzero(SCORED)[0]:
              rank = np.argsort(-cnt[j], kind='stable')
              for ri, R in enumerate(RES):
                  res = set(rank[:R].tolist())
                  for c in names:
                      if c in orders:
                          posts = [e for e in orders[c][j] if e not in res]
                      else:
                          A, B, k = c.split(':'); k = int(k)
                          head = [e for e in orders[A][j] if e not in res][:k]
                          posts = head + [e for e in orders[B][j] if e not in res and e not in head]
                      w = win[c][ri]
                      w[0] += sum(yu[j][e] for e in res)
                      for wi, W in enumerate(WIN):
                          w[1 + wi] += yu[j][posts[:W]].sum()
                      w[-2] += yu[j].sum() - sum(yu[j][e] for e in res); w[-1] += 1
                      if ri == len(RES) - 1:
                          wpair[c][j] += [yu[j][posts[:WIN[-1]]].sum(), 1]
    cnt += yu
    z = base + ca2 * margin(a.learn, t0, t1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction='sum') / m / P
    ob.zero_grad(); oc.zero_grad(); loss.backward(); ob.step(); oc.step()
    if s % 500 == 0: print('step', s, '/', S, '%.0fs' % (time.time() - t), flush=True)
print(f"{S} steps, {n.sum()} tokens, scored from step {s0} / token {a.from_token}; first margin {a.first}, learning on {a.learn}")
print('rank ' + ' '.join(f'{c:>7}' for c in plans))
for k in range(PLAN):
    print(f'{k + 1:>4} ' + ' '.join(f'{hit[c][k, 0] / hit[c][k, 1]:7.3f}' for c in plans))
print('rec@13 ' + ' '.join(f'{rec13[c][0] / rec13[c][1]:7.3f}' for c in plans))
for c in plans:
    print(f'calibration {c}: ' + '  '.join(f'{b / 10:.1f}:{v[2] / v[1]:.2f}->{v[0] / v[1]:.2f}({int(v[1])})' for b, v in enumerate(cal[c]) if v[1]))
print('window (a layer-step): resident R | resident hits | picked among the first W posts for W in ' + a.window + ' | non-resident picks')
for c in names:
    for ri, R in enumerate(RES):
        w = win[c][ri]; L_ = w[-1]
        print(f'  {c:>14} R={R}: res {w[0] / L_:5.2f} | ' + ' '.join(f'{w[1 + i] / L_:5.2f}' for i in range(len(WIN))) + f' | nonres {w[-2] / L_:5.2f}')
print(f'per pair (l), picked of the first {WIN[-1]} posts at R={RES[-1]}:')
for c in names:
    print(f'  {c:>14}: ' + ' '.join(f'{j + 3}:{wpair[c][j, 0] / wpair[c][j, 1]:.2f}' for j in range(P) if wpair[c][j, 1]))
