#!/usr/bin/env python3
"""resident_slots.py [run]: what a resident expert slot is worth, from `plan_eval` (track-experts). Per MoE layer, over
the run's decode/verify layer runs: for each expert e, hits(e) = runs where e is in the rows plan with its resident bit
set and e was picked; rate(e) = hits(e) / the layer's runs (how often a resident expert is picked a layer-step). The
layer's residents ranked by rate: the sum of the lowest three rates is what three slots a layer cost (098's q4 analog),
the lowest one and a half q2's; the total over all residents is the layer's resident hits a layer-step (098's model:
1.27 cody / 1.67 survey at R = 6). A resident expert that is picked is in the plan at ranks 1-8 at .9+, so the plan's
24 entries see nearly all resident hits. (rockem.predictor's reading, 2026-10-26.)"""
import json, os, sys, urllib.request
from collections import defaultdict

URL = os.environ.get("PROW_URL", "http://192.168.88.3:7381")


def sql(q):
    req = urllib.request.Request(f"{URL}/api/sql?project=rockem&format=json", data=q.encode(), headers={"Content-Type": "text/plain"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    cols = d["columns"] if isinstance(d, dict) and "columns" in d else None
    rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
    return [dict(zip(cols, r)) for r in rows] if cols else rows


prefix = sys.argv[1] if len(sys.argv) > 1 else ""
run = sql(f"SELECT run FROM runs WHERE run LIKE '{prefix}%' ORDER BY started DESC LIMIT 1")[0]["run"] if prefix else sql("SELECT run FROM runs WHERE host = 'metatron' ORDER BY started DESC LIMIT 1")[0]["run"]
rows = sql(f"SELECT layer, tokens, rows, rows_resident, picked FROM plan_eval WHERE run = '{run}' AND tokens <= 4")
runs = defaultdict(int)
present = defaultdict(lambda: defaultdict(int))
hits = defaultdict(lambda: defaultdict(int))
for r in rows:
    l = int(r["layer"]); runs[l] += 1
    ids = [int(x) for x in r["rows"].split()]; res = int(r["rows_resident"]); picked = {int(x) for x in r["picked"].split()}
    for k, e in enumerate(ids):
        if e < 288 and res >> k & 1:
            present[l][e] += 1
            hits[l][e] += e in picked
print(f"resident slots: run {run[:8]}, {len(rows)} layer runs (steps of <= 4 tokens), {len(runs)} MoE layers")
print(f"{'layer':>5} {'runs':>6} {'residents':>9} {'total':>6} {'lowest1':>8} {'lowest3':>8} {'low 1.5':>8} {'top3':>6}")
tot = defaultdict(float)
for l in sorted(runs):
    rates = sorted(hits[l][e] / runs[l] for e in present[l])
    n = len(rates)
    if n == 0:
        continue
    total = sum(rates); low1 = rates[0]; low3 = sum(rates[:3]); low15 = rates[0] + 0.5 * (rates[1] if n > 1 else 0); top3 = sum(rates[-3:])
    for k, v in (("total", total), ("low1", low1), ("low3", low3), ("low15", low15), ("top3", top3), ("n", n)):
        tot[k] += v
    print(f"{l:>5} {runs[l]:>6} {n:>9} {total:>6.2f} {low1:>8.3f} {low3:>8.3f} {low15:>8.3f} {top3:>6.2f}")
L = len([l for l in runs if present[l]])
print(f"{'mean':>5} {'':>6} {tot['n']/L:>9.1f} {tot['total']/L:>6.2f} {tot['low1']/L:>8.3f} {tot['low3']/L:>8.3f} {tot['low15']/L:>8.3f} {tot['top3']/L:>6.2f}")
