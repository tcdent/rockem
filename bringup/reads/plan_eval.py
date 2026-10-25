#!/usr/bin/env python3
"""plan_eval.py [run] [since]: how good the decode predictor's plans are (`plan_eval` rows, track-experts, aff85+).

Each row is one MoE layer of a decode or verify step: the early plan (posted at the previous layer's slot) and the rows
plan (its amend, 093), each `plan::PLAN` experts best first with their keys Π(1 + eᶻ) and residency, and the experts the
step's tokens picked. For both plans: the hit rate at each rank, overall and among the experts not resident on the cards
(those the prefetch posts); the prefetch's own view (the first `--cap` non-resident experts: how many are picked, and
what share of the non-resident picks they cover); and the calibration — the predicted P(picked) = 1 − 1/key (exact if
the tokens' calibrated picks are independent) against the rate observed, in bins. `run` is a prefix of the run id
(none: metatron's newest); `since` a UTC timestamp."""
import json, os, sys, urllib.request

URL = os.environ.get("PROW_URL", "http://192.168.88.3:7381")
PLAN, CAP = 24, int(os.environ.get("CAP", "13"))


def sql(q: str):
    req = urllib.request.Request(f"{URL}/api/sql?project=rockem&format=json", data=q.encode(), headers={"Content-Type": "text/plain"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    cols = d["columns"] if isinstance(d, dict) and "columns" in d else None
    rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
    return [dict(zip(cols, r)) for r in rows] if cols else rows


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    since = sys.argv[2] if len(sys.argv) > 2 else "1970-01-01"
    if prefix:
        run = sql(f"SELECT run FROM runs WHERE run LIKE '{prefix}%' ORDER BY started DESC LIMIT 1")[0]["run"]
    else:
        run = sql("SELECT run FROM runs WHERE host = 'metatron' ORDER BY started DESC LIMIT 1")[0]["run"]
    rows = sql(f"SELECT tokens, early, early_keys, early_resident, rows, rows_keys, rows_resident, amended, picked FROM plan_eval "
               f"WHERE run = '{run}' AND ts >= TIMESTAMPTZ '{since}'")
    print(f"plan_eval: {len(rows)} layer runs of run {run[:8]} (decode and verify steps)")
    for which in ("early", "rows"):
        rs = [r for r in rows if which == "early" or r["amended"]]
        if not rs:
            continue
        hit = [[0, 0] for _ in range(PLAN)]      # all: picked, seen
        hit_nr = [[0, 0] for _ in range(PLAN)]   # not resident: picked, seen
        res = [0] * PLAN
        cal = {}                                 # p bin → [picked, seen]
        posted = [0, 0]                          # picked among the posted, posted
        cover = [0, 0]                           # non-resident picks covered by the posted, non-resident picks
        for r in rs:
            ids = [int(x) for x in r[which].split()]
            keys = [float(x) for x in r[f"{which}_keys"].split()]
            resident = int(r[f"{which}_resident"])
            picked = {int(x) for x in r["picked"].split()}
            nr_picks = 0
            taken = []
            for k, (e, key) in enumerate(zip(ids, keys)):
                if e >= 288:
                    continue
                is_res = resident >> k & 1
                got = e in picked
                hit[k][0] += got
                hit[k][1] += 1
                res[k] += is_res
                if not is_res:
                    hit_nr[k][0] += got
                    hit_nr[k][1] += 1
                    if len(taken) < CAP:
                        taken.append(e)
                p = 1 - 1 / key if key > 0 else 0.0
                b = min(int(p * 10), 9)
                c = cal.setdefault(b, [0, 0, 0.0])
                c[0] += got
                c[1] += 1
                c[2] += p
            posted[0] += sum(e in picked for e in taken)
            posted[1] += len(taken)
            # the non-resident picks: an expert picked and not resident is one a prefetch could have helped
            in_plan_res = {e for k, e in enumerate(ids) if e < 288 and resident >> k & 1}
            nr = [e for e in picked if e not in in_plan_res]
            cover[0] += sum(e in taken for e in nr)
            cover[1] += len(nr)
        print(f"\n== the {which} plan ({len(rs)} layer runs)")
        print(f"{'rank':>4} {'P(picked)':>10} {'resident':>9} {'P(picked | not resident)':>25}")
        for k in range(PLAN):
            if hit[k][1] == 0:
                continue
            nr = f"{hit_nr[k][0] / hit_nr[k][1]:.3f}" if hit_nr[k][1] else "-"
            print(f"{k + 1:>4} {hit[k][0] / hit[k][1]:>10.3f} {res[k] / hit[k][1]:>9.3f} {nr:>25}")
        print(f"the prefetch's {CAP} (non-resident, best first): {posted[0] / max(posted[1], 1):.3f} picked "
              f"({posted[0] / len(rs):.2f} of {posted[1] / len(rs):.2f} a layer); they cover {cover[0] / max(cover[1], 1):.3f} "
              f"of the non-resident picks ({cover[1] / len(rs):.2f} a layer)")
        print(f"{'predicted P':>12} {'experts':>8} {'mean predicted':>15} {'observed':>9}")
        for b in sorted(cal):
            c = cal[b]
            print(f"{f'{b / 10:.1f}–{(b + 1) / 10:.1f}':>12} {c[1]:>8} {c[2] / c[1]:>15.3f} {c[0] / c[1]:>9.3f}")


main()
