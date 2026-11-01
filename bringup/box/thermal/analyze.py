#!/usr/bin/env python3
"""analyze.py — how metatron warmed and cooled around a load.py window, from voria.

  analyze.py --label warm4k [--before 5] [--after 25]   (minutes of context either side)

For every temperature series (k10temp Tctl per socket, amdgpu junction per card, the IT8613E
thermistor, the BMC's VR, inlet and outlet sensors) and socket power: the baseline before the
window, the peak and when it came, the value at the window's end, then the cool-down: the steepest
fall in a minute, the time constant of the decay toward baseline (least squares on log(T - base)
over the first --fit minutes after the end), and the time to come within 2 °C of baseline.
Writes bringup/box/logs/thermal/<label>.analysis.csv and prints the table.
"""
import argparse, csv, json, math, os, sys, urllib.parse, urllib.request

VORIA = os.environ.get("VORIA_URL", "http://192.168.88.3:8428")
HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.normpath(os.path.join(HERE, "..", "logs", "thermal"))

SERIES = {
    "socket0 Tctl": 'host_hwmon_celsius{host="metatron",chip="k10temp",dev="0000:00:18.3",sensor="Tctl"}',
    "socket1 Tctl": 'host_hwmon_celsius{host="metatron",chip="k10temp",dev="0000:00:19.3",sensor="Tctl"}',
    "card0 c3 junction": 'host_hwmon_celsius{host="metatron",chip="amdgpu",dev="0000:c3:00.0",sensor="junction"}',
    "card1 a3 junction": 'host_hwmon_celsius{host="metatron",chip="amdgpu",dev="0000:a3:00.0",sensor="junction"}',
    "board thermistor": 'host_hwmon_celsius{host="metatron",chip="it8613",sensor="temp2"}',
    "BMC vcore VR": 'ipmi_sensor_value{host="metatron",status_code="ok",name="cpu_vcore_vr_t"}',
    "BMC vsoc VR": 'ipmi_sensor_value{host="metatron",status_code="ok",name="cpu_vsoc_vr_t"}',
    "BMC inlet": 'ipmi_sensor_value{host="metatron",status_code="ok",name="inlet_temp"}',
    "BMC outlet": 'ipmi_sensor_value{host="metatron",status_code="ok",name="outlet_temp"}',
    "socket power W": 'sum(host_hwmon_watts{host="metatron",chip="amd_hsmp_hwmon"})',
    "cards power W": 'sum(host_hwmon_watts{host="metatron",chip="amdgpu",sensor="PPT"})',
    "CPU fan2 rpm": 'host_hwmon_rpm{host="metatron",chip="it8613",sensor="fan2"}',
    "card0 fan rpm": 'host_hwmon_rpm{host="metatron",chip="amdgpu",dev="0000:c3:00.0"}',
}


def fetch(query, start, end, step=10):
    q = urllib.parse.urlencode({"query": query, "start": start, "end": end, "step": step})
    with urllib.request.urlopen(f"{VORIA}/api/v1/query_range?{q}", timeout=30) as r:
        res = json.load(r)["data"]["result"]
    return [(float(t), float(v)) for t, v in res[0]["values"]] if res else []


def window(label):
    rows = [json.loads(l) for l in open(os.path.join(LOGS, f"{label}.jsonl"))]
    starts = [r for r in rows if r["kind"] == "start"]; ends = [r for r in rows if r["kind"] == "end"]
    reqs = [r for r in rows if r["kind"] == "request"]
    if not starts or not reqs:
        sys.exit("no start or no request in the log")
    # a run that died without its end row ends with its last request
    end = ends[-1]["t"] if ends else reqs[-1]["t"]
    return starts[-1]["t"], end, reqs


def stats(series, t0, t1, fit_min):
    before = [v for t, v in series if t < t0]
    during = [(t, v) for t, v in series if t0 <= t <= t1]
    after = [(t, v) for t, v in series if t > t1]
    if not before or not during:
        return None
    base = sum(before) / len(before)
    peak_t, peak = max(during, key=lambda p: p[1])
    end_v = during[-1][1]
    out = {"base": base, "peak": peak, "rise": peak - base, "t_peak_min": (peak_t - t0) / 60, "end": end_v}
    # steepest 60 s fall after the end
    fall = 0.0
    for i in range(len(after)):
        for j in range(i + 1, len(after)):
            dt = after[j][0] - after[i][0]
            if 55 <= dt <= 65:
                fall = min(fall, after[j][1] - after[i][1]); break
    out["fall60s"] = fall
    # time constant: log-linear fit of (T - base) over the first fit_min minutes after the end
    pts = [(t - t1, v - base) for t, v in after if t - t1 <= fit_min * 60 and v - base > 0.5]
    if len(pts) >= 6:
        xs = [p[0] for p in pts]; ys = [math.log(p[1]) for p in pts]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs); sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = sxy / sxx if sxx else 0
        out["tau_min"] = (-1 / slope) / 60 if slope < 0 else float("inf")
    else:
        out["tau_min"] = None
    settle = next((t - t1 for t, v in after if v - base <= 2.0), None)
    out["settle_min"] = settle / 60 if settle is not None else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--before", type=float, default=5)
    ap.add_argument("--after", type=float, default=25)
    ap.add_argument("--fit", type=float, default=10, help="minutes of cool-down fitted for the time constant")
    a = ap.parse_args()
    t0, t1, reqs = window(a.label)
    print(f"[{a.label}] window {t1 - t0:.0f} s, {len(reqs)} requests, "
          f"{sum(r.get('prompt_tokens') or 0 for r in reqs)} prompt + {sum(r.get('completion_tokens') or 0 for r in reqs)} completion tokens")
    rows = []
    hdr = f"{'series':20} {'base':>6} {'peak':>6} {'rise':>5} {'t_peak':>6} {'end':>6} {'fall/min':>8} {'tau':>6} {'settle':>6}"
    print(hdr)
    for name, q in SERIES.items():
        s = fetch(q, t0 - a.before * 60, t1 + a.after * 60)
        st = stats(s, t0, t1, a.fit) if s else None
        if not st:
            print(f"{name:20} (no data)"); continue
        f = lambda v, w=6, d=1: (f"{v:{w}.{d}f}" if isinstance(v, (int, float)) and v not in (None, float('inf')) else f"{'—':>{w}}")
        print(f"{name:20} {f(st['base'])} {f(st['peak'])} {f(st['rise'],5)} {f(st['t_peak_min'])} {f(st['end'])} {f(st['fall60s'],8)} {f(st['tau_min'])} {f(st['settle_min'])}")
        rows.append({"series": name, **st})
    out = os.path.join(LOGS, f"{a.label}.analysis.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["series", "base", "peak", "rise", "t_peak_min", "end", "fall60s", "tau_min", "settle_min"])
        w.writeheader(); w.writerows(rows)
    print(f"csv: {out}")


if __name__ == "__main__":
    main()
