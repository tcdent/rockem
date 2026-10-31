#!/usr/bin/env python3
"""load.py — warm metatron through the rockem API in a controlled way and record the window.

  load.py --label warm4k --minutes 3 --context 4096 --max-tokens 256 [--mode prefill|decode|mixed]

Serial requests (the stack is serial-first) back to back for --minutes. Each prompt carries a fresh
random prefix so the node's prefix cache never short-cuts the prefill. Every 20 s the guard reads
voria (10 s host samples, 30 s BMC samples) and aborts the run when any limit is crossed, since the
chassis fans sit at a fixed 50 % today (tcdent, 2026-10-31). The window's start/end and every
request land in bringup/box/logs/thermal/<label>.jsonl for analyze.py.

Limits (°C): --tctl-max 85 (k10temp Tctl, either socket) --vr-max 90 (BMC Vcore/VSoC VR)
             --junction-max 95 (either card)
"""
import argparse, json, os, random, sys, time, urllib.request, urllib.parse

API = os.environ.get("ROCKEM_API", "http://192.168.88.3:8765")
VORIA = os.environ.get("VORIA_URL", "http://192.168.88.3:8428")
HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.normpath(os.path.join(HERE, "..", "logs", "thermal"))

WORDS = ("the model reads every position of the prompt and attends across the whole window while the experts "
         "on the host stream their weights from memory one layer at a time so the sockets draw their full power "
         "and the cards hold the attention state for the window").split()


def prompt_of(n_tokens: int, rng: random.Random) -> str:
    # ~0.75 tokens a word for this tokenizer on plain English; the first request calibrates
    words = [rng.choice(WORDS) for _ in range(int(n_tokens * prompt_of.ratio))]
    nonce = f"session {rng.randrange(1 << 48):x}. "
    return nonce + " ".join(words) + "\n\nSummarize the text above in detail, then continue it."


prompt_of.ratio = 0.78


def request(prompt: str, max_tokens: int) -> dict:
    body = json.dumps({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "stream": False}).encode()
    req = urllib.request.Request(f"{API}/v1/chat/completions", data=body, headers={"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    u = d.get("usage", {})
    return {"t0": t0, "s": time.time() - t0, "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "finish": d["choices"][0].get("finish_reason")}


def voria(query: str):
    q = urllib.parse.urlencode({"query": query})
    with urllib.request.urlopen(f"{VORIA}/api/v1/query?{q}", timeout=10) as r:
        d = json.load(r)["data"]["result"]
    return float(d[0]["value"][1]) if d else None


def guard() -> dict:
    return {
        "tctl": voria('max(host_hwmon_celsius{host="metatron",chip="k10temp",sensor="Tctl"})'),
        "vr": voria('max(ipmi_sensor_value{host="metatron",status_code="ok",name=~"cpu_v(core|soc)_vr_t"})'),
        "junction": voria('max(host_hwmon_celsius{host="metatron",chip="amdgpu",sensor="junction"})'),
        "watts": voria('sum(host_hwmon_watts{host="metatron",chip="amd_hsmp_hwmon"})'),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--minutes", type=float, required=True)
    ap.add_argument("--context", type=int, default=4096, help="prompt tokens a request")
    ap.add_argument("--max-tokens", type=int, default=256, help="completion tokens a request")
    ap.add_argument("--mode", choices=["prefill", "decode", "mixed"], default="mixed",
                    help="prefill: long prompt, 32 tokens out; decode: 256-token prompt, --max-tokens out; mixed: both as given")
    ap.add_argument("--tctl-max", type=float, default=85)
    ap.add_argument("--vr-max", type=float, default=90)
    ap.add_argument("--junction-max", type=float, default=95)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "prefill":
        a.max_tokens = 32
    elif a.mode == "decode":
        a.context = 256
    os.makedirs(LOGS, exist_ok=True)
    path = os.path.join(LOGS, f"{a.label}.jsonl")
    rng = random.Random(a.seed)
    log = open(path, "a")

    def emit(kind, **kw):
        row = {"kind": kind, "t": time.time(), **kw}
        log.write(json.dumps(row) + "\n"); log.flush()
        return row

    g = guard()
    emit("start", label=a.label, minutes=a.minutes, context=a.context, max_tokens=a.max_tokens, mode=a.mode, guard=g)
    print(f"[{a.label}] start: {g}", flush=True)
    t_start = time.time(); t_end = t_start + 60 * a.minutes; last_guard = time.time(); n = 0; aborted = None
    while time.time() < t_end:
        try:
            r = request(prompt_of(a.context, rng), a.max_tokens)
        except Exception as e:  # the API's error is the data here, not a crash
            emit("error", error=str(e)); print(f"[{a.label}] request error: {e}", flush=True); time.sleep(5); continue
        n += 1
        if n == 1 and r["prompt_tokens"]:
            prompt_of.ratio *= a.context / max(r["prompt_tokens"], 1)  # calibrate words → tokens
        emit("request", n=n, **r)
        print(f"[{a.label}] #{n} {r['prompt_tokens']} + {r['completion_tokens']} tokens in {r['s']:.1f} s ({r['finish']})", flush=True)
        if time.time() - last_guard >= 20:
            g = guard(); last_guard = time.time(); emit("guard", **g)
            print(f"[{a.label}] guard: {g}", flush=True)
            if (g["tctl"] or 0) >= a.tctl_max or (g["vr"] or 0) >= a.vr_max or (g["junction"] or 0) >= a.junction_max:
                aborted = g; break
    g = guard()
    emit("end", requests=n, aborted=aborted, guard=g, elapsed_s=time.time() - t_start)
    print(f"[{a.label}] end after {n} requests, {time.time() - t_start:.0f} s; {'ABORTED ' + str(aborted) if aborted else 'clean'}; {g}", flush=True)
    print(f"log: {path}")
    return 2 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
