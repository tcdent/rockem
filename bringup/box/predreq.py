"""The phase-3 dump's 24 decode requests: the 18k coding-session prompt (requests/cody-decode512.json) with its last
question replaced by each of these, 450 tokens each, written to requests/predreq/ (2026-10-21)."""
import json, pathlib
here = pathlib.Path(__file__).parent / "requests"
d = json.load(open(here / "cody-decode512.json"))
base = d["messages"][0]["content"]
cut = base.rindex("\n\n") + 2
QS = [l.strip() for l in (pathlib.Path(__file__).parent / "predreq.txt").read_text().splitlines() if l.strip()]
(here / "predreq").mkdir(exist_ok=True)
for i, q in enumerate(QS):
    json.dump({"model": d["model"], "messages": [{"role": "user", "content": base[:cut] + q}], "max_tokens": 450}, open(here / "predreq" / f"q{i:02d}.json", "w"))
