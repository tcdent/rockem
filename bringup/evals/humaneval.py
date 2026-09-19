"""HumanEval (Chen et al. 2021): 164 Python functions from a signature and docstring,
scored by their unit tests. `data/HumanEval.jsonl.gz` is the released set
(github.com/openai/human-eval, MIT). Zero-shot: the prompt is the problem's text as
released; generation stops at the next top-level statement, the stop sequences the
released harness and the Codex paper use; pass@1 is one sample per problem under the
model's recipe. The published number for DeepSeek-V2-Lite (base) is 29.9, from the
model card, whose decoding is the paper's and not stated there."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

METRIC = "pass_at_1"
PUBLISHED = 29.9
PUBLISHED_SOURCE = "DeepSeek-V2-Lite model card, base model"
STOPS = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
MAX_TOKENS = 512
TIMEOUT = 10.0


def problems() -> list[dict]:
    with gzip.open(Path(__file__).with_name("data") / "HumanEval.jsonl.gz", "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt(p: dict) -> str:
    return p["prompt"]


def score(p: dict, completion: str) -> tuple[bool, str]:
    """Run the program in a subprocess with a timeout: model-written code, so it is
    never exec'd in this process; there is no other sandbox here."""
    program = p["prompt"] + completion + "\n\n" + p["test"] + "\n\n" + f"check({p['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "prog.py"
        path.write_text(program)
        try:
            r = subprocess.run([sys.executable, str(path)], cwd=d, capture_output=True, text=True, timeout=TIMEOUT, env={"PATH": os.environ.get("PATH", "")})
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if r.returncode == 0:
        return True, "passed"
    tail = (r.stderr.strip().splitlines() or ["exit %d" % r.returncode])[-1]
    return False, tail[:200]
