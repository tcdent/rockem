"""MMLU (Hendrycks et al. 2021): 14,042 four-choice questions over 57 subjects, five-shot
from each subject's dev set, in the released prompt format. Scored the way the DeepSeek-V2
paper scores it ("we adopt perplexity-based evaluation for datasets including ... MMLU"):
the choice whose answer letter is most probable after "Answer:", read through the SCORE
verb, so nothing is generated. The published number for DeepSeek-V2-Lite (base) is 58.3
(model card, 5-shot). `data/mmlu/` is the released archive
(people.eecs.berkeley.edu/~hendrycks/data.tar, MIT), fetched by `run.py` when absent."""

from __future__ import annotations

import csv
import io
import tarfile
import urllib.request
from pathlib import Path

METRIC = "accuracy"
PUBLISHED = 58.3
PUBLISHED_SOURCE = "DeepSeek-V2-Lite model card, base model, 5-shot"
KIND = "score"
SHOTS = 5
CHOICES = ["A", "B", "C", "D"]
DATA = Path(__file__).with_name("data") / "mmlu"
URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"


def _fetch() -> None:
    if (DATA / "data" / "test").exists():
        return
    DATA.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(URL, timeout=600).read()
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        t.extractall(DATA, filter="data")


def _rows(path: Path) -> list[list[str]]:
    with open(path, newline="") as f:
        return [r for r in csv.reader(f) if r]


def _subject(name: str) -> str:
    return name.replace("_", " ")


def _format(row: list[str], with_answer: bool) -> str:
    q, a, b, c, d, ans = row[:6]
    s = f"{q.strip()}\nA. {a.strip()}\nB. {b.strip()}\nC. {c.strip()}\nD. {d.strip()}\nAnswer:"
    return s + (f" {ans.strip()}\n\n" if with_answer else "")


def problems() -> list[dict]:
    _fetch()
    out = []
    for test in sorted((DATA / "data" / "test").glob("*_test.csv")):
        subject = test.name[: -len("_test.csv")]
        dev = _rows(DATA / "data" / "dev" / f"{subject}_dev.csv")[:SHOTS]
        for i, row in enumerate(_rows(test)):
            out.append({"task_id": f"{subject}/{i}", "subject": subject, "dev": dev, "row": row, "answer": row[5].strip()})
    return out


def prompt(p: dict, shots: int = SHOTS) -> str:
    """Five-shot; fewer when the question would not fit the node's context (the
    harness's usual retreat), which a few long-passage questions need at 2,048."""
    head = f"The following are multiple choice questions (with answers) about {_subject(p['subject'])}.\n\n" + "".join(_format(r, True) for r in p["dev"][:shots])
    return head + _format(p["row"], False)


def continuations(p: dict) -> list[str]:
    """The texts scored after the prompt, one per choice; the released format's answer is ' A'."""
    return [f" {c}" for c in CHOICES]


def score(p: dict, lps: list[float]) -> tuple[bool, str]:
    """`lps[i]` is the log-probability of continuation i given the prompt."""
    pick = CHOICES[max(range(len(lps)), key=lambda i: lps[i])]
    return pick == p["answer"], f"picked {pick}, answer {p['answer']}, ln p {[round(v, 3) for v in lps]}"
