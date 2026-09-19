"""GSM8K (Cobbe et al. 2021): 1,319 grade-school math problems, eight-shot with
chain-of-thought exemplars, scored by exact match of the final number. The
DeepSeek-V2 paper evaluates it generation-based (8-shot); the published number
for DeepSeek-V2-Lite (base) is 41.1 (model card). `data/gsm8k/` is the released
test set and the eight exemplars are the standard CoT prompt of Wei et al.
(2022), the ones every harness uses; fetched by `run.py` when absent."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

METRIC = "accuracy"
PUBLISHED = 41.1
PUBLISHED_SOURCE = "DeepSeek-V2-Lite model card, base model, 8-shot"
STOPS = ["\n\nQuestion:", "\nQuestion:", "\n\n\n"]
MAX_TOKENS = 320
DATA = Path(__file__).with_name("data") / "gsm8k"
URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

# the eight chain-of-thought exemplars of Wei et al. 2022 (Table 20), as lm-evaluation-harness's gsm8k_cot uses them
SHOTS = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."),
]
HEAD = "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a in SHOTS)


def _fetch() -> None:
    if (DATA / "test.jsonl").exists():
        return
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "test.jsonl").write_bytes(urllib.request.urlopen(URL, timeout=600).read())


def problems() -> list[dict]:
    _fetch()
    out = []
    with open(DATA / "test.jsonl") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            out.append({"task_id": f"gsm8k/{i}", "question": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")})
    return out


def prompt(p: dict) -> str:
    return HEAD + f"Question: {p['question']}\nAnswer:"


def _number(text: str) -> str | None:
    m = re.search(r"The answer is\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if m is None:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        m_val = nums[-1]
    else:
        m_val = m.group(1)
    v = m_val.replace(",", "").rstrip(".")
    return v[:-2] if v.endswith(".0") else v


def score(p: dict, completion: str) -> tuple[bool, str]:
    got = _number(completion)
    return got == p["answer"], f"got {got}, answer {p['answer']}"
