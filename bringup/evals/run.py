#!/usr/bin/env python3
"""Run a published benchmark against a node and keep the number.

  uv run python run.py humaneval --node 192.168.88.4:7332 --bundle ../../bundles/v2lite-q4gptq [--limit N]

The node is reached the way the API reaches it: `rockem.wire.Client`, the bundle's
tokenizer through `rockem.edge.Edge`, stop sequences through `rockem.edge.Stops`. What
the benchmark sees is therefore what a client of the service sees, under the model's
own sampling recipe. A task module gives the problems, how each becomes a prompt, where
its generation stops, and how a completion is scored; `run` does the rest and appends
the result to the model's `evals.toml` next to the published number, with the bundle's
identity and recipe, so a change in a kernel or a recipe shows up against the same
anchor the paper used. Every completion is written under `results/` for reading."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import importlib
import json
import sys
import time
from pathlib import Path

from rockem.bundle import Bundle
from rockem.edge import Edge, Stops
from rockem.wire import Client, ContextFull, Finish

HERE = Path(__file__).resolve().parent
MODELS = HERE.parents[1] / "models"


async def generate(addr: tuple[str, int], bundle: Bundle, edge: Edge, ids: list[int], stops: list[str], max_tokens: int) -> tuple[str, str, int]:
    """One prompt on a fresh connection: the completion text, why it ended, ids received."""
    client = await Client.connect(addr, bundle.bos, bundle.eos, bundle.vocab_size, bundle.identity)
    try:
        await client.prompt(ids)
        dec, st, text, n = edge.decoder(), Stops(stops), "", 0
        async for id in client.generate():
            n += 1
            if id == bundle.eos:
                text += st.flush()
                return text, "eos", n
            text += st.feed(dec.feed(id))
            if st.matched is not None:
                return text, "stop", n  # closing the connection is the abort
            if n >= max_tokens:
                text += st.flush()
                return text, "max_tokens", n
        text += st.flush()
        return text, "context_full" if client.finish is Finish.LENGTH else "closed", n
    finally:
        client.close()


async def loglik(addr: tuple[str, int], bundle: Bundle, prompt_ids: list[int], cont_ids: list[int]) -> float:
    """ln p(continuation | prompt), summed over the continuation's ids, on a fresh
    connection. The prompt is fed unscored, so a repeated prompt is served from the
    node's store and only its last position is recomputed."""
    client = await Client.connect(addr, bundle.bos, bundle.eos, bundle.vocab_size, bundle.identity)
    try:
        await client.prompt(prompt_ids)
        return sum(await client.score(cont_ids))
    finally:
        client.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--node", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--subset", help="score tasks: only problems whose id starts with this (a subject), still recorded")
    ap.add_argument("--no-record", action="store_true", help="do not append to the model's evals.toml")
    ap.add_argument("--resume", help="a results file of an interrupted run: its problems are kept, the rest run")
    a = ap.parse_args()
    task = importlib.import_module(a.task)
    host, port = a.node.rsplit(":", 1)
    addr = (host, int(port))
    bundle = Bundle(a.bundle)
    edge = Edge(bundle)
    problems = task.problems()
    if a.subset:
        problems = [p for p in problems if p["task_id"].startswith(a.subset)]
    if a.limit:
        problems = problems[: a.limit]
    stamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out = HERE / "results" / f"{a.task}-{bundle.identity:016x}-{stamp}.jsonl"
    out.parent.mkdir(exist_ok=True)
    t0, toks, passed = time.time(), 0, 0
    kind = getattr(task, "KIND", "generate")
    done: dict[str, dict] = {}
    if a.resume:
        for line in open(a.resume):
            r = json.loads(line)
            done[r["id"]] = r
    with open(out, "w") as f:
        for i, p in enumerate(problems):
            if p["task_id"] in done:
                r = done[p["task_id"]]
                passed += r["pass"]
                toks += r["prompt_ids"] + r["ids"]
                f.write(json.dumps(r) + "\n")
                continue
            ids = edge.encode(task.prompt(p))
            if kind == "score":
                # a continuation's ids are those its text adds to the prompt's, so a merge
                # across the boundary is scored as the prompt's own token would be; a prompt
                # the node's context cannot hold retreats to fewer shots
                shots = getattr(task, "SHOTS", 0)
                while True:
                    try:
                        lps = []
                        for cont in task.continuations(p):
                            full = edge.encode(task.prompt(p, shots) + cont)
                            assert full[: len(ids)] == ids, "the continuation changed the prompt's tokens"
                            lps.append(asyncio.run(loglik(addr, bundle, ids, full[len(ids) :])))
                        break
                    except ContextFull:
                        if shots == 0:
                            raise
                        shots -= 1
                        ids = edge.encode(task.prompt(p, shots))
                ok, detail = task.score(p, lps)
                if shots != getattr(task, "SHOTS", 0):
                    detail += f" ({shots}-shot: the full prompt did not fit)"
                reason, n, text = "scored", len(lps), ""
            else:
                text, reason, n = asyncio.run(generate(addr, bundle, edge, ids, task.STOPS, task.MAX_TOKENS))
                ok, detail = task.score(p, text)
            passed += ok
            toks += len(ids) + n
            f.write(json.dumps({"id": p["task_id"], "prompt_ids": len(ids), "completion": text, "reason": reason, "ids": n, "pass": ok, "detail": detail}) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(problems)}] {p['task_id']}: {'pass' if ok else 'FAIL'} ({reason}, {n} ids) running {100 * passed / (i + 1):.1f} %", file=sys.stderr)
    secs = time.time() - t0
    score = 100 * passed / len(problems)
    print(f"{a.task}: {task.METRIC} {score:.1f} ({passed}/{len(problems)}); published {task.PUBLISHED} ({task.PUBLISHED_SOURCE}); {toks} ids in {secs:.0f} s = {toks / secs:.0f} ids/s; {out.relative_to(HERE)}")
    if not a.no_record and not a.limit:
        subset = f'subset = "{a.subset}"\n' if a.subset else ""

        rec = MODELS / bundle.manifest.model / "evals.toml"
        with open(rec, "a") as f:
            f.write(f'\n[[{a.task}.runs]]\ndate = "{stamp[:10]}"\nbundle = "{Path(a.bundle).name}"\nidentity = "{bundle.identity:016x}"\nrecipe = "{bundle.manifest.recipe}"\nnode = "{a.node}"\n{subset}{task.METRIC} = {score:.1f}\npassed = {passed}\nn = {len(problems)}\nids_per_s = {toks / secs:.0f}\nresults = "bringup/evals/{out.relative_to(HERE)}"\n')
        print(f"recorded in {rec.relative_to(HERE.parents[1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
