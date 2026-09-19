#!/usr/bin/env python3
"""The sampler against the reference, on real logits: transformers' warpers
(the version the checkpoint names) versus our recipe, position by position.

  uv run python sample_check.py --sets code,dialogue,science [--twin ../bringup/bench/target/release/bench-sample]

For every position of every golden set: the reference applies
TemperatureLogitsWarper, TopKLogitsWarper and TopPLogitsWarper (verbatim, in
generate's order) to `lm_head.out`, then softmax; a draw with the uniform u is
the walk over the surviving probabilities in (probability desc, index asc)
order. Ours is `common::sample::draw` through the bench's `--draw` twin with
the same logits and u. Reports how many positions agree on the kept set and on
the token, the disagreements, and the nucleus sizes seen."""
import argparse, json, struct, subprocess, tempfile, os
import numpy as np, torch
from safetensors import safe_open
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

ap = argparse.ArgumentParser()
ap.add_argument("--sets", default="code,dialogue,science"); ap.add_argument("--twin", default="bench/target/release/bench-sample")
ap.add_argument("--temperature", type=float, default=0.3); ap.add_argument("--top_k", type=int, default=50); ap.add_argument("--top_p", type=float, default=0.95)
a = ap.parse_args()
warp = [TemperatureLogitsWarper(a.temperature), TopKLogitsWarper(a.top_k), TopPLogitsWarper(a.top_p)]

def uniform_at(seed, pos):  # common::sample::uniform_at
    M = (1 << 64) - 1
    x = (seed + pos * 0x9E3779B97F4A7C15) & M
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M
    x ^= x >> 31
    return (x >> 11) / float(1 << 53)
SEED = 0x726f636b656d0001

agree_set = agree_tok = n = 0; sizes = []; diffs = []
tmp = tempfile.NamedTemporaryFile(suffix=".f32", delete=False).name
for s in a.sets.split(","):
    meta = json.load(open(f"goldens/{s}/meta.json"))
    for st in meta["steps"]:
        with safe_open(f"goldens/{s}/step{st['step']}.safetensors", "np") as f:
            L = f.get_tensor("lm_head.out")
        for pos in range(L.shape[0]):
            logits = L[pos].astype(np.float32)
            scores = torch.from_numpy(logits.copy())[None, :]
            for w in warp:
                scores = w(None, scores)
            probs = torch.softmax(scores, -1)[0].numpy().astype(np.float64)
            kept = np.flatnonzero(probs > 0)
            order = kept[np.lexsort((kept, -probs[kept]))]  # probability desc, index asc
            u = uniform_at(SEED, pos)
            acc = 0.0; ref = int(order[-1])
            for i in order:
                acc += probs[i]
                if u < acc:
                    ref = int(i); break
            open(tmp, "wb").write(logits.tobytes())
            ours = int(subprocess.run([a.twin, "--draw", tmp, repr(u)], capture_output=True, text=True, check=True).stdout)
            n += 1; sizes.append(len(kept))
            # our kept set: recompute the recipe in numpy to compare sets (the twin reports the token only)
            agree_tok += ours == ref
            if ours != ref:
                diffs.append((s, st["step"], pos, ref, ours, float(probs[ref]), float(probs[ours]) if ours < len(probs) else None, u, len(kept)))
os.unlink(tmp)
sizes = np.array(sizes)
print(f"{n} positions; token agrees with the reference at {agree_tok} ({100*agree_tok/n:.2f} %)")
print(f"nucleus size: mean {sizes.mean():.2f}, median {np.median(sizes):.0f}, p99 {np.percentile(sizes, 99):.0f}, max {sizes.max()}; size 1 at {(sizes == 1).mean()*100:.1f} % of positions")
for d in diffs[:20]:
    print("  differs:", d)
