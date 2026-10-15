#!/usr/bin/env python3
"""The SCORE verb against two anchors: the node's own logits (the op, exact to ~1e-6)
and the reference's (the recipe's error, in nats).

  uv run python score_check.py --node <ip>:7332 --bundle ../bundles/v2lite-q4gptq --debug <dir the node wrote> --sets legal,code

For each golden set's sequence: BOS, SCORE, then every id, reading ln p per id from the
node. The node's own logits come from `rockem.debug=<dir>` dumps (pos<N>.f32, the logits
after computing position N; position N+1's id is scored by them), which must have been
written by this node for exactly these sequences. The reference's come from the goldens'
`lm_head.out`, position by position."""
import argparse, asyncio, json, sys
from pathlib import Path
import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
from rockem.bundle import Bundle  # noqa: E402
from rockem.wire import Client  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--node", required=True); ap.add_argument("--bundle", required=True)
ap.add_argument("--sets", default="legal,code"); ap.add_argument("--debug", help="the node's rockem.debug dir, for the exact check")
a = ap.parse_args()
host, port = a.node.rsplit(":", 1); addr = (host, int(port))
b = Bundle(a.bundle)

def logsoftmax(v):
    v = v.astype(np.float64); m = v.max(); return v - m - np.log(np.exp(v - m).sum())

async def score(ids):
    c = await Client.connect(addr, b.prefix, b.eos_tokens, b.vocab_size, b.identity)
    try:
        await c.prompt([])
        return await c.score(ids)
    finally:
        c.close()

for s in a.sets.split(","):
    meta = json.load(open(f"goldens/{s}/meta.json"))
    ids = list(meta["steps"][0]["input_ids"]) + [st["input_ids"][-1] for st in meta["steps"][1:]]
    assert ids[0] == b.bos
    got = np.array(asyncio.run(score(ids[1:])))  # got[i] = ln p(ids[i+1] | ids[..=i])
    # the reference's logits at every position: step 0 holds the prompt's, later steps one each
    ref = []
    with safe_open(f"goldens/{s}/step0.safetensors", "np") as f:
        L = f.get_tensor("lm_head.out")
        ref.extend(logsoftmax(L[p])[ids[p + 1]] for p in range(L.shape[0] - 1))
        last = L[-1]
    for st in meta["steps"][1:]:
        with safe_open(f"goldens/{s}/step{st['step']}.safetensors", "np") as f:
            cur = f.get_tensor("lm_head.out")[-1]
        pos = len(ref) + 1
        ref.append(logsoftmax(last)[ids[pos]])
        last = cur
    ref = np.array(ref[: len(got)])
    print(f"{s}: {len(got)} positions scored; vs the reference: mean ln p ours {got.mean():.4f} ref {ref.mean():.4f}, mean |Δ| {np.abs(got - ref).mean():.4f} nats, max |Δ| {np.abs(got - ref).max():.4f}")
    if a.debug:
        own = []
        for p in range(1, len(ids)):
            path = Path(a.debug) / f"pos{p}.f32"
            if not path.exists():
                break
            own.append(logsoftmax(np.fromfile(path, dtype=np.float32))[ids[p + 1]] if p + 1 < len(ids) else None)
        own = np.array([v for v in own if v is not None])
        n = min(len(own), len(got) - 1)
        d = np.abs(got[1 : 1 + n] - own[:n])
        print(f"  vs the node's own logits ({n} positions): max |Δ| {d.max():.2e}, mean {d.mean():.2e}  {'OK' if d.max() < 1e-4 else 'MISMATCH'}")
