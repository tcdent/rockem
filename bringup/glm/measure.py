#!/usr/bin/env python3
"""Measurements over oracle runs (analysis.md §2–§7):

  uv run python measure.py ranges  <run>...            # §5/§6: per-layer |residual| max/rms, position 0 against the rest, expert in/out ranges
  uv run python measure.py routing <run>...            # §7: experts touched, histogram concentration, pick margins (8th vs 9th score)
  uv run python measure.py kl <reference run> <run>... # §3/§4: KL(ref‖run) per position, top-1 agreement, NLL delta, from the logits

A run is a directory oracle.py wrote (edges.safetensors, routing.safetensors, meta.json).
"""
import json, os, sys
import torch
from safetensors.torch import load_file


def load(run):
    e = load_file(os.path.join(run, "edges.safetensors"))
    r = load_file(os.path.join(run, "routing.safetensors")) if os.path.exists(os.path.join(run, "routing.safetensors")) else {}
    m = json.load(open(os.path.join(run, "meta.json")))
    return e, r, m


def ranges(runs):
    for run in runs:
        e, r, m = load(run)
        L = m["layers"]
        print(f"== {run}: {len(m['ids'])} positions, {L} layers")
        print("layer  |resid| max   rms   pos0 max   pos0/rest max ratio | ffn_in max  routed out max  shared out max")
        for l in range(L):
            h = e[f"l{l}.out"]
            rest = h[1:].abs().max() if h.shape[0] > 1 else torch.tensor(float("nan"))
            fi = e.get(f"l{l}.ffn_in"); ro = e.get(f"l{l}.routed"); so = e.get(f"l{l}.shared")
            print(f"{l:5d}  {h.abs().max():10.1f} {h.pow(2).mean().sqrt():6.2f} {h[0].abs().max():9.1f}  {h[0].abs().max() / rest:6.2f}        | "
                  + (f"{fi.abs().max():8.1f}" if fi is not None else "       -") + (f"  {ro.abs().max():12.1f}" if ro is not None else "             -") + (f"  {so.abs().max():12.1f}" if so is not None else ""))


def routing(runs):
    for run in runs:
        e, r, m = load(run)
        print(f"== {run}: {len(m['ids'])} positions")
        print("layer  touched/288  top-8 share  top-16 share  min margin(8th-9th)  median margin")
        for l in range(m["layers"]):
            if f"l{l}.topk" not in r:
                continue
            idx = r[f"l{l}.topk"]; sc = r[f"l{l}.scores"]
            S = idx.shape[0]
            hist = torch.bincount(idx.reshape(-1).long(), minlength=288).float()
            share = hist.sort(descending=True).values
            srt = torch.sigmoid(sc).sort(-1, descending=True).values
            margin = srt[:, 7] - srt[:, 8]
            print(f"{l:5d}  {int((hist > 0).sum()):11d}  {share[:8].sum() / hist.sum():10.2f}  {share[:16].sum() / hist.sum():11.2f}  {margin.min():18.4f}  {margin.median():13.4f}")


def kl(ref, runs):
    er, rr, mr = load(ref)
    lr = torch.log_softmax(er["logits"], -1)
    ids = torch.tensor(mr["ids"])
    for run in runs:
        e, r, m = load(run)
        lq = torch.log_softmax(e["logits"], -1)
        k = (lr.exp() * (lr - lq)).sum(-1)
        top = (lr.argmax(-1) == lq.argmax(-1)).float().mean()
        nll_r = -lr[torch.arange(len(ids) - 1), ids[1:]].mean(); nll_q = -lq[torch.arange(len(ids) - 1), ids[1:]].mean()
        flips = sum(int(not torch.equal(rr[k2], r[k2])) for k2 in rr if k2.endswith(".topk"))
        print(f"{run}: KL mean {k.mean():.4f} max {k.max():.4f} nats  top-1 agreement {top:.3f}  NLL {nll_r:.4f} -> {nll_q:.4f} (+{nll_q - nll_r:.4f})  layers with a routing change {flips}  q4 {m.get('q4')}")


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    {"ranges": lambda: ranges(rest), "routing": lambda: routing(rest), "kl": lambda: kl(rest[0], rest[1:])}[cmd]()
