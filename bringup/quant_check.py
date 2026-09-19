#!/usr/bin/env python3
"""Reconstruction error of packed expert streams against the bf16 weights, on the
oracle's own expert inputs — the per-matrix quantity GPTQ minimizes, measured
outside of it.

  uv run python quant_check.py --snapshot <dir> --layer 10 --calib code,dialogue,science --streams rtn=<file> ref=<file> rs=<file>

For every expert of the layer and every stream: ‖X·(W−Q)ᵀ‖² / ‖X·Wᵀ‖² for gate/up
on the layer's post-norm inputs X, and for down on that expert's SwiGLU hidden
states of X. Prints per-stream mean/max over the layer's experts and the pairwise
winner counts, so a new quantizer is checked against the reference on the number
it is supposed to optimize rather than on byte identity."""
import argparse, json
import numpy as np
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--snapshot", required=True); ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--calib", default="code,dialogue,science"); ap.add_argument("--streams", nargs="+", required=True)
a = ap.parse_args()
cfg = json.load(open(f"{a.snapshot}/config.json"))
H, D_FF, E, L0 = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["n_routed_experts"], cfg["first_k_dense_replace"]
GROUP, TILE = 128, 4
RECORD = (2 * D_FF * H // 2) + 4 * (2 * D_FF * (H // GROUP)) + (H * D_FF // 2) + 4 * (H * (D_FF // GROUP))
idx = json.load(open(f"{a.snapshot}/model.safetensors.index.json"))["weight_map"]; files = {}
def W(name):
    f = idx[name]
    if f not in files: files[f] = safe_open(f"{a.snapshot}/{f}", "pt")
    return files[f].get_tensor(name).float().numpy()

def unpack(m, k, wb, sb):
    """Inverse of quant.py's pack: tiled biased-u4 bytes + [t][g][r] scales → dequantized [m][k] f32."""
    t = np.frombuffer(wb, np.uint8).reshape(m // TILE, k // GROUP, TILE, 2, 32)       # (t, g, r, c, b)
    nib = np.stack([t & 15, t >> 4], -1)                                              # (t, g, r, c, b, half)
    q = nib.transpose(0, 2, 1, 3, 5, 4).reshape(m, k).astype(np.float32) - 8
    s = np.frombuffer(sb, "<f4").reshape(m // TILE, k // GROUP, TILE).transpose(0, 2, 1).reshape(m, k // GROUP)
    return q * np.repeat(s, GROUP, 1)

bw, bs, dw, ds = 2 * D_FF * H // 2, 4 * (2 * D_FF * (H // GROUP)), H * D_FF // 2, 4 * (H * (D_FF // GROUP))
def record(path, e):
    with open(path, "rb") as f:
        f.seek(((a.layer - L0) * E + e) * RECORD); r = f.read(RECORD)
    return unpack(2 * D_FF, H, r[:bw], r[bw:bw + bs]), unpack(H, D_FF, r[bw + bs:bw + bs + dw], r[bw + bs + dw:])

X = []
for s in a.calib.split(","):
    meta = json.load(open(f"goldens/{s}/meta.json"))
    for st in meta["steps"]:
        with safe_open(f"goldens/{s}/step{st['step']}.safetensors", "np") as f:
            X.append(f.get_tensor(f"model.layers.{a.layer}.post_attention_layernorm.out"))
X = np.concatenate(X, 0)
streams = dict(s.split("=", 1) for s in a.streams)
names = list(streams)
errs = {n: {"gate/up": [], "down": []} for n in names}
for e in range(E):
    p = f"model.layers.{a.layer}.mlp.experts.{e}."
    gu = np.concatenate([W(p + "gate_proj.weight"), W(p + "up_proj.weight")], 0); d = W(p + "down_proj.weight")
    ygu = X @ gu.T
    hid = ygu[:, :D_FF] / (1 + np.exp(-ygu[:, :D_FF])) * ygu[:, D_FF:]
    yd = hid @ d.T
    for n in names:
        qgu, qd = record(streams[n], e)
        errs[n]["gate/up"].append(((X @ (gu - qgu).T) ** 2).sum() / (ygu ** 2).sum())
        errs[n]["down"].append(((hid @ (d - qd).T) ** 2).sum() / (yd ** 2).sum())
print(f"layer {a.layer}, {X.shape[0]} calibration rows, {E} experts; relative reconstruction error ‖X(W−Q)ᵀ‖²/‖XWᵀ‖²")
for part in ("gate/up", "down"):
    print(f"  {part}:")
    for n in names:
        v = np.array(errs[n][part])
        print(f"    {n:>6}: mean {v.mean():.3e}  median {np.median(v):.3e}  max {v.max():.3e}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            vi, vj = np.array(errs[names[i]][part]), np.array(errs[names[j]][part])
            print(f"    {names[i]} < {names[j]} on {(vi < vj).sum()}/{E} experts; ratio {names[i]}/{names[j]} mean {np.mean(vi / vj):.3f}")
