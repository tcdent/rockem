#!/usr/bin/env python3
"""GPTQ (Frantar et al.) for the routed experts, written straight into the bundle's
q4g128 record format so the forward driver and the node read it unchanged. This is the
reference implementation the bundle builder's Rust quantizer (`bundle --calib`,
`common::q4::gptq`) is checked against with quant_check.py; the shipped path is the Rust one.

  uv run python quant.py --snapshot <dir> --out <bundle dir> --calib code,dialogue,science [--exclude-pos0] [--rtn]

Symmetric INT4, group 128, scale = absmax/7 per (row, group) from the error-fed weights
at the group's first column (standard GPTQ), integers chosen with Hessian error feedback
over the calibration inputs. The calibration set is every position of the named golden
sets (the post-attention-norm output at each MoE layer), optionally without position 0.
`--rtn` writes round-to-nearest with the same code path, as a control."""
import argparse, json, os, sys, time, struct
import numpy as np
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--snapshot", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--calib", default="code,dialogue,science"); ap.add_argument("--exclude-pos0", action="store_true")
ap.add_argument("--rtn", action="store_true"); ap.add_argument("--damp", type=float, default=0.01)
ap.add_argument("--layers", default="1-26")
ap.add_argument("--threads", type=int, default=8); ap.add_argument("--batch", type=int, default=16, help="experts per GPTQ pass")
a = ap.parse_args()
torch.set_num_threads(a.threads)
cfg = json.load(open(f"{a.snapshot}/config.json"))
H, D_FF, E, L0, NL = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["n_routed_experts"], cfg["first_k_dense_replace"], cfg["num_hidden_layers"]
GROUP, TILE = 128, 4

def round_away(x):
    """Round half away from zero, as `common::math::round` does (torch/numpy round is half-to-even)."""
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)
idx = json.load(open(f"{a.snapshot}/model.safetensors.index.json"))["weight_map"]
files = {}
def W(name):
    f = idx[name]
    if f not in files: files[f] = safe_open(f"{a.snapshot}/{f}", "pt")
    return files[f].get_tensor(name).float()
lo, hi = (int(v) for v in a.layers.split("-"))
layers = range(max(lo, L0), min(hi, NL - 1) + 1)

# calibration inputs per layer: every position's FFN input
calib = {l: [] for l in layers}
for s in a.calib.split(","):
    meta = json.load(open(f"goldens/{s}/meta.json"))
    for st in meta["steps"]:
        with safe_open(f"goldens/{s}/step{st['step']}.safetensors", "np") as f:
            for l in layers:
                x = f.get_tensor(f"model.layers.{l}.post_attention_layernorm.out")
                if a.exclude_pos0 and st["step"] == 0: x = x[1:]
                calib[l].append(x)
calib = {l: torch.from_numpy(np.concatenate(v, 0)) for l, v in calib.items()}
print(f"calibration: {a.calib}, {calib[layers[0]].shape[0]} positions per layer{' (position 0 excluded)' if a.exclude_pos0 else ''}", flush=True)

def hinv(Hh, damp):
    """Upper Cholesky factor U of H⁻¹ (H⁻¹ = UᵀU) for a batch of Hessians [E][k][k]; dead
    columns (zero diagonal) get a unit diagonal so the inverse exists (their weights are zeroed
    by the caller). Returns (U, dead)."""
    diag = torch.diagonal(Hh, dim1=-2, dim2=-1)
    dead = diag == 0
    diag[dead] = 1.0
    Hh = Hh + (damp * diag.mean(-1))[:, None, None] * torch.eye(Hh.shape[-1])
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hh))    # H = LLᵀ → H⁻¹, as the reference GPTQ does
    return torch.linalg.cholesky(Hinv, upper=True), dead

def gptq(Wm, U, dead):
    """GPTQ over a batch of matrices. Wm [E][m][k] f32, U [E or 1][k][k] the upper Cholesky
    factor of each matrix's H⁻¹ (shared when E' = 1), dead [E or 1][k]. Returns (q int8
    [E][m][k] in [-8,7], scales f32 [E][m][k/GROUP]). Same algorithm as the standard
    blocked GPTQ, column by column with error feedback; the work is laid out as [E][k][m]
    so each column's rank-1 update is a contiguous slab across all E matrices at once."""
    E, m, k = Wm.shape
    Wt = Wm.transpose(1, 2).contiguous()                    # [E][k][m]
    for e in range(E):
        Wt[e, dead[min(e, dead.shape[0] - 1)], :] = 0.0
    q = torch.empty(E, k, m, dtype=torch.int8); scales = torch.empty(E, k // GROUP, m)
    B = GROUP
    for i1 in range(0, k, B):
        i2 = min(i1 + B, k)
        Wb = Wt[:, i1:i2, :].clone(); Err = torch.zeros(E, i2 - i1, m); Hb = U[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % GROUP == 0:
                s = Wb[:, i:i + GROUP, :].abs().amax(1) / 7.0; s[s == 0] = 1.0
                scales[:, col // GROUP, :] = s
            w = Wb[:, i, :]
            qi = torch.clamp(round_away(w * (1.0 / s)), -8, 7)
            q[:, col, :] = qi.to(torch.int8)
            err = (w - qi * s) / Hb[:, i, i, None]
            Wb[:, i:, :] -= Hb[:, i, i:, None] * err[:, None, :]
            Err[:, i, :] = err
        Wt[:, i2:, :] -= U[:, i1:i2, i2:].transpose(1, 2) @ Err
    return q.transpose(1, 2), scales.transpose(1, 2)

def rtn(Wm):
    m, k = Wm.shape
    g = Wm.numpy().reshape(m, k // GROUP, GROUP); s = np.abs(g).max(-1) / 7.0; s[s == 0] = 1.0
    inv = (1.0 / s)[..., None]
    return np.clip(round_away(g * inv), -8, 7).astype(np.int8).reshape(m, k), s.astype(np.float32)

def pack(q, scales):
    """q [m][k] in [-8,7] → the tiled biased-u4 stream and [t][g][r] scales, as bytes."""
    m, k = q.shape
    nib = (q.astype(np.int16) + 8).astype(np.uint8)
    t = nib.reshape(m // TILE, TILE, k // GROUP, 2, 2, 32)      # (t, r, g, c, half, b)
    t = t.transpose(0, 2, 1, 3, 5, 4)                          # (t, g, r, c, b, half)
    bytes_ = (t[..., 0] | (t[..., 1] << 4)).astype(np.uint8).reshape(-1)
    sc = scales.reshape(m // TILE, TILE, k // GROUP).transpose(0, 2, 1).astype("<f4").reshape(-1)
    return bytes_.tobytes(), sc.tobytes()

RECORD = (2 * D_FF * H // 2) + 4 * (2 * D_FF * (H // GROUP)) + (H * D_FF // 2) + 4 * (H * (D_FF // GROUP))
os.makedirs(a.out, exist_ok=True)
path = f"{a.out}/experts.q4g128.bin"
out = open(path, "r+b" if os.path.exists(path) else "w+b")
out.truncate((NL - L0) * E * RECORD)
t0 = time.time()
for l in layers:
    X = calib[l]
    Ugu, dead_gu = hinv((X.T @ X)[None], a.damp)                     # one Hessian for every expert's gate/up
    for e0 in range(0, E, a.batch):
        es = range(e0, min(e0 + a.batch, E))
        gu = torch.stack([torch.cat([W(f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight"), W(f"model.layers.{l}.mlp.experts.{e}.up_proj.weight")], 0) for e in es])
        d = torch.stack([W(f"model.layers.{l}.mlp.experts.{e}.down_proj.weight") for e in es])
        if a.rtn:
            packed = [(rtn(gu[i]), rtn(d[i])) for i in range(len(es))]
        else:
            qg, sg = gptq(gu, Ugu, dead_gu)
            hgu = X @ gu.transpose(1, 2)                                   # [E][n][2·D_FF]
            hcal = hgu[:, :, :D_FF] / (1 + torch.exp(-hgu[:, :, :D_FF])) * hgu[:, :, D_FF:]   # the down projection's real inputs
            Ud, dead_d = hinv(hcal.transpose(1, 2) @ hcal, a.damp)
            qd, sd = gptq(d, Ud, dead_d)
            packed = [((qg[i].numpy(), sg[i].numpy()), (qd[i].numpy(), sd[i].numpy())) for i in range(len(es))]
        for i, e in enumerate(es):
            (bw, bs), (dw, ds) = pack(*packed[i][0]), pack(*packed[i][1])
            rec = bw + bs + dw + ds
            assert len(rec) == RECORD
            out.seek(((l - L0) * E + e) * RECORD); out.write(rec)
    print(f"layer {l} done ({time.time()-t0:.0f}s)", flush=True)
out.close()
print(f"wrote {path}")
