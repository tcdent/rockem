#!/usr/bin/env python3
"""YaRN rotary tables and the attention softmax scale for the pinned V2-Lite config,
computed with the pinned modeling file's own functions, so the Rust side has a
number to cross-check against.

  uv run python rope_tables.py            # print
  uv run python rope_tables.py --out goldens/v2lite   # also write rope.json
"""
import argparse, json, os, sys
import torch
from goldens import load_oracle, ORACLE_DIR

_, _ = load_oracle()
m = sys.modules["oracle.modeling_deepseek"]

cfg = json.load(open(os.path.join(ORACLE_DIR, "config.json")))
rs = cfg["rope_scaling"]
dim, theta = cfg["qk_rope_head_dim"], cfg["rope_theta"]

emb = m.DeepseekV2YarnRotaryEmbedding(
    dim, max_position_embeddings=cfg["max_position_embeddings"], base=theta,
    scaling_factor=rs["factor"], original_max_position_embeddings=rs["original_max_position_embeddings"],
    beta_fast=rs["beta_fast"], beta_slow=rs["beta_slow"], mscale=rs["mscale"], mscale_all_dim=rs["mscale_all_dim"])
inv_freq = emb.inv_freq.tolist()
mscale = float(m.yarn_get_mscale(rs["factor"], rs["mscale"]) / m.yarn_get_mscale(rs["factor"], rs["mscale_all_dim"]))
q_head_dim = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
softmax_scale = q_head_dim ** -0.5
ms = m.yarn_get_mscale(rs["factor"], rs["mscale_all_dim"])
softmax_scale *= ms * ms
low, high = m.yarn_find_correction_range(rs["beta_fast"], rs["beta_slow"], dim, theta, rs["original_max_position_embeddings"])

out = dict(dim=dim, theta=theta, factor=rs["factor"], correction_range=[low, high],
           cos_sin_mscale=mscale, softmax_scale=softmax_scale, inv_freq=inv_freq,
           cos_pos1=emb.cos_cached[1].tolist(), sin_pos1=emb.sin_cached[1].tolist())
print(json.dumps({k: v for k, v in out.items() if k not in ("cos_pos1", "sin_pos1")}, indent=1))

ap = argparse.ArgumentParser(); ap.add_argument("--out"); a = ap.parse_args()
if a.out:
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "rope.json"), "w"), indent=1)
    # the tables at long positions, for the model crate's `rope_matches_the_oracle_at_long_positions`
    positions = [1, 4095, 4096, 16383, 65535, 99999, 163839]
    emb._set_cos_sin_cache(seq_len=cfg["max_position_embeddings"], device="cpu", dtype=torch.float32)
    long = dict(positions=positions, cos=[emb.cos_cached[p].tolist() for p in positions], sin=[emb.sin_cached[p].tolist() for p in positions])
    json.dump(long, open(os.path.join(a.out, "rope_long.json"), "w"))
    print("wrote rope.json and rope_long.json")
