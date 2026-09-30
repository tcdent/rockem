#!/usr/bin/env python3
"""The checkpoint's bytes by tensor class, from the shards' headers alone (no tensor is read):
what a token reads, what stays resident, at the checkpoint's own dtypes and at 4 bits.

  uv run python bytes.py --ckpt ~/hf-cache/glm-5.3-flash [--md]
"""
import argparse, collections, glob, json, os, re, struct

DT = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I8": 1, "U8": 1}


def classify(name):
    n = re.sub(r"\.\d+\.", ".N.", name)
    n = n.replace("model.language_model.", "")
    if n.startswith("model.visual") or "vision" in n:
        return "vision tower"
    if "mlp.experts." in n:
        return "routed experts"
    if "shared_experts" in n:
        return "shared experts"
    if re.search(r"layers\.N\.mlp\.(gate|up|down)_proj", n):
        return "dense FFN (layers 0-2)"
    if "mlp.gate" in n:
        return "router"
    if "indexer" in n:
        return "MLA indexer"
    if "self_attn" in n and any(k in n for k in ("q_a_", "q_b_", "kv_a_", "kv_b_")):
        return "MLA projections"
    if "self_attn" in n and any(k in n for k in ("conv1d", "forget", "f_a_", "f_b_", "A_log", "dt_bias", "b_proj", "g_a_", "g_b_", "o_norm", "q_proj", "k_proj", "v_proj")):
        return "KDA projections"
    if "self_attn.o_proj" in n:
        return "attention o_proj (KDA + MLA)"
    if "hc_" in n:
        return "mHC"
    if "layernorm" in n or n.endswith("norm.weight"):
        return "norms"
    if "embed_tokens" in n:
        return "embeddings"
    if n.startswith("lm_head"):
        return "lm_head"
    if any(k in n for k in ("eh_proj", "enorm", "hnorm", "shared_head")):
        return "MTP head"
    return "other: " + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    by = collections.defaultdict(lambda: [0, 0, 0, collections.Counter()])  # tensors, params, bytes, dtypes
    for f in sorted(glob.glob(os.path.join(a.ckpt, "model-*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k == "__metadata__":
                continue
            c = classify(k)
            p = 1
            for d in v["shape"]:
                p *= d
            if k.endswith("_scale_inv"):
                by[c][2] += p * DT[v["dtype"]]  # scales are bytes, not parameters
                continue
            by[c][0] += 1
            by[c][1] += p
            by[c][2] += p * DT[v["dtype"]]
            by[c][3][v["dtype"]] += 1
    rows = sorted(by.items(), key=lambda t: -t[1][2])
    tot_p = sum(v[1] for _, v in rows)
    tot_b = sum(v[2] for _, v in rows)
    if a.md:
        print("| class | tensors | parameters | bytes at the checkpoint's dtypes | at 4 bits + scales |\n|---|---|---|---|---|")
        for c, (t, p, b, dts) in rows:
            print(f"| {c} | {t} | {p / 1e9:.2f}B | {b / 1e9:.1f} GB ({', '.join(f'{k}×{v}' for k, v in dts.items())}) | {p * 0.5 * (1 + 4 / 128) / 1e9:.1f} GB |")
        print(f"| **total** | | **{tot_p / 1e9:.1f}B** | **{tot_b / 1e9:.1f} GB** | |")
    else:
        for c, (t, p, b, dts) in rows:
            print(f"{c:32s} {t:6d} tensors {p / 1e9:8.2f}B params {b / 1e9:8.1f} GB  {dict(dts)}")
        print(f"{'total':32s} {'':6s}         {tot_p / 1e9:8.1f}B        {tot_b / 1e9:8.1f} GB")


if __name__ == "__main__":
    main()
