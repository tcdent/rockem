#!/usr/bin/env python3
"""Where two `rockem.debug` dumps of the same position first part: `dump_where.py <dir a> <dir b> <pos>`
prints the first layer whose residual trace or router picks differ, or that every layer agrees and
whether the logits do. Standard library only: it runs on the GPU box."""
import sys, array
a, b, pos = sys.argv[1], sys.argv[2], int(sys.argv[3])
H, TOP_K = 2048, 6
def load(path, code):
    v = array.array(code); v.frombytes(open(path, "rb").read()); return v
la, lb = load(f"{a}/pos{pos}.layers.f32", "f"), load(f"{b}/pos{pos}.layers.f32", "f")
ta, tb = load(f"{a}/pos{pos}.topk.u32", "I"), load(f"{b}/pos{pos}.topk.u32", "I")
for l in range(len(la) // H):
    ha, hb = la[l * H:(l + 1) * H], lb[l * H:(l + 1) * H]
    ka, kb = ta[l * TOP_K:(l + 1) * TOP_K], tb[l * TOP_K:(l + 1) * TOP_K]
    if ha != hb or ka != kb:
        d = max(abs(x - y) for x, y in zip(ha, hb))
        print(f"pos {pos}: first parts at layer {l}: residual {'same' if ha == hb else f'max |Δ| {d:.3e}'}, picks {'same' if ka == kb else f'{ka.tolist()} vs {kb.tolist()}'}")
        break
else:
    ga, gb = load(f"{a}/pos{pos}.f32", "f"), load(f"{b}/pos{pos}.f32", "f")
    print(f"pos {pos}: every layer the same; logits {'same' if ga == gb else 'differ'}")
