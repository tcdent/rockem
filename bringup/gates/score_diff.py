#!/usr/bin/env python3
"""Two SCORE outputs of the same ids (the f32 bits in hex, as score_client.py writes them) compared:
the numeric gate for a change that is not bit-identical (a stored precision). Prints the mean and
max |Δ logprob|, the share of positions that moved, and each file's NLL."""
import sys, math

def load(p):
    # score_client.py writes each logprob as the hex of its f32 bits, space-separated
    import struct
    return [struct.unpack("<f", struct.pack("<I", int(w, 16)))[0] for w in open(p).read().split()]

a, b = load(sys.argv[1]), load(sys.argv[2])
n = min(len(a), len(b))
d = [abs(x - y) for x, y in zip(a[:n], b[:n])]
moved = sum(1 for x in d if x > 0)
print(f"{n} positions: mean |Δ logprob| {sum(d)/n:.3e}, max {max(d):.3e}, {moved} moved ({100*moved/n:.1f} %); NLL {-sum(a[:n])/n:.5f} vs {-sum(b[:n])/n:.5f}")
