#!/usr/bin/env python3
"""The routing-trace replay: what a per-layer expert cache on the cards would hit.

`routing_replay.py <trace> [<trace> ...]` over files from `forward-glm --routing-trace` (lines
`pos layer e0..e7`). Per cache size (experts a layer on the cards), the share of picks served:
LRU and LFU (refreshed within the text), Belady's offline optimum (the ceiling of any refresh
policy), and a static list from the text itself (the ceiling of a hotlist; with a second trace,
the first's list applied to the second: whether a hotlist travels). A layer's cache is its own;
numbers are means over the MoE layers, picks weighted equally.
"""
import sys
from collections import Counter, defaultdict

SIZES = [8, 16, 24, 32, 48, 64, 96, 128, 192]
TOP_K = 8


def load(path):
    layers = defaultdict(list)  # layer -> [picks per position, in order]
    for line in open(path):
        f = line.split()
        if len(f) < 2 + TOP_K:
            continue
        layers[int(f[1])].append([int(x) for x in f[2:2 + TOP_K]])
    return layers


def lru(seq, size):
    cache, hits, n = {}, 0, 0
    for t, picks in enumerate(seq):
        for e in picks:
            n += 1
            if e in cache:
                hits += 1
            elif len(cache) >= size:
                victim = min(cache, key=cache.get)
                del cache[victim]
            cache[e] = t
        for e in picks:
            cache[e] = t
    return hits / n


def lfu(seq, size):
    cache, freq, hits, n = set(), Counter(), 0, 0
    for picks in seq:
        for e in picks:
            n += 1
            freq[e] += 1
            if e in cache:
                hits += 1
            else:
                if len(cache) >= size:
                    victim = min(cache, key=lambda x: freq[x])
                    cache.discard(victim)
                cache.add(e)
    return hits / n


def belady(seq, size):
    flat = [e for picks in seq for e in picks]
    nxt = {}
    future = [0] * len(flat)
    for i in range(len(flat) - 1, -1, -1):
        future[i] = nxt.get(flat[i], float("inf"))
        nxt[flat[i]] = i
    cache, hits = {}, 0
    for i, e in enumerate(flat):
        if e in cache:
            hits += 1
        elif len(cache) >= size:
            victim = max(cache, key=cache.get)
            del cache[victim]
        cache[e] = future[i]
    return hits / len(flat)


def static(seq, size, from_seq=None):
    src = from_seq if from_seq is not None else seq
    top = {e for e, _ in Counter(e for picks in src for e in picks).most_common(size)}
    flat = [e for picks in seq for e in picks]
    return sum(e in top for e in flat) / len(flat)


def mean(f, layers, size, **kw):
    return sum(f(seq, size, **kw) for seq in layers.values()) / len(layers)


def main():
    traces = [load(p) for p in sys.argv[1:]]
    for path, layers in zip(sys.argv[1:], traces):
        positions = len(next(iter(layers.values())))
        print(f"{path}: {positions} positions, {len(layers)} MoE layers")
        print(f"  {'experts/layer':>14} {'LRU':>6} {'LFU':>6} {'Belady':>7} {'static(self)':>13}" + (f" {'static(from ' + sys.argv[1] + ')':>22}" if layers is not traces[0] else ""))
        for size in SIZES:
            row = f"  {size:>14} {mean(lru, layers, size):6.3f} {mean(lfu, layers, size):6.3f} {mean(belady, layers, size):7.3f} {mean(static, layers, size):13.3f}"
            if layers is not traces[0]:
                first = traces[0]
                row += f" {sum(static(layers[l], size, from_seq=first[l]) for l in layers if l in first) / len(layers):22.3f}"
            print(row)


if __name__ == "__main__":
    main()
