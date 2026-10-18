#!/usr/bin/env python3
"""The routing-trace replay: what a per-layer expert cache on the cards would hit.

`routing_replay.py <trace> [<trace> ...]` over files from `forward-glm --routing-trace` (lines
`pos layer e0..e7`). Per cache size (experts a layer on the cards), the share of picks served:
LRU and LFU (refreshed within the text), the node's rule (`node`: least-picked replaced past a margin
of 4), ARC, Belady's offline optimum (the ceiling of any refresh
policy), and a static list from the text itself (the ceiling of a hotlist; with a second trace,
the first's list applied to the second: whether a hotlist travels). A layer's cache is its own;
numbers are means over the MoE layers, picks weighted equally.
"""
import sys
from collections import Counter, defaultdict

SIZES = [8, 16, 18, 21, 24, 32, 48, 64, 96, 128, 192]
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


def resident(seq, size, margin=2, half=32):
    """The node's rule (glm_kfd.rs `Resident`): a miss takes a free slot, else replaces the layer's
    least-picked resident once the miss's count leads it by `margin`; the counts halved every `half`
    positions (2026-10-18; before that lifetime counts and a margin of 4: 0.406 / 0.285 at 21 slots)."""
    cache, count, hits, n = set(), Counter(), 0, 0
    for t, picks in enumerate(seq):
        if half and t and t % half == 0:
            for k in list(count):
                count[k] >>= 1
        for e in picks:
            n += 1
            count[e] += 1
            if e in cache:
                hits += 1
            elif len(cache) < size:
                cache.add(e)
            else:
                victim = min(cache, key=lambda x: count[x])
                if count[e] >= count[victim] + margin:
                    cache.discard(victim)
                    cache.add(e)
    return hits / n


def arc(seq, size):
    """ARC (Megiddo & Modha 2003): T1 recent-once, T2 recent-twice, ghosts B1/B2 steer the split p."""
    from collections import OrderedDict
    t1, t2, b1, b2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    p, hits, n = 0, 0, 0

    def replace(e):
        nonlocal p
        if t1 and (len(t1) > p or (e in b2 and len(t1) == p)):
            old, _ = t1.popitem(last=False)
            b1[old] = None
        else:
            old, _ = t2.popitem(last=False)
            b2[old] = None

    for picks in seq:
        for e in picks:
            n += 1
            if e in t1 or e in t2:
                hits += 1
                t1.pop(e, None)
                t2.pop(e, None)
                t2[e] = None
            elif e in b1:
                p = min(size, p + max(1, len(b2) // max(1, len(b1))))
                replace(e)
                del b1[e]
                t2[e] = None
            elif e in b2:
                p = max(0, p - max(1, len(b1) // max(1, len(b2))))
                replace(e)
                del b2[e]
                t2[e] = None
            else:
                if len(t1) + len(b1) == size:
                    if len(t1) < size:
                        b1.popitem(last=False)
                        replace(e)
                    else:
                        t1.popitem(last=False)
                elif len(t1) + len(b1) < size and len(t1) + len(t2) + len(b1) + len(b2) >= size:
                    if len(t1) + len(t2) + len(b1) + len(b2) == 2 * size:
                        b2.popitem(last=False)
                    replace(e)
                t1[e] = None
    return hits / n


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
        print(f"  {'experts/layer':>14} {'LRU':>6} {'LFU':>6} {'node':>6} {'ARC':>6} {'Belady':>7} {'static(self)':>13}" + (f" {'static(from ' + sys.argv[1] + ')':>22}" if layers is not traces[0] else ""))
        for size in SIZES:
            row = f"  {size:>14} {mean(lru, layers, size):6.3f} {mean(lfu, layers, size):6.3f} {mean(resident, layers, size):6.3f} {mean(arc, layers, size):6.3f} {mean(belady, layers, size):7.3f} {mean(static, layers, size):13.3f}"
            if layers is not traces[0]:
                first = traces[0]
                row += f" {sum(static(layers[l], size, from_seq=first[l]) for l in layers if l in first) / len(layers):22.3f}"
            print(row)


if __name__ == "__main__":
    main()
