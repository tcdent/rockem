#!/usr/bin/env python3
"""turns_wire.py <bundle dir> <label> [sizes] [reps] [gen] [sends]: turn prefills shaped like a coding agent's, over the wire.

The 18k conversation (requests/cody-user.json) rendered by the API's own edge is the first prompt; every turn after it
is the previous turn's whole transcript (its prompt and the id the node generated) followed by `size` new ids of
log-like text — a tool's output — and one generated id, so the node resumes each turn from the cached end of the last,
as an agent's turns do. Each size runs in groups of four consecutive turns (a lab node cycles a setting per request,
076) `reps` times. The time a turn takes is the client's: the prompt sent to the first id back — the pause before a
reply. Printed per turn, and kept as logs/turns-<label>.tsv. `gen` ids are generated a turn (1); with `sends` > 1 each
turn's prompt is sent again that many times in all, every send's ids compared with the first's (the same prompt resumed
from the same state must generate the same ids). Run from api/ (`uv run python ../bringup/box/turns_wire.py`)."""
import asyncio, json, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "api"))
from rockem.bundle import Bundle  # noqa: E402
from rockem.edge import Edge  # noqa: E402
from rockem.wire import Client  # noqa: E402

NODE = ("192.168.88.6", 7332)
WORDS = ["src", "node", "glm", "kfd", "rows", "exp4", "bench", "card", "step", "store", "wire", "prow", "lane", "wave"]


def tool_output(rng: random.Random, edge: Edge, size: int) -> list[int]:
    lines, ids = [], []
    while len(ids) < size:
        for _ in range(64):
            lines.append(f"{rng.choice(['drwxr-xr-x', '-rw-r--r--', '-rwxr-xr-x'])} 1 slomp slomp {rng.randint(100, 99999):>6} Oct 25 "
                         f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d} {'/'.join(rng.choice(WORDS) for _ in range(rng.randint(1, 4)))}_{rng.randint(0, 9999)}"
                         f".{rng.choice(['rs', 'toml', 'md', 'json', 'sh'])}")
        ids = edge.encode("\n".join(lines))
    return ids[:size]


async def turn(m, ids: list[int], gen: int = 1) -> tuple[float, list[int]]:
    c = await Client.connect(NODE, m.prefix, m.eos_tokens, m.vocab_size, m.identity, model=m.name)
    try:
        t0 = time.monotonic()
        await c.prompt(ids)
        out = []
        # the node sends `gen` ids and keeps the sequence open for more: stop at the last
        t1 = None
        async for i in c.generate(gen):
            t1 = t1 or time.monotonic()
            out.append(i)
            if len(out) == gen:
                break
        return t1 - t0, out
    finally:
        c.close()


async def main():
    bundle, label = sys.argv[1], sys.argv[2]
    sizes = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "128,300,700,1500,3000").split(",")]
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    gen = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    sends = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    m = Bundle(bundle).model("glm-5.3-flash")
    edge = Edge(m)
    base = edge.encode_chat(json.load(open(os.path.join(HERE, "requests", "cody-user.json")))["messages"])
    dt, out = await turn(m, base)
    transcript = base + out
    print(f"prime: {len(base)} ids, {dt:.2f} s", flush=True)
    rng = random.Random(76)
    rows = []
    k = 0
    for size in sizes:
        for rep in range(reps):
            for slot in range(4):
                k += 1
                ids = transcript + tool_output(rng, edge, size)
                first = None
                for send in range(sends):
                    dt, out = await turn(m, ids, gen)
                    same = "" if first is None else ("\tsame" if out == first else f"\tDIFFERENT {first} vs {out}")
                    first = first or out
                    rows.append((k, size, rep, slot, len(ids), dt))
                    print(f"{k}\tsize {size}\trep {rep}\tslot {slot}\tsend {send}\tcontext {len(ids)}\t{dt:.3f} s{same}", flush=True)
                transcript = ids + out
    with open(os.path.join(HERE, "logs", f"turns-{label}.tsv"), "w") as f:
        f.write("turn\tsize\trep\tslot\tcontext\tseconds\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


asyncio.run(main())
