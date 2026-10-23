#!/usr/bin/env python3
"""The gates' wire client: `wire.py <host> <ids...>` prompts the V2-Lite node, GENERATEs, prints the tokens.
SLOW=<s> sends the ids one at a time with that pause (sequential prefill); CAP=<n> stops
after n tokens. Uses api/rockem/wire.py, the same code the edge speaks."""
import asyncio, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "api"))
from rockem.wire import Client  # noqa: E402

BOS, EOS, VOCAB, MODEL = 100000, 100001, 102400, "deepseek-v2-lite"

async def main():
    host, ids = sys.argv[1], [int(t) for t in sys.argv[2:]]
    slow, cap = float(os.environ.get("SLOW", "0")), int(os.environ.get("CAP", "16"))
    c = await Client.connect((host, 7332), BOS, EOS, VOCAB, model=MODEL)
    t0 = time.time()
    if slow:
        await c.prompt(ids[:1])
        for w in ids[1:]:
            await c.feed([w]); await asyncio.sleep(slow)
    else:
        await c.prompt(ids)
    out, t1 = [], None
    async for t in c.generate():
        if t1 is None: t1 = time.time()
        out.append(t)
        if len(out) >= cap: break
    t2 = time.time()
    c.close()
    rate = f", then {(len(out) - 1) / (t2 - t1):.1f} tok/s" if len(out) > 1 else ""
    print(f"prompt {len(ids)} ids, first token after {t1 - t0:.3f}s{rate}; tokens: {out}")

asyncio.run(main())
