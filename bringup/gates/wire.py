#!/usr/bin/env python3
"""The gates' wire client: `wire.py <host> <ids...>` prompts, GENERATEs, prints the tokens.
SLOW=<s> sends the ids one at a time with that pause (sequential prefill); CAP=<n> stops
after n tokens. Uses api/rockem/wire.py, the same code the edge speaks."""
import asyncio, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "api"))
from rockem.wire import Client  # noqa: E402

BOS, EOS, VOCAB = 100000, 100001, 102400

async def main():
    host, ids = sys.argv[1], [int(t) for t in sys.argv[2:]]
    slow, cap = float(os.environ.get("SLOW", "0")), int(os.environ.get("CAP", "16"))
    c = await Client.connect((host, 7332), BOS, EOS, VOCAB)
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
    c.close()
    print(f"prompt {len(ids)} ids, first token after {t1 - t0:.3f}s; tokens: {out}")

asyncio.run(main())
