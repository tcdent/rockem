#!/usr/bin/env python3
"""`score_client.py <host> <ids...>`: SCORE the ids on a fresh sequence and print every ln p as
its exact bits, one line. For run-to-run comparisons of a node without rockem.debug."""
import asyncio, os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "api"))
from rockem.wire import Client  # noqa: E402
BOS, EOS, VOCAB = 100000, 100001, 102400

async def main():
    host, ids = sys.argv[1], [int(t) for t in sys.argv[2:]]
    if ids and ids[0] == BOS:
        ids = ids[1:]
    c = await Client.connect((host, 7332), BOS, EOS, VOCAB)
    c.writer.write(c.session.start())
    await c.writer.drain()
    lp = await c.score(ids)
    c.close()
    print(" ".join(f"{struct.unpack('<I', struct.pack('<f', v))[0]:08x}" for v in lp))

asyncio.run(main())
