#!/usr/bin/env python3
"""`score_client.py <host> <bundle dir> <model> <ids...>`: SCORE the ids on a fresh sequence of the bundle's
named model and print every ln p as its exact bits, one line. For run-to-run and build-to-build comparisons
of a node without rockem.debug: a prompt longer than a step goes through the step path."""
import asyncio, os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "api"))
from rockem.bundle import Bundle  # noqa: E402
from rockem.wire import Client  # noqa: E402
async def main():
    host, m, ids = sys.argv[1], Bundle(sys.argv[2]).model(sys.argv[3]), [int(t) for t in sys.argv[4:]]
    if ids[: len(m.prefix)] == list(m.prefix):
        ids = ids[len(m.prefix):]
    c = await Client.connect((host, 7332), m.prefix, m.eos_tokens, m.vocab_size, m.identity, model=m.name)
    # BOS, then the rest of the model's prefix as plain ids (SCORE needs the prefix complete)
    c.writer.write(c.session.start() + c.session.feed(list(m.prefix[1:])))
    await c.writer.drain()
    lp = await c.score(ids)
    c.close()
    print(" ".join(f"{struct.unpack('<I', struct.pack('<f', v))[0]:08x}" for v in lp))
asyncio.run(main())
