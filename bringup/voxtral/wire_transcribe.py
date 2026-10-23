#!/usr/bin/env python3
"""`wire_transcribe.py <host> <bundle dir> <set dir>`: a transcription straight over the wire (v5): the set's
`prompt.json` ids and `windows.f32` (bringup/voxtral/windows.py) through the API's client to the node's Voxtral
model — AUDIO windows, the placeholders, GENERATE — and the ids back, decoded through the bundle's tokenizer. The
node-side gate the forward-voxtral driver cannot give: the engine, AUDIO's receive, the store's window digest."""
import asyncio, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "api"))
from rockem.bundle import Bundle  # noqa: E402
from rockem.wire import AudioSpec, Client  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

async def main():
    host, bundle, sdir = sys.argv[1], sys.argv[2], sys.argv[3]
    m = Bundle(bundle).model("voxtral-small-24b")
    prompt = json.load(open(os.path.join(sdir, "prompt.json")))
    ids, n = prompt["ids"], prompt["windows"]
    raw = open(os.path.join(sdir, "windows.f32"), "rb").read()
    per = m.audio["window"][0] * m.audio["window"][1] * 4
    assert len(raw) == n * per, (len(raw), n, per)
    windows = [raw[i * per:(i + 1) * per] for i in range(n)]
    tok = Tokenizer.from_file(os.path.join(bundle, "voxtral-small-24b", "tokenizer.json"))
    t0 = time.time()
    spec = AudioSpec(placeholder=m.audio["placeholder"], window_bytes=per, rows=m.audio["rows"])
    c = await Client.connect((host, 7332), m.prefix, m.eos_tokens, m.vocab_size, m.identity, model=m.name, audio=spec)
    t1 = time.time()
    await c.prompt(ids, windows=windows)
    out, t_first = [], None
    async for id in c.generate(limit=64):
        if t_first is None:
            t_first = time.time()
        out.append(id)
    t2 = time.time()
    c.close()
    print(f"transcript ({len(out)} ids): {tok.decode(out, skip_special_tokens=True)!r}")
    print(f"connect {t1-t0:.2f}s; prompt + first token {t_first-t1:.2f}s ({len(ids)} ids, {n} window(s)); {len(out)} tokens at {(t2-t_first)*1e3/max(1,len(out)-1):.0f} ms each")

asyncio.run(main())
