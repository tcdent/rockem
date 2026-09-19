# The wire

How a client talks to a node. The node's side is `node/src/wire.rs`, and this document
and that file's header say the same thing; if they ever disagree, the code is wrong.

> "What we're really doing is just streaming over TCP, and so I think building a protocol
> on a byte stream that is oriented specifically towards streaming lets us keep this super
> simple."

## Transport

One TCP connection carries one session at a time. Both directions are a stream of
little-endian unsigned 32-bit words. There is no framing beyond TCP's own: no length
fields, no delimiters, no escaping. A reader that started at the connection is always at a
word boundary, and control values live above the vocabulary, so no value ever needs to be
distinguished from data by anything but its size.

The stream in is the transcript the model sees. The stream out is the transcript it
produces. Nothing is transformed on either side, so a captured stream is a replay.

## Words

| value | meaning |
|---|---|
| `0 ≤ w < VOCAB_SIZE` (102,400 for V2-Lite) | a token id |
| `BOS` (100,000) | client → node: start a sequence. The attention cache becomes the sink's (the shipped state of position 0). BOS itself is never computed; the sequence continues from `FIRST_POS`. The draw at every later position is a function of the model's seed and the position, so a transcript always continues the same way. |
| `EOS` (100,001) | node → client: the generation ended. EOS is a real token: it is sampled, written, and fed back into the cache like any other. |
| `GENERATE` (`0xFFFFFFFF`) | client → node: sample from here. |
| `SCORE` (`0xFFFFFFFE`) | client → node: from here, answer every id fed with its log-probability. |
| anything else above the vocabulary | reserved; the node closes the connection |

BOS, EOS and the vocabulary size are constants of the model, and the bundle's manifest
states them in its `[model]` table (`docs/bundle.md`) so a client reads them from the
same directory the node loads. GENERATE and SCORE are constants of the protocol.

## Scoring

After SCORE, every id the client feeds is answered, before it is fed, with one word
holding an f32: ln p of that id given everything before it, under the model's own
distribution (no temperature, no top-k, no top-p; the sampling recipe is for draws). The
word is the f32's bits, little-endian, in the same four bytes an id would take; a client
knows which it is reading because it sent SCORE. Ids fed before SCORE are context and are
not scored. Scoring persists for the sequence: a GENERATE after it still streams ids, and
ids fed after that are scored again. BOS ends it.

```
client → node   BOS, id, id, …, SCORE, id, id, id
node → client   f32, f32, f32                       (one per id after SCORE, as each is fed)
```

This is what likelihood-scored benchmarks and rerankers need: the prompt fed unscored,
then the candidate's ids scored and summed. A prompt shared by several candidates is
served from the node's store on the second and later connections, with only its last
position recomputed, so four choices cost four candidates and one prompt.

## Header

At connect, framed by position: the first sixteen bytes the client writes, and the first
twenty the node writes back.

| offset | size | field | client → node | node → client |
|---|---|---|---|---|
| 0 | 4 | magic, the bytes `rock` | yes | yes |
| 4 | 4 | version, u32 LE, currently 3 | yes | yes |
| 8 | 8 | identity, u64 LE | the identity expected, or 0 | the node's |
| 16 | 4 | positions, u32 LE | | the positions a sequence may hold after BOS |

Positions is the node's context budget: how many ids, prompt and generated together, a
sequence can hold after its BOS. It is a property of the node, set at boot
(`rockem.ctx`, 2,048 by default), not of the bundle, because it is a memory budget chosen
per card; so the node reports it rather than the manifest. A client uses it to refuse a
prompt that cannot fit before sending it, and to tell a full context from a dropped
connection by its own count.

The identity is FNV-1a over the bytes of the bundle's `manifest.toml`, 64-bit, offset
basis `0xcbf29ce484222325`, prime `0x100000001b3`. The manifest names the checkpoint
revision, the recipe, the calibration text and the hash of every file, so equal
identities mean the client's tokenizer and the node's weights came from the same
checkpoint. A client sends `0` to accept any node. The node refuses a bad magic, a
different version, or a nonzero identity that is not its own, by closing the connection
without writing its header. Otherwise it writes its header and reads words.

## A session

```
client → node   header
node → client   header
client → node   BOS, id, id, id, …, GENERATE
node → client   id, id, id, …, EOS          (each word written the moment it is sampled)
client → node   id, id, …, GENERATE          (optional: continue the same sequence)
node → client   id, …, EOS
client → node   BOS, …                       (optional: a new sequence on the same connection)
```

Prompt ids are fed to the model as they arrive, in whatever steps arrive: ids already in
the socket's buffer go through the model together, up to 32 a step, and a trickle of
single ids goes one at a time. A position's result does not depend on how many stepped
with it (the node's own gate: every position's dumps byte-identical either way). On
GENERATE the node samples, writes the id, feeds it back, and repeats until it writes EOS
or the context is full. Every id that crosses the wire in either direction is in the
cache, so a continuation after EOS is exactly the transcript so far plus the new ids.

The node keeps the cache rows of every transcript it has computed, in host memory,
keyed by a running hash of the ids. As a prompt's ids arrive they are matched against
that store, and nothing is computed until the first id the node has not seen after that
prefix; the card is then seeded from the longest match and computing resumes there. A
client sees only that a repeated or extended transcript answers faster. The rows a hit
installs are the bytes a miss would have produced, so this is invisible to the stream.

The node does not read the socket while generating. Abort is closing the connection: the
node's next write fails, it stops, and it returns to accepting. There is no stop verb and
no token limit on the wire; the node runs to EOS or a full context (the header's
positions), and a full context closes the connection, since the sequence cannot continue.

A node serves one connection at a time, because it holds one attention cache. Nagle's
algorithm is off on the node's socket, so a four-byte token leaves when it is sampled
rather than when a segment fills.

## Inspecting it

The stream is readable with nothing but netcat. The bytes below are the header with
identity 0, then BOS, five prompt ids, and GENERATE:

```
(printf 'rock\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
 printf '\xa0\x86\x01\x00\x25\x02\x00\x00\xbd\x17\x00\x00\x18\x01\x00\x00\x47\x1c\x00\x00\x3d\x01\x00\x00\xff\xff\xff\xff'
 sleep 2) | nc -q 1 <node> 7332 | xxd
```

The first twenty bytes back are the node's header; every four after that are a token.
The console is the same protocol in text, for a human at a serial port: decimal ids on a
line, the newline is GENERATE, and the node writes token bytes instead of ids. It exists
so the wire can be exercised by hand and is not the interface a client speaks.

The edge's client is `api/rockem/wire.py`: a `Session` that is the protocol without a
socket (bytes out, ids in, the budget mirrored), and a `Client` around it. A client in any
language is small. In Python: open a socket, set `TCP_NODELAY`, send
`b"rock" + struct.pack("<IQ", 3, 0)`, read twenty bytes (`struct.unpack("<4sIQI", …)`
gives the magic, version, identity and positions), then `struct.pack("<I", w)` per word
out and `struct.unpack("<I", sock.recv(4))` per word in until EOS. Encoding the chat
template's output with the bundle's tokenizer yields BOS as the first id, and that id is
the wire's BOS, so a client sends what the tokenizer produced.

## Versioning

The version in the header covers the word alphabet and the header layout. Anything added
later, a token limit, a seed override, a reference to a compiled grammar for structured
output, is a new verb, a new header field, or both, under a new version. A node speaks
exactly one version and refuses the others; the client decides what to do with that.
Version 3 added SCORE and the f32 words it answers with. Version 2 added the positions
field to the node's header; version 1 had sixteen bytes each way and no way for a client
to learn the budget.

The one addition already sketched: structured output is a mask on the logits before the
draw. The grammar is compiled at the edge, where the tokenizer is, into an automaton over
token ids, and sent with the request or referenced by hash; the node holds a state, masks
by lookup, samples, and advances by the id it drew. It never sees a character.
