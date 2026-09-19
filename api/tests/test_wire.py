import asyncio
import struct

import pytest
from rockem import bundle, wire
from rockem.wire import ContextFull, Disconnected, Finish, Header, Refused, Session, WireError

from fake_node import BOS, EOS, VOCAB, FakeNode

IDENTITY = 0x0123456789ABCDEF


def test_hello_is_the_documented_bytes():
    assert wire.hello(0) == b"rock\x03\x00\x00\x00" + b"\x00" * 8
    assert wire.hello(IDENTITY) == b"rock" + struct.pack("<IQ", 3, IDENTITY)


def test_header_parses_and_refuses():
    h = wire.parse_header(b"rock" + struct.pack("<IQI", 3, IDENTITY, 2047))
    assert h == Header(IDENTITY, 2047)
    with pytest.raises(WireError):
        wire.parse_header(b"rocj" + struct.pack("<IQI", 3, 0, 1))
    with pytest.raises(WireError):
        wire.parse_header(b"rock" + struct.pack("<IQI", 1, 0, 1))
    with pytest.raises(Refused):
        wire.parse_header(b"")


def test_identity_is_fnv1a64():
    assert bundle.identity(b"") == 0xCBF29CE484222325
    assert bundle.identity(b"a") == 0xAF63DC4C8601EC8C
    assert bundle.identity(b"foobar") == 0x85944171F73967E8


def session(positions=8) -> Session:
    return Session(Header(IDENTITY, positions), BOS, EOS, VOCAB)


def test_session_mirrors_the_budget():
    s = session(4)
    with pytest.raises(WireError):
        s.feed([1])
    assert s.start() == struct.pack("<I", BOS)
    assert s.feed([1, 2, 3]) == struct.pack("<3I", 1, 2, 3)
    assert s.remaining == 1
    with pytest.raises(ContextFull):
        s.feed([4, 5])
    with pytest.raises(WireError):
        s.feed([BOS])
    with pytest.raises(WireError):
        s.feed([VOCAB])
    assert s.generate() == b"\xff\xff\xff\xff"
    # one id fills the context, one more is the draw the node writes before it closes
    assert s.receive(struct.pack("<2I", 9, 10)) == [9, 10]
    assert s.remaining == -1
    with pytest.raises(WireError):
        s.receive(struct.pack("<I", 11))
    assert s.eof() is Finish.LENGTH


def test_session_buffers_partial_words_and_reads_eos():
    s = session()
    s.start()
    s.feed([1])
    s.generate()
    word = struct.pack("<I", 5)
    assert s.receive(word[:3]) == []
    assert s.receive(word[3:] + struct.pack("<I", EOS)) == [5, EOS]
    assert s.ended
    with pytest.raises(Disconnected):  # a close after EOS is not the protocol
        s.eof()
    s.feed([6])  # continuing resets
    assert not s.ended
    s.generate()
    s.receive(word[:1])
    with pytest.raises(Disconnected):
        s.eof()
    with pytest.raises(WireError):
        session().start() and session().receive(struct.pack("<I", wire.GENERATE))


def test_session_eof_with_room_is_a_disconnect():
    s = session()
    s.start()
    s.feed([1])
    s.generate()
    s.receive(struct.pack("<I", 2))
    with pytest.raises(Disconnected):
        s.eof()


@pytest.fixture
def node():
    n = FakeNode(IDENTITY, max_ctx=8, draws=[10, 11, 12])
    yield n
    n.close()


def connect(node, identity=0):
    return wire.Client.connect(node.addr, BOS, EOS, VOCAB, identity=identity)


async def collect(gen):
    return [id async for id in gen]


@pytest.mark.anyio
async def test_client_generates_to_eos_and_continues(node):
    async with await connect(node, IDENTITY) as c:
        assert c.positions == 7
        await c.prompt([1, 2])
        assert await collect(c.generate()) == [10, 11, 12, EOS]
        assert c.finish is Finish.STOP
        assert c.session.remaining == 7 - 2 - 4
        await c.feed([3])  # continue the same sequence after EOS
        with pytest.raises(ContextFull):
            await c.feed([4, 5])
        await c.prompt([BOS, 9])  # a new sequence on the same connection; the tokenizer's BOS is the wire's
        assert await collect(c.generate()) == [10, 11, 12, EOS]
        with pytest.raises(WireError):
            await c.prompt([9, BOS])
    await asyncio.sleep(0.05)
    assert node.transcripts == [[1, 2, 10, 11, 12, EOS, 3], [9, 10, 11, 12, EOS]]


@pytest.mark.anyio
async def test_client_reads_a_full_context_as_length(node):
    node.draws = [20] * 100  # never EOS
    async with await connect(node) as c:
        await c.prompt([1, 2, 3, 4, 5])  # 2 positions left
        ids = await collect(c.generate())
        assert ids == [20, 20, 20]  # the two that fit, and the draw written before the close
        assert c.finish is Finish.LENGTH
    async with await connect(node) as c:
        await c.prompt([1] * 7)  # exactly full
        assert await collect(c.generate()) == [20]
        assert c.finish is Finish.LENGTH
        with pytest.raises(ContextFull):
            await c.prompt([1] * 8)


@pytest.mark.anyio
async def test_client_abort_is_close_and_the_node_returns_to_accepting(node):
    node.draws = [20] * 100
    c = await connect(node)
    await c.prompt([1])
    g = c.generate()
    assert await anext(g) == 20
    c.close()
    async with await connect(node) as c2:
        await c2.prompt([2])
        assert await anext(c2.generate()) == 20


@pytest.mark.anyio
async def test_client_refused_on_identity(node):
    with pytest.raises(Refused):
        await connect(node, identity=IDENTITY ^ 1)
    async with await connect(node, identity=IDENTITY) as c:
        assert c.header.identity == IDENTITY
