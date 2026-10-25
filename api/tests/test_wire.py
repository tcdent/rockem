import contextlib
import asyncio
import struct

import pytest
from rockem import bundle, wire
from rockem.wire import ContextFull, Disconnected, Finish, Header, Refused, Session, WireError

from fake_node import BOS, EOS, VOCAB, FakeNode

IDENTITY = 0x0123456789ABCDEF


def test_hello_is_the_documented_bytes():
    assert wire.hello("tiny") == b"rock\x05\x00\x00\x00" + b"\x00" * 8 + b"tiny" + b"\x00" * 28
    assert wire.hello("glm-5.3-flash", IDENTITY) == b"rock" + struct.pack("<IQ", 5, IDENTITY) + b"glm-5.3-flash".ljust(32, b"\x00")
    assert len(wire.hello("x" * 32)) == wire.HELLO == 48
    for bad in ("", "x" * 33):
        with pytest.raises(WireError):
            wire.hello(bad)  # no default model, and the name fits its field


def test_header_parses_and_refuses():
    h = wire.parse_header(b"rock" + struct.pack("<IQI", 5, IDENTITY, 2047))
    assert h == Header(IDENTITY, 2047)
    with pytest.raises(WireError):
        wire.parse_header(b"rocj" + struct.pack("<IQI", 5, 0, 1))
    with pytest.raises(WireError):
        wire.parse_header(b"rock" + struct.pack("<IQI", 4, 0, 1))
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
    assert s.generate() == b"\xff\xff\xff\xff\x00\x00\x00\x00"
    assert s.generate(7) == b"\xff\xff\xff\xff\x07\x00\x00\x00"
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


def connect(node, identity=0, model="tiny"):
    return wire.Client.connect(node.addr, BOS, EOS, VOCAB, identity=identity, model=model)


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
    with pytest.raises(Refused):
        await connect(node, model="other")  # a model the node does not serve
    assert node.hellos[-1] == "other"
    async with await connect(node, identity=IDENTITY) as c:
        assert c.header.identity == IDENTITY


def test_session_takes_a_prefix_and_several_end_ids():
    s = Session(Header(IDENTITY, 4), (BOS, 5), {EOS, 9}, VOCAB)
    assert (s.bos, s.prefix, s.ends) == (BOS, (BOS, 5), frozenset({EOS, 9}))
    s.start()
    s.feed([1])
    s.generate()
    assert s.receive(struct.pack("<2I", 3, 9)) == [3, 9] and s.ended  # any end id ends the turn


@pytest.mark.anyio
async def test_client_sends_the_whole_prefix_once(node):
    # a two-id prefix: the second id travels like any id after BOS (this fake node feeds it;
    # the real node checks and swallows it, outside the budget, which is why the mirror skips it)
    async with await wire.Client.connect(node.addr, (BOS, 5), {EOS}, VOCAB, model="tiny") as c:
        await c.prompt([BOS, 5, 1, 2])  # what a template that writes the prefix yields
        assert c.session.remaining == 7 - 2
        await collect(c.generate())
        await c.prompt([1])  # raw ids: the prefix goes first
        await collect(c.generate())
        with pytest.raises(WireError):
            await c.prompt([BOS, 1])  # half a prefix
    await asyncio.sleep(0.05)
    assert [t[:3] for t in node.transcripts[:2]] == [[5, 1, 2], [5, 1, 10]]


@pytest.mark.anyio
async def test_close_is_a_reset_that_drops_the_unsent_prompt(node):
    """2026-10-24: a cancelled long prompt kept the node busy, because an ordinary close's FIN
    waits behind the unsent prompt. The client's close resets instead: the node stops reading."""
    node.max_ctx, node.read_delay, node.rcvbuf = 1_000_000, 0.001, 32_768  # slow, with a full buffer, as in production
    c = await connect(node)
    big = [1] * 400_000  # 1.6 MB of prompt: far more than the kernel buffers hold
    task = asyncio.ensure_future(c.prompt(big))
    await asyncio.sleep(0.3)
    c.close()  # the cancel
    task.cancel()
    with contextlib.suppress(BaseException):
        await task
    await asyncio.sleep(1.0)
    read_then = node.words
    await asyncio.sleep(1.0)
    assert node.words == read_then, "the node is still reading the cancelled prompt"
    assert node.words < 50_000


@pytest.mark.anyio
async def test_a_node_connection_notices_a_dead_machine(node):
    """2026-10-25: a node panic reset its machine and closed nothing; the API waited 250 s and on."""
    import socket as _socket

    async with await connect(node) as c:
        s = c.writer.get_extra_info("socket")
        assert s.getsockopt(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE) == 1
        idle = s.getsockopt(_socket.IPPROTO_TCP, _socket.TCP_KEEPIDLE)
        intvl, cnt = s.getsockopt(_socket.IPPROTO_TCP, _socket.TCP_KEEPINTVL), s.getsockopt(_socket.IPPROTO_TCP, _socket.TCP_KEEPCNT)
        assert idle + intvl * cnt <= 15  # a dead machine is noticed within seconds
        assert s.getsockopt(_socket.IPPROTO_TCP, _socket.TCP_USER_TIMEOUT) == wire.USER_TIMEOUT_MS


@pytest.mark.anyio
async def test_a_machine_that_is_down_fails_the_connect(monkeypatch):
    monkeypatch.setattr(wire, "CONNECT_TIMEOUT", 0.5)
    import time as _time

    t0 = _time.time()
    with pytest.raises(OSError):
        await wire.Client.connect(("10.255.255.1", 7332), BOS, EOS, VOCAB, model="tiny")  # nothing answers there
    assert _time.time() - t0 < 2


@pytest.mark.anyio
async def test_a_node_that_vanishes_mid_reply_ends_the_stream(engine, node):
    """The node's machine resets mid-reply: the next read fails, the client gets an error, the node's
    place in line is free again."""
    from rockem import core

    node.draws, node.delay, node.close_after = [2] * 50, 0.01, 3
    run = await engine.start(core.Request(prompt="hello"))
    events = []
    with pytest.raises((WireError, OSError)):  # a reset reads as OSError, a clean close as WireError
        async for ev in run.events():
            events.append(ev)
    assert engine.scheduler.free == 1
