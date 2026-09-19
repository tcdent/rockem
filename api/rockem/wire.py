"""The wire, from the client's side (docs/wire.md).

`Session` is the protocol without a socket: it produces the bytes to send,
parses the bytes received into ids, and mirrors the node's position counter,
so a prompt that cannot fit is refused before it is sent and an end of
stream is read for what it is: a full context, a finished generation, or a
node that went away. `Client` is the asyncio socket around it; anything with a
byte stream drives a `Session` the same way.
"""

from __future__ import annotations

import asyncio
import enum
import socket
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass

VERSION = 3
GENERATE = 0xFFFFFFFF
SCORE = 0xFFFFFFFE
MAGIC = b"rock"
HELLO = 16  # client → node: magic, version, identity
HEADER = 20  # node → client: the same, then positions
WORD = 4


class WireError(Exception):
    """The peer did not speak the wire."""


class Refused(WireError):
    """The node closed the connection before writing its header: a bad magic, another
    version, or an identity that is not its own."""


class ContextFull(WireError):
    """The ids would not fit in the positions the node has."""


class Disconnected(WireError):
    """The stream ended where the protocol did not: the node stopped or the socket broke."""


@dataclass(frozen=True)
class Header:
    """The node's header: its identity and how many ids a sequence may hold after BOS."""

    identity: int
    positions: int


def hello(identity: int = 0) -> bytes:
    """The client's header. Identity 0 accepts any node."""
    return MAGIC + struct.pack("<IQ", VERSION, identity)


def parse_header(data: bytes) -> Header:
    if len(data) != HEADER:
        raise Refused(f"{len(data)} header bytes, expected {HEADER}")
    magic, version, identity, positions = struct.unpack("<4sIQI", data)
    if magic != MAGIC:
        raise WireError(f"magic {magic!r}")
    if version != VERSION:
        raise WireError(f"version {version}, this client speaks {VERSION}")
    return Header(identity, positions)


class Finish(enum.Enum):
    STOP = "stop"  # the node wrote EOS
    LENGTH = "length"  # the context filled: the node closed after the last id it could hold


class Session:
    """One sequence on one connection, after the headers. Send what the `bytes` methods
    return; pass what arrives to `receive`; call `eof` when the stream ends.

    Positions: the node holds `positions` ids after BOS, prompt and generated together.
    Each id sent or received takes one. When a draw lands on a full context the node
    writes the id and then closes, so exactly one id can arrive past the budget; it is
    real output and never entered the cache."""

    def __init__(self, header: Header, bos: int, eos: int, vocab_size: int):
        self.positions = header.positions
        self.bos, self.eos, self.vocab_size = bos, eos, vocab_size
        self.remaining = 0
        self.started = False
        self.ended = False  # the last word received was EOS
        self.scoring = False  # SCORE was sent: every id fed is answered with its ln p, an f32 in the word
        self.scores_due = 0
        self._buf = b""

    # --- out ---

    def start(self) -> bytes:
        """BOS: a new sequence; the budget is whole again."""
        self.remaining = self.positions
        self.started, self.ended = True, False
        self.scoring, self.scores_due = False, 0
        self._buf = b""
        return struct.pack("<I", self.bos)

    def feed(self, ids: list[int]) -> bytes:
        """Prompt ids, or ids appended after EOS to continue. Refused whole if they do not fit."""
        if not self.started:
            raise WireError("feed before BOS")
        if len(ids) > self.remaining:
            raise ContextFull(f"{len(ids)} ids, {self.remaining} positions left")
        for id in ids:
            if not 0 <= id < self.vocab_size or id == self.bos:
                raise WireError(f"id {id} is not a token")
        self.remaining -= len(ids)
        self.ended = False
        if self.scoring:
            self.scores_due += len(ids)
        return struct.pack(f"<{len(ids)}I", *ids)

    def score(self) -> bytes:
        """SCORE: from here every id fed is answered with ln p(id | everything before), the
        model's own distribution, before it is fed. Read them with `receive_scores`."""
        if not self.started:
            raise WireError("SCORE before BOS")
        self.scoring = True
        return struct.pack("<I", SCORE)

    def generate(self) -> bytes:
        if not self.started:
            raise WireError("GENERATE before BOS")
        self.ended = False
        return struct.pack("<I", GENERATE)

    # --- in ---

    def receive(self, data: bytes) -> list[int]:
        """The ids in `data`, whole words only; a partial word waits for the next call.
        EOS is returned like any id and marks the generation ended."""
        self._buf += data
        n = len(self._buf) // WORD
        ids = list(struct.unpack_from(f"<{n}I", self._buf))
        self._buf = self._buf[n * WORD :]
        for id in ids:
            if id >= self.vocab_size:
                raise WireError(f"word {id:#x} from the node is not a token")
            if self.remaining < 0:
                raise WireError("an id after the one past the budget")
            self.remaining -= 1
            self.ended = id == self.eos
        return ids

    def receive_scores(self, data: bytes) -> list[float]:
        """The scores in `data`, whole words only, one per id fed since SCORE."""
        self._buf += data
        n = min(len(self._buf) // WORD, self.scores_due)
        vals = list(struct.unpack_from(f"<{n}f", self._buf))
        self._buf = self._buf[n * WORD :]
        self.scores_due -= n
        return vals

    def eof(self) -> Finish:
        """The stream ended. A full context is the only end the protocol allows before EOS."""
        if self._buf:
            raise Disconnected(f"{len(self._buf)} bytes of a partial word")
        if self.ended:
            raise Disconnected("the node closed after EOS")
        if self.remaining <= 0:
            return Finish.LENGTH
        raise Disconnected(f"the node closed with {self.remaining} positions left")


class Client:
    """One connection, one session, asyncio. `connect` exchanges the headers; `prompt`,
    `generate` and `close` are the three moves. Closing mid-generation is the abort."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, header: Header, bos: int, eos: int, vocab_size: int):
        self.reader, self.writer = reader, writer
        self.header = header
        self.session = Session(header, bos, eos, vocab_size)
        self.finish: Finish | None = None

    @classmethod
    async def connect(cls, addr: tuple[str, int], bos: int, eos: int, vocab_size: int, identity: int = 0) -> Client:
        reader, writer = await asyncio.open_connection(*addr)
        sock = writer.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            writer.write(hello(identity))
            await writer.drain()
            data = await reader.read(HEADER)
            while 0 < len(data) < HEADER:
                more = await reader.read(HEADER - len(data))
                if not more:
                    break
                data += more
            if len(data) == 0:
                raise Refused(f"{addr[0]}:{addr[1]} closed without a header")
            return cls(reader, writer, parse_header(data), bos, eos, vocab_size)
        except BaseException:
            writer.close()
            raise

    @property
    def positions(self) -> int:
        return self.header.positions

    async def prompt(self, ids: list[int]) -> None:
        """A new sequence: BOS then the ids. A leading BOS in `ids`, which is what encoding
        a rendered chat template yields, is that BOS; otherwise one is sent first. Nothing
        is sent if the ids do not fit."""
        s = self.session
        if ids and ids[0] == s.bos:
            ids = ids[1:]
        bos = s.start()
        self.writer.write(bos + s.feed(ids))
        await self.writer.drain()

    async def feed(self, ids: list[int]) -> None:
        """More ids on the current sequence, after EOS."""
        self.writer.write(self.session.feed(ids))
        await self.writer.drain()

    async def score(self, ids: list[int]) -> list[float]:
        """ln p of each id given everything before it, feeding them; SCORE is sent
        the first time. The sequence continues afterwards (more ids, GENERATE)."""
        s = self.session
        head = b"" if s.scoring else s.score()
        self.writer.write(head + s.feed(ids))
        await self.writer.drain()
        out: list[float] = []
        while len(out) < len(ids):
            data = await self.reader.read(WORD * (len(ids) - len(out)))
            if not data:
                raise Disconnected(f"the node closed with {len(ids) - len(out)} scores due")
            out.extend(s.receive_scores(data))
        return out

    async def generate(self) -> AsyncIterator[int]:
        """GENERATE, then every id as it arrives, EOS included, until EOS or the stream ends.
        `self.finish` says which; `Disconnected` if the end was neither."""
        s = self.session
        self.finish = None
        self.writer.write(s.generate())
        await self.writer.drain()
        while True:
            data = await self.reader.read(65536)
            if not data:
                self.finish = s.eof()
                return
            for id in s.receive(data):
                yield id
                if id == s.eos:
                    self.finish = Finish.STOP
                    return

    def close(self) -> None:
        self.writer.close()

    async def __aenter__(self) -> Client:
        return self

    async def __aexit__(self, *_) -> None:
        self.close()
