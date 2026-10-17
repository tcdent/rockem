"""A node with no model: the serve loop of node/src/main.rs over a scripted draw, so the
client can be tested against the wire's exact behavior, the budget's edge included."""

from __future__ import annotations

import socket
import struct
import threading

from rockem import wire


def read_exact(sock, n):
    """`n` bytes, or fewer at end of stream; a reset peer is an end of stream too, as
    `read_exact(..).is_ok()` makes it on the node."""
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
    return buf

BOS, EOS, VOCAB = 100000, 100001, 102400
FIRST_POS = 1


class FakeNode:
    """`draws` is the id the model 'samples' at each generated position of a sequence,
    restarting at BOS; after the list runs out it draws EOS."""

    def __init__(self, identity: int, max_ctx: int, draws: list[int], bos: int = BOS, eos: int = EOS, vocab: int = VOCAB):
        self.identity, self.max_ctx, self.draws = identity, max_ctx, draws
        self.bos, self.eos, self.vocab = bos, eos, vocab
        self.close_after: int | None = None  # drop the connection after this many generated ids
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(8)
        self.addr = self.srv.getsockname()
        self.transcripts: list[list[int]] = []  # every sequence this node saw, ids after BOS
        self.sessions = 0
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def _serve(self):
        while True:
            try:
                s, _ = self.srv.accept()
            except OSError:
                return
            with s:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sessions += 1
                h = read_exact(s, wire.HELLO)
                if len(h) != wire.HELLO or h[:4] != wire.MAGIC:
                    continue
                version, want = struct.unpack("<IQ", h[4:])
                if version != wire.VERSION or want not in (0, self.identity):
                    continue
                s.sendall(wire.MAGIC + struct.pack("<IQI", wire.VERSION, self.identity, self.max_ctx - FIRST_POS))
                pos, started, drawn, ids = 0, False, 0, []
                while True:
                    w = read_exact(s, 4)
                    if len(w) != 4:
                        break
                    (word,) = struct.unpack("<I", w)
                    if word == self.bos:
                        pos, started, drawn, ids = FIRST_POS, True, 0, []
                        self.transcripts.append(ids)
                        ok = True
                    elif word == wire.GENERATE:
                        lw = read_exact(s, 4)
                        if len(lw) != 4:
                            break
                        (limit,) = struct.unpack("<I", lw)
                        ok = started and self._generate(s, pos, drawn, ids, limit or None)
                        if ok:
                            pos, drawn = self._pos, self._drawn
                    elif word < self.vocab:
                        ok = started and pos < self.max_ctx
                        if ok:
                            ids.append(word)
                            pos += 1
                    else:
                        ok = False
                    if not ok:
                        break

    def _generate(self, s, pos, drawn, ids, limit=None) -> bool:
        # Session::generate: sample, emit, feed, until EOS or `limit` ids; false when the context filled
        n = 0
        while limit is None or n < limit:
            n += 1
            id = self.draws[drawn] if drawn < len(self.draws) else self.eos
            drawn += 1
            try:
                s.sendall(struct.pack("<I", id))
            except OSError:
                return False
            if self.close_after is not None and drawn >= self.close_after:
                return False
            if pos >= self.max_ctx:
                return False
            ids.append(id)
            pos += 1
            if id == self.eos:
                self._pos, self._drawn = pos, drawn
                return True
        self._pos, self._drawn = pos, drawn
        return True

    def close(self):
        self.srv.close()
