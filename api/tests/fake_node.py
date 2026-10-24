"""A node with no model: the serve loop of node/src/main.rs over a scripted draw, so the
client can be tested against the wire's exact behavior, the budget's edge included."""

from __future__ import annotations

import socket
import select
import struct
import time
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

    def __init__(self, identity: int, max_ctx: int, draws: list[int], bos: int = BOS, eos: int = EOS, vocab: int = VOCAB, models: set[str] = frozenset({"tiny"})):
        self.identity, self.max_ctx, self.draws = identity, max_ctx, draws
        self.models = set(models)  # the names this node serves; the hello must name one
        self.hellos: list[str] = []
        self.bos, self.eos, self.vocab = bos, eos, vocab
        # models with an audio encoder: name → {bos, eos, vocab, placeholder, window_bytes, rows}
        self.audio: dict[str, dict] = {}
        self.windows: list[bytes] = []  # every AUDIO window received
        self.delay = 0.0  # seconds before each generated id
        self.read_delay = 0.0  # seconds before reading each word: a node busy computing its prompt
        self.words = 0  # words read after the header, every connection
        self.rcvbuf = 0  # a small receive buffer (bytes), so a long prompt waits on the sender's side
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
                if self.rcvbuf:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf)
                self.sessions += 1
                h = read_exact(s, wire.HELLO)
                if len(h) != wire.HELLO or h[:4] != wire.MAGIC:
                    continue
                version, want = struct.unpack("<IQ", h[4:16])
                name = h[16:].rstrip(b"\0").decode(errors="replace")
                self.hellos.append(name)
                if version != wire.VERSION or want not in (0, self.identity) or (name not in self.models and name not in self.audio):
                    continue
                a = self.audio.get(name)
                bos, eos, vocab = (a["bos"], a["eos"], a["vocab"]) if a else (self.bos, self.eos, self.vocab)
                rows = 0
                s.sendall(wire.MAGIC + struct.pack("<IQI", wire.VERSION, self.identity, self.max_ctx - FIRST_POS))
                pos, started, drawn, ids = 0, False, 0, []
                while True:
                    if self.read_delay:
                        time.sleep(self.read_delay)
                        # the node asks the socket before every step (poll POLLRDHUP): a hang-up or an
                        # error ends the sequence even with the prompt's words still queued
                        p = select.poll()
                        p.register(s, select.POLLRDHUP | select.POLLERR | select.POLLHUP)
                        if any(ev & (select.POLLERR | select.POLLHUP) for _, ev in p.poll(0)):
                            break
                    w = read_exact(s, 4)
                    self.words += len(w) == 4
                    if len(w) != 4:
                        break
                    (word,) = struct.unpack("<I", w)
                    if word == bos:
                        pos, started, drawn, ids, rows = FIRST_POS, True, 0, [], 0
                        self.transcripts.append(ids)
                        ok = True
                    elif word == wire.AUDIO:
                        # the spec's AUDIO: one window of the model's size, rows queued for placeholders
                        if not (started and a):
                            break
                        win = read_exact(s, a["window_bytes"])
                        if len(win) != a["window_bytes"]:
                            break
                        self.windows.append(win)
                        rows += a["rows"]
                        ok = True
                    elif word == wire.GENERATE:
                        lw = read_exact(s, 4)
                        if len(lw) != 4 or rows:  # rows not taken by placeholders: refused
                            break
                        (limit,) = struct.unpack("<I", lw)
                        ok = started and self._generate(s, pos, drawn, ids, limit or None, eos)
                        if ok:
                            pos, drawn = self._pos, self._drawn
                    elif word < vocab:
                        if a and word == a["placeholder"]:
                            if not rows:  # a placeholder with no row: refused
                                break
                            rows -= 1
                        ok = started and pos < self.max_ctx
                        if ok:
                            ids.append(word)
                            pos += 1
                    else:
                        ok = False
                    if not ok:
                        break

    def _generate(self, s, pos, drawn, ids, limit=None, eos=None) -> bool:
        eos = self.eos if eos is None else eos
        # Session::generate: sample, emit, feed, until EOS or `limit` ids; false when the context filled
        n = 0
        while limit is None or n < limit:
            n += 1
            id = self.draws[drawn] if drawn < len(self.draws) else eos
            drawn += 1
            if self.delay:
                time.sleep(self.delay)
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
            if id == eos:
                self._pos, self._drawn = pos, drawn
                return True
        self._pos, self._drawn = pos, drawn
        return True

    def close(self):
        self.srv.close()
