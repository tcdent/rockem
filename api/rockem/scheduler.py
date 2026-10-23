"""The nodes' occupancy. A node serves one connection at a time and parks a second
connector silently, so the edge never connects to a node it is using: each node is a
token in a queue, taken for the length of one request and put back after. Waiters are
served in arrival order. One process holds the one table, and every model shares it: a
node serves one connection at a time across all its models, so a chat and a transcription
wait in the same line (docs/queue/094)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from .bundle import Model
from .wire import AudioSpec, Client

Addr = tuple[str, int]


class Scheduler:
    def __init__(self, nodes: list[Addr], identity: int = 0):
        self.nodes = list(nodes)
        self.identity = identity
        self._free: asyncio.Queue[Addr] = asyncio.Queue()
        for n in self.nodes:
            self._free.put_nowait(n)

    @property
    def free(self) -> int:
        return self._free.qsize()

    @property
    def waiting(self) -> int:
        return len(self._free._getters)  # noqa: SLF001 — asyncio has no public count of waiters

    @asynccontextmanager
    async def node(self, model: Model) -> AsyncIterator[Client]:
        """A client on the next free node, speaking `model`; the node is free again on exit,
        and the connection is closed, which is the abort if a generation is running."""
        audio = AudioSpec.of(model.audio) if model.audio else None
        addr = await self._free.get()
        try:
            client = await Client.connect(addr, model.prefix, model.eos_tokens, model.vocab_size, self.identity, model=model.name, audio=audio)
        except BaseException:
            self._free.put_nowait(addr)
            raise
        try:
            yield client
        finally:
            client.close()
            self._free.put_nowait(addr)
