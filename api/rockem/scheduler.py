"""The nodes' occupancy. A node serves one connection at a time and parks a second
connector silently, so the edge never connects to a node it is using: each node is a
token in a queue, taken for the length of one request and put back after. Waiters are
served in arrival order. One process holds the one table."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from .wire import Client

Addr = tuple[str, int]


class Scheduler:
    def __init__(self, nodes: list[Addr], bos: int, eos: int, vocab_size: int, identity: int = 0):
        self.nodes = list(nodes)
        self.bos, self.eos, self.vocab_size, self.identity = bos, eos, vocab_size, identity
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
    async def node(self) -> AsyncIterator[Client]:
        """A connected client on the next free node; the node is free again on exit,
        and the connection is closed, which is the abort if a generation is running."""
        addr = await self._free.get()
        try:
            client = await Client.connect(addr, self.bos, self.eos, self.vocab_size, self.identity)
        except BaseException:
            self._free.put_nowait(addr)
            raise
        try:
            yield client
        finally:
            client.close()
            self._free.put_nowait(addr)
