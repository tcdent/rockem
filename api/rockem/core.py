"""One path from a canonical request to a stream of events. Adapters map their request
shape onto `Request` and their response shape off `Delta` and `Done`; nothing in here
knows an HTTP surface. A non-streaming response is `fold` over the same stream."""

from __future__ import annotations

import enum
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .edge import Edge, Stops
from .scheduler import Scheduler
from .wire import ContextFull, Finish


@dataclass
class Request:
    """Either `messages` for the chat template or `prompt` as raw text, never both."""

    messages: list[dict[str, Any]] | None = None
    prompt: str | None = None
    max_tokens: int | None = None  # None: whatever the node's budget allows
    stop: list[str] = field(default_factory=list)


class Reason(enum.Enum):
    EOS = "eos"  # the model ended the turn
    STOP_SEQUENCE = "stop_sequence"  # a stop sequence appeared in the text
    MAX_TOKENS = "max_tokens"  # the request's limit
    CONTEXT_FULL = "context_full"  # the node's budget


@dataclass
class Delta:
    text: str
    ids: list[int]


@dataclass
class Done:
    reason: Reason
    stop_sequence: str | None
    prompt_tokens: int  # ids fed after BOS
    completion_tokens: int  # ids received, EOS included: the positions the generation took
    positions: int  # the node's budget


Event = Delta | Done


class Engine:
    def __init__(self, edge: Edge, scheduler: Scheduler):
        self.edge, self.scheduler = edge, scheduler

    def ids(self, req: Request) -> list[int]:
        if req.messages is not None:
            return self.edge.encode_chat(req.messages)
        return self.edge.encode(req.prompt or "")

    async def start(self, req: Request) -> Run:
        """Encode, take a node, send the prompt. `ContextFull` before anything is sent
        and the node is free again; connection errors likewise."""
        ids = self.ids(req)
        cm = self.scheduler.node()
        client = await cm.__aenter__()
        try:
            await client.prompt(ids)
        except BaseException as e:
            await cm.__aexit__(type(e), e, e.__traceback__)
            raise
        return Run(self, req, ids, cm, client)


class Run:
    """A request on a node. Iterating `events` streams it; closing the iterator early,
    which is what a client disconnect does, aborts it and frees the node."""

    def __init__(self, engine: Engine, req: Request, ids: list[int], cm, client):
        self.engine, self.req, self.client, self._cm = engine, req, client, cm
        self.prompt_tokens = len(ids) - (1 if ids and ids[0] == client.session.bos else 0)

    async def events(self) -> AsyncIterator[Event]:
        req, client = self.req, self.client
        decoder, stops = self.engine.edge.decoder(), Stops(req.stop)
        limit = req.max_tokens  # None: the node's budget decides, and its close says so
        n, pending = 0, []
        reason: Reason | None = None
        try:
            async for id in client.generate():
                n += 1
                if id == client.session.eos:
                    reason = Reason.EOS
                    break
                pending.append(id)
                text = stops.feed(decoder.feed(id))
                if stops.matched is not None:
                    reason = Reason.STOP_SEQUENCE
                    if text:
                        yield Delta(text, pending)
                        pending = []
                    break
                if text:
                    yield Delta(text, pending)
                    pending = []
                if limit is not None and n >= limit:
                    reason = Reason.MAX_TOKENS
                    break
            if reason is None:
                reason = Reason.EOS if client.finish is Finish.STOP else Reason.CONTEXT_FULL
            tail = stops.feed(decoder.flush()) + stops.flush()
            if tail or pending:
                yield Delta(tail, pending)
            yield Done(reason, stops.matched, self.prompt_tokens, n, client.positions)
        finally:
            await self._cm.__aexit__(None, None, None)

    async def fold(self) -> tuple[str, list[int], Done]:
        text, ids, done = [], [], None
        async for ev in self.events():
            if isinstance(ev, Delta):
                text.append(ev.text)
                ids.extend(ev.ids)
            else:
                done = ev
        assert done is not None
        return "".join(text), ids, done


__all__ = ["ContextFull", "Delta", "Done", "Engine", "Event", "Reason", "Request", "Run"]
