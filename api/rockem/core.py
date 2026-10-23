"""One path from a canonical request to a stream of events. Adapters map their request
shape onto `Request` and their response shape off the events; nothing in here knows an
HTTP surface. A non-streaming response is `fold` over the same stream."""

from __future__ import annotations

import enum
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import logging

from .calls import Call, Parser, Text, Thinking
from .edge import Edge, Stops
from .scheduler import Scheduler
from .wire import ContextFull, Finish


@dataclass
class Request:
    """Either `messages` for the chat template, `prompt` as raw text, or `ids` already made
    (a transcription's prompt, with its audio `windows`). Messages and tools are
    template-shaped (see `Edge.render`)."""

    messages: list[dict[str, Any]] | None = None
    prompt: str | None = None
    tools: list[dict[str, Any]] | None = None
    max_tokens: int | None = None  # None: whatever the node's budget allows
    stop: list[str] = field(default_factory=list)
    reasoning_effort: str = "low"  # the template's variable; low unless asked (tcdent 2026-10-21)
    ids: list[int] | None = None
    windows: list[bytes] = field(default_factory=list)  # audio encoder windows, sent before the placeholders


class Reason(enum.Enum):
    END_TURN = "end_turn"  # the model ended its turn: EOS, or the text opened the next turn
    TOOL_USE = "tool_use"  # the turn ended having made tool calls
    STOP_SEQUENCE = "stop_sequence"  # one of the request's stop sequences appeared
    MAX_TOKENS = "max_tokens"  # the request's limit
    CONTEXT_FULL = "context_full"  # the node's budget
    NO_SPEECH = "no_speech"  # transcription: the clip has no voice in it; the node was not asked


@dataclass
class Delta:
    text: str
    ids: list[int]


@dataclass
class Started:
    """A node took the request and has its prompt; what follows is the node's work."""

    prompt_tokens: int


@dataclass
class Reasoning:
    """The model's reasoning before its reply, between its think markers (markers dropped)."""

    text: str
    ids: list[int]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    ids: list[int]  # the ids the call was generated as, markers included


@dataclass
class Done:
    reason: Reason
    stop_sequence: str | None
    prompt_tokens: int  # ids fed after BOS
    completion_tokens: int  # ids received, EOS included: the positions the generation took
    positions: int  # the node's budget


Event = Started | Reasoning | Delta | ToolCall | Done
log = logging.getLogger("rockem")


class Engine:
    def __init__(self, edge: Edge, scheduler: Scheduler):
        self.edge, self.scheduler = edge, scheduler
        # Replies this process generated, by their text: the ids they were generated as. A chat's
        # next turn carries the reply back as text; rendered and re-tokenized it differs from the
        # generated ids (the template strips and re-wraps it), so the node's cache would stop
        # matching at the reply and recompute it. Rendering such a turn from the remembered ids
        # keeps the match to the reply's end, where the node kept the state (2026-10-16).
        self.replies: OrderedDict[tuple[str, str], list[int]] = OrderedDict()  # (reasoning, text) → ids

    def remember(self, reasoning: str, text: str, ids: list[int]) -> None:
        if not (text or reasoning) or not ids:
            return
        key = (reasoning, text)
        self.replies[key] = list(ids)
        self.replies.move_to_end(key)
        while len(self.replies) > 256:
            self.replies.popitem(last=False)

    def ids(self, req: Request) -> list[int]:
        if req.ids is not None:
            return req.ids
        if req.messages is None:
            return self.edge.encode(req.prompt or "")
        msgs, tools = req.messages, req.tools
        e = req.reasoning_effort
        full = self.edge.render(msgs, tools, reasoning_effort=e)
        out: list[int] = []
        done = ""  # the rendered text encoded so far
        for i, m in enumerate(msgs):
            content = m.get("content") if m.get("role") == "assistant" else None
            ids = self.replies.get((m.get("reasoning_content") or "", content)) if isinstance(content, str) else None
            if ids is None:
                continue
            before = self.edge.render(msgs[:i], tools, add_generation_prompt=False, reasoning_effort=e)
            opened = self.edge.render(msgs[:i], tools, add_generation_prompt=True, reasoning_effort=e)
            after = self.edge.render(msgs[: i + 1], tools, add_generation_prompt=False, reasoning_effort=e)
            if not (full.startswith(after) and after.startswith(before) and opened.startswith(before) and len(before) >= len(done)):
                continue
            # the text up to this turn's opening (the generation prompt it was generated after), then the ids as generated
            out.extend(self.edge.encode(before[len(done):] + opened[len(before):]))
            out.extend(ids)
            done = after
            log.info("turn %d rendered from its %d generated ids", i, len(ids))
        out.extend(self.edge.encode(full[len(done):]))
        return out

    async def start(self, req: Request) -> Run:
        """Encode, take a node, send the prompt. `ContextFull` before anything is sent
        and the node is free again; connection errors likewise."""
        ids = self.ids(req)
        fmt = self.edge.format
        # the generation prompt opened the model's reasoning: the reply starts inside it
        inside = bool(req.messages is not None and fmt and fmt.think and self.edge.render(req.messages, req.tools, reasoning_effort=req.reasoning_effort).endswith(fmt.think[0]))
        cm = self.scheduler.node(self.edge.model)
        client = await cm.__aenter__()
        try:
            await client.prompt(ids, req.windows)
        except BaseException as e:
            await cm.__aexit__(type(e), e, e.__traceback__)
            raise
        return Run(self, req, ids, cm, client, inside)


class Run:
    """A request on a node. Iterating `events` streams it; closing the iterator early,
    which is what a client disconnect does, aborts it and frees the node.

    Text goes decoder → call parser → stop matcher: a call's arguments are never matched
    against stop sequences, and text before a call is emitted before it. For a chat
    request the model's turn markers are stops too, since a model that does not sample
    EOS ends its turn by opening the next one. And a turn that has made calls ends at the
    first text after them that is not another call: in every tool-call format a call is
    followed by another call or by the end of the turn, so that text is a model that
    skipped its end writing the next turn (V2-Lite writes `Tool:` straight after a call)."""

    def __init__(self, engine: Engine, req: Request, ids: list[int], cm, client, inside: bool = False):
        self.engine, self.req, self.client, self._cm = engine, req, client, cm
        self.inside = inside
        prefix = client.session.prefix
        self.prompt_tokens = len(ids) - (len(prefix) if tuple(ids[: len(prefix)]) == prefix else 0)

    async def events(self) -> AsyncIterator[Event]:
        req, client, fmt = self.req, self.client, self.engine.edge.format
        chat = req.messages is not None
        turn_stops = list(fmt.turn_stops) if (chat and fmt) else []
        decoder, stops = self.engine.edge.decoder(), Stops(req.stop + turn_stops)
        parser = Parser(fmt) if (req.tools and fmt) else None
        limit = req.max_tokens  # None: the node's budget decides, and its close says so
        n, pending, calls = 0, [], 0
        reason: Reason | None = None
        past_calls = False  # text after the turn's calls ended it
        memo_think, memo_text, memo_ids = [], [], []  # the reply as delivered, and the ids it was generated as
        thinker = Thinking(fmt.think, self.inside) if (chat and fmt and fmt.think) else None

        def out(ev: Event) -> Event:
            if isinstance(ev, Reasoning):
                memo_think.append(ev.text)
                memo_ids.extend(ev.ids)
            elif isinstance(ev, Delta):
                memo_text.append(ev.text)
                memo_ids.extend(ev.ids)
            elif isinstance(ev, ToolCall):
                memo_ids.extend(ev.ids)
            return ev

        def route(text: str) -> list[Event]:
            nonlocal pending, calls, past_calls
            out: list[Event] = []
            for item in parser.feed(text) if parser else [Text(text)]:
                if isinstance(item, Text) and calls:
                    if item.text.strip():
                        past_calls = True
                        break
                    continue  # whitespace between calls
                if isinstance(item, Text):
                    t = stops.feed(item.text)
                    if t:
                        out.append(Delta(t, pending))
                        pending = []
                    if stops.matched is not None:
                        break
                else:
                    t = stops.flush()  # held text was not a stop: a call follows it
                    if t:
                        out.append(Delta(t, []))
                    out.append(ToolCall(f"call_{uuid.uuid4().hex[:12]}", item.name, item.arguments, pending))
                    pending = []
                    calls += 1
            return out

        def split(text: str, pieces=None) -> list[Event]:
            nonlocal pending
            evs: list[Event] = []
            for reasoning, piece in (thinker.feed(text) if pieces is None else pieces) if thinker else [(False, text)]:
                if reasoning:
                    evs.append(Reasoning(piece, pending))
                    pending = []
                else:
                    evs.extend(route(piece))
            return evs

        yield Started(self.prompt_tokens)
        try:
            async for id in client.generate(limit or 0):
                n += 1
                if id in client.session.ends:
                    reason = Reason.END_TURN
                    break
                pending.append(id)
                for ev in split(decoder.feed(id)):
                    yield out(ev)
                if stops.matched is not None:
                    reason = Reason.END_TURN if stops.matched in turn_stops else Reason.STOP_SEQUENCE
                    break
                if past_calls:
                    reason = Reason.END_TURN
                    break
                if limit is not None and n >= limit:
                    reason = Reason.MAX_TOKENS
                    break
            if reason is None:
                reason = Reason.END_TURN if client.finish is Finish.STOP else Reason.CONTEXT_FULL
            if stops.matched is None and not past_calls:
                for ev in split(decoder.flush()):
                    yield out(ev)
                if thinker:
                    for ev in split("", thinker.flush()):
                        yield out(ev)
                held = parser.flush() if parser else ""
                tail = stops.feed(held) if not (calls and held.strip()) else ""
                tail += stops.flush()
                if tail or pending:
                    yield out(Delta(tail, pending))
            elif pending:
                yield out(Delta("", pending))
            if reason is Reason.END_TURN and calls:
                reason = Reason.TOOL_USE
            if chat:
                self.engine.remember("".join(memo_think), "".join(memo_text), memo_ids)
            yield Done(reason, stops.matched if reason is Reason.STOP_SEQUENCE else None, self.prompt_tokens, n, client.positions)
        finally:
            await self._cm.__aexit__(None, None, None)

    async def fold(self) -> tuple[str, list[int], list[ToolCall], Done]:
        """The reply's text (reasoning left out), every id, the calls, and the end."""
        text, ids, calls, done = [], [], [], None
        async for ev in self.events():
            if isinstance(ev, Reasoning):
                ids.extend(ev.ids)
            elif isinstance(ev, Delta):
                text.append(ev.text)
                ids.extend(ev.ids)
            elif isinstance(ev, ToolCall):
                calls.append(ev)
                ids.extend(ev.ids)
            elif isinstance(ev, Done):
                done = ev
        assert done is not None
        return "".join(text), ids, calls, done


__all__ = ["ContextFull", "Delta", "Done", "Engine", "Event", "Reason", "Reasoning", "Request", "Run", "Started", "ToolCall"]
