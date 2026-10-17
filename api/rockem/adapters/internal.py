"""rockem's own surface (docs/api.md): the canonical request and its events, as they are.
Always a stream: a chunked HTTP/1.1 response of `data: {json}` lines, one event each,
typed by `type`. Messages, tools and calls are flat here ({id, name, arguments}); the
template-shaped nesting is made in `to_request`, so clients never write a model's or a
template's conventions. The server holds no state: every request carries the whole
conversation, and the node's prefix store makes the repeated part cost nothing."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .. import core
from ..wire import WireError


class CallIn(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[CallIn] | None = None  # assistant: the calls it made
    tool_call_id: str | None = None  # tool: the call this is the result of

    @model_validator(mode="after")
    def shape(self):
        if self.tool_calls and self.role != "assistant":
            raise ValueError("tool_calls belong to assistant messages")
        if (self.role == "tool") != (self.tool_call_id is not None):
            raise ValueError("a tool message, and only a tool message, carries tool_call_id")
        return self


class Tool(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class Request(BaseModel):
    messages: list[Message] | None = None
    prompt: str | None = None
    tools: list[Tool] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    stop: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def one_of(self):
        if (self.messages is None) == (self.prompt is None):
            raise ValueError("exactly one of messages or prompt")
        if self.tools and self.messages is None:
            raise ValueError("tools need messages")
        return self


class Usage(BaseModel):
    prompt_tokens: int  # ids fed after BOS
    completion_tokens: int  # ids generated, EOS included
    positions: int  # the node's budget a sequence may hold


class TextEvent(BaseModel):
    type: Literal["text"] = "text"
    text: str
    ids: list[int]


class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any]
    ids: list[int]


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    reason: core.Reason
    stop_sequence: str | None
    usage: Usage
    model: str
    identity: str


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


Event = TextEvent | ToolCallEvent | DoneEvent | ErrorEvent


def _message(m: Message) -> dict[str, Any]:
    d: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls]
    if m.tool_call_id is not None:
        d["tool_call_id"] = m.tool_call_id
    return d


def to_request(body: Request) -> core.Request:
    return core.Request(
        messages=[_message(m) for m in body.messages] if body.messages is not None else None,
        prompt=body.prompt,
        tools=[{"type": "function", "function": t.model_dump()} for t in body.tools] if body.tools else None,
        max_tokens=body.max_tokens,
        stop=body.stop,
    )


def _line(ev: BaseModel) -> bytes:
    return b"data: " + ev.model_dump_json().encode() + b"\n\n"


async def stream(events: AsyncIterator[core.Event], model: str, identity: str) -> AsyncIterator[bytes]:
    """Every event as it happens; `done` last, or `error` last if the generation broke."""
    try:
        async for ev in events:
            if isinstance(ev, core.Delta):
                yield _line(TextEvent(text=ev.text, ids=ev.ids))
            elif isinstance(ev, core.ToolCall):
                yield _line(ToolCallEvent(id=ev.id, name=ev.name, arguments=ev.arguments, ids=ev.ids))
            else:
                usage = Usage(prompt_tokens=ev.prompt_tokens, completion_tokens=ev.completion_tokens, positions=ev.positions)
                yield _line(DoneEvent(reason=ev.reason, stop_sequence=ev.stop_sequence, usage=usage, model=model, identity=identity))
    except (WireError, OSError) as e:
        yield _line(ErrorEvent(message=f"node: {e}"))
