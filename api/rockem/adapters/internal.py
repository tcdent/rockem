"""The internal surface: the canonical request and the events, as they are."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .. import core


class Message(BaseModel):
    role: str
    content: str


class Request(BaseModel):
    messages: list[Message] | None = None
    prompt: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    stop: list[str] = Field(default_factory=list, max_length=8)
    stream: bool = False

    @model_validator(mode="after")
    def one_of(self):
        if (self.messages is None) == (self.prompt is None):
            raise ValueError("exactly one of messages or prompt")
        return self


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    positions: int


class Response(BaseModel):
    text: str
    ids: list[int]
    reason: core.Reason
    stop_sequence: str | None
    usage: Usage
    model: str
    identity: str
    temperature: float | None = None  # the model's own; not a request field


def to_request(body: Request) -> core.Request:
    return core.Request(
        messages=[m.model_dump() for m in body.messages] if body.messages is not None else None,
        prompt=body.prompt,
        max_tokens=body.max_tokens,
        stop=body.stop,
    )


def _sse(event: str, data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


async def stream(events: AsyncIterator[core.Event], model: str, identity: str) -> AsyncIterator[bytes]:
    async for ev in events:
        if isinstance(ev, core.Delta):
            yield _sse("delta", {"text": ev.text, "ids": ev.ids})
        else:
            yield _sse("done", {"reason": ev.reason.value, "stop_sequence": ev.stop_sequence, "usage": Usage(prompt_tokens=ev.prompt_tokens, completion_tokens=ev.completion_tokens, positions=ev.positions).model_dump(), "model": model, "identity": identity})


def fold(text: str, ids: list[int], done: core.Done, model: str, identity: str) -> Response:
    return Response(text=text, ids=ids, reason=done.reason, stop_sequence=done.stop_sequence, usage=Usage(prompt_tokens=done.prompt_tokens, completion_tokens=done.completion_tokens, positions=done.positions), model=model, identity=identity)


Literal  # re-exported for the views' annotations
