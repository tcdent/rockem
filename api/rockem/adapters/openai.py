"""OpenAI chat completions, the subset the node can honor. Sampling fields are accepted
and ignored: the model samples at its own constants."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .. import core


class Message(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None


class Request(BaseModel):
    model_config = ConfigDict(extra="allow")  # temperature, top_p, seed, …: accepted, ignored
    model: str
    messages: list[Message]
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Response(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


def _text(content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if part.get("type") != "text":
            raise ValueError(f"content part {part.get('type')!r}: only text")
        parts.append(part.get("text", ""))
    return "".join(parts)


def to_request(body: Request) -> core.Request:
    stop = [body.stop] if isinstance(body.stop, str) else list(body.stop or [])
    return core.Request(
        messages=[{"role": m.role, "content": _text(m.content)} for m in body.messages],
        max_tokens=body.max_completion_tokens or body.max_tokens,
        stop=stop,
    )


def _finish(done: core.Done) -> str:
    return "stop" if done.reason in (core.Reason.EOS, core.Reason.STOP_SEQUENCE) else "length"


def _usage(done: core.Done) -> Usage:
    return Usage(prompt_tokens=done.prompt_tokens, completion_tokens=done.completion_tokens, total_tokens=done.prompt_tokens + done.completion_tokens)


def _chunk(id: str, created: int, model: str, delta: dict[str, Any], finish: str | None, usage: Usage | None = None) -> bytes:
    d: dict[str, Any] = {"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        d["usage"] = usage.model_dump()
    return f"data: {json.dumps(d, ensure_ascii=False)}\n\n".encode()


async def stream(events: AsyncIterator[core.Event], model: str, include_usage: bool) -> AsyncIterator[bytes]:
    id, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
    yield _chunk(id, created, model, {"role": "assistant", "content": ""}, None)
    async for ev in events:
        if isinstance(ev, core.Delta):
            if ev.text:
                yield _chunk(id, created, model, {"content": ev.text}, None)
        else:
            yield _chunk(id, created, model, {}, _finish(ev), _usage(ev) if include_usage else None)
    yield b"data: [DONE]\n\n"


def fold(text: str, done: core.Done, model: str) -> Response:
    return Response(id=f"chatcmpl-{uuid.uuid4().hex[:24]}", created=int(time.time()), model=model, choices=[Choice(message=ChoiceMessage(content=text), finish_reason=_finish(done))], usage=_usage(done))
