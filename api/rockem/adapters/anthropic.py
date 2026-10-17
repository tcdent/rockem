"""Anthropic messages, the subset the node can honor: text blocks, a system prompt, stop
sequences, max_tokens. Sampling fields are accepted and ignored."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .. import core


class Message(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class Request(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[Message]
    max_tokens: int = Field(ge=1)
    system: str | list[dict[str, Any]] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False


class TextBlock(BaseModel):
    type: str = "text"
    text: str


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class Response(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    model: str
    content: list[TextBlock]
    stop_reason: str
    stop_sequence: str | None
    usage: Usage


def _text(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if block.get("type") != "text":
            raise ValueError(f"content block {block.get('type')!r}: only text")
        parts.append(block.get("text", ""))
    return "".join(parts)


def to_request(body: Request) -> core.Request:
    messages = []
    if body.system:
        messages.append({"role": "system", "content": _text(body.system)})
    messages += [{"role": m.role, "content": _text(m.content)} for m in body.messages]
    return core.Request(messages=messages, max_tokens=body.max_tokens, stop=list(body.stop_sequences or []))


def _stop(done: core.Done) -> str:
    return {core.Reason.END_TURN: "end_turn", core.Reason.TOOL_USE: "end_turn", core.Reason.STOP_SEQUENCE: "stop_sequence", core.Reason.MAX_TOKENS: "max_tokens", core.Reason.CONTEXT_FULL: "max_tokens"}[done.reason]


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


async def stream(events: AsyncIterator[core.Event], model: str, prompt_tokens: int) -> AsyncIterator[bytes]:
    id = f"msg_{uuid.uuid4().hex[:24]}"
    yield _sse("message_start", {"type": "message_start", "message": {"id": id, "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": prompt_tokens, "output_tokens": 0}}})
    yield _sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
    async for ev in events:
        if isinstance(ev, core.ToolCall):
            continue  # tools are not mapped on this surface yet
        if isinstance(ev, core.Delta):
            if ev.text:
                yield _sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": ev.text}})
        else:
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
            yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": _stop(ev), "stop_sequence": ev.stop_sequence}, "usage": {"output_tokens": ev.completion_tokens}})
            yield _sse("message_stop", {"type": "message_stop"})


def fold(text: str, done: core.Done, model: str) -> Response:
    return Response(id=f"msg_{uuid.uuid4().hex[:24]}", model=model, content=[TextBlock(text=text)], stop_reason=_stop(done), stop_sequence=done.stop_sequence, usage=Usage(input_tokens=done.prompt_tokens, output_tokens=done.completion_tokens))
