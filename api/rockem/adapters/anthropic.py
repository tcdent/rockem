"""Anthropic's Messages API, as Claude Code speaks it (ANTHROPIC_BASE_URL at this service).

Mapped onto the core, nothing invented:
- `system` (a string or text blocks, cache_control ignored) is the system message;
- an assistant turn's `thinking` blocks are its reasoning, its `text` blocks its content, its
  `tool_use` blocks its calls; a user turn's `tool_result` blocks are tool messages, in order,
  before the rest of its text;
- `tools` are `{name, description, input_schema}`; server tools without a schema are left out;
- the reply streams as Anthropic's events: a `thinking` block (reasoning, with a signature, which
  Claude Code sends back and which is never checked), `text` blocks, and one `tool_use` block a
  call, its input whole in one `input_json_delta`; `stop_reason` `tool_use` when the turn called.
Sampling fields, `thinking`, `tool_choice`, `metadata` and cache_control are accepted and
ignored: the model samples and reasons at its own constants. Images and documents are refused.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .. import core
from ..wire import WireError
from .internal import batched

SIGNATURE = "rockem"  # thinking blocks need one to be sent back; nothing verifies it


class Message(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class Request(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[Message]
    max_tokens: int | None = Field(default=None, ge=1)
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class Response(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    model: str
    content: list[dict[str, Any]]
    stop_reason: str
    stop_sequence: str | None
    usage: Usage


def _text(blocks: str | list[dict[str, Any]], where: str) -> str:
    if isinstance(blocks, str):
        return blocks
    parts = []
    for b in blocks:
        if b.get("type") != "text":
            raise ValueError(f"{where}: {b.get('type')!r} content is not supported by this model")
        parts.append(b.get("text", ""))
    return "".join(parts)


def _message(m: Message) -> list[dict[str, Any]]:
    """One Anthropic message as the core's (template-shaped) messages: usually one, but a user
    turn carrying tool results becomes a tool message for each, then the user's own text."""
    if isinstance(m.content, str):
        return [{"role": m.role, "content": m.content}]
    if m.role == "assistant":
        text, thinking, calls = [], [], []
        for b in m.content:
            kind = b.get("type")
            if kind == "text":
                text.append(b.get("text", ""))
            elif kind == "thinking":
                thinking.append(b.get("thinking", ""))
            elif kind == "redacted_thinking":
                pass
            elif kind == "tool_use":
                calls.append({"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": b.get("input") or {}}})
            else:
                raise ValueError(f"assistant: {kind!r} content is not supported by this model")
        out: dict[str, Any] = {"role": "assistant", "content": "".join(text)}
        if thinking:
            out["reasoning_content"] = "".join(thinking)
        if calls:
            out["tool_calls"] = calls
        return [out]
    out, rest = [], []
    for b in m.content:
        if b.get("type") == "tool_result":
            content = _text(b.get("content") or "", "tool_result")
            out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": content})
        else:
            rest.append(b)
    if rest or not out:
        out.append({"role": m.role, "content": _text(rest, m.role)})
    return out


def to_request(body: Request) -> core.Request:
    messages = []
    if body.system:
        messages.append({"role": "system", "content": _text(body.system, "system")})
    for m in body.messages:
        messages += _message(m)
    tools = [
        {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t["input_schema"]}}
        for t in body.tools or []
        if "input_schema" in t
    ]
    return core.Request(messages=messages, tools=tools or None, max_tokens=body.max_tokens, stop=list(body.stop_sequences or []))


def _stop(done: core.Done) -> str:
    return {
        core.Reason.END_TURN: "end_turn",
        core.Reason.TOOL_USE: "tool_use",
        core.Reason.STOP_SEQUENCE: "stop_sequence",
        core.Reason.MAX_TOKENS: "max_tokens",
        core.Reason.CONTEXT_FULL: "max_tokens",
    }.get(done.reason, "end_turn")


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def _tool_id(ev: core.ToolCall) -> str:
    return "toolu_" + ev.id.removeprefix("call_")


async def stream(events: AsyncIterator[core.Event], model: str, prompt_tokens: int) -> AsyncIterator[bytes]:
    """Anthropic's event stream: content blocks opened, filled and closed in order."""
    id = f"msg_{uuid.uuid4().hex[:24]}"
    yield _sse("message_start", {"type": "message_start", "message": {"id": id, "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": prompt_tokens, "output_tokens": 0}}})
    index, open_kind = -1, None

    def close() -> list[bytes]:
        nonlocal open_kind
        out = []
        if open_kind == "thinking":
            out.append(_sse("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "signature_delta", "signature": SIGNATURE}}))
        if open_kind is not None:
            out.append(_sse("content_block_stop", {"type": "content_block_stop", "index": index}))
        open_kind = None
        return out

    def open_(kind: str, block: dict[str, Any]) -> list[bytes]:
        nonlocal index, open_kind
        out = close()
        index += 1
        open_kind = kind
        return out + [_sse("content_block_start", {"type": "content_block_start", "index": index, "content_block": block})]

    try:
        async for ev in batched(events):
            if isinstance(ev, core.Reasoning):
                if not ev.text:
                    continue
                if open_kind != "thinking":
                    for b in open_("thinking", {"type": "thinking", "thinking": "", "signature": ""}):
                        yield b
                yield _sse("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "thinking_delta", "thinking": ev.text}})
            elif isinstance(ev, core.Delta):
                if not ev.text:
                    continue
                if open_kind != "text":
                    for b in open_("text", {"type": "text", "text": ""}):
                        yield b
                yield _sse("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": ev.text}})
            elif isinstance(ev, core.ToolCall):
                for b in open_("tool_use", {"type": "tool_use", "id": _tool_id(ev), "name": ev.name, "input": {}}):
                    yield b
                yield _sse("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": json.dumps(ev.arguments, ensure_ascii=False)}})
                for b in close():
                    yield b
            elif isinstance(ev, core.Done):
                for b in close():
                    yield b
                yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": _stop(ev), "stop_sequence": ev.stop_sequence}, "usage": {"output_tokens": ev.completion_tokens}})
                yield _sse("message_stop", {"type": "message_stop"})
    except (WireError, OSError) as e:
        yield _sse("error", {"type": "error", "error": {"type": "api_error", "message": f"node: {e}"}})


async def collect(events: AsyncIterator[core.Event]) -> list[core.Event]:
    return [ev async for ev in events]


def fold(events: list[core.Event], model: str) -> Response:
    """The non-streaming response: the same blocks, whole."""
    content: list[dict[str, Any]] = []
    done = None
    for ev in events:
        last = content[-1] if content else None
        if isinstance(ev, core.Reasoning) and ev.text:
            if last and last["type"] == "thinking":
                last["thinking"] += ev.text
            else:
                content.append({"type": "thinking", "thinking": ev.text, "signature": SIGNATURE})
        elif isinstance(ev, core.Delta) and ev.text:
            if last and last["type"] == "text":
                last["text"] += ev.text
            else:
                content.append({"type": "text", "text": ev.text})
        elif isinstance(ev, core.ToolCall):
            content.append({"type": "tool_use", "id": _tool_id(ev), "name": ev.name, "input": ev.arguments})
        elif isinstance(ev, core.Done):
            done = ev
    assert done is not None
    return Response(id=f"msg_{uuid.uuid4().hex[:24]}", model=model, content=content, stop_reason=_stop(done), stop_sequence=done.stop_sequence, usage=Usage(input_tokens=done.prompt_tokens, output_tokens=done.completion_tokens))


def error(status: int, message: str) -> dict[str, Any]:
    """Anthropic's error body. Over the context it is the 400 Claude Code reads as 'compact now'."""
    kind = {400: "invalid_request_error", 404: "not_found_error", 413: "request_too_large", 422: "invalid_request_error", 499: "api_error", 503: "overloaded_error"}.get(status, "api_error")
    return {"type": "error", "error": {"type": kind, "message": message}}
