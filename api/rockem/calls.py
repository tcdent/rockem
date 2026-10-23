"""Tool calls out of generated text. A model's chat template defines its call syntax
(it renders past calls); this module reads that syntax back, one `Format` per model,
keyed by the bundle's model name. Each format is checked by round trip: what its
model's template renders for a call, its parser returns as that call.

A format is two markers that bracket a call, a decode from the bracketed text to
(name, arguments), and the strings that open the next turn in that model's transcript,
which end the assistant's turn when the model does not end it with EOS itself."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class CallError(ValueError):
    """The text between a call's markers is not a call in this model's syntax."""


@dataclass(frozen=True)
class Format:
    start: str
    end: str
    decode: Callable[[str], tuple[str, dict[str, Any]]]
    turn_stops: tuple[str, ...]
    think: tuple[str, str] | None = None  # the markers around the model's reasoning, if it reasons


def _json_call(span: str) -> tuple[str, dict[str, Any]]:
    try:
        obj = json.loads(span)
        name, args = obj["name"], obj.get("arguments", {})
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise CallError(f"{span!r}: {e}") from e
    if isinstance(args, str):
        args = json.loads(args)
    if not isinstance(name, str) or not isinstance(args, dict):
        raise CallError(f"{span!r}: name or arguments of the wrong type")
    return name, args


_ARG = re.compile(r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>", re.S)


def _glm_call(span: str) -> tuple[str, dict[str, Any]]:
    # name<arg_key>k</arg_key><arg_value>v</arg_value>…; a value is JSON unless it was a string
    name, _, rest = span.partition("<arg_key>")
    rest = "<arg_key>" + rest if rest else ""
    args: dict[str, Any] = {}
    for k, v in _ARG.findall(rest):
        try:
            args[k] = json.loads(v)
        except json.JSONDecodeError:
            args[k] = v
    if not name.strip() or _ARG.sub("", rest).strip():
        raise CallError(f"{span!r}: not name<arg_key>…</arg_value> pairs")
    return name.strip(), args


FORMATS: dict[str, Format] = {
    # rockem's own format for a checkpoint with none (models/deepseek-v2-lite/chat_template.jinja)
    "deepseek-v2-lite": Format("<tool_call>", "</tool_call>", _json_call, ("\nUser:", "\nTool:", "\nAssistant:")),
    # the checkpoint's native format (its chat_template.jinja); <|user|> and <|observation|> open the next turn
    "glm-5.3-flash": Format("<tool_call>", "</tool_call>", _glm_call, ("<|user|>", "<|observation|>"), ("<think>", "</think>")),
}


@dataclass
class Text:
    text: str


@dataclass
class Call:
    name: str
    arguments: dict[str, Any]
    raw: str  # the text between the markers, as generated


class Parser:
    """Streamed text in; plain text and complete calls out, in order. Text that could be the
    start of a call marker is held until it resolves; a call is emitted when an end marker
    arrives after which it decodes. A call still open when the generation ends comes back
    as the text it was (`flush`)."""

    def __init__(self, fmt: Format):
        self.fmt = fmt
        self._buf = ""
        self._in_call = False
        self._scan = 0  # within a call: where to look for the next end marker

    def feed(self, text: str) -> list[Text | Call]:
        self._buf += text
        out: list[Text | Call] = []
        while True:
            if self._in_call:
                # the call ends at the first end marker after which it decodes: an argument may
                # contain the marker itself (code that mentions it), and a cut there never decodes
                i = self._buf.find(self.fmt.end, self._scan)
                if i < 0:
                    return out
                try:
                    name, args = self.fmt.decode(self._buf[:i])
                except CallError:
                    self._scan = i + len(self.fmt.end)
                    continue
                span, self._buf = self._buf[:i], self._buf[i + len(self.fmt.end) :]
                self._in_call, self._scan = False, 0
                out.append(Call(name, args, span))
                continue
            i = self._buf.find(self.fmt.start)
            if i >= 0:
                if i:
                    out.append(Text(self._buf[:i]))
                self._buf = self._buf[i + len(self.fmt.start) :]
                self._in_call = True
                continue
            hold = max((k for k in range(1, len(self.fmt.start)) if self._buf.endswith(self.fmt.start[:k])), default=0)
            cut = len(self._buf) - hold
            if cut:
                out.append(Text(self._buf[:cut]))
            self._buf = self._buf[cut:]
            return out

    @property
    def in_call(self) -> bool:
        return self._in_call

    def flush(self) -> str:
        """What is held at the end of a generation: an unfinished call comes back as text."""
        out = (self.fmt.start if self._in_call else "") + self._buf
        self._buf, self._in_call, self._scan = "", False, 0
        return out


class Thinking:
    """Streamed text in; (reasoning, text) pieces out, in order, the markers dropped. A reply
    is reasoning while inside the model's think markers: from the start when the prompt
    already opened them (the generation prompt ends with the open marker), or when the reply
    itself begins with the open marker. Text that could be part of a marker is held."""

    def __init__(self, markers: tuple[str, str], inside: bool):
        self.open, self.close = markers
        self.inside = inside
        self._at_start = not inside  # only a reply's beginning can open reasoning
        self._buf = ""

    def feed(self, text: str) -> list[tuple[bool, str]]:
        self._buf += text
        out: list[tuple[bool, str]] = []
        while self._buf:
            if self.inside:
                i = self._buf.find(self.close)
                if i >= 0:
                    if i:
                        out.append((True, self._buf[:i]))
                    self._buf, self.inside = self._buf[i + len(self.close) :], False
                    continue
                return out + self._emit(True, self.close)
            if self._at_start:
                lead = self._buf.lstrip()
                if lead.startswith(self.open):
                    self._buf, self.inside, self._at_start = lead[len(self.open) :], True, False
                    continue
                if not lead or self.open.startswith(lead):
                    return out  # still could open
                self._at_start = False
            out.append((False, self._buf))
            self._buf = ""
        return out

    def _emit(self, reasoning: bool, marker: str) -> list[tuple[bool, str]]:
        hold = max((k for k in range(1, len(marker)) if self._buf.endswith(marker[:k])), default=0)
        cut = len(self._buf) - hold
        piece, self._buf = self._buf[:cut], self._buf[cut:]
        return [(reasoning, piece)] if piece else []

    def flush(self) -> list[tuple[bool, str]]:
        piece, self._buf = self._buf, ""
        return [(self.inside, piece)] if piece else []
