"""Where text stops. The chat template, the tokenizer, the decoder and stop sequences,
all from one bundle; past this module everything is ids."""

from __future__ import annotations

import json
from typing import Any

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .bundle import Bundle, Decoder
from .calls import FORMATS, Format


class TemplateError(Exception):
    """The chat template refused the messages."""


def template_env() -> ImmutableSandboxedEnvironment:
    """The environment transformers renders chat templates in, so a template behaves as
    its model's publishers tested it: sandboxed, trimmed blocks, loop controls, their
    `tojson` (non-ASCII kept unless asked) and `raise_exception`."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message: str):
        raise TemplateError(message)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


class Edge:
    def __init__(self, bundle: Bundle):
        self.bundle = bundle
        self.bos_text = bundle.vocab[bundle.bos].decode()
        self.eos_text = bundle.vocab[bundle.eos].decode()
        self._template = template_env().from_string(bundle.chat_template)
        # how this model's calls and turns read back out of its text; None: no format known
        self.format: Format | None = FORMATS.get(bundle.manifest.model)

    def render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, add_generation_prompt: bool = True) -> str:
        """The transcript the template makes of the messages and tools, BOS as text at the
        front. Messages and tools are in the shape chat templates take (transformers'): an
        assistant message's `tool_calls` are `{id, type, function: {name, arguments}}`, a
        result is `{role: "tool", tool_call_id, content}`, a tool is `{type, function}`.
        Everything the model reads about tools is the template's; the edge adds nothing."""
        if tools and self.format is None:
            raise TemplateError(f"{self.bundle.manifest.model}: no tool-call format for this model")
        try:
            return self._template.render(messages=messages, tools=tools or None, bos_token=self.bos_text, eos_token=self.eos_text, add_generation_prompt=add_generation_prompt)
        except TemplateError:
            raise
        except Exception as e:
            raise TemplateError(str(e)) from e

    def encode(self, text: str) -> list[int]:
        """Ids for text. The tokenizer adds nothing: BOS is present only if the text has it."""
        return self.bundle.tokenizer.encode(text).ids

    def encode_chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> list[int]:
        return self.encode(self.render(messages, tools))

    def decoder(self) -> Decoder:
        return self.bundle.vocab.decoder()


class Stops:
    """Stop sequences over streamed text. Text in, the text that is safe to emit out: a
    suffix that could be the start of a stop sequence is held until it resolves."""

    def __init__(self, stops: list[str]):
        self.stops = [s for s in stops if s]
        self.matched: str | None = None
        self._buf = ""

    def feed(self, text: str) -> str:
        """Text to emit. Once a stop is found, `matched` names it and nothing more is emitted."""
        if self.matched is not None:
            return ""
        self._buf += text
        hit = min(((i, s) for s in self.stops if (i := self._buf.find(s)) >= 0), default=None)
        if hit is not None:
            i, self.matched = hit
            out, self._buf = self._buf[:i], ""
            return out
        hold = max((k for s in self.stops for k in range(1, min(len(s), len(self._buf) + 1)) if self._buf.endswith(s[:k])), default=0)
        cut = len(self._buf) - hold
        out, self._buf = self._buf[:cut], self._buf[cut:]
        return out

    def flush(self) -> str:
        out, self._buf = self._buf, ""
        return out if self.matched is None else ""
