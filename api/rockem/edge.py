"""Where text stops. The chat template, the tokenizer, the decoder and stop sequences,
all from one model in a bundle; past this module everything is ids."""

from __future__ import annotations

import json
from typing import Any

from functools import cached_property

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .bundle import Decoder, Model
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
    def __init__(self, model: Model):
        self.model = model
        self.bos_text = model.vocab[model.bos].decode()
        self.eos_text = model.vocab[model.eos].decode()
        # how this model's calls and turns read back out of its text; None: no format known
        self.format: Format | None = FORMATS.get(model.name)

    @cached_property
    def _template(self):
        """The model's chat template, read when first rendered: a transcription-only model has none."""
        return template_env().from_string(self.model.chat_template)

    def render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, add_generation_prompt: bool = True, reasoning_effort: str | None = None) -> str:
        """The transcript the template makes of the messages and tools, BOS as text at the
        front. Messages and tools are in the shape chat templates take (transformers'): an
        assistant message's `tool_calls` are `{id, type, function: {name, arguments}}`, a
        result is `{role: "tool", tool_call_id, content}`, a tool is `{type, function}`.
        Everything the model reads about tools is the template's; the edge adds nothing.
        `reasoning_effort` is the template variable of that name (GLM-5.3-Flash: low, high,
        max); a template that does not read it ignores it."""
        if tools and self.format is None:
            raise TemplateError(f"{self.model.name}: no tool-call format for this model")
        try:
            return self._template.render(messages=messages, tools=tools or None, bos_token=self.bos_text, eos_token=self.eos_text, add_generation_prompt=add_generation_prompt, reasoning_effort=reasoning_effort)
        except TemplateError:
            raise
        except Exception as e:
            raise TemplateError(str(e)) from e

    def render_noted(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, reasoning_effort: str | None = None) -> tuple[str, list[int]]:
        """`render(messages, tools)` and where each message's text starts in it, from one render:
        the template is handed a list that notes, as the template takes each message, how much text
        it has produced so far. A template may loop over the messages more than once (GLM's
        does: a pass for the last user's index, then the render); the last pass's notes are the
        starts. Equal to the per-prefix renders' lengths (tests/test_edge_noting.py, per model)."""
        if tools and self.format is None:
            raise TemplateError(f"{self.model.name}: no tool-call format for this model")
        produced = [0]
        notes: list[int] = []

        class Noting(list):
            def __iter__(self):
                for m in list.__iter__(self):
                    notes.append(produced[0])
                    yield m

        parts: list[str] = []
        try:
            for piece in self._template.generate(messages=Noting(messages), tools=tools or None, bos_token=self.bos_text, eos_token=self.eos_text, add_generation_prompt=True, reasoning_effort=reasoning_effort):
                parts.append(piece)
                produced[0] += len(piece)
        except TemplateError:
            raise
        except Exception as e:
            raise TemplateError(str(e)) from e
        starts = notes[len(notes) - len(messages) :] if messages else []
        if len(notes) % max(len(messages), 1) or any(b < a for a, b in zip(starts, starts[1:])):
            raise TemplateError(f"{self.model.name}: the template's message loop cannot be noted")
        return "".join(parts), starts

    @cached_property
    def generation_prompt(self) -> str:
        """The text a generation prompt adds after a turn (GLM: `<|assistant|><think>`): the
        opening every reply was generated after."""
        one = [{"role": "user", "content": ""}]
        bare = self.render(one, add_generation_prompt=False)
        opened = self.render(one, add_generation_prompt=True)
        if not opened.startswith(bare):
            raise TemplateError(f"{self.model.name}: the generation prompt does not follow the conversation")
        return opened[len(bare) :]

    def encode(self, text: str) -> list[int]:
        """Ids for text. The tokenizer adds nothing: BOS is present only if the text has it."""
        return self.model.tokenizer.encode(text).ids

    @cached_property
    def split_marks(self) -> tuple[str, ...]:
        """Special tokens text can be split right before without changing its ids: the tokenizer
        takes added tokens out first and encodes the text between them on its own, unless the token
        swallows the whitespace on its left (lstrip). GLM's turns all start with one."""
        added = self.model.tokenizer.get_added_tokens_decoder().values() if hasattr(self.model.tokenizer, "get_added_tokens_decoder") else ()
        return tuple(t.content for t in added if t.special and not t.lstrip)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """`encode` of each text, in parallel (the tokenizer is Rust): the same ids as one by one."""
        return [e.ids for e in self.model.tokenizer.encode_batch(texts)]  # encode's defaults, as `encode`

    def encode_chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> list[int]:
        return self.encode(self.render(messages, tools))

    def decoder(self) -> Decoder:
        return self.model.vocab.decoder()


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
