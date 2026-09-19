"""Where text stops. The chat template, the tokenizer, the decoder and stop sequences,
all from one bundle; past this module everything is ids."""

from __future__ import annotations

import json
from typing import Any

from jinja2.sandbox import ImmutableSandboxedEnvironment

from .bundle import Bundle, Decoder


class TemplateError(Exception):
    """The chat template refused the messages."""


class Edge:
    def __init__(self, bundle: Bundle):
        self.bundle = bundle
        self.bos_text = bundle.vocab[bundle.bos].decode()
        self.eos_text = bundle.vocab[bundle.eos].decode()
        # the environment transformers renders with, so the template behaves as published
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **kw)

        def raise_exception(message: str):
            raise TemplateError(message)

        env.globals["raise_exception"] = raise_exception
        self._template = env.from_string(bundle.chat_template)

    def render(self, messages: list[dict[str, Any]], add_generation_prompt: bool = True) -> str:
        """The transcript the template makes of the messages, BOS as text at the front."""
        try:
            return self._template.render(messages=messages, bos_token=self.bos_text, eos_token=self.eos_text, add_generation_prompt=add_generation_prompt)
        except TemplateError:
            raise
        except Exception as e:
            raise TemplateError(str(e)) from e

    def encode(self, text: str) -> list[int]:
        """Ids for text. The tokenizer adds nothing: BOS is present only if the text has it."""
        return self.bundle.tokenizer.encode(text).ids

    def encode_chat(self, messages: list[dict[str, Any]]) -> list[int]:
        return self.encode(self.render(messages))

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
