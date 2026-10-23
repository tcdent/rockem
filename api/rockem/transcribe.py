"""Transcription: audio in, text out, a step of its own that the user reviews before sending
the text as an ordinary turn (tcdent 2026-10-23). The audio never enters a conversation.

The edge does all the preprocessing (`audio.Spectrogram`) and renders the model's
transcription prompt; the node receives finished windows as AUDIO and the placeholder ids
they fill (docs/queue/094). The prompt is per model, like a tool-call format: Voxtral's is
mistral-common's `encode_transcription`, verified token for token on the model's tekken.json
(mistral-common 1.12.0):

    <s> [INST] [BEGIN_AUDIO] [AUDIO]×rows·windows [/INST] (lang:xx) [TRANSCRIBE]
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from . import core
from .audio import Spectrogram
from .bundle import Model

MEL_FILTERS_FILE = "mel_filters.f32"


class TranscribeError(ValueError):
    """The request cannot be transcribed by this model."""


def _voxtral(m: Model, placeholders: int, language: str | None) -> list[int]:
    tok = m.tokenizer

    def special(s: str) -> int:
        i = tok.token_to_id(s)
        if i is None:
            raise TranscribeError(f"{m.name}: the tokenizer has no {s}")
        return i

    ids = [m.bos, special("[INST]"), special("[BEGIN_AUDIO]")] + [m.audio["placeholder"]] * placeholders + [special("[/INST]")]
    if language is not None:
        ids += tok.encode(f"lang:{language}", add_special_tokens=False).ids  # no space, as mistral-common writes it
    return ids + [special("[TRANSCRIBE]")]


# the model's manifest name → its transcription prompt
FORMATS: dict[str, Callable[[Model, int, str | None], list[int]]] = {
    "voxtral-small-24b": _voxtral,
}

LANGUAGE = re.compile(r"^[a-z]{2,3}$")


@dataclass
class Transcriber:
    engine: core.Engine
    spectrogram: Spectrogram
    prompt: Callable[[Model, int, str | None], list[int]]

    @classmethod
    def of(cls, engine: core.Engine) -> Transcriber:
        m = engine.edge.model
        if not m.audio:
            raise TranscribeError(f"{m.name}: no [model.audio] in the manifest")
        if m.name not in FORMATS:
            raise TranscribeError(f"{m.name}: no transcription prompt known for this model")
        return cls(engine, Spectrogram.of(m.audio, m.read(MEL_FILTERS_FILE)), FORMATS[m.name])

    @property
    def model(self) -> Model:
        return self.engine.edge.model

    def request(self, audio: bytes, language: str | None = None, max_tokens: int | None = None) -> core.Request:
        """The clip's windows and the prompt they fill. `AudioError` for bytes that are not audio."""
        if language is not None and not LANGUAGE.match(language):
            raise TranscribeError(f"language {language!r}: a code like en or fr")
        windows = self.spectrogram.windows(audio)
        ids = self.prompt(self.model, len(windows) * self.model.audio["rows"], language)
        return core.Request(ids=ids, windows=windows, max_tokens=max_tokens)
