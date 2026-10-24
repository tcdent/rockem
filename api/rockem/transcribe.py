"""Transcription: audio in, text out, a step of its own that the user reviews before sending
the text as an ordinary turn (tcdent 2026-10-23). The audio never enters a conversation.

The edge does all the preprocessing (`audio.Spectrogram`) and renders the model's
transcription prompt; the node receives finished windows as AUDIO and the placeholder ids
they fill (docs/queue/094). The prompt is per model, like a tool-call format: Voxtral's is
mistral-common's `encode_transcription`, verified token for token on the model's tekken.json
(mistral-common 1.12.0):

    <s> [INST] [BEGIN_AUDIO] [AUDIO]×rows·windows [/INST] (lang:xx) [TRANSCRIBE]

Granite Speech 4.1's is its chat template around the model card's raw-ASR request (`_granite`).
Since 2026-10-24 the service transcribes with Granite only (tcdent); Voxtral's format stays for
its bundles.
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


class NoSpeech(Exception):
    """The clip has less voice in it than a single short word: silence, hiss or noise. Sent to
    the model, these come back as invented text (silence became a football season, hiss a
    paragraph about the color pink, 2026-10-23), so they are answered without the node."""

    def __init__(self, seconds: float):
        super().__init__(f"{seconds:.2f} s of voiced audio")
        self.seconds = seconds


# Less voiced audio than this is no speech: a single short word measures ~0.34 s
MIN_VOICED_SECONDS = 0.2


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


def _granite(m: Model, placeholders: int, language: str | None) -> list[int]:
    """Granite Speech's chat template, `USER: {content}\n ASSISTANT:`, with the model card's "ASR,
    raw" request as content: `<|audio|>can you transcribe the speech into a written format?`
    (lowercase, no punctuation: tcdent wants word accuracy, not cleanup). No BOS: the sequence
    begins with the text. The one `<|audio|>` is the clip's placeholders. Language is not part
    of the prompt (the model detects it; the English request serves every language): ignored."""
    tok = m.tokenizer

    def encode(text: str) -> list[int]:
        # Line by line, the newlines on their own. The checkpoint's processor loads GPT2Tokenizer,
        # whose split rule never joins a newline to punctuation ("?" "\n" = 30, 198), while its
        # tokenizer.json's newer rule makes "?\n" one token (5380). The reference's prompt and its
        # golden transcripts use the former (rockem.lead's oracle, 2026-10-24).
        out: list[int] = []
        for i, line in enumerate(text.split("\n")):
            if i:
                out += tok.encode("\n", add_special_tokens=False).ids
            if line:
                out += tok.encode(line, add_special_tokens=False).ids
        return out

    before = encode("USER: ")
    after = encode("can you transcribe the speech into a written format?\n ASSISTANT:")
    return before + [m.audio["placeholder"]] * placeholders + after


# the model's manifest name → its transcription prompt
FORMATS: dict[str, Callable[[Model, int, str | None], list[int]]] = {
    "voxtral-small-24b": _voxtral,
    "granite-speech-4.1-2b": _granite,
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
        """The clip's windows and the prompt they fill. `AudioError` for bytes that are not audio,
        `NoSpeech` for a clip with no voice in it."""
        if language is not None and not LANGUAGE.match(language):
            raise TranscribeError(f"language {language!r}: a code like en or fr")
        signal = self.spectrogram.signal(audio)
        voiced = self.spectrogram.voiced_seconds(signal)
        if voiced < MIN_VOICED_SECONDS:
            raise NoSpeech(voiced)
        windows = self.spectrogram.windows_of(signal)
        ids = self.prompt(self.model, len(windows) * self.model.audio["rows"], language)
        return core.Request(ids=ids, windows=windows, max_tokens=max_tokens)
