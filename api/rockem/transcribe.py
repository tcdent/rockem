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

import functools
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


def _voxtral(m: Model, placeholders: int, language: str | None, keywords: list[str]) -> list[int]:
    if keywords:
        raise TranscribeError(f"{m.name}: keywords are not something this model was trained to take")
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


@functools.cache
def _gpt2_split(tokenizer_json: str):
    """Granite's tokenizer with the split rule its reference uses. The checkpoint's processor loads
    GPT2Tokenizer (vocab.json + merges.txt): GPT-2's pre-tokenizer regex. Its tokenizer.json carries
    a newer cl100k-style rule that splits punctuation, digits and newlines differently ("?\n" one
    token there, two in the reference). Same vocabulary and merges, GPT-2's byte-level split: equal
    to GPT2Tokenizer on the prompt and on keyword text (tests/test_granite.py, 2026-10-27)."""
    from tokenizers import Tokenizer, pre_tokenizers

    tok = Tokenizer.from_str(tokenizer_json)
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    return tok


def _granite(m: Model, placeholders: int, language: str | None, keywords: list[str]) -> list[int]:
    """Granite Speech's chat template, `USER: {content}\n ASSISTANT:`, with one of the model card's
    trained requests as content (its "Preferred prompt by task" table):
      - raw ASR, `can you transcribe the speech into a written format?`: lowercase, unpunctuated
        (tcdent wants word accuracy, not cleanup);
      - keyword-biased ASR, `transcribe the speech to text. Keywords: <kw1>, <kw2>, ...`: the
        spellings of names, acronyms and jargon the speaker may use (tcdent 2026-10-27).
    No BOS: the sequence begins with the text. The one `<|audio|>` is the clip's placeholders.
    Language is not part of the prompt (the model detects it; the English request serves every
    language): ignored."""
    tok = _gpt2_split(m.read("tokenizer.json").decode())
    request = f"transcribe the speech to text. Keywords: {', '.join(keywords)}" if keywords else "can you transcribe the speech into a written format?"
    before = tok.encode("USER: ", add_special_tokens=False).ids
    after = tok.encode(f"{request}\n ASSISTANT:", add_special_tokens=False).ids
    return before + [m.audio["placeholder"]] * placeholders + after


# the model's manifest name → its transcription prompt
FORMATS: dict[str, Callable[[Model, int, str | None, list[str]], list[int]]] = {
    "voxtral-small-24b": _voxtral,
    "granite-speech-4.1-2b": _granite,
}

LANGUAGE = re.compile(r"^[a-z]{2,3}$")


@dataclass
class Transcriber:
    engine: core.Engine
    spectrogram: Spectrogram
    prompt: Callable[[Model, int, str | None, list[str]], list[int]]

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

    def request(self, audio: bytes, language: str | None = None, max_tokens: int | None = None, keywords: list[str] | None = None) -> core.Request:
        """The clip's windows and the prompt they fill. `AudioError` for bytes that are not audio,
        `NoSpeech` for a clip with no voice in it."""
        keywords = [k.strip() for k in keywords or []]
        if any(not k or "," in k or "\n" in k for k in keywords):
            raise TranscribeError("keywords: each a non-empty word or phrase, without commas or newlines (the list is comma-separated)")
        if language is not None and not LANGUAGE.match(language):
            raise TranscribeError(f"language {language!r}: a code like en or fr")
        signal = self.spectrogram.signal(audio)
        voiced = self.spectrogram.voiced_seconds(signal)
        if voiced < MIN_VOICED_SECONDS:
            raise NoSpeech(voiced)
        windows = self.spectrogram.windows_of(signal)
        ids = self.prompt(self.model, len(windows) * self.model.audio["rows"], language, keywords)
        return core.Request(ids=ids, windows=windows, max_tokens=max_tokens)
