"""Granite Speech 4.1 2B at the edge, against the real bundle and rockem.lead's reference
(bringup/granite/oracle.py, bundles/goldens-granite/libri2). Skipped on a host without them."""

from pathlib import Path

import numpy as np
import pytest
from rockem.audio import Spectrogram
from rockem.bundle import Bundle
from rockem.transcribe import FORMATS

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "bundles/glm-5.3-flash-q4rtn"
GOLDEN = ROOT / "bundles/goldens-granite/libri2"
CLIP = ROOT / "bringup/voxtral/data/libri2.wav"
have = BUNDLE.exists() and GOLDEN.exists() and CLIP.exists() and b"granite-speech-4.1-2b" in (BUNDLE / "manifest.toml").read_bytes()
pytestmark = pytest.mark.skipif(not have, reason="no Granite bundle and golden on this host")


@pytest.fixture(scope="module")
def granite():
    return Bundle(BUNDLE).model("granite-speech-4.1-2b")


def test_the_log_mel_is_the_references(granite):
    spec = Spectrogram.of(granite.audio, granite.read("mel_filters.f32"))
    assert (spec.fft, spec.fft_window, spec.filters.shape) == (512, 400, (80, 257))
    mine = spec.log_mel(spec.signal(CLIP.read_bytes()))
    ref = np.fromfile(GOLDEN / "mel.f32", dtype="<f4").reshape(80, -1)
    d = np.abs(mine[:, : ref.shape[1]] - ref)  # the reference keeps fewer frames: only the tail differs
    # the reference is float32: the same math in float32 vs float64 differs by up to 1.5e-5 (2026-10-24)
    assert d.max() < 3e-5 and d.mean() < 1e-6


def test_the_prompt_is_the_references(granite):
    ids = FORMATS[granite.name](granite, 150, None, [])
    assert ids == [6584, 25, 220] + [100352] * 150 + [4919, 499, 1380, 3191, 279, 8982, 1139, 264, 5439, 3645, 30, 198, 36660, 3931, 2891, 25]
    assert FORMATS[granite.name](granite, 30, "fr", []) == FORMATS[granite.name](granite, 30, None, [])  # language ignored


SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--ibm-granite--granite-speech-4.1-2b/snapshots/de575db64086f84fdc79da4932d1076e965bc546"


def test_keywords_take_the_model_cards_keyword_prompt(granite):
    """The keyword-biased request of the model card, its text exactly; tokenized as the reference."""
    from rockem.transcribe import _gpt2_split

    ids = FORMATS[granite.name](granite, 30, None, ["rockem", "GLM-5.3-Flash"])
    tok = _gpt2_split(granite.read("tokenizer.json").decode())
    text = tok.decode(ids, skip_special_tokens=False)
    assert text.endswith("transcribe the speech to text. Keywords: rockem, GLM-5.3-Flash\n ASSISTANT:")
    assert ids.count(granite.audio["placeholder"]) == 30


@pytest.mark.skipif(not (SNAPSHOT / "vocab.json").exists(), reason="no Granite checkpoint (for GPT2Tokenizer)")
def test_the_split_is_gpt2tokenizers(granite):
    transformers = pytest.importorskip("transformers")
    from rockem.transcribe import _gpt2_split

    ref = transformers.GPT2Tokenizer.from_pretrained(str(SNAPSHOT))
    tok = _gpt2_split(granite.read("tokenizer.json").decode())
    for t in ["USER: <|audio|>can you transcribe the speech into a written format?\n ASSISTANT:",
              "transcribe the speech to text. Keywords: rockem, GLM-5.3-Flash, C++, (OpenAI), 12345, tcdent's, naïve, Zürich, x86_64\n ASSISTANT:",
              "a  b\t\tc   \n\n  d?!\n x"]:
        assert tok.encode(t, add_special_tokens=False).ids == ref.encode(t, add_special_tokens=False)
