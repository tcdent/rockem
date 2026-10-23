"""A tiny bundle with a real tokenizer, a fake node, and the app wired to both."""

import hashlib
import struct

import httpx
import pytest
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

from rockem.app import app, build

from fake_node import FakeNode

WORDS = ["hello", "world", "paris", "stop", "x", "y"]  # ids 0..5; 6 BOS, 7 EOS
# ids 8.. decode only (vocab.bin); the node draws them: a call in the tiny model's format, a turn marker
EXTRA = ["<tool_call>", '{"name": "read", "arguments": {"path": "a"}}', "</tool_call>", "\nUser:", "<think>", "</think>"]
CALL_START, CALL_BODY, CALL_END, NEXT_TURN, THINK_OPEN, THINK_CLOSE = 8, 9, 10, 11, 12, 13
BOS, EOS, VOCAB = 6, 7, 14
IDENTITY = 0x0123456789ABCDEF


@pytest.fixture
def anyio_backend():
    return "asyncio"


def tokenizer_json() -> str:
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="x"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens([AddedToken("<bos>", special=True), AddedToken("<eos>", special=True)])
    return tok.to_str()


def vocab_bin() -> bytes:
    tokens = [(" " + w).encode() for w in WORDS] + [b"<bos>", b"<eos>"] + [t.encode() for t in EXTRA]
    offs, body = [], b""
    for t in tokens:
        offs.append(len(body))
        body += t
    offs.append(len(body))
    return struct.pack(f"<{len(offs)}I", *offs) + body


# "ears": a transcription model in the same bundle. Words 0..4; Voxtral's special tokens after.
EAR_WORDS = ["hello", "world", "heard", "you", "x"]
EAR_SPECIALS = ["<s>", "</s>", "[INST]", "[/INST]", "[BEGIN_AUDIO]", "[AUDIO]", "[TRANSCRIBE]"]
E_BOS, E_EOS, E_INST, E_END_INST, E_BEGIN, E_AUDIO, E_TRANSCRIBE = range(5, 12)
EAR_VOCAB = 12
EAR_AUDIO = {"placeholder": E_AUDIO, "window": [4, 100], "rows": 3, "sample_rate": 16000, "hop": 160, "fft_window": 400}


def ears_files() -> dict[str, bytes]:
    import numpy as np

    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(EAR_WORDS)}, unk_token="x"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens([AddedToken(t, special=True) for t in EAR_SPECIALS])
    tokens = [(" " + w).encode() for w in EAR_WORDS] + [t.encode() for t in EAR_SPECIALS]
    offs, body = [], b""
    for t in tokens:
        offs.append(len(body))
        body += t
    offs.append(len(body))
    filters = np.random.default_rng(0).random((4, 201), dtype=np.float32)
    return {
        "tokenizer.json": tok.to_str().encode(),
        "vocab.bin": struct.pack(f"<{len(offs)}I", *offs) + body,
        "mel_filters.f32": filters.astype("<f4").tobytes(),
    }


def wav(seconds: float, rate: int = 16000) -> bytes:
    import io
    import math
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i * 0.05))) for i in range(int(seconds * rate))))
    return buf.getvalue()


@pytest.fixture
def tiny_bundle(tmp_path):
    files = {
        "vocab.bin": vocab_bin(),
        "tokenizer.json": tokenizer_json().encode(),
        "tokenizer_config.json": b'{"chat_template": "{{ bos_token }}{% for m in messages %}{{ m.content }} {% endfor %}{% if add_generation_prompt %}{% endif %}"}',
    }
    every = {f"tiny/{n}": b for n, b in files.items()} | {f"ears/{n}": b for n, b in ears_files().items()}
    for n, b in every.items():
        (tmp_path / n).parent.mkdir(exist_ok=True)
        (tmp_path / n).write_bytes(b)
    listed = "\n".join(f'"{n}" = "sha256:{hashlib.sha256(b).hexdigest()}"' for n, b in every.items())
    a = EAR_AUDIO
    (tmp_path / "manifest.toml").write_text(
        f'[[model]]\nname = "tiny"\nbos = {BOS}\neos = {EOS}\nvocab_size = {VOCAB}\n\n'
        f'[[model]]\nname = "ears"\nbos = {E_BOS}\neos = {E_EOS}\nvocab_size = {EAR_VOCAB}\n\n'
        f'[model.audio]\nplaceholder = {a["placeholder"]}\nwindow = {a["window"]}\nrows = {a["rows"]}\n'
        f'sample_rate = {a["sample_rate"]}\nhop = {a["hop"]}\nfft_window = {a["fft_window"]}\n\n'
        f'[files]\n{listed}\n"tiny/experts.q4g128.bin" = "sha256:00"\n'
    )
    return tmp_path


@pytest.fixture
def node():
    n = FakeNode(IDENTITY, max_ctx=12, draws=[2, 3], bos=BOS, eos=EOS, vocab=VOCAB, models={"tiny"})
    a = EAR_AUDIO
    n.audio["ears"] = {"bos": E_BOS, "eos": E_EOS, "vocab": EAR_VOCAB, "placeholder": a["placeholder"], "window_bytes": 4 * a["window"][0] * a["window"][1], "rows": a["rows"]}
    yield n
    n.close()


@pytest.fixture
def engine(tiny_bundle, node, monkeypatch):
    from rockem import calls

    # the tiny model speaks the JSON call format and opens its next turn with "\nUser:"
    monkeypatch.setitem(calls.FORMATS, "tiny", calls.FORMATS["deepseek-v2-lite"])
    return build(str(tiny_bundle), [node.addr], "tiny", "any")


@pytest.fixture
def transcribing(engine, monkeypatch):
    from rockem import transcribe
    from rockem.app import transcriber

    monkeypatch.setitem(transcribe.FORMATS, "ears", transcribe.FORMATS["voxtral-small-24b"])  # Voxtral's prompt
    t = transcriber(engine, "ears")
    app.state.transcriber = t
    yield t
    del app.state.transcriber


@pytest.fixture
async def client(engine):
    app.state.engine = engine
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://edge") as c:
        yield c
