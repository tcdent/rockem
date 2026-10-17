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
EXTRA = ["<tool_call>", '{"name": "read", "arguments": {"path": "a"}}', "</tool_call>", "\nUser:"]
CALL_START, CALL_BODY, CALL_END, NEXT_TURN = 8, 9, 10, 11
BOS, EOS, VOCAB = 6, 7, 12
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


@pytest.fixture
def tiny_bundle(tmp_path):
    files = {
        "vocab.bin": vocab_bin(),
        "tokenizer.json": tokenizer_json().encode(),
        "tokenizer_config.json": b'{"chat_template": "{{ bos_token }}{% for m in messages %}{{ m.content }} {% endfor %}{% if add_generation_prompt %}{% endif %}"}',
    }
    for n, b in files.items():
        (tmp_path / n).write_bytes(b)
    listed = "\n".join(f'"{n}" = "sha256:{hashlib.sha256(b).hexdigest()}"' for n, b in files.items())
    (tmp_path / "manifest.toml").write_text(
        f'recipe = "experts:q4g128-a8-rtn"\nrecord_bytes = 64\nexperts = 2\nlayers = [1, 3]\n\n'
        f'[model]\nname = "tiny"\nbos = {BOS}\neos = {EOS}\nvocab_size = {VOCAB}\n\n[files]\n{listed}\n"experts.q4g128.bin" = "sha256:00"\n'
    )
    return tmp_path


@pytest.fixture
def node():
    n = FakeNode(IDENTITY, max_ctx=12, draws=[2, 3], bos=BOS, eos=EOS, vocab=VOCAB)
    yield n
    n.close()


@pytest.fixture
def engine(tiny_bundle, node, monkeypatch):
    from rockem import calls

    # the tiny model speaks the JSON call format and opens its next turn with "\nUser:"
    monkeypatch.setitem(calls.FORMATS, "tiny", calls.FORMATS["deepseek-v2-lite"])
    return build(str(tiny_bundle), [node.addr], "any")


@pytest.fixture
async def client(engine):
    app.state.engine = engine
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://edge") as c:
        yield c
