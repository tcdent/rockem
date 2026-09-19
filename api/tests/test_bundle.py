import hashlib
import struct
from pathlib import Path

import pytest
from rockem.bundle import Bundle, BundleError, Manifest, Vocab, identity

REAL = Path(__file__).resolve().parents[2] / "bundles" / "v2lite-q4gptq"

TOKENS = [b"", b"a", b"b", "é".encode(), "é".encode()[:1], "é".encode()[1:], b"<bos>", b"<eos>"]


def vocab_bin(tokens: list[bytes]) -> bytes:
    offs, body = [], b""
    for t in tokens:
        offs.append(len(body))
        body += t
    offs.append(len(body))
    return struct.pack(f"<{len(offs)}I", *offs) + body


def manifest(files: dict[str, bytes], bos=6, eos=7, vocab_size=8) -> bytes:
    listed = "\n".join(f'"{n}" = "sha256:{hashlib.sha256(b).hexdigest()}"' for n, b in files.items())
    return (
        'recipe = "experts:q4g128-a8-rtn"\nrecord_bytes = 64\nexperts = 2\nlayers = [1, 3]\n\n'
        f'[model]\nname = "tiny"\nbos = {bos}\neos = {eos}\nvocab_size = {vocab_size}\n\n'
        f'[files]\n{listed}\n"experts.q4g128.bin" = "sha256:00"\n\n# checkpoint.toml\n# repo = "x/y"\n# revision = "abc"\n# pinned = 2026-09-16\n'
    ).encode()


@pytest.fixture
def tiny(tmp_path):
    vb = vocab_bin(TOKENS)
    files = {"vocab.bin": vb, "tokenizer_config.json": b'{"chat_template": "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"}'}
    for n, b in files.items():
        (tmp_path / n).write_bytes(b)
    (tmp_path / "manifest.toml").write_bytes(manifest(files))
    return tmp_path


def test_manifest_fields(tiny):
    b = Bundle(tiny)
    m = b.manifest
    assert (m.model, m.recipe, m.calibration, m.record_bytes, m.experts, m.layers) == ("tiny", "experts:q4g128-a8-rtn", None, 64, 2, (1, 3))
    assert (b.bos, b.eos, b.vocab_size) == (6, 7, 8)
    assert m.checkpoint.repo == "x/y" and m.checkpoint.revision == "abc" and m.checkpoint.pinned == "2026-09-16"
    assert set(m.files) == {"vocab.bin", "tokenizer_config.json", "experts.q4g128.bin"}
    assert b.identity == identity((tiny / "manifest.toml").read_bytes())
    assert b.present() == ["vocab.bin", "tokenizer_config.json"]  # the weights are listed, not here


def test_files_are_verified(tiny):
    b = Bundle(tiny)
    assert b.read("vocab.bin") == vocab_bin(TOKENS)
    with pytest.raises(BundleError):
        b.read("experts.q4g128.bin")  # absent
    with pytest.raises(BundleError):
        b.read("sink.f32")  # unlisted
    (tiny / "vocab.bin").write_bytes(b"x" * 40)
    with pytest.raises(BundleError):
        Bundle(tiny).read("vocab.bin")  # hash


def test_manifest_rejects_the_malformed(tiny):
    with pytest.raises(BundleError):
        Manifest.parse(b"model = 1\n[model]\n")
    with pytest.raises(BundleError):
        Manifest.parse(manifest({}).replace(b"vocab_size = 8", b""))
    with pytest.raises(BundleError):
        Bundle(tiny / "nope")


def test_vocab_and_streaming_decode(tiny):
    v = Bundle(tiny).vocab
    assert len(v) == 8 and v[0] == b"" and v[1] == b"a" and v[6] == b"<bos>"
    d = v.decoder()
    assert [d.feed(i) for i in (1, 3, 4, 5, 2)] == ["a", "é", "", "é", "b"]
    assert d.feed(4) == "" and d.flush() == "�"
    with pytest.raises(BundleError):
        Vocab(vocab_bin(TOKENS)[:-1], 8)
    with pytest.raises(BundleError):
        Vocab(b"\x00" * 8, 8)


def test_chat_template(tiny):
    assert Bundle(tiny).chat_template.startswith("{{ bos_token }}")


@pytest.mark.skipif(not (REAL / "manifest.toml").exists(), reason="no built bundle on this host")
def test_real_bundle_round_trip():
    b = Bundle(REAL)
    assert b.manifest.model == "deepseek-v2-lite" and (b.bos, b.eos, b.vocab_size) == (100000, 100001, 102400)
    assert b.manifest.checkpoint.repo == "deepseek-ai/DeepSeek-V2-Lite"
    v = b.vocab
    assert len(v) == 102400 and v[b.bos] == "<｜begin▁of▁sentence｜>".encode() and v[102399] == b""
    text = "Hello, wörld — 你好"
    assert b.tokenizer.encode(text).ids[0] != b.bos  # the tokenizer adds nothing itself
    ids = b.tokenizer.encode(v[b.bos].decode() + text).ids  # the template writes bos_token as text
    assert ids[0] == b.bos and ids.count(b.bos) == 1  # and it is the wire's BOS, once
    d = v.decoder()
    assert "".join(d.feed(i) for i in ids[1:]) + d.flush() == text
    assert "User: " in b.chat_template
