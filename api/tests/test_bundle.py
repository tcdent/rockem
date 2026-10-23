import hashlib
import struct
from pathlib import Path

import pytest
from rockem.bundle import Bundle, BundleError, Manifest, Vocab, identity

GLM = Path(__file__).resolve().parents[2] / "bundles" / "glm-5.3-flash-q4rtn"

TOKENS = [b"", b"a", b"b", "é".encode(), "é".encode()[:1], "é".encode()[1:], b"<bos>", b"<eos>"]
TEMPLATE = b'{"chat_template": "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"}'


def vocab_bin(tokens: list[bytes]) -> bytes:
    offs, body = [], b""
    for t in tokens:
        offs.append(len(body))
        body += t
    offs.append(len(body))
    return struct.pack(f"<{len(offs)}I", *offs) + body


def manifest(files: dict[str, bytes], extra_model: str = "") -> bytes:
    listed = "\n".join(f'"{n}" = "sha256:{hashlib.sha256(b).hexdigest()}"' for n, b in files.items())
    return (
        '[[model]]\nname = "tiny"\nrecipe = "experts:q4g128-a8-rtn"\nrecord_bytes = 64\nexperts = 2\nlayers = [1, 3]\nbos = 6\neos = 7\nvocab_size = 8\n\n'
        '[[model]]\nname = "ears"\nbos = 1\neos = 2\nvocab_size = 8\nprefix = [1, 3]\neos_tokens = [2, 4]\n\n'
        '[model.audio]\nplaceholder = 5\nwindow = [128, 3000]\nrows = 375\n\n'
        f"{extra_model}"
        f'[files]\n{listed}\n"tiny/experts.q4g128.bin" = "sha256:00"\n\n# checkpoint.toml\n# repo = "x/y"\n# revision = "abc"\n# pinned = 2026-09-16\n'
    ).encode()


@pytest.fixture
def tiny(tmp_path):
    files = {"tiny/vocab.bin": vocab_bin(TOKENS), "tiny/tokenizer_config.json": TEMPLATE, "ears/vocab.bin": vocab_bin(TOKENS[::-1])}
    for n, b in files.items():
        (tmp_path / n).parent.mkdir(exist_ok=True)
        (tmp_path / n).write_bytes(b)
    (tmp_path / "manifest.toml").write_bytes(manifest(files))
    return tmp_path


def test_manifest_lists_its_models(tiny):
    b = Bundle(tiny)
    m = b.manifest
    assert list(m.models) == ["tiny", "ears"]
    assert m.checkpoint.repo == "x/y" and m.checkpoint.revision == "abc" and m.checkpoint.pinned == "2026-09-16"
    assert b.identity == identity((tiny / "manifest.toml").read_bytes())
    t, e = b.model("tiny"), b.model("ears")
    assert (t.recipe, t.record_bytes, t.experts, t.layers) == ("experts:q4g128-a8-rtn", 64, 2, (1, 3))
    assert (e.recipe, e.experts, e.layers) == (None, None, None)  # each model's own; ears has none
    assert (t.bos, t.eos, t.vocab_size, t.prefix, t.eos_tokens, t.audio) == (6, 7, 8, (6,), frozenset({7}), None)  # defaults: [bos], {eos}
    assert (e.prefix, e.eos_tokens, e.audio["rows"]) == ((1, 3), frozenset({2, 4}), 375)
    assert t.identity == e.identity == b.identity  # one bundle, one identity
    assert b.present() == ["tiny/vocab.bin", "tiny/tokenizer_config.json", "ears/vocab.bin"]  # weights listed, not here
    with pytest.raises(BundleError):
        b.model("other")


def test_each_model_reads_its_own_files(tiny):
    b = Bundle(tiny)
    assert b.model("tiny").vocab[1] == b"a" and b.model("ears").vocab[1] == TOKENS[-2]
    assert b.model("tiny").chat_template.startswith("{{ bos_token }}")
    with pytest.raises(BundleError):
        b.model("ears").chat_template  # ears has no template
    with pytest.raises(BundleError):
        b.read("tiny/experts.q4g128.bin")  # absent
    (tiny / "tiny/vocab.bin").write_bytes(b"x" * 40)
    with pytest.raises(BundleError):
        Bundle(tiny).model("tiny").vocab  # hash


def test_manifest_rejects_the_malformed(tiny):
    files = {}
    with pytest.raises(BundleError):
        Manifest.parse(b'[model]\nname = "tiny"\nbos = 1\neos = 2\nvocab_size = 3\n[files]\n')  # one table, not a list
    with pytest.raises(BundleError):
        Manifest.parse(manifest(files).replace(b"vocab_size = 8\n\n[[model]]", b"\n\n[[model]]"))  # a model without its constants
    with pytest.raises(BundleError):
        Manifest.parse(manifest(files, '[[model]]\nname = "tiny"\nbos = 1\neos = 2\nvocab_size = 3\n\n'))  # a name twice
    with pytest.raises(BundleError):
        Manifest.parse(manifest(files, f'[[model]]\nname = "{"x" * 33}"\nbos = 1\neos = 2\nvocab_size = 3\n\n'))  # longer than the hello's field
    with pytest.raises(BundleError):
        Bundle(tiny / "nope")


def test_vocab_and_streaming_decode(tiny):
    v = Bundle(tiny).model("tiny").vocab
    assert len(v) == 8 and v[0] == b"" and v[1] == b"a" and v[6] == b"<bos>"
    d = v.decoder()
    assert [d.feed(i) for i in (1, 3, 4, 5, 2)] == ["a", "é", "", "é", "b"]
    assert d.feed(4) == "" and d.flush() == "�"
    with pytest.raises(BundleError):
        Vocab(vocab_bin(TOKENS)[:-1], 8)
    with pytest.raises(BundleError):
        Vocab(b"\x00" * 8, 8)


@pytest.mark.skipif(not (GLM / "manifest.toml").exists() or b"[[model]]" not in (GLM / "manifest.toml").read_bytes(), reason="no GLM bundle with a [[model]] list on this host")
def test_real_glm_bundle():
    m = Bundle(GLM).model("glm-5.3-flash")
    assert m.recipe.startswith("experts:q4g128") and m.layers == (3, 45)
    assert m.prefix == (154822, 154824) and m.eos_tokens == frozenset({154820, 154827, 154829}) and m.vocab_size == 154880
    v = m.vocab
    ids = m.tokenizer.encode("[gMASK]<sop>Hello, wörld — 你好").ids
    assert ids[:2] == list(m.prefix)  # the template writes the prefix as text; it encodes to the wire's prefix
    d = v.decoder()
    assert "".join(d.feed(i) for i in ids[2:]) + d.flush() == "Hello, wörld — 你好"
    assert "<|assistant|>" in m.chat_template
