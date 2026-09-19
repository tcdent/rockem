"""A bundle, read from the edge (docs/bundle.md).

The edge opens the manifest, the vocabulary and the tokenizer files, and
never the weights; a bundle directory here may hold only those, with the
weight files listed in the manifest but absent. Every file opened is checked
against the manifest's sha256. The identity is FNV-1a over the manifest's
bytes, computed here exactly as the node computes it, and never stored.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import struct
import tomllib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

MANIFEST = "manifest.toml"
VOCAB_FILE = "vocab.bin"
TOKENIZER_FILE = "tokenizer.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

FNV_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def identity(manifest: bytes) -> int:
    """FNV-1a, 64-bit, over the manifest's bytes exactly as written."""
    h = FNV_BASIS
    for b in manifest:
        h = ((h ^ b) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


class BundleError(Exception):
    """The directory is not a bundle, or a file in it is not the one the manifest names."""


@dataclass(frozen=True)
class Checkpoint:
    repo: str
    revision: str
    pinned: str


@dataclass(frozen=True)
class Manifest:
    """`manifest.toml`, parsed. `files` maps a name to its sha256 hex; `raw` is the bytes the identity covers."""

    raw: bytes
    model: str
    recipe: str
    calibration: str | None
    record_bytes: int
    experts: int
    layers: tuple[int, int]
    bos: int
    eos: int
    vocab_size: int
    files: dict[str, str]
    checkpoint: Checkpoint | None

    @classmethod
    def parse(cls, raw: bytes) -> Manifest:
        try:
            t = tomllib.loads(raw.decode())
            files = {name: cls._sha(v) for name, v in t["files"].items()}
            m = t["model"]
            return cls(
                raw=raw,
                model=m["name"],
                recipe=t["recipe"],
                calibration=t.get("calibration"),
                record_bytes=t["record_bytes"],
                experts=t["experts"],
                layers=(t["layers"][0], t["layers"][1]),
                bos=m["bos"],
                eos=m["eos"],
                vocab_size=m["vocab_size"],
                files=files,
                checkpoint=cls._checkpoint(raw),
            )
        except (tomllib.TOMLDecodeError, KeyError, IndexError, TypeError, UnicodeDecodeError) as e:
            raise BundleError(f"manifest: {e!r}") from e

    @property
    def identity(self) -> int:
        return identity(self.raw)

    @staticmethod
    def _sha(v: str) -> str:
        if not v.startswith("sha256:"):
            raise KeyError(f"file hash {v!r}")
        return v[len("sha256:") :]

    @staticmethod
    def _checkpoint(raw: bytes) -> Checkpoint | None:
        # the model's checkpoint.toml, embedded verbatim as comments after `# checkpoint.toml`
        lines = raw.decode().splitlines()
        try:
            start = lines.index("# checkpoint.toml") + 1
        except ValueError:
            return None
        body = "\n".join(l[2:] for l in lines[start:] if l.startswith("# "))
        try:
            c = tomllib.loads(body)
            return Checkpoint(repo=c["repo"], revision=c["revision"], pinned=str(c["pinned"]))
        except (tomllib.TOMLDecodeError, KeyError):
            return None


class Vocab:
    """`vocab.bin`: token bytes by id. `[u32; vocab_size + 1]` offsets, then the bytes."""

    def __init__(self, data: bytes, vocab_size: int):
        n = vocab_size + 1
        if len(data) < 4 * n:
            raise BundleError(f"{VOCAB_FILE}: {len(data)} bytes, offsets alone need {4 * n}")
        self._off = struct.unpack_from(f"<{n}I", data)
        self._bytes = data[4 * n :]
        if self._off[-1] != len(self._bytes):
            raise BundleError(f"{VOCAB_FILE}: last offset {self._off[-1]} != {len(self._bytes)} bytes of tokens")
        self.size = vocab_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, id: int) -> bytes:
        return self._bytes[self._off[id] : self._off[id + 1]]

    def decoder(self) -> Decoder:
        return Decoder(self)


class Decoder:
    """Ids in, text out, streamed: a token can end inside a multi-byte character, so
    what comes back for an id is the text completed by its bytes, possibly empty."""

    def __init__(self, vocab: Vocab):
        self._vocab = vocab
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, id: int) -> str:
        return self._utf8.decode(self._vocab[id])

    def flush(self) -> str:
        return self._utf8.decode(b"", final=True)


class Bundle:
    """A bundle directory. Opening it reads and parses the manifest; each other file is
    read on first use and checked against the manifest's hash."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            raw = (self.path / MANIFEST).read_bytes()
        except OSError as e:
            raise BundleError(f"{self.path}: {e}") from e
        self.manifest = Manifest.parse(raw)

    @property
    def identity(self) -> int:
        return self.manifest.identity

    @property
    def bos(self) -> int:
        return self.manifest.bos

    @property
    def eos(self) -> int:
        return self.manifest.eos

    @property
    def vocab_size(self) -> int:
        return self.manifest.vocab_size

    def present(self) -> list[str]:
        """The listed files that are in the directory, in manifest order."""
        return [f for f in self.manifest.files if (self.path / f).exists()]

    def read(self, name: str) -> bytes:
        """A listed file's bytes, verified against the manifest."""
        if name not in self.manifest.files:
            raise BundleError(f"{name}: not in the manifest")
        try:
            data = (self.path / name).read_bytes()
        except OSError as e:
            raise BundleError(f"{name}: {e}") from e
        want = self.manifest.files[name]
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise BundleError(f"{name}: sha256 {got} != manifest's {want}")
        return data

    @cached_property
    def vocab(self) -> Vocab:
        return Vocab(self.read(VOCAB_FILE), self.vocab_size)

    @cached_property
    def tokenizer(self):
        """The checkpoint's tokenizer, `tokenizers.Tokenizer`, from `tokenizer.json`. Encoding
        the rendered chat template with it yields BOS as the first id."""
        from tokenizers import Tokenizer

        return Tokenizer.from_str(self.read(TOKENIZER_FILE).decode())

    @cached_property
    def chat_template(self) -> str:
        """The Jinja source of the checkpoint's chat template, from `tokenizer_config.json`."""
        try:
            return json.loads(self.read(TOKENIZER_CONFIG_FILE))["chat_template"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise BundleError(f"{TOKENIZER_CONFIG_FILE}: {e!r}") from e
