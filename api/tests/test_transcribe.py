"""The transcribe step end to end: audio bytes → windows and Voxtral's prompt → AUDIO on the
wire → text events. The spectrogram itself was checked against transformers' Whisper feature
extractor and mistral-common's loader (max |diff| 1.2e-7, 2026-10-23); here, the plumbing."""

import base64
import json

import pytest
from rockem import wire
from rockem.wire import AudioSpec, Header, Session, WireError

from conftest import E_AUDIO, E_BEGIN, E_BOS, E_END_INST, E_EOS, E_INST, E_TRANSCRIBE, EAR_VOCAB, IDENTITY, wav

pytestmark = pytest.mark.anyio


def events(r) -> list[dict]:
    assert r.headers["content-type"].startswith("text/event-stream"), r.text
    return [json.loads(c[6:]) for c in r.text.split("\n\n") if c.startswith("data: ")]


def test_the_prompt_and_windows(transcribing):
    req = transcribing.request(wav(2.5))  # 2.5 s → padded to three 1 s windows
    assert len(req.windows) == 3 and all(len(w) == 4 * 4 * 100 for w in req.windows)
    assert req.ids == [E_BOS, E_INST, E_BEGIN] + [E_AUDIO] * 9 + [E_END_INST, E_TRANSCRIBE]
    with_lang = transcribing.request(wav(0.5), language="en").ids
    assert with_lang[:3] == [E_BOS, E_INST, E_BEGIN] and with_lang[-1] == E_TRANSCRIBE and len(with_lang) == 3 + 3 + 1 + 3 + 1
    with pytest.raises(ValueError):
        transcribing.request(wav(0.5), language="English")


async def test_transcribe_end_to_end(client, transcribing, node):
    node.draws, node.max_ctx = [2, 3], 64  # " heard you", then the model's end; room for 13 prompt ids
    r = await client.post("/rockem/v1/transcribe", json={"audio": base64.b64encode(wav(2.5)).decode()})
    evs = events(r)
    assert [e["type"] for e in evs] == ["started", "text", "done"]
    assert evs[1]["text"] == " heard you" and evs[-1]["reason"] == "end_turn" and evs[-1]["model"] == "ears"
    assert node.hellos[-1] == "ears"
    assert node.windows == transcribing.request(wav(2.5)).windows  # the node got the edge's windows, in order
    assert node.transcripts[-1][:14] == [E_INST, E_BEGIN] + [E_AUDIO] * 9 + [E_END_INST, E_TRANSCRIBE] + [2]


async def test_transcribe_refusals(client, transcribing, engine):
    assert (await client.post("/rockem/v1/transcribe", json={"audio": "not base64!"})).status_code == 400
    assert (await client.post("/rockem/v1/transcribe", json={"audio": base64.b64encode(b"just bytes").decode()})).status_code == 400
    r = await client.post("/rockem/v1/transcribe", json={"audio": base64.b64encode(wav(0.5)).decode(), "language": "English"})
    assert r.status_code == 400
    # the chat model and the transcription model wait in the same line: one scheduler
    assert transcribing.engine.scheduler is engine.scheduler


async def test_no_transcription_model_is_a_503(client):
    assert (await client.post("/rockem/v1/transcribe", json={"audio": ""})).status_code == 503


def test_session_counts_rows_against_placeholders():
    spec = AudioSpec(placeholder=E_AUDIO, window_bytes=16, rows=2)
    s = Session(Header(IDENTITY, 100), E_BOS, E_EOS, EAR_VOCAB, spec)
    with pytest.raises(WireError):
        s.audio(b"\0" * 16)  # before BOS
    s.start()
    with pytest.raises(WireError):
        s.audio(b"\0" * 15)  # not the model's window size
    assert s.audio(b"\0" * 16)[:4] == wire.AUDIO.to_bytes(4, "little")
    with pytest.raises(WireError):
        s.generate()  # two rows not taken
    s.feed([E_AUDIO])
    with pytest.raises(WireError):
        s.score()  # one row still queued
    s.feed([E_AUDIO])
    with pytest.raises(WireError):
        s.feed([E_AUDIO])  # a placeholder with no row
    s.generate()
    plain = Session(Header(IDENTITY, 100), E_BOS, E_EOS, EAR_VOCAB)
    plain.start()
    with pytest.raises(WireError):
        plain.audio(b"\0" * 16)  # a model without an encoder
