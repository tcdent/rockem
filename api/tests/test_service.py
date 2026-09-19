import asyncio
import json

import pytest
from rockem import core
from rockem.edge import Stops

from conftest import BOS, EOS

pytestmark = pytest.mark.anyio


def test_stops_hold_back_a_possible_prefix():
    s = Stops(["User:", "\n\n"])
    assert s.feed("Paris. Us") == "Paris. "
    assert s.feed("er") == ""
    assert s.feed(": what") == "" and s.matched == "User:"
    assert s.feed("more") == "" and s.flush() == ""
    s = Stops(["User:"])
    assert s.feed("Us") == "" and s.feed("ual") == "Usual" and s.flush() == ""
    s = Stops([])
    assert s.feed("x") == "x" and s.matched is None
    s = Stops(["ab", "abc"])
    assert s.feed("xab") == "x" and s.matched == "ab"


def test_edge_renders_and_encodes(engine):
    e = engine.edge
    assert e.render([{"role": "user", "content": "hello world"}]) == "<bos>hello world "
    assert e.encode_chat([{"role": "user", "content": "hello world"}]) == [BOS, 0, 1]
    assert e.encode("paris") == [2]
    d = e.decoder()
    assert d.feed(0) + d.feed(2) == " hello paris"


async def test_engine_streams_to_eos(engine):
    run = await engine.start(core.Request(messages=[{"role": "user", "content": "hello"}]))
    assert run.prompt_tokens == 1
    events = [ev async for ev in run.events()]
    assert events[:-1] == [core.Delta(" paris", [2]), core.Delta(" stop", [3])]
    done = events[-1]
    assert (done.reason, done.stop_sequence, done.prompt_tokens, done.completion_tokens, done.positions) == (core.Reason.EOS, None, 1, 3, 11)
    assert engine.scheduler.free == 1


async def test_engine_stop_sequence_max_tokens_and_context(engine, node):
    text, ids, done = await (await engine.start(core.Request(prompt="hello", stop=["stop"]))).fold()
    assert (text, ids, done.reason, done.stop_sequence) == (" paris ", [2, 3], core.Reason.STOP_SEQUENCE, "stop")
    text, ids, done = await (await engine.start(core.Request(prompt="hello", max_tokens=1))).fold()
    assert (text, ids, done.reason, done.completion_tokens) == (" paris", [2], core.Reason.MAX_TOKENS, 1)
    node.draws = [4] * 100
    text, ids, done = await (await engine.start(core.Request(prompt="hello world", max_tokens=None))).fold()
    assert done.reason is core.Reason.CONTEXT_FULL and done.completion_tokens == 11 - 2 + 1 and len(ids) == 10
    with pytest.raises(core.ContextFull):
        await engine.start(core.Request(prompt=" ".join(["hello"] * 12)))
    assert engine.scheduler.free == 1


async def test_engine_abort_frees_the_node(engine, node):
    node.draws = [4] * 100
    run = await engine.start(core.Request(prompt="hello"))
    assert engine.scheduler.free == 0
    events = run.events()
    assert isinstance(await anext(events), core.Delta)
    await events.aclose()
    assert engine.scheduler.free == 1
    text, _, done = await (await engine.start(core.Request(prompt="hello", max_tokens=2))).fold()
    assert text == " x x"


async def test_scheduler_serializes_in_arrival_order(engine, node):
    order = []

    async def one(tag):
        run = await engine.start(core.Request(prompt="hello"))
        order.append(tag)
        await run.fold()

    await asyncio.gather(one("a"), one("b"), one("c"))
    assert order == ["a", "b", "c"] and node.sessions == 3


async def test_internal_adapter(client):
    r = await client.post("/v1/generate", json={"messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 200, r.text
    b = r.json()
    assert (b["text"], b["ids"], b["reason"], b["model"]) == (" paris stop", [2, 3], "eos", "tiny")
    assert b["usage"] == {"prompt_tokens": 1, "completion_tokens": 3, "positions": 11}
    r = await client.post("/v1/generate", json={"prompt": "hello", "stream": True, "stop": ["stop"]})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [(e.split("\n")[0], json.loads(e.split("\n")[1][6:])) for e in r.text.strip().split("\n\n")]
    assert events[0] == ("event: delta", {"text": " pari", "ids": [2]})  # the "s" could begin "stop", so it waits
    assert events[1] == ("event: delta", {"text": "s ", "ids": [3]})
    assert events[-1][0] == "event: done" and events[-1][1]["reason"] == "stop_sequence" and events[-1][1]["stop_sequence"] == "stop"
    assert (await client.post("/v1/generate", json={"prompt": "x", "messages": []})).status_code == 422
    assert (await client.post("/v1/generate", json={"prompt": " ".join(["hello"] * 12)})).status_code == 413


async def test_openai_adapter(client):
    body = {"model": "whatever", "messages": [{"role": "user", "content": "hello"}], "temperature": 0.9, "max_tokens": 5}
    r = await client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["object"] == "chat.completion" and b["model"] == "tiny"
    assert b["choices"][0]["message"] == {"role": "assistant", "content": " paris stop"} and b["choices"][0]["finish_reason"] == "stop"
    assert b["usage"] == {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4}
    r = await client.post("/v1/chat/completions", json={**body, "stream": True, "stream_options": {"include_usage": True}})
    lines = [l for l in r.text.split("\n") if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == " paris stop"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop" and chunks[-1]["usage"]["total_tokens"] == 4
    r = await client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": [{"type": "image_url"}]}]})
    assert r.status_code == 400


async def test_anthropic_adapter(client):
    body = {"model": "m", "max_tokens": 1, "system": "hello", "messages": [{"role": "user", "content": [{"type": "text", "text": "world"}]}]}
    r = await client.post("/v1/messages", json=body)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["type"] == "message" and b["content"] == [{"type": "text", "text": " paris"}] and b["stop_reason"] == "max_tokens"
    assert b["usage"] == {"input_tokens": 2, "output_tokens": 1}
    r = await client.post("/v1/messages", json={**body, "stream": True, "max_tokens": 10, "stop_sequences": ["stop"]})
    names = [l[7:] for l in r.text.split("\n") if l.startswith("event: ")]
    assert names[:2] == ["message_start", "content_block_start"] and names[-3:] == ["content_block_stop", "message_delta", "message_stop"]
    assert set(names[2:-3]) == {"content_block_delta"} and len(names) > 5
    delta = [json.loads(l[6:]) for l in r.text.split("\n") if l.startswith("data: ")][-2]
    assert delta["delta"] == {"stop_reason": "stop_sequence", "stop_sequence": "stop"}
    assert (await client.post("/v1/messages", json={"model": "m", "messages": []})).status_code == 422


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200 and r.json()["nodes"] == 1 and r.json()["free"] == 1 and r.json()["model"] == "tiny"
