import asyncio
import json

import pytest
from rockem import core
from rockem.edge import Stops

from conftest import BOS, CALL_BODY, CALL_END, CALL_START, EOS, NEXT_TURN, THINK_CLOSE, THINK_OPEN

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
    assert isinstance(events.pop(0), core.Started)
    assert events[:-1] == [core.Delta(" paris", [2]), core.Delta(" stop", [3])]
    done = events[-1]
    assert (done.reason, done.stop_sequence, done.prompt_tokens, done.completion_tokens, done.positions) == (core.Reason.END_TURN, None, 1, 3, 11)
    assert engine.scheduler.free == 1


async def test_engine_stop_sequence_max_tokens_and_context(engine, node):
    text, ids, _, done = await (await engine.start(core.Request(prompt="hello", stop=["stop"]))).fold()
    assert (text, ids, done.reason, done.stop_sequence) == (" paris ", [2, 3], core.Reason.STOP_SEQUENCE, "stop")
    text, ids, _, done = await (await engine.start(core.Request(prompt="hello", max_tokens=1))).fold()
    assert (text, ids, done.reason, done.completion_tokens) == (" paris", [2], core.Reason.MAX_TOKENS, 1)
    node.draws = [4] * 100
    text, ids, _, done = await (await engine.start(core.Request(prompt="hello world", max_tokens=None))).fold()
    assert done.reason is core.Reason.CONTEXT_FULL and done.completion_tokens == 11 - 2 + 1 and len(ids) == 10
    with pytest.raises(core.ContextFull):
        await engine.start(core.Request(prompt=" ".join(["hello"] * 12)))
    assert engine.scheduler.free == 1


async def test_engine_abort_frees_the_node(engine, node):
    node.draws = [4] * 100
    run = await engine.start(core.Request(prompt="hello"))
    assert engine.scheduler.free == 0
    events = run.events()
    assert isinstance(await anext(events), core.Started)
    assert isinstance(await anext(events), core.Delta)
    await events.aclose()
    assert engine.scheduler.free == 1
    text, _, _, done = await (await engine.start(core.Request(prompt="hello", max_tokens=2))).fold()
    assert text == " x x"


async def test_scheduler_serializes_in_arrival_order(engine, node):
    order = []

    async def one(tag):
        run = await engine.start(core.Request(prompt="hello"))
        order.append(tag)
        await run.fold()

    await asyncio.gather(one("a"), one("b"), one("c"))
    assert order == ["a", "b", "c"] and node.sessions == 3


def data_events(r) -> list[dict]:
    """The events of a rockem stream: one `data: {json}` line each, blank-line separated."""
    assert r.headers["content-type"].startswith("text/event-stream")
    chunks = [c for c in r.text.split("\n\n") if c]
    assert all(c.startswith("data: ") and "\n" not in c for c in chunks), chunks
    return [json.loads(c[6:]) for c in chunks]


async def test_internal_adapter(client):
    r = await client.post("/rockem/v1/generate", json={"messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 200, r.text
    evs = data_events(r)
    assert [e["type"] for e in evs] == ["started", "text", "done"]  # the two tokens arrive within one flush
    assert evs[0] == {"type": "started", "prompt_tokens": 1}
    assert "".join(e["text"] for e in evs if e["type"] == "text") == " paris stop"
    done = evs[-1]
    assert (done["reason"], done["model"], done["stop_sequence"]) == ("end_turn", "tiny", None)
    assert done["usage"] == {"prompt_tokens": 1, "completion_tokens": 3, "positions": 11}
    r = await client.post("/rockem/v1/generate", json={"prompt": "hello", "stop": ["stop"]})
    evs = data_events(r)[1:]
    assert evs[0] == {"type": "text", "text": " paris ", "ids": [2, 3]}  # held "s" (could begin "stop") and the rest, one flush
    assert evs[-1]["reason"] == "stop_sequence" and evs[-1]["stop_sequence"] == "stop"
    assert (await client.post("/rockem/v1/generate", json={"prompt": "x", "messages": []})).status_code == 422
    assert (await client.post("/rockem/v1/generate", json={"prompt": " ".join(["hello"] * 12)})).status_code == 413


async def test_internal_adapter_reports_a_broken_generation(client, node):
    node.draws = [2] * 100
    node.close_after = 1  # the node goes away after one id
    evs = data_events(await client.post("/rockem/v1/generate", json={"prompt": "hello"}))
    assert [e["type"] for e in evs] == ["started", "text", "error"] and "node" in evs[-1]["message"]


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


async def test_engine_tool_call_then_turn_marker(engine, node):
    tools = [{"type": "function", "function": {"name": "read", "description": "", "parameters": {"type": "object"}}}]
    node.draws = [2, CALL_START, CALL_BODY, CALL_END, NEXT_TURN, 4]
    run = await engine.start(core.Request(messages=[{"role": "user", "content": "hello"}], tools=tools))
    events = [ev async for ev in run.events()]
    assert isinstance(events.pop(0), core.Started)
    assert events[0] == core.Delta(" paris", [2])
    call = events[1]
    assert isinstance(call, core.ToolCall) and (call.name, call.arguments, call.ids) == ("read", {"path": "a"}, [CALL_START, CALL_BODY, CALL_END])
    assert events[-1].reason is core.Reason.TOOL_USE and events[-1].stop_sequence is None
    assert not any(isinstance(e, core.Delta) and "User" in e.text for e in events)


async def test_engine_without_tools_leaves_calls_as_text_and_ends_at_the_turn(engine, node):
    node.draws = [CALL_START, CALL_BODY, CALL_END, NEXT_TURN]
    text, _, calls, done = await (await engine.start(core.Request(messages=[{"role": "user", "content": "hello"}]))).fold()
    assert calls == [] and text.startswith("<tool_call>") and done.reason is core.Reason.END_TURN
    # a raw prompt has no turns: the marker is just text
    node.draws = [NEXT_TURN, 2]
    text, _, _, done = await (await engine.start(core.Request(prompt="hello"))).fold()
    assert text == "\nUser: paris" and done.reason is core.Reason.END_TURN


async def test_engine_unfinished_call_comes_back_as_text(engine, node):
    tools = [{"type": "function", "function": {"name": "read", "parameters": {}}}]
    node.draws = [CALL_START, CALL_BODY]
    text, _, calls, done = await (await engine.start(core.Request(messages=[{"role": "user", "content": "hello"}], tools=tools, max_tokens=2))).fold()
    assert calls == [] and text == '<tool_call>{"name": "read", "arguments": {"path": "a"}}' and done.reason is core.Reason.MAX_TOKENS


async def test_internal_adapter_tools_round_trip(client, node):
    node.draws = [CALL_START, CALL_BODY, CALL_END, EOS]
    body = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "name": "read", "arguments": {"path": "b"}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "world"},
        ],
        "tools": [{"name": "read", "description": "read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}],
    }
    r = await client.post("/rockem/v1/generate", json=body)
    assert r.status_code == 200, r.text
    evs = data_events(r)
    assert [e["type"] for e in evs] == ["started", "tool_call", "done"] and evs[-1]["reason"] == "tool_use"
    call = evs[1]
    assert (call["name"], call["arguments"]) == ("read", {"path": "a"}) and call["id"].startswith("call_")
    bad = {"messages": [{"role": "user", "content": "x", "tool_call_id": "c"}]}
    assert (await client.post("/rockem/v1/generate", json=bad)).status_code == 422


async def test_engine_text_after_calls_ends_the_turn(engine, node):
    # V2-Lite skips EOS and writes the next turn straight after its call
    tools = [{"type": "function", "function": {"name": "read", "parameters": {}}}]
    node.draws = [2, CALL_START, CALL_BODY, CALL_END, CALL_START, CALL_BODY, CALL_END, 3, 4]
    run = await engine.start(core.Request(messages=[{"role": "user", "content": "hello"}], tools=tools))
    events = [ev async for ev in run.events()]
    assert isinstance(events.pop(0), core.Started)
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["Delta", "ToolCall", "ToolCall", "Delta", "Done"], kinds
    assert events[0].text == " paris" and events[3].text == "" and events[3].ids == [3]
    assert events[-1].reason is core.Reason.TOOL_USE


@pytest.fixture
def thinking_engine(engine):
    import dataclasses

    engine.edge.format = dataclasses.replace(engine.edge.format, think=("<think>", "</think>"))
    return engine


def contiguous(seq, part):
    return any(seq[i : i + len(part)] == part for i in range(len(seq)))


async def test_reasoning_is_its_own_event_and_never_a_call(thinking_engine, node):
    e = thinking_engine
    node.draws = [THINK_OPEN, 0, THINK_CLOSE, 2, EOS]
    run = await e.start(core.Request(messages=[{"role": "user", "content": "hello"}]))
    events = [ev async for ev in run.events()]
    assert events[:3] == [core.Started(1), core.Reasoning(" hello", [THINK_OPEN, 0]), core.Delta(" paris", [THINK_CLOSE, 2])]
    assert events[-1].reason is core.Reason.END_TURN
    # a call written while reasoning is reasoning, not a call
    tools = [{"type": "function", "function": {"name": "read", "parameters": {}}}]
    node.draws = [THINK_OPEN, CALL_START, CALL_BODY, CALL_END, THINK_CLOSE, 2, EOS]
    run = await e.start(core.Request(messages=[{"role": "user", "content": "hello"}], tools=tools))
    events = [ev async for ev in run.events()]
    assert not any(isinstance(ev, core.ToolCall) for ev in events) and events[-1].reason is core.Reason.END_TURN
    assert "".join(ev.text for ev in events if isinstance(ev, core.Reasoning)).startswith("<tool_call>")


async def test_a_reply_sent_back_with_its_reasoning_is_rendered_from_its_ids(thinking_engine, node):
    e = thinking_engine
    node.draws = [THINK_OPEN, 0, THINK_CLOSE, 2, EOS]
    await (await e.start(core.Request(messages=[{"role": "user", "content": "hello"}]))).fold()
    generated = [THINK_OPEN, 0, THINK_CLOSE, 2]
    turn = [{"role": "user", "content": "hello"}, None, {"role": "user", "content": "world"}]
    turn[1] = {"role": "assistant", "content": " paris", "reasoning_content": " hello"}
    assert contiguous(e.ids(core.Request(messages=turn)), generated)
    turn[1] = {"role": "assistant", "content": " paris"}  # reasoning dropped: not the turn that was generated
    assert not contiguous(e.ids(core.Request(messages=turn)), generated)


async def test_internal_adapter_thinking(client, engine, node):
    import dataclasses

    engine.edge.format = dataclasses.replace(engine.edge.format, think=("<think>", "</think>"))
    node.draws = [THINK_OPEN, 0, THINK_CLOSE, 2, EOS]
    evs = data_events(await client.post("/rockem/v1/generate", json={"messages": [{"role": "user", "content": "hello"}]}))
    assert [e["type"] for e in evs] == ["started", "thinking", "text", "done"]
    assert (evs[1]["text"], evs[2]["text"]) == (" hello", " paris")
    body = {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": " paris", "thinking": " hello"}, {"role": "user", "content": "x"}]}
    assert (await client.post("/rockem/v1/generate", json=body)).status_code == 200
    bad = {"messages": [{"role": "user", "content": "x", "thinking": "no"}]}
    assert (await client.post("/rockem/v1/generate", json=bad)).status_code == 422


async def test_internal_adapter_flushes_by_time(client, node, monkeypatch):
    from rockem.adapters import internal

    monkeypatch.setattr(internal, "FLUSH_SECONDS", 0.05)
    node.draws, node.delay = [2, 3, 4, 2], 0.03  # a token every 30 ms: a flush every second or third token
    evs = data_events(await client.post("/rockem/v1/generate", json={"prompt": "hello"}))
    texts = [e for e in evs if e["type"] == "text"]
    assert 1 < len(texts) < 4 and [i for e in texts for i in e["ids"]] == [2, 3, 4, 2]
    assert "".join(e["text"] for e in texts) == " paris stop x paris"


async def test_reasoning_effort_defaults_low_and_is_checked(client, engine, monkeypatch):
    seen = []
    render = engine.edge.render
    monkeypatch.setattr(engine.edge, "render", lambda *a, **k: (seen.append(k.get("reasoning_effort")), render(*a, **k))[1])
    await client.post("/rockem/v1/generate", json={"messages": [{"role": "user", "content": "hello"}]})
    await client.post("/rockem/v1/generate", json={"messages": [{"role": "user", "content": "hello"}], "reasoning_effort": "max"})
    assert seen[0] == "low" and seen[-1] == "max"
    bad = {"messages": [{"role": "user", "content": "hello"}], "reasoning_effort": "medium"}
    assert (await client.post("/rockem/v1/generate", json=bad)).status_code == 422


async def test_a_non_streaming_fold_aborts_when_the_client_leaves(engine, node):
    import asyncio

    from fastapi import HTTPException
    from rockem.views import fold_or_abort

    node.draws, node.delay = [2] * 50, 0.05  # a slow generation: 2.5 s if nobody stops it
    left = asyncio.Event()

    async def receive():
        await left.wait()
        return {"type": "http.disconnect"}

    run = await engine.start(core.Request(prompt="hello"))
    task = asyncio.ensure_future(fold_or_abort(run, receive))
    await asyncio.sleep(0.12)
    assert engine.scheduler.free == 0
    left.set()  # the client goes away
    with pytest.raises(HTTPException):
        await task
    assert engine.scheduler.free == 1  # the node connection closed: the abort, at once
    node.delay = 0

    async def stays():
        await asyncio.Event().wait()

    text, _, _, done = await fold_or_abort(await engine.start(core.Request(prompt="hello", max_tokens=1)), stays)
    assert done.reason is core.Reason.MAX_TOKENS  # a fold that finishes first returns as before
