"""The one-render path against the many-render one it replaced, per template we serve: the noted
message starts equal the per-prefix renders' lengths at every prefix, and Engine.ids equals the
per-turn algorithm (kept here as the reference) with remembered turns."""

import json
from pathlib import Path

import pytest
from rockem import calls, core
from rockem.edge import Edge, template_env

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = {
    "glm-5.3-flash": Path(__file__).parent / "fixtures/glm-5.3-flash.chat_template.jinja",
    "deepseek-v2-lite": ROOT / "models/deepseek-v2-lite/chat_template.jinja",
}
TOOLS = [{"type": "function", "function": {"name": "shell", "description": "Run a command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]


def conversation():
    msgs = [{"role": "system", "content": "You are Codey."}, {"role": "user", "content": "List the files."}]
    for i in range(6):
        msgs.append({"role": "assistant", "content": f"Looking, step {i}.", "reasoning_content": f"Reason {i}: maybe ls.", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "shell", "arguments": {"command": f"ls dir{i}"}}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"file_{i}.rs\nfile_{i}b.rs"})
        if i % 2:
            msgs.append({"role": "user", "content": f"And then {i}?"})
    msgs.append({"role": "assistant", "content": "Done: twelve files.", "reasoning_content": "All listed."})
    msgs.append({"role": "user", "content": "Thanks."})
    return msgs


class FakeModel:
    """Just enough of a bundle model for Edge: the template, a vocabulary for BOS/EOS text, and
    a stand-in tokenizer (each character one id), so the ids test runs without a bundle."""

    def __init__(self, name, template):
        self.name, self.chat_template, self.bos, self.eos = name, template, 0, 1
        self.vocab = [b"<B>", b"<E>"]

        class Enc:
            def __init__(self, ids):
                self.ids = ids

        class Tok:
            def encode(self, text):
                return Enc([ord(c) for c in text])

            def encode_batch(self, texts):
                return [self.encode(t) for t in texts]

        self.tokenizer = Tok()


@pytest.fixture(params=list(TEMPLATES))
def edge(request):
    return Edge(FakeModel(request.param, TEMPLATES[request.param].read_text()))


def test_noted_starts_equal_the_per_prefix_renders(edge):
    msgs = conversation()
    full, starts = edge.render_noted(msgs, TOOLS)
    assert full == edge.render(msgs, TOOLS)
    for i in range(len(msgs) + 1):
        assert edge.render(msgs[:i], TOOLS, add_generation_prompt=False) == full[: (starts + [len(full) - len(edge.generation_prompt)])[i]], i
    assert full.endswith(edge.generation_prompt)


def per_turn_ids(engine, msgs, tools):
    """The path this replaced (core.Engine.ids before 2026-10-25), verbatim in effect."""
    e, edge = None, engine.edge
    full = edge.render(msgs, tools, reasoning_effort=e)
    out, done = [], ""
    for i, m in enumerate(msgs):
        content = m.get("content") if m.get("role") == "assistant" else None
        ids = engine.replies.get((m.get("reasoning_content") or "", content)) if isinstance(content, str) else None
        if ids is None:
            continue
        before = edge.render(msgs[:i], tools, add_generation_prompt=False, reasoning_effort=e)
        opened = edge.render(msgs[:i], tools, add_generation_prompt=True, reasoning_effort=e)
        after = edge.render(msgs[: i + 1], tools, add_generation_prompt=False, reasoning_effort=e)
        if not (full.startswith(after) and after.startswith(before) and opened.startswith(before) and len(before) >= len(done)):
            continue
        out.extend(edge.encode(before[len(done):] + opened[len(before):]))
        out.extend(ids)
        done = after
    out.extend(edge.encode(full[len(done):]))
    return out


def test_ids_equal_the_per_turn_path(edge, monkeypatch):
    msgs = conversation()
    engine = core.Engine(edge, None)
    for k, m in enumerate(m for m in msgs if m["role"] == "assistant"):
        if k % 2 == 0:  # some turns remembered, some not
            engine.remember(m["reasoning_content"], m["content"], [900_000 + k] * (k + 3))
    req = core.Request(messages=msgs, tools=TOOLS, reasoning_effort=None)
    assert engine.ids(req) == per_turn_ids(engine, msgs, TOOLS)
    engine.replies.clear()
    assert engine.ids(req) == per_turn_ids(engine, msgs, TOOLS)  # nothing remembered: one encode of the text


GLM = ROOT / "bundles/glm-5.3-flash-q4rtn"


@pytest.mark.skipif(not (GLM / "manifest.toml").exists(), reason="no GLM bundle on this host")
def test_ids_equal_the_per_turn_path_with_glms_tokenizer():
    """With the real tokenizer the text is also cut before every special-token turn marker."""
    from rockem.bundle import Bundle

    edge = Edge(Bundle(GLM).model("glm-5.3-flash"))
    assert "<|assistant|>" in edge.split_marks and "<|observation|>" in edge.split_marks
    msgs = conversation()
    engine = core.Engine(edge, None)
    for k, m in enumerate(m for m in msgs if m["role"] == "assistant"):
        if k % 2:
            engine.remember(m["reasoning_content"], m["content"], edge.encode("<think>" + m["reasoning_content"] + "</think>" + m["content"]))
    req = core.Request(messages=msgs, tools=TOOLS, reasoning_effort=None)
    assert engine.ids(req) == per_turn_ids(engine, msgs, TOOLS)
