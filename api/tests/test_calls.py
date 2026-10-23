"""Formats against their own templates: a call the template renders, the parser returns."""

from pathlib import Path

from rockem.calls import FORMATS, Call, Parser, Text
from rockem.edge import template_env

ROOT = Path(__file__).resolve().parents[2]
TOOLS = [{"type": "function", "function": {"name": "edit_file", "description": "Edit a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "edits": {"type": "array"}}}}}]
CALLS = [
    {"id": "c1", "type": "function", "function": {"name": "edit_file", "arguments": {"path": "src/a.rs", "edits": [{"old": "</tool_call>", "new": "User: ünï"}]}}},
    {"id": "c2", "type": "function", "function": {"name": "shell", "arguments": {"command": "ls -la", "timeout": 30}}},
]


def rendered_assistant(template: str, before: str) -> str:
    """The text the template writes for an assistant turn holding CALLS, cut from the turn
    that precedes it to the end."""
    msgs = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": CALLS}]
    full = template_env().from_string(template).render(messages=msgs, tools=TOOLS, bos_token="<B>", eos_token="<E>", add_generation_prompt=False)
    return full[full.rindex(before) + len(before) :]


def parse(fmt, text: str, step: int = 3) -> list:
    p, out = Parser(fmt), []
    for i in range(0, len(text), step):  # streamed in small pieces, markers split across them
        out += p.feed(text[i : i + step])
    rest = p.flush()
    return out + ([Text(rest)] if rest else [])


def calls_of(items) -> list:
    return [(c.name, c.arguments) for c in items if isinstance(c, Call)]


def expected():
    return [(c["function"]["name"], c["function"]["arguments"]) for c in CALLS]


def test_deepseek_v2_lite_round_trip():
    tpl = (ROOT / "models/deepseek-v2-lite/chat_template.jinja").read_text()
    text = rendered_assistant(tpl, "Assistant:")
    assert calls_of(parse(FORMATS["deepseek-v2-lite"], text)) == expected()


def test_glm_round_trip_on_its_native_template():
    # the checkpoint's own template (zai-org/GLM-5.3-Flash chat_template.jinja), verbatim
    tpl = (Path(__file__).parent / "fixtures/glm-5.3-flash.chat_template.jinja").read_text()
    text = rendered_assistant(tpl, "<|assistant|>")
    assert calls_of(parse(FORMATS["glm-5.3-flash"], text)) == expected()


def test_parser_text_order_holds_and_bad_calls():
    fmt = FORMATS["deepseek-v2-lite"]
    items = parse(fmt, 'Let me look. <tool_call>{"name": "a", "arguments": {}}</tool_call> then <tool_c', step=1)
    shape, last = [], None  # runs of text collapsed: text, call, text
    for i in items:
        kind = "call" if isinstance(i, Call) else "text"
        if kind != last:
            shape.append(kind)
        last = kind
    assert shape == ["text", "call", "text"]
    assert "".join(i.text for i in items if isinstance(i, Text)) == "Let me look.  then <tool_c"
    # never decodes: held as an open call, and given back as text when the generation ends
    bad = parse(fmt, "<tool_call>not json</tool_call>x")
    assert calls_of(bad) == [] and "".join(i.text for i in bad) == "<tool_call>not json</tool_call>x"
    assert calls_of(parse(fmt, '<tool_call>{"name": "a", "arguments": "{\\"k\\": 1}"}</tool_call>')) == [("a", {"k": 1})]


def test_glm_reads_reasoning_effort():
    tpl = template_env().from_string((Path(__file__).parent / "fixtures/glm-5.3-flash.chat_template.jinja").read_text())
    msgs = [{"role": "user", "content": "hi"}]
    for effort, line in [("low", "Reasoning Effort: Low"), ("high", "Reasoning Effort: High"), ("max", "Reasoning Effort: Max")]:
        assert tpl.render(messages=msgs, add_generation_prompt=True, reasoning_effort=effort).startswith("[gMASK]<sop><|system|>" + line)
