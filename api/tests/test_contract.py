"""The contract with the Rust client (api/client), pinned by files both sides test against:
`request.json` is a request Rust builds and this service accepts, and each `*.stream` is a
body this service writes and Rust decodes. Run with REGEN=1 to rewrite the streams after a
deliberate change to the format; otherwise a difference fails here before it reaches Rust."""

import json
import os
import re
from pathlib import Path

import pytest
from rockem.adapters import internal

from conftest import CALL_BODY, CALL_END, CALL_START, EOS

FIXTURES = Path(__file__).resolve().parents[1] / "client/tests/fixtures"
pytestmark = pytest.mark.anyio


def pinned(name: str, body: str):
    body = re.sub(r"call_[0-9a-f]{12}", "call_0", body)  # call ids are random; the shape is the contract
    body = re.sub(r'"identity":"[0-9a-f]{16}"', '"identity":"0000000000000000"', body)  # the test bundle's hash
    path = FIXTURES / name
    if os.environ.get("REGEN"):
        path.write_text(body)
    assert body == path.read_text(), f"{name}: the service's stream changed; REGEN=1 if deliberate"


def test_request_fixture_is_a_valid_request():
    internal.Request.model_validate(json.loads((FIXTURES / "request.json").read_text()))


async def test_streams(client, node):
    req = json.loads((FIXTURES / "request.json").read_text())
    node.draws, node.max_ctx = [2, CALL_START, CALL_BODY, CALL_END, EOS], 64  # room for the whole conversation
    r = await client.post("/rockem/v1/generate", json=req)
    assert r.status_code == 200, r.text
    pinned("tool_call.stream", r.text)

    node.draws, node.max_ctx = [2, 3, 4], 12
    r = await client.post("/rockem/v1/generate", json={"prompt": "hello", "stop": ["stop"]})
    pinned("stop_sequence.stream", r.text)

    node.draws, node.close_after = [2] * 10, 1
    r = await client.post("/rockem/v1/generate", json={"prompt": "hello"})
    pinned("error.stream", r.text)
