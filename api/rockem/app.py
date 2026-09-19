"""The API service. One process: the bundle, the edge and the scheduler are built once
at startup and every view reaches them through the app's state.

    ROCKEM_BUNDLE=/path/to/bundle ROCKEM_NODES=192.168.88.4:7332 uv run uvicorn rockem.app:app

    ROCKEM_BUNDLE      the bundle directory (the edge's files are enough; docs/bundle.md)
    ROCKEM_NODES       comma-separated host:port of every node
    ROCKEM_IDENTITY    `bundle` (default): send the bundle's identity, so a node built from
                       another bundle refuses us; `any`: send 0 and accept any node
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastcbv import APIRouter

from .bundle import Bundle
from .core import Engine
from .edge import Edge
from .scheduler import Addr, Scheduler
from .views import AnthropicMessagesView, HealthView, InternalView, OpenAIChatView


def parse_nodes(spec: str) -> list[Addr]:
    nodes = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        host, _, port = item.rpartition(":")
        nodes.append((host, int(port)))
    if not nodes:
        raise ValueError("ROCKEM_NODES: no nodes")
    return nodes


def build(bundle_path: str, nodes: list[Addr], identity: str = "bundle") -> Engine:
    bundle = Bundle(bundle_path)
    want = bundle.identity if identity == "bundle" else 0
    return Engine(Edge(bundle), Scheduler(nodes, bundle.bos, bundle.eos, bundle.vocab_size, want))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "engine"):
        app.state.engine = build(os.environ["ROCKEM_BUNDLE"], parse_nodes(os.environ["ROCKEM_NODES"]), os.environ.get("ROCKEM_IDENTITY", "bundle"))
    yield


router = APIRouter()
router.add_view("/v1/generate", InternalView, tags=["internal"])
router.add_view("/v1/chat/completions", OpenAIChatView, tags=["openai"])
router.add_view("/v1/messages", AnthropicMessagesView, tags=["anthropic"])
router.add_view("/health", HealthView, tags=["internal"])


@router.get("/", include_in_schema=False)
async def chat() -> HTMLResponse:
    """A page to talk to the service by hand: `chat.html`, next to this file."""
    return HTMLResponse((Path(__file__).parent / "chat.html").read_text())

app = FastAPI(title="rockem", lifespan=lifespan)
app.include_router(router)
