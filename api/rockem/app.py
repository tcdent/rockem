"""The API service. One process: the bundle, the edge and the scheduler are built once
at startup and every view reaches them through the app's state.

    ROCKEM_BUNDLE=/path/to/bundle ROCKEM_NODES=192.168.88.4:7332 uv run uvicorn rockem.app:app

    ROCKEM_BUNDLE      the bundle directory (the edge's files are enough; docs/bundle.md)
    ROCKEM_NODES       comma-separated host:port of every node
    ROCKEM_MODEL       the manifest name of the model the chat surfaces speak (required: the
                       wire has no default model, docs/queue/094)
    ROCKEM_TRANSCRIBE_MODEL  the manifest name of the model /rockem/v1/transcribe speaks; unset,
                       the endpoint answers 503. Both models share the nodes and their one line.
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
from .transcribe import Transcriber
from .views import RUNS, AnthropicMessagesView, HealthView, InternalView, OpenAIChatView, TranscribeView


def parse_nodes(spec: str) -> list[Addr]:
    nodes = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        host, _, port = item.rpartition(":")
        nodes.append((host, int(port)))
    if not nodes:
        raise ValueError("ROCKEM_NODES: no nodes")
    return nodes


def build(bundle_path: str, nodes: list[Addr], model: str, identity: str = "bundle") -> Engine:
    m = Bundle(bundle_path).model(model)
    want = m.identity if identity == "bundle" else 0
    return Engine(Edge(m), Scheduler(nodes, want))


def transcriber(chat: Engine, model: str) -> Transcriber:
    """The transcription model from the chat engine's bundle, on the chat engine's scheduler:
    one node serves one connection at a time whichever model it speaks."""
    return Transcriber.of(Engine(Edge(chat.edge.model.bundle.model(model)), chat.scheduler))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "engine"):
        app.state.engine = build(os.environ["ROCKEM_BUNDLE"], parse_nodes(os.environ["ROCKEM_NODES"]), os.environ["ROCKEM_MODEL"], os.environ.get("ROCKEM_IDENTITY", "bundle"))
        if os.environ.get("ROCKEM_TRANSCRIBE_MODEL"):
            app.state.transcriber = transcriber(app.state.engine, os.environ["ROCKEM_TRANSCRIBE_MODEL"])
    yield


router = APIRouter()
router.add_view("/rockem/v1/generate", InternalView, tags=["rockem"])
router.add_view("/rockem/v1/transcribe", TranscribeView, tags=["rockem"])
router.add_view("/v1/chat/completions", OpenAIChatView, tags=["openai"])
router.add_view("/v1/messages", AnthropicMessagesView, tags=["anthropic"])
router.add_view("/health", HealthView, tags=["internal"])


@router.get("/", include_in_schema=False)
async def chat() -> HTMLResponse:
    """A page to talk to the service by hand: `chat.html`, next to this file."""
    return HTMLResponse((Path(__file__).parent / "chat.html").read_text())

class ReleaseRuns:
    """Gives back the node of every run a request took, when the request ends, however it ends:
    finished, failed, or its client gone before the stream began (then neither the body nor a
    response's background task ever runs). The views register their runs (views.RUNS)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        runs: list = []
        scope[RUNS] = runs  # the list itself: routes see copies of the scope, all holding this list
        try:
            await self.app(scope, receive, send)
        finally:
            for run in runs:
                await run.release()


app = FastAPI(title="rockem", lifespan=lifespan)
app.add_middleware(ReleaseRuns)
app.include_router(router)
