"""The views. One base holds the path from a canonical request to a response; each
adapter is a subclass whose `post` names its own request and response types, so the
OpenAPI document is per adapter and the view code is written once."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request as HttpRequest
from fastapi.responses import StreamingResponse
from fastcbv import BaseView

from . import core
from .adapters import anthropic, internal, openai
from .edge import TemplateError
from .wire import ContextFull, Refused, WireError


def get_engine(request: HttpRequest) -> core.Engine:
    return request.app.state.engine


class GenerationView(BaseView):
    engine: Annotated[core.Engine, Depends(get_engine)]

    @property
    def model(self) -> str:
        return self.engine.edge.bundle.manifest.model

    @property
    def identity(self) -> str:
        return f"{self.engine.edge.bundle.identity:016x}"

    async def _start(self, req: core.Request) -> core.Run:
        """Refusals happen here, before any response starts: a prompt the template rejects
        or the node cannot hold is a 4xx, a node that cannot be reached is a 503."""
        try:
            return await self.engine.start(req)
        except (TemplateError, ValueError) as e:
            raise HTTPException(400, str(e)) from e
        except ContextFull as e:
            raise HTTPException(413, str(e)) from e
        except (Refused, WireError, OSError) as e:
            raise HTTPException(503, f"node: {e}") from e

    @staticmethod
    def _sse(body: AsyncIterator[bytes]) -> StreamingResponse:
        return StreamingResponse(body, media_type="text/event-stream", headers={"cache-control": "no-cache", "x-accel-buffering": "no"})


class InternalView(GenerationView):
    async def post(self, body: internal.Request) -> StreamingResponse:
        """Generate: a stream of `data:` lines, one event each (docs/api.md)."""
        try:
            req = internal.to_request(body)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        run = await self._start(req)
        return self._sse(internal.stream(run.events(), self.model, self.identity))


class OpenAIChatView(GenerationView):
    async def post(self, body: openai.Request) -> openai.Response:
        try:
            req = openai.to_request(body)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        run = await self._start(req)
        if body.stream:
            include_usage = bool((body.stream_options or {}).get("include_usage"))
            return self._sse(openai.stream(run.events(), self.model, include_usage))
        text, _, _, done = await run.fold()
        return openai.fold(text, done, self.model)


class AnthropicMessagesView(GenerationView):
    async def post(self, body: anthropic.Request) -> anthropic.Response:
        try:
            req = anthropic.to_request(body)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        run = await self._start(req)
        if body.stream:
            return self._sse(anthropic.stream(run.events(), self.model, run.prompt_tokens))
        text, _, _, done = await run.fold()
        return anthropic.fold(text, done, self.model)


class HealthView(BaseView):
    engine: Annotated[core.Engine, Depends(get_engine)]

    async def get(self) -> dict:
        s = self.engine.scheduler
        return {"model": self.engine.edge.bundle.manifest.model, "identity": f"{self.engine.edge.bundle.identity:016x}", "nodes": len(s.nodes), "free": s.free, "waiting": s.waiting}
