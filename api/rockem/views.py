"""The views. One base holds the path from a canonical request to a response; each
adapter is a subclass whose `post` names its own request and response types, so the
OpenAPI document is per adapter and the view code is written once."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request as HttpRequest
from fastapi.responses import StreamingResponse
from fastcbv import BaseView

from . import core
from .adapters import anthropic, internal, openai
from .edge import TemplateError
from .audio import AudioError
from .transcribe import NoSpeech, Transcriber
from .wire import ContextFull, Refused, WireError


def get_engine(request: HttpRequest) -> core.Engine:
    return request.app.state.engine


def get_transcriber(request: HttpRequest) -> Transcriber:
    t = getattr(request.app.state, "transcriber", None)
    if t is None:
        raise HTTPException(503, "no transcription model on this service (ROCKEM_TRANSCRIBE_MODEL)")
    return t


async def start(engine: core.Engine, req: core.Request) -> core.Run:
    """Refusals happen here, before any response starts: a prompt the template rejects or
    the node cannot hold is a 4xx, a node that cannot be reached is a 503."""
    try:
        return await engine.start(req)
    except (TemplateError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    except ContextFull as e:
        raise HTTPException(413, str(e)) from e
    except (Refused, WireError, OSError) as e:
        raise HTTPException(503, f"node: {e}") from e


async def fold_or_abort(run: core.Run, receive) -> tuple:
    """`run.fold()`, unless the client goes away first: then the run is cancelled, which closes
    its node connection, the wire's abort. A streaming response gets this from the response
    machinery; a non-streaming one folds inside the handler, which nothing cancels on a
    disconnect, so the node would compute to the end for nobody (rockem.lead, 2026-10-23)."""

    async def gone() -> None:
        while (await receive())["type"] != "http.disconnect":
            pass

    fold, left = asyncio.ensure_future(run.fold()), asyncio.ensure_future(gone())
    try:
        await asyncio.wait({fold, left}, return_when=asyncio.FIRST_COMPLETED)
        if fold.done():
            return fold.result()
        fold.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await fold
        raise HTTPException(499, "the client closed the connection")
    finally:
        left.cancel()
        if not fold.done():
            fold.cancel()


def sse(body: AsyncIterator[bytes]) -> StreamingResponse:
    return StreamingResponse(body, media_type="text/event-stream", headers={"cache-control": "no-cache", "x-accel-buffering": "no"})


class GenerationView(BaseView):
    engine: Annotated[core.Engine, Depends(get_engine)]

    @property
    def model(self) -> str:
        return self.engine.edge.model.name

    @property
    def identity(self) -> str:
        return f"{self.engine.edge.model.identity:016x}"

    async def _start(self, req: core.Request) -> core.Run:
        return await start(self.engine, req)

    @staticmethod
    def _sse(body: AsyncIterator[bytes]) -> StreamingResponse:
        return sse(body)


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
        text, _, _, done = await fold_or_abort(run, self.request.receive)
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
        text, _, _, done = await fold_or_abort(run, self.request.receive)
        return anthropic.fold(text, done, self.model)


async def no_speech() -> AsyncIterator[core.Event]:
    """A transcription of a clip with no voice: the end at once, no text, no node taken."""
    yield core.Done(core.Reason.NO_SPEECH, None, 0, 0, 0)


class TranscribeView(BaseView):
    transcriber: Annotated[Transcriber, Depends(get_transcriber)]

    async def post(self, body: internal.TranscribeRequest) -> StreamingResponse:
        """Transcribe: a stream of `data:` lines, one event each (docs/api.md), the text in
        `text` events."""
        try:
            audio = base64.b64decode(body.audio, validate=True)
        except binascii.Error as e:
            raise HTTPException(400, f"audio: not base64: {e}") from e
        t = self.transcriber
        try:
            req = t.request(audio, body.language, body.max_tokens)
        except NoSpeech:
            return sse(internal.stream(no_speech(), t.model.name, f"{t.model.identity:016x}"))
        except (AudioError, ValueError) as e:
            raise HTTPException(400, str(e)) from e
        run = await start(t.engine, req)
        return sse(internal.stream(run.events(), t.model.name, f"{t.model.identity:016x}"))


class HealthView(BaseView):
    engine: Annotated[core.Engine, Depends(get_engine)]

    async def get(self) -> dict:
        s = self.engine.scheduler
        return {"model": self.engine.edge.model.name, "identity": f"{self.engine.edge.model.identity:016x}", "nodes": len(s.nodes), "free": s.free, "waiting": s.waiting}
