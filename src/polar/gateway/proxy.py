"""HTTP client for forwarding requests to SGLang with SSE streaming support."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)


class UpstreamError(RuntimeError):
    """Base class for upstream gateway failures."""


class UpstreamHTTPError(UpstreamError):
    """Raised when the upstream returns a non-2xx status."""

    def __init__(self, status_code: int, body: dict[str, Any] | str | None = None):
        self.status_code = status_code
        self.body = body
        super().__init__(self._build_message(status_code, body))

    @staticmethod
    def _build_message(status_code: int, body: dict[str, Any] | str | None) -> str:
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return message
            message = body.get("message")
            if isinstance(message, str) and message:
                return message
        if isinstance(body, str) and body:
            return body
        return f"Upstream request failed with status {status_code}"


class UpstreamTimeoutError(UpstreamError):
    """Raised when the upstream times out."""


class UpstreamTransportError(UpstreamError):
    """Raised for connection and transport failures."""


class OpenedStream:
    """An already-opened upstream stream with a primed first chunk."""

    def __init__(self, response: httpx.Response):
        self._response = response
        self._lines = response.aiter_lines()
        self._buffered_chunks: list[dict[str, Any]] = []
        self._done = False
        self._closed = False

    async def prime(self) -> None:
        """Read the first parseable chunk before the HTTP 200 is committed downstream."""
        first_chunk = await self._next_chunk()
        if first_chunk is not None:
            self._buffered_chunks.append(first_chunk)

    async def aiter_chunks(self) -> AsyncIterator[dict[str, Any]]:
        """Iterate buffered and live SSE data chunks."""
        while self._buffered_chunks:
            yield self._buffered_chunks.pop(0)

        while not self._done:
            chunk = await self._next_chunk()
            if chunk is None:
                return
            yield chunk

    async def _next_chunk(self) -> dict[str, Any] | None:
        while not self._done:
            try:
                line = await self._lines.__anext__()
            except StopAsyncIteration:
                self._done = True
                return None
            except httpx.TimeoutException as exc:
                raise UpstreamTimeoutError("Upstream streaming response timed out") from exc
            except httpx.RequestError as exc:
                raise UpstreamTransportError(
                    f"Upstream streaming connection failed: {exc}"
                ) from exc

            line = line.strip()
            if not line or not line.startswith("data: "):
                continue

            data = line[6:]
            if data == "[DONE]":
                self._done = True
                return None

            try:
                return json.loads(data)
            except json.JSONDecodeError:
                logger.warning("Failed to parse SSE chunk: %s", data[:200])

        return None

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._response.aclose()


class SGLangClient:
    """Direct httpx client to SGLang's OpenAI-compatible API.

    Per-call bound comes from the session's remaining-timeout budget
    (`_await_with_budget` at the gateway node). The internal httpx timeout
    is a high liveness ceiling so that a stuck SGLang can't pin a request
    past the session deadline.
    """

    _LIVENESS_TIMEOUT_SECONDS = 900.0

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self._LIVENESS_TIMEOUT_SECONDS, connect=30),
            )
        return self._client

    async def _read_error_body(self, response: httpx.Response) -> dict[str, Any] | str | None:
        content = await response.aread()
        if not content:
            return None

        text = content.decode("utf-8", errors="replace").strip()
        if not text:
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return

        body = await self._read_error_body(response)
        await response.aclose()
        raise UpstreamHTTPError(response.status_code, body)

    @staticmethod
    def _translate_transport_error(exc: httpx.RequestError) -> UpstreamError:
        if isinstance(exc, httpx.TimeoutException):
            return UpstreamTimeoutError("Upstream request timed out")
        return UpstreamTransportError(f"Upstream request failed: {exc}")

    async def completion(self, request: dict[str, Any]) -> dict[str, Any]:
        """Non-streaming chat completion. Returns the full JSON response."""
        client = await self._get_client()
        request_copy = request.copy()
        request_copy.pop("stream", None)
        request_copy["stream"] = False

        try:
            resp = await client.post(
                "/v1/chat/completions",
                json=request_copy,
                headers={"Content-Type": "application/json"},
            )
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc

        await self._raise_for_status(resp)
        return resp.json()

    async def open_completion_stream(self, request: dict[str, Any]) -> OpenedStream:
        """Open, validate, and prime a streaming response before returning it."""
        client = await self._get_client()
        request_copy = request.copy()
        request_copy["stream"] = True
        response: httpx.Response | None = None

        try:
            upstream_request = client.build_request(
                "POST",
                "/v1/chat/completions",
                json=request_copy,
                headers={"Content-Type": "application/json"},
            )
            response = await client.send(upstream_request, stream=True)
            await self._raise_for_status(response)

            stream = OpenedStream(response)
            try:
                await stream.prime()
            except Exception:
                await stream.aclose()
                raise
            return stream
        except UpstreamError:
            if response is not None and not response.is_closed:
                await response.aclose()
            raise
        except httpx.RequestError as exc:
            if response is not None and not response.is_closed:
                await response.aclose()
            raise self._translate_transport_error(exc) from exc

    async def completion_stream(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Backwards-compatible helper for callers that still expect an iterator."""
        stream = await self.open_completion_stream(request)
        try:
            async for chunk in stream.aiter_chunks():
                yield chunk
        finally:
            await stream.aclose()

    async def list_models(self) -> dict[str, Any]:
        """Passthrough GET /v1/models."""
        client = await self._get_client()
        try:
            resp = await client.get("/v1/models")
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc
        await self._raise_for_status(resp)
        return resp.json()

    async def health(self) -> dict[str, Any]:
        """Passthrough GET /health."""
        client = await self._get_client()
        try:
            resp = await client.get("/health")
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc
        await self._raise_for_status(resp)
        content = await resp.aread()
        if not content:
            return {"status": "ok"}

        text = content.decode("utf-8", errors="replace").strip()
        if not text:
            return {"status": "ok"}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"status": "ok", "body": text}

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
