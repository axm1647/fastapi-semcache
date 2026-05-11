"""ASGI I/O helpers for middleware adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

DEFAULT_MAX_BODY_BYTES: int = 10 * 1024 * 1024
"""Default cap (10 MiB) for request and response bodies when a limit is enabled."""


@dataclass
class TeeSend:
    """ASGI send wrapper that forwards messages while buffering response body chunks.

    Forwards each message to ``real_send`` immediately so the client sees streaming
    behavior. Accumulates ``http.response.body`` payloads in a side buffer until
    the stream completes or ``max_body_bytes`` is exceeded.

    Attributes:
        real_send: Underlying ASGI ``send`` callable.
        max_body_bytes: Optional cap on buffered body size; when exceeded,
            ``over_limit`` is set, buffered chunks are discarded, and forwarding
            continues.
        merge_into_start: Optional header map appended to ``http.response.start``
            before forwarding (for example ``X-Cache: MISS``).
        status_code: Populated from the first ``http.response.start`` message.
        headers: Response headers from ``http.response.start`` (latin-1 decoded).
        over_limit: True when buffering was stopped due to ``max_body_bytes``.

    Example:
        After ``await app(scope, replay_receive, tee)``, inspect ``tee.body``,
        ``tee.status_code``, and ``tee.headers``; skip cache store when
        ``tee.over_limit``.
    """

    real_send: Send
    max_body_bytes: int | None = None
    merge_into_start: Mapping[str, str] | None = None

    status_code: int = field(default=500, init=False)
    headers: dict[str, str] = field(default_factory=dict, init=False)
    _body_chunks: list[bytes] = field(default_factory=list, init=False)
    _buffered_total: int = field(default=0, init=False)
    over_limit: bool = field(default=False, init=False)

    async def __call__(self, message: Message) -> None:
        """Forward ``message`` to ``real_send`` and update tee state.

        Args:
            message: ASGI HTTP response message.
        """
        msg_type = message["type"]

        if msg_type == "http.response.start":
            self.status_code = int(message["status"])
            raw_headers = list(message.get("headers", []))
            merged: list[tuple[bytes, bytes]] = []
            if self.merge_into_start:
                for hk, hv in self.merge_into_start.items():
                    merged.append((hk.encode("latin-1"), hv.encode("latin-1")))
            combined = raw_headers + merged
            parsed: dict[str, str] = {}
            for key, value in combined:
                if isinstance(key, bytes) and isinstance(value, bytes):
                    parsed[key.decode("latin-1")] = value.decode("latin-1")
            self.headers = parsed
            out_message = dict(message)
            out_message["headers"] = combined
            await self.real_send(out_message)
            return

        if msg_type == "http.response.body":
            chunk = message.get("body", b"")
            if not isinstance(chunk, bytes):
                chunk = b""
            chunk = chunk or b""
            await self.real_send(message)
            if self.over_limit:
                return
            if self.max_body_bytes is not None:
                next_total = self._buffered_total + len(chunk)
                if next_total > self.max_body_bytes:
                    self.over_limit = True
                    self._body_chunks.clear()
                    return
            if chunk:
                self._body_chunks.append(chunk)
                self._buffered_total += len(chunk)

    @property
    def body(self) -> bytes:
        """Concatenated buffered body bytes after the response stream completes."""
        return b"".join(self._body_chunks)


async def read_body(
    receive: Receive, max_body_bytes: int | None = DEFAULT_MAX_BODY_BYTES
) -> bytes:
    """Read and buffer the incoming request body from ASGI receive.

    Args:
        receive: ASGI receive callable for the current request.
        max_body_bytes: Maximum total size; when None, no limit is enforced.
            When exceeded, remaining body data is consumed then an HTTP 413 is raised.

    Returns:
        Full request body bytes.

    Raises:
        HTTPException: When the body exceeds max_body_bytes (status 413).
    """
    chunks: list[bytes] = []
    total = 0

    async def drain_remaining_request_body() -> None:
        """Consume the rest of the request body so the connection can close cleanly."""
        while True:
            message = await receive()
            msg_type = message["type"]
            if msg_type == "http.disconnect":
                return
            if msg_type != "http.request":
                continue
            if not bool(message.get("more_body", False)):
                return

    while True:
        message = await receive()
        msg_type = message["type"]
        if msg_type == "http.disconnect":
            break
        if msg_type != "http.request":
            continue
        chunk = message.get("body", b"")
        if not isinstance(chunk, bytes):
            chunk = b""
        more_body = bool(message.get("more_body", False))

        if max_body_bytes is not None:
            next_total = total + len(chunk)
            if next_total > max_body_bytes:
                if more_body:
                    await drain_remaining_request_body()
                raise HTTPException(
                    status_code=413,
                    detail="Request body exceeds configured maximum size.",
                )

        if chunk:
            chunks.append(chunk)
            total += len(chunk)
        if not more_body:
            break

    return b"".join(chunks)


async def call_downstream(
    app: ASGIApp,
    scope: Scope,
    body: bytes,
    max_body_bytes: int | None = DEFAULT_MAX_BODY_BYTES,
) -> Response:
    """Invoke downstream ASGI app and buffer its response.

    This function always returns a plain, fully-buffered ``starlette.responses.Response``
    whose ``.body`` attribute is a ``bytes`` object. Callers that access ``.body``
    (e.g. ``prepare_response_for_client``) rely on this contract. Do not replace
    the return type with ``StreamingResponse`` or any subclass that omits ``.body``.

    Args:
        app: Downstream ASGI application.
        scope: Current request ASGI scope.
        body: Full buffered request body.
        max_body_bytes: Maximum total downstream response body size; when None,
            no limit is enforced. When exceeded, returns HTTP 502 with a short body.

    Returns:
        Fully-buffered Starlette ``Response`` whose ``.body`` contains the
        complete downstream response body as ``bytes``, or HTTP 502 when the
        buffered response exceeds ``max_body_bytes``.
    """
    status_code = 500
    response_headers: dict[str, str] = {}
    response_body: list[bytes] = []
    body_sent = False
    response_over_limit = False
    buffered_total = 0

    async def replay_receive() -> Message:
        nonlocal body_sent
        if body_sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        body_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def capture_send(message: Message) -> None:
        nonlocal status_code, response_headers, response_over_limit, buffered_total
        msg_type = message["type"]
        if msg_type == "http.response.start":
            status_code = int(message["status"])
            raw_headers = message.get("headers", [])
            parsed_headers: dict[str, str] = {}
            for key, value in raw_headers:
                if isinstance(key, bytes) and isinstance(value, bytes):
                    parsed_headers[key.decode("latin-1")] = value.decode("latin-1")
            response_headers = parsed_headers
            return
        if msg_type == "http.response.body":
            chunk = message.get("body", b"")
            if not isinstance(chunk, bytes):
                chunk = b""
            if response_over_limit:
                return
            if max_body_bytes is None:
                if chunk:
                    response_body.append(chunk)
                return
            next_total = buffered_total + len(chunk)
            if next_total > max_body_bytes:
                response_over_limit = True
                response_body.clear()
                return
            if chunk:
                response_body.append(chunk)
                buffered_total += len(chunk)
            return

    await app(scope, replay_receive, capture_send)
    if response_over_limit:
        return Response(
            content=b"Bad Gateway",
            status_code=502,
            media_type="text/plain",
        )
    return Response(
        content=b"".join(response_body),
        status_code=status_code,
        headers=response_headers,
    )


async def send_response(response: Response, scope: Scope, send: Send) -> None:
    """Emit a Starlette response over ASGI send.

    Args:
        response: Response object to emit.
        scope: Current request ASGI scope.
        send: ASGI send callable.
    """

    async def _unused_receive() -> Message:
        return {"type": "http.disconnect"}

    await response(scope, _unused_receive, send)
