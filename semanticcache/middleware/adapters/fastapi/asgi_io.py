"""ASGI I/O helpers for middleware adapters."""

from __future__ import annotations

from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

DEFAULT_MAX_BODY_BYTES: int = 10 * 1024 * 1024
"""Default cap (10 MiB) for request and response bodies when a limit is enabled."""


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
