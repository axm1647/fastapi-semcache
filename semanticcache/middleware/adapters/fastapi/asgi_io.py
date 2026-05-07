"""ASGI I/O helpers for middleware adapters."""

from __future__ import annotations

from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send


async def read_body(receive: Receive) -> bytes:
    """Read and buffer the incoming request body from ASGI receive.

    Args:
        receive: ASGI receive callable for the current request.

    Returns:
        Full request body bytes.
    """
    chunks: list[bytes] = []
    while True:
        message = await receive()
        msg_type = message["type"]
        if msg_type == "http.disconnect":
            break
        if msg_type != "http.request":
            continue
        chunk = message.get("body", b"")
        if isinstance(chunk, bytes) and chunk:
            chunks.append(chunk)
        if not bool(message.get("more_body", False)):
            break
    return b"".join(chunks)


async def call_downstream(app: ASGIApp, scope: Scope, body: bytes) -> Response:
    """Invoke downstream ASGI app and buffer its response.

    Args:
        app: Downstream ASGI application.
        scope: Current request ASGI scope.
        body: Full buffered request body.

    Returns:
        Buffered Starlette response built from downstream ASGI messages.
    """
    status_code = 500
    response_headers: dict[str, str] = {}
    response_body: list[bytes] = []
    body_sent = False

    async def replay_receive() -> Message:
        nonlocal body_sent
        if body_sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        body_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def capture_send(message: Message) -> None:
        nonlocal status_code, response_headers
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
            if isinstance(chunk, bytes) and chunk:
                response_body.append(chunk)
            return

    await app(scope, replay_receive, capture_send)
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
