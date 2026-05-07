from dataclasses import dataclass
from starlette.requests import Request
from starlette.responses import Response
from collections.abc import Awaitable, Callable


@dataclass(frozen=True, slots=True)
class ResponseValidationContext:
    """Hold response details passed to a cache store validator."""

    request: Request
    request_body: bytes
    response: Response
    payload: dict[str, object]
    model: str | None
    scope: str | None


type ResponseShapeValidator = Callable[
    [ResponseValidationContext],
    bool | Awaitable[bool],
]
