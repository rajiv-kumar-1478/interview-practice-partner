"""Starlette middleware for structured JSON request/response logging."""

from __future__ import annotations

import time
import uuid

import structlog
import structlog.contextvars
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log each inbound request and outbound response as structured JSON.

    Binds a ``correlation_id`` (UUID) to the ``structlog`` context vars at
    the start of each request so that all log entries emitted during that
    request share the same identifier.  Clears the context vars on request
    completion.

    Logs:
    - ``http.request``: method, path, ``from_number`` (if present in form body)
    - ``http.response``: status code, latency in milliseconds

    Satisfies Requirement 12.3.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

        start_time = time.monotonic()

        # Try to extract from_number from cached body (set by TwilioSignatureMiddleware)
        from_number: str | None = None
        cached_body: bytes | None = getattr(request.state, "body", None)
        if cached_body:
            try:
                from urllib.parse import unquote_plus

                raw_str = cached_body.decode("utf-8")
                for part in raw_str.split("&"):
                    if "=" in part:
                        key, _, value = part.partition("=")
                        if unquote_plus(key) == "From":
                            from_number = unquote_plus(value)
                            break
            except Exception:  # noqa: BLE001
                pass

        log = logger.bind(
            method=request.method,
            path=request.url.path,
            from_number=from_number,
        )
        log.info("http.request")

        try:
            response = await call_next(request)
        except Exception:
            latency_ms = round((time.monotonic() - start_time) * 1000, 2)
            logger.error(
                "http.response",
                status_code=500,
                latency_ms=latency_ms,
            )
            raise
        finally:
            structlog.contextvars.clear_contextvars()

        latency_ms = round((time.monotonic() - start_time) * 1000, 2)
        logger.info(
            "http.response",
            status_code=response.status_code,
            latency_ms=latency_ms,
        )

        return response
