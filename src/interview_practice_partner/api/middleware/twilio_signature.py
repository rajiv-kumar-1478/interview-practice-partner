"""Starlette middleware for Twilio HMAC-SHA1 webhook signature validation."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

import structlog
from twilio.request_validator import RequestValidator

logger = structlog.get_logger(__name__)

# Paths that bypass signature validation (e.g. health checks)
_SKIP_PATHS = frozenset(["/health"])


class TwilioSignatureMiddleware(BaseHTTPMiddleware):
    """Validate the ``X-Twilio-Signature`` header on inbound webhook requests.

    Reads the raw request body before FastAPI parses it, validates the
    HMAC-SHA1 signature using the Twilio ``RequestValidator``, and stores
    the raw body in ``request.state.body`` so downstream handlers can still
    parse the form data.

    Returns HTTP 403 if the signature is absent or invalid.
    Skips validation for paths listed in ``_SKIP_PATHS`` (e.g. ``/health``).

    Satisfies Requirements 1.2, 1.3.
    """

    def __init__(self, app, auth_token: str) -> None:
        super().__init__(app)
        self._validator = RequestValidator(auth_token)

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip validation for health and other non-webhook paths
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        # Read the raw body once and cache it in request state so downstream
        # handlers (form parsers) can still consume it.
        body: bytes = await request.body()
        request.state.body = body

        # Only validate POST requests (Twilio webhooks are always POST)
        if request.method != "POST":
            return await call_next(request)

        signature = request.headers.get("X-Twilio-Signature", "")

        # Reconstruct the full URL that Twilio signed
        url = str(request.url)

        # Parse the form body into a dict for validation
        params: dict[str, str] = {}
        if body:
            try:
                from urllib.parse import parse_qs, unquote_plus

                # parse_qs returns lists; Twilio validation needs single values
                raw_str = body.decode("utf-8")
                for part in raw_str.split("&"):
                    if "=" in part:
                        key, _, value = part.partition("=")
                        params[unquote_plus(key)] = unquote_plus(value)
            except Exception:  # noqa: BLE001
                logger.warning("twilio_signature_body_parse_error")

        is_valid = self._validator.validate(url, params, signature)

        if not is_valid:
            logger.warning(
                "twilio_signature_invalid",
                path=request.url.path,
                method=request.method,
            )
            return Response(content="Forbidden", status_code=403)

        return await call_next(request)
