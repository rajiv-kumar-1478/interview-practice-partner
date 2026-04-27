"""Unit tests for Twilio signature middleware.

Tests:
- Valid signature (mocked to return True) passes through and returns HTTP 200
- Invalid signature (mocked to return False) returns HTTP 403
- Missing X-Twilio-Signature header returns HTTP 403

Requirements: 1.2, 1.3
"""

from __future__ import annotations

from urllib.parse import urlencode

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from httpx import ASGITransport
from hypothesis import given, settings
from hypothesis import strategies as st
from twilio.request_validator import RequestValidator

from interview_practice_partner.api.middleware.twilio_signature import TwilioSignatureMiddleware


# ---------------------------------------------------------------------------
# Minimal test app
# ---------------------------------------------------------------------------

def _build_test_app(auth_token: str = "test_auth_token") -> FastAPI:
    """Build a minimal FastAPI app with TwilioSignatureMiddleware registered.

    The single POST /webhook route returns 200 so we can confirm pass-through.
    """
    app = FastAPI()
    app.add_middleware(TwilioSignatureMiddleware, auth_token=auth_token)

    @app.post("/webhook")
    async def webhook() -> PlainTextResponse:
        return PlainTextResponse("OK", status_code=200)

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _form_body(**kwargs: str) -> bytes:
    """Encode keyword arguments as application/x-www-form-urlencoded bytes."""
    return urlencode(kwargs).encode()


_SAMPLE_BODY = _form_body(
    MessageSid="SM_test_001",
    From="whatsapp:+447700900001",
    To="whatsapp:+14155238886",
    Body="Hello",
    NumMedia="0",
)

_CONTENT_TYPE_HEADER = {"Content-Type": "application/x-www-form-urlencoded"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTwilioSignatureMiddleware:
    """Unit tests for TwilioSignatureMiddleware.

    RequestValidator.validate() is mocked to control pass/fail without
    requiring a real Twilio account or correct HMAC computation.
    """

    async def test_valid_signature_returns_200(self, mocker) -> None:
        """A request whose signature validates successfully passes through (HTTP 200).

        Requirements: 1.2
        """
        mocker.patch(
            "twilio.request_validator.RequestValidator.validate",
            return_value=True,
        )

        app = _build_test_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/webhook",
                content=_SAMPLE_BODY,
                headers={
                    **_CONTENT_TYPE_HEADER,
                    "X-Twilio-Signature": "valid_signature_value",
                },
            )

        assert response.status_code == 200

    async def test_invalid_signature_returns_403(self, mocker) -> None:
        """A request whose signature fails validation is rejected with HTTP 403.

        Requirements: 1.3
        """
        mocker.patch(
            "twilio.request_validator.RequestValidator.validate",
            return_value=False,
        )

        app = _build_test_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/webhook",
                content=_SAMPLE_BODY,
                headers={
                    **_CONTENT_TYPE_HEADER,
                    "X-Twilio-Signature": "invalid_signature_value",
                },
            )

        assert response.status_code == 403

    async def test_missing_signature_header_returns_403(self, mocker) -> None:
        """A request with no X-Twilio-Signature header is rejected with HTTP 403.

        When the header is absent the middleware passes an empty string to
        RequestValidator.validate(), which returns False — resulting in 403.

        Requirements: 1.3
        """
        mocker.patch(
            "twilio.request_validator.RequestValidator.validate",
            return_value=False,
        )

        app = _build_test_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/webhook",
                content=_SAMPLE_BODY,
                headers=_CONTENT_TYPE_HEADER,  # no X-Twilio-Signature
            )

        assert response.status_code == 403

    async def test_missing_signature_header_calls_validate_with_empty_string(
        self, mocker
    ) -> None:
        """Middleware passes an empty string to validate() when header is absent.

        This confirms the middleware's fallback behaviour: it does not raise
        a KeyError but instead defaults to "" and lets the validator decide.

        Requirements: 1.3
        """
        mock_validate = mocker.patch(
            "twilio.request_validator.RequestValidator.validate",
            return_value=False,
        )

        app = _build_test_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            await client.post(
                "/webhook",
                content=_SAMPLE_BODY,
                headers=_CONTENT_TYPE_HEADER,
            )

        # validate() must have been called; the third positional arg is the signature
        mock_validate.assert_called_once()
        _url, _params, signature_arg = mock_validate.call_args.args
        assert signature_arg == ""

    async def test_valid_signature_body_is_accessible_downstream(self, mocker) -> None:
        """Raw body is cached in request.state.body so downstream handlers can read it.

        Requirements: 1.2
        """
        mocker.patch(
            "twilio.request_validator.RequestValidator.validate",
            return_value=True,
        )

        from starlette.requests import Request
        from starlette.responses import Response as StarletteResponse

        received_body: list[bytes] = []

        # Build a Starlette app (not FastAPI) to avoid FastAPI's request-body
        # validation layer, which would return 422 before our handler runs.
        from starlette.applications import Starlette
        from starlette.routing import Route

        async def webhook(request: Request) -> StarletteResponse:
            received_body.append(request.state.body)
            return StarletteResponse("OK", status_code=200)

        starlette_app = Starlette(routes=[Route("/webhook", webhook, methods=["POST"])])
        starlette_app.add_middleware(TwilioSignatureMiddleware, auth_token="test_auth_token")

        async with httpx.AsyncClient(
            transport=ASGITransport(app=starlette_app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/webhook",
                content=_SAMPLE_BODY,
                headers={
                    **_CONTENT_TYPE_HEADER,
                    "X-Twilio-Signature": "valid_signature_value",
                },
            )

        assert response.status_code == 200
        assert len(received_body) == 1
        assert received_body[0] == _SAMPLE_BODY


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Strategies for generating arbitrary but printable ASCII tokens, paths, and
# form-body parameter names/values.  We restrict to printable ASCII to avoid
# encoding edge-cases that are orthogonal to the signature-validation property.

_auth_token_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=8,
    max_size=64,
)

_url_path_strategy = st.just("webhook")

# Form body: a dict of 0–5 key/value pairs, each key/value being short ASCII
_form_params_strategy = st.dictionaries(
    keys=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
        min_size=1,
        max_size=16,
    ),
    values=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
        min_size=0,
        max_size=64,
    ),
    min_size=0,
    max_size=5,
)


def _build_app_for_token(auth_token: str) -> FastAPI:
    """Build a minimal FastAPI app with TwilioSignatureMiddleware for the given token."""
    app = FastAPI()
    app.add_middleware(TwilioSignatureMiddleware, auth_token=auth_token)

    @app.post("/webhook")
    async def webhook() -> PlainTextResponse:
        return PlainTextResponse("OK", status_code=200)

    return app


# Feature: interview-practice-partner, Property 1: Signature Validation Accepts Valid and Rejects Invalid Requests
@given(
    auth_token=_auth_token_strategy,
    path=_url_path_strategy,
    params=_form_params_strategy,
)
@settings(max_examples=100)
async def test_property_1_valid_signature_accepted_invalid_rejected(
    auth_token: str,
    path: str,
    params: dict[str, str],
) -> None:
    """Property 1: Signature Validation Accepts Valid and Rejects Invalid Requests.

    For any inbound POST body and URL:
    - A request signed with the correct auth token is accepted (HTTP 200).
    - A request with an incorrect/tampered signature is rejected (HTTP 403).
    - A request with an absent signature is rejected (HTTP 403).

    Validates: Requirements 1.2, 1.3
    """
    app = _build_app_for_token(auth_token)
    base_url = "http://testserver"
    full_url = f"{base_url}/{path}"

    # Compute the real HMAC-SHA1 signature using the Twilio RequestValidator
    validator = RequestValidator(auth_token)
    valid_signature = validator.compute_signature(full_url, params)

    # Encode the form body exactly as the middleware will receive it
    encoded_body = urlencode(params).encode()
    content_type_header = {"Content-Type": "application/x-www-form-urlencoded"}

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url=base_url
    ) as client:

        # --- Sub-property A: correct signature → HTTP 200 ---
        response_valid = await client.post(
            f"/{path}",
            content=encoded_body,
            headers={
                **content_type_header,
                "X-Twilio-Signature": valid_signature,
            },
        )
        assert response_valid.status_code == 200, (
            f"Expected 200 for valid signature, got {response_valid.status_code}. "
            f"auth_token={auth_token!r}, path={path!r}, params={params!r}"
        )

        # --- Sub-property B: tampered signature → HTTP 403 ---
        # Tamper by appending a character to the valid signature
        tampered_signature = valid_signature + "X"
        response_tampered = await client.post(
            f"/{path}",
            content=encoded_body,
            headers={
                **content_type_header,
                "X-Twilio-Signature": tampered_signature,
            },
        )
        assert response_tampered.status_code == 403, (
            f"Expected 403 for tampered signature, got {response_tampered.status_code}. "
            f"auth_token={auth_token!r}, path={path!r}, params={params!r}"
        )

        # --- Sub-property C: absent signature → HTTP 403 ---
        response_absent = await client.post(
            f"/{path}",
            content=encoded_body,
            headers=content_type_header,  # no X-Twilio-Signature header
        )
        assert response_absent.status_code == 403, (
            f"Expected 403 for absent signature, got {response_absent.status_code}. "
            f"auth_token={auth_token!r}, path={path!r}, params={params!r}"
        )
