# Feature: interview-practice-partner, Property 3: Delivery Failure Logs Contain Required Fields
"""Property-based tests for delivery failure log fields.

Any delivery status callback with ``failed`` or ``undelivered`` status must
produce a log entry containing both ``phone_number`` and ``message_sid`` fields.

Validates: Requirements 1.6
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from urllib.parse import urlencode

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from interview_practice_partner.api.schemas import DeliveryStatusCallback
from interview_practice_partner.api.routers.webhook import status_callback

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for failure statuses — only the two statuses that trigger logging
_failure_status_strategy = st.sampled_from(["failed", "undelivered"])

# Strategy for non-failure statuses — must NOT trigger error logging
_non_failure_status_strategy = st.sampled_from(["sent", "delivered", "queued", "sending"])

# Strategy for arbitrary MessageSid values (Twilio uses alphanumeric + hyphens)
_message_sid_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="-_",
    ),
    min_size=1,
    max_size=64,
)

# Strategy for phone numbers — E.164-style WhatsApp numbers
_phone_number_strategy = st.one_of(
    st.from_regex(r"whatsapp:\+1[0-9]{10}", fullmatch=True),
    st.from_regex(r"whatsapp:\+44[0-9]{10}", fullmatch=True),
    st.from_regex(r"\+1[0-9]{10}", fullmatch=True),
    st.from_regex(r"\+44[0-9]{10}", fullmatch=True),
)

# Strategy for optional error codes (Twilio error codes are numeric strings)
_error_code_strategy = st.one_of(
    st.none(),
    st.from_regex(r"[0-9]{5}", fullmatch=True),
)

# Strategy for optional error messages
_error_message_strategy = st.one_of(
    st.none(),
    st.text(min_size=0, max_size=100),
)


# ---------------------------------------------------------------------------
# Helper: build a DeliveryStatusCallback DTO
# ---------------------------------------------------------------------------


def _make_callback(
    message_sid: str,
    message_status: str,
    to: str,
    error_code: str | None = None,
    error_message: str | None = None,
) -> DeliveryStatusCallback:
    """Construct a ``DeliveryStatusCallback`` DTO directly."""
    return DeliveryStatusCallback(
        message_sid=message_sid,
        message_status=message_status,
        to=to,
        error_code=error_code,
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# Helper: invoke the status_callback handler with a mocked Request
# ---------------------------------------------------------------------------


def _make_mock_request(
    message_sid: str,
    message_status: str,
    to: str,
    error_code: str | None = None,
    error_message: str | None = None,
) -> MagicMock:
    """Build a mock FastAPI Request whose state.body contains the form payload."""
    params: dict[str, str] = {
        "MessageSid": message_sid,
        "MessageStatus": message_status,
        "To": to,
    }
    if error_code is not None:
        params["ErrorCode"] = error_code
    if error_message is not None:
        params["ErrorMessage"] = error_message

    body = urlencode(params).encode("utf-8")

    mock_request = MagicMock()
    mock_request.state.body = body
    return mock_request


# ---------------------------------------------------------------------------
# Property 3a: Failure status produces a log entry with phone_number and message_sid
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    message_sid=_message_sid_strategy,
    message_status=_failure_status_strategy,
    phone_number=_phone_number_strategy,
    error_code=_error_code_strategy,
    error_message=_error_message_strategy,
)
@settings(max_examples=100)
async def test_property_3_failure_status_logs_phone_number_and_message_sid(
    message_sid: str,
    message_status: str,
    phone_number: str,
    error_code: str | None,
    error_message: str | None,
) -> None:
    """Property 3: Delivery Failure Logs Contain Required Fields.

    For any ``DeliveryStatusCallback`` with ``message_status`` of ``"failed"``
    or ``"undelivered"``, the webhook handler must emit a log entry that
    contains both ``phone_number`` and ``message_sid`` fields.

    **Validates: Requirements 1.6**
    """
    mock_request = _make_mock_request(
        message_sid=message_sid,
        message_status=message_status,
        to=phone_number,
        error_code=error_code,
        error_message=error_message,
    )

    logged_events: list[dict] = []

    mock_logger = MagicMock()

    def capture_error(event: str, **kwargs: object) -> None:
        logged_events.append({"event": event, **kwargs})

    mock_logger.error.side_effect = capture_error
    mock_logger.info.return_value = None

    with patch("interview_practice_partner.api.routers.webhook.logger", mock_logger):
        await status_callback(mock_request)

    # At least one log entry must have been emitted for a failure status
    assert len(logged_events) >= 1, (
        f"Expected at least one log entry for status={message_status!r}, "
        f"message_sid={message_sid!r}, phone_number={phone_number!r}, "
        f"but no log entries were captured."
    )

    # The log entry must contain both phone_number and message_sid
    failure_log = logged_events[0]

    assert "phone_number" in failure_log, (
        f"Expected log entry to contain 'phone_number' field for "
        f"status={message_status!r}, message_sid={message_sid!r}, "
        f"phone_number={phone_number!r}. "
        f"Got log entry: {failure_log!r}"
    )
    assert "message_sid" in failure_log, (
        f"Expected log entry to contain 'message_sid' field for "
        f"status={message_status!r}, message_sid={message_sid!r}, "
        f"phone_number={phone_number!r}. "
        f"Got log entry: {failure_log!r}"
    )

    # The values must match the input
    assert failure_log["phone_number"] == phone_number, (
        f"Expected phone_number={phone_number!r} in log entry, "
        f"but got {failure_log['phone_number']!r}"
    )
    assert failure_log["message_sid"] == message_sid, (
        f"Expected message_sid={message_sid!r} in log entry, "
        f"but got {failure_log['message_sid']!r}"
    )


# ---------------------------------------------------------------------------
# Property 3b: Non-failure status does NOT produce an error log entry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    message_sid=_message_sid_strategy,
    message_status=_non_failure_status_strategy,
    phone_number=_phone_number_strategy,
)
@settings(max_examples=100)
async def test_property_3_non_failure_status_does_not_log_error(
    message_sid: str,
    message_status: str,
    phone_number: str,
) -> None:
    """Property 3 (inverse): Non-failure statuses must NOT produce an error log entry.

    For any ``DeliveryStatusCallback`` with a non-failure ``message_status``
    (e.g. ``"sent"``, ``"delivered"``), the webhook handler must NOT emit an
    error-level log entry.

    **Validates: Requirements 1.6**
    """
    mock_request = _make_mock_request(
        message_sid=message_sid,
        message_status=message_status,
        to=phone_number,
    )

    error_logged: list[dict] = []

    mock_logger = MagicMock()

    def capture_error(event: str, **kwargs: object) -> None:
        error_logged.append({"event": event, **kwargs})

    mock_logger.error.side_effect = capture_error
    mock_logger.info.return_value = None

    with patch("interview_practice_partner.api.routers.webhook.logger", mock_logger):
        await status_callback(mock_request)

    assert len(error_logged) == 0, (
        f"Expected no error log entries for non-failure status={message_status!r}, "
        f"but got: {error_logged!r}"
    )


# ---------------------------------------------------------------------------
# Property 3c: Both "failed" and "undelivered" are treated as failure statuses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    message_sid=_message_sid_strategy,
    phone_number=_phone_number_strategy,
)
@settings(max_examples=100)
async def test_property_3_both_failed_and_undelivered_trigger_log(
    message_sid: str,
    phone_number: str,
) -> None:
    """Property 3: Both 'failed' and 'undelivered' statuses trigger a failure log.

    Both failure status values must independently produce a log entry with
    ``phone_number`` and ``message_sid`` fields.

    **Validates: Requirements 1.6**
    """
    for failure_status in ("failed", "undelivered"):
        mock_request = _make_mock_request(
            message_sid=message_sid,
            message_status=failure_status,
            to=phone_number,
        )

        logged_events: list[dict] = []

        mock_logger = MagicMock()

        def capture_error(event: str, **kwargs: object) -> None:
            logged_events.append({"event": event, **kwargs})

        mock_logger.error.side_effect = capture_error
        mock_logger.info.return_value = None

        with patch("interview_practice_partner.api.routers.webhook.logger", mock_logger):
            await status_callback(mock_request)

        assert len(logged_events) >= 1, (
            f"Expected a log entry for status={failure_status!r}, "
            f"message_sid={message_sid!r}, phone_number={phone_number!r}"
        )

        log_entry = logged_events[0]
        assert "phone_number" in log_entry, (
            f"Missing 'phone_number' in log for status={failure_status!r}: {log_entry!r}"
        )
        assert "message_sid" in log_entry, (
            f"Missing 'message_sid' in log for status={failure_status!r}: {log_entry!r}"
        )
        assert log_entry["phone_number"] == phone_number, (
            f"phone_number mismatch for status={failure_status!r}: "
            f"expected {phone_number!r}, got {log_entry['phone_number']!r}"
        )
        assert log_entry["message_sid"] == message_sid, (
            f"message_sid mismatch for status={failure_status!r}: "
            f"expected {message_sid!r}, got {log_entry['message_sid']!r}"
        )
