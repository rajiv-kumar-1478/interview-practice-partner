# Feature: interview-practice-partner, Property 22: Structured Logs Contain Correlation Identifiers
"""Property-based tests for correlation ID presence in structured logs.

All log entries emitted during a single call to
``MessageOrchestrationService.handle()`` must share a consistent
``correlation_id`` field derived from the inbound ``MessageSid``.

Validates: Requirements 12.3
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

import structlog
import structlog.contextvars
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from interview_practice_partner.api.schemas import InboundMessage
from interview_practice_partner.domain.enums import Role, Stage
from interview_practice_partner.domain.models import SessionState
from interview_practice_partner.services.orchestration import MessageOrchestrationService

# ---------------------------------------------------------------------------
# Custom log capture that includes structlog context vars
# ---------------------------------------------------------------------------


@contextmanager
def capture_logs_with_contextvars() -> Generator[list[dict], None, None]:
    """Capture structlog log entries including context vars (e.g. ``correlation_id``).

    ``structlog.testing.capture_logs()`` bypasses the normal processor chain
    and does NOT merge context vars.  This context manager configures structlog
    with a processor chain that explicitly merges context vars before capturing,
    so that ``correlation_id`` (bound via
    ``structlog.contextvars.bind_contextvars``) appears in every captured entry.
    """
    cap: list[dict] = []

    def capturing_processor(
        logger: object, method: str, event_dict: dict
    ) -> dict:
        # Merge context vars (correlation_id, etc.) into the event dict first
        structlog.contextvars.merge_contextvars(logger, method, event_dict)
        cap.append(event_dict.copy())
        # Raise DropEvent so structlog doesn't try to render/output anything
        raise structlog.DropEvent()

    old_processors = structlog.get_config().get("processors", [])
    structlog.configure(processors=[capturing_processor])
    try:
        yield cap
    finally:
        structlog.configure(processors=old_processors)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Twilio MessageSid values: alphanumeric strings, typically 34 chars starting
# with "SM", but we test arbitrary non-empty strings to cover the property
# universally.
_message_sid_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="-_",
    ),
    min_size=1,
    max_size=64,
)

# E.164-style WhatsApp phone numbers
_phone_number_strategy = st.one_of(
    st.from_regex(r"whatsapp:\+1[0-9]{10}", fullmatch=True),
    st.from_regex(r"whatsapp:\+44[0-9]{10}", fullmatch=True),
)

# Arbitrary non-empty message bodies
_body_strategy = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service() -> tuple[
    MessageOrchestrationService,
    MagicMock,  # session_repo
    MagicMock,  # idempotency_repo
    MagicMock,  # session_service
    MagicMock,  # messaging_service
]:
    """Build a ``MessageOrchestrationService`` with all external deps mocked."""
    from datetime import datetime, timezone

    now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    updated_session = SessionState(
        session_id="sess-prop22-001",
        phone_number="whatsapp:+15550001234",
        stage=Stage.ROLE_SELECTION,
        role=Role.UNKNOWN,
        created_at=now,
        updated_at=now,
    )

    idempotency_repo = MagicMock()
    idempotency_repo.is_processed = AsyncMock(return_value=False)
    idempotency_repo.mark_processed = AsyncMock()

    session_repo = MagicMock()
    session_repo.get = AsyncMock(return_value=None)  # new user each time
    session_repo.save = AsyncMock()

    session_service = MagicMock()
    session_service.transition = AsyncMock(
        return_value=("Which role would you like to practise for?", updated_session)
    )

    messaging_service = MagicMock()
    messaging_service.send_message = AsyncMock()

    svc = MessageOrchestrationService(
        session_repository=session_repo,
        idempotency_repository=idempotency_repo,
        session_service=session_service,
        messaging_service=messaging_service,
    )
    return svc, session_repo, idempotency_repo, session_service, messaging_service


def _make_inbound_message(
    message_sid: str,
    from_number: str,
    body: str,
) -> InboundMessage:
    return InboundMessage(
        message_sid=message_sid,
        from_number=from_number,
        to_number="whatsapp:+14155238886",
        body=body,
        num_media=0,
    )


# ---------------------------------------------------------------------------
# Property 22: Structured Logs Contain Correlation Identifiers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    message_sid=_message_sid_strategy,
    from_number=_phone_number_strategy,
    body=_body_strategy,
)
@settings(max_examples=100)
async def test_property_22_all_log_entries_share_correlation_id(
    message_sid: str,
    from_number: str,
    body: str,
) -> None:
    """Property 22: Structured Logs Contain Correlation Identifiers.

    All log entries emitted during a single call to
    ``MessageOrchestrationService.handle()`` must:

    1. Contain a ``correlation_id`` field.
    2. Share the same ``correlation_id`` value across all entries.
    3. Have a ``correlation_id`` equal to the inbound ``MessageSid``.

    **Validates: Requirements 12.3**
    """
    # Ensure structlog context vars are clean before each test run
    structlog.contextvars.clear_contextvars()

    svc, _, _, _, _ = _make_service()
    msg = _make_inbound_message(
        message_sid=message_sid,
        from_number=from_number,
        body=body,
    )

    # capture_logs_with_contextvars() configures structlog with a processor
    # that explicitly merges context vars (including correlation_id) before
    # capturing each log entry.  This is necessary because
    # structlog.testing.capture_logs() bypasses merge_contextvars.
    with capture_logs_with_contextvars() as captured:
        await svc.handle(msg)

    # The orchestration service emits several log entries during a normal
    # happy-path flow (new_session_created, session_loaded, transition_complete,
    # session_persisted, reply_sent, message_sid_marked_processed).
    assert len(captured) >= 1, (
        f"Expected at least one log entry from handle(), "
        f"but got none for message_sid={message_sid!r}"
    )

    # Every log entry must contain a correlation_id field
    entries_missing_correlation_id = [
        entry for entry in captured if "correlation_id" not in entry
    ]
    assert len(entries_missing_correlation_id) == 0, (
        f"The following log entries are missing 'correlation_id': "
        f"{entries_missing_correlation_id!r}. "
        f"All {len(captured)} entries: {captured!r}"
    )

    # All correlation_id values must be identical
    correlation_ids = {entry["correlation_id"] for entry in captured}
    assert len(correlation_ids) == 1, (
        f"Expected all log entries to share the same correlation_id, "
        f"but found multiple values: {correlation_ids!r}. "
        f"All entries: {captured!r}"
    )

    # The correlation_id must equal the MessageSid
    actual_correlation_id = next(iter(correlation_ids))
    assert actual_correlation_id == message_sid, (
        f"Expected correlation_id={message_sid!r} (the MessageSid), "
        f"but got {actual_correlation_id!r}. "
        f"All entries: {captured!r}"
    )


@pytest.mark.asyncio
@given(
    message_sid=_message_sid_strategy,
    from_number=_phone_number_strategy,
    body=_body_strategy,
)
@settings(max_examples=100)
async def test_property_22_correlation_id_cleared_after_handle(
    message_sid: str,
    from_number: str,
    body: str,
) -> None:
    """Property 22 (cleanup): structlog context vars are cleared after handle() completes.

    The ``correlation_id`` bound during one request must not leak into
    subsequent requests.  After ``handle()`` returns, the structlog context
    vars must be empty (or at least not contain ``correlation_id``).

    **Validates: Requirements 12.3**
    """
    structlog.contextvars.clear_contextvars()

    svc, _, _, _, _ = _make_service()
    msg = _make_inbound_message(
        message_sid=message_sid,
        from_number=from_number,
        body=body,
    )

    with capture_logs_with_contextvars():
        await svc.handle(msg)

    # After handle() completes, the context vars must be cleared so that
    # the correlation_id from this request does not bleed into the next one.
    context_after = structlog.contextvars.get_contextvars()
    assert "correlation_id" not in context_after, (
        f"Expected 'correlation_id' to be cleared from structlog context vars "
        f"after handle() completed, but found: {context_after!r}"
    )
