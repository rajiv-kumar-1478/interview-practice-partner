"""Unit tests for MessageOrchestrationService.

Covers:
- Happy-path flow: idempotency check → session load → transition → persist → send
- Duplicate MessageSid is suppressed (no state mutation, no send)
- Redis unavailability returns error message to user without processing
- LLM failure returns fallback message to user

Requirements: 1.7, 2.6, 12.6
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from interview_practice_partner.api.schemas import InboundMessage
from interview_practice_partner.domain.enums import Role, Stage
from interview_practice_partner.domain.exceptions import (
    LLMError,
    SessionStoreUnavailableError,
)
from interview_practice_partner.domain.models import SessionState
from interview_practice_partner.services.orchestration import MessageOrchestrationService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

_PHONE = "whatsapp:+15550001234"
_MESSAGE_SID = "SM1234567890abcdef"


def _make_inbound_message(
    message_sid: str = _MESSAGE_SID,
    from_number: str = _PHONE,
    body: str = "Hello, I want to practise for a software engineer interview.",
) -> InboundMessage:
    return InboundMessage(
        message_sid=message_sid,
        from_number=from_number,
        to_number="whatsapp:+14155238886",
        body=body,
        num_media=0,
    )


def _make_session(
    phone: str = _PHONE,
    stage: Stage = Stage.INIT,
) -> SessionState:
    return SessionState(
        session_id="sess-test-0001",
        phone_number=phone,
        stage=stage,
        role=Role.UNKNOWN,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_updated_session(phone: str = _PHONE) -> SessionState:
    """Return a session that looks like it has been transitioned."""
    return SessionState(
        session_id="sess-test-0001",
        phone_number=phone,
        stage=Stage.ROLE_SELECTION,
        role=Role.UNKNOWN,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_service(
    *,
    is_processed: bool = False,
    existing_session: SessionState | None = None,
    transition_reply: str = "Which role would you like to practise for?",
    transition_session: SessionState | None = None,
    session_store_error_on: str | None = None,
    llm_error_on_transition: bool = False,
) -> tuple[
    MessageOrchestrationService,
    MagicMock,  # session_repo
    MagicMock,  # idempotency_repo
    MagicMock,  # session_service
    MagicMock,  # messaging_service
]:
    """Build a MessageOrchestrationService with all dependencies mocked.

    Args:
        is_processed: Whether the idempotency check returns True (duplicate).
        existing_session: Session returned by session_repo.get (None = new user).
        transition_reply: The reply text returned by session_service.transition.
        transition_session: The updated session returned by session_service.transition.
            Defaults to a freshly-built updated session.
        session_store_error_on: If set to "get", "save", or "is_processed",
            that call raises SessionStoreUnavailableError.
        llm_error_on_transition: If True, session_service.transition raises LLMError.
    """
    if transition_session is None:
        transition_session = _make_updated_session()

    # --- idempotency_repo ---
    idempotency_repo = MagicMock()
    if session_store_error_on == "is_processed":
        idempotency_repo.is_processed = AsyncMock(
            side_effect=SessionStoreUnavailableError("Redis down")
        )
    else:
        idempotency_repo.is_processed = AsyncMock(return_value=is_processed)
    idempotency_repo.mark_processed = AsyncMock()

    # --- session_repo ---
    session_repo = MagicMock()
    if session_store_error_on == "get":
        session_repo.get = AsyncMock(
            side_effect=SessionStoreUnavailableError("Redis down")
        )
    else:
        session_repo.get = AsyncMock(return_value=existing_session)

    if session_store_error_on == "save":
        session_repo.save = AsyncMock(
            side_effect=SessionStoreUnavailableError("Redis down")
        )
    else:
        session_repo.save = AsyncMock()

    # --- session_service ---
    session_service = MagicMock()
    if llm_error_on_transition:
        session_service.transition = AsyncMock(
            side_effect=LLMError("LLM API call failed")
        )
    else:
        session_service.transition = AsyncMock(
            return_value=(transition_reply, transition_session)
        )

    # --- messaging_service ---
    messaging_service = MagicMock()
    messaging_service.send_message = AsyncMock()

    svc = MessageOrchestrationService(
        session_repository=session_repo,
        idempotency_repository=idempotency_repo,
        session_service=session_service,
        messaging_service=messaging_service,
    )
    return svc, session_repo, idempotency_repo, session_service, messaging_service


# ===========================================================================
# Happy-path flow — Requirement 1.7, 2.6
# ===========================================================================


class TestHappyPath:
    """The full pipeline executes in the correct order for a new, non-duplicate message."""

    async def test_idempotency_check_is_called_first(self) -> None:
        """is_processed is called with the correct MessageSid."""
        svc, _, idempotency_repo, _, _ = _make_service()
        msg = _make_inbound_message()
        await svc.handle(msg)
        idempotency_repo.is_processed.assert_awaited_once_with(_MESSAGE_SID)

    async def test_session_is_loaded_after_idempotency_check(self) -> None:
        """session_repo.get is called with the sender's phone number."""
        svc, session_repo, _, _, _ = _make_service()
        msg = _make_inbound_message()
        await svc.handle(msg)
        session_repo.get.assert_awaited_once_with(_PHONE)

    async def test_transition_is_called_with_session_and_body(self) -> None:
        """session_service.transition receives the loaded session and message body."""
        existing = _make_session()
        svc, _, _, session_service, _ = _make_service(existing_session=existing)
        msg = _make_inbound_message(body="I want to practise for sales.")
        await svc.handle(msg)
        session_service.transition.assert_awaited_once_with(
            existing, "I want to practise for sales.",
            num_media=0,
            media_content_type=None,
            media_url=None,
        )

    async def test_new_session_created_when_none_exists(self) -> None:
        """When session_repo.get returns None, transition is called with a new session."""
        svc, _, _, session_service, _ = _make_service(existing_session=None)
        msg = _make_inbound_message()
        await svc.handle(msg)
        # transition must have been called
        session_service.transition.assert_awaited_once()
        # The session passed to transition should have the correct phone number
        called_session: SessionState = session_service.transition.call_args[0][0]
        assert called_session.phone_number == _PHONE
        assert called_session.stage == Stage.INIT

    async def test_updated_session_is_persisted(self) -> None:
        """session_repo.save is called with the updated session returned by transition."""
        updated = _make_updated_session()
        svc, session_repo, _, _, _ = _make_service(transition_session=updated)
        msg = _make_inbound_message()
        await svc.handle(msg)
        session_repo.save.assert_awaited_once_with(updated)

    async def test_reply_is_sent_to_sender(self) -> None:
        """messaging_service.send_message is called with the reply and sender's number."""
        reply_text = "Which role would you like to practise for?"
        svc, _, _, _, messaging_service = _make_service(transition_reply=reply_text)
        msg = _make_inbound_message()
        await svc.handle(msg)
        messaging_service.send_message.assert_awaited_once_with(reply_text, _PHONE, media_url=None)

    async def test_message_sid_is_marked_processed_after_send(self) -> None:
        """idempotency_repo.mark_processed is called with the MessageSid after sending."""
        svc, _, idempotency_repo, _, _ = _make_service()
        msg = _make_inbound_message()
        await svc.handle(msg)
        idempotency_repo.mark_processed.assert_awaited_once_with(_MESSAGE_SID)

    async def test_pipeline_order_save_before_send(self) -> None:
        """Session is persisted before the reply is sent to the user."""
        call_order: list[str] = []

        updated = _make_updated_session()
        svc, session_repo, _, _, messaging_service = _make_service(
            transition_session=updated
        )

        async def record_save(session: SessionState) -> None:
            call_order.append("save")

        async def record_send(text: str, to: str, media_url: str | None = None) -> None:
            call_order.append("send")

        session_repo.save.side_effect = record_save
        messaging_service.send_message.side_effect = record_send

        await svc.handle(_make_inbound_message())

        assert call_order.index("save") < call_order.index("send"), (
            "Session must be persisted before the reply is sent"
        )

    async def test_mark_processed_called_after_send(self) -> None:
        """mark_processed is called after the reply is sent."""
        call_order: list[str] = []

        svc, _, idempotency_repo, _, messaging_service = _make_service()

        async def record_send(text: str, to: str, media_url: str | None = None) -> None:
            call_order.append("send")

        async def record_mark(sid: str) -> None:
            call_order.append("mark")

        messaging_service.send_message.side_effect = record_send
        idempotency_repo.mark_processed.side_effect = record_mark

        await svc.handle(_make_inbound_message())

        assert call_order.index("send") < call_order.index("mark"), (
            "mark_processed must be called after the reply is sent"
        )


# ===========================================================================
# Duplicate MessageSid suppression — Requirement 1.7
# ===========================================================================


class TestDuplicateSuppression:
    """A duplicate MessageSid must be silently dropped with no side effects."""

    async def test_duplicate_does_not_call_session_get(self) -> None:
        """When is_processed returns True, session_repo.get is never called."""
        svc, session_repo, _, _, _ = _make_service(is_processed=True)
        await svc.handle(_make_inbound_message())
        session_repo.get.assert_not_awaited()

    async def test_duplicate_does_not_call_transition(self) -> None:
        """When is_processed returns True, session_service.transition is never called."""
        svc, _, _, session_service, _ = _make_service(is_processed=True)
        await svc.handle(_make_inbound_message())
        session_service.transition.assert_not_awaited()

    async def test_duplicate_does_not_persist_session(self) -> None:
        """When is_processed returns True, session_repo.save is never called."""
        svc, session_repo, _, _, _ = _make_service(is_processed=True)
        await svc.handle(_make_inbound_message())
        session_repo.save.assert_not_awaited()

    async def test_duplicate_does_not_send_reply(self) -> None:
        """When is_processed returns True, no outbound message is sent."""
        svc, _, _, _, messaging_service = _make_service(is_processed=True)
        await svc.handle(_make_inbound_message())
        messaging_service.send_message.assert_not_awaited()

    async def test_duplicate_does_not_mark_processed_again(self) -> None:
        """When is_processed returns True, mark_processed is not called again."""
        svc, _, idempotency_repo, _, _ = _make_service(is_processed=True)
        await svc.handle(_make_inbound_message())
        idempotency_repo.mark_processed.assert_not_awaited()

    async def test_second_call_with_same_sid_is_suppressed(self) -> None:
        """Simulates two webhook deliveries of the same message."""
        # First call: not processed yet
        svc, session_repo, idempotency_repo, _, messaging_service = _make_service(
            is_processed=False
        )
        msg = _make_inbound_message()
        await svc.handle(msg)
        assert messaging_service.send_message.await_count == 1
        assert session_repo.save.await_count == 1

        # Second call: now processed
        svc2, session_repo2, _, _, messaging_service2 = _make_service(is_processed=True)
        await svc2.handle(msg)
        messaging_service2.send_message.assert_not_awaited()
        session_repo2.save.assert_not_awaited()


# ===========================================================================
# Redis unavailability — Requirement 2.6
# ===========================================================================


class TestRedisUnavailability:
    """When Redis is unavailable, an error message is sent and processing stops."""

    async def test_redis_down_on_is_processed_sends_error_message(self) -> None:
        """SessionStoreUnavailableError from is_processed triggers error message to user."""
        svc, _, _, _, messaging_service = _make_service(
            session_store_error_on="is_processed"
        )
        await svc.handle(_make_inbound_message())
        messaging_service.send_message.assert_awaited_once()
        sent_text: str = messaging_service.send_message.call_args[0][0]
        # The error message should be informative (not empty, not a generic crash message)
        assert len(sent_text) > 0
        assert "try again" in sent_text.lower() or "trouble" in sent_text.lower()

    async def test_redis_down_on_is_processed_sends_to_correct_number(self) -> None:
        """Error message is sent to the original sender's phone number."""
        svc, _, _, _, messaging_service = _make_service(
            session_store_error_on="is_processed"
        )
        await svc.handle(_make_inbound_message())
        sent_to: str = messaging_service.send_message.call_args[0][1]
        assert sent_to == _PHONE

    async def test_redis_down_on_is_processed_does_not_call_transition(self) -> None:
        """When is_processed raises, transition is never called."""
        svc, _, _, session_service, _ = _make_service(
            session_store_error_on="is_processed"
        )
        await svc.handle(_make_inbound_message())
        session_service.transition.assert_not_awaited()

    async def test_redis_down_on_session_get_sends_error_message(self) -> None:
        """SessionStoreUnavailableError from session_repo.get triggers error message."""
        svc, _, _, _, messaging_service = _make_service(
            session_store_error_on="get"
        )
        await svc.handle(_make_inbound_message())
        messaging_service.send_message.assert_awaited_once()
        sent_text: str = messaging_service.send_message.call_args[0][0]
        assert len(sent_text) > 0

    async def test_redis_down_on_session_get_does_not_call_transition(self) -> None:
        """When session_repo.get raises, transition is never called."""
        svc, _, _, session_service, _ = _make_service(
            session_store_error_on="get"
        )
        await svc.handle(_make_inbound_message())
        session_service.transition.assert_not_awaited()

    async def test_redis_down_on_session_save_sends_error_message(self) -> None:
        """SessionStoreUnavailableError from session_repo.save triggers error message."""
        svc, _, _, _, messaging_service = _make_service(
            session_store_error_on="save"
        )
        await svc.handle(_make_inbound_message())
        messaging_service.send_message.assert_awaited_once()
        sent_text: str = messaging_service.send_message.call_args[0][0]
        assert len(sent_text) > 0

    async def test_redis_down_on_session_save_does_not_send_normal_reply(self) -> None:
        """When save raises, the normal interview reply is NOT sent — only the error message."""
        normal_reply = "Which role would you like to practise for?"
        svc, _, _, _, messaging_service = _make_service(
            transition_reply=normal_reply,
            session_store_error_on="save",
        )
        await svc.handle(_make_inbound_message())
        # Only one send call — the error message, not the normal reply
        assert messaging_service.send_message.await_count == 1
        sent_text: str = messaging_service.send_message.call_args[0][0]
        assert sent_text != normal_reply

    async def test_redis_error_message_is_informative_not_generic_crash(self) -> None:
        """The Redis error message is specific to session unavailability, not a generic crash."""
        from interview_practice_partner.services.orchestration import (
            _MSG_SESSION_UNAVAILABLE,
            _MSG_UNEXPECTED_ERROR,
        )

        svc, _, _, _, messaging_service = _make_service(
            session_store_error_on="get"
        )
        await svc.handle(_make_inbound_message())
        sent_text: str = messaging_service.send_message.call_args[0][0]
        # Should be the session-unavailable message, not the generic unexpected error
        assert sent_text == _MSG_SESSION_UNAVAILABLE
        assert sent_text != _MSG_UNEXPECTED_ERROR


# ===========================================================================
# LLM failure — Requirement 12.6
# ===========================================================================


class TestLLMFailure:
    """When the LLM fails after retries, a fallback message is sent to the user."""

    async def test_llm_error_sends_fallback_message(self) -> None:
        """LLMError from session_service.transition triggers a fallback message."""
        svc, _, _, _, messaging_service = _make_service(llm_error_on_transition=True)
        await svc.handle(_make_inbound_message())
        messaging_service.send_message.assert_awaited_once()
        sent_text: str = messaging_service.send_message.call_args[0][0]
        assert len(sent_text) > 0

    async def test_llm_error_sends_to_correct_number(self) -> None:
        """Fallback message is sent to the original sender's phone number."""
        svc, _, _, _, messaging_service = _make_service(llm_error_on_transition=True)
        await svc.handle(_make_inbound_message())
        sent_to: str = messaging_service.send_message.call_args[0][1]
        assert sent_to == _PHONE

    async def test_llm_error_does_not_persist_session(self) -> None:
        """When transition raises LLMError, session_repo.save is never called."""
        svc, session_repo, _, _, _ = _make_service(llm_error_on_transition=True)
        await svc.handle(_make_inbound_message())
        session_repo.save.assert_not_awaited()

    async def test_llm_error_does_not_mark_processed(self) -> None:
        """When transition raises LLMError, mark_processed is never called."""
        svc, _, idempotency_repo, _, _ = _make_service(llm_error_on_transition=True)
        await svc.handle(_make_inbound_message())
        idempotency_repo.mark_processed.assert_not_awaited()

    async def test_llm_error_message_is_specific_fallback(self) -> None:
        """The LLM error message is the specific fallback, not the generic crash message."""
        from interview_practice_partner.services.orchestration import (
            _MSG_LLM_ERROR,
            _MSG_UNEXPECTED_ERROR,
        )

        svc, _, _, _, messaging_service = _make_service(llm_error_on_transition=True)
        await svc.handle(_make_inbound_message())
        sent_text: str = messaging_service.send_message.call_args[0][0]
        assert sent_text == _MSG_LLM_ERROR
        assert sent_text != _MSG_UNEXPECTED_ERROR

    async def test_llm_error_message_is_not_empty(self) -> None:
        """The fallback message is non-empty and user-friendly."""
        svc, _, _, _, messaging_service = _make_service(llm_error_on_transition=True)
        await svc.handle(_make_inbound_message())
        sent_text: str = messaging_service.send_message.call_args[0][0]
        assert len(sent_text.strip()) > 0


# ===========================================================================
# Unhandled exception — Requirement 12.6
# ===========================================================================


class TestUnhandledException:
    """Unexpected exceptions are caught, logged, and a generic error message is sent."""

    async def test_unexpected_exception_sends_generic_error(self) -> None:
        """An unhandled exception during transition triggers the generic error message."""
        from interview_practice_partner.services.orchestration import _MSG_UNEXPECTED_ERROR

        svc, _, _, session_service, messaging_service = _make_service()
        session_service.transition = AsyncMock(
            side_effect=RuntimeError("Something totally unexpected")
        )

        await svc.handle(_make_inbound_message())

        messaging_service.send_message.assert_awaited_once()
        sent_text: str = messaging_service.send_message.call_args[0][0]
        assert sent_text == _MSG_UNEXPECTED_ERROR

    async def test_unexpected_exception_sends_to_correct_number(self) -> None:
        """Generic error message is sent to the original sender's phone number."""
        svc, _, _, session_service, messaging_service = _make_service()
        session_service.transition = AsyncMock(
            side_effect=ValueError("Unexpected value error")
        )

        await svc.handle(_make_inbound_message())

        sent_to: str = messaging_service.send_message.call_args[0][1]
        assert sent_to == _PHONE

    async def test_unexpected_exception_does_not_persist_session(self) -> None:
        """An unhandled exception does not result in a session save."""
        svc, session_repo, _, session_service, _ = _make_service()
        session_service.transition = AsyncMock(
            side_effect=RuntimeError("Boom")
        )

        await svc.handle(_make_inbound_message())

        session_repo.save.assert_not_awaited()


# ===========================================================================
# Property test — Requirement 1.5
# Feature: interview-practice-partner, Property 2: Outbound Messages Are Sent to the Correct Phone Number
# ===========================================================================

# Strategy: generate arbitrary E.164-formatted WhatsApp phone numbers.
# E.164 format: +<country_code><subscriber_number>, 7–15 digits total.
# WhatsApp prefix: "whatsapp:" prepended to the E.164 number.
_e164_phone_numbers = st.builds(
    lambda country_code, subscriber: f"whatsapp:+{country_code}{subscriber}",
    country_code=st.integers(min_value=1, max_value=999).map(str),
    subscriber=st.text(
        alphabet="0123456789",
        min_size=6,
        max_size=12,
    ),
)


class TestOutboundPhoneNumberCorrectness:
    """Property 2: Outbound Messages Are Sent to the Correct Phone Number.

    **Validates: Requirements 1.5**

    For any inbound from_number, the outbound reply uses that same number as
    destination. This property holds across all valid E.164 WhatsApp phone
    number formats.
    """

    @given(from_number=_e164_phone_numbers)
    @settings(max_examples=100)
    def test_outbound_reply_uses_inbound_from_number(self, from_number: str) -> None:
        """For any inbound from_number, send_message is called with that exact number."""
        import asyncio

        # Build a fresh service for each generated phone number
        svc, _, _, _, messaging_service = _make_service()

        msg = InboundMessage(
            message_sid="SM_property_test_001",
            from_number=from_number,
            to_number="whatsapp:+14155238886",
            body="Hello, I want to practise for a software engineer interview.",
            num_media=0,
        )

        asyncio.run(svc.handle(msg))

        # The outbound send_message must have been called exactly once
        messaging_service.send_message.assert_awaited_once()

        # The destination (second positional argument) must equal the inbound from_number
        call_args_positional = messaging_service.send_message.call_args[0]
        call_kwargs = messaging_service.send_message.call_args[1]
        destination = call_args_positional[1] if len(call_args_positional) > 1 else call_kwargs.get("to")

        assert destination == from_number, (
            f"Expected send_message destination to be {from_number!r}, "
            f"but got {destination!r}"
        )
