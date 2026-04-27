# Feature: interview-practice-partner, Property 5 & 6: New Session Creation and SessionState Structural Invariants
"""Property-based tests for new session creation and SessionState structural invariants.

Validates: Requirements 2.2, 2.4, 3.1
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from interview_practice_partner.domain.enums import Role, Stage
from interview_practice_partner.domain.models import FeedbackReport, SessionState
from interview_practice_partner.repositories.redis_session import RedisSessionRepository
from interview_practice_partner.services.session import SessionService

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# E.164 phone numbers: + followed by 7–15 digits
_e164_strategy = st.from_regex(r"\+[1-9]\d{6,14}", fullmatch=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_service() -> SessionService:
    """Build a SessionService with mocked LLM and sub-services.

    The LLM returns a low-confidence role-selection response so that the
    INIT → ROLE_SELECTION transition completes without hitting a real API.
    """
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = (
        '{"role": "unknown", "confidence": "low", "message": "Which role?"}'
    )

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_role_selection_prompt.return_value = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]

    mock_interview = AsyncMock()
    mock_interview.generate_question.return_value = "Tell me about yourself."

    mock_feedback = AsyncMock()

    return SessionService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
        interview_service=mock_interview,
        feedback_service=mock_feedback,
    )


def _new_session(phone_number: str) -> SessionState:
    """Create a brand-new SessionState at INIT stage (mirrors orchestration logic)."""
    now = datetime.now(tz=timezone.utc)
    return SessionState(
        session_id=str(uuid.uuid4()),
        phone_number=phone_number,
        stage=Stage.INIT,
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Property 5: New Session Created for Unknown Phone Number
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(phone_number=_e164_strategy)
@settings(max_examples=100)
async def test_property_5_unknown_phone_number_returns_none_from_repository(
    phone_number: str,
) -> None:
    """Property 5 (part A): Repository returns None for any unknown E.164 number.

    For any E.164 phone number that has never been stored, the session
    repository MUST return None — confirming that no stale state exists
    and a fresh session will be created.

    **Validates: Requirements 2.2, 3.1**
    """
    redis_client = fakeredis.aioredis.FakeRedis()
    repo = RedisSessionRepository(redis_client=redis_client, ttl_seconds=86400)

    result = await repo.get(phone_number)

    assert result is None, (
        f"Expected None for unknown phone number {phone_number!r}, got {result!r}"
    )


@pytest.mark.asyncio
@given(phone_number=_e164_strategy)
@settings(max_examples=100)
async def test_property_5_new_session_created_with_role_selection_stage(
    phone_number: str,
) -> None:
    """Property 5 (part B): New session for unknown phone number has stage=ROLE_SELECTION.

    For any E.164 phone number without existing state, the orchestration
    pipeline creates a new SessionState at INIT and immediately transitions
    it to ROLE_SELECTION on the first inbound message.

    **Validates: Requirements 2.2, 3.1**
    """
    # Simulate the orchestration pipeline:
    # 1. repo.get() returns None (no existing session)
    # 2. _new_session() creates a fresh INIT session
    # 3. SessionService.transition() moves it to ROLE_SELECTION
    session = _new_session(phone_number)

    assert session.stage == Stage.INIT, (
        f"Expected new session to start at INIT, got {session.stage}"
    )

    service = _make_session_service()
    _reply, updated_session = await service.transition(session, "Hello, I want to practise.")

    assert updated_session.stage == Stage.ROLE_SELECTION, (
        f"Expected stage=ROLE_SELECTION after first message for {phone_number!r}, "
        f"got {updated_session.stage}"
    )
    assert updated_session.phone_number == phone_number, (
        f"Expected phone_number to be preserved as {phone_number!r}, "
        f"got {updated_session.phone_number!r}"
    )


@given(
    session_id=st.text(min_size=1),
    phone_number=st.text(min_size=1),
    stage=st.sampled_from(list(Stage)),
    role=st.sampled_from(list(Role)),
    off_topic_count=st.integers(min_value=0),
    consecutive_out_of_scope_count=st.integers(min_value=0),
    clarification_turn_count=st.integers(min_value=0),
    requested_short_session=st.booleans(),
    is_complete=st.booleans(),
    created_at=st.datetimes(),
    updated_at=st.datetimes(),
)
@settings(max_examples=100)
def test_session_state_contains_all_required_fields(
    session_id: str,
    phone_number: str,
    stage: Stage,
    role: Role,
    off_topic_count: int,
    consecutive_out_of_scope_count: int,
    clarification_turn_count: int,
    requested_short_session: bool,
    is_complete: bool,
    created_at: datetime,
    updated_at: datetime,
) -> None:
    """Property 6: SessionState Contains All Required Fields.

    For any valid combination of inputs used to construct a SessionState,
    the resulting object always contains all required fields with correct
    types and valid default values.

    Validates: Requirements 2.4
    """
    session = SessionState(
        session_id=session_id,
        phone_number=phone_number,
        stage=stage,
        role=role,
        off_topic_count=off_topic_count,
        consecutive_out_of_scope_count=consecutive_out_of_scope_count,
        clarification_turn_count=clarification_turn_count,
        requested_short_session=requested_short_session,
        is_complete=is_complete,
        created_at=created_at,
        updated_at=updated_at,
    )

    # session_id is a non-empty string
    assert isinstance(session.session_id, str)
    assert len(session.session_id) > 0

    # phone_number is a non-empty string
    assert isinstance(session.phone_number, str)
    assert len(session.phone_number) > 0

    # stage is a valid Stage enum member
    assert isinstance(session.stage, Stage)
    assert session.stage in Stage

    # role is a valid Role enum member
    assert isinstance(session.role, Role)
    assert session.role in Role

    # questions is a list
    assert isinstance(session.questions, list)

    # responses is a list
    assert isinstance(session.responses, list)

    # off_topic_count is an int >= 0
    assert isinstance(session.off_topic_count, int)
    assert session.off_topic_count >= 0

    # consecutive_out_of_scope_count is an int >= 0
    assert isinstance(session.consecutive_out_of_scope_count, int)
    assert session.consecutive_out_of_scope_count >= 0

    # clarification_turn_count is an int >= 0
    assert isinstance(session.clarification_turn_count, int)
    assert session.clarification_turn_count >= 0

    # requested_short_session is a bool
    assert isinstance(session.requested_short_session, bool)

    # is_complete is a bool
    assert isinstance(session.is_complete, bool)

    # feedback_report is either None or a FeedbackReport instance
    assert session.feedback_report is None or isinstance(session.feedback_report, FeedbackReport)

    # completed_at is either None or a datetime
    assert session.completed_at is None or isinstance(session.completed_at, datetime)

    # context_summary is either None or a str
    assert session.context_summary is None or isinstance(session.context_summary, str)

    # created_at is a datetime
    assert isinstance(session.created_at, datetime)

    # updated_at is a datetime
    assert isinstance(session.updated_at, datetime)
