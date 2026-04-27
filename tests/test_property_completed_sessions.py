# Feature: interview-practice-partner, Property 7: Completed Sessions Are Marked Correctly
"""Property-based tests for completed session persistence invariants.

For any session transitioned to COMPLETE, the persisted state must have
``is_complete=True`` and a non-null ``completed_at`` timestamp.

Validates: Requirements 2.5
"""

from datetime import datetime, timezone

import fakeredis.aioredis
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from interview_practice_partner.domain.enums import Role, Stage
from interview_practice_partner.domain.models import SessionState
from interview_practice_partner.repositories.redis_session import RedisSessionRepository


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# E.164-style phone numbers: +1 followed by 10 digits
_phone_strategy = st.from_regex(r"\+1[2-9]\d{9}", fullmatch=True)

# Arbitrary non-empty UUIDs / identifiers
_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-"),
    min_size=1,
    max_size=36,
)

_datetime_strategy = st.datetimes(timezones=st.just(timezone.utc))


def _completed_session_strategy() -> st.SearchStrategy[SessionState]:
    """Build a ``SessionState`` that has been transitioned to COMPLETE."""
    return st.builds(
        SessionState,
        session_id=_id_strategy,
        phone_number=_phone_strategy,
        stage=st.just(Stage.COMPLETE),
        role=st.sampled_from(list(Role)),
        off_topic_count=st.integers(min_value=0, max_value=20),
        consecutive_out_of_scope_count=st.integers(min_value=0, max_value=10),
        clarification_turn_count=st.integers(min_value=0, max_value=5),
        requested_short_session=st.booleans(),
        # Completed sessions must have is_complete=True and a completed_at timestamp
        is_complete=st.just(True),
        completed_at=_datetime_strategy.map(lambda dt: dt),
        created_at=_datetime_strategy,
        updated_at=_datetime_strategy,
        questions=st.just([]),
        responses=st.just([]),
        feedback_report=st.none(),
        context_summary=st.none() | st.text(min_size=1, max_size=200),
    )


# ---------------------------------------------------------------------------
# Property 7: Completed Sessions Are Marked Correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_completed_session_strategy())
@settings(max_examples=100)
async def test_property_7_completed_sessions_are_marked_correctly(
    session: SessionState,
) -> None:
    """Property 7: Completed Sessions Are Marked Correctly.

    For any session transitioned to COMPLETE, the state persisted to and
    retrieved from the session store must satisfy:
      - ``is_complete`` is ``True``
      - ``completed_at`` is not ``None``
      - ``stage`` is ``Stage.COMPLETE``

    This validates that the persistence layer faithfully round-trips the
    completion markers required by Requirement 2.5.

    Validates: Requirements 2.5
    """
    redis_client = fakeredis.aioredis.FakeRedis()
    repo = RedisSessionRepository(redis_client=redis_client, ttl_seconds=86400)

    # Persist the completed session
    await repo.save(session)

    # Retrieve it back from the store
    retrieved = await repo.get(session.phone_number)

    assert retrieved is not None, (
        f"Expected a session for {session.phone_number} but got None"
    )

    # Core invariants for a completed session (Requirement 2.5)
    assert retrieved.is_complete is True, (
        f"Expected is_complete=True for a COMPLETE session, got {retrieved.is_complete}"
    )
    assert retrieved.completed_at is not None, (
        "Expected completed_at to be non-null for a COMPLETE session"
    )
    assert isinstance(retrieved.completed_at, datetime), (
        f"Expected completed_at to be a datetime, got {type(retrieved.completed_at)}"
    )
    assert retrieved.stage == Stage.COMPLETE, (
        f"Expected stage=COMPLETE after round-trip, got {retrieved.stage}"
    )


@pytest.mark.asyncio
@given(session=_completed_session_strategy())
@settings(max_examples=100)
async def test_property_7_completed_session_round_trip_preserves_all_completion_fields(
    session: SessionState,
) -> None:
    """Property 7 (extended): Round-trip preserves all completion-related fields exactly.

    Ensures that ``is_complete``, ``completed_at``, and ``stage`` survive
    serialisation to Redis JSON and deserialisation back without mutation.

    Validates: Requirements 2.5
    """
    redis_client = fakeredis.aioredis.FakeRedis()
    repo = RedisSessionRepository(redis_client=redis_client, ttl_seconds=86400)

    await repo.save(session)
    retrieved = await repo.get(session.phone_number)

    assert retrieved is not None

    # Completion fields must be identical after round-trip
    assert retrieved.is_complete == session.is_complete
    assert retrieved.completed_at == session.completed_at
    assert retrieved.stage == session.stage
    assert retrieved.session_id == session.session_id
    assert retrieved.phone_number == session.phone_number
