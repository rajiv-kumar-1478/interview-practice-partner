"""Unit tests for RedisSessionRepository.

Covers:
- Round-trip serialisation: save then get returns identical SessionState
- TTL is applied on save
- SessionStoreUnavailableError is raised when Redis is unavailable (get, save, delete)

Requirements: 2.1, 2.3, 2.6
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import fakeredis.aioredis
import pytest
import redis.exceptions

from interview_practice_partner.domain.enums import (
    EvaluationDimension,
    QuestionType,
    Role,
    Stage,
)
from interview_practice_partner.domain.exceptions import SessionStoreUnavailableError
from interview_practice_partner.domain.models import (
    DimensionScore,
    FeedbackReport,
    Question,
    SessionState,
    UserResponse,
)
from interview_practice_partner.repositories.redis_session import RedisSessionRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

TTL_SECONDS = 3600


def _minimal_session(phone: str = "+15550001111") -> SessionState:
    """Return a minimal SessionState with only required fields populated."""
    return SessionState(
        session_id="sess-0001",
        phone_number=phone,
        stage=Stage.INIT,
        role=Role.UNKNOWN,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _full_session(phone: str = "+15550002222") -> SessionState:
    """Return a fully-populated SessionState exercising every field."""
    question = Question(
        question_id="q-0001",
        text="Tell me about yourself.",
        question_type=QuestionType.BEHAVIOURAL,
        asked_at=_NOW,
        skipped=False,
    )
    response = UserResponse(
        response_id="r-0001",
        question_id="q-0001",
        text="I am a software engineer.",
        word_count=5,
        is_off_topic=False,
        received_at=_NOW,
    )
    dim_score = DimensionScore(
        dimension=EvaluationDimension.COMMUNICATION_CLARITY,
        qualitative_assessment="Clear and concise.",
        score=4,
    )
    report = FeedbackReport(
        report_id="rep-0001",
        session_id="sess-0002",
        dimension_scores=[dim_score],
        strengths=["Good communication"],
        improvements=["Add more examples"],
        actionable_recommendations=["Practice STAR method"],
        generated_at=_NOW,
    )
    return SessionState(
        session_id="sess-0002",
        phone_number=phone,
        stage=Stage.COMPLETE,
        role=Role.SOFTWARE_ENGINEER,
        questions=[question],
        responses=[response],
        off_topic_count=1,
        consecutive_out_of_scope_count=0,
        clarification_turn_count=2,
        requested_short_session=True,
        feedback_report=report,
        created_at=_NOW,
        updated_at=_NOW,
        completed_at=_NOW,
        is_complete=True,
        context_summary="Candidate performed well.",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_redis() -> fakeredis.aioredis.FakeRedis:
    return fakeredis.aioredis.FakeRedis()


@pytest.fixture
def repo(fake_redis: fakeredis.aioredis.FakeRedis) -> RedisSessionRepository:
    return RedisSessionRepository(redis_client=fake_redis, ttl_seconds=TTL_SECONDS)


# ---------------------------------------------------------------------------
# Round-trip serialisation — Requirements 2.1, 2.3
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """save then get must return an identical SessionState."""

    async def test_minimal_session_round_trip(
        self, repo: RedisSessionRepository
    ) -> None:
        """Saving a minimal session and retrieving it returns the same object."""
        session = _minimal_session()
        await repo.save(session)
        retrieved = await repo.get(session.phone_number)
        assert retrieved is not None
        assert retrieved == session

    async def test_full_session_round_trip(
        self, repo: RedisSessionRepository
    ) -> None:
        """Saving a fully-populated session and retrieving it returns the same object."""
        session = _full_session()
        await repo.save(session)
        retrieved = await repo.get(session.phone_number)
        assert retrieved is not None
        assert retrieved == session

    async def test_round_trip_preserves_stage(
        self, repo: RedisSessionRepository
    ) -> None:
        session = _minimal_session()
        session.stage = Stage.INTERVIEW
        await repo.save(session)
        retrieved = await repo.get(session.phone_number)
        assert retrieved is not None
        assert retrieved.stage == Stage.INTERVIEW

    async def test_round_trip_preserves_role(
        self, repo: RedisSessionRepository
    ) -> None:
        session = _minimal_session()
        session.role = Role.SALES_REPRESENTATIVE
        await repo.save(session)
        retrieved = await repo.get(session.phone_number)
        assert retrieved is not None
        assert retrieved.role == Role.SALES_REPRESENTATIVE

    async def test_round_trip_preserves_questions_list(
        self, repo: RedisSessionRepository
    ) -> None:
        session = _full_session()
        await repo.save(session)
        retrieved = await repo.get(session.phone_number)
        assert retrieved is not None
        assert len(retrieved.questions) == len(session.questions)
        assert retrieved.questions[0] == session.questions[0]

    async def test_round_trip_preserves_feedback_report(
        self, repo: RedisSessionRepository
    ) -> None:
        session = _full_session()
        await repo.save(session)
        retrieved = await repo.get(session.phone_number)
        assert retrieved is not None
        assert retrieved.feedback_report == session.feedback_report

    async def test_get_returns_none_for_unknown_phone(
        self, repo: RedisSessionRepository
    ) -> None:
        result = await repo.get("+19999999999")
        assert result is None

    async def test_save_overwrites_existing_session(
        self, repo: RedisSessionRepository
    ) -> None:
        """A second save with updated fields replaces the first."""
        session = _minimal_session()
        await repo.save(session)

        session.stage = Stage.ROLE_SELECTION
        await repo.save(session)

        retrieved = await repo.get(session.phone_number)
        assert retrieved is not None
        assert retrieved.stage == Stage.ROLE_SELECTION

    async def test_delete_removes_session(
        self, repo: RedisSessionRepository
    ) -> None:
        session = _minimal_session()
        await repo.save(session)
        await repo.delete(session.phone_number)
        result = await repo.get(session.phone_number)
        assert result is None


# ---------------------------------------------------------------------------
# TTL — Requirement 2.6
# ---------------------------------------------------------------------------


class TestTTL:
    """TTL must be applied on every save."""

    async def test_ttl_is_set_after_save(
        self,
        repo: RedisSessionRepository,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        session = _minimal_session()
        await repo.save(session)
        key = f"session:{session.phone_number}"
        ttl = await fake_redis.ttl(key)
        assert ttl > 0
        assert ttl <= TTL_SECONDS

    async def test_ttl_is_refreshed_on_second_save(
        self,
        repo: RedisSessionRepository,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        """Each save must reset the TTL to the configured value."""
        session = _minimal_session()
        await repo.save(session)
        key = f"session:{session.phone_number}"

        # Manually reduce the TTL to simulate time passing
        await fake_redis.expire(key, 10)
        ttl_before = await fake_redis.ttl(key)
        assert ttl_before <= 10

        # A second save should restore the full TTL
        await repo.save(session)
        ttl_after = await fake_redis.ttl(key)
        assert ttl_after > 10
        assert ttl_after <= TTL_SECONDS

    async def test_ttl_uses_configured_value(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        """Repository respects the ttl_seconds passed at construction time."""
        custom_ttl = 120
        repo = RedisSessionRepository(redis_client=fake_redis, ttl_seconds=custom_ttl)
        session = _minimal_session(phone="+15550003333")
        await repo.save(session)
        key = f"session:{session.phone_number}"
        ttl = await fake_redis.ttl(key)
        assert ttl > 0
        assert ttl <= custom_ttl


# ---------------------------------------------------------------------------
# Error handling — Requirement 2.6
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """SessionStoreUnavailableError must be raised when Redis is unreachable."""

    # get -------------------------------------------------------------------

    async def test_get_raises_on_connection_error(
        self, repo: RedisSessionRepository
    ) -> None:
        with patch.object(
            repo._redis,
            "get",
            new=AsyncMock(side_effect=redis.exceptions.ConnectionError("refused")),
        ):
            with pytest.raises(SessionStoreUnavailableError):
                await repo.get("+15550001111")

    async def test_get_raises_on_timeout_error(
        self, repo: RedisSessionRepository
    ) -> None:
        with patch.object(
            repo._redis,
            "get",
            new=AsyncMock(side_effect=redis.exceptions.TimeoutError("timed out")),
        ):
            with pytest.raises(SessionStoreUnavailableError):
                await repo.get("+15550001111")

    # save ------------------------------------------------------------------

    async def test_save_raises_on_connection_error(
        self, repo: RedisSessionRepository
    ) -> None:
        with patch.object(
            repo._redis,
            "setex",
            new=AsyncMock(side_effect=redis.exceptions.ConnectionError("refused")),
        ):
            with pytest.raises(SessionStoreUnavailableError):
                await repo.save(_minimal_session())

    async def test_save_raises_on_timeout_error(
        self, repo: RedisSessionRepository
    ) -> None:
        with patch.object(
            repo._redis,
            "setex",
            new=AsyncMock(side_effect=redis.exceptions.TimeoutError("timed out")),
        ):
            with pytest.raises(SessionStoreUnavailableError):
                await repo.save(_minimal_session())

    # delete ----------------------------------------------------------------

    async def test_delete_raises_on_connection_error(
        self, repo: RedisSessionRepository
    ) -> None:
        with patch.object(
            repo._redis,
            "delete",
            new=AsyncMock(side_effect=redis.exceptions.ConnectionError("refused")),
        ):
            with pytest.raises(SessionStoreUnavailableError):
                await repo.delete("+15550001111")

    async def test_delete_raises_on_timeout_error(
        self, repo: RedisSessionRepository
    ) -> None:
        with patch.object(
            repo._redis,
            "delete",
            new=AsyncMock(side_effect=redis.exceptions.TimeoutError("timed out")),
        ):
            with pytest.raises(SessionStoreUnavailableError):
                await repo.delete("+15550001111")

    async def test_error_message_contains_phone_number(
        self, repo: RedisSessionRepository
    ) -> None:
        """The exception message should include the phone number for diagnostics."""
        import re

        phone = "+15550001111"
        with patch.object(
            repo._redis,
            "get",
            new=AsyncMock(side_effect=redis.exceptions.ConnectionError("refused")),
        ):
            with pytest.raises(SessionStoreUnavailableError, match=re.escape(phone)):
                await repo.get(phone)
