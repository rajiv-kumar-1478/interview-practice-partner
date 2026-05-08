"""Integration tests for the Interview Practice Partner webhook flow.

Covers:
- Task 10.1: Full webhook flow (new user → role selection → interview → feedback)
- Task 10.2: Idempotency — duplicate MessageSid is suppressed
- Task 10.3: Redis session round-trip via RedisSessionRepository

Requirements: 1.1, 1.4, 1.5, 1.7, 2.1, 2.3
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import urlencode

import fakeredis.aioredis
import httpx
import pytest
from httpx import ASGITransport

from interview_practice_partner.config import Settings
from interview_practice_partner.domain.enums import (
    EvaluationDimension,
    QuestionType,
    Role,
    Stage,
)
from interview_practice_partner.domain.models import (
    DimensionScore,
    FeedbackReport,
    Question,
    SessionState,
    UserResponse,
)
from interview_practice_partner.main import create_app
from interview_practice_partner.repositories.redis_session import RedisSessionRepository


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

_TEST_SETTINGS = Settings(
    twilio_account_sid="ACtest123",
    twilio_auth_token="test_token",
    twilio_whatsapp_number="whatsapp:+14155238886",
    llm_api_key="test_key",
    redis_url="redis://localhost:6379/0",
    groq_api_key="test_groq_key",
    elevenlabs_api_key="test_elevenlabs_key",
    elevenlabs_voice_id="test_voice_id",
)

_FROM_NUMBER = "whatsapp:+447700900001"
_TO_NUMBER = "whatsapp:+14155238886"


def _form_body(**kwargs: str) -> bytes:
    """Encode keyword arguments as an ``application/x-www-form-urlencoded`` body."""
    return urlencode(kwargs).encode()


def _make_message(
    body: str,
    message_sid: str = "SM_test_001",
    from_number: str = _FROM_NUMBER,
    to_number: str = _TO_NUMBER,
) -> bytes:
    """Build a Twilio-style form-encoded webhook payload."""
    return _form_body(
        MessageSid=message_sid,
        From=from_number,
        To=to_number,
        Body=body,
        NumMedia="0",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def app_with_fakeredis(mocker):
    """Create a FastAPI app wired to fakeredis with Twilio and LLM mocked out.

    ASGITransport does not trigger the FastAPI lifespan, so we set
    app.state.redis and app.state.twilio_client directly after create_app().
    This is the simplest and most reliable approach for integration tests.
    """
    app = create_app(_TEST_SETTINGS)

    # Set up fakeredis directly on app state — bypasses the lifespan Redis init
    fake_redis = fakeredis.aioredis.FakeRedis()
    app.state.redis = fake_redis

    # Mock Twilio client and set it directly on app state
    mock_twilio = mocker.MagicMock()
    mock_message = mocker.MagicMock()
    mock_message.status = "sent"
    mock_message.sid = "SMtest123"
    mock_twilio.messages.create.return_value = mock_message
    app.state.twilio_client = mock_twilio

    # Bypass Twilio signature validation so test requests are accepted
    mocker.patch(
        "twilio.request_validator.RequestValidator.validate",
        return_value=True,
    )

    yield app, fake_redis, mock_twilio


# ---------------------------------------------------------------------------
# Task 10.1 — Full webhook flow integration test
# Requirements: 1.1, 1.4, 1.5, 2.1, 2.3
# ---------------------------------------------------------------------------


class TestFullWebhookFlow:
    """Simulate a complete session: new user → role selection → interview → feedback."""

    async def test_new_user_receives_role_selection_prompt(
        self, app_with_fakeredis, mocker
    ) -> None:
        """First message from an unknown number triggers role-selection prompt.

        Requirements: 1.1, 1.4, 2.1
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        # Mock LLM to return a role-selection clarification prompt
        mocker.patch(
            "interview_practice_partner.llm.openai_client.OpenAIClient.complete",
            return_value=json.dumps(
                {
                    "role": "unknown",
                    "confidence": "low",
                    "message": "Which role are you preparing for?",
                }
            ),
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhook/whatsapp",
                content=_make_message("Hello!", message_sid="SM_flow_001"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        assert response.status_code == 200

        # Twilio send must have been called with the correct destination
        mock_twilio.messages.create.assert_called_once()
        call_kwargs = mock_twilio.messages.create.call_args.kwargs
        assert call_kwargs["to"] == _FROM_NUMBER

        # Session must be persisted in Redis
        session = await RedisSessionRepository(
            redis_client=fake_redis, ttl_seconds=86400
        ).get(_FROM_NUMBER)
        assert session is not None
        assert session.phone_number == _FROM_NUMBER
        assert session.stage == Stage.ROLE_SELECTION

    async def test_role_in_opening_message_starts_interview_directly(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Opening message containing SWE role skips role-selection and goes to round type selection.

        Requirements: 1.1, 1.4, 2.1, 2.3
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        # LLM returns a question for the interview stage
        mocker.patch(
            "interview_practice_partner.llm.openai_client.OpenAIClient.complete",
            return_value="Tell me about a challenging project you worked on.",
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhook/whatsapp",
                content=_make_message(
                    "I want to practise for a software engineer interview",
                    message_sid="SM_flow_002",
                ),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        assert response.status_code == 200

        # Session must be in ROUND_TYPE_SELECTION stage with the correct role
        # (SOFTWARE_ENGINEER goes to round type selection before starting interview)
        session = await RedisSessionRepository(
            redis_client=fake_redis, ttl_seconds=86400
        ).get(_FROM_NUMBER)
        assert session is not None
        assert session.stage == Stage.ROUND_TYPE_SELECTION
        assert session.role == Role.SOFTWARE_ENGINEER

        # Twilio send must target the sender
        mock_twilio.messages.create.assert_called_once()
        assert mock_twilio.messages.create.call_args.kwargs["to"] == _FROM_NUMBER

    async def test_session_state_persisted_after_each_turn(
        self, app_with_fakeredis, mocker
    ) -> None:
        """SessionState is updated and persisted after every webhook call.

        Requirements: 2.1, 2.3
        """
        app, fake_redis, mock_twilio = app_with_fakeredis
        repo = RedisSessionRepository(redis_client=fake_redis, ttl_seconds=86400)

        # Turn 1: opening message with role → INTERVIEW
        mocker.patch(
            "interview_practice_partner.llm.openai_client.OpenAIClient.complete",
            return_value="Describe a time you handled a difficult customer.",
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/webhook/whatsapp",
                content=_make_message(
                    "I want to practise for a sales representative interview",
                    message_sid="SM_persist_001",
                ),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        session_after_turn1 = await repo.get(_FROM_NUMBER)
        assert session_after_turn1 is not None
        assert session_after_turn1.stage == Stage.INTERVIEW
        assert session_after_turn1.role == Role.SALES_REPRESENTATIVE

        # Turn 2: answer the question — LLM evaluates and returns next question
        # The handle_response flow makes 3 LLM calls:
        #   1. classify_intent → {"intent": "answer"}
        #   2. evaluate_response → evaluation JSON
        #   3. generate_question → next question text
        intent_response = json.dumps({"intent": "answer"})
        eval_response = json.dumps(
            {
                "is_off_topic": False,
                "is_short": False,
                "follow_up_warranted": False,
                "follow_up_text": "",
                "difficulty_signal": "maintain",
            }
        )
        next_question = "How do you handle rejection from a prospect?"

        mocker.patch(
            "interview_practice_partner.llm.openai_client.OpenAIClient.complete",
            side_effect=[intent_response, eval_response, next_question],
        )

        # Use a response with ≥15 words to pass the short-response threshold
        long_answer = (
            "I stayed calm and listened carefully to their concerns before offering "
            "a tailored solution that addressed their specific needs and resolved the issue."
        )
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/webhook/whatsapp",
                content=_make_message(
                    long_answer,
                    message_sid="SM_persist_002",
                ),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        session_after_turn2 = await repo.get(_FROM_NUMBER)
        assert session_after_turn2 is not None
        # A response should have been recorded
        assert len(session_after_turn2.responses) >= 1

    async def test_twilio_send_uses_from_number_as_destination(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Outbound Twilio message must be addressed to the inbound from_number.

        Requirements: 1.5
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        custom_from = "whatsapp:+12025550199"

        mocker.patch(
            "interview_practice_partner.llm.openai_client.OpenAIClient.complete",
            return_value=json.dumps(
                {
                    "role": "unknown",
                    "confidence": "low",
                    "message": "Which role are you preparing for?",
                }
            ),
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/webhook/whatsapp",
                content=_make_message(
                    "Hi there",
                    message_sid="SM_dest_001",
                    from_number=custom_from,
                ),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        mock_twilio.messages.create.assert_called_once()
        assert mock_twilio.messages.create.call_args.kwargs["to"] == custom_from

    async def test_webhook_always_returns_200(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Webhook endpoint must return HTTP 200 regardless of message content.

        Requirements: 1.4
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        mocker.patch(
            "interview_practice_partner.llm.openai_client.OpenAIClient.complete",
            return_value=json.dumps(
                {
                    "role": "unknown",
                    "confidence": "low",
                    "message": "Which role are you preparing for?",
                }
            ),
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhook/whatsapp",
                content=_make_message("Hello", message_sid="SM_200_001"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Task 10.2 — Idempotency integration test
# Requirements: 1.7
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Duplicate MessageSid must be suppressed — exactly one send and one state mutation."""

    async def test_duplicate_message_sid_suppressed(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Submitting the same MessageSid twice results in exactly one Twilio send.

        Requirements: 1.7
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        mocker.patch(
            "interview_practice_partner.llm.openai_client.OpenAIClient.complete",
            return_value=json.dumps(
                {
                    "role": "unknown",
                    "confidence": "low",
                    "message": "Which role are you preparing for?",
                }
            ),
        )

        payload = _make_message("Hello again", message_sid="SM_idem_001")
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1 = await client.post("/webhook/whatsapp", content=payload, headers=headers)
            r2 = await client.post("/webhook/whatsapp", content=payload, headers=headers)

        assert r1.status_code == 200
        assert r2.status_code == 200

        # Twilio send must have been called exactly once
        assert mock_twilio.messages.create.call_count == 1

    async def test_duplicate_message_sid_does_not_mutate_session_twice(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Session state must be mutated exactly once for a duplicate MessageSid.

        Requirements: 1.7
        """
        app, fake_redis, mock_twilio = app_with_fakeredis
        repo = RedisSessionRepository(redis_client=fake_redis, ttl_seconds=86400)

        mocker.patch(
            "interview_practice_partner.llm.openai_client.OpenAIClient.complete",
            return_value=json.dumps(
                {
                    "role": "unknown",
                    "confidence": "low",
                    "message": "Which role are you preparing for?",
                }
            ),
        )

        payload = _make_message("Hello", message_sid="SM_idem_002")
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/webhook/whatsapp", content=payload, headers=headers)

        # Capture session state after first request
        session_after_first = await repo.get(_FROM_NUMBER)
        assert session_after_first is not None
        first_updated_at = session_after_first.updated_at

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/webhook/whatsapp", content=payload, headers=headers)

        # Session state must not have changed after the duplicate
        session_after_second = await repo.get(_FROM_NUMBER)
        assert session_after_second is not None
        assert session_after_second.updated_at == first_updated_at

    async def test_different_message_sids_are_processed_independently(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Two distinct MessageSids from the same number are both processed.

        Requirements: 1.7
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        mocker.patch(
            "interview_practice_partner.llm.openai_client.OpenAIClient.complete",
            return_value=json.dumps(
                {
                    "role": "unknown",
                    "confidence": "low",
                    "message": "Which role are you preparing for?",
                }
            ),
        )

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/webhook/whatsapp",
                content=_make_message("First message", message_sid="SM_unique_001"),
                headers=headers,
            )
            await client.post(
                "/webhook/whatsapp",
                content=_make_message("Second message", message_sid="SM_unique_002"),
                headers=headers,
            )

        # Both messages should have triggered a Twilio send
        assert mock_twilio.messages.create.call_count == 2


# ---------------------------------------------------------------------------
# Task 10.3 — Redis session round-trip integration test
# Requirements: 2.1, 2.3
# ---------------------------------------------------------------------------


class TestRedisSessionRoundTrip:
    """Save a SessionState via RedisSessionRepository, retrieve it, assert equality."""

    @pytest.fixture
    def fake_redis(self) -> fakeredis.aioredis.FakeRedis:
        return fakeredis.aioredis.FakeRedis()

    @pytest.fixture
    def repo(self, fake_redis: fakeredis.aioredis.FakeRedis) -> RedisSessionRepository:
        return RedisSessionRepository(redis_client=fake_redis, ttl_seconds=86400)

    def _minimal_session(self, phone: str = "+15550001111") -> SessionState:
        return SessionState(
            session_id="sess-integration-001",
            phone_number=phone,
            stage=Stage.INIT,
            role=Role.UNKNOWN,
            created_at=_NOW,
            updated_at=_NOW,
        )

    def _full_session(self, phone: str = "+15550002222") -> SessionState:
        question = Question(
            question_id="q-int-001",
            text="Tell me about a time you led a team.",
            question_type=QuestionType.BEHAVIOURAL,
            asked_at=_NOW,
            skipped=False,
        )
        response = UserResponse(
            response_id="r-int-001",
            question_id="q-int-001",
            text="I led a cross-functional team to deliver a product on time.",
            word_count=12,
            is_off_topic=False,
            received_at=_NOW,
        )
        dim_score = DimensionScore(
            dimension=EvaluationDimension.COMMUNICATION_CLARITY,
            qualitative_assessment="Clear and structured response.",
            score=4,
        )
        report = FeedbackReport(
            report_id="rep-int-001",
            session_id="sess-integration-002",
            dimension_scores=[dim_score],
            strengths=["Strong leadership narrative"],
            improvements=["Include more quantitative outcomes"],
            actionable_recommendations=["Use the STAR method consistently"],
            generated_at=_NOW,
        )
        return SessionState(
            session_id="sess-integration-002",
            phone_number=phone,
            stage=Stage.COMPLETE,
            role=Role.SOFTWARE_ENGINEER,
            questions=[question],
            responses=[response],
            off_topic_count=0,
            consecutive_out_of_scope_count=0,
            clarification_turn_count=1,
            requested_short_session=False,
            feedback_report=report,
            created_at=_NOW,
            updated_at=_NOW,
            completed_at=_NOW,
            is_complete=True,
            context_summary="Candidate demonstrated strong leadership skills.",
        )

    async def test_minimal_session_round_trip(
        self, repo: RedisSessionRepository
    ) -> None:
        """Save a minimal SessionState and retrieve it — all fields must be identical.

        Requirements: 2.1, 2.3
        """
        session = self._minimal_session()
        await repo.save(session)
        retrieved = await repo.get(session.phone_number)

        assert retrieved is not None
        assert retrieved == session

    async def test_full_session_round_trip(
        self, repo: RedisSessionRepository
    ) -> None:
        """Save a fully-populated SessionState and retrieve it — all fields identical.

        Requirements: 2.1, 2.3
        """
        session = self._full_session()
        await repo.save(session)
        retrieved = await repo.get(session.phone_number)

        assert retrieved is not None
        assert retrieved == session

    async def test_round_trip_preserves_stage(
        self, repo: RedisSessionRepository
    ) -> None:
        """Stage enum value survives serialisation round-trip."""
        for stage in Stage:
            phone = f"+1555000{stage.value[:4].lower()}"
            session = self._minimal_session(phone=phone)
            session.stage = stage
            await repo.save(session)
            retrieved = await repo.get(phone)
            assert retrieved is not None
            assert retrieved.stage == stage

    async def test_round_trip_preserves_role(
        self, repo: RedisSessionRepository
    ) -> None:
        """Role enum value survives serialisation round-trip."""
        for role in Role:
            phone = f"+1555001{role.value[:4].lower()}"
            session = self._minimal_session(phone=phone)
            session.role = role
            await repo.save(session)
            retrieved = await repo.get(phone)
            assert retrieved is not None
            assert retrieved.role == role

    async def test_round_trip_preserves_questions_and_responses(
        self, repo: RedisSessionRepository
    ) -> None:
        """Questions and responses lists survive serialisation round-trip."""
        session = self._full_session()
        await repo.save(session)
        retrieved = await repo.get(session.phone_number)

        assert retrieved is not None
        assert len(retrieved.questions) == len(session.questions)
        assert retrieved.questions[0] == session.questions[0]
        assert len(retrieved.responses) == len(session.responses)
        assert retrieved.responses[0] == session.responses[0]

    async def test_round_trip_preserves_feedback_report(
        self, repo: RedisSessionRepository
    ) -> None:
        """FeedbackReport (including nested DimensionScore) survives round-trip."""
        session = self._full_session()
        await repo.save(session)
        retrieved = await repo.get(session.phone_number)

        assert retrieved is not None
        assert retrieved.feedback_report is not None
        assert retrieved.feedback_report == session.feedback_report
        assert retrieved.feedback_report.dimension_scores[0].score == 4

    async def test_round_trip_preserves_boolean_flags(
        self, repo: RedisSessionRepository
    ) -> None:
        """Boolean flags (is_complete, requested_short_session) survive round-trip."""
        session = self._full_session()
        assert session.is_complete is True
        assert session.requested_short_session is False

        await repo.save(session)
        retrieved = await repo.get(session.phone_number)

        assert retrieved is not None
        assert retrieved.is_complete is True
        assert retrieved.requested_short_session is False

    async def test_round_trip_preserves_counters(
        self, repo: RedisSessionRepository
    ) -> None:
        """Integer counters survive serialisation round-trip."""
        session = self._minimal_session()
        session.off_topic_count = 3
        session.consecutive_out_of_scope_count = 2
        session.clarification_turn_count = 1

        await repo.save(session)
        retrieved = await repo.get(session.phone_number)

        assert retrieved is not None
        assert retrieved.off_topic_count == 3
        assert retrieved.consecutive_out_of_scope_count == 2
        assert retrieved.clarification_turn_count == 1

    async def test_round_trip_preserves_optional_fields(
        self, repo: RedisSessionRepository
    ) -> None:
        """Optional fields (context_summary, completed_at) survive round-trip."""
        session = self._full_session()
        assert session.context_summary is not None
        assert session.completed_at is not None

        await repo.save(session)
        retrieved = await repo.get(session.phone_number)

        assert retrieved is not None
        assert retrieved.context_summary == session.context_summary
        assert retrieved.completed_at == session.completed_at

    async def test_get_returns_none_for_unknown_phone(
        self, repo: RedisSessionRepository
    ) -> None:
        """Retrieving a session for an unknown phone number returns None."""
        result = await repo.get("+19999999999")
        assert result is None

    async def test_save_overwrites_existing_session(
        self, repo: RedisSessionRepository
    ) -> None:
        """A second save with updated stage replaces the first persisted state."""
        session = self._minimal_session()
        await repo.save(session)

        session.stage = Stage.INTERVIEW
        await repo.save(session)

        retrieved = await repo.get(session.phone_number)
        assert retrieved is not None
        assert retrieved.stage == Stage.INTERVIEW
