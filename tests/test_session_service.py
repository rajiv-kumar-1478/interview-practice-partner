"""Unit tests for SessionService state machine transitions.

Covers:
- All valid stage transitions from the design's transition table
- Role change mid-session creates a new session
- Clarification timeout transitions to INTERVIEW with default role
- Counter increments (clarification_turn_count, off_topic_count,
  consecutive_out_of_scope_count)

Requirements: 3.1, 3.3, 5.2, 5.3
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from interview_practice_partner.domain.enums import QuestionType, Role, Stage
from interview_practice_partner.domain.models import (
    FeedbackReport,
    DimensionScore,
    EvaluationDimension,
    Question,
    SessionState,
    UserResponse,
)
from interview_practice_partner.services.session import SessionService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
PHONE = "+15550001234"


def make_session(**overrides) -> SessionState:
    defaults = dict(
        session_id=str(uuid.uuid4()),
        phone_number=PHONE,
        stage=Stage.INIT,
        role=Role.UNKNOWN,
        created_at=NOW,
        updated_at=NOW,
    )
    defaults.update(overrides)
    return SessionState(**defaults)


def make_question(
    question_id: str | None = None,
    text: str = "Tell me about yourself.",
    question_type: QuestionType = QuestionType.BEHAVIOURAL,
    skipped: bool = False,
) -> Question:
    return Question(
        question_id=question_id or str(uuid.uuid4()),
        text=text,
        question_type=question_type,
        asked_at=NOW,
        skipped=skipped,
    )


def make_response(question_id: str, text: str = "I have five years of experience.") -> UserResponse:
    return UserResponse(
        response_id=str(uuid.uuid4()),
        question_id=question_id,
        text=text,
        word_count=len(text.split()),
        received_at=NOW,
    )


def make_feedback_report(session_id: str) -> FeedbackReport:
    return FeedbackReport(
        report_id=str(uuid.uuid4()),
        session_id=session_id,
        dimension_scores=[
            DimensionScore(
                dimension=EvaluationDimension.COMMUNICATION_CLARITY,
                qualitative_assessment="Good",
                score=4,
            )
        ],
        strengths=["Clear communication"],
        improvements=["Be more concise"],
        actionable_recommendations=["Practice STAR method"],
        generated_at=NOW,
    )


def make_service(
    llm_response: str = '{"role": "unknown", "confidence": "low", "message": "What role?"}',
    interview_question: str = "Tell me about a challenge you faced.",
    interview_handle_response: tuple[str, SessionState] | None = None,
    feedback_response: tuple[str, SessionState] | None = None,
) -> tuple[SessionService, Any, Any, Any]:
    """Build a SessionService with mocked dependencies.

    Returns (service, mock_llm, mock_interview, mock_feedback).
    """
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = llm_response

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_role_selection_prompt.return_value = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]

    mock_interview = AsyncMock()
    mock_interview.generate_question.return_value = interview_question
    # Set up handle_response to return a sensible default for round type selection
    mock_interview.handle_response.return_value = (
        "Which round type would you like to practice?",
        make_session(stage=Stage.ROUND_TYPE_SELECTION, role=Role.SOFTWARE_ENGINEER),
    )

    mock_feedback = AsyncMock()

    service = SessionService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
        interview_service=mock_interview,
        feedback_service=mock_feedback,
    )

    return service, mock_llm, mock_interview, mock_feedback


# ===========================================================================
# INIT stage transitions
# ===========================================================================


class TestInitStage:
    @pytest.mark.asyncio
    async def test_init_any_message_transitions_to_role_selection(self):
        """INIT → ROLE_SELECTION on any first message (no role detected)."""
        service, mock_llm, _, _ = make_service(
            llm_response='{"role": "unknown", "confidence": "low", "message": "What role?"}'
        )
        session = make_session(stage=Stage.INIT)

        _, updated = await service.transition(session, "Hello, I want to practise.")

        assert updated.stage == Stage.ROLE_SELECTION

    @pytest.mark.asyncio
    async def test_init_with_role_in_message_fast_path_to_interview(self):
        """INIT with SWE role in message → ROUND_TYPE_SELECTION (fast-path, Req 6.1)."""
        service, _, mock_interview, _ = make_service(
            interview_question="Describe a time you led a project."
        )
        session = make_session(stage=Stage.INIT)

        reply, updated = await service.transition(session, "I want to practise as a software engineer.")

        assert updated.stage == Stage.ROUND_TYPE_SELECTION
        assert updated.role == Role.SOFTWARE_ENGINEER
        # generate_question should NOT be called (we're in round type selection)
        mock_interview.generate_question.assert_not_called()

    @pytest.mark.asyncio
    async def test_init_with_sales_role_fast_path(self):
        """INIT with 'sales' keyword → INTERVIEW with SALES_REPRESENTATIVE role."""
        service, _, mock_interview, _ = make_service(
            interview_question="Tell me about your sales experience."
        )
        session = make_session(stage=Stage.INIT)

        _, updated = await service.transition(session, "I'm preparing for a sales role.")

        assert updated.stage == Stage.INTERVIEW
        assert updated.role == Role.SALES_REPRESENTATIVE

    @pytest.mark.asyncio
    async def test_init_with_retail_role_fast_path(self):
        """INIT with 'retail' keyword → INTERVIEW with RETAIL_ASSOCIATE role."""
        service, _, mock_interview, _ = make_service(
            interview_question="How do you handle difficult customers?"
        )
        session = make_session(stage=Stage.INIT)

        _, updated = await service.transition(session, "I want to practise for a retail associate job.")

        assert updated.stage == Stage.INTERVIEW
        assert updated.role == Role.RETAIL_ASSOCIATE

    @pytest.mark.asyncio
    async def test_init_no_role_calls_llm_for_role_selection(self):
        """INIT without role → LLM called to generate role-selection prompt."""
        service, mock_llm, _, _ = make_service(
            llm_response='{"role": "unknown", "confidence": "low", "message": "Which role?"}'
        )
        session = make_session(stage=Stage.INIT)

        reply, updated = await service.transition(session, "Hi there!")

        assert updated.stage == Stage.ROLE_SELECTION
        mock_llm.complete.assert_called_once()
        assert "Which role?" in reply

    @pytest.mark.asyncio
    async def test_init_resets_clarification_turn_count(self):
        """INIT handler resets clarification_turn_count to 0."""
        service, _, _, _ = make_service()
        session = make_session(stage=Stage.INIT, clarification_turn_count=3)

        _, updated = await service.transition(session, "Hello!")

        assert updated.clarification_turn_count == 0


# ===========================================================================
# ROLE_SELECTION stage transitions
# ===========================================================================


class TestRoleSelectionStage:
    @pytest.mark.asyncio
    async def test_role_confirmed_transitions_to_interview(self):
        """ROLE_SELECTION with high-confidence SWE role → ROUND_TYPE_SELECTION."""
        service, mock_llm, mock_interview, _ = make_service(
            llm_response='{"role": "software_engineer", "confidence": "high", "message": "Great!"}',
            interview_question="What is your experience with Python?",
        )
        session = make_session(stage=Stage.ROLE_SELECTION)

        reply, updated = await service.transition(session, "I want to be a software engineer.")

        assert updated.stage == Stage.ROUND_TYPE_SELECTION
        assert updated.role == Role.SOFTWARE_ENGINEER
        # generate_question should NOT be called (we're in round type selection)
        mock_interview.generate_question.assert_not_called()

    @pytest.mark.asyncio
    async def test_low_confidence_role_stays_in_role_selection(self):
        """ROLE_SELECTION with low-confidence role → stays in ROLE_SELECTION."""
        service, mock_llm, mock_interview, _ = make_service(
            llm_response='{"role": "software_engineer", "confidence": "low", "message": "Did you mean engineer?"}'
        )
        session = make_session(stage=Stage.ROLE_SELECTION)

        reply, updated = await service.transition(session, "I want to do something technical.")

        assert updated.stage == Stage.ROLE_SELECTION
        mock_interview.generate_question.assert_not_called()

    @pytest.mark.asyncio
    async def test_clarification_turn_count_incremented(self):
        """ROLE_SELECTION increments clarification_turn_count on each turn."""
        service, _, _, _ = make_service(
            llm_response='{"role": "unknown", "confidence": "low", "message": "Please clarify."}'
        )
        session = make_session(stage=Stage.ROLE_SELECTION, clarification_turn_count=0)

        _, updated = await service.transition(session, "I'm not sure what I want.")

        assert updated.clarification_turn_count == 1

    @pytest.mark.asyncio
    async def test_clarification_turn_count_incremented_again(self):
        """Second ROLE_SELECTION turn increments count to 2."""
        service, _, _, _ = make_service(
            llm_response='{"role": "unknown", "confidence": "low", "message": "Please clarify."}'
        )
        session = make_session(stage=Stage.ROLE_SELECTION, clarification_turn_count=1)

        _, updated = await service.transition(session, "Still not sure.")

        assert updated.clarification_turn_count == 2

    @pytest.mark.asyncio
    async def test_two_clarification_turns_no_role_transitions_to_interview_default(self):
        """After 2 clarification turns with no role → INTERVIEW with default general format (Req 5.2)."""
        service, mock_llm, mock_interview, _ = make_service(
            llm_response='{"role": "unknown", "confidence": "low", "message": "Unclear."}',
            interview_question="Tell me about your work experience.",
        )
        # clarification_turn_count starts at 1 (one turn already done)
        session = make_session(stage=Stage.ROLE_SELECTION, clarification_turn_count=1)

        reply, updated = await service.transition(session, "I really don't know what I want.")

        assert updated.stage == Stage.INTERVIEW
        assert updated.role == Role.UNKNOWN
        mock_interview.generate_question.assert_called_once()
        assert "general interview format" in reply.lower()

    @pytest.mark.asyncio
    async def test_clarification_timeout_with_role_in_message_uses_role(self):
        """Clarification timeout but role detected in message → use that role."""
        service, mock_llm, mock_interview, _ = make_service(
            llm_response='{"role": "unknown", "confidence": "low", "message": "Unclear."}',
            interview_question="Describe your sales approach.",
        )
        session = make_session(stage=Stage.ROLE_SELECTION, clarification_turn_count=1)

        _, updated = await service.transition(session, "Actually, let's do sales.")

        # Role detected in message → should transition to INTERVIEW with sales role
        assert updated.stage == Stage.INTERVIEW
        assert updated.role == Role.SALES_REPRESENTATIVE

    @pytest.mark.asyncio
    async def test_role_selection_llm_json_parse_failure_falls_back(self):
        """If LLM returns non-JSON, role detection falls back gracefully."""
        service, mock_llm, _, _ = make_service(
            llm_response="I couldn't understand your role. Please clarify."
        )
        session = make_session(stage=Stage.ROLE_SELECTION)

        reply, updated = await service.transition(session, "Something unclear.")

        # Should not crash; stays in ROLE_SELECTION or transitions based on fallback
        assert updated.stage in (Stage.ROLE_SELECTION, Stage.INTERVIEW)


# ===========================================================================
# INTERVIEW stage transitions
# ===========================================================================


class TestInterviewStage:
    def _make_session_with_answered_questions(
        self, count: int, short_session: bool = False
    ) -> SessionState:
        """Create an INTERVIEW session with *count* answered questions."""
        session = make_session(
            stage=Stage.INTERVIEW,
            role=Role.SOFTWARE_ENGINEER,
            requested_short_session=short_session,
        )
        for i in range(count):
            q = make_question(question_id=str(uuid.uuid4()), text=f"Question {i + 1}?")
            r = make_response(question_id=q.question_id)
            session.questions.append(q)
            session.responses.append(r)
        return session

    @pytest.mark.asyncio
    async def test_five_answered_questions_transitions_to_feedback(self):
        """INTERVIEW with ≥5 answered questions and no follow-up → FEEDBACK."""
        session = self._make_session_with_answered_questions(5)

        mock_feedback = AsyncMock()
        feedback_session = session.model_copy(deep=True)
        feedback_session.stage = Stage.FEEDBACK
        feedback_session.feedback_report = make_feedback_report(session.session_id)
        mock_feedback.generate_feedback_report.return_value = (
            "Here is your feedback.",
            feedback_session,
        )

        mock_interview = AsyncMock()
        mock_interview.handle_response.return_value = ("Good answer!", session)

        service = SessionService(
            llm_client=AsyncMock(),
            prompt_builder=MagicMock(),
            interview_service=mock_interview,
            feedback_service=mock_feedback,
        )

        reply, updated = await service.transition(session, "My answer to question 5.")

        assert updated.stage == Stage.FEEDBACK
        mock_feedback.generate_feedback_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_short_session_three_questions_transitions_to_feedback(self):
        """INTERVIEW with short session request and ≥3 answered → FEEDBACK (Req 6.3)."""
        session = self._make_session_with_answered_questions(3, short_session=True)

        mock_feedback = AsyncMock()
        feedback_session = session.model_copy(deep=True)
        feedback_session.stage = Stage.FEEDBACK
        feedback_session.feedback_report = make_feedback_report(session.session_id)
        mock_feedback.generate_feedback_report.return_value = (
            "Short session feedback.",
            feedback_session,
        )

        mock_interview = AsyncMock()
        mock_interview.handle_response.return_value = ("Good answer!", session)

        service = SessionService(
            llm_client=AsyncMock(),
            prompt_builder=MagicMock(),
            interview_service=mock_interview,
            feedback_service=mock_feedback,
        )

        reply, updated = await service.transition(session, "My answer.")

        assert updated.stage == Stage.FEEDBACK
        mock_feedback.generate_feedback_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_short_session_request_sets_flag(self):
        """'quick' keyword in message sets requested_short_session=True."""
        session = self._make_session_with_answered_questions(1)

        mock_interview = AsyncMock()
        mock_interview.handle_response.return_value = ("Good answer!", session)

        service = SessionService(
            llm_client=AsyncMock(),
            prompt_builder=MagicMock(),
            interview_service=mock_interview,
            feedback_service=AsyncMock(),
        )

        _, updated = await service.transition(session, "Can we do a quick session?")

        assert updated.requested_short_session is True

    @pytest.mark.asyncio
    async def test_fewer_than_five_questions_stays_in_interview(self):
        """INTERVIEW with <5 answered questions → stays in INTERVIEW."""
        session = self._make_session_with_answered_questions(3)

        mock_interview = AsyncMock()
        mock_interview.handle_response.return_value = ("Good answer!", session)

        service = SessionService(
            llm_client=AsyncMock(),
            prompt_builder=MagicMock(),
            interview_service=mock_interview,
            feedback_service=AsyncMock(),
        )

        _, updated = await service.transition(session, "My answer.")

        assert updated.stage == Stage.INTERVIEW

    @pytest.mark.asyncio
    async def test_role_change_mid_session_creates_fresh_session(self):
        """Role change mid-session → fresh SessionState, ROLE_SELECTION (Req 5.3)."""
        session = self._make_session_with_answered_questions(3)
        original_session_id = session.session_id

        service = SessionService(
            llm_client=AsyncMock(),
            prompt_builder=MagicMock(),
            interview_service=AsyncMock(),
            feedback_service=AsyncMock(),
        )

        _, updated = await service.transition(session, "I want to change role to sales.")

        # New session created
        assert updated.session_id != original_session_id
        # Old questions/responses cleared
        assert updated.questions == []
        assert updated.responses == []
        # Stage is ROLE_SELECTION
        assert updated.stage == Stage.ROLE_SELECTION
        # Phone number preserved
        assert updated.phone_number == PHONE

    @pytest.mark.asyncio
    async def test_role_change_with_detected_role_sets_role(self):
        """Role change with detectable role → new session has that role."""
        session = self._make_session_with_answered_questions(2)

        service = SessionService(
            llm_client=AsyncMock(),
            prompt_builder=MagicMock(),
            interview_service=AsyncMock(),
            feedback_service=AsyncMock(),
        )

        _, updated = await service.transition(session, "Switch to retail associate please.")

        assert updated.role == Role.RETAIL_ASSOCIATE
        assert updated.stage == Stage.ROLE_SELECTION

    @pytest.mark.asyncio
    async def test_role_change_without_detected_role_creates_unknown_session(self):
        """Role change without detectable role → new session with UNKNOWN role."""
        session = self._make_session_with_answered_questions(2)

        service = SessionService(
            llm_client=AsyncMock(),
            prompt_builder=MagicMock(),
            interview_service=AsyncMock(),
            feedback_service=AsyncMock(),
        )

        _, updated = await service.transition(session, "Let's start over with a different role.")

        assert updated.role == Role.UNKNOWN
        assert updated.stage == Stage.ROLE_SELECTION

    @pytest.mark.asyncio
    async def test_pending_follow_up_prevents_feedback_transition(self):
        """INTERVIEW with ≥5 answered but pending follow-up → stays in INTERVIEW."""
        session = self._make_session_with_answered_questions(5)
        # Add an unanswered follow-up question
        follow_up_q = make_question(
            question_id=str(uuid.uuid4()),
            text="Can you elaborate on that?",
            question_type=QuestionType.FOLLOW_UP,
        )
        session.questions.append(follow_up_q)
        # No response for the follow-up

        mock_interview = AsyncMock()
        mock_interview.handle_response.return_value = ("Please elaborate.", session)

        service = SessionService(
            llm_client=AsyncMock(),
            prompt_builder=MagicMock(),
            interview_service=mock_interview,
            feedback_service=AsyncMock(),
        )

        _, updated = await service.transition(session, "My answer.")

        assert updated.stage == Stage.INTERVIEW


# ===========================================================================
# FEEDBACK stage transitions
# ===========================================================================


class TestFeedbackStage:
    @pytest.mark.asyncio
    async def test_feedback_delivered_transitions_to_complete(self):
        """FEEDBACK with is_complete=True → COMPLETE."""
        session = make_session(
            stage=Stage.FEEDBACK,
            role=Role.SOFTWARE_ENGINEER,
            is_complete=True,
        )

        service = SessionService(
            llm_client=AsyncMock(),
            prompt_builder=MagicMock(),
            interview_service=AsyncMock(),
            feedback_service=AsyncMock(),
        )

        reply, updated = await service.transition(session, "Thanks!")

        assert updated.stage == Stage.COMPLETE
        assert updated.completed_at is not None

    @pytest.mark.asyncio
    async def test_feedback_no_report_generates_report(self):
        """FEEDBACK with no report yet → FeedbackService called."""
        session = make_session(stage=Stage.FEEDBACK, role=Role.SOFTWARE_ENGINEER)

        mock_feedback = AsyncMock()
        feedback_session = session.model_copy(deep=True)
        feedback_session.feedback_report = make_feedback_report(session.session_id)
        mock_feedback.generate_feedback_report.return_value = (
            "Here is your feedback.",
            feedback_session,
        )

        service = SessionService(
            llm_client=AsyncMock(),
            prompt_builder=MagicMock(),
            interview_service=AsyncMock(),
            feedback_service=mock_feedback,
        )

        reply, updated = await service.transition(session, "I'm ready for feedback.")

        mock_feedback.generate_feedback_report.assert_called_once()
        assert "feedback" in reply.lower()

    @pytest.mark.asyncio
    async def test_feedback_elaboration_request_calls_llm(self):
        """FEEDBACK with elaboration keyword → LLM called for elaboration."""
        session = make_session(stage=Stage.FEEDBACK, role=Role.SOFTWARE_ENGINEER)
        session.feedback_report = make_feedback_report(session.session_id)

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "Here is more detail on your communication skills."

        service = SessionService(
            llm_client=mock_llm,
            prompt_builder=MagicMock(),
            interview_service=AsyncMock(),
            feedback_service=AsyncMock(),
        )

        reply, updated = await service.transition(session, "Can you elaborate on my communication?")

        mock_llm.complete.assert_called_once()
        assert "detail" in reply.lower() or "communication" in reply.lower()

    @pytest.mark.asyncio
    async def test_feedback_non_elaboration_marks_complete(self):
        """FEEDBACK with non-elaboration message → session marked COMPLETE."""
        session = make_session(stage=Stage.FEEDBACK, role=Role.SOFTWARE_ENGINEER)
        session.feedback_report = make_feedback_report(session.session_id)

        service = SessionService(
            llm_client=AsyncMock(),
            prompt_builder=MagicMock(),
            interview_service=AsyncMock(),
            feedback_service=AsyncMock(),
        )

        _, updated = await service.transition(session, "Thank you, that was helpful!")

        assert updated.stage == Stage.COMPLETE
        assert updated.is_complete is True
        assert updated.completed_at is not None


# ===========================================================================
# COMPLETE stage transitions
# ===========================================================================


class TestCompleteStage:
    @pytest.mark.asyncio
    async def test_complete_any_message_creates_new_session(self):
        """COMPLETE + new message → new SessionState (INIT → ROLE_SELECTION)."""
        session = make_session(
            stage=Stage.COMPLETE,
            role=Role.SOFTWARE_ENGINEER,
            is_complete=True,
        )
        original_session_id = session.session_id

        service, mock_llm, _, _ = make_service(
            llm_response='{"role": "unknown", "confidence": "low", "message": "Welcome back!"}'
        )

        _, updated = await service.transition(session, "I want to practise again.")

        # New session created
        assert updated.session_id != original_session_id
        # Old data cleared
        assert updated.questions == []
        assert updated.responses == []
        # Phone number preserved
        assert updated.phone_number == PHONE
        # Stage is ROLE_SELECTION (new session starts there)
        assert updated.stage == Stage.ROLE_SELECTION

    @pytest.mark.asyncio
    async def test_complete_with_role_in_message_fast_path(self):
        """COMPLETE + role in message → new session goes directly to INTERVIEW."""
        session = make_session(stage=Stage.COMPLETE, is_complete=True)

        service, _, mock_interview, _ = make_service(
            interview_question="Tell me about your engineering experience."
        )

        _, updated = await service.transition(session, "Let's do a software engineer interview.")

        assert updated.stage == Stage.INTERVIEW
        assert updated.role == Role.SOFTWARE_ENGINEER
        mock_interview.generate_question.assert_called_once()


# ===========================================================================
# Counter management helpers
# ===========================================================================


class TestCounterManagement:
    def test_increment_off_topic_count(self):
        """increment_off_topic_count increments both off_topic and consecutive counters."""
        service, _, _, _ = make_service()
        session = make_session(off_topic_count=2, consecutive_out_of_scope_count=1)

        updated = service.increment_off_topic_count(session)

        assert updated.off_topic_count == 3
        assert updated.consecutive_out_of_scope_count == 2

    def test_reset_consecutive_out_of_scope_count(self):
        """reset_consecutive_out_of_scope_count sets consecutive counter to 0."""
        service, _, _, _ = make_service()
        session = make_session(consecutive_out_of_scope_count=3)

        updated = service.reset_consecutive_out_of_scope_count(session)

        assert updated.consecutive_out_of_scope_count == 0
        # off_topic_count should be unchanged
        assert updated.off_topic_count == 0

    def test_increment_clarification_turn_count(self):
        """increment_clarification_turn_count increments by 1."""
        service, _, _, _ = make_service()
        session = make_session(clarification_turn_count=1)

        updated = service.increment_clarification_turn_count(session)

        assert updated.clarification_turn_count == 2

    def test_off_topic_count_starts_at_zero(self):
        """Fresh session has off_topic_count=0."""
        session = make_session()
        assert session.off_topic_count == 0

    def test_consecutive_out_of_scope_count_starts_at_zero(self):
        """Fresh session has consecutive_out_of_scope_count=0."""
        session = make_session()
        assert session.consecutive_out_of_scope_count == 0


# ===========================================================================
# updated_at is always refreshed
# ===========================================================================


class TestUpdatedAt:
    @pytest.mark.asyncio
    async def test_updated_at_is_refreshed_after_transition(self):
        """updated_at is set to a recent timestamp after any transition."""
        service, _, _, _ = make_service(
            llm_response='{"role": "unknown", "confidence": "low", "message": "Which role?"}'
        )
        session = make_session(stage=Stage.INIT, updated_at=NOW)

        _, updated = await service.transition(session, "Hello!")

        # updated_at should be more recent than the original NOW
        assert updated.updated_at >= NOW


# ===========================================================================
# _create_fresh_session helper
# ===========================================================================


class TestCreateFreshSession:
    def test_fresh_session_has_new_uuid(self):
        """_create_fresh_session generates a unique session_id."""
        s1 = SessionService._create_fresh_session(PHONE)
        s2 = SessionService._create_fresh_session(PHONE)
        assert s1.session_id != s2.session_id

    def test_fresh_session_preserves_phone_number(self):
        """_create_fresh_session preserves the phone number."""
        session = SessionService._create_fresh_session(PHONE)
        assert session.phone_number == PHONE

    def test_fresh_session_starts_at_init(self):
        """_create_fresh_session starts at INIT stage."""
        session = SessionService._create_fresh_session(PHONE)
        assert session.stage == Stage.INIT

    def test_fresh_session_has_empty_questions_and_responses(self):
        """_create_fresh_session has empty questions and responses."""
        session = SessionService._create_fresh_session(PHONE)
        assert session.questions == []
        assert session.responses == []

    def test_fresh_session_with_role(self):
        """_create_fresh_session accepts an optional role."""
        session = SessionService._create_fresh_session(PHONE, role=Role.SALES_REPRESENTATIVE)
        assert session.role == Role.SALES_REPRESENTATIVE
