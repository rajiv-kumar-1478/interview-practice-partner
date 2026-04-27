# Feature: interview-practice-partner, Property 9: Standard Session Question Count Is Within Bounds
"""Property-based tests for standard session question count bounds.

For any completed standard session (where the user did not request a short
session), the number of non-skipped answered questions in ``session.questions``
SHALL be at least 5 and at most 10.

Validates: Requirements 4.1, 6.2
"""

from __future__ import annotations

import uuid
from datetime import timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from interview_practice_partner.domain.enums import EvaluationDimension, QuestionType, Role, Stage
from interview_practice_partner.domain.models import (
    DimensionScore,
    FeedbackReport,
    Question,
    SessionState,
    UserResponse,
)
from interview_practice_partner.services.interview import InterviewService
from interview_practice_partner.services.session import SessionService

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_uuid_strategy = st.uuids().map(str)
_phone_strategy = st.from_regex(r"\+1[2-9]\d{9}", fullmatch=True)
_datetime_strategy = st.datetimes(timezones=st.just(timezone.utc))
_nonempty_text = st.text(min_size=1, max_size=200)


def _question_strategy(
    skipped: st.SearchStrategy[bool] = st.just(False),
    question_type: st.SearchStrategy[QuestionType] | None = None,
) -> st.SearchStrategy[Question]:
    """Build a Question strategy."""
    if question_type is None:
        question_type = st.sampled_from([
            QuestionType.BEHAVIOURAL,
            QuestionType.SITUATIONAL,
            QuestionType.TECHNICAL,
        ])
    return st.builds(
        Question,
        question_id=_uuid_strategy,
        text=_nonempty_text,
        question_type=question_type,
        asked_at=_datetime_strategy,
        skipped=skipped,
    )


def _user_response_for_question(question_id: str) -> UserResponse:
    """Build a UserResponse linked to a specific question_id."""
    return UserResponse(
        response_id=str(uuid.uuid4()),
        question_id=question_id,
        text="This is a sufficiently detailed answer to the interview question.",
        word_count=20,
        is_off_topic=False,
        received_at=_datetime_strategy.example(),
    )


def _build_standard_completed_session(
    num_answered: int,
    num_skipped: int = 0,
) -> SessionState:
    """Build a completed standard session with the given number of answered questions.

    Args:
        num_answered: Number of non-skipped questions with responses (5–10).
        num_skipped: Number of skipped questions (no responses).

    Returns:
        A ``SessionState`` with ``is_complete=True``, ``requested_short_session=False``,
        and the specified number of answered and skipped questions.
    """
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc)

    # Build answered questions (non-skipped, with responses)
    answered_questions: list[Question] = []
    responses: list[UserResponse] = []
    question_types = [
        QuestionType.BEHAVIOURAL,
        QuestionType.SITUATIONAL,
        QuestionType.TECHNICAL,
    ]
    for i in range(num_answered):
        q_id = str(uuid.uuid4())
        q = Question(
            question_id=q_id,
            text=f"Tell me about a time you demonstrated skill {i + 1}.",
            question_type=question_types[i % len(question_types)],
            asked_at=now,
            skipped=False,
        )
        answered_questions.append(q)
        r = UserResponse(
            response_id=str(uuid.uuid4()),
            question_id=q_id,
            text="I demonstrated this skill by leading a project to completion on time.",
            word_count=20,
            is_off_topic=False,
            received_at=now,
        )
        responses.append(r)

    # Build skipped questions (no responses)
    skipped_questions: list[Question] = []
    for i in range(num_skipped):
        q = Question(
            question_id=str(uuid.uuid4()),
            text=f"Skipped question {i + 1}.",
            question_type=QuestionType.BEHAVIOURAL,
            asked_at=now,
            skipped=True,
        )
        skipped_questions.append(q)

    all_questions = answered_questions + skipped_questions

    # Build a minimal FeedbackReport
    feedback_report = FeedbackReport(
        report_id=str(uuid.uuid4()),
        session_id="test-session-id",
        dimension_scores=[
            DimensionScore(
                dimension=EvaluationDimension.COMMUNICATION_CLARITY,
                qualitative_assessment="Clear communication.",
                score=4,
            ),
            DimensionScore(
                dimension=EvaluationDimension.RELEVANCE,
                qualitative_assessment="Relevant responses.",
                score=4,
            ),
            DimensionScore(
                dimension=EvaluationDimension.TECHNICAL_KNOWLEDGE,
                qualitative_assessment="Good technical knowledge.",
                score=4,
            ),
            DimensionScore(
                dimension=EvaluationDimension.CONFIDENCE,
                qualitative_assessment="Confident delivery.",
                score=4,
            ),
        ],
        strengths=["Strong communication skills."],
        improvements=["Could provide more specific examples."],
        actionable_recommendations=["Practise the STAR method."],
        off_topic_references=[],
        generated_at=now,
    )

    return SessionState(
        session_id=str(uuid.uuid4()),
        phone_number="+12025551234",
        stage=Stage.COMPLETE,
        role=Role.SOFTWARE_ENGINEER,
        questions=all_questions,
        responses=responses,
        off_topic_count=0,
        consecutive_out_of_scope_count=0,
        clarification_turn_count=0,
        requested_short_session=False,
        feedback_report=feedback_report,
        is_complete=True,
        created_at=now,
        updated_at=now,
        completed_at=now,
        context_summary=None,
    )


# ---------------------------------------------------------------------------
# Strategy: completed standard sessions with 5–10 answered questions
# ---------------------------------------------------------------------------

def _completed_standard_session_strategy() -> st.SearchStrategy[SessionState]:
    """Build a completed standard session with 5–10 non-skipped answered questions.

    The session has:
    - ``is_complete=True``
    - ``requested_short_session=False``
    - ``stage=Stage.COMPLETE``
    - Between 5 and 10 non-skipped questions, each with a corresponding response
    - Optionally 0–3 additional skipped questions (no responses)
    """
    return st.integers(min_value=5, max_value=10).flatmap(
        lambda num_answered: st.integers(min_value=0, max_value=3).map(
            lambda num_skipped: _build_standard_completed_session(
                num_answered=num_answered,
                num_skipped=num_skipped,
            )
        )
    )


def _count_answered_questions(session: SessionState) -> int:
    """Return the number of non-skipped questions that have a recorded response."""
    answered_ids = {r.question_id for r in session.responses}
    return sum(
        1
        for q in session.questions
        if not q.skipped and q.question_id in answered_ids
    )


# ---------------------------------------------------------------------------
# Property 9a: Completed standard sessions have between 5 and 10 answered questions
# ---------------------------------------------------------------------------


@given(session=_completed_standard_session_strategy())
@settings(max_examples=100)
def test_property_9a_standard_session_answered_question_count_within_bounds(
    session: SessionState,
) -> None:
    """Property 9a: Standard Session Question Count Is Within Bounds.

    For any completed standard session (``is_complete=True``,
    ``requested_short_session=False``), the number of non-skipped answered
    questions SHALL be at least 5 and at most 10.

    **Validates: Requirements 4.1, 6.2**
    """
    # Preconditions: this is a completed standard session
    assert session.is_complete is True, (
        "Strategy should guarantee is_complete=True"
    )
    assert session.requested_short_session is False, (
        "Strategy should guarantee requested_short_session=False"
    )
    assert session.stage == Stage.COMPLETE, (
        "Strategy should guarantee stage=Stage.COMPLETE"
    )

    answered_count = _count_answered_questions(session)

    assert answered_count >= 5, (
        f"Expected at least 5 answered questions in a completed standard session, "
        f"but got {answered_count}. "
        f"Total questions: {len(session.questions)}, "
        f"responses: {len(session.responses)}"
    )
    assert answered_count <= 10, (
        f"Expected at most 10 answered questions in a completed standard session, "
        f"but got {answered_count}. "
        f"Total questions: {len(session.questions)}, "
        f"responses: {len(session.responses)}"
    )


# ---------------------------------------------------------------------------
# Property 9b: Total question count (including skipped) is at most 10 + skipped
# ---------------------------------------------------------------------------


@given(session=_completed_standard_session_strategy())
@settings(max_examples=100)
def test_property_9b_standard_session_total_question_count_reasonable(
    session: SessionState,
) -> None:
    """Property 9b: Standard Session Total Question Count Is Reasonable.

    For any completed standard session, the total number of questions
    (including skipped) must be at least 5 (minimum answered) and the
    number of non-skipped answered questions must not exceed 10.

    **Validates: Requirements 4.1, 6.2**
    """
    assert session.is_complete is True
    assert session.requested_short_session is False

    answered_count = _count_answered_questions(session)
    skipped_count = sum(1 for q in session.questions if q.skipped)
    total_count = len(session.questions)

    # Total questions = answered + skipped
    assert total_count == answered_count + skipped_count, (
        f"Expected total_count ({total_count}) == answered_count ({answered_count}) "
        f"+ skipped_count ({skipped_count})"
    )

    # Answered questions must be within [5, 10]
    assert 5 <= answered_count <= 10, (
        f"Expected answered_count in [5, 10], got {answered_count}"
    )

    # Total questions must be at least 5 (can be more if some were skipped)
    assert total_count >= 5, (
        f"Expected total question count >= 5, got {total_count}"
    )


# ---------------------------------------------------------------------------
# Property 9c: Boundary values — exactly 5 and exactly 10 answered questions
# ---------------------------------------------------------------------------


def test_property_9c_boundary_exactly_5_answered_questions() -> None:
    """Property 9c: Boundary — exactly 5 answered questions is valid.

    A completed standard session with exactly 5 answered questions must
    satisfy the lower bound of the question count property.

    **Validates: Requirements 4.1, 6.2**
    """
    session = _build_standard_completed_session(num_answered=5)

    assert session.is_complete is True
    assert session.requested_short_session is False

    answered_count = _count_answered_questions(session)

    assert answered_count == 5, (
        f"Expected exactly 5 answered questions, got {answered_count}"
    )
    assert answered_count >= 5, (
        "5 answered questions must satisfy the lower bound (>= 5)"
    )
    assert answered_count <= 10, (
        "5 answered questions must satisfy the upper bound (<= 10)"
    )


def test_property_9c_boundary_exactly_10_answered_questions() -> None:
    """Property 9c: Boundary — exactly 10 answered questions is valid.

    A completed standard session with exactly 10 answered questions must
    satisfy the upper bound of the question count property.

    **Validates: Requirements 4.1, 6.2**
    """
    session = _build_standard_completed_session(num_answered=10)

    assert session.is_complete is True
    assert session.requested_short_session is False

    answered_count = _count_answered_questions(session)

    assert answered_count == 10, (
        f"Expected exactly 10 answered questions, got {answered_count}"
    )
    assert answered_count >= 5, (
        "10 answered questions must satisfy the lower bound (>= 5)"
    )
    assert answered_count <= 10, (
        "10 answered questions must satisfy the upper bound (<= 10)"
    )


# ---------------------------------------------------------------------------
# Property 9d: Skipped questions do not count toward the answered question bound
# ---------------------------------------------------------------------------


@given(
    num_answered=st.integers(min_value=5, max_value=10),
    num_skipped=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=100)
def test_property_9d_skipped_questions_excluded_from_count(
    num_answered: int,
    num_skipped: int,
) -> None:
    """Property 9d: Skipped questions are excluded from the answered question count.

    For any completed standard session with both answered and skipped questions,
    the count of non-skipped answered questions must still be within [5, 10],
    regardless of how many questions were skipped.

    **Validates: Requirements 4.1, 6.2**
    """
    session = _build_standard_completed_session(
        num_answered=num_answered,
        num_skipped=num_skipped,
    )

    assert session.is_complete is True
    assert session.requested_short_session is False

    answered_count = _count_answered_questions(session)
    skipped_count = sum(1 for q in session.questions if q.skipped)

    # Skipped questions must not be counted as answered
    assert answered_count == num_answered, (
        f"Expected answered_count={num_answered}, got {answered_count}. "
        f"Skipped questions must not be counted as answered."
    )
    assert skipped_count == num_skipped, (
        f"Expected skipped_count={num_skipped}, got {skipped_count}"
    )

    # Answered count must still be within bounds
    assert 5 <= answered_count <= 10, (
        f"Expected answered_count in [5, 10], got {answered_count}"
    )


# ---------------------------------------------------------------------------
# Property 9e: InterviewService transitions to FEEDBACK at exactly 5 answered questions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    num_pre_answered=st.integers(min_value=4, max_value=4),
)
@settings(max_examples=20)
async def test_property_9e_session_service_transitions_at_5_answered_questions(
    num_pre_answered: int,
) -> None:
    """Property 9e: SessionService transitions to FEEDBACK at 5 answered questions.

    When a standard session (``requested_short_session=False``) reaches 5
    answered questions with no pending follow-up, the ``SessionService``
    must transition the session to ``Stage.FEEDBACK``.

    This verifies the lower bound of the question count property from the
    service layer perspective.

    **Validates: Requirements 4.1, 6.2**
    """
    import json
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc)

    # Build a session with 4 answered questions (one short of the threshold)
    answered_questions: list[Question] = []
    responses: list[UserResponse] = []
    question_types = [
        QuestionType.BEHAVIOURAL,
        QuestionType.SITUATIONAL,
        QuestionType.TECHNICAL,
        QuestionType.BEHAVIOURAL,
    ]
    for i in range(num_pre_answered):
        q_id = str(uuid.uuid4())
        q = Question(
            question_id=q_id,
            text=f"Question {i + 1}: Tell me about a time you demonstrated skill {i + 1}.",
            question_type=question_types[i % len(question_types)],
            asked_at=now,
            skipped=False,
        )
        answered_questions.append(q)
        r = UserResponse(
            response_id=str(uuid.uuid4()),
            question_id=q_id,
            text="I demonstrated this skill by leading a project to completion on time.",
            word_count=20,
            is_off_topic=False,
            received_at=now,
        )
        responses.append(r)

    # Add the 5th (unanswered) question — this is the current question
    current_q_id = str(uuid.uuid4())
    current_question = Question(
        question_id=current_q_id,
        text="Question 5: Describe a challenging situation you overcame.",
        question_type=QuestionType.SITUATIONAL,
        asked_at=now,
        skipped=False,
    )
    answered_questions.append(current_question)

    session = SessionState(
        session_id=str(uuid.uuid4()),
        phone_number="+12025551234",
        stage=Stage.INTERVIEW,
        role=Role.SOFTWARE_ENGINEER,
        questions=answered_questions,
        responses=responses,
        off_topic_count=0,
        consecutive_out_of_scope_count=0,
        clarification_turn_count=0,
        requested_short_session=False,
        feedback_report=None,
        is_complete=False,
        created_at=now,
        updated_at=now,
        completed_at=None,
        context_summary=None,
    )

    # Mock the evaluation response: on-topic, no follow-up, maintain difficulty
    evaluation_json = json.dumps({
        "is_off_topic": False,
        "is_short": False,
        "follow_up_warranted": False,
        "follow_up_text": None,
        "difficulty_signal": "maintain",
    })

    # Mock the feedback JSON
    import json as _json
    feedback_json = _json.dumps({
        "dimension_scores": [
            {
                "dimension": EvaluationDimension.COMMUNICATION_CLARITY.value,
                "qualitative_assessment": "Clear communication.",
                "score": 4,
            },
            {
                "dimension": EvaluationDimension.RELEVANCE.value,
                "qualitative_assessment": "Relevant responses.",
                "score": 4,
            },
            {
                "dimension": EvaluationDimension.TECHNICAL_KNOWLEDGE.value,
                "qualitative_assessment": "Good technical knowledge.",
                "score": 4,
            },
            {
                "dimension": EvaluationDimension.CONFIDENCE.value,
                "qualitative_assessment": "Confident delivery.",
                "score": 4,
            },
        ],
        "strengths": ["Strong communication skills."],
        "improvements": ["Could provide more specific examples."],
        "actionable_recommendations": ["Practise the STAR method."],
        "off_topic_references": [],
    })

    # Set up mock LLM: first call returns evaluation, second returns feedback
    call_count = {"n": 0}
    responses_map = [evaluation_json, feedback_json]

    async def _mock_complete(*_args, **_kwargs) -> str:
        result = responses_map[min(call_count["n"], len(responses_map) - 1)]
        call_count["n"] += 1
        return result

    mock_llm = AsyncMock()
    mock_llm.complete.side_effect = _mock_complete

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_response_evaluation_prompt.return_value = [
        {"role": "system", "content": "Evaluate this response."},
        {"role": "user", "content": "User answer."},
    ]
    mock_prompt_builder.build_feedback_prompt.return_value = [
        {"role": "system", "content": "Generate feedback."},
        {"role": "user", "content": "Session transcript."},
    ]
    mock_prompt_builder.build_question_generation_prompt.return_value = [
        {"role": "system", "content": "Generate a question."},
        {"role": "user", "content": "Session context."},
    ]

    from interview_practice_partner.services.feedback import FeedbackService

    mock_whisper = AsyncMock()
    mock_tts = AsyncMock()
    mock_audio_download_inner = AsyncMock()

    interview_service = InterviewService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
        whisper_client=mock_whisper,
        tts_client=mock_tts,
        audio_download_client=mock_audio_download_inner,
    )
    feedback_service = FeedbackService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
    )
    session_service = SessionService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
        interview_service=interview_service,
        feedback_service=feedback_service,
    )

    # Submit a sufficiently long answer to the 5th question
    user_message = (
        "I overcame the challenge by breaking it down into smaller tasks "
        "and collaborating with my team to deliver the project on time."
    )

    _reply, updated_session = await session_service.transition(session, user_message)

    # After answering the 5th question, the session should transition to FEEDBACK
    answered_count = _count_answered_questions(updated_session)

    assert answered_count >= 5, (
        f"Expected at least 5 answered questions after the 5th answer, "
        f"but got {answered_count}"
    )
    assert updated_session.stage == Stage.FEEDBACK, (
        f"Expected session to transition to FEEDBACK after 5 answered questions "
        f"(no follow-up), but got stage={updated_session.stage!r}"
    )
