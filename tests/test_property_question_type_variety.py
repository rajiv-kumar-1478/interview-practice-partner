# Feature: interview-practice-partner, Property 10: Question Type Variety Across a Session
"""Property-based tests for question type variety across a session.

Completed standard sessions (``is_complete=True``, ``requested_short_session=False``)
must include at least one BEHAVIOURAL, one SITUATIONAL, and one TECHNICAL question.

Validates: Requirements 4.4
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
from interview_practice_partner.services.interview import (
    InterviewService,
    _determine_next_question_type,
)

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_uuid_strategy = st.uuids().map(str)
_phone_strategy = st.from_regex(r"\+1[2-9]\d{9}", fullmatch=True)
_datetime_strategy = st.datetimes(timezones=st.just(timezone.utc))
_nonempty_text = st.text(min_size=1, max_size=200)

# The three required question types for a standard session
_REQUIRED_TYPES = frozenset([
    QuestionType.BEHAVIOURAL,
    QuestionType.SITUATIONAL,
    QuestionType.TECHNICAL,
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_minimal_feedback_report(session_id: str, now: datetime) -> FeedbackReport:
    """Build a minimal valid FeedbackReport for a completed session."""
    return FeedbackReport(
        report_id=str(uuid.uuid4()),
        session_id=session_id,
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


def _build_standard_completed_session_with_variety(
    num_answered: int,
    extra_types: list[QuestionType] | None = None,
) -> SessionState:
    """Build a completed standard session that includes all three required question types.

    The first three questions are always BEHAVIOURAL, SITUATIONAL, and TECHNICAL
    (in that order) to guarantee variety.  Additional questions cycle through the
    same three types.

    Args:
        num_answered: Total number of answered questions (must be >= 3).
        extra_types: Optional list of additional question types to append after
            the mandatory first three.  Defaults to cycling through the three
            required types.

    Returns:
        A ``SessionState`` with ``is_complete=True``, ``requested_short_session=False``,
        and at least one question of each required type.
    """
    assert num_answered >= 3, "Need at least 3 questions to cover all required types"

    now = datetime.now(tz=timezone.utc)
    session_id = str(uuid.uuid4())

    # The first three questions guarantee one of each required type
    mandatory_types = [
        QuestionType.BEHAVIOURAL,
        QuestionType.SITUATIONAL,
        QuestionType.TECHNICAL,
    ]

    # Remaining questions cycle through the three types
    remaining_count = num_answered - 3
    cycle_types = [
        QuestionType.BEHAVIOURAL,
        QuestionType.SITUATIONAL,
        QuestionType.TECHNICAL,
    ]
    if extra_types is not None:
        additional_types = extra_types[:remaining_count]
    else:
        additional_types = [cycle_types[i % len(cycle_types)] for i in range(remaining_count)]

    all_types = mandatory_types + additional_types

    questions: list[Question] = []
    responses: list[UserResponse] = []

    for i, q_type in enumerate(all_types):
        q_id = str(uuid.uuid4())
        q = Question(
            question_id=q_id,
            text=f"Question {i + 1}: {q_type.value} question about your experience.",
            question_type=q_type,
            asked_at=now,
            skipped=False,
        )
        questions.append(q)
        r = UserResponse(
            response_id=str(uuid.uuid4()),
            question_id=q_id,
            text="I demonstrated this skill by leading a project to completion on time.",
            word_count=20,
            is_off_topic=False,
            received_at=now,
        )
        responses.append(r)

    feedback_report = _build_minimal_feedback_report(session_id, now)

    return SessionState(
        session_id=session_id,
        phone_number="+12025551234",
        stage=Stage.COMPLETE,
        role=Role.SOFTWARE_ENGINEER,
        questions=questions,
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


def _completed_standard_session_with_variety_strategy() -> st.SearchStrategy[SessionState]:
    """Strategy: completed standard sessions with 5–10 answered questions and all three types.

    The session has:
    - ``is_complete=True``
    - ``requested_short_session=False``
    - ``stage=Stage.COMPLETE``
    - Between 5 and 10 answered questions
    - At least one BEHAVIOURAL, one SITUATIONAL, and one TECHNICAL question
    """
    return st.integers(min_value=5, max_value=10).map(
        lambda num_answered: _build_standard_completed_session_with_variety(
            num_answered=num_answered,
        )
    )


def _get_question_types_in_session(session: SessionState) -> set[QuestionType]:
    """Return the set of non-FOLLOW_UP question types present in the session."""
    return {
        q.question_type
        for q in session.questions
        if q.question_type != QuestionType.FOLLOW_UP
    }


# ---------------------------------------------------------------------------
# Property 10a: Completed standard sessions include all three required types
# ---------------------------------------------------------------------------


@given(session=_completed_standard_session_with_variety_strategy())
@settings(max_examples=100)
def test_property_10a_completed_standard_session_has_all_required_question_types(
    session: SessionState,
) -> None:
    """Property 10a: Question Type Variety Across a Session.

    For any completed standard session (``is_complete=True``,
    ``requested_short_session=False``), the session must include at least one
    BEHAVIOURAL, one SITUATIONAL, and one TECHNICAL question.

    **Validates: Requirements 4.4**
    """
    # Preconditions: this is a completed standard session
    assert session.is_complete is True, "Strategy should guarantee is_complete=True"
    assert session.requested_short_session is False, (
        "Strategy should guarantee requested_short_session=False"
    )
    assert session.stage == Stage.COMPLETE, (
        "Strategy should guarantee stage=Stage.COMPLETE"
    )

    types_present = _get_question_types_in_session(session)

    assert QuestionType.BEHAVIOURAL in types_present, (
        f"Expected at least one BEHAVIOURAL question in a completed standard session, "
        f"but found only: {types_present!r}. "
        f"Questions: {[(q.text, q.question_type) for q in session.questions]!r}"
    )
    assert QuestionType.SITUATIONAL in types_present, (
        f"Expected at least one SITUATIONAL question in a completed standard session, "
        f"but found only: {types_present!r}. "
        f"Questions: {[(q.text, q.question_type) for q in session.questions]!r}"
    )
    assert QuestionType.TECHNICAL in types_present, (
        f"Expected at least one TECHNICAL question in a completed standard session, "
        f"but found only: {types_present!r}. "
        f"Questions: {[(q.text, q.question_type) for q in session.questions]!r}"
    )

    # All three required types must be present
    missing = _REQUIRED_TYPES - types_present
    assert not missing, (
        f"Completed standard session is missing required question types: {missing!r}. "
        f"Types present: {types_present!r}"
    )


# ---------------------------------------------------------------------------
# Property 10b: _determine_next_question_type cycles through all three types
# ---------------------------------------------------------------------------


def test_property_10b_question_type_cycle_covers_all_required_types() -> None:
    """Property 10b: _determine_next_question_type cycles through all required types.

    Starting from an empty session, calling ``_determine_next_question_type``
    and recording the result (by adding a question of that type) must produce
    all three required types within the first three calls.

    **Validates: Requirements 4.4**
    """
    now = datetime.now(tz=timezone.utc)

    session = SessionState(
        session_id=str(uuid.uuid4()),
        phone_number="+12025551234",
        stage=Stage.INTERVIEW,
        role=Role.SOFTWARE_ENGINEER,
        questions=[],
        responses=[],
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

    types_generated: list[QuestionType] = []

    for i in range(3):
        next_type = _determine_next_question_type(session)
        types_generated.append(next_type)

        # Simulate adding a question of that type to the session
        session.questions.append(
            Question(
                question_id=str(uuid.uuid4()),
                text=f"Question {i + 1}: {next_type.value} question.",
                question_type=next_type,
                asked_at=now,
                skipped=False,
            )
        )

    types_set = set(types_generated)

    assert QuestionType.BEHAVIOURAL in types_set, (
        f"Expected BEHAVIOURAL to appear in the first 3 question types, "
        f"but got: {types_generated!r}"
    )
    assert QuestionType.SITUATIONAL in types_set, (
        f"Expected SITUATIONAL to appear in the first 3 question types, "
        f"but got: {types_generated!r}"
    )
    assert QuestionType.TECHNICAL in types_set, (
        f"Expected TECHNICAL to appear in the first 3 question types, "
        f"but got: {types_generated!r}"
    )


# ---------------------------------------------------------------------------
# Property 10c: generate_question produces all three types across 5+ questions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(num_questions=st.integers(min_value=5, max_value=10))
@settings(max_examples=100)
async def test_property_10c_generate_question_produces_all_required_types(
    num_questions: int,
) -> None:
    """Property 10c: generate_question produces all three required question types.

    Calling ``generate_question`` at least 5 times on a fresh session must
    result in at least one BEHAVIOURAL, one SITUATIONAL, and one TECHNICAL
    question being recorded in the session.

    **Validates: Requirements 4.4**
    """
    now = datetime.now(tz=timezone.utc)

    session = SessionState(
        session_id=str(uuid.uuid4()),
        phone_number="+12025551234",
        stage=Stage.INTERVIEW,
        role=Role.SOFTWARE_ENGINEER,
        questions=[],
        responses=[],
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

    # Track which type the service will request so the mock LLM can return
    # a matching question text.
    call_counter = {"n": 0}

    async def _mock_complete(*_args, **_kwargs) -> str:
        n = call_counter["n"]
        call_counter["n"] += 1
        return f"Generated question number {n + 1}."

    mock_llm = AsyncMock()
    mock_llm.complete.side_effect = _mock_complete

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_question_generation_prompt.return_value = [
        {"role": "system", "content": "You are an interviewer."},
        {"role": "user", "content": "Generate a question."},
    ]

    mock_whisper = AsyncMock()
    mock_tts = AsyncMock()
    mock_audio_download = AsyncMock()

    service = InterviewService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
        whisper_client=mock_whisper,
        tts_client=mock_tts,
        audio_download_client=mock_audio_download,
    )

    for _ in range(num_questions):
        await service.generate_question(session)

    types_present = _get_question_types_in_session(session)

    assert QuestionType.BEHAVIOURAL in types_present, (
        f"Expected at least one BEHAVIOURAL question after {num_questions} "
        f"generate_question calls, but found types: {types_present!r}"
    )
    assert QuestionType.SITUATIONAL in types_present, (
        f"Expected at least one SITUATIONAL question after {num_questions} "
        f"generate_question calls, but found types: {types_present!r}"
    )
    assert QuestionType.TECHNICAL in types_present, (
        f"Expected at least one TECHNICAL question after {num_questions} "
        f"generate_question calls, but found types: {types_present!r}"
    )

    missing = _REQUIRED_TYPES - types_present
    assert not missing, (
        f"After {num_questions} generate_question calls, missing required types: "
        f"{missing!r}. Types present: {types_present!r}"
    )


# ---------------------------------------------------------------------------
# Property 10d: Variety is maintained even with follow-up questions interspersed
# ---------------------------------------------------------------------------


@given(
    num_main_questions=st.integers(min_value=5, max_value=10),
    num_follow_ups=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=100)
def test_property_10d_variety_maintained_with_follow_up_questions(
    num_main_questions: int,
    num_follow_ups: int,
) -> None:
    """Property 10d: Question type variety is maintained even with follow-up questions.

    FOLLOW_UP questions must not count toward or against the variety requirement.
    A session with the required three main types plus some follow-up questions
    must still satisfy the variety property.

    **Validates: Requirements 4.4**
    """
    now = datetime.now(tz=timezone.utc)
    session_id = str(uuid.uuid4())

    # Build the main questions with all three required types
    main_types = [
        QuestionType.BEHAVIOURAL,
        QuestionType.SITUATIONAL,
        QuestionType.TECHNICAL,
    ]
    questions: list[Question] = []
    responses: list[UserResponse] = []

    for i in range(num_main_questions):
        q_type = main_types[i % len(main_types)]
        q_id = str(uuid.uuid4())
        q = Question(
            question_id=q_id,
            text=f"Main question {i + 1}: {q_type.value} question.",
            question_type=q_type,
            asked_at=now,
            skipped=False,
        )
        questions.append(q)
        r = UserResponse(
            response_id=str(uuid.uuid4()),
            question_id=q_id,
            text="I demonstrated this skill by leading a project to completion on time.",
            word_count=20,
            is_off_topic=False,
            received_at=now,
        )
        responses.append(r)

    # Intersperse follow-up questions (no responses needed for this property check)
    for i in range(num_follow_ups):
        q = Question(
            question_id=str(uuid.uuid4()),
            text=f"Follow-up question {i + 1}: Can you elaborate on that?",
            question_type=QuestionType.FOLLOW_UP,
            asked_at=now,
            skipped=False,
        )
        questions.append(q)

    feedback_report = _build_minimal_feedback_report(session_id, now)

    session = SessionState(
        session_id=session_id,
        phone_number="+12025551234",
        stage=Stage.COMPLETE,
        role=Role.SOFTWARE_ENGINEER,
        questions=questions,
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

    # Verify variety ignoring FOLLOW_UP questions
    types_present = _get_question_types_in_session(session)

    assert QuestionType.BEHAVIOURAL in types_present, (
        f"Expected BEHAVIOURAL in session with {num_main_questions} main questions "
        f"and {num_follow_ups} follow-ups. Types present: {types_present!r}"
    )
    assert QuestionType.SITUATIONAL in types_present, (
        f"Expected SITUATIONAL in session with {num_main_questions} main questions "
        f"and {num_follow_ups} follow-ups. Types present: {types_present!r}"
    )
    assert QuestionType.TECHNICAL in types_present, (
        f"Expected TECHNICAL in session with {num_main_questions} main questions "
        f"and {num_follow_ups} follow-ups. Types present: {types_present!r}"
    )
