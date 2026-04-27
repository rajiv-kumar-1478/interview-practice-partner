# Feature: interview-practice-partner, Property 14: Skipped Questions Are Recorded in Session State
"""Property-based tests for skipped question recording.

Skipped questions must have ``skipped=True`` in the session state, and the
``FeedbackReport`` must contain no negative assessment attributable solely to
the skip (i.e., skipped questions are not penalised in the feedback).

Validates: Requirements 6.4
"""

from __future__ import annotations

import json
from datetime import timezone
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
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
    Question,
    SessionState,
    UserResponse,
)
from interview_practice_partner.repositories.redis_session import RedisSessionRepository
from interview_practice_partner.services.feedback import FeedbackService
from interview_practice_partner.services.interview import InterviewService

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_nonempty_text = st.text(min_size=1, max_size=200)

# Use UUIDs for IDs to guarantee uniqueness within a generated session
_uuid_strategy = st.uuids().map(str)

_phone_strategy = st.from_regex(r"\+1[2-9]\d{9}", fullmatch=True)

_datetime_strategy = st.datetimes(timezones=st.just(timezone.utc))


def _question_strategy(skipped: st.SearchStrategy[bool] = st.just(False)) -> st.SearchStrategy[Question]:
    """Build a Question strategy with configurable skipped flag."""
    return st.builds(
        Question,
        question_id=_uuid_strategy,
        text=_nonempty_text,
        question_type=st.sampled_from(list(QuestionType)),
        asked_at=_datetime_strategy,
        skipped=skipped,
    )


def _user_response_strategy(
    is_off_topic: st.SearchStrategy[bool] = st.booleans(),
) -> st.SearchStrategy[UserResponse]:
    """Build a UserResponse strategy with configurable is_off_topic flag."""
    return st.builds(
        UserResponse,
        response_id=_uuid_strategy,
        question_id=_uuid_strategy,
        text=_nonempty_text,
        word_count=st.integers(min_value=15, max_value=200),
        is_off_topic=is_off_topic,
        received_at=_datetime_strategy,
    )


def _interview_session_with_unanswered_questions() -> st.SearchStrategy[SessionState]:
    """Build an INTERVIEW SessionState with 1–5 unanswered, non-skipped questions."""
    return st.builds(
        SessionState,
        session_id=_uuid_strategy,
        phone_number=_phone_strategy,
        stage=st.just(Stage.INTERVIEW),
        role=st.sampled_from([r for r in Role if r != Role.UNKNOWN]),
        questions=st.lists(_question_strategy(skipped=st.just(False)), min_size=1, max_size=5),
        responses=st.just([]),
        off_topic_count=st.just(0),
        consecutive_out_of_scope_count=st.just(0),
        clarification_turn_count=st.just(0),
        requested_short_session=st.booleans(),
        is_complete=st.just(False),
        created_at=_datetime_strategy,
        updated_at=_datetime_strategy,
        completed_at=st.none(),
        feedback_report=st.none(),
        context_summary=st.none(),
    )


def _session_with_mixed_skipped_questions() -> st.SearchStrategy[SessionState]:
    """Build a SessionState with 0–5 questions where each may be skipped."""
    return st.builds(
        SessionState,
        session_id=_uuid_strategy,
        phone_number=_phone_strategy,
        stage=st.just(Stage.INTERVIEW),
        role=st.sampled_from(list(Role)),
        questions=st.lists(
            _question_strategy(skipped=st.booleans()),
            min_size=0,
            max_size=5,
        ),
        responses=st.just([]),
        off_topic_count=st.just(0),
        consecutive_out_of_scope_count=st.just(0),
        clarification_turn_count=st.just(0),
        requested_short_session=st.booleans(),
        is_complete=st.just(False),
        created_at=_datetime_strategy,
        updated_at=_datetime_strategy,
        completed_at=st.none(),
        feedback_report=st.none(),
        context_summary=st.none(),
    )


def _feedback_session_strategy() -> st.SearchStrategy[SessionState]:
    """Build a FEEDBACK SessionState with skipped questions and on-topic responses."""
    return st.builds(
        SessionState,
        session_id=_uuid_strategy,
        phone_number=_phone_strategy,
        stage=st.just(Stage.FEEDBACK),
        role=st.sampled_from([r for r in Role if r != Role.UNKNOWN]),
        questions=st.lists(
            _question_strategy(skipped=st.booleans()),
            min_size=3,
            max_size=5,
        ),
        responses=st.lists(
            _user_response_strategy(is_off_topic=st.just(False)),
            min_size=1,
            max_size=3,
        ),
        off_topic_count=st.just(0),
        consecutive_out_of_scope_count=st.just(0),
        clarification_turn_count=st.just(0),
        requested_short_session=st.booleans(),
        is_complete=st.just(False),
        created_at=_datetime_strategy,
        updated_at=_datetime_strategy,
        completed_at=st.none(),
        feedback_report=st.none(),
        context_summary=st.none(),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_FEEDBACK_JSON = json.dumps({
    "dimension_scores": [
        {
            "dimension": EvaluationDimension.COMMUNICATION_CLARITY.value,
            "qualitative_assessment": "Clear and concise communication.",
            "score": 4,
        },
        {
            "dimension": EvaluationDimension.RELEVANCE.value,
            "qualitative_assessment": "Responses were relevant to the questions.",
            "score": 4,
        },
        {
            "dimension": EvaluationDimension.TECHNICAL_KNOWLEDGE.value,
            "qualitative_assessment": "Demonstrated solid technical knowledge.",
            "score": 4,
        },
        {
            "dimension": EvaluationDimension.CONFIDENCE.value,
            "qualitative_assessment": "Spoke with confidence throughout.",
            "score": 4,
        },
    ],
    "strengths": ["Good communication skills."],
    "improvements": ["Could provide more specific examples."],
    "actionable_recommendations": ["Practise the STAR method for behavioural questions."],
    "off_topic_references": [],
})


def _make_interview_service() -> InterviewService:
    """Build an InterviewService with mocked LLM and PromptBuilder."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "Tell me about a time you solved a difficult problem."

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_question_generation_prompt.return_value = [
        {"role": "system", "content": "You are an interviewer."},
        {"role": "user", "content": "Generate a question."},
    ]

    mock_whisper = AsyncMock()
    mock_tts = AsyncMock()
    mock_audio_download = AsyncMock()

    return InterviewService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
        whisper_client=mock_whisper,
        tts_client=mock_tts,
        audio_download_client=mock_audio_download,
    )


def _make_feedback_service() -> FeedbackService:
    """Build a FeedbackService with mocked LLM and PromptBuilder."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = _VALID_FEEDBACK_JSON

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_feedback_prompt.return_value = [
        {"role": "system", "content": "You are a feedback generator."},
        {"role": "user", "content": "Generate feedback."},
    ]

    return FeedbackService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
    )


# ---------------------------------------------------------------------------
# Property 14a: handle_skip sets skipped=True on the current question
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_interview_session_with_unanswered_questions())
@settings(max_examples=100)
async def test_property_14a_handle_skip_sets_skipped_true(
    session: SessionState,
) -> None:
    """Property 14a: handle_skip sets skipped=True on the most recent unanswered question.

    For any INTERVIEW session with at least one unanswered, non-skipped question,
    calling ``handle_skip`` must set ``skipped=True`` on the most recent unanswered
    question, return a non-empty reply, and add one new question to the session.

    **Validates: Requirements 6.4**
    """
    service = _make_interview_service()

    initial_question_count = len(session.questions)

    # Identify the most recent unanswered, non-skipped question before the call
    answered_ids = {r.question_id for r in session.responses}
    expected_skipped_question = None
    for q in reversed(session.questions):
        if not q.skipped and q.question_id not in answered_ids:
            expected_skipped_question = q
            break

    # There must be at least one unanswered question (guaranteed by strategy)
    assert expected_skipped_question is not None, (
        "Strategy should guarantee at least one unanswered, non-skipped question"
    )

    reply, updated_session = await service.handle_skip(session)

    # The reply must be a non-empty string
    assert isinstance(reply, str), (
        f"Expected handle_skip to return a str reply, got {type(reply)!r}"
    )
    assert len(reply.strip()) > 0, (
        "Expected handle_skip to return a non-empty reply string"
    )

    # The most recent unanswered question must now be skipped
    skipped_question = next(
        (q for q in updated_session.questions if q.question_id == expected_skipped_question.question_id),
        None,
    )
    assert skipped_question is not None, (
        f"Expected question {expected_skipped_question.question_id} to still be in session"
    )
    assert skipped_question.skipped is True, (
        f"Expected question {expected_skipped_question.question_id} to have skipped=True "
        f"after handle_skip, but got skipped={skipped_question.skipped}"
    )

    # A new question must have been generated and appended
    assert len(updated_session.questions) == initial_question_count + 1, (
        f"Expected session to have {initial_question_count + 1} questions after handle_skip "
        f"(one skipped + one new), but got {len(updated_session.questions)}"
    )


# ---------------------------------------------------------------------------
# Property 14b: All skipped questions are preserved through a Redis round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_session_with_mixed_skipped_questions())
@settings(max_examples=100)
async def test_property_14b_skipped_flag_preserved_through_redis_round_trip(
    session: SessionState,
) -> None:
    """Property 14b: All skipped questions in session state have skipped=True after round-trip.

    For any ``SessionState`` with a mix of skipped and non-skipped questions,
    every question's ``skipped`` flag must be faithfully preserved through a
    Redis save → get round-trip.

    **Validates: Requirements 6.4**
    """
    redis_client = fakeredis.aioredis.FakeRedis()
    repo = RedisSessionRepository(redis_client=redis_client, ttl_seconds=86400)

    # Record the original skipped state of each question
    original_skipped = {q.question_id: q.skipped for q in session.questions}
    original_skipped_count = sum(1 for skipped in original_skipped.values() if skipped)

    # Persist and retrieve
    await repo.save(session)
    retrieved = await repo.get(session.phone_number)

    assert retrieved is not None, (
        f"Expected a session for {session.phone_number} but got None after round-trip"
    )

    # Every question that was skipped must still be skipped
    for q in retrieved.questions:
        expected = original_skipped.get(q.question_id)
        assert expected is not None, (
            f"Question {q.question_id} appeared after round-trip but was not in original session"
        )
        assert q.skipped == expected, (
            f"Expected question {q.question_id} to have skipped={expected} after round-trip, "
            f"but got skipped={q.skipped}"
        )

    # The count of skipped questions must be preserved
    retrieved_skipped_count = sum(1 for q in retrieved.questions if q.skipped)
    assert retrieved_skipped_count == original_skipped_count, (
        f"Expected {original_skipped_count} skipped questions after round-trip, "
        f"but got {retrieved_skipped_count}"
    )

    # The total question count must be preserved
    assert len(retrieved.questions) == len(session.questions), (
        f"Expected {len(session.questions)} questions after round-trip, "
        f"but got {len(retrieved.questions)}"
    )


# ---------------------------------------------------------------------------
# Property 14c: FeedbackReport does not penalise skipped questions via off_topic_references
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_feedback_session_strategy())
@settings(max_examples=100)
async def test_property_14c_feedback_report_no_off_topic_references_for_skipped_questions(
    session: SessionState,
) -> None:
    """Property 14c: FeedbackReport does not penalise skipped questions via off_topic_references.

    For any session where questions are skipped but no responses are off-topic
    (``is_off_topic=False`` on all responses, ``off_topic_count=0``), the
    generated ``FeedbackReport`` must have an empty ``off_topic_references`` list.

    **Validates: Requirements 6.4**
    """
    service = _make_feedback_service()

    # Preconditions guaranteed by strategy, but assert explicitly
    assert session.off_topic_count == 0, (
        "Strategy should guarantee off_topic_count=0"
    )
    assert all(not r.is_off_topic for r in session.responses), (
        "Strategy should guarantee all responses have is_off_topic=False"
    )

    _reply, updated_session = await service.generate_feedback_report(session)

    report = updated_session.feedback_report
    assert report is not None, "Expected feedback_report to be populated after generate_feedback_report"

    # Skipped questions must NOT cause off_topic_references to be populated
    assert report.off_topic_references == [], (
        f"Expected off_topic_references to be empty when no responses are off-topic "
        f"(off_topic_count=0), but got: {report.off_topic_references!r}"
    )

    # Structural invariants: at least one entry in each list
    assert len(report.strengths) >= 1, (
        f"Expected at least one strength in FeedbackReport, got {report.strengths!r}"
    )
    assert len(report.improvements) >= 1, (
        f"Expected at least one improvement in FeedbackReport, got {report.improvements!r}"
    )
    assert len(report.actionable_recommendations) >= 1, (
        f"Expected at least one actionable recommendation in FeedbackReport, "
        f"got {report.actionable_recommendations!r}"
    )

    # All 4 EvaluationDimension values must be present
    present_dimensions = {ds.dimension for ds in report.dimension_scores}
    for dim in EvaluationDimension:
        assert dim in present_dimensions, (
            f"Expected EvaluationDimension.{dim.name} to be present in dimension_scores, "
            f"but only found: {[d.name for d in present_dimensions]}"
        )


# ---------------------------------------------------------------------------
# Property 14d: Skipped question texts do not appear in off_topic_references
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_feedback_session_strategy())
@settings(max_examples=100)
async def test_property_14d_skipped_question_texts_not_in_off_topic_references(
    session: SessionState,
) -> None:
    """Property 14d: Skipped questions do not appear in off_topic_references.

    For any session with skipped questions and ``off_topic_count=0``, the text
    of skipped questions must NOT appear in ``off_topic_references`` of the
    generated ``FeedbackReport``.

    This is a direct check that the skip mechanism is not confused with
    off-topic detection.

    **Validates: Requirements 6.4**
    """
    service = _make_feedback_service()

    # Collect the texts of all skipped questions
    skipped_question_texts = [q.text for q in session.questions if q.skipped]

    # Preconditions
    assert session.off_topic_count == 0, (
        "Strategy should guarantee off_topic_count=0"
    )
    assert all(not r.is_off_topic for r in session.responses), (
        "Strategy should guarantee all responses have is_off_topic=False"
    )

    _reply, updated_session = await service.generate_feedback_report(session)

    report = updated_session.feedback_report
    assert report is not None, "Expected feedback_report to be populated after generate_feedback_report"

    # off_topic_references must be empty (no off-topic responses)
    assert report.off_topic_references == [], (
        f"Expected off_topic_references to be empty when off_topic_count=0, "
        f"but got: {report.off_topic_references!r}"
    )

    # None of the skipped question texts should appear in off_topic_references
    for skipped_text in skipped_question_texts:
        assert skipped_text not in report.off_topic_references, (
            f"Skipped question text {skipped_text!r} must not appear in "
            f"off_topic_references, but it was found: {report.off_topic_references!r}"
        )
