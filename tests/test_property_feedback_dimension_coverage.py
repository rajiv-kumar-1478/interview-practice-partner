# Feature: interview-practice-partner, Property 17: FeedbackReport Covers All Evaluation Dimensions
"""Property-based tests for FeedbackReport dimension coverage.

Every completed session's ``FeedbackReport`` must have a ``DimensionScore``
for each ``EvaluationDimension`` value (COMMUNICATION_CLARITY, RELEVANCE,
TECHNICAL_KNOWLEDGE, CONFIDENCE).

Validates: Requirements 9.1, 9.2
"""

from __future__ import annotations

import json
import uuid
from datetime import timezone
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
    Question,
    SessionState,
    UserResponse,
)
from interview_practice_partner.services.feedback import FeedbackService

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_uuid_strategy = st.uuids().map(str)
_phone_strategy = st.from_regex(r"\+1[2-9]\d{9}", fullmatch=True)
_datetime_strategy = st.datetimes(timezones=st.just(timezone.utc))
_nonempty_text = st.text(min_size=1, max_size=200)

_ALL_DIMENSIONS = list(EvaluationDimension)


def _question_strategy() -> st.SearchStrategy[Question]:
    """Build a Question strategy."""
    return st.builds(
        Question,
        question_id=_uuid_strategy,
        text=_nonempty_text,
        question_type=st.sampled_from(list(QuestionType)),
        asked_at=_datetime_strategy,
        skipped=st.booleans(),
    )


def _user_response_strategy() -> st.SearchStrategy[UserResponse]:
    """Build a UserResponse strategy."""
    return st.builds(
        UserResponse,
        response_id=_uuid_strategy,
        question_id=_uuid_strategy,
        text=_nonempty_text,
        word_count=st.integers(min_value=1, max_value=300),
        is_off_topic=st.booleans(),
        received_at=_datetime_strategy,
    )


def _completed_session_strategy() -> st.SearchStrategy[SessionState]:
    """Build a completed SessionState (stage=COMPLETE, is_complete=True).

    Generates sessions with varying numbers of questions and responses,
    varying roles, and varying off-topic counts to exercise the full
    range of inputs to ``generate_feedback_report``.
    """
    return st.builds(
        SessionState,
        session_id=_uuid_strategy,
        phone_number=_phone_strategy,
        stage=st.just(Stage.COMPLETE),
        role=st.sampled_from([r for r in Role if r != Role.UNKNOWN]),
        questions=st.lists(_question_strategy(), min_size=1, max_size=10),
        responses=st.lists(_user_response_strategy(), min_size=0, max_size=10),
        off_topic_count=st.integers(min_value=0, max_value=5),
        consecutive_out_of_scope_count=st.just(0),
        clarification_turn_count=st.just(0),
        requested_short_session=st.booleans(),
        is_complete=st.just(True),
        created_at=_datetime_strategy,
        updated_at=_datetime_strategy,
        completed_at=_datetime_strategy,
        feedback_report=st.none(),
        context_summary=st.none(),
    )


# ---------------------------------------------------------------------------
# LLM response strategies — vary what the LLM returns to test robustness
# ---------------------------------------------------------------------------

def _full_dimension_scores_json() -> list[dict]:
    """Return dimension score dicts covering all four dimensions."""
    return [
        {
            "dimension": dim.value,
            "qualitative_assessment": f"Assessment for {dim.value}.",
            "score": 4,
        }
        for dim in EvaluationDimension
    ]


def _partial_dimension_scores_json(num_dimensions: int) -> list[dict]:
    """Return dimension score dicts covering only the first ``num_dimensions`` dimensions."""
    return [
        {
            "dimension": dim.value,
            "qualitative_assessment": f"Assessment for {dim.value}.",
            "score": 3,
        }
        for dim in list(EvaluationDimension)[:num_dimensions]
    ]


def _make_feedback_json(dimension_scores: list[dict]) -> str:
    """Build a valid LLM feedback JSON response with the given dimension scores."""
    return json.dumps({
        "dimension_scores": dimension_scores,
        "strengths": ["Clear communication throughout the session."],
        "improvements": ["Work on providing more specific examples."],
        "actionable_recommendations": ["Use the STAR method to structure your answers."],
        "off_topic_references": [],
    })


def _make_feedback_service(llm_response: str) -> FeedbackService:
    """Build a FeedbackService with a mocked LLM returning ``llm_response``."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = llm_response

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_feedback_prompt.return_value = [
        {"role": "system", "content": "You are a feedback generator."},
        {"role": "user", "content": "Generate feedback for this session."},
    ]

    return FeedbackService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
    )


# ---------------------------------------------------------------------------
# Property 17a: All four dimensions present when LLM returns all four
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_completed_session_strategy())
@settings(max_examples=100)
async def test_property_17a_all_dimensions_present_when_llm_returns_all(
    session: SessionState,
) -> None:
    """Property 17a: FeedbackReport covers all EvaluationDimension values.

    For any completed session, when the LLM returns all four dimension scores,
    the generated ``FeedbackReport`` must have a ``DimensionScore`` for each
    ``EvaluationDimension`` value.

    **Validates: Requirements 9.1, 9.2**
    """
    service = _make_feedback_service(
        llm_response=_make_feedback_json(_full_dimension_scores_json())
    )

    _reply, updated_session = await service.generate_feedback_report(session)

    report = updated_session.feedback_report
    assert report is not None, (
        "Expected feedback_report to be populated after generate_feedback_report"
    )

    present_dimensions = {ds.dimension for ds in report.dimension_scores}

    for dim in EvaluationDimension:
        assert dim in present_dimensions, (
            f"Expected EvaluationDimension.{dim.name} to be present in dimension_scores, "
            f"but only found: {[d.name for d in present_dimensions]!r}. "
            f"Session role: {session.role!r}, "
            f"question count: {len(session.questions)}, "
            f"response count: {len(session.responses)}"
        )

    assert present_dimensions == set(EvaluationDimension), (
        f"Expected exactly the four EvaluationDimension values in dimension_scores, "
        f"but found: {[d.name for d in present_dimensions]!r}"
    )


# ---------------------------------------------------------------------------
# Property 17b: All four dimensions present when LLM returns only a subset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    session=_completed_session_strategy(),
    num_llm_dimensions=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=100)
async def test_property_17b_all_dimensions_present_when_llm_returns_partial(
    session: SessionState,
    num_llm_dimensions: int,
) -> None:
    """Property 17b: FeedbackReport covers all dimensions even when LLM omits some.

    For any completed session, even when the LLM returns fewer than four
    dimension scores (0–3), the ``FeedbackService`` must fill in the missing
    dimensions so that the final ``FeedbackReport`` always has a ``DimensionScore``
    for each ``EvaluationDimension`` value.

    **Validates: Requirements 9.1, 9.2**
    """
    service = _make_feedback_service(
        llm_response=_make_feedback_json(
            _partial_dimension_scores_json(num_llm_dimensions)
        )
    )

    _reply, updated_session = await service.generate_feedback_report(session)

    report = updated_session.feedback_report
    assert report is not None, (
        "Expected feedback_report to be populated after generate_feedback_report"
    )

    present_dimensions = {ds.dimension for ds in report.dimension_scores}

    for dim in EvaluationDimension:
        assert dim in present_dimensions, (
            f"Expected EvaluationDimension.{dim.name} to be present in dimension_scores "
            f"even though LLM only returned {num_llm_dimensions} dimension(s). "
            f"Present dimensions: {[d.name for d in present_dimensions]!r}. "
            f"Session role: {session.role!r}, "
            f"question count: {len(session.questions)}"
        )


# ---------------------------------------------------------------------------
# Property 17c: All four dimensions present when LLM returns invalid JSON
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_completed_session_strategy())
@settings(max_examples=100)
async def test_property_17c_all_dimensions_present_on_invalid_llm_response(
    session: SessionState,
) -> None:
    """Property 17c: FeedbackReport covers all dimensions even on LLM failure.

    For any completed session, when the LLM returns invalid JSON (triggering
    the fallback path), the generated ``FeedbackReport`` must still have a
    ``DimensionScore`` for each ``EvaluationDimension`` value.

    **Validates: Requirements 9.1, 9.2**
    """
    service = _make_feedback_service(llm_response="This is not valid JSON at all!")

    _reply, updated_session = await service.generate_feedback_report(session)

    report = updated_session.feedback_report
    assert report is not None, (
        "Expected feedback_report to be populated even when LLM returns invalid JSON"
    )

    present_dimensions = {ds.dimension for ds in report.dimension_scores}

    for dim in EvaluationDimension:
        assert dim in present_dimensions, (
            f"Expected EvaluationDimension.{dim.name} to be present in dimension_scores "
            f"even on LLM failure (fallback path). "
            f"Present dimensions: {[d.name for d in present_dimensions]!r}. "
            f"Session role: {session.role!r}"
        )

    assert present_dimensions == set(EvaluationDimension), (
        f"Expected exactly the four EvaluationDimension values in fallback dimension_scores, "
        f"but found: {[d.name for d in present_dimensions]!r}"
    )


# ---------------------------------------------------------------------------
# Property 17d: All four dimensions present when LLM raises an exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_completed_session_strategy())
@settings(max_examples=100)
async def test_property_17d_all_dimensions_present_on_llm_exception(
    session: SessionState,
) -> None:
    """Property 17d: FeedbackReport covers all dimensions even when LLM raises an exception.

    For any completed session, when the LLM call raises an exception (triggering
    the fallback path), the generated ``FeedbackReport`` must still have a
    ``DimensionScore`` for each ``EvaluationDimension`` value.

    **Validates: Requirements 9.1, 9.2**
    """
    mock_llm = AsyncMock()
    mock_llm.complete.side_effect = RuntimeError("LLM service unavailable")

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_feedback_prompt.return_value = [
        {"role": "system", "content": "You are a feedback generator."},
        {"role": "user", "content": "Generate feedback."},
    ]

    service = FeedbackService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
    )

    _reply, updated_session = await service.generate_feedback_report(session)

    report = updated_session.feedback_report
    assert report is not None, (
        "Expected feedback_report to be populated even when LLM raises an exception"
    )

    present_dimensions = {ds.dimension for ds in report.dimension_scores}

    for dim in EvaluationDimension:
        assert dim in present_dimensions, (
            f"Expected EvaluationDimension.{dim.name} to be present in dimension_scores "
            f"even when LLM raises an exception. "
            f"Present dimensions: {[d.name for d in present_dimensions]!r}. "
            f"Session role: {session.role!r}"
        )

    assert present_dimensions == set(EvaluationDimension), (
        f"Expected exactly the four EvaluationDimension values in dimension_scores "
        f"after LLM exception, but found: {[d.name for d in present_dimensions]!r}"
    )


# ---------------------------------------------------------------------------
# Property 17e: Each DimensionScore has a non-empty qualitative_assessment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_completed_session_strategy())
@settings(max_examples=100)
async def test_property_17e_each_dimension_score_has_non_empty_assessment(
    session: SessionState,
) -> None:
    """Property 17e: Each DimensionScore has a non-empty qualitative_assessment.

    For any completed session, every ``DimensionScore`` in the generated
    ``FeedbackReport`` must have a non-empty ``qualitative_assessment`` string.

    **Validates: Requirements 9.1, 9.2**
    """
    service = _make_feedback_service(
        llm_response=_make_feedback_json(_full_dimension_scores_json())
    )

    _reply, updated_session = await service.generate_feedback_report(session)

    report = updated_session.feedback_report
    assert report is not None, (
        "Expected feedback_report to be populated after generate_feedback_report"
    )

    for ds in report.dimension_scores:
        assert isinstance(ds.qualitative_assessment, str), (
            f"Expected qualitative_assessment to be a str for dimension "
            f"{ds.dimension.name}, got {type(ds.qualitative_assessment)!r}"
        )
        assert len(ds.qualitative_assessment.strip()) > 0, (
            f"Expected non-empty qualitative_assessment for dimension "
            f"{ds.dimension.name}, but got: {ds.qualitative_assessment!r}"
        )


# ---------------------------------------------------------------------------
# Property 17f: Each DimensionScore has a valid score in range [1, 5]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_completed_session_strategy())
@settings(max_examples=100)
async def test_property_17f_each_dimension_score_is_in_valid_range(
    session: SessionState,
) -> None:
    """Property 17f: Each DimensionScore has a score in the valid range [1, 5].

    For any completed session, every ``DimensionScore`` in the generated
    ``FeedbackReport`` must have a ``score`` value between 1 and 5 inclusive.

    **Validates: Requirements 9.1, 9.2**
    """
    service = _make_feedback_service(
        llm_response=_make_feedback_json(_full_dimension_scores_json())
    )

    _reply, updated_session = await service.generate_feedback_report(session)

    report = updated_session.feedback_report
    assert report is not None, (
        "Expected feedback_report to be populated after generate_feedback_report"
    )

    for ds in report.dimension_scores:
        assert 1 <= ds.score <= 5, (
            f"Expected score for dimension {ds.dimension.name} to be in [1, 5], "
            f"but got: {ds.score}"
        )
