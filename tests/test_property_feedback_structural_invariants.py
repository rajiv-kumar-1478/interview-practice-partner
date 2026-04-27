# Feature: interview-practice-partner, Property 18: FeedbackReport Structural Invariants
"""Property-based tests for FeedbackReport structural invariants.

Every completed session's ``FeedbackReport`` must have:
- At least one entry in ``strengths``
- At least one entry in ``improvements``
- At least one entry in ``actionable_recommendations``

These invariants must hold regardless of what the LLM returns — including
valid JSON, partial JSON, empty lists, invalid JSON, and LLM exceptions.

Validates: Requirements 9.3, 9.5
"""

from __future__ import annotations

import json
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
# LLM response helpers
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


def _make_feedback_json(
    strengths: list[str] | None = None,
    improvements: list[str] | None = None,
    actionable_recommendations: list[str] | None = None,
) -> str:
    """Build a valid LLM feedback JSON response with the given list fields."""
    return json.dumps({
        "dimension_scores": _full_dimension_scores_json(),
        "strengths": strengths if strengths is not None else ["Clear communication."],
        "improvements": improvements if improvements is not None else ["Work on depth."],
        "actionable_recommendations": (
            actionable_recommendations
            if actionable_recommendations is not None
            else ["Use the STAR method."]
        ),
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


def _assert_structural_invariants(report, *, context: str = "") -> None:
    """Assert that the FeedbackReport satisfies all structural invariants."""
    prefix = f"{context} — " if context else ""

    assert report is not None, (
        f"{prefix}Expected feedback_report to be populated after generate_feedback_report"
    )

    assert len(report.strengths) >= 1, (
        f"{prefix}Expected at least one entry in strengths, "
        f"but got {len(report.strengths)} entries: {report.strengths!r}"
    )

    assert len(report.improvements) >= 1, (
        f"{prefix}Expected at least one entry in improvements, "
        f"but got {len(report.improvements)} entries: {report.improvements!r}"
    )

    assert len(report.actionable_recommendations) >= 1, (
        f"{prefix}Expected at least one entry in actionable_recommendations, "
        f"but got {len(report.actionable_recommendations)} entries: "
        f"{report.actionable_recommendations!r}"
    )

    # Each entry must be a non-empty string
    for strength in report.strengths:
        assert isinstance(strength, str) and len(strength.strip()) > 0, (
            f"{prefix}Each strength must be a non-empty string, got: {strength!r}"
        )

    for improvement in report.improvements:
        assert isinstance(improvement, str) and len(improvement.strip()) > 0, (
            f"{prefix}Each improvement must be a non-empty string, got: {improvement!r}"
        )

    for rec in report.actionable_recommendations:
        assert isinstance(rec, str) and len(rec.strip()) > 0, (
            f"{prefix}Each actionable_recommendation must be a non-empty string, "
            f"got: {rec!r}"
        )


# ---------------------------------------------------------------------------
# Property 18a: Structural invariants hold when LLM returns valid full lists
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_completed_session_strategy())
@settings(max_examples=100)
async def test_property_18a_structural_invariants_with_valid_llm_response(
    session: SessionState,
) -> None:
    """Property 18a: FeedbackReport structural invariants hold with a valid LLM response.

    For any completed session, when the LLM returns a well-formed JSON response
    with non-empty strengths, improvements, and actionable_recommendations, the
    generated ``FeedbackReport`` must have at least one entry in each of those
    three lists.

    **Validates: Requirements 9.3, 9.5**
    """
    service = _make_feedback_service(llm_response=_make_feedback_json())

    _reply, updated_session = await service.generate_feedback_report(session)

    _assert_structural_invariants(
        updated_session.feedback_report,
        context=(
            f"valid LLM response, role={session.role!r}, "
            f"questions={len(session.questions)}, responses={len(session.responses)}"
        ),
    )


# ---------------------------------------------------------------------------
# Property 18b: Structural invariants hold when LLM returns empty lists
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_completed_session_strategy())
@settings(max_examples=100)
async def test_property_18b_structural_invariants_when_llm_returns_empty_lists(
    session: SessionState,
) -> None:
    """Property 18b: FeedbackReport structural invariants hold when LLM returns empty lists.

    For any completed session, when the LLM returns a JSON response with empty
    ``strengths``, ``improvements``, and ``actionable_recommendations`` lists,
    the ``FeedbackService`` must fill in safe defaults so that the final
    ``FeedbackReport`` still has at least one entry in each list.

    **Validates: Requirements 9.3, 9.5**
    """
    service = _make_feedback_service(
        llm_response=_make_feedback_json(
            strengths=[],
            improvements=[],
            actionable_recommendations=[],
        )
    )

    _reply, updated_session = await service.generate_feedback_report(session)

    _assert_structural_invariants(
        updated_session.feedback_report,
        context=(
            f"LLM returned empty lists, role={session.role!r}, "
            f"questions={len(session.questions)}, responses={len(session.responses)}"
        ),
    )


# ---------------------------------------------------------------------------
# Property 18c: Structural invariants hold when LLM returns null/missing fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_completed_session_strategy())
@settings(max_examples=100)
async def test_property_18c_structural_invariants_when_llm_omits_list_fields(
    session: SessionState,
) -> None:
    """Property 18c: FeedbackReport structural invariants hold when LLM omits list fields.

    For any completed session, when the LLM returns a JSON response that omits
    ``strengths``, ``improvements``, and ``actionable_recommendations`` entirely,
    the ``FeedbackService`` must fill in safe defaults so that the final
    ``FeedbackReport`` still has at least one entry in each list.

    **Validates: Requirements 9.3, 9.5**
    """
    # JSON with no strengths/improvements/recommendations keys at all
    minimal_json = json.dumps({
        "dimension_scores": _full_dimension_scores_json(),
    })
    service = _make_feedback_service(llm_response=minimal_json)

    _reply, updated_session = await service.generate_feedback_report(session)

    _assert_structural_invariants(
        updated_session.feedback_report,
        context=(
            f"LLM omitted list fields, role={session.role!r}, "
            f"questions={len(session.questions)}"
        ),
    )


# ---------------------------------------------------------------------------
# Property 18d: Structural invariants hold when LLM returns invalid JSON
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_completed_session_strategy())
@settings(max_examples=100)
async def test_property_18d_structural_invariants_on_invalid_llm_json(
    session: SessionState,
) -> None:
    """Property 18d: FeedbackReport structural invariants hold when LLM returns invalid JSON.

    For any completed session, when the LLM returns a response that cannot be
    parsed as JSON (triggering the fallback path), the generated ``FeedbackReport``
    must still have at least one entry in each of ``strengths``, ``improvements``,
    and ``actionable_recommendations``.

    **Validates: Requirements 9.3, 9.5**
    """
    service = _make_feedback_service(llm_response="This is not valid JSON at all!")

    _reply, updated_session = await service.generate_feedback_report(session)

    _assert_structural_invariants(
        updated_session.feedback_report,
        context=(
            f"invalid LLM JSON (fallback path), role={session.role!r}, "
            f"questions={len(session.questions)}"
        ),
    )


# ---------------------------------------------------------------------------
# Property 18e: Structural invariants hold when LLM raises an exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_completed_session_strategy())
@settings(max_examples=100)
async def test_property_18e_structural_invariants_on_llm_exception(
    session: SessionState,
) -> None:
    """Property 18e: FeedbackReport structural invariants hold when LLM raises an exception.

    For any completed session, when the LLM call raises an exception (triggering
    the fallback path), the generated ``FeedbackReport`` must still have at least
    one entry in each of ``strengths``, ``improvements``, and
    ``actionable_recommendations``.

    **Validates: Requirements 9.3, 9.5**
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

    _assert_structural_invariants(
        updated_session.feedback_report,
        context=(
            f"LLM exception (fallback path), role={session.role!r}, "
            f"questions={len(session.questions)}"
        ),
    )


# ---------------------------------------------------------------------------
# Property 18f: Structural invariants hold across varying LLM response shapes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    session=_completed_session_strategy(),
    num_strengths=st.integers(min_value=0, max_value=5),
    num_improvements=st.integers(min_value=0, max_value=5),
    num_recommendations=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=100)
async def test_property_18f_structural_invariants_across_varying_list_sizes(
    session: SessionState,
    num_strengths: int,
    num_improvements: int,
    num_recommendations: int,
) -> None:
    """Property 18f: FeedbackReport structural invariants hold for any list size from LLM.

    For any completed session and any combination of list sizes returned by the
    LLM (including zero), the generated ``FeedbackReport`` must always have at
    least one entry in each of ``strengths``, ``improvements``, and
    ``actionable_recommendations``.

    **Validates: Requirements 9.3, 9.5**
    """
    strengths = [f"Strength {i + 1}." for i in range(num_strengths)]
    improvements = [f"Improvement {i + 1}." for i in range(num_improvements)]
    recommendations = [f"Recommendation {i + 1}." for i in range(num_recommendations)]

    service = _make_feedback_service(
        llm_response=_make_feedback_json(
            strengths=strengths,
            improvements=improvements,
            actionable_recommendations=recommendations,
        )
    )

    _reply, updated_session = await service.generate_feedback_report(session)

    _assert_structural_invariants(
        updated_session.feedback_report,
        context=(
            f"LLM returned {num_strengths} strengths, {num_improvements} improvements, "
            f"{num_recommendations} recommendations; role={session.role!r}"
        ),
    )
