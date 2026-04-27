# Feature: interview-practice-partner, Property 15: High Off-Topic Count Surfaces in Feedback
"""Property-based tests for high off-topic count surfacing in feedback.

When a session has ``off_topic_count > 2``, the generated ``FeedbackReport``
must reference focus/relevance in either ``improvements`` or
``actionable_recommendations``.

This property must hold regardless of what the LLM returns — including valid
JSON with no focus mention, partial JSON, invalid JSON, and LLM exceptions.

Validates: Requirements 7.2, 9.4
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
from interview_practice_partner.services.feedback import (
    FeedbackService,
    _FOCUS_KEYWORDS,
)

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


def _user_response_strategy(*, is_off_topic: bool | None = None) -> st.SearchStrategy[UserResponse]:
    """Build a UserResponse strategy.

    If ``is_off_topic`` is provided, all generated responses will have that
    value; otherwise it is drawn randomly.
    """
    off_topic_st = (
        st.just(is_off_topic) if is_off_topic is not None else st.booleans()
    )
    return st.builds(
        UserResponse,
        response_id=_uuid_strategy,
        question_id=_uuid_strategy,
        text=_nonempty_text,
        word_count=st.integers(min_value=1, max_value=300),
        is_off_topic=off_topic_st,
        received_at=_datetime_strategy,
    )


def _high_off_topic_session_strategy() -> st.SearchStrategy[SessionState]:
    """Build a completed SessionState with ``off_topic_count > 2``.

    Generates sessions with varying numbers of questions and responses,
    varying roles, and off_topic_count drawn from [3, 10] to exercise the
    full range of inputs that trigger the focus/relevance note.
    """
    return st.builds(
        SessionState,
        session_id=_uuid_strategy,
        phone_number=_phone_strategy,
        stage=st.just(Stage.COMPLETE),
        role=st.sampled_from([r for r in Role if r != Role.UNKNOWN]),
        questions=st.lists(_question_strategy(), min_size=1, max_size=10),
        responses=st.lists(_user_response_strategy(), min_size=0, max_size=10),
        off_topic_count=st.integers(min_value=3, max_value=10),
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


def _make_feedback_json_no_focus(
    improvements: list[str] | None = None,
    actionable_recommendations: list[str] | None = None,
) -> str:
    """Build a valid LLM feedback JSON response that does NOT mention focus/relevance.

    This simulates an LLM that failed to include a focus/relevance note despite
    the session having a high off-topic count — the service must add it.
    """
    return json.dumps({
        "dimension_scores": _full_dimension_scores_json(),
        "strengths": ["Clear communication throughout the session."],
        "improvements": improvements or ["Work on providing more specific examples."],
        "actionable_recommendations": actionable_recommendations or [
            "Use the STAR method to structure your answers."
        ],
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


def _contains_focus_reference(texts: list[str]) -> bool:
    """Return True if any string in *texts* contains a focus/relevance keyword."""
    combined = " ".join(texts).lower()
    return any(kw in combined for kw in _FOCUS_KEYWORDS)


def _assert_focus_reference_present(report, *, context: str = "") -> None:
    """Assert that the FeedbackReport references focus/relevance."""
    prefix = f"{context} — " if context else ""

    assert report is not None, (
        f"{prefix}Expected feedback_report to be populated after generate_feedback_report"
    )

    has_focus = _contains_focus_reference(report.improvements) or _contains_focus_reference(
        report.actionable_recommendations
    )

    assert has_focus, (
        f"{prefix}Expected improvements or actionable_recommendations to reference "
        f"focus/relevance (one of: {sorted(_FOCUS_KEYWORDS)!r}), but got:\n"
        f"  improvements={report.improvements!r}\n"
        f"  actionable_recommendations={report.actionable_recommendations!r}"
    )


# ---------------------------------------------------------------------------
# Property 15a: Focus/relevance note present when LLM omits it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_high_off_topic_session_strategy())
@settings(max_examples=100)
async def test_property_15a_focus_note_added_when_llm_omits_it(
    session: SessionState,
) -> None:
    """Property 15a: Focus/relevance note is added when LLM omits it.

    For any session with ``off_topic_count > 2``, when the LLM returns a
    response that does not mention focus or relevance, the ``FeedbackService``
    must add a focus/relevance note to ``improvements`` or
    ``actionable_recommendations``.

    **Validates: Requirements 7.2, 9.4**
    """
    service = _make_feedback_service(
        llm_response=_make_feedback_json_no_focus()
    )

    _reply, updated_session = await service.generate_feedback_report(session)

    _assert_focus_reference_present(
        updated_session.feedback_report,
        context=(
            f"LLM omitted focus note, off_topic_count={session.off_topic_count}, "
            f"role={session.role!r}, questions={len(session.questions)}, "
            f"responses={len(session.responses)}"
        ),
    )


# ---------------------------------------------------------------------------
# Property 15b: Focus/relevance note present when LLM already includes it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_high_off_topic_session_strategy())
@settings(max_examples=100)
async def test_property_15b_focus_note_preserved_when_llm_includes_it(
    session: SessionState,
) -> None:
    """Property 15b: Focus/relevance note is preserved when LLM already includes it.

    For any session with ``off_topic_count > 2``, when the LLM already includes
    a focus/relevance reference in its response, the ``FeedbackReport`` must
    still contain that reference (the service must not strip it).

    **Validates: Requirements 7.2, 9.4**
    """
    service = _make_feedback_service(
        llm_response=_make_feedback_json_no_focus(
            improvements=["Work on staying focused and relevant to each question."]
        )
    )

    _reply, updated_session = await service.generate_feedback_report(session)

    _assert_focus_reference_present(
        updated_session.feedback_report,
        context=(
            f"LLM included focus note, off_topic_count={session.off_topic_count}, "
            f"role={session.role!r}"
        ),
    )


# ---------------------------------------------------------------------------
# Property 15c: Focus/relevance note present when LLM returns invalid JSON
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_high_off_topic_session_strategy())
@settings(max_examples=100)
async def test_property_15c_focus_note_present_on_invalid_llm_json(
    session: SessionState,
) -> None:
    """Property 15c: Focus/relevance note is present even when LLM returns invalid JSON.

    For any session with ``off_topic_count > 2``, when the LLM returns invalid
    JSON (triggering the fallback path), the ``FeedbackService`` must still
    ensure that ``improvements`` or ``actionable_recommendations`` references
    focus/relevance.

    **Validates: Requirements 7.2, 9.4**
    """
    service = _make_feedback_service(llm_response="This is not valid JSON at all!")

    _reply, updated_session = await service.generate_feedback_report(session)

    _assert_focus_reference_present(
        updated_session.feedback_report,
        context=(
            f"invalid LLM JSON (fallback path), off_topic_count={session.off_topic_count}, "
            f"role={session.role!r}"
        ),
    )


# ---------------------------------------------------------------------------
# Property 15d: Focus/relevance note present when LLM raises an exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_high_off_topic_session_strategy())
@settings(max_examples=100)
async def test_property_15d_focus_note_present_on_llm_exception(
    session: SessionState,
) -> None:
    """Property 15d: Focus/relevance note is present even when LLM raises an exception.

    For any session with ``off_topic_count > 2``, when the LLM call raises an
    exception (triggering the fallback path), the ``FeedbackService`` must still
    ensure that ``improvements`` or ``actionable_recommendations`` references
    focus/relevance.

    **Validates: Requirements 7.2, 9.4**
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

    _assert_focus_reference_present(
        updated_session.feedback_report,
        context=(
            f"LLM exception (fallback path), off_topic_count={session.off_topic_count}, "
            f"role={session.role!r}"
        ),
    )


# ---------------------------------------------------------------------------
# Property 15e: Focus/relevance note present across varying off-topic counts > 2
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    session=_high_off_topic_session_strategy(),
    num_improvements=st.integers(min_value=0, max_value=5),
    num_recommendations=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=100)
async def test_property_15e_focus_note_present_across_varying_list_sizes(
    session: SessionState,
    num_improvements: int,
    num_recommendations: int,
) -> None:
    """Property 15e: Focus/relevance note is present for any list size from LLM.

    For any session with ``off_topic_count > 2`` and any combination of
    improvements/recommendations list sizes returned by the LLM (including
    zero), the generated ``FeedbackReport`` must always reference focus/relevance
    in ``improvements`` or ``actionable_recommendations``.

    **Validates: Requirements 7.2, 9.4**
    """
    # Generate improvements and recommendations that do NOT mention focus/relevance
    improvements = [f"Improvement {i + 1}: work on depth." for i in range(num_improvements)]
    recommendations = [f"Recommendation {i + 1}: practise daily." for i in range(num_recommendations)]

    service = _make_feedback_service(
        llm_response=_make_feedback_json_no_focus(
            improvements=improvements,
            actionable_recommendations=recommendations,
        )
    )

    _reply, updated_session = await service.generate_feedback_report(session)

    _assert_focus_reference_present(
        updated_session.feedback_report,
        context=(
            f"LLM returned {num_improvements} improvements, {num_recommendations} recommendations; "
            f"off_topic_count={session.off_topic_count}, role={session.role!r}"
        ),
    )
