# Feature: interview-practice-partner, Property 21: No Question Is Repeated Within a Session
"""Property-based tests for question uniqueness within a session.

All ``question.text`` values in a session must be unique — the same question
must never be asked twice within a single session.

Validates: Requirements 11.4
"""

from __future__ import annotations

import itertools
from datetime import timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from interview_practice_partner.domain.enums import QuestionType, Role, Stage
from interview_practice_partner.domain.models import Question, SessionState
from interview_practice_partner.services.interview import InterviewService

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_nonempty_text = st.text(min_size=1, max_size=200)
_uuid_strategy = st.uuids().map(str)
_phone_strategy = st.from_regex(r"\+1[2-9]\d{9}", fullmatch=True)
_datetime_strategy = st.datetimes(timezones=st.just(timezone.utc))


def _question_strategy() -> st.SearchStrategy[Question]:
    """Build a Question strategy with unique-ish text."""
    return st.builds(
        Question,
        question_id=_uuid_strategy,
        text=_nonempty_text,
        question_type=st.sampled_from(list(QuestionType)),
        asked_at=_datetime_strategy,
        skipped=st.just(False),
    )


def _session_with_unique_questions(
    min_questions: int = 0,
    max_questions: int = 8,
) -> st.SearchStrategy[SessionState]:
    """Build an INTERVIEW SessionState whose existing questions all have unique texts."""
    return st.builds(
        SessionState,
        session_id=_uuid_strategy,
        phone_number=_phone_strategy,
        stage=st.just(Stage.INTERVIEW),
        role=st.sampled_from([r for r in Role if r != Role.UNKNOWN]),
        questions=st.lists(
            _question_strategy(),
            min_size=min_questions,
            max_size=max_questions,
        ).filter(
            lambda qs: len({q.text for q in qs}) == len(qs)  # all texts unique
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service_with_unique_questions(question_texts: list[str]) -> InterviewService:
    """Build an InterviewService whose mock LLM returns successive unique question texts.

    Each call to ``llm.complete`` returns the next text from ``question_texts``,
    cycling through them in order.  This simulates an LLM that always produces
    a fresh, distinct question.
    """
    call_counter = {"n": 0}

    async def _unique_complete(*_args, **_kwargs) -> str:
        text = question_texts[call_counter["n"] % len(question_texts)]
        call_counter["n"] += 1
        return text

    mock_llm = AsyncMock()
    mock_llm.complete.side_effect = _unique_complete

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


# ---------------------------------------------------------------------------
# Property 21a: Existing questions in a session all have unique texts
# ---------------------------------------------------------------------------


@given(session=_session_with_unique_questions(min_questions=0, max_questions=10))
@settings(max_examples=100)
def test_property_21a_existing_session_questions_are_unique(
    session: SessionState,
) -> None:
    """Property 21a: All question.text values in a session are unique.

    For any generated session whose questions were built with unique texts,
    the ``question.text`` values must all be distinct — no two questions share
    the same text.

    **Validates: Requirements 11.4**
    """
    texts = [q.text for q in session.questions]
    unique_texts = set(texts)

    assert len(unique_texts) == len(texts), (
        f"Expected all question texts to be unique, but found duplicates.\n"
        f"Total questions: {len(texts)}, unique texts: {len(unique_texts)}\n"
        f"Duplicate texts: {[t for t in texts if texts.count(t) > 1]!r}"
    )


# ---------------------------------------------------------------------------
# Property 21b: generate_question appends a question with a unique text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    session=_session_with_unique_questions(min_questions=0, max_questions=5),
    num_new_questions=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=100)
async def test_property_21b_generate_question_produces_unique_texts(
    session: SessionState,
    num_new_questions: int,
) -> None:
    """Property 21b: generate_question appends questions with unique texts.

    For any INTERVIEW session with 0–5 existing unique questions, calling
    ``generate_question`` 1–5 times with a mock LLM that returns distinct
    texts must result in a session where all ``question.text`` values remain
    unique.

    **Validates: Requirements 11.4**
    """
    # Build a pool of question texts that are guaranteed to be distinct from
    # each other AND from any existing question texts in the session.
    existing_texts = {q.text for q in session.questions}
    new_texts = [
        f"__unique_question_{i}__"
        for i in range(num_new_questions)
        if f"__unique_question_{i}__" not in existing_texts
    ]
    # Ensure we have enough unique texts (the generated texts use a prefix
    # that is extremely unlikely to collide with hypothesis-generated texts,
    # but we guard defensively).
    if len(new_texts) < num_new_questions:
        # Fall back to a larger index range to avoid collisions
        new_texts = [
            f"__unique_question_fallback_{i}__"
            for i in range(num_new_questions * 10)
            if f"__unique_question_fallback_{i}__" not in existing_texts
        ][:num_new_questions]

    service = _make_service_with_unique_questions(new_texts)

    for _ in range(num_new_questions):
        await service.generate_question(session)

    # All question texts in the session must be unique
    all_texts = [q.text for q in session.questions]
    unique_texts = set(all_texts)

    assert len(unique_texts) == len(all_texts), (
        f"Expected all question texts to be unique after {num_new_questions} "
        f"generate_question call(s), but found duplicates.\n"
        f"Total questions: {len(all_texts)}, unique texts: {len(unique_texts)}\n"
        f"Duplicate texts: {[t for t in all_texts if all_texts.count(t) > 1]!r}"
    )


# ---------------------------------------------------------------------------
# Property 21c: No two questions in a session share the same text (pairwise check)
# ---------------------------------------------------------------------------


@given(session=_session_with_unique_questions(min_questions=2, max_questions=10))
@settings(max_examples=100)
def test_property_21c_no_two_questions_share_same_text(
    session: SessionState,
) -> None:
    """Property 21c: No two questions in a session share the same text (pairwise).

    For any session with at least 2 questions, every pair of questions must
    have distinct ``text`` values.  This is a pairwise formulation of the
    uniqueness property.

    **Validates: Requirements 11.4**
    """
    for q1, q2 in itertools.combinations(session.questions, 2):
        assert q1.text != q2.text, (
            f"Found two questions with identical text in the same session.\n"
            f"Question 1 (id={q1.question_id!r}): {q1.text!r}\n"
            f"Question 2 (id={q2.question_id!r}): {q2.text!r}"
        )


# ---------------------------------------------------------------------------
# Property 21d: generate_question does not duplicate an existing question text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_session_with_unique_questions(min_questions=1, max_questions=5))
@settings(max_examples=100)
async def test_property_21d_generate_question_does_not_duplicate_existing_text(
    session: SessionState,
) -> None:
    """Property 21d: generate_question does not add a question with an existing text.

    For any INTERVIEW session with at least one existing question, calling
    ``generate_question`` once with a mock LLM that returns a text NOT already
    in the session must result in the new question having a unique text.

    **Validates: Requirements 11.4**
    """
    existing_texts = {q.text for q in session.questions}
    initial_count = len(session.questions)

    # The mock LLM returns a text that is guaranteed to be new
    new_text = "__brand_new_unique_question_text__"
    assert new_text not in existing_texts, (
        "Test setup error: the new text collided with an existing question text."
    )

    service = _make_service_with_unique_questions([new_text])
    await service.generate_question(session)

    # One new question must have been appended
    assert len(session.questions) == initial_count + 1, (
        f"Expected {initial_count + 1} questions after generate_question, "
        f"but got {len(session.questions)}"
    )

    # The new question's text must not duplicate any existing text
    new_question = session.questions[-1]
    assert new_question.text not in existing_texts, (
        f"generate_question appended a question whose text already existed in the session.\n"
        f"New question text: {new_question.text!r}\n"
        f"Existing texts: {existing_texts!r}"
    )

    # All texts in the session must still be unique
    all_texts = [q.text for q in session.questions]
    assert len(set(all_texts)) == len(all_texts), (
        f"Expected all question texts to remain unique after generate_question, "
        f"but found duplicates: {[t for t in all_texts if all_texts.count(t) > 1]!r}"
    )
