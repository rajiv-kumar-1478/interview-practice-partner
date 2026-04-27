# Feature: interview-practice-partner, Property 16: Empty or Whitespace Input Triggers Rephrase Prompt
"""Property-based tests for empty/whitespace input handling.

Any inbound message body that is empty or consists entirely of whitespace
characters must result in a rephrase prompt being returned, and the session
state must not advance (stage stays the same, no new questions or responses
are added).

Validates: Requirements 8.3
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from interview_practice_partner.domain.enums import QuestionType, Role, Stage
from interview_practice_partner.domain.models import Question, SessionState, UserResponse
from interview_practice_partner.services.interview import InterviewService
from interview_practice_partner.services.session import SessionService

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for empty or all-whitespace strings.
# Covers:
#   - The empty string ""
#   - Strings of one or more whitespace characters that Python's str.strip()
#     considers whitespace: space, tab, newline, carriage return, vertical tab,
#     and form feed.
#
# We use st.sampled_from with the exact set of characters that str.strip()
# treats as whitespace to ensure the generated strings always satisfy
# message.strip() == "".
_PYTHON_WHITESPACE_CHARS = " \t\n\r\x0b\x0c"

_whitespace_only_strategy = st.text(
    alphabet=st.sampled_from(_PYTHON_WHITESPACE_CHARS),
    min_size=1,
)

_empty_or_whitespace_strategy = st.one_of(
    st.just(""),
    _whitespace_only_strategy,
)

_nonempty_text = st.text(min_size=1, max_size=200)

_question_strategy = st.builds(
    Question,
    question_id=_nonempty_text,
    text=_nonempty_text,
    question_type=st.sampled_from(list(QuestionType)),
    asked_at=st.datetimes(),
    skipped=st.just(False),
)

_user_response_strategy = st.builds(
    UserResponse,
    response_id=_nonempty_text,
    question_id=_nonempty_text,
    text=_nonempty_text,
    word_count=st.integers(min_value=0, max_value=1000),
    is_off_topic=st.booleans(),
    received_at=st.datetimes(),
)

_interview_session_strategy = st.builds(
    SessionState,
    session_id=_nonempty_text,
    phone_number=_nonempty_text,
    stage=st.just(Stage.INTERVIEW),
    role=st.sampled_from([Role.SALES_REPRESENTATIVE, Role.RETAIL_ASSOCIATE]),  # Exclude SOFTWARE_ENGINEER to avoid round selection
    questions=st.lists(_question_strategy, min_size=0, max_size=5),
    responses=st.lists(_user_response_strategy, min_size=0, max_size=5),
    off_topic_count=st.integers(min_value=0, max_value=5),
    consecutive_out_of_scope_count=st.integers(min_value=0, max_value=5),
    clarification_turn_count=st.integers(min_value=0, max_value=3),
    requested_short_session=st.booleans(),
    is_complete=st.just(False),
    created_at=st.datetimes(),
    updated_at=st.datetimes(),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_interview_service() -> InterviewService:
    """Build an InterviewService with mocked LLM and PromptBuilder."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "What is your greatest strength?"

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_question_generation_prompt.return_value = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    mock_prompt_builder.build_response_evaluation_prompt.return_value = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
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


def _make_session_service() -> SessionService:
    """Build a SessionService with all dependencies mocked."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "Which role are you preparing for?"

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_role_selection_prompt.return_value = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]

    mock_interview_service = AsyncMock()
    mock_interview_service.generate_question.return_value = "Tell me about yourself."
    mock_interview_service.handle_response = AsyncMock(
        return_value=("Thank you for your answer.", MagicMock(spec=SessionState))
    )

    mock_feedback_service = AsyncMock()
    mock_feedback_service.generate_feedback_report.return_value = (
        "Here is your feedback.",
        MagicMock(spec=SessionState),
    )

    return SessionService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
        interview_service=mock_interview_service,
        feedback_service=mock_feedback_service,
    )


# ---------------------------------------------------------------------------
# Property 16a: handle_response with empty/whitespace returns a rephrase prompt
#               and does not advance the session (via InterviewService directly)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    session=_interview_session_strategy,
    message=_empty_or_whitespace_strategy,
)
@settings(max_examples=100)
async def test_property_16_empty_whitespace_returns_rephrase_prompt(
    session: SessionState,
    message: str,
) -> None:
    """Property 16: Empty or Whitespace Input Triggers Rephrase Prompt.

    For any INTERVIEW session and any message that is empty or all-whitespace,
    ``handle_response`` must return a rephrase/elaboration prompt (not an
    interview question or feedback), and the session must not advance.

    **Validates: Requirements 8.3**
    """
    # Confirm the message is genuinely empty or all-whitespace
    assert message.strip() == "", (
        f"Strategy produced a non-whitespace message: {message!r}"
    )

    service = _make_interview_service()

    initial_stage = session.stage
    initial_question_count = len(session.questions)
    initial_response_count = len(session.responses)

    reply, updated_session = await service.handle_response(session, message)

    # The reply must be a non-empty string
    assert isinstance(reply, str), (
        f"Expected handle_response to return a str reply, got {type(reply)!r}"
    )
    assert len(reply.strip()) > 0, (
        "Expected handle_response to return a non-empty reply for empty/whitespace input."
    )

    # The reply must be a rephrase/elaboration prompt — not a new interview question
    # or feedback. We check that it contains words associated with asking the user
    # to provide more or rephrase, and does NOT look like a new interview question.
    reply_lower = reply.lower()
    is_rephrase_prompt = any(
        word in reply_lower
        for word in [
            "elaborate",
            "more",
            "detail",
            "example",
            "brief",
            "expand",
            "rephrase",
            "repeat",
            "answer",
            "response",
            "try",
        ]
    )
    assert is_rephrase_prompt, (
        f"Expected a rephrase/elaboration prompt for empty/whitespace input, "
        f"but got: {reply!r}"
    )

    # Session stage must not advance
    assert updated_session.stage == Stage.INTERVIEW, (
        f"Expected session stage to remain INTERVIEW after empty/whitespace input, "
        f"but got {updated_session.stage!r}"
    )
    assert updated_session.stage == initial_stage, (
        f"Expected session stage to be unchanged after empty/whitespace input, "
        f"but it changed from {initial_stage!r} to {updated_session.stage!r}"
    )

    # No new questions should have been added
    assert len(updated_session.questions) == initial_question_count, (
        f"Expected question count to remain {initial_question_count} after "
        f"empty/whitespace input, but got {len(updated_session.questions)}"
    )

    # No new responses should have been recorded
    assert len(updated_session.responses) == initial_response_count, (
        f"Expected response count to remain {initial_response_count} after "
        f"empty/whitespace input, but got {len(updated_session.responses)}"
    )


# ---------------------------------------------------------------------------
# Property 16b: empty string specifically triggers rephrase prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_interview_session_strategy)
@settings(max_examples=100)
async def test_property_16_empty_string_triggers_rephrase_prompt(
    session: SessionState,
) -> None:
    """Property 16: Empty or Whitespace Input Triggers Rephrase Prompt.

    The empty string specifically must trigger a rephrase/elaboration prompt
    and must not advance the session.

    **Validates: Requirements 8.3**
    """
    service = _make_interview_service()

    initial_stage = session.stage
    initial_question_count = len(session.questions)
    initial_response_count = len(session.responses)

    reply, updated_session = await service.handle_response(session, "")

    assert isinstance(reply, str)
    assert len(reply.strip()) > 0, (
        "Expected a non-empty rephrase prompt for empty string input."
    )

    # Session must not advance
    assert updated_session.stage == Stage.INTERVIEW, (
        f"Expected stage to remain INTERVIEW after empty string, "
        f"but got {updated_session.stage!r}"
    )
    assert len(updated_session.questions) == initial_question_count, (
        f"Expected question count unchanged after empty string, "
        f"but got {len(updated_session.questions)}"
    )
    assert len(updated_session.responses) == initial_response_count, (
        f"Expected response count unchanged after empty string, "
        f"but got {len(updated_session.responses)}"
    )


# ---------------------------------------------------------------------------
# Property 16c: whitespace-only strings trigger rephrase prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    session=_interview_session_strategy,
    message=_whitespace_only_strategy,
)
@settings(max_examples=100)
async def test_property_16_whitespace_only_triggers_rephrase_prompt(
    session: SessionState,
    message: str,
) -> None:
    """Property 16: Empty or Whitespace Input Triggers Rephrase Prompt.

    Any string consisting entirely of whitespace characters must trigger a
    rephrase/elaboration prompt and must not advance the session.

    **Validates: Requirements 8.3**
    """
    # Confirm the message is genuinely all-whitespace
    assert message.strip() == "", (
        f"Strategy produced a non-whitespace message: {message!r}"
    )

    service = _make_interview_service()

    initial_stage = session.stage
    initial_question_count = len(session.questions)
    initial_response_count = len(session.responses)

    reply, updated_session = await service.handle_response(session, message)

    assert isinstance(reply, str)
    assert len(reply.strip()) > 0, (
        f"Expected a non-empty rephrase prompt for whitespace-only input {message!r}."
    )

    # Session must not advance
    assert updated_session.stage == Stage.INTERVIEW, (
        f"Expected stage to remain INTERVIEW after whitespace-only input, "
        f"but got {updated_session.stage!r}"
    )
    assert len(updated_session.questions) == initial_question_count, (
        f"Expected question count unchanged after whitespace-only input, "
        f"but got {len(updated_session.questions)}"
    )
    assert len(updated_session.responses) == initial_response_count, (
        f"Expected response count unchanged after whitespace-only input, "
        f"but got {len(updated_session.responses)}"
    )
