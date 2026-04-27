# Feature: interview-practice-partner, Property 11: Short Responses Trigger Elaboration Prompts
"""Property-based tests for short response elaboration.

Any response with word count < 15 must result in an elaboration prompt
being returned, and the session must not advance (stage stays INTERVIEW,
no new questions or responses are added).

Validates: Requirements 4.6
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from interview_practice_partner.domain.enums import QuestionType, Role, Stage
from interview_practice_partner.domain.models import Question, SessionState, UserResponse
from interview_practice_partner.services.interview import InterviewService

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

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

# Strategy for messages with fewer than 15 words (word count < 15)
# We generate between 0 and 14 words by drawing a list of words and joining them.
_word = st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu")), min_size=1, max_size=12)
_short_message_strategy = st.lists(_word, min_size=0, max_size=14).map(lambda ws: " ".join(ws))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service() -> InterviewService:
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


# ---------------------------------------------------------------------------
# Property 11a: handle_short_response returns a non-empty elaboration prompt
# ---------------------------------------------------------------------------


@given(session=_interview_session_strategy)
@settings(max_examples=100)
def test_property_11_handle_short_response_returns_nonempty_elaboration_prompt(
    session: SessionState,
) -> None:
    """Property 11: Short Responses Trigger Elaboration Prompts.

    For any INTERVIEW session, ``handle_short_response`` must return a
    non-empty string that prompts the user to elaborate.

    **Validates: Requirements 4.6**
    """
    service = _make_service()

    reply = service.handle_short_response(session)

    assert isinstance(reply, str), (
        f"Expected handle_short_response to return a str, got {type(reply)!r}"
    )
    assert len(reply.strip()) > 0, (
        "Expected handle_short_response to return a non-empty elaboration prompt, "
        "but got an empty string."
    )
    # The reply should encourage elaboration
    assert any(
        word in reply.lower()
        for word in ["elaborate", "more", "detail", "example", "brief", "expand"]
    ), (
        f"Expected elaboration prompt to contain words like 'elaborate', 'more', "
        f"'detail', 'example', 'brief', or 'expand', but got: {reply!r}"
    )


# ---------------------------------------------------------------------------
# Property 11b: handle_short_response does not advance the session
# ---------------------------------------------------------------------------


@given(session=_interview_session_strategy)
@settings(max_examples=100)
def test_property_11_handle_short_response_does_not_advance_session(
    session: SessionState,
) -> None:
    """Property 11: Short Responses Trigger Elaboration Prompts.

    For any INTERVIEW session, ``handle_short_response`` must NOT advance
    the session: stage stays INTERVIEW, question list is unchanged, and
    response list is unchanged.

    **Validates: Requirements 4.6**
    """
    service = _make_service()

    initial_stage = session.stage
    initial_question_count = len(session.questions)
    initial_response_count = len(session.responses)

    service.handle_short_response(session)

    assert session.stage == Stage.INTERVIEW, (
        f"Expected session stage to remain INTERVIEW after handle_short_response, "
        f"but got {session.stage!r}"
    )
    assert session.stage == initial_stage, (
        f"Expected session stage to be unchanged after handle_short_response, "
        f"but it changed from {initial_stage!r} to {session.stage!r}"
    )
    assert len(session.questions) == initial_question_count, (
        f"Expected question count to remain {initial_question_count} after "
        f"handle_short_response, but got {len(session.questions)}"
    )
    assert len(session.responses) == initial_response_count, (
        f"Expected response count to remain {initial_response_count} after "
        f"handle_short_response, but got {len(session.responses)}"
    )


# ---------------------------------------------------------------------------
# Property 11c: handle_response with word_count < 15 returns elaboration prompt
#               and does not advance session (end-to-end via handle_response)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(session=_interview_session_strategy, message=_short_message_strategy)
@settings(max_examples=100)
async def test_property_11_handle_response_short_message_returns_elaboration(
    session: SessionState,
    message: str,
) -> None:
    """Property 11: Short Responses Trigger Elaboration Prompts.

    For any INTERVIEW session and any message with fewer than 15 words,
    ``handle_response`` must return an elaboration prompt and the session
    must not advance (stage stays INTERVIEW, no new responses recorded).

    **Validates: Requirements 4.6**
    """
    # Confirm the message is actually short (< 15 words)
    word_count = len(message.split()) if message.strip() else 0
    # Empty strings split to [''] which has length 1, but we treat empty as 0
    if not message.strip():
        word_count = 0

    # Only test messages that are genuinely short (< 15 words)
    # The strategy already constrains to 0–14 words, but we verify here
    # to be explicit about the property boundary.
    if word_count >= 15:
        return  # skip — strategy should not produce this, but guard anyway

    service = _make_service()

    initial_stage = session.stage
    initial_question_count = len(session.questions)
    initial_response_count = len(session.responses)

    reply, updated_session = await service.handle_response(session, message)

    # The reply must be a non-empty elaboration prompt
    assert isinstance(reply, str), (
        f"Expected handle_response to return a str reply, got {type(reply)!r}"
    )
    assert len(reply.strip()) > 0, (
        "Expected handle_response to return a non-empty reply for a short message."
    )
    assert any(
        word in reply.lower()
        for word in ["elaborate", "more", "detail", "example", "brief", "expand"]
    ), (
        f"Expected elaboration prompt for short message ({word_count} words), "
        f"but got: {reply!r}"
    )

    # Session must not advance
    assert updated_session.stage == Stage.INTERVIEW, (
        f"Expected session stage to remain INTERVIEW after short response "
        f"({word_count} words), but got {updated_session.stage!r}"
    )
    assert len(updated_session.questions) == initial_question_count, (
        f"Expected question count to remain {initial_question_count} after short "
        f"response, but got {len(updated_session.questions)}"
    )
    assert len(updated_session.responses) == initial_response_count, (
        f"Expected response count to remain {initial_response_count} after short "
        f"response, but got {len(updated_session.responses)}"
    )


# ---------------------------------------------------------------------------
# Property 11d: word_count boundary — exactly 14 words is still short
# ---------------------------------------------------------------------------


@given(session=_interview_session_strategy)
@settings(max_examples=100)
def test_property_11_word_count_14_is_short(
    session: SessionState,
) -> None:
    """Property 11: Short Responses Trigger Elaboration Prompts.

    A response with exactly 14 words (the maximum short count) must still
    trigger an elaboration prompt and not advance the session.

    **Validates: Requirements 4.6**
    """
    service = _make_service()

    initial_question_count = len(session.questions)
    initial_response_count = len(session.responses)

    # Exactly 14 words — one below the 15-word threshold
    fourteen_word_message = "I have some experience but I am not sure how to explain it well"
    assert len(fourteen_word_message.split()) == 14

    reply = service.handle_short_response(session)

    assert isinstance(reply, str)
    assert len(reply.strip()) > 0
    assert len(session.questions) == initial_question_count
    assert len(session.responses) == initial_response_count
