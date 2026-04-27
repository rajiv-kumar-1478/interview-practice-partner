# Feature: voice-note-support, Property 4: Mode Commands Do Not Advance Interview State
"""Property-based tests for mode commands not advancing interview state.

**Validates: Requirements 6.5**
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from interview_practice_partner.domain.enums import QuestionType, Role, Stage
from interview_practice_partner.domain.models import Question, SessionState, UserResponse
from interview_practice_partner.services.interview import InterviewService

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_session_id_strategy = st.uuids().map(str)
_phone_number_strategy = st.from_regex(r"\+[1-9]\d{6,14}", fullmatch=True)
_datetime_strategy = st.datetimes(timezones=st.just(timezone.utc))
_preferred_mode_strategy = st.sampled_from(["voice", "text"])

# Mode command variants (case-insensitive, with/without whitespace)
_mode_command_strategy = st.sampled_from([
    "voice mode",
    "text mode",
    "VOICE MODE",
    "TEXT MODE",
    "Voice Mode",
    "Text Mode",
    "  voice mode  ",
    "  text mode  ",
    "\tvoice mode\t",
    "\ttext mode\t",
])


def _make_question(
    question_id: str | None = None,
    text: str = "Tell me about yourself.",
    question_type: QuestionType = QuestionType.BEHAVIOURAL,
    skipped: bool = False,
) -> Question:
    """Helper to create a Question instance."""
    return Question(
        question_id=question_id or str(uuid.uuid4()),
        text=text,
        question_type=question_type,
        asked_at=datetime.now(timezone.utc),
        skipped=skipped,
    )


def _make_response(question_id: str, text: str = "I have five years of experience.") -> UserResponse:
    """Helper to create a UserResponse instance."""
    return UserResponse(
        response_id=str(uuid.uuid4()),
        question_id=question_id,
        text=text,
        word_count=len(text.split()),
        received_at=datetime.now(timezone.utc),
    )


def _st_interview_session_with_questions():
    """Strategy that generates SessionState in INTERVIEW stage with at least one question.

    Generates sessions with:
    - Stage.INTERVIEW
    - 1-5 questions (at least one unanswered to simulate active interview)
    - 0-4 responses (may be fewer than questions)
    - Random preferred_mode
    """
    @st.composite
    def _build_session(draw):
        session_id = draw(_session_id_strategy)
        phone_number = draw(_phone_number_strategy)
        created_at = draw(_datetime_strategy)
        updated_at = draw(_datetime_strategy)
        preferred_mode = draw(_preferred_mode_strategy)
        
        # Generate 1-5 questions
        num_questions = draw(st.integers(min_value=1, max_value=5))
        questions = []
        for i in range(num_questions):
            q_id = str(uuid.uuid4())
            q = _make_question(
                question_id=q_id,
                text=f"Question {i + 1}: Tell me about your experience.",
                question_type=draw(st.sampled_from(list(QuestionType))),
                skipped=False,
            )
            questions.append(q)
        
        # Generate 0 to (num_questions - 1) responses
        # This ensures at least one question is unanswered (active interview)
        num_responses = draw(st.integers(min_value=0, max_value=max(0, num_questions - 1)))
        responses = []
        for i in range(num_responses):
            r = _make_response(
                question_id=questions[i].question_id,
                text=f"Response {i + 1}: I have experience with this.",
            )
            responses.append(r)
        
        return SessionState(
            session_id=session_id,
            phone_number=phone_number,
            stage=Stage.INTERVIEW,
            role=Role.SOFTWARE_ENGINEER,  # Fixed role for simplicity
            questions=questions,
            responses=responses,
            created_at=created_at,
            updated_at=updated_at,
            preferred_mode=preferred_mode,
        )
    
    return _build_session()


# ---------------------------------------------------------------------------
# Property 4: Mode Commands Do Not Advance Interview State
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    session=_st_interview_session_with_questions(),
    mode_command=_mode_command_strategy,
)
@settings(max_examples=100)
async def test_property_4_mode_commands_do_not_advance_interview_state(
    session: SessionState,
    mode_command: str,
) -> None:
    """Property 4: Mode Commands Do Not Advance Interview State.

    For any SessionState in the INTERVIEW stage with at least one active
    question, sending a mode command ("voice mode" or "text mode") SHALL
    leave the questions list and responses list unchanged. The current
    active question SHALL remain the same question after the command is
    processed.

    This property ensures that mode commands are purely control commands
    that do not affect the interview flow or state progression.

    **Validates: Requirements 6.5**
    """
    # Arrange: Create a mock InterviewService with minimal dependencies
    mock_llm = AsyncMock()
    mock_prompt_builder = AsyncMock()
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
    
    # Capture the state before processing the mode command
    questions_before = list(session.questions)
    responses_before = list(session.responses)
    questions_count_before = len(session.questions)
    responses_count_before = len(session.responses)
    
    # Determine which mode is being requested
    normalized_command = mode_command.strip().lower()
    if normalized_command == "voice mode":
        requested_mode = "voice"
    elif normalized_command == "text mode":
        requested_mode = "text"
    else:
        # Skip invalid commands (shouldn't happen with our strategy)
        return
    
    # Act: Process the mode command directly via handle_mode_command
    # Note: Task 5.11 (routing in handle_response) is not yet implemented,
    # so we test handle_mode_command directly
    _, updated_session = await service.handle_mode_command(session, requested_mode)
    
    # Assert: Questions and responses lists must be unchanged
    assert len(updated_session.questions) == questions_count_before, (
        f"Mode command {mode_command!r} changed questions count: "
        f"before={questions_count_before}, after={len(updated_session.questions)}"
    )
    
    assert len(updated_session.responses) == responses_count_before, (
        f"Mode command {mode_command!r} changed responses count: "
        f"before={responses_count_before}, after={len(updated_session.responses)}"
    )
    
    # Verify questions list content is unchanged (same question IDs in same order)
    assert [q.question_id for q in updated_session.questions] == [q.question_id for q in questions_before], (
        f"Mode command {mode_command!r} changed questions list: "
        f"before={[q.question_id for q in questions_before]}, "
        f"after={[q.question_id for q in updated_session.questions]}"
    )
    
    # Verify responses list content is unchanged (same response IDs in same order)
    assert [r.response_id for r in updated_session.responses] == [r.response_id for r in responses_before], (
        f"Mode command {mode_command!r} changed responses list: "
        f"before={[r.response_id for r in responses_before]}, "
        f"after={[r.response_id for r in updated_session.responses]}"
    )
    
    # Verify the preferred_mode was updated (this is the only expected change)
    assert updated_session.preferred_mode == requested_mode, (
        f"Expected preferred_mode to be {requested_mode!r} after {mode_command!r}, "
        f"got {updated_session.preferred_mode!r}"
    )


@pytest.mark.asyncio
@given(
    session=_st_interview_session_with_questions(),
)
@settings(max_examples=100)
async def test_property_4_both_mode_commands_preserve_state(
    session: SessionState,
) -> None:
    """Property 4 (variant): Both mode commands preserve state independently.

    This variant tests that both "voice mode" and "text mode" commands
    preserve interview state, and that switching modes multiple times
    does not accumulate any state changes.

    **Validates: Requirements 6.5**
    """
    # Arrange
    mock_llm = AsyncMock()
    mock_prompt_builder = AsyncMock()
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
    
    # Capture initial state
    questions_initial = list(session.questions)
    responses_initial = list(session.responses)
    
    # Act: Switch to voice mode
    _, session_after_voice = await service.handle_mode_command(session, "voice")
    
    # Assert: State unchanged after voice mode
    assert [q.question_id for q in session_after_voice.questions] == [q.question_id for q in questions_initial]
    assert [r.response_id for r in session_after_voice.responses] == [r.response_id for r in responses_initial]
    assert session_after_voice.preferred_mode == "voice"
    
    # Act: Switch to text mode
    _, session_after_text = await service.handle_mode_command(session_after_voice, "text")
    
    # Assert: State still unchanged after text mode
    assert [q.question_id for q in session_after_text.questions] == [q.question_id for q in questions_initial]
    assert [r.response_id for r in session_after_text.responses] == [r.response_id for r in responses_initial]
    assert session_after_text.preferred_mode == "text"
    
    # Act: Switch back to voice mode
    _, session_after_voice_again = await service.handle_mode_command(session_after_text, "voice")
    
    # Assert: State still unchanged after multiple switches
    assert [q.question_id for q in session_after_voice_again.questions] == [q.question_id for q in questions_initial]
    assert [r.response_id for r in session_after_voice_again.responses] == [r.response_id for r in responses_initial]
    assert session_after_voice_again.preferred_mode == "voice"
