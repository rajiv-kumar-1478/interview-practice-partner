# Feature: voice-note-support, Property 3: Auto Mode Switch Matches Input Type
"""Property-based tests for InterviewService auto mode switching.

**Validates: Requirements 5.1, 5.2**
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

# Valid audio content types: starts with "audio/" or equals "application/ogg"
_audio_content_type_strategy = st.one_of(
    st.from_regex(r"audio/[a-z0-9]+", fullmatch=True),
    st.just("application/ogg"),
)

# Non-audio content types (for text messages, num_media=0 so content type is irrelevant,
# but we also test num_media>0 with non-audio types to ensure they are treated as text)
_non_audio_content_type_strategy = st.one_of(
    st.none(),
    st.text(min_size=1).filter(
        lambda s: not s.lower().startswith("audio/") and s.lower() != "application/ogg"
    ),
)


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


@st.composite
def _st_interview_session_with_questions(draw) -> SessionState:
    """Strategy that generates SessionState in INTERVIEW stage with at least one question.

    Generates sessions with:
    - Stage.INTERVIEW
    - 1-5 questions (at least one unanswered to simulate active interview)
    - 0-4 responses (may be fewer than questions)
    - Random preferred_mode (to verify the switch happens regardless of initial mode)
    """
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

    # Generate 0 to (num_questions - 1) responses so at least one question is unanswered
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
        role=Role.SOFTWARE_ENGINEER,
        questions=questions,
        responses=responses,
        created_at=created_at,
        updated_at=updated_at,
        preferred_mode=preferred_mode,
    )


def _make_service_with_mocks(
    transcribed_text: str = "I have extensive experience with Python and distributed systems.",
    llm_reply: str = '{"is_off_topic": false, "is_short": false, "follow_up_warranted": false, "follow_up_text": null, "difficulty_signal": "maintain"}',
    next_question: str = "Can you describe a challenging project you worked on?",
) -> InterviewService:
    """Create an InterviewService with all external dependencies mocked.

    Mocks:
    - LLM client: returns a valid evaluation JSON and a next question text
    - Whisper client: returns a non-empty transcription
    - TTS client: not called in these tests (mode switching happens before TTS)
    - Audio download client: returns dummy audio bytes
    """
    mock_llm = AsyncMock()
    # complete() is called for intent classification and response evaluation
    # and question generation — return appropriate values for each call
    mock_llm.complete = AsyncMock(side_effect=[
        # First call: intent classification → "answer"
        '{"intent": "answer"}',
        # Second call: response evaluation
        llm_reply,
        # Third call: generate next question
        next_question,
    ])

    mock_prompt_builder = AsyncMock()
    mock_prompt_builder.build_intent_classification_prompt = AsyncMock(return_value=[])
    mock_prompt_builder.build_response_evaluation_prompt = AsyncMock(return_value=[])
    mock_prompt_builder.build_question_generation_prompt = AsyncMock(return_value=[])

    mock_whisper = AsyncMock()
    mock_whisper.transcribe = AsyncMock(return_value=transcribed_text)

    mock_tts = AsyncMock()

    mock_audio_download = AsyncMock()
    mock_audio_download.download = AsyncMock(return_value=b"fake_audio_bytes")

    return InterviewService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
        whisper_client=mock_whisper,
        tts_client=mock_tts,
        audio_download_client=mock_audio_download,
    )


# ---------------------------------------------------------------------------
# Property 3a: Voice note input → preferred_mode becomes "voice"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    session=_st_interview_session_with_questions(),
    audio_content_type=_audio_content_type_strategy,
)
@settings(max_examples=100)
async def test_property_3a_voice_note_sets_preferred_mode_to_voice(
    session: SessionState,
    audio_content_type: str,
) -> None:
    """Property 3a: When handle_response is called with a voice note, preferred_mode becomes "voice".

    For any SessionState (regardless of initial preferred_mode) and any valid
    audio content type, processing a voice note via handle_response SHALL set
    preferred_mode to "voice" in the returned SessionState.

    **Validates: Requirements 5.1**
    """
    service = _make_service_with_mocks()

    # Act: process a voice note (num_media=1, audio content type, with media_url)
    _, updated_session = await service.handle_response(
        session,
        user_message="",  # voice notes have no text body
        num_media=1,
        media_content_type=audio_content_type,
        media_url="https://api.twilio.com/media/0",
    )

    assert updated_session.preferred_mode == "voice", (
        f"Expected preferred_mode='voice' after voice note with content_type={audio_content_type!r}, "
        f"but got preferred_mode={updated_session.preferred_mode!r}. "
        f"Initial preferred_mode was {session.preferred_mode!r}."
    )


# ---------------------------------------------------------------------------
# Property 3b: Text message input → preferred_mode becomes "text"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    session=_st_interview_session_with_questions(),
)
@settings(max_examples=100)
async def test_property_3b_text_message_sets_preferred_mode_to_text(
    session: SessionState,
) -> None:
    """Property 3b: When handle_response is called with a plain text message, preferred_mode becomes "text".

    For any SessionState (regardless of initial preferred_mode), processing a
    plain text message (num_media=0) via handle_response SHALL set preferred_mode
    to "text" in the returned SessionState.

    **Validates: Requirements 5.2**
    """
    service = _make_service_with_mocks()

    # Act: process a plain text message (num_media=0, no media)
    _, updated_session = await service.handle_response(
        session,
        user_message="I have extensive experience with Python and distributed systems.",
        num_media=0,
        media_content_type=None,
        media_url=None,
    )

    assert updated_session.preferred_mode == "text", (
        f"Expected preferred_mode='text' after plain text message, "
        f"but got preferred_mode={updated_session.preferred_mode!r}. "
        f"Initial preferred_mode was {session.preferred_mode!r}."
    )


# ---------------------------------------------------------------------------
# Property 3c: Mode switch is independent of initial preferred_mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    session=_st_interview_session_with_questions(),
    audio_content_type=_audio_content_type_strategy,
)
@settings(max_examples=100)
async def test_property_3c_voice_note_switches_from_text_mode_to_voice(
    session: SessionState,
    audio_content_type: str,
) -> None:
    """Property 3c: Voice note switches preferred_mode to "voice" even when initial mode is "text".

    This specifically tests Requirement 5.1: WHEN a Voice_Note is detected and
    the current Preferred_Mode is "text", THE InterviewService SHALL update the
    Preferred_Mode to "voice".

    **Validates: Requirements 5.1**
    """
    # Force initial mode to "text" to test the specific scenario in Req 5.1
    session.preferred_mode = "text"

    service = _make_service_with_mocks()

    _, updated_session = await service.handle_response(
        session,
        user_message="",
        num_media=1,
        media_content_type=audio_content_type,
        media_url="https://api.twilio.com/media/0",
    )

    assert updated_session.preferred_mode == "voice", (
        f"Expected preferred_mode='voice' after voice note (initial mode was 'text'), "
        f"but got preferred_mode={updated_session.preferred_mode!r}."
    )


@pytest.mark.asyncio
@given(
    session=_st_interview_session_with_questions(),
)
@settings(max_examples=100)
async def test_property_3d_text_message_switches_from_voice_mode_to_text(
    session: SessionState,
) -> None:
    """Property 3d: Text message switches preferred_mode to "text" even when initial mode is "voice".

    This specifically tests Requirement 5.2: WHEN a text message is detected and
    the current Preferred_Mode is "voice", THE InterviewService SHALL update the
    Preferred_Mode to "text".

    **Validates: Requirements 5.2**
    """
    # Force initial mode to "voice" to test the specific scenario in Req 5.2
    session.preferred_mode = "voice"

    service = _make_service_with_mocks()

    _, updated_session = await service.handle_response(
        session,
        user_message="I have extensive experience with Python and distributed systems.",
        num_media=0,
        media_content_type=None,
        media_url=None,
    )

    assert updated_session.preferred_mode == "text", (
        f"Expected preferred_mode='text' after plain text message (initial mode was 'voice'), "
        f"but got preferred_mode={updated_session.preferred_mode!r}."
    )
