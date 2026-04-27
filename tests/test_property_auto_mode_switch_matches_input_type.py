# Feature: voice-note-support, Property 3: Auto Mode Switch Matches Input Type
"""Property-based tests for automatic mode switching in InterviewService.

**Validates: Requirements 5.1, 5.2**
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest

from interview_practice_partner.domain.enums import QuestionType, Role, Stage
from interview_practice_partner.domain.models import Question, SessionState
from interview_practice_partner.services.interview import InterviewService

# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------

NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
PHONE = "+15550001234"


def make_session(**overrides) -> SessionState:
    """Create a test SessionState with sensible defaults."""
    defaults = dict(
        session_id=str(uuid.uuid4()),
        phone_number=PHONE,
        stage=Stage.INTERVIEW,
        role=Role.SOFTWARE_ENGINEER,
        created_at=NOW,
        updated_at=NOW,
    )
    defaults.update(overrides)
    return SessionState(**defaults)


def make_question(
    question_id: str | None = None,
    text: str = "Tell me about yourself.",
    question_type: QuestionType = QuestionType.BEHAVIOURAL,
) -> Question:
    """Create a test Question with sensible defaults."""
    return Question(
        question_id=question_id or str(uuid.uuid4()),
        text=text,
        question_type=question_type,
        asked_at=NOW,
    )


def make_service() -> tuple[InterviewService, AsyncMock, MagicMock]:
    """Build an InterviewService with mocked dependencies for property testing."""
    # Mock LLM client
    mock_llm = AsyncMock()
    
    # Mock prompt builder
    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_question_generation_prompt.return_value = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    mock_prompt_builder.build_response_evaluation_prompt.return_value = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]

    # Mock audio clients
    mock_whisper = AsyncMock()
    mock_whisper.transcribe.return_value = "This is a comprehensive and detailed transcribed voice note response that contains more than fifteen words to ensure it bypasses the short response detection logic and allows the mode switching functionality to work correctly during property-based testing."
    
    mock_tts = AsyncMock()
    mock_tts.synthesise.return_value = b"fake_audio_bytes"
    
    mock_audio_download = AsyncMock()
    mock_audio_download.download.return_value = b"fake_audio_data"

    service = InterviewService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
        whisper_client=mock_whisper,
        tts_client=mock_tts,
        audio_download_client=mock_audio_download,
    )
    return service, mock_llm, mock_prompt_builder


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for initial preferred_mode values
_initial_mode_strategy = st.sampled_from(["voice", "text"])

# Strategy for voice note parameters (num_media > 0 and audio content type)
_voice_note_params_strategy = st.tuples(
    st.integers(min_value=1, max_value=5),  # num_media > 0
    st.sampled_from([
        "audio/ogg", "audio/mpeg", "audio/wav", "audio/mp4", 
        "application/ogg", "AUDIO/OGG", "Audio/Mpeg"
    ])  # audio content types (including case variations)
)

# Strategy for text message parameters (num_media == 0 or non-audio content type)
_text_message_params_strategy = st.one_of(
    # num_media == 0 (no media)
    st.tuples(st.just(0), st.one_of(st.none(), st.text())),
    # num_media > 0 but non-audio content type
    st.tuples(
        st.integers(min_value=1, max_value=5),
        st.sampled_from([
            "image/jpeg", "video/mp4", "text/plain", "application/pdf",
            "image/png", "video/avi", None
        ])
    )
)

# Strategy for user message text (non-mode-command, long enough to avoid short response handling)
_user_message_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
    min_size=100, 
    max_size=300
).map(
    lambda s: " ".join(s.split()[:20] + ["additional", "words", "to", "ensure", "sufficient", "length", "for", "testing", "purposes", "and", "avoid", "short", "response", "handling", "logic"])
).filter(
    lambda s: s.strip().lower() not in ("voice mode", "text mode")
)

# Strategy for media URLs
_media_url_strategy = st.one_of(
    st.none(),
    st.from_regex(r"https://api\.twilio\.com/[a-zA-Z0-9/]+", fullmatch=True)
)


# ---------------------------------------------------------------------------
# Property 3: Auto Mode Switch Matches Input Type
# ---------------------------------------------------------------------------


@given(
    initial_mode=_initial_mode_strategy,
    voice_params=_voice_note_params_strategy,
    user_message=_user_message_strategy,
    media_url=_media_url_strategy,
)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_property_3a_voice_note_switches_to_voice_mode(
    initial_mode: str,
    voice_params: tuple[int, str],
    user_message: str,
    media_url: str | None,
) -> None:
    """Property 3a: Voice notes automatically switch preferred_mode to 'voice'.

    **Validates: Requirement 5.1**
    """
    num_media, media_content_type = voice_params
    
    # Arrange: create session with initial mode and a question
    session = make_session(preferred_mode=initial_mode)
    session.questions.append(make_question())
    
    service, mock_llm, _ = make_service()
    
    # Mock LLM responses for intent classification and evaluation
    mock_llm.complete.side_effect = [
        json.dumps({"intent": "answer"}),  # Intent classification
        json.dumps({  # Response evaluation
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        }),
        "Next question?"  # Question generation
    ]
    
    # Act: process voice note message
    reply, updated_session = await service.handle_response(
        session,
        user_message,
        num_media=num_media,
        media_content_type=media_content_type,
        media_url=media_url or "https://api.twilio.com/test/media/123",
    )
    
    # Assert: preferred_mode should be 'voice' regardless of initial mode
    assert updated_session.preferred_mode == "voice", (
        f"Expected preferred_mode='voice' after voice note, got {updated_session.preferred_mode!r}. "
        f"Initial mode: {initial_mode!r}, num_media: {num_media}, content_type: {media_content_type!r}"
    )
    
    # Verify the message was processed (response recorded)
    assert len(updated_session.responses) == 1
    assert isinstance(reply, str)
    assert len(reply) > 0


@given(
    initial_mode=_initial_mode_strategy,
    text_params=_text_message_params_strategy,
    user_message=_user_message_strategy,
)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_property_3b_text_message_switches_to_text_mode(
    initial_mode: str,
    text_params: tuple[int, str | None],
    user_message: str,
) -> None:
    """Property 3b: Text messages automatically switch preferred_mode to 'text'.

    **Validates: Requirement 5.2**
    """
    num_media, media_content_type = text_params
    
    # Arrange: create session with initial mode and a question
    session = make_session(preferred_mode=initial_mode)
    session.questions.append(make_question())
    
    service, mock_llm, _ = make_service()
    
    # Mock LLM responses for intent classification and evaluation
    mock_llm.complete.side_effect = [
        json.dumps({"intent": "answer"}),  # Intent classification
        json.dumps({  # Response evaluation
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        }),
        "Next question?"  # Question generation
    ]
    
    # Act: process text message
    reply, updated_session = await service.handle_response(
        session,
        user_message,
        num_media=num_media,
        media_content_type=media_content_type,
    )
    
    # Assert: preferred_mode should be 'text' regardless of initial mode
    assert updated_session.preferred_mode == "text", (
        f"Expected preferred_mode='text' after text message, got {updated_session.preferred_mode!r}. "
        f"Initial mode: {initial_mode!r}, num_media: {num_media}, content_type: {media_content_type!r}"
    )
    
    # Verify the message was processed (response recorded)
    assert len(updated_session.responses) == 1
    assert isinstance(reply, str)
    assert len(reply) > 0


@given(
    initial_mode=_initial_mode_strategy,
    is_voice_note=st.booleans(),
    user_message=_user_message_strategy,
)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_property_3c_mode_switch_is_deterministic(
    initial_mode: str,
    is_voice_note: bool,
    user_message: str,
) -> None:
    """Property 3c: Mode switching is deterministic based on input type.

    **Validates: Requirements 5.1, 5.2**
    """
    # Arrange: create session with initial mode and a question
    session = make_session(preferred_mode=initial_mode)
    session.questions.append(make_question())
    
    service, mock_llm, _ = make_service()
    
    # Mock LLM responses
    mock_llm.complete.side_effect = [
        json.dumps({"intent": "answer"}),
        json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        }),
        "Next question?"
    ]
    
    # Set up message parameters based on input type
    if is_voice_note:
        num_media, media_content_type = 1, "audio/ogg"
        media_url = "https://api.twilio.com/test/media/123"
        expected_mode = "voice"
    else:
        num_media, media_content_type = 0, None
        media_url = None
        expected_mode = "text"
    
    # Act: process message
    reply, updated_session = await service.handle_response(
        session,
        user_message,
        num_media=num_media,
        media_content_type=media_content_type,
        media_url=media_url,
    )
    
    # Assert: mode should match input type regardless of initial mode
    assert updated_session.preferred_mode == expected_mode, (
        f"Mode switching not deterministic. Expected {expected_mode!r}, got {updated_session.preferred_mode!r}. "
        f"Initial mode: {initial_mode!r}, is_voice_note: {is_voice_note}"
    )


@given(
    voice_params=_voice_note_params_strategy,
    user_message=_user_message_strategy,
)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_property_3d_voice_mode_switch_is_silent(
    voice_params: tuple[int, str],
    user_message: str,
) -> None:
    """Property 3d: Automatic mode switches are silent (no notification to user).

    **Validates: Requirement 5.3**
    """
    num_media, media_content_type = voice_params
    
    # Arrange: create session starting in text mode
    session = make_session(preferred_mode="text")
    session.questions.append(make_question())
    
    service, mock_llm, _ = make_service()
    
    # Mock LLM responses
    mock_llm.complete.side_effect = [
        json.dumps({"intent": "answer"}),
        json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        }),
        "Next question?"
    ]
    
    # Act: process voice note (should switch from text to voice silently)
    reply, updated_session = await service.handle_response(
        session,
        user_message,
        num_media=num_media,
        media_content_type=media_content_type,
        media_url="https://api.twilio.com/test/media/123",
    )
    
    # Assert: mode switched to voice
    assert updated_session.preferred_mode == "voice"
    
    # Assert: reply should not contain mode switch notification
    reply_lower = reply.lower()
    mode_switch_phrases = [
        "voice mode", "text mode", "switched to", "mode is now", 
        "changed to voice", "now in voice", "mode changed"
    ]
    
    for phrase in mode_switch_phrases:
        assert phrase not in reply_lower, (
            f"Automatic mode switch should be silent, but reply contains '{phrase}': {reply!r}"
        )


@given(
    text_params=_text_message_params_strategy,
    user_message=_user_message_strategy,
)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_property_3e_text_mode_switch_is_silent(
    text_params: tuple[int, str | None],
    user_message: str,
) -> None:
    """Property 3e: Automatic text mode switches are silent (no notification to user).

    **Validates: Requirement 5.3**
    """
    num_media, media_content_type = text_params
    
    # Arrange: create session starting in voice mode
    session = make_session(preferred_mode="voice")
    session.questions.append(make_question())
    
    service, mock_llm, _ = make_service()
    
    # Mock LLM responses
    mock_llm.complete.side_effect = [
        json.dumps({"intent": "answer"}),
        json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        }),
        "Next question?"
    ]
    
    # Act: process text message (should switch from voice to text silently)
    reply, updated_session = await service.handle_response(
        session,
        user_message,
        num_media=num_media,
        media_content_type=media_content_type,
    )
    
    # Assert: mode switched to text
    assert updated_session.preferred_mode == "text"
    
    # Assert: reply should not contain mode switch notification
    reply_lower = reply.lower()
    mode_switch_phrases = [
        "voice mode", "text mode", "switched to", "mode is now", 
        "changed to text", "now in text", "mode changed"
    ]
    
    for phrase in mode_switch_phrases:
        assert phrase not in reply_lower, (
            f"Automatic mode switch should be silent, but reply contains '{phrase}': {reply!r}"
        )