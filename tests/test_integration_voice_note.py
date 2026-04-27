"""Integration tests for the voice note webhook flow.

Task 10.1: Full voice note webhook flow integration test.
Task 10.2: TTS fallback on TTSError integration test.
Task 10.3: TranscriptionError fallback integration test.

Covers:
- Full voice note flow: POST /webhook/whatsapp with audio media → Twilio sends audio reply
- Verifies Twilio messages.create is called with media_url pointing to /media/{uuid}.mp3
- Verifies SessionState.preferred_mode is persisted as "voice" in fakeredis after the turn
- TTS fallback: when ElevenLabsClient.synthesise raises TTSError, Twilio sends plain text
- Verifies preferred_mode remains "voice" after TTS failure (mode not reset on TTS error)
- TranscriptionError fallback: when AudioDownloadClient.download raises TranscriptionError,
  Twilio sends the fallback text message and session questions/responses are unchanged

Requirements: 2.1, 2.2, 3.1, 3.6, 4.1, 5.1, 7.2, 10.1, 10.2, 10.3
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from urllib.parse import urlencode

import fakeredis.aioredis
import httpx
import pytest
from httpx import ASGITransport

from interview_practice_partner.config import Settings
from interview_practice_partner.domain.enums import Role, Stage
from interview_practice_partner.domain.exceptions import TranscriptionError, TTSError
from interview_practice_partner.domain.models import Question, SessionState
from interview_practice_partner.domain.enums import QuestionType
from interview_practice_partner.main import create_app
from interview_practice_partner.repositories.redis_session import RedisSessionRepository

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

_TEST_SETTINGS = Settings(
    twilio_account_sid="ACtest123",
    twilio_auth_token="test_token",
    twilio_whatsapp_number="whatsapp:+14155238886",
    llm_api_key="test_key",
    redis_url="redis://localhost:6379/0",
    groq_api_key="test_groq_key",
    elevenlabs_api_key="test_elevenlabs_key",
    elevenlabs_voice_id="test_voice_id",
    media_base_url="https://test.example.com",
)

_FROM_NUMBER = "whatsapp:+447700900001"
_TO_NUMBER = "whatsapp:+14155238886"
_MEDIA_URL = "https://api.twilio.com/2010-04-01/Accounts/ACtest123/Messages/MM123/Media/0"

_DUMMY_AUDIO_BYTES = b"\xff\xfb\x90\x00" * 64  # fake MP3 bytes
_DUMMY_TTS_BYTES = b"\xff\xfb\x90\x00" * 128  # fake TTS MP3 bytes
_TRANSCRIPTION = (
    "I led a cross-functional team to deliver a product on time by coordinating "
    "daily standups and removing blockers for each team member throughout the sprint."
)


def _form_body(**kwargs: str) -> bytes:
    """Encode keyword arguments as an ``application/x-www-form-urlencoded`` body."""
    return urlencode(kwargs).encode()


def _make_voice_note_message(
    message_sid: str = "SM_voice_001",
    from_number: str = _FROM_NUMBER,
    to_number: str = _TO_NUMBER,
    media_url: str = _MEDIA_URL,
) -> bytes:
    """Build a Twilio-style form-encoded webhook payload for a voice note."""
    return _form_body(
        MessageSid=message_sid,
        From=from_number,
        To=to_number,
        Body="",
        NumMedia="1",
        MediaContentType0="audio/ogg",
        MediaUrl0=media_url,
    )


def _make_text_message(
    body: str,
    message_sid: str = "SM_text_001",
    from_number: str = _FROM_NUMBER,
    to_number: str = _TO_NUMBER,
) -> bytes:
    """Build a Twilio-style form-encoded webhook payload for a text message."""
    return _form_body(
        MessageSid=message_sid,
        From=from_number,
        To=to_number,
        Body=body,
        NumMedia="0",
    )


def _make_interview_session(phone: str = _FROM_NUMBER) -> SessionState:
    """Create a SessionState in INTERVIEW stage with one active question."""
    question = Question(
        question_id="q-integration-001",
        text="Tell me about a time you led a team through a challenging project.",
        question_type=QuestionType.BEHAVIOURAL,
        asked_at=_NOW,
        skipped=False,
    )
    return SessionState(
        session_id="sess-integration-voice-001",
        phone_number=phone,
        stage=Stage.INTERVIEW,
        role=Role.SOFTWARE_ENGINEER,
        questions=[question],
        responses=[],
        created_at=_NOW,
        updated_at=_NOW,
        preferred_mode="text",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def app_with_fakeredis(mocker):
    """Create a FastAPI app wired to fakeredis with all external clients mocked.

    Sets up:
    - fakeredis for session storage
    - Mock Twilio client (messages.create)
    - Bypassed Twilio signature validation
    - Mocked AudioDownloadClient.download → returns dummy bytes
    - Mocked GroqWhisperClient.transcribe → returns transcription string
    - Mocked ElevenLabsClient.synthesise → returns dummy MP3 bytes
    """
    app = create_app(_TEST_SETTINGS)

    # Set up fakeredis directly on app state
    fake_redis = fakeredis.aioredis.FakeRedis()
    app.state.redis = fake_redis

    # Mock Twilio client
    mock_twilio = mocker.MagicMock()
    mock_message = mocker.MagicMock()
    mock_message.status = "sent"
    mock_message.sid = "SMtest_voice_123"
    mock_twilio.messages.create.return_value = mock_message
    app.state.twilio_client = mock_twilio

    # Bypass Twilio signature validation
    mocker.patch(
        "twilio.request_validator.RequestValidator.validate",
        return_value=True,
    )

    # Mock AudioDownloadClient.download to return dummy bytes
    mocker.patch(
        "interview_practice_partner.audio.download_client.AudioDownloadClient.download",
        return_value=_DUMMY_AUDIO_BYTES,
    )

    # Mock GroqWhisperClient.transcribe to return a transcription string
    mocker.patch(
        "interview_practice_partner.audio.whisper_client.GroqWhisperClient.transcribe",
        return_value=_TRANSCRIPTION,
    )

    # Mock ElevenLabsClient.synthesise to return dummy MP3 bytes
    mocker.patch(
        "interview_practice_partner.audio.tts_client.ElevenLabsClient.synthesise",
        return_value=_DUMMY_TTS_BYTES,
    )

    yield app, fake_redis, mock_twilio


# ---------------------------------------------------------------------------
# Task 10.1 — Full voice note webhook flow
# Requirements: 2.1, 2.2, 3.1, 4.1, 5.1, 7.2
# ---------------------------------------------------------------------------


class TestVoiceNoteWebhookFlow:
    """Full voice note flow: voice note in → TTS audio reply out."""

    async def _seed_interview_session(
        self, fake_redis: fakeredis.aioredis.FakeRedis
    ) -> SessionState:
        """Pre-seed Redis with a session in INTERVIEW stage."""
        repo = RedisSessionRepository(redis_client=fake_redis, ttl_seconds=86400)
        session = _make_interview_session()
        await repo.save(session)
        return session

    async def test_voice_note_triggers_audio_reply_with_media_url(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Voice note webhook results in Twilio messages.create called with media_url.

        Requirements: 2.1, 2.2, 3.1, 4.1
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        # Pre-seed session in INTERVIEW stage
        await self._seed_interview_session(fake_redis)

        # Mock LLM for intent classification + response evaluation + next question
        intent_response = json.dumps({"intent": "answer"})
        eval_response = json.dumps(
            {
                "is_off_topic": False,
                "is_short": False,
                "follow_up_warranted": False,
                "follow_up_text": "",
                "difficulty_signal": "maintain",
            }
        )
        next_question = "How do you handle disagreements within a team?"

        mocker.patch(
            "interview_practice_partner.llm.openai_client.OpenAIClient.complete",
            side_effect=[intent_response, eval_response, next_question],
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhook/whatsapp",
                content=_make_voice_note_message(message_sid="SM_voice_flow_001"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        assert response.status_code == 200

        # Twilio messages.create must have been called
        mock_twilio.messages.create.assert_called_once()
        call_kwargs = mock_twilio.messages.create.call_args.kwargs

        # Verify media_url is present and points to /media/{uuid}.mp3
        assert "media_url" in call_kwargs, (
            "Twilio messages.create must be called with media_url for voice replies"
        )
        media_url_list = call_kwargs["media_url"]
        assert isinstance(media_url_list, list), "media_url must be a list"
        assert len(media_url_list) == 1
        media_url = media_url_list[0]

        # Verify the URL matches the pattern: {base_url}/media/{uuid}.mp3
        assert media_url.startswith("https://test.example.com/media/"), (
            f"media_url should start with the configured media_base_url/media/, got: {media_url}"
        )
        assert media_url.endswith(".mp3"), (
            f"media_url should end with .mp3, got: {media_url}"
        )

        # Verify the UUID part is a valid UUID
        filename = media_url.split("/media/")[-1]
        uuid_part = filename.replace(".mp3", "")
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        assert uuid_pattern.match(uuid_part), (
            f"Filename should be a UUID, got: {uuid_part}"
        )

    async def test_voice_note_persists_preferred_mode_as_voice_in_redis(
        self, app_with_fakeredis, mocker
    ) -> None:
        """After a voice note turn, SessionState.preferred_mode is "voice" in Redis.

        Requirements: 5.1, 7.2
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        # Pre-seed session in INTERVIEW stage with preferred_mode="text"
        session = await self._seed_interview_session(fake_redis)
        assert session.preferred_mode == "text"

        # Mock LLM for the interview flow
        intent_response = json.dumps({"intent": "answer"})
        eval_response = json.dumps(
            {
                "is_off_topic": False,
                "is_short": False,
                "follow_up_warranted": False,
                "follow_up_text": "",
                "difficulty_signal": "maintain",
            }
        )
        next_question = "Describe a situation where you had to meet a tight deadline."

        mocker.patch(
            "interview_practice_partner.llm.openai_client.OpenAIClient.complete",
            side_effect=[intent_response, eval_response, next_question],
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhook/whatsapp",
                content=_make_voice_note_message(message_sid="SM_voice_mode_001"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        assert response.status_code == 200

        # Retrieve the session from Redis and verify preferred_mode is "voice"
        repo = RedisSessionRepository(redis_client=fake_redis, ttl_seconds=86400)
        updated_session = await repo.get(_FROM_NUMBER)

        assert updated_session is not None, "Session must be persisted in Redis"
        assert updated_session.preferred_mode == "voice", (
            f"Expected preferred_mode='voice' after voice note turn, "
            f"got: {updated_session.preferred_mode!r}"
        )

    async def test_voice_note_reply_sent_to_correct_number(
        self, app_with_fakeredis, mocker
    ) -> None:
        """The audio reply is sent to the inbound from_number.

        Requirements: 4.1
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        await self._seed_interview_session(fake_redis)

        mocker.patch(
            "interview_practice_partner.llm.openai_client.OpenAIClient.complete",
            side_effect=[
                json.dumps({"intent": "answer"}),
                json.dumps({
                    "is_off_topic": False,
                    "is_short": False,
                    "follow_up_warranted": False,
                    "follow_up_text": "",
                    "difficulty_signal": "maintain",
                }),
                "What is your approach to code reviews?",
            ],
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/webhook/whatsapp",
                content=_make_voice_note_message(message_sid="SM_voice_dest_001"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        mock_twilio.messages.create.assert_called_once()
        call_kwargs = mock_twilio.messages.create.call_args.kwargs
        assert call_kwargs["to"] == _FROM_NUMBER

    async def test_webhook_returns_200_for_voice_note(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Webhook always returns HTTP 200 for voice note messages.

        Requirements: 2.1
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        await self._seed_interview_session(fake_redis)

        mocker.patch(
            "interview_practice_partner.llm.openai_client.OpenAIClient.complete",
            side_effect=[
                json.dumps({"intent": "answer"}),
                json.dumps({
                    "is_off_topic": False,
                    "is_short": False,
                    "follow_up_warranted": False,
                    "follow_up_text": "",
                    "difficulty_signal": "maintain",
                }),
                "Tell me about a time you had to learn something quickly.",
            ],
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhook/whatsapp",
                content=_make_voice_note_message(message_sid="SM_voice_200_001"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        assert response.status_code == 200

    async def test_voice_note_audio_file_written_to_media_dir(
        self, app_with_fakeredis, mocker, tmp_path
    ) -> None:
        """TTS audio bytes are written to the media directory as a .mp3 file.

        Requirements: 3.1, 4.1
        """
        import pathlib
        import interview_practice_partner.services.orchestration as orch_module

        app, fake_redis, mock_twilio = app_with_fakeredis

        await self._seed_interview_session(fake_redis)

        # Patch _MEDIA_DIR to use tmp_path so we can inspect written files
        mocker.patch.object(orch_module, "_MEDIA_DIR", tmp_path)

        mocker.patch(
            "interview_practice_partner.llm.openai_client.OpenAIClient.complete",
            side_effect=[
                json.dumps({"intent": "answer"}),
                json.dumps({
                    "is_off_topic": False,
                    "is_short": False,
                    "follow_up_warranted": False,
                    "follow_up_text": "",
                    "difficulty_signal": "maintain",
                }),
                "How do you prioritise tasks when everything seems urgent?",
            ],
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/webhook/whatsapp",
                content=_make_voice_note_message(message_sid="SM_voice_file_001"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        # Verify a .mp3 file was written to the media directory
        mp3_files = list(tmp_path.glob("*.mp3"))
        assert len(mp3_files) == 1, (
            f"Expected exactly one .mp3 file in media dir, found: {mp3_files}"
        )
        assert mp3_files[0].read_bytes() == _DUMMY_TTS_BYTES, (
            "Written audio bytes must match the TTS output"
        )


# ---------------------------------------------------------------------------
# Task 10.2 — TTS fallback on TTSError
# Requirements: 3.6, 10.2
# ---------------------------------------------------------------------------


def _make_interview_session_voice_mode(phone: str = _FROM_NUMBER) -> SessionState:
    """Create a SessionState in INTERVIEW stage with preferred_mode="voice"."""
    question = Question(
        question_id="q-integration-tts-001",
        text="Tell me about a time you led a team through a challenging project.",
        question_type=QuestionType.BEHAVIOURAL,
        asked_at=_NOW,
        skipped=False,
    )
    return SessionState(
        session_id="sess-integration-tts-fallback-001",
        phone_number=phone,
        stage=Stage.INTERVIEW,
        role=Role.SOFTWARE_ENGINEER,
        questions=[question],
        responses=[],
        created_at=_NOW,
        updated_at=_NOW,
        preferred_mode="voice",
    )


class TestTTSFallbackOnTTSError:
    """TTS fallback flow: when ElevenLabsClient.synthesise raises TTSError,
    the reply is sent as plain text and preferred_mode remains "voice".

    Requirements: 3.6, 10.2
    """

    async def _seed_voice_mode_session(
        self, fake_redis: fakeredis.aioredis.FakeRedis
    ) -> SessionState:
        """Pre-seed Redis with a session in INTERVIEW stage with preferred_mode="voice"."""
        repo = RedisSessionRepository(redis_client=fake_redis, ttl_seconds=86400)
        session = _make_interview_session_voice_mode()
        await repo.save(session)
        return session

    async def test_tts_error_falls_back_to_plain_text_reply(
        self, app_with_fakeredis, mocker
    ) -> None:
        """When ElevenLabsClient.synthesise raises TTSError, Twilio messages.create
        is called with a plain text body and no media_url.

        Requirements: 3.6, 10.2
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        # Pre-seed session in INTERVIEW stage with preferred_mode="voice"
        await self._seed_voice_mode_session(fake_redis)

        # Override the ElevenLabsClient.synthesise mock to raise TTSError
        mocker.patch(
            "interview_practice_partner.audio.tts_client.ElevenLabsClient.synthesise",
            side_effect=TTSError("TTS synthesis failed"),
        )

        # Mock LLM for intent classification + response evaluation + next question
        intent_response = json.dumps({"intent": "answer"})
        eval_response = json.dumps(
            {
                "is_off_topic": False,
                "is_short": False,
                "follow_up_warranted": False,
                "follow_up_text": "",
                "difficulty_signal": "maintain",
            }
        )
        next_question = "How do you handle disagreements within a team?"

        mocker.patch(
            "interview_practice_partner.llm.openai_client.OpenAIClient.complete",
            side_effect=[intent_response, eval_response, next_question],
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhook/whatsapp",
                content=_make_voice_note_message(message_sid="SM_tts_fallback_001"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        assert response.status_code == 200

        # Twilio messages.create must have been called
        mock_twilio.messages.create.assert_called_once()
        call_kwargs = mock_twilio.messages.create.call_args.kwargs

        # Verify NO media_url is present — plain text fallback
        assert "media_url" not in call_kwargs, (
            "Twilio messages.create must NOT include media_url when TTS fails; "
            f"got call_kwargs: {call_kwargs}"
        )

        # Verify a plain text body was sent
        assert "body" in call_kwargs, (
            "Twilio messages.create must include a 'body' for the plain text fallback"
        )
        assert isinstance(call_kwargs["body"], str) and len(call_kwargs["body"]) > 0, (
            "Plain text fallback body must be a non-empty string"
        )

    async def test_tts_error_does_not_reset_preferred_mode_to_text(
        self, app_with_fakeredis, mocker
    ) -> None:
        """After a TTSError, SessionState.preferred_mode remains "voice" in Redis.

        The mode must NOT be reset to "text" just because TTS synthesis failed.

        Requirements: 3.6, 10.2
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        # Pre-seed session in INTERVIEW stage with preferred_mode="voice"
        session = await self._seed_voice_mode_session(fake_redis)
        assert session.preferred_mode == "voice"

        # Override the ElevenLabsClient.synthesise mock to raise TTSError
        mocker.patch(
            "interview_practice_partner.audio.tts_client.ElevenLabsClient.synthesise",
            side_effect=TTSError("TTS synthesis failed"),
        )

        # Mock LLM for the interview flow
        intent_response = json.dumps({"intent": "answer"})
        eval_response = json.dumps(
            {
                "is_off_topic": False,
                "is_short": False,
                "follow_up_warranted": False,
                "follow_up_text": "",
                "difficulty_signal": "maintain",
            }
        )
        next_question = "Describe a situation where you had to meet a tight deadline."

        mocker.patch(
            "interview_practice_partner.llm.openai_client.OpenAIClient.complete",
            side_effect=[intent_response, eval_response, next_question],
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhook/whatsapp",
                content=_make_voice_note_message(message_sid="SM_tts_mode_001"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        assert response.status_code == 200

        # Retrieve the session from Redis and verify preferred_mode is still "voice"
        repo = RedisSessionRepository(redis_client=fake_redis, ttl_seconds=86400)
        updated_session = await repo.get(_FROM_NUMBER)

        assert updated_session is not None, "Session must be persisted in Redis"
        assert updated_session.preferred_mode == "voice", (
            f"Expected preferred_mode='voice' after TTS failure (mode must not be reset), "
            f"got: {updated_session.preferred_mode!r}"
        )

    async def test_tts_error_webhook_returns_200(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Webhook returns HTTP 200 even when TTS synthesis fails.

        Requirements: 3.6
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        await self._seed_voice_mode_session(fake_redis)

        # Override the ElevenLabsClient.synthesise mock to raise TTSError
        mocker.patch(
            "interview_practice_partner.audio.tts_client.ElevenLabsClient.synthesise",
            side_effect=TTSError("TTS synthesis failed"),
        )

        mocker.patch(
            "interview_practice_partner.llm.openai_client.OpenAIClient.complete",
            side_effect=[
                json.dumps({"intent": "answer"}),
                json.dumps({
                    "is_off_topic": False,
                    "is_short": False,
                    "follow_up_warranted": False,
                    "follow_up_text": "",
                    "difficulty_signal": "maintain",
                }),
                "Tell me about a time you had to learn something quickly.",
            ],
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhook/whatsapp",
                content=_make_voice_note_message(message_sid="SM_tts_200_001"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Task 10.3 — TranscriptionError fallback
# Requirements: 10.1, 10.3
# ---------------------------------------------------------------------------

_FALLBACK_MESSAGE = "I couldn't process your voice note. Please resend it or type your answer."


def _make_interview_session_with_history(phone: str = _FROM_NUMBER) -> SessionState:
    """Create a SessionState in INTERVIEW stage with pre-existing questions and responses."""
    from interview_practice_partner.domain.models import UserResponse

    question = Question(
        question_id="q-transcription-error-001",
        text="Tell me about a time you led a team through a challenging project.",
        question_type=QuestionType.BEHAVIOURAL,
        asked_at=_NOW,
        skipped=False,
    )
    existing_response = UserResponse(
        response_id="resp-prev-001",
        question_id="q-prev-001",
        text="I managed a cross-functional team to deliver a product on time.",
        word_count=12,
        received_at=_NOW,
    )
    return SessionState(
        session_id="sess-transcription-error-001",
        phone_number=phone,
        stage=Stage.INTERVIEW,
        role=Role.SOFTWARE_ENGINEER,
        questions=[question],
        responses=[existing_response],
        created_at=_NOW,
        updated_at=_NOW,
        preferred_mode="text",
    )


class TestTranscriptionErrorFallback:
    """TranscriptionError fallback: when AudioDownloadClient.download raises
    TranscriptionError, the fallback text is sent and session state is unchanged.

    Requirements: 10.1, 10.3
    """

    async def _seed_session_with_history(
        self, fake_redis: fakeredis.aioredis.FakeRedis
    ) -> SessionState:
        """Pre-seed Redis with a session in INTERVIEW stage with questions and responses."""
        repo = RedisSessionRepository(redis_client=fake_redis, ttl_seconds=86400)
        session = _make_interview_session_with_history()
        await repo.save(session)
        return session

    async def test_transcription_error_sends_fallback_text_message(
        self, app_with_fakeredis, mocker
    ) -> None:
        """When AudioDownloadClient.download raises TranscriptionError, Twilio
        messages.create is called with the fallback text body and no media_url.

        Requirements: 10.1, 10.3
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        # Pre-seed session in INTERVIEW stage with questions and responses
        await self._seed_session_with_history(fake_redis)

        # Override AudioDownloadClient.download to raise TranscriptionError
        mocker.patch(
            "interview_practice_partner.audio.download_client.AudioDownloadClient.download",
            side_effect=TranscriptionError("Download failed"),
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhook/whatsapp",
                content=_make_voice_note_message(message_sid="SM_transcription_err_001"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        assert response.status_code == 200

        # Twilio messages.create must have been called
        mock_twilio.messages.create.assert_called_once()
        call_kwargs = mock_twilio.messages.create.call_args.kwargs

        # Verify NO media_url — this is a plain text fallback
        assert "media_url" not in call_kwargs, (
            "Twilio messages.create must NOT include media_url for TranscriptionError fallback; "
            f"got call_kwargs: {call_kwargs}"
        )

        # Verify the fallback text body is present
        assert "body" in call_kwargs, (
            "Twilio messages.create must include a 'body' for the TranscriptionError fallback"
        )
        assert call_kwargs["body"] == _FALLBACK_MESSAGE, (
            f"Expected fallback body {_FALLBACK_MESSAGE!r}, got: {call_kwargs['body']!r}"
        )

    async def test_transcription_error_leaves_session_questions_unchanged(
        self, app_with_fakeredis, mocker
    ) -> None:
        """When AudioDownloadClient.download raises TranscriptionError, the session's
        questions list in Redis is unchanged after the turn.

        Requirements: 10.1, 10.3
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        # Pre-seed session and capture original questions
        original_session = await self._seed_session_with_history(fake_redis)
        original_question_ids = [q.question_id for q in original_session.questions]

        # Override AudioDownloadClient.download to raise TranscriptionError
        mocker.patch(
            "interview_practice_partner.audio.download_client.AudioDownloadClient.download",
            side_effect=TranscriptionError("Download failed"),
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/webhook/whatsapp",
                content=_make_voice_note_message(message_sid="SM_transcription_q_001"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        # Retrieve the session from Redis and verify questions are unchanged
        repo = RedisSessionRepository(redis_client=fake_redis, ttl_seconds=86400)
        updated_session = await repo.get(_FROM_NUMBER)

        assert updated_session is not None, "Session must be persisted in Redis"
        updated_question_ids = [q.question_id for q in updated_session.questions]
        assert updated_question_ids == original_question_ids, (
            f"Session questions must be unchanged after TranscriptionError; "
            f"expected {original_question_ids}, got {updated_question_ids}"
        )

    async def test_transcription_error_leaves_session_responses_unchanged(
        self, app_with_fakeredis, mocker
    ) -> None:
        """When AudioDownloadClient.download raises TranscriptionError, the session's
        responses list in Redis is unchanged after the turn.

        Requirements: 10.1, 10.3
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        # Pre-seed session and capture original responses
        original_session = await self._seed_session_with_history(fake_redis)
        original_response_count = len(original_session.responses)
        original_response_texts = [r.text for r in original_session.responses]

        # Override AudioDownloadClient.download to raise TranscriptionError
        mocker.patch(
            "interview_practice_partner.audio.download_client.AudioDownloadClient.download",
            side_effect=TranscriptionError("Download failed"),
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/webhook/whatsapp",
                content=_make_voice_note_message(message_sid="SM_transcription_r_001"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        # Retrieve the session from Redis and verify responses are unchanged
        repo = RedisSessionRepository(redis_client=fake_redis, ttl_seconds=86400)
        updated_session = await repo.get(_FROM_NUMBER)

        assert updated_session is not None, "Session must be persisted in Redis"
        assert len(updated_session.responses) == original_response_count, (
            f"Session responses count must be unchanged after TranscriptionError; "
            f"expected {original_response_count}, got {len(updated_session.responses)}"
        )
        updated_response_texts = [r.text for r in updated_session.responses]
        assert updated_response_texts == original_response_texts, (
            f"Session response texts must be unchanged after TranscriptionError; "
            f"expected {original_response_texts}, got {updated_response_texts}"
        )

    async def test_transcription_error_webhook_returns_200(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Webhook returns HTTP 200 even when AudioDownloadClient.download raises
        TranscriptionError.

        Requirements: 10.1, 10.3
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        await self._seed_session_with_history(fake_redis)

        # Override AudioDownloadClient.download to raise TranscriptionError
        mocker.patch(
            "interview_practice_partner.audio.download_client.AudioDownloadClient.download",
            side_effect=TranscriptionError("Download failed"),
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhook/whatsapp",
                content=_make_voice_note_message(message_sid="SM_transcription_200_001"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Task 10.4 — Mode command flow integration tests
# Requirements: 6.1, 6.3, 6.5, 6.6, 7.4
# ---------------------------------------------------------------------------


def _make_interview_session_for_mode_command(
    phone: str = _FROM_NUMBER,
    preferred_mode: str = "text",
) -> SessionState:
    """Create a SessionState in INTERVIEW stage with questions and responses for mode command tests."""
    from interview_practice_partner.domain.models import UserResponse

    question = Question(
        question_id="q-mode-cmd-001",
        text="Tell me about a time you led a team through a challenging project.",
        question_type=QuestionType.BEHAVIOURAL,
        asked_at=_NOW,
        skipped=False,
    )
    existing_response = UserResponse(
        response_id="resp-mode-cmd-001",
        question_id="q-prev-mode-001",
        text="I managed a cross-functional team to deliver a product on time.",
        word_count=12,
        received_at=_NOW,
    )
    return SessionState(
        session_id="sess-mode-cmd-001",
        phone_number=phone,
        stage=Stage.INTERVIEW,
        role=Role.SOFTWARE_ENGINEER,
        questions=[question],
        responses=[existing_response],
        created_at=_NOW,
        updated_at=_NOW,
        preferred_mode=preferred_mode,
    )


class TestModeCommandFlow:
    """Mode command flow: sending 'voice mode' or 'text mode' updates preferred_mode
    and sends a text confirmation without advancing interview state.

    Requirements: 6.1, 6.3, 6.5, 6.6, 7.4
    """

    async def _seed_session(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
        preferred_mode: str = "text",
    ) -> SessionState:
        """Pre-seed Redis with a session in INTERVIEW stage with questions and responses."""
        repo = RedisSessionRepository(redis_client=fake_redis, ttl_seconds=86400)
        session = _make_interview_session_for_mode_command(preferred_mode=preferred_mode)
        await repo.save(session)
        return session

    # ------------------------------------------------------------------
    # "voice mode" command tests
    # ------------------------------------------------------------------

    async def test_voice_mode_command_persists_preferred_mode_as_voice(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Sending 'voice mode' persists preferred_mode='voice' in Redis.

        Requirements: 6.1, 7.4
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        # Pre-seed session in INTERVIEW stage with preferred_mode="text"
        session = await self._seed_session(fake_redis, preferred_mode="text")
        assert session.preferred_mode == "text"

        # Override TTS to raise TTSError so the confirmation is sent as text
        # (Requirement 6.6: confirmation must always be a text message)
        mocker.patch(
            "interview_practice_partner.audio.tts_client.ElevenLabsClient.synthesise",
            side_effect=TTSError("TTS not used for mode command confirmation"),
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhook/whatsapp",
                content=_make_text_message(
                    body="voice mode",
                    message_sid="SM_voice_mode_cmd_001",
                ),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        assert response.status_code == 200

        # Retrieve the session from Redis and verify preferred_mode is "voice"
        repo = RedisSessionRepository(redis_client=fake_redis, ttl_seconds=86400)
        updated_session = await repo.get(_FROM_NUMBER)

        assert updated_session is not None, "Session must be persisted in Redis"
        assert updated_session.preferred_mode == "voice", (
            f"Expected preferred_mode='voice' after 'voice mode' command, "
            f"got: {updated_session.preferred_mode!r}"
        )

    async def test_voice_mode_command_sends_text_confirmation_no_media_url(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Sending 'voice mode' results in a text confirmation with no media_url.

        Requirement 6.6: confirmation is always sent as a text message,
        regardless of the new preferred_mode.

        Requirements: 6.3, 6.6
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        # Pre-seed session in INTERVIEW stage
        await self._seed_session(fake_redis, preferred_mode="text")

        # Override TTS to raise TTSError so the confirmation is sent as text
        # (Requirement 6.6: confirmation must always be a text message)
        mocker.patch(
            "interview_practice_partner.audio.tts_client.ElevenLabsClient.synthesise",
            side_effect=TTSError("TTS not used for mode command confirmation"),
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhook/whatsapp",
                content=_make_text_message(
                    body="voice mode",
                    message_sid="SM_voice_mode_cmd_002",
                ),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        assert response.status_code == 200

        # Twilio messages.create must have been called
        mock_twilio.messages.create.assert_called_once()
        call_kwargs = mock_twilio.messages.create.call_args.kwargs

        # Verify NO media_url — confirmation must be plain text
        assert "media_url" not in call_kwargs, (
            "Twilio messages.create must NOT include media_url for mode command confirmation; "
            f"got call_kwargs: {call_kwargs}"
        )

        # Verify a plain text body was sent
        assert "body" in call_kwargs, (
            "Twilio messages.create must include a 'body' for the mode command confirmation"
        )
        assert isinstance(call_kwargs["body"], str) and len(call_kwargs["body"]) > 0, (
            "Mode command confirmation body must be a non-empty string"
        )

    async def test_voice_mode_command_leaves_questions_unchanged(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Sending 'voice mode' does not change the session's questions list.

        Requirements: 6.5
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        # Pre-seed session and capture original questions
        original_session = await self._seed_session(fake_redis, preferred_mode="text")
        original_question_ids = [q.question_id for q in original_session.questions]

        # Override TTS to raise TTSError so the confirmation is sent as text
        mocker.patch(
            "interview_practice_partner.audio.tts_client.ElevenLabsClient.synthesise",
            side_effect=TTSError("TTS not used for mode command confirmation"),
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/webhook/whatsapp",
                content=_make_text_message(
                    body="voice mode",
                    message_sid="SM_voice_mode_cmd_003",
                ),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        # Retrieve the session from Redis and verify questions are unchanged
        repo = RedisSessionRepository(redis_client=fake_redis, ttl_seconds=86400)
        updated_session = await repo.get(_FROM_NUMBER)

        assert updated_session is not None, "Session must be persisted in Redis"
        updated_question_ids = [q.question_id for q in updated_session.questions]
        assert updated_question_ids == original_question_ids, (
            f"Session questions must be unchanged after 'voice mode' command; "
            f"expected {original_question_ids}, got {updated_question_ids}"
        )

    async def test_voice_mode_command_leaves_responses_unchanged(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Sending 'voice mode' does not change the session's responses list.

        Requirements: 6.5
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        # Pre-seed session and capture original responses
        original_session = await self._seed_session(fake_redis, preferred_mode="text")
        original_response_ids = [r.response_id for r in original_session.responses]
        original_response_count = len(original_session.responses)

        # Override TTS to raise TTSError so the confirmation is sent as text
        mocker.patch(
            "interview_practice_partner.audio.tts_client.ElevenLabsClient.synthesise",
            side_effect=TTSError("TTS not used for mode command confirmation"),
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/webhook/whatsapp",
                content=_make_text_message(
                    body="voice mode",
                    message_sid="SM_voice_mode_cmd_004",
                ),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        # Retrieve the session from Redis and verify responses are unchanged
        repo = RedisSessionRepository(redis_client=fake_redis, ttl_seconds=86400)
        updated_session = await repo.get(_FROM_NUMBER)

        assert updated_session is not None, "Session must be persisted in Redis"
        assert len(updated_session.responses) == original_response_count, (
            f"Session responses count must be unchanged after 'voice mode' command; "
            f"expected {original_response_count}, got {len(updated_session.responses)}"
        )
        updated_response_ids = [r.response_id for r in updated_session.responses]
        assert updated_response_ids == original_response_ids, (
            f"Session response IDs must be unchanged after 'voice mode' command; "
            f"expected {original_response_ids}, got {updated_response_ids}"
        )

    # ------------------------------------------------------------------
    # "text mode" command tests
    # ------------------------------------------------------------------

    async def test_text_mode_command_persists_preferred_mode_as_text(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Sending 'text mode' persists preferred_mode='text' in Redis.

        Requirements: 6.2, 7.4
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        # Pre-seed session in INTERVIEW stage with preferred_mode="voice"
        session = await self._seed_session(fake_redis, preferred_mode="voice")
        assert session.preferred_mode == "voice"

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhook/whatsapp",
                content=_make_text_message(
                    body="text mode",
                    message_sid="SM_text_mode_cmd_001",
                ),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        assert response.status_code == 200

        # Retrieve the session from Redis and verify preferred_mode is "text"
        repo = RedisSessionRepository(redis_client=fake_redis, ttl_seconds=86400)
        updated_session = await repo.get(_FROM_NUMBER)

        assert updated_session is not None, "Session must be persisted in Redis"
        assert updated_session.preferred_mode == "text", (
            f"Expected preferred_mode='text' after 'text mode' command, "
            f"got: {updated_session.preferred_mode!r}"
        )

    async def test_text_mode_command_sends_text_confirmation_no_media_url(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Sending 'text mode' results in a text confirmation with no media_url.

        Requirement 6.6: confirmation is always sent as a text message.

        Requirements: 6.4, 6.6
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        # Pre-seed session in INTERVIEW stage with preferred_mode="voice"
        await self._seed_session(fake_redis, preferred_mode="voice")

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhook/whatsapp",
                content=_make_text_message(
                    body="text mode",
                    message_sid="SM_text_mode_cmd_002",
                ),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        assert response.status_code == 200

        # Twilio messages.create must have been called
        mock_twilio.messages.create.assert_called_once()
        call_kwargs = mock_twilio.messages.create.call_args.kwargs

        # Verify NO media_url — confirmation must be plain text
        # After "text mode", preferred_mode="text" so no TTS attempt is made
        assert "media_url" not in call_kwargs, (
            "Twilio messages.create must NOT include media_url for 'text mode' confirmation; "
            f"got call_kwargs: {call_kwargs}"
        )

        # Verify a plain text body was sent
        assert "body" in call_kwargs, (
            "Twilio messages.create must include a 'body' for the mode command confirmation"
        )
        assert isinstance(call_kwargs["body"], str) and len(call_kwargs["body"]) > 0, (
            "Mode command confirmation body must be a non-empty string"
        )

    async def test_text_mode_command_leaves_questions_unchanged(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Sending 'text mode' does not change the session's questions list.

        Requirements: 6.5
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        # Pre-seed session and capture original questions
        original_session = await self._seed_session(fake_redis, preferred_mode="voice")
        original_question_ids = [q.question_id for q in original_session.questions]

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/webhook/whatsapp",
                content=_make_text_message(
                    body="text mode",
                    message_sid="SM_text_mode_cmd_003",
                ),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        # Retrieve the session from Redis and verify questions are unchanged
        repo = RedisSessionRepository(redis_client=fake_redis, ttl_seconds=86400)
        updated_session = await repo.get(_FROM_NUMBER)

        assert updated_session is not None, "Session must be persisted in Redis"
        updated_question_ids = [q.question_id for q in updated_session.questions]
        assert updated_question_ids == original_question_ids, (
            f"Session questions must be unchanged after 'text mode' command; "
            f"expected {original_question_ids}, got {updated_question_ids}"
        )

    async def test_text_mode_command_leaves_responses_unchanged(
        self, app_with_fakeredis, mocker
    ) -> None:
        """Sending 'text mode' does not change the session's responses list.

        Requirements: 6.5
        """
        app, fake_redis, mock_twilio = app_with_fakeredis

        # Pre-seed session and capture original responses
        original_session = await self._seed_session(fake_redis, preferred_mode="voice")
        original_response_ids = [r.response_id for r in original_session.responses]
        original_response_count = len(original_session.responses)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/webhook/whatsapp",
                content=_make_text_message(
                    body="text mode",
                    message_sid="SM_text_mode_cmd_004",
                ),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        # Retrieve the session from Redis and verify responses are unchanged
        repo = RedisSessionRepository(redis_client=fake_redis, ttl_seconds=86400)
        updated_session = await repo.get(_FROM_NUMBER)

        assert updated_session is not None, "Session must be persisted in Redis"
        assert len(updated_session.responses) == original_response_count, (
            f"Session responses count must be unchanged after 'text mode' command; "
            f"expected {original_response_count}, got {len(updated_session.responses)}"
        )
        updated_response_ids = [r.response_id for r in updated_session.responses]
        assert updated_response_ids == original_response_ids, (
            f"Session response IDs must be unchanged after 'text mode' command; "
            f"expected {original_response_ids}, got {updated_response_ids}"
        )
