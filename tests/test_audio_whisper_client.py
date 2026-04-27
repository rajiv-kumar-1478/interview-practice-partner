"""Unit tests for GroqWhisperClient.

Covers:
- transcribe calls the Groq API with the correct model, file tuple, and response_format="text"
- transcribe returns stripped text on success
- transcribe raises TranscriptionError on APIError
- transcribe logs audio_size_bytes and latency_ms on success

Requirements: 2.2, 2.4, 2.5, 2.7
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog.testing

from interview_practice_partner.audio.whisper_client import GroqWhisperClient
from interview_practice_partner.domain.exceptions import TranscriptionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_settings(
    groq_api_key: str = "test-groq-key",
    groq_whisper_model: str = "whisper-large-v3-turbo",
) -> MagicMock:
    """Return a minimal Settings stub with Groq fields."""
    settings = MagicMock()
    settings.groq_api_key = groq_api_key
    settings.groq_whisper_model = groq_whisper_model
    return settings


def make_client(
    groq_api_key: str = "test-groq-key",
    groq_whisper_model: str = "whisper-large-v3-turbo",
    mock_async_openai: MagicMock | None = None,
) -> tuple[GroqWhisperClient, MagicMock]:
    """Build a GroqWhisperClient with a mocked AsyncOpenAI instance.

    Returns (client, mock_async_openai_instance).
    """
    settings = make_settings(groq_api_key=groq_api_key, groq_whisper_model=groq_whisper_model)

    with patch(
        "interview_practice_partner.audio.whisper_client.AsyncOpenAI"
    ) as MockAsyncOpenAI:
        mock_instance = mock_async_openai or MagicMock()
        MockAsyncOpenAI.return_value = mock_instance
        client = GroqWhisperClient(settings=settings)

    return client, mock_instance


# ===========================================================================
# API call correctness
# ===========================================================================


class TestTranscribeAPICall:
    @pytest.mark.asyncio
    async def test_transcribe_calls_create_with_correct_model(self):
        """transcribe passes the configured model name to the Groq API."""
        client, mock_openai = make_client(groq_whisper_model="whisper-large-v3-turbo")

        mock_openai.audio.transcriptions.create = AsyncMock(return_value="Hello world")

        await client.transcribe(b"audio data", "voice_note.ogg")

        call_kwargs = mock_openai.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["model"] == "whisper-large-v3-turbo"

    @pytest.mark.asyncio
    async def test_transcribe_calls_create_with_response_format_text(self):
        """transcribe always passes response_format='text' to the Groq API."""
        client, mock_openai = make_client()

        mock_openai.audio.transcriptions.create = AsyncMock(return_value="Hello world")

        await client.transcribe(b"audio data", "voice_note.ogg")

        call_kwargs = mock_openai.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["response_format"] == "text"

    @pytest.mark.asyncio
    async def test_transcribe_passes_file_as_tuple(self):
        """transcribe wraps audio bytes in a (filename, BytesIO, mime) tuple."""
        client, mock_openai = make_client()

        mock_openai.audio.transcriptions.create = AsyncMock(return_value="Hello world")

        audio_bytes = b"raw audio bytes"
        filename = "voice_note.ogg"
        await client.transcribe(audio_bytes, filename)

        call_kwargs = mock_openai.audio.transcriptions.create.call_args.kwargs
        file_arg = call_kwargs["file"]

        # Must be a 3-tuple: (filename, BytesIO, mime_type)
        assert isinstance(file_arg, tuple)
        assert len(file_arg) == 3

        file_name, file_obj, mime_type = file_arg
        assert file_name == filename
        assert isinstance(file_obj, io.BytesIO)
        assert file_obj.read() == audio_bytes
        assert mime_type == "audio/ogg"

    @pytest.mark.asyncio
    async def test_transcribe_uses_custom_model_from_settings(self):
        """transcribe uses the model name from settings, not a hardcoded value."""
        custom_model = "whisper-large-v3"
        client, mock_openai = make_client(groq_whisper_model=custom_model)

        mock_openai.audio.transcriptions.create = AsyncMock(return_value="Transcribed text")

        await client.transcribe(b"audio", "audio.ogg")

        call_kwargs = mock_openai.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["model"] == custom_model


# ===========================================================================
# Return value
# ===========================================================================


class TestTranscribeReturnValue:
    @pytest.mark.asyncio
    async def test_transcribe_returns_stripped_text(self):
        """transcribe strips leading and trailing whitespace from the API response."""
        client, mock_openai = make_client()

        mock_openai.audio.transcriptions.create = AsyncMock(
            return_value="  Hello, this is a test.  \n"
        )

        result = await client.transcribe(b"audio", "voice_note.ogg")

        assert result == "Hello, this is a test."

    @pytest.mark.asyncio
    async def test_transcribe_returns_plain_string(self):
        """transcribe returns a plain str, not a wrapper object."""
        client, mock_openai = make_client()

        mock_openai.audio.transcriptions.create = AsyncMock(return_value="Plain text response")

        result = await client.transcribe(b"audio", "voice_note.ogg")

        assert isinstance(result, str)
        assert result == "Plain text response"

    @pytest.mark.asyncio
    async def test_transcribe_returns_empty_string_when_api_returns_empty(self):
        """transcribe returns an empty string when the API returns an empty response."""
        client, mock_openai = make_client()

        mock_openai.audio.transcriptions.create = AsyncMock(return_value="")

        result = await client.transcribe(b"audio", "voice_note.ogg")

        assert result == ""

    @pytest.mark.asyncio
    async def test_transcribe_strips_only_whitespace_not_content(self):
        """transcribe strips surrounding whitespace but preserves internal content."""
        client, mock_openai = make_client()

        inner_text = "Hello   world   with   spaces"
        mock_openai.audio.transcriptions.create = AsyncMock(
            return_value=f"  {inner_text}  "
        )

        result = await client.transcribe(b"audio", "voice_note.ogg")

        assert result == inner_text


# ===========================================================================
# Error handling
# ===========================================================================


class TestTranscribeErrorHandling:
    @pytest.mark.asyncio
    async def test_transcribe_raises_transcription_error_on_api_error(self):
        """transcribe raises TranscriptionError when the Groq API returns an APIError."""
        from openai import APIError

        client, mock_openai = make_client()

        # APIError requires request and body arguments
        api_error = APIError(
            message="API rate limit exceeded",
            request=MagicMock(),
            body=None,
        )
        mock_openai.audio.transcriptions.create = AsyncMock(side_effect=api_error)

        with pytest.raises(TranscriptionError) as exc_info:
            await client.transcribe(b"audio", "voice_note.ogg")

        assert "Groq Whisper API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_transcription_error_chains_original_api_error(self):
        """TranscriptionError raised from APIError has the original as __cause__."""
        from openai import APIError

        client, mock_openai = make_client()

        api_error = APIError(
            message="Service unavailable",
            request=MagicMock(),
            body=None,
        )
        mock_openai.audio.transcriptions.create = AsyncMock(side_effect=api_error)

        with pytest.raises(TranscriptionError) as exc_info:
            await client.transcribe(b"audio", "voice_note.ogg")

        assert exc_info.value.__cause__ is api_error

    @pytest.mark.asyncio
    async def test_transcribe_raises_transcription_error_on_unexpected_exception(self):
        """transcribe raises TranscriptionError on any unexpected exception."""
        client, mock_openai = make_client()

        mock_openai.audio.transcriptions.create = AsyncMock(
            side_effect=RuntimeError("Unexpected network failure")
        )

        with pytest.raises(TranscriptionError) as exc_info:
            await client.transcribe(b"audio", "voice_note.ogg")

        assert "Unexpected transcription error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_transcription_error_is_exception_subclass(self):
        """TranscriptionError can be caught as a plain Exception."""
        from openai import APIError

        client, mock_openai = make_client()

        api_error = APIError(
            message="Bad request",
            request=MagicMock(),
            body=None,
        )
        mock_openai.audio.transcriptions.create = AsyncMock(side_effect=api_error)

        with pytest.raises(Exception):
            await client.transcribe(b"audio", "voice_note.ogg")


# ===========================================================================
# Structured logging
# ===========================================================================


class TestTranscribeLogging:
    @pytest.mark.asyncio
    async def test_transcribe_logs_audio_size_bytes_on_success(self):
        """transcribe logs audio_size_bytes after a successful transcription."""
        client, mock_openai = make_client()

        mock_openai.audio.transcriptions.create = AsyncMock(return_value="Hello world")

        audio_bytes = b"x" * 1024  # 1 KB

        with structlog.testing.capture_logs() as logs:
            await client.transcribe(audio_bytes, "voice_note.ogg")

        # Find the completion log event
        complete_events = [e for e in logs if e.get("event") == "groq_whisper.transcribe_complete"]
        assert len(complete_events) == 1

    @pytest.mark.asyncio
    async def test_transcribe_logs_latency_ms_on_success(self):
        """transcribe logs latency_ms after a successful transcription."""
        client, mock_openai = make_client()

        mock_openai.audio.transcriptions.create = AsyncMock(return_value="Hello world")

        with structlog.testing.capture_logs() as logs:
            await client.transcribe(b"audio data", "voice_note.ogg")

        complete_events = [e for e in logs if e.get("event") == "groq_whisper.transcribe_complete"]
        assert len(complete_events) == 1
        assert "latency_ms" in complete_events[0]
        assert isinstance(complete_events[0]["latency_ms"], int)

    @pytest.mark.asyncio
    async def test_transcribe_logs_start_event(self):
        """transcribe logs a start event before calling the API."""
        client, mock_openai = make_client()

        mock_openai.audio.transcriptions.create = AsyncMock(return_value="Hello world")

        with structlog.testing.capture_logs() as logs:
            await client.transcribe(b"audio data", "voice_note.ogg")

        start_events = [e for e in logs if e.get("event") == "groq_whisper.transcribe_start"]
        assert len(start_events) == 1

    @pytest.mark.asyncio
    async def test_transcribe_logs_audio_size_bytes_value(self):
        """transcribe logs the correct audio_size_bytes value."""
        client, mock_openai = make_client()

        mock_openai.audio.transcriptions.create = AsyncMock(return_value="Hello world")

        audio_bytes = b"y" * 2048  # 2 KB

        with structlog.testing.capture_logs() as logs:
            await client.transcribe(audio_bytes, "voice_note.ogg")

        # audio_size_bytes is bound at the start and appears in all log events
        start_events = [e for e in logs if e.get("event") == "groq_whisper.transcribe_start"]
        assert len(start_events) == 1
        assert start_events[0].get("audio_size_bytes") == 2048

    @pytest.mark.asyncio
    async def test_transcribe_logs_error_on_api_error(self):
        """transcribe logs an error event when the API raises an APIError."""
        from openai import APIError

        client, mock_openai = make_client()

        api_error = APIError(
            message="Internal server error",
            request=MagicMock(),
            body=None,
        )
        mock_openai.audio.transcriptions.create = AsyncMock(side_effect=api_error)

        with structlog.testing.capture_logs() as logs:
            with pytest.raises(TranscriptionError):
                await client.transcribe(b"audio", "voice_note.ogg")

        error_events = [e for e in logs if e.get("event") == "groq_whisper.api_error"]
        assert len(error_events) == 1

    @pytest.mark.asyncio
    async def test_transcribe_latency_ms_is_non_negative(self):
        """transcribe logs a non-negative latency_ms value."""
        client, mock_openai = make_client()

        mock_openai.audio.transcriptions.create = AsyncMock(return_value="Result")

        with structlog.testing.capture_logs() as logs:
            await client.transcribe(b"audio", "voice_note.ogg")

        complete_events = [e for e in logs if e.get("event") == "groq_whisper.transcribe_complete"]
        assert complete_events[0]["latency_ms"] >= 0
