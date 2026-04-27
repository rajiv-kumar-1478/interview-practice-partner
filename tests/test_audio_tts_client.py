"""Unit tests for ElevenLabsClient.

Covers:
- synthesise calls ElevenLabs with the correct voice ID, model ID, and output format
- synthesise returns concatenated bytes from the async generator
- synthesise raises TTSError on ApiError
- synthesise logs char_count and latency_ms on success

Requirements: 3.2, 3.3, 3.4, 3.5, 3.7
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog.testing

from interview_practice_partner.audio.tts_client import ElevenLabsClient
from interview_practice_partner.domain.exceptions import TTSError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_settings(
    elevenlabs_api_key: str = "test-eleven-key",
    elevenlabs_voice_id: str = "test-voice-id",
    elevenlabs_model_id: str = "eleven_turbo_v2_5",
) -> MagicMock:
    """Return a minimal Settings stub with ElevenLabs fields."""
    settings = MagicMock()
    settings.elevenlabs_api_key = elevenlabs_api_key
    settings.elevenlabs_voice_id = elevenlabs_voice_id
    settings.elevenlabs_model_id = elevenlabs_model_id
    return settings


async def _async_chunks(chunks: list[bytes]):
    """Async generator that yields the given byte chunks."""
    for chunk in chunks:
        yield chunk


def make_client(
    elevenlabs_api_key: str = "test-eleven-key",
    elevenlabs_voice_id: str = "test-voice-id",
    elevenlabs_model_id: str = "eleven_turbo_v2_5",
) -> tuple[ElevenLabsClient, MagicMock]:
    """Build an ElevenLabsClient with a mocked AsyncElevenLabs instance.

    Returns (client, mock_async_elevenlabs_instance).
    """
    settings = make_settings(
        elevenlabs_api_key=elevenlabs_api_key,
        elevenlabs_voice_id=elevenlabs_voice_id,
        elevenlabs_model_id=elevenlabs_model_id,
    )

    with patch(
        "interview_practice_partner.audio.tts_client.AsyncElevenLabs"
    ) as MockAsyncElevenLabs:
        mock_instance = MagicMock()
        MockAsyncElevenLabs.return_value = mock_instance
        client = ElevenLabsClient(settings=settings)

    return client, mock_instance


# ===========================================================================
# API call correctness
# ===========================================================================


class TestSynthesiseAPICall:
    @pytest.mark.asyncio
    async def test_synthesise_calls_generate_with_correct_voice_id(self):
        """synthesise passes the configured voice ID to the ElevenLabs API."""
        client, mock_eleven = make_client(elevenlabs_voice_id="my-voice-123")

        mock_eleven.generate = AsyncMock(
            return_value=_async_chunks([b"audio"])
        )

        await client.synthesise("Hello world")

        call_kwargs = mock_eleven.generate.call_args.kwargs
        assert call_kwargs["voice"] == "my-voice-123"

    @pytest.mark.asyncio
    async def test_synthesise_calls_generate_with_correct_model_id(self):
        """synthesise passes the configured model ID to the ElevenLabs API."""
        client, mock_eleven = make_client(elevenlabs_model_id="eleven_turbo_v2_5")

        mock_eleven.generate = AsyncMock(
            return_value=_async_chunks([b"audio"])
        )

        await client.synthesise("Hello world")

        call_kwargs = mock_eleven.generate.call_args.kwargs
        assert call_kwargs["model"] == "eleven_turbo_v2_5"

    @pytest.mark.asyncio
    async def test_synthesise_calls_generate_with_mp3_output_format(self):
        """synthesise always requests mp3_44100_128 output format."""
        client, mock_eleven = make_client()

        mock_eleven.generate = AsyncMock(
            return_value=_async_chunks([b"audio"])
        )

        await client.synthesise("Hello world")

        call_kwargs = mock_eleven.generate.call_args.kwargs
        assert call_kwargs["output_format"] == "mp3_44100_128"

    @pytest.mark.asyncio
    async def test_synthesise_passes_text_to_generate(self):
        """synthesise passes the input text to the ElevenLabs generate call."""
        client, mock_eleven = make_client()

        mock_eleven.generate = AsyncMock(
            return_value=_async_chunks([b"audio"])
        )

        await client.synthesise("Tell me about yourself")

        call_kwargs = mock_eleven.generate.call_args.kwargs
        assert call_kwargs["text"] == "Tell me about yourself"

    @pytest.mark.asyncio
    async def test_synthesise_uses_custom_voice_id_from_settings(self):
        """synthesise uses the voice ID from settings, not a hardcoded value."""
        custom_voice = "21m00Tcm4TlvDq8ikWAM"
        client, mock_eleven = make_client(elevenlabs_voice_id=custom_voice)

        mock_eleven.generate = AsyncMock(
            return_value=_async_chunks([b"audio"])
        )

        await client.synthesise("Some text")

        call_kwargs = mock_eleven.generate.call_args.kwargs
        assert call_kwargs["voice"] == custom_voice

    @pytest.mark.asyncio
    async def test_synthesise_uses_custom_model_id_from_settings(self):
        """synthesise uses the model ID from settings, not a hardcoded value."""
        custom_model = "eleven_multilingual_v2"
        client, mock_eleven = make_client(elevenlabs_model_id=custom_model)

        mock_eleven.generate = AsyncMock(
            return_value=_async_chunks([b"audio"])
        )

        await client.synthesise("Some text")

        call_kwargs = mock_eleven.generate.call_args.kwargs
        assert call_kwargs["model"] == custom_model


# ===========================================================================
# Return value — concatenated bytes
# ===========================================================================


class TestSynthesiseReturnValue:
    @pytest.mark.asyncio
    async def test_synthesise_returns_concatenated_bytes_from_generator(self):
        """synthesise concatenates all chunks from the async generator."""
        client, mock_eleven = make_client()

        chunks = [b"chunk1", b"chunk2", b"chunk3"]
        mock_eleven.generate = AsyncMock(return_value=_async_chunks(chunks))

        result = await client.synthesise("Hello")

        assert result == b"chunk1chunk2chunk3"

    @pytest.mark.asyncio
    async def test_synthesise_returns_bytes_type(self):
        """synthesise returns a bytes object."""
        client, mock_eleven = make_client()

        mock_eleven.generate = AsyncMock(return_value=_async_chunks([b"audio data"]))

        result = await client.synthesise("Hello")

        assert isinstance(result, bytes)

    @pytest.mark.asyncio
    async def test_synthesise_returns_empty_bytes_for_empty_generator(self):
        """synthesise returns empty bytes when the generator yields nothing."""
        client, mock_eleven = make_client()

        mock_eleven.generate = AsyncMock(return_value=_async_chunks([]))

        result = await client.synthesise("Hello")

        assert result == b""

    @pytest.mark.asyncio
    async def test_synthesise_returns_single_chunk_unchanged(self):
        """synthesise returns the single chunk as-is when generator yields one item."""
        client, mock_eleven = make_client()

        audio_data = b"\xff\xfb\x90\x00" * 100  # fake MP3 header bytes
        mock_eleven.generate = AsyncMock(return_value=_async_chunks([audio_data]))

        result = await client.synthesise("Hello")

        assert result == audio_data

    @pytest.mark.asyncio
    async def test_synthesise_concatenates_many_chunks(self):
        """synthesise correctly concatenates a large number of chunks."""
        client, mock_eleven = make_client()

        chunks = [bytes([i]) for i in range(256)]
        mock_eleven.generate = AsyncMock(return_value=_async_chunks(chunks))

        result = await client.synthesise("Long text")

        assert result == bytes(range(256))


# ===========================================================================
# Error handling
# ===========================================================================


class TestSynthesiseErrorHandling:
    @pytest.mark.asyncio
    async def test_synthesise_raises_tts_error_on_api_error(self):
        """synthesise raises TTSError when the ElevenLabs API returns an ApiError."""
        from elevenlabs.core import ApiError

        client, mock_eleven = make_client()

        api_error = ApiError(status_code=429, body="Rate limit exceeded")
        mock_eleven.generate = AsyncMock(side_effect=api_error)

        with pytest.raises(TTSError) as exc_info:
            await client.synthesise("Hello world")

        assert "ElevenLabs API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tts_error_chains_original_api_error(self):
        """TTSError raised from ApiError has the original as __cause__."""
        from elevenlabs.core import ApiError

        client, mock_eleven = make_client()

        api_error = ApiError(status_code=500, body="Internal server error")
        mock_eleven.generate = AsyncMock(side_effect=api_error)

        with pytest.raises(TTSError) as exc_info:
            await client.synthesise("Hello world")

        assert exc_info.value.__cause__ is api_error

    @pytest.mark.asyncio
    async def test_synthesise_raises_tts_error_on_unexpected_exception(self):
        """synthesise raises TTSError on any unexpected exception."""
        client, mock_eleven = make_client()

        mock_eleven.generate = AsyncMock(
            side_effect=RuntimeError("Unexpected network failure")
        )

        with pytest.raises(TTSError) as exc_info:
            await client.synthesise("Hello world")

        assert "Unexpected TTS error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tts_error_is_exception_subclass(self):
        """TTSError can be caught as a plain Exception."""
        from elevenlabs.core import ApiError

        client, mock_eleven = make_client()

        api_error = ApiError(status_code=400, body="Bad request")
        mock_eleven.generate = AsyncMock(side_effect=api_error)

        with pytest.raises(Exception):
            await client.synthesise("Hello world")

    @pytest.mark.asyncio
    async def test_synthesise_raises_tts_error_on_api_error_with_various_status_codes(self):
        """synthesise raises TTSError for different HTTP error status codes."""
        from elevenlabs.core import ApiError

        client, mock_eleven = make_client()

        for status_code in [400, 401, 403, 404, 429, 500, 503]:
            api_error = ApiError(status_code=status_code, body=f"Error {status_code}")
            mock_eleven.generate = AsyncMock(side_effect=api_error)

            with pytest.raises(TTSError):
                await client.synthesise("Hello")


# ===========================================================================
# Structured logging
# ===========================================================================


class TestSynthesiseLogging:
    @pytest.mark.asyncio
    async def test_synthesise_logs_char_count_on_success(self):
        """synthesise logs char_count after a successful synthesis."""
        client, mock_eleven = make_client()

        mock_eleven.generate = AsyncMock(return_value=_async_chunks([b"audio"]))

        text = "Hello, this is a test message."
        with structlog.testing.capture_logs() as logs:
            await client.synthesise(text)

        # char_count is bound at the start and appears in all log events
        start_events = [e for e in logs if e.get("event") == "elevenlabs.synthesise_start"]
        assert len(start_events) == 1
        assert start_events[0].get("char_count") == len(text)

    @pytest.mark.asyncio
    async def test_synthesise_logs_latency_ms_on_success(self):
        """synthesise logs latency_ms after a successful synthesis."""
        client, mock_eleven = make_client()

        mock_eleven.generate = AsyncMock(return_value=_async_chunks([b"audio"]))

        with structlog.testing.capture_logs() as logs:
            await client.synthesise("Hello world")

        complete_events = [
            e for e in logs if e.get("event") == "elevenlabs.synthesise_complete"
        ]
        assert len(complete_events) == 1
        assert "latency_ms" in complete_events[0]
        assert isinstance(complete_events[0]["latency_ms"], int)

    @pytest.mark.asyncio
    async def test_synthesise_logs_start_event(self):
        """synthesise logs a start event before calling the API."""
        client, mock_eleven = make_client()

        mock_eleven.generate = AsyncMock(return_value=_async_chunks([b"audio"]))

        with structlog.testing.capture_logs() as logs:
            await client.synthesise("Hello world")

        start_events = [e for e in logs if e.get("event") == "elevenlabs.synthesise_start"]
        assert len(start_events) == 1

    @pytest.mark.asyncio
    async def test_synthesise_logs_complete_event(self):
        """synthesise logs a completion event after successful synthesis."""
        client, mock_eleven = make_client()

        mock_eleven.generate = AsyncMock(return_value=_async_chunks([b"audio"]))

        with structlog.testing.capture_logs() as logs:
            await client.synthesise("Hello world")

        complete_events = [
            e for e in logs if e.get("event") == "elevenlabs.synthesise_complete"
        ]
        assert len(complete_events) == 1

    @pytest.mark.asyncio
    async def test_synthesise_logs_latency_ms_is_non_negative(self):
        """synthesise logs a non-negative latency_ms value."""
        client, mock_eleven = make_client()

        mock_eleven.generate = AsyncMock(return_value=_async_chunks([b"audio"]))

        with structlog.testing.capture_logs() as logs:
            await client.synthesise("Hello world")

        complete_events = [
            e for e in logs if e.get("event") == "elevenlabs.synthesise_complete"
        ]
        assert complete_events[0]["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_synthesise_logs_correct_char_count_value(self):
        """synthesise logs the exact character count of the input text."""
        client, mock_eleven = make_client()

        mock_eleven.generate = AsyncMock(return_value=_async_chunks([b"audio"]))

        text = "A" * 512
        with structlog.testing.capture_logs() as logs:
            await client.synthesise(text)

        start_events = [e for e in logs if e.get("event") == "elevenlabs.synthesise_start"]
        assert start_events[0].get("char_count") == 512

    @pytest.mark.asyncio
    async def test_synthesise_logs_error_on_api_error(self):
        """synthesise logs an error event when the API raises an ApiError."""
        from elevenlabs.core import ApiError

        client, mock_eleven = make_client()

        api_error = ApiError(status_code=500, body="Internal server error")
        mock_eleven.generate = AsyncMock(side_effect=api_error)

        with structlog.testing.capture_logs() as logs:
            with pytest.raises(TTSError):
                await client.synthesise("Hello world")

        error_events = [e for e in logs if e.get("event") == "elevenlabs.api_error"]
        assert len(error_events) == 1

    @pytest.mark.asyncio
    async def test_synthesise_does_not_log_complete_on_error(self):
        """synthesise does not log a completion event when an error occurs."""
        from elevenlabs.core import ApiError

        client, mock_eleven = make_client()

        api_error = ApiError(status_code=500, body="Internal server error")
        mock_eleven.generate = AsyncMock(side_effect=api_error)

        with structlog.testing.capture_logs() as logs:
            with pytest.raises(TTSError):
                await client.synthesise("Hello world")

        complete_events = [
            e for e in logs if e.get("event") == "elevenlabs.synthesise_complete"
        ]
        assert len(complete_events) == 0
