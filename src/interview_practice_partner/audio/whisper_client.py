"""WhisperClient ABC and GroqWhisperClient implementation for speech-to-text transcription."""

import io
import time
from abc import ABC, abstractmethod

import structlog
from openai import AsyncOpenAI, APIError

from interview_practice_partner.config import Settings
from interview_practice_partner.domain.exceptions import TranscriptionError

logger = structlog.get_logger(__name__)


class WhisperClient(ABC):
    """Abstract base class for speech-to-text provider clients.

    Follows the same pattern as LLMClient. Implementations must provide
    an async ``transcribe`` method that accepts raw audio bytes and returns
    the transcribed text as a plain string.
    """

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        """Transcribe *audio_bytes* and return the plain-text result.

        Args:
            audio_bytes: Raw audio file bytes (e.g. OGG, MP3, WAV).
            filename: A filename hint including the file extension
                      (e.g. ``"voice_note.ogg"``). Used by the Whisper API
                      to determine the audio format.

        Returns:
            The transcribed text as a plain string. May be empty if the
            audio contains no recognisable speech.

        Raises:
            TranscriptionError: If the API call fails for any reason.
        """
        ...


class GroqWhisperClient(WhisperClient):
    """Groq Whisper API implementation of WhisperClient.

    Uses the OpenAI-compatible SDK pointed at the Groq API base URL.
    API key and model are loaded from Settings.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self._model = settings.groq_whisper_model

    async def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        """Transcribe audio bytes via the Groq Whisper API.

        Args:
            audio_bytes: Raw audio bytes.
            filename: Filename hint with extension (e.g. ``"audio.ogg"``).

        Returns:
            Transcribed text string (may be empty).

        Raises:
            TranscriptionError: On any API error.
        """
        log = logger.bind(audio_size_bytes=len(audio_bytes), filename=filename)
        log.info("groq_whisper.transcribe_start")
        start = time.monotonic()

        try:
            audio_file = (filename, io.BytesIO(audio_bytes), "audio/ogg")
            response = await self._client.audio.transcriptions.create(
                model=self._model,
                file=audio_file,
                response_format="text",
            )
            latency_ms = int((time.monotonic() - start) * 1000)
            log.info(
                "groq_whisper.transcribe_complete",
                latency_ms=latency_ms,
                transcript_length=len(response),
            )
            return response.strip()
        except APIError as exc:
            log.error(
                "groq_whisper.api_error",
                status_code=getattr(exc, "status_code", None),
                message=str(exc),
            )
            raise TranscriptionError(f"Groq Whisper API error: {exc}") from exc
        except Exception as exc:
            log.error("groq_whisper.unexpected_error", message=str(exc))
            raise TranscriptionError(f"Unexpected transcription error: {exc}") from exc
