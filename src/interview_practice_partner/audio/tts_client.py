"""ElevenLabs text-to-speech client."""

import time

import structlog
from elevenlabs import AsyncElevenLabs
from elevenlabs.core import ApiError

from interview_practice_partner.config import Settings
from interview_practice_partner.domain.exceptions import TTSError

logger = structlog.get_logger(__name__)


class ElevenLabsClient:
    """ElevenLabs text-to-speech client.

    Synthesises text into MP3 audio bytes using the ElevenLabs API.
    API key, voice ID, and model ID are loaded from Settings.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = AsyncElevenLabs(api_key=settings.elevenlabs_api_key)
        self._voice_id = settings.elevenlabs_voice_id
        self._model_id = settings.elevenlabs_model_id

    async def synthesise(self, text: str) -> bytes:
        """Synthesise *text* into MP3 audio bytes.

        Args:
            text: The text to synthesise. Must be non-empty.

        Returns:
            Raw MP3 audio bytes.

        Raises:
            TTSError: On any API error.
        """
        log = logger.bind(char_count=len(text), voice_id=self._voice_id)
        log.info("elevenlabs.synthesise_start")
        start = time.monotonic()

        try:
            audio_generator = await self._client.generate(
                text=text,
                voice=self._voice_id,
                model=self._model_id,
                output_format="mp3_44100_128",
            )
            # Collect all chunks from the async generator
            chunks: list[bytes] = []
            async for chunk in audio_generator:
                chunks.append(chunk)
            audio_bytes = b"".join(chunks)

            latency_ms = int((time.monotonic() - start) * 1000)
            log.info(
                "elevenlabs.synthesise_complete",
                latency_ms=latency_ms,
                audio_size_bytes=len(audio_bytes),
            )
            return audio_bytes
        except ApiError as exc:
            log.error(
                "elevenlabs.api_error",
                status_code=getattr(exc, "status_code", None),
                message=str(exc),
            )
            raise TTSError(f"ElevenLabs API error: {exc}") from exc
        except Exception as exc:
            log.error("elevenlabs.unexpected_error", message=str(exc))
            raise TTSError(f"Unexpected TTS error: {exc}") from exc
