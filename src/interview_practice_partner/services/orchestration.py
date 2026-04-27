"""MessageOrchestrationService — single entry point for inbound message processing.

Coordinates the full request flow for each inbound WhatsApp message:
  1. Check idempotency (MessageSid in Redis) — if duplicate, return immediately
  2. Load SessionState from Redis (or create new if not found)
  3. Call SessionService.transition(session, message.body) → (reply, updated_session)
  4. Persist updated SessionState to Redis
  5. If preferred_mode is "voice": synthesise TTS audio, write to disk, send via media_url
     Otherwise: send reply as plain text
  6. Mark MessageSid as processed in Redis (idempotency)

Error handling:
  - SessionStoreUnavailableError → send informative error message, return
  - LLMError → send safe fallback message, return
  - TTSError → fall back to plain text reply, log the fallback event
  - Any other exception → log full stack trace, send "Something went wrong" message
"""

from __future__ import annotations

import pathlib
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
import structlog.contextvars

from interview_practice_partner.api.schemas import InboundMessage
from interview_practice_partner.audio.tts_client import ElevenLabsClient
from interview_practice_partner.config import Settings
from interview_practice_partner.domain.exceptions import (
    LLMError,
    SessionStoreUnavailableError,
    TTSError,
)
from interview_practice_partner.domain.enums import Stage
from interview_practice_partner.domain.models import SessionState
from interview_practice_partner.repositories.base import SessionRepository
from interview_practice_partner.repositories.idempotency import IdempotencyRepository
from interview_practice_partner.services.messaging import TwilioMessagingService
from interview_practice_partner.services.session import SessionService

# Directory where TTS audio files are written (same as media router)
_MEDIA_DIR = pathlib.Path("/tmp/ipp_media")
_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# User-facing error messages
# ---------------------------------------------------------------------------

_MSG_SESSION_UNAVAILABLE = (
    "I'm having trouble accessing your session right now. "
    "Please try again in a moment."
)

_MSG_LLM_ERROR = (
    "I'm having a technical issue. Please send your message again."
)

_MSG_UNEXPECTED_ERROR = (
    "Something went wrong on my end. Please try again."
)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _new_session(phone_number: str) -> SessionState:
    """Create a brand-new SessionState for *phone_number*."""
    now = _now()
    return SessionState(
        session_id=str(uuid.uuid4()),
        phone_number=phone_number,
        stage=Stage.INIT,
        created_at=now,
        updated_at=now,
    )


class MessageOrchestrationService:
    """Single entry point for processing inbound WhatsApp messages.

    Coordinates idempotency checking, session loading, state machine
    transitions, session persistence, and outbound message delivery.

    Args:
        session_repository: Repository for loading and persisting SessionState.
        idempotency_repository: Repository for MessageSid deduplication.
        session_service: Service owning the session state machine.
        messaging_service: Service for sending outbound WhatsApp messages.
        tts_client: Optional ElevenLabs TTS client for voice replies.
        settings: Application settings (used for media_base_url).
    """

    def __init__(
        self,
        session_repository: SessionRepository,
        idempotency_repository: IdempotencyRepository,
        session_service: SessionService,
        messaging_service: TwilioMessagingService,
        tts_client: Optional[ElevenLabsClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._session_repo = session_repository
        self._idempotency_repo = idempotency_repository
        self._session_service = session_service
        self._messaging = messaging_service
        self._tts = tts_client
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle(self, message: InboundMessage) -> None:
        """Process a single inbound WhatsApp message end-to-end.

        This is the single entry point for all inbound message processing.
        It coordinates the full request flow and handles all error cases.

        Args:
            message: The parsed inbound message DTO from the Twilio webhook.
        """
        # Bind correlation_id to structlog context vars for this request.
        # All log entries within this call will automatically include it.
        structlog.contextvars.bind_contextvars(correlation_id=message.message_sid)

        log = logger.bind(
            from_number=message.from_number,
            message_sid=message.message_sid,
        )

        try:
            await self._process(message, log)
        except SessionStoreUnavailableError as exc:
            log.error("session_store_unavailable", error=str(exc))
            await self._safe_send(_MSG_SESSION_UNAVAILABLE, message.from_number, log)
        except LLMError as exc:
            log.error("llm_error", error=str(exc))
            await self._safe_send(_MSG_LLM_ERROR, message.from_number, log)
        except Exception:
            log.exception("unhandled_exception")
            await self._safe_send(_MSG_UNEXPECTED_ERROR, message.from_number, log)
        finally:
            structlog.contextvars.clear_contextvars()

    # ------------------------------------------------------------------
    # Internal processing pipeline
    # ------------------------------------------------------------------

    async def _process(self, message: InboundMessage, log: structlog.BoundLogger) -> None:
        """Execute the full message processing pipeline.

        Raises:
            SessionStoreUnavailableError: If Redis is unreachable.
            LLMError: If the LLM call fails after all retries.
        """
        # Step 1: Idempotency check — suppress duplicate Twilio deliveries
        is_duplicate = await self._idempotency_repo.is_processed(message.message_sid)
        if is_duplicate:
            log.info("duplicate_message_suppressed")
            return

        # Step 2: Load session state (or create a new one)
        session = await self._session_repo.get(message.from_number)
        if session is None:
            log.info("new_session_created")
            session = _new_session(message.from_number)

        log.info(
            "session_loaded",
            session_id=session.session_id,
            stage=session.stage.value,
        )

        # Step 3: Run the state machine transition
        reply, updated_session = await self._session_service.transition(
            session, message.body,
            num_media=message.num_media,
            media_content_type=message.media_content_type_0,
            media_url=message.media_url_0,
        )

        log.info(
            "transition_complete",
            new_stage=updated_session.stage.value,
            reply_length=len(reply),
        )

        # Step 4: Persist the updated session state
        await self._session_repo.save(updated_session)
        log.info("session_persisted", session_id=updated_session.session_id)

        # Step 5: Send the reply to the user
        # If preferred_mode is "voice" and TTS is available, synthesise audio reply
        media_url: str | None = None
        if updated_session.preferred_mode == "voice" and self._tts is not None:
            media_url = await self._synthesise_and_get_url(reply, log)

        await self._messaging.send_message(reply, message.from_number, media_url=media_url)
        log.info("reply_sent")

        # Step 6: Mark the MessageSid as processed (idempotency)
        await self._idempotency_repo.mark_processed(message.message_sid)
        log.info("message_sid_marked_processed")

    # ------------------------------------------------------------------
    # Safe send helper
    # ------------------------------------------------------------------

    async def _safe_send(
        self,
        text: str,
        to_number: str,
        log: structlog.BoundLogger,
    ) -> None:
        """Attempt to send *text* to *to_number*, swallowing any send errors.

        Used in error handlers to ensure we always attempt to notify the user
        even when the primary processing pipeline has failed.

        Args:
            text: The error/fallback message to send.
            to_number: The destination WhatsApp phone number.
            log: The bound structlog logger for this request.
        """
        try:
            await self._messaging.send_message(text, to_number)
        except Exception:
            log.exception("error_message_send_failed", to_number=to_number)

    # ------------------------------------------------------------------
    # TTS synthesis helper
    # ------------------------------------------------------------------

    async def _synthesise_and_get_url(
        self,
        text: str,
        log: structlog.BoundLogger,
    ) -> str | None:
        """Synthesise *text* to MP3 audio and return the public media URL.

        Writes the audio bytes to ``/tmp/ipp_media/{uuid}.mp3`` and returns
        the publicly accessible URL for the file.

        On ``TTSError``, logs the fallback event and returns ``None`` so the
        caller falls back to sending a plain text reply.

        Args:
            text: The reply text to synthesise.
            log: The bound structlog logger for this request.

        Returns:
            The public media URL string, or ``None`` if synthesis failed.
        """
        assert self._tts is not None  # guarded by caller
        try:
            audio_bytes = await self._tts.synthesise(text)
        except TTSError as exc:
            log.error(
                "tts_fallback_to_text",
                error=str(exc),
            )
            return None

        # Write audio bytes to a UUID-named temp file
        filename = f"{uuid.uuid4()}.mp3"
        file_path = _MEDIA_DIR / filename
        file_path.write_bytes(audio_bytes)
        log.info("tts_audio_written", filename=filename, size_bytes=len(audio_bytes))

        # Construct the public URL
        base_url = (self._settings.media_base_url if self._settings else "").rstrip("/")
        media_url = f"{base_url}/media/{filename}"
        return media_url
