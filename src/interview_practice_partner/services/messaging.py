"""TwilioMessagingService — outbound WhatsApp message sending with chunking support."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from twilio.rest import Client

from interview_practice_partner.config import Settings

logger = structlog.get_logger(__name__)

_MAX_CHARS = 4096


class TwilioMessagingService:
    """Sends outbound WhatsApp messages via the Twilio Messages API.

    Handles chunking of long messages at word boundaries to stay within
    WhatsApp's 4096-character message limit.
    """

    def __init__(self, client: "Client", settings: Settings) -> None:
        self._client = client
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_message(
        self,
        text: str,
        to_number: str,
        media_url: str | None = None,
    ) -> None:
        """Send *text* to *to_number*, splitting into chunks if necessary.

        When *media_url* is provided, the message is sent as a WhatsApp media
        message without chunking. The *text* body may be empty for media-only
        messages.
        When *media_url* is None, behaves identically to the existing
        implementation (chunked text send).

        Each chunk is sent as a separate Twilio message in sequential order.
        Delivery failures (status ``failed`` / ``undelivered``) are logged
        with ``phone_number`` and ``message_sid``.

        Args:
            text: The message body to send. May be empty when sending media.
            to_number: The destination E.164 WhatsApp number
                       (e.g. ``whatsapp:+447700900000``).
            media_url: Optional publicly accessible URL of the media file.
                       When provided, the message is sent as a media message
                       and chunking is skipped.
        """
        if media_url is not None:
            await self._send_single(text, to_number, media_url=media_url)
        else:
            chunks = self.send_chunked(text)
            for chunk in chunks:
                await self._send_single(chunk, to_number)

    @staticmethod
    def send_chunked(text: str, max_chars: int = _MAX_CHARS) -> list[str]:
        """Split *text* into chunks of at most *max_chars* characters.

        Splits are made at word boundaries (spaces) so that no word is
        broken mid-way.  If a single word is longer than *max_chars* it
        is placed in its own chunk (unavoidable hard split).

        Args:
            text: The text to split.
            max_chars: Maximum number of characters per chunk (default 4096).

        Returns:
            A list of non-empty chunk strings.  Returns ``[""]`` for an
            empty input so callers always receive at least one element.
        """
        if not text:
            return [""]

        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        current_start = 0
        text_len = len(text)

        while current_start < text_len:
            # Remaining text fits in one chunk
            if current_start + max_chars >= text_len:
                chunks.append(text[current_start:])
                break

            # Find the last space within the allowed window
            split_at = text.rfind(" ", current_start, current_start + max_chars)

            if split_at == -1 or split_at <= current_start:
                # No space found — hard-split at max_chars (single long word)
                split_at = current_start + max_chars
                chunks.append(text[current_start:split_at])
                current_start = split_at
            else:
                chunks.append(text[current_start:split_at])
                # Skip the space itself
                current_start = split_at + 1

        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _send_single(
        self,
        chunk: str,
        to_number: str,
        media_url: str | None = None,
    ) -> None:
        """Send a single chunk via the Twilio Messages API.

        Wraps the synchronous Twilio SDK call in ``asyncio.to_thread`` so
        it does not block the event loop.

        Args:
            chunk: The message body to send.
            to_number: The destination E.164 WhatsApp number.
            media_url: Optional publicly accessible URL of the media file.
                       When provided, passed as ``media_url=[media_url]`` to
                       the Twilio API.
        """
        log = logger.bind(to_number=to_number)

        if media_url is not None:
            log.info("twilio.send_message", body_length=len(chunk), media_url=media_url)
            message = await asyncio.to_thread(
                self._client.messages.create,
                from_=self._settings.twilio_whatsapp_number,
                to=to_number,
                body=chunk,
                media_url=[media_url],
            )
        else:
            log.info("twilio.send_message", body_length=len(chunk))
            message = await asyncio.to_thread(
                self._client.messages.create,
                from_=self._settings.twilio_whatsapp_number,
                to=to_number,
                body=chunk,
            )

        if message.status in ("failed", "undelivered"):
            log.error(
                "twilio.delivery_failure",
                phone_number=to_number,
                message_sid=message.sid,
                status=message.status,
            )
        else:
            log.info(
                "twilio.message_sent",
                message_sid=message.sid,
                status=message.status,
            )
