"""Audio download client for fetching Twilio voice note media."""

import structlog
import httpx

from interview_practice_partner.config import Settings
from interview_practice_partner.domain.exceptions import TranscriptionError

logger = structlog.get_logger(__name__)


class AudioDownloadClient:
    """Downloads voice note audio bytes from a Twilio media URL.

    Uses HTTP Basic Auth with the Twilio account SID and auth token,
    as required by Twilio's media access API.
    """

    def __init__(self, settings: Settings) -> None:
        self._auth = (settings.twilio_account_sid, settings.twilio_auth_token)

    async def download(self, media_url: str) -> bytes:
        """Download audio bytes from *media_url*.

        Args:
            media_url: The Twilio media URL from the inbound webhook payload
                       (e.g. ``https://api.twilio.com/2010-04-01/Accounts/.../Messages/.../Media/0``).

        Returns:
            Raw audio bytes.

        Raises:
            TranscriptionError: If the download fails (network error, 4xx/5xx response).
        """
        log = logger.bind(media_url=media_url)
        log.info("audio_download.start")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    media_url,
                    auth=self._auth,
                    follow_redirects=True,
                    timeout=30.0,
                )
                response.raise_for_status()
                audio_bytes = response.content
                log.info("audio_download.complete", size_bytes=len(audio_bytes))
                return audio_bytes
        except httpx.HTTPStatusError as exc:
            log.error(
                "audio_download.http_error",
                status_code=exc.response.status_code,
                message=str(exc),
            )
            raise TranscriptionError(
                f"Failed to download voice note: HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            log.error("audio_download.request_error", message=str(exc))
            raise TranscriptionError(
                f"Failed to download voice note: {exc}"
            ) from exc
