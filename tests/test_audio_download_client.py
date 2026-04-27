"""Unit tests for AudioDownloadClient.

Covers:
- download performs GET with Basic Auth, follow_redirects=True, timeout=30.0
- download returns response.content on success
- download raises TranscriptionError with HTTP status code on HTTPStatusError
- download raises TranscriptionError on RequestError
- download logs media_url and size_bytes on success
- download logs status_code and message on HTTP error

Requirements: 2.1, 10.3
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import structlog.testing

from interview_practice_partner.audio.download_client import AudioDownloadClient
from interview_practice_partner.domain.exceptions import TranscriptionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_settings(
    twilio_account_sid: str = "ACtest123",
    twilio_auth_token: str = "authtoken456",
) -> MagicMock:
    """Return a minimal Settings stub with Twilio fields."""
    settings = MagicMock()
    settings.twilio_account_sid = twilio_account_sid
    settings.twilio_auth_token = twilio_auth_token
    return settings


def make_client(
    twilio_account_sid: str = "ACtest123",
    twilio_auth_token: str = "authtoken456",
) -> AudioDownloadClient:
    """Build an AudioDownloadClient with stub settings."""
    settings = make_settings(
        twilio_account_sid=twilio_account_sid,
        twilio_auth_token=twilio_auth_token,
    )
    return AudioDownloadClient(settings=settings)


def make_mock_response(content: bytes = b"audio data", status_code: int = 200) -> MagicMock:
    """Return a mock httpx.Response."""
    response = MagicMock()
    response.content = content
    response.status_code = status_code
    response.raise_for_status = MagicMock()
    return response


# ===========================================================================
# HTTP request correctness
# ===========================================================================


class TestDownloadRequestBehaviour:
    @pytest.mark.asyncio
    async def test_download_uses_basic_auth_from_settings(self):
        """download passes (account_sid, auth_token) as Basic Auth."""
        client = make_client(twilio_account_sid="ACabc", twilio_auth_token="tokxyz")

        mock_response = make_mock_response(b"bytes")
        mock_get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = mock_get
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await client.download("https://api.twilio.com/media/0")

        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs["auth"] == ("ACabc", "tokxyz")

    @pytest.mark.asyncio
    async def test_download_uses_follow_redirects_true(self):
        """download passes follow_redirects=True to the GET request."""
        client = make_client()

        mock_response = make_mock_response(b"bytes")
        mock_get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = mock_get
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await client.download("https://api.twilio.com/media/0")

        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs["follow_redirects"] is True

    @pytest.mark.asyncio
    async def test_download_uses_timeout_30(self):
        """download passes timeout=30.0 to the GET request."""
        client = make_client()

        mock_response = make_mock_response(b"bytes")
        mock_get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = mock_get
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await client.download("https://api.twilio.com/media/0")

        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs["timeout"] == 30.0

    @pytest.mark.asyncio
    async def test_download_calls_raise_for_status(self):
        """download calls raise_for_status() on the response."""
        client = make_client()

        mock_response = make_mock_response(b"bytes")
        mock_get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = mock_get
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await client.download("https://api.twilio.com/media/0")

        mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_passes_media_url_as_positional_arg(self):
        """download passes the media_url as the first positional argument to GET."""
        client = make_client()
        url = "https://api.twilio.com/2010-04-01/Accounts/AC123/Messages/MM456/Media/0"

        mock_response = make_mock_response(b"bytes")
        mock_get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = mock_get
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await client.download(url)

        call_args = mock_get.call_args
        assert call_args.args[0] == url


# ===========================================================================
# Return value
# ===========================================================================


class TestDownloadReturnValue:
    @pytest.mark.asyncio
    async def test_download_returns_response_content(self):
        """download returns the raw bytes from response.content."""
        client = make_client()
        expected_bytes = b"\x00\x01\x02\x03audio content"

        mock_response = make_mock_response(expected_bytes)
        mock_get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = mock_get
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.download("https://api.twilio.com/media/0")

        assert result == expected_bytes

    @pytest.mark.asyncio
    async def test_download_returns_bytes_type(self):
        """download returns a bytes object."""
        client = make_client()

        mock_response = make_mock_response(b"some audio")
        mock_get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = mock_get
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client.download("https://api.twilio.com/media/0")

        assert isinstance(result, bytes)


# ===========================================================================
# Error handling
# ===========================================================================


class TestDownloadErrorHandling:
    @pytest.mark.asyncio
    async def test_download_raises_transcription_error_on_http_status_error(self):
        """download raises TranscriptionError when the server returns a 4xx/5xx."""
        client = make_client()

        mock_response = MagicMock()
        mock_response.status_code = 403
        http_error = httpx.HTTPStatusError(
            message="403 Forbidden",
            request=MagicMock(),
            response=mock_response,
        )

        mock_get = AsyncMock(side_effect=http_error)

        with patch("httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = mock_get
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(TranscriptionError) as exc_info:
                await client.download("https://api.twilio.com/media/0")

        assert "403" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_download_raises_transcription_error_on_request_error(self):
        """download raises TranscriptionError on network/connection errors."""
        client = make_client()

        request_error = httpx.ConnectError("Connection refused")

        mock_get = AsyncMock(side_effect=request_error)

        with patch("httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = mock_get
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(TranscriptionError):
                await client.download("https://api.twilio.com/media/0")

    @pytest.mark.asyncio
    async def test_download_chains_http_status_error_as_cause(self):
        """TranscriptionError raised from HTTPStatusError has the original as __cause__."""
        client = make_client()

        mock_response = MagicMock()
        mock_response.status_code = 404
        http_error = httpx.HTTPStatusError(
            message="404 Not Found",
            request=MagicMock(),
            response=mock_response,
        )

        mock_get = AsyncMock(side_effect=http_error)

        with patch("httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = mock_get
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(TranscriptionError) as exc_info:
                await client.download("https://api.twilio.com/media/0")

        assert exc_info.value.__cause__ is http_error

    @pytest.mark.asyncio
    async def test_download_chains_request_error_as_cause(self):
        """TranscriptionError raised from RequestError has the original as __cause__."""
        client = make_client()

        request_error = httpx.TimeoutException("Request timed out")

        mock_get = AsyncMock(side_effect=request_error)

        with patch("httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = mock_get
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(TranscriptionError) as exc_info:
                await client.download("https://api.twilio.com/media/0")

        assert exc_info.value.__cause__ is request_error

    @pytest.mark.asyncio
    async def test_download_http_error_includes_status_code_in_message(self):
        """TranscriptionError message includes the HTTP status code."""
        client = make_client()

        mock_response = MagicMock()
        mock_response.status_code = 401
        http_error = httpx.HTTPStatusError(
            message="401 Unauthorized",
            request=MagicMock(),
            response=mock_response,
        )

        mock_get = AsyncMock(side_effect=http_error)

        with patch("httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = mock_get
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(TranscriptionError) as exc_info:
                await client.download("https://api.twilio.com/media/0")

        assert "HTTP 401" in str(exc_info.value)


# ===========================================================================
# Structured logging
# ===========================================================================


class TestDownloadLogging:
    @pytest.mark.asyncio
    async def test_download_logs_start_event(self):
        """download logs an audio_download.start event before the request."""
        client = make_client()

        mock_response = make_mock_response(b"audio")
        mock_get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = mock_get
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            with structlog.testing.capture_logs() as logs:
                await client.download("https://api.twilio.com/media/0")

        start_events = [e for e in logs if e.get("event") == "audio_download.start"]
        assert len(start_events) == 1

    @pytest.mark.asyncio
    async def test_download_logs_complete_with_size_bytes(self):
        """download logs audio_download.complete with size_bytes on success."""
        client = make_client()
        audio_content = b"x" * 512

        mock_response = make_mock_response(audio_content)
        mock_get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = mock_get
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            with structlog.testing.capture_logs() as logs:
                await client.download("https://api.twilio.com/media/0")

        complete_events = [e for e in logs if e.get("event") == "audio_download.complete"]
        assert len(complete_events) == 1
        assert complete_events[0]["size_bytes"] == 512

    @pytest.mark.asyncio
    async def test_download_logs_media_url_in_start_event(self):
        """download logs the media_url in the start event."""
        client = make_client()
        url = "https://api.twilio.com/2010-04-01/Accounts/AC123/Media/0"

        mock_response = make_mock_response(b"audio")
        mock_get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = mock_get
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            with structlog.testing.capture_logs() as logs:
                await client.download(url)

        start_events = [e for e in logs if e.get("event") == "audio_download.start"]
        assert len(start_events) == 1
        assert start_events[0].get("media_url") == url

    @pytest.mark.asyncio
    async def test_download_logs_http_error_event(self):
        """download logs audio_download.http_error with status_code on HTTPStatusError."""
        client = make_client()

        mock_response = MagicMock()
        mock_response.status_code = 500
        http_error = httpx.HTTPStatusError(
            message="500 Internal Server Error",
            request=MagicMock(),
            response=mock_response,
        )

        mock_get = AsyncMock(side_effect=http_error)

        with patch("httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = mock_get
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            with structlog.testing.capture_logs() as logs:
                with pytest.raises(TranscriptionError):
                    await client.download("https://api.twilio.com/media/0")

        error_events = [e for e in logs if e.get("event") == "audio_download.http_error"]
        assert len(error_events) == 1
        assert error_events[0]["status_code"] == 500

    @pytest.mark.asyncio
    async def test_download_logs_request_error_event(self):
        """download logs audio_download.request_error on network failure."""
        client = make_client()

        request_error = httpx.ConnectError("Connection refused")

        mock_get = AsyncMock(side_effect=request_error)

        with patch("httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.get = mock_get
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            with structlog.testing.capture_logs() as logs:
                with pytest.raises(TranscriptionError):
                    await client.download("https://api.twilio.com/media/0")

        error_events = [e for e in logs if e.get("event") == "audio_download.request_error"]
        assert len(error_events) == 1
