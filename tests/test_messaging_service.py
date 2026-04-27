"""Unit tests for TwilioMessagingService.

Covers:
- Message under 4096 chars is sent as a single chunk
- Message over 4096 chars is split into multiple chunks each ≤ 4096 chars
- Delivery failure is logged with required fields (phone_number, message_sid)
- send_chunked splits at word boundaries
- send_chunked handles edge cases (empty string, exact boundary, single long word)

Requirements: 1.5, 1.6, 9.7, 10.6
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from interview_practice_partner.services.messaging import TwilioMessagingService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PHONE = "whatsapp:+15550001234"
FROM_NUMBER = "whatsapp:+14155238886"


def make_service(
    message_status: str = "sent",
    message_sid: str = "SM1234567890abcdef",
) -> tuple[TwilioMessagingService, MagicMock, MagicMock]:
    """Build a TwilioMessagingService with mocked Twilio client and settings."""
    mock_message = MagicMock()
    mock_message.status = message_status
    mock_message.sid = message_sid

    mock_messages = MagicMock()
    mock_messages.create.return_value = mock_message

    mock_client = MagicMock()
    mock_client.messages = mock_messages

    mock_settings = MagicMock()
    mock_settings.twilio_whatsapp_number = FROM_NUMBER

    service = TwilioMessagingService(client=mock_client, settings=mock_settings)
    return service, mock_client, mock_settings


# ===========================================================================
# send_chunked — unit tests
# ===========================================================================


class TestSendChunked:
    def test_short_text_returns_single_chunk(self):
        """Text under max_chars is returned as a single-element list."""
        text = "Hello, this is a short message."
        chunks = TwilioMessagingService.send_chunked(text, max_chars=4096)
        assert chunks == [text]

    def test_empty_string_returns_single_empty_chunk(self):
        """Empty string returns ['']."""
        chunks = TwilioMessagingService.send_chunked("", max_chars=4096)
        assert chunks == [""]

    def test_exact_boundary_returns_single_chunk(self):
        """Text of exactly max_chars is returned as a single chunk."""
        text = "a" * 4096
        chunks = TwilioMessagingService.send_chunked(text, max_chars=4096)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_text_one_over_boundary_splits_into_two(self):
        """Text of max_chars + 1 is split into two chunks."""
        # 4096 'a's + space + 'b' = 4098 chars; split at the space
        text = "a" * 4096 + " b"
        chunks = TwilioMessagingService.send_chunked(text, max_chars=4096)
        assert len(chunks) == 2
        assert all(len(c) <= 4096 for c in chunks)

    def test_split_at_word_boundary_not_mid_word(self):
        """Chunks do not break words in the middle."""
        # Build a text where a word straddles the 4096-char boundary
        word_a = "hello " * 682  # 6 * 682 = 4092 chars
        word_b = "world"         # would push past 4096 if appended
        text = word_a + word_b
        chunks = TwilioMessagingService.send_chunked(text, max_chars=4096)
        for chunk in chunks:
            assert len(chunk) <= 4096
        # Reassembled text should equal original (spaces stripped between chunks)
        reassembled = " ".join(chunks)
        assert reassembled == text

    def test_multiple_chunks_all_within_limit(self):
        """All chunks from a very long text are within max_chars."""
        text = ("word " * 2000).rstrip()  # ~10000 chars
        chunks = TwilioMessagingService.send_chunked(text, max_chars=4096)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 4096

    def test_chunks_reassemble_to_original(self):
        """Chunks joined with spaces reconstruct the original text."""
        text = ("The quick brown fox jumps over the lazy dog. " * 200).rstrip()
        chunks = TwilioMessagingService.send_chunked(text, max_chars=4096)
        reassembled = " ".join(chunks)
        assert reassembled == text

    def test_single_word_longer_than_max_chars_is_hard_split(self):
        """A single word longer than max_chars is hard-split at the boundary."""
        long_word = "x" * 5000
        chunks = TwilioMessagingService.send_chunked(long_word, max_chars=4096)
        assert len(chunks) == 2
        assert len(chunks[0]) == 4096
        assert len(chunks[1]) == 904
        assert "".join(chunks) == long_word

    def test_custom_max_chars(self):
        """send_chunked respects a custom max_chars value."""
        text = "one two three four five"
        chunks = TwilioMessagingService.send_chunked(text, max_chars=10)
        for chunk in chunks:
            assert len(chunk) <= 10

    def test_no_empty_chunks_in_output(self):
        """No chunk in the output is an empty string (for non-empty input)."""
        text = ("word " * 1000).rstrip()
        chunks = TwilioMessagingService.send_chunked(text, max_chars=4096)
        for chunk in chunks:
            assert chunk != ""


# ===========================================================================
# send_message — integration with Twilio client mock
# ===========================================================================


class TestSendMessage:
    @pytest.mark.asyncio
    async def test_short_message_calls_create_once(self):
        """A message under 4096 chars triggers exactly one Twilio API call."""
        service, mock_client, _ = make_service()
        await service.send_message("Hello!", PHONE)
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_short_message_uses_correct_from_and_to(self):
        """Twilio create is called with the correct from_ and to parameters."""
        service, mock_client, _ = make_service()
        await service.send_message("Hello!", PHONE)
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["from_"] == FROM_NUMBER
        assert call_kwargs.kwargs["to"] == PHONE

    @pytest.mark.asyncio
    async def test_short_message_body_is_passed_correctly(self):
        """The message body is passed verbatim to Twilio create."""
        service, mock_client, _ = make_service()
        body = "This is my interview question."
        await service.send_message(body, PHONE)
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["body"] == body

    @pytest.mark.asyncio
    async def test_long_message_calls_create_multiple_times(self):
        """A message over 4096 chars triggers multiple Twilio API calls."""
        service, mock_client, _ = make_service()
        long_text = ("word " * 1000).rstrip()  # ~5000 chars
        await service.send_message(long_text, PHONE)
        assert mock_client.messages.create.call_count >= 2

    @pytest.mark.asyncio
    async def test_long_message_each_chunk_within_limit(self):
        """Each chunk sent to Twilio is at most 4096 characters."""
        service, mock_client, _ = make_service()
        long_text = ("word " * 1000).rstrip()
        await service.send_message(long_text, PHONE)
        for call in mock_client.messages.create.call_args_list:
            body = call.kwargs["body"]
            assert len(body) <= 4096

    @pytest.mark.asyncio
    async def test_chunks_sent_in_order(self):
        """Chunks are sent in sequential order (first chunk first)."""
        service, mock_client, _ = make_service()
        # Build text where first chunk is identifiable
        first_part = "FIRST " + "word " * 818  # ~4096 chars
        second_part = "SECOND part"
        text = first_part.rstrip() + " " + second_part
        await service.send_message(text, PHONE)
        calls = mock_client.messages.create.call_args_list
        assert len(calls) >= 2
        first_body = calls[0].kwargs["body"]
        second_body = calls[1].kwargs["body"]
        assert "FIRST" in first_body
        assert "SECOND" in second_body


# ===========================================================================
# Delivery failure logging
# ===========================================================================


class TestDeliveryFailureLogging:
    @pytest.mark.asyncio
    async def test_failed_status_is_logged(self):
        """Delivery failure with status 'failed' is logged with phone_number and message_sid."""
        service, _, _ = make_service(message_status="failed", message_sid="SMfailed123")

        with patch.object(
            service._client.messages, "create"
        ) as mock_create:
            mock_msg = MagicMock()
            mock_msg.status = "failed"
            mock_msg.sid = "SMfailed123"
            mock_create.return_value = mock_msg

            logged_events: list[dict] = []

            import structlog
            original_logger = structlog.get_logger

            def capture_logger(name=None):
                log = original_logger(name)

                class CapturingLogger:
                    def bind(self, **kw):
                        return self

                    def info(self, event, **kw):
                        logged_events.append({"event": event, **kw})

                    def error(self, event, **kw):
                        logged_events.append({"event": event, **kw})

                return CapturingLogger()

            with patch("interview_practice_partner.services.messaging.logger") as mock_log:
                bound_log = MagicMock()
                mock_log.bind.return_value = bound_log

                await service.send_message("Hello!", PHONE)

                # Verify error was logged with required fields
                error_calls = bound_log.error.call_args_list
                assert len(error_calls) == 1
                call_kwargs = error_calls[0].kwargs
                assert call_kwargs["phone_number"] == PHONE
                assert call_kwargs["message_sid"] == "SMfailed123"

    @pytest.mark.asyncio
    async def test_undelivered_status_is_logged(self):
        """Delivery failure with status 'undelivered' is logged with required fields."""
        service, mock_client, _ = make_service(
            message_status="undelivered", message_sid="SMundelivered456"
        )

        with patch("interview_practice_partner.services.messaging.logger") as mock_log:
            bound_log = MagicMock()
            mock_log.bind.return_value = bound_log

            await service.send_message("Hello!", PHONE)

            error_calls = bound_log.error.call_args_list
            assert len(error_calls) == 1
            call_kwargs = error_calls[0].kwargs
            assert call_kwargs["phone_number"] == PHONE
            assert call_kwargs["message_sid"] == "SMundelivered456"

    @pytest.mark.asyncio
    async def test_sent_status_does_not_log_error(self):
        """Successful delivery (status 'sent') does not log an error."""
        service, _, _ = make_service(message_status="sent")

        with patch("interview_practice_partner.services.messaging.logger") as mock_log:
            bound_log = MagicMock()
            mock_log.bind.return_value = bound_log

            await service.send_message("Hello!", PHONE)

            bound_log.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_delivered_status_does_not_log_error(self):
        """Successful delivery (status 'delivered') does not log an error."""
        service, mock_client, _ = make_service(message_status="delivered")

        with patch("interview_practice_partner.services.messaging.logger") as mock_log:
            bound_log = MagicMock()
            mock_log.bind.return_value = bound_log

            await service.send_message("Hello!", PHONE)

            bound_log.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_outbound_call_is_logged(self):
        """Each outbound Twilio call is logged via structlog."""
        service, _, _ = make_service()

        with patch("interview_practice_partner.services.messaging.logger") as mock_log:
            bound_log = MagicMock()
            mock_log.bind.return_value = bound_log

            await service.send_message("Hello!", PHONE)

            # At least one info log for the outbound call
            assert bound_log.info.call_count >= 1


# ===========================================================================
# Property test — Property 19: Outbound Messages Respect the 4096-Character Limit
# ===========================================================================

# Feature: interview-practice-partner, Property 19: Outbound Messages Respect the 4096-Character Limit

from hypothesis import given, settings
from hypothesis import strategies as st


class TestMessageChunkingProperty:
    """Property 19: Outbound Messages Respect the 4096-Character Limit.

    Validates: Requirements 9.7
    """

    @given(text=st.text(min_size=0, max_size=20000))
    @settings(max_examples=100)
    def test_every_chunk_is_at_most_4096_characters(self, text: str):
        """Every chunk produced by send_chunked is at most 4096 characters.

        **Validates: Requirements 9.7**
        """
        chunks = TwilioMessagingService.send_chunked(text)
        for chunk in chunks:
            assert len(chunk) <= 4096, (
                f"Chunk of length {len(chunk)} exceeds 4096-character limit. "
                f"Input length: {len(text)}"
            )


# ===========================================================================
# media_url support — Requirements 4.1, 4.3, 4.4
# ===========================================================================


class TestSendMessageWithMediaUrl:
    """Tests for send_message when media_url is provided (Requirements 4.1, 4.3, 4.4)."""

    @pytest.mark.asyncio
    async def test_send_message_with_media_url_calls_twilio_with_media_url_param(self):
        """When media_url is provided, Twilio create is called with media_url=[...].

        Requirements: 4.1
        """
        service, mock_client, _ = make_service()
        url = "https://example.com/audio.mp3"

        await service.send_message("", PHONE, media_url=url)

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["media_url"] == [url]

    @pytest.mark.asyncio
    async def test_send_message_with_media_url_passes_correct_from_and_to(self):
        """When media_url is provided, from_ and to are still set correctly.

        Requirements: 4.1
        """
        service, mock_client, _ = make_service()
        url = "https://example.com/audio.mp3"

        await service.send_message("", PHONE, media_url=url)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["from_"] == FROM_NUMBER
        assert call_kwargs["to"] == PHONE

    @pytest.mark.asyncio
    async def test_send_message_with_media_url_skips_chunking(self):
        """When media_url is provided, Twilio create is called exactly once (no chunking).

        Requirements: 4.1
        """
        service, mock_client, _ = make_service()
        # Even a very long text body should not be chunked when media_url is set
        long_text = "word " * 2000
        url = "https://example.com/audio.mp3"

        await service.send_message(long_text, PHONE, media_url=url)

        assert mock_client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_send_message_with_media_url_logs_media_url_and_message_sid(self):
        """When media_url is provided, the log includes media_url and message_sid.

        Requirements: 4.3
        """
        service, _, _ = make_service(message_sid="SMmedia123")
        url = "https://example.com/audio.mp3"

        with patch("interview_practice_partner.services.messaging.logger") as mock_log:
            bound_log = MagicMock()
            mock_log.bind.return_value = bound_log

            await service.send_message("", PHONE, media_url=url)

            # Check that the outbound info log includes media_url
            info_calls = bound_log.info.call_args_list
            media_log_calls = [
                c for c in info_calls if c.kwargs.get("media_url") == url
            ]
            assert len(media_log_calls) >= 1, (
                "Expected at least one info log with media_url set"
            )

            # Check that the success log includes message_sid
            sid_log_calls = [
                c for c in info_calls if c.kwargs.get("message_sid") == "SMmedia123"
            ]
            assert len(sid_log_calls) >= 1, (
                "Expected at least one info log with message_sid set"
            )

    @pytest.mark.asyncio
    async def test_send_message_without_media_url_still_chunks_long_text(self):
        """When media_url is None, long messages are still chunked as before.

        Requirements: 4.1 (backward compatibility)
        """
        service, mock_client, _ = make_service()
        long_text = ("word " * 1000).rstrip()  # ~5000 chars

        await service.send_message(long_text, PHONE)  # no media_url

        assert mock_client.messages.create.call_count >= 2
        for call in mock_client.messages.create.call_args_list:
            assert len(call.kwargs["body"]) <= 4096
            # No media_url should be present in text-only calls
            assert "media_url" not in call.kwargs

    @pytest.mark.asyncio
    async def test_send_message_without_media_url_short_text_unchanged(self):
        """When media_url is None, short messages behave identically to before.

        Requirements: 4.1 (backward compatibility)
        """
        service, mock_client, _ = make_service()
        body = "Hello, this is a short message."

        await service.send_message(body, PHONE)  # no media_url

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["body"] == body
        assert "media_url" not in call_kwargs

    @pytest.mark.asyncio
    async def test_delivery_failure_with_media_url_logs_phone_number_and_error_details(self):
        """Delivery failure when media_url is set logs phone_number and error details.

        Requirements: 4.4
        """
        service, _, _ = make_service(message_status="failed", message_sid="SMfailedmedia")
        url = "https://example.com/audio.mp3"

        with patch("interview_practice_partner.services.messaging.logger") as mock_log:
            bound_log = MagicMock()
            mock_log.bind.return_value = bound_log

            await service.send_message("", PHONE, media_url=url)

            error_calls = bound_log.error.call_args_list
            assert len(error_calls) == 1, "Expected exactly one error log on delivery failure"
            call_kwargs = error_calls[0].kwargs
            assert call_kwargs["phone_number"] == PHONE
            assert call_kwargs["message_sid"] == "SMfailedmedia"

    @pytest.mark.asyncio
    async def test_delivery_undelivered_with_media_url_logs_phone_number_and_error_details(self):
        """Undelivered status when media_url is set logs phone_number and error details.

        Requirements: 4.4
        """
        service, _, _ = make_service(
            message_status="undelivered", message_sid="SMundeliveredmedia"
        )
        url = "https://example.com/audio.mp3"

        with patch("interview_practice_partner.services.messaging.logger") as mock_log:
            bound_log = MagicMock()
            mock_log.bind.return_value = bound_log

            await service.send_message("", PHONE, media_url=url)

            error_calls = bound_log.error.call_args_list
            assert len(error_calls) == 1
            call_kwargs = error_calls[0].kwargs
            assert call_kwargs["phone_number"] == PHONE
            assert call_kwargs["message_sid"] == "SMundeliveredmedia"

    @pytest.mark.asyncio
    async def test_successful_delivery_with_media_url_does_not_log_error(self):
        """Successful delivery with media_url does not log an error.

        Requirements: 4.4
        """
        service, _, _ = make_service(message_status="sent")
        url = "https://example.com/audio.mp3"

        with patch("interview_practice_partner.services.messaging.logger") as mock_log:
            bound_log = MagicMock()
            mock_log.bind.return_value = bound_log

            await service.send_message("", PHONE, media_url=url)

            bound_log.error.assert_not_called()
