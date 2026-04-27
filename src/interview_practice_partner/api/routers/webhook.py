"""Router for Twilio inbound message and delivery status webhook endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

import structlog

from interview_practice_partner.api.dependencies import get_orchestration_service
from interview_practice_partner.api.schemas import DeliveryStatusCallback, InboundMessage
from interview_practice_partner.services.orchestration import MessageOrchestrationService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhook"])


def _parse_form_body(body: bytes) -> dict[str, str]:
    """Parse a URL-encoded form body into a plain dict.

    Args:
        body: Raw bytes of the ``application/x-www-form-urlencoded`` body.

    Returns:
        A ``{field: value}`` dict.  If the body is empty or unparseable,
        returns an empty dict.
    """
    params: dict[str, str] = {}
    if not body:
        return params
    try:
        from urllib.parse import unquote_plus

        raw_str = body.decode("utf-8")
        for part in raw_str.split("&"):
            if "=" in part:
                key, _, value = part.partition("=")
                params[unquote_plus(key)] = unquote_plus(value)
    except Exception:  # noqa: BLE001
        logger.warning("webhook_form_parse_error")
    return params


@router.post("/whatsapp")
async def whatsapp_webhook(
    request: Request,
    orchestration_service: MessageOrchestrationService = Depends(get_orchestration_service),
) -> Response:
    """Receive an inbound WhatsApp message from Twilio.

    Parses the Twilio form body into an ``InboundMessage`` DTO and delegates
    processing to ``MessageOrchestrationService``.  Always returns HTTP 200
    so Twilio does not retry the delivery.

    Requirements: 1.1, 1.4.
    """
    # Use the raw body cached by TwilioSignatureMiddleware (avoids double-read)
    body: bytes = getattr(request.state, "body", b"") or b""
    params = _parse_form_body(body)

    message = InboundMessage(
        message_sid=params.get("MessageSid", ""),
        from_number=params.get("From", ""),
        to_number=params.get("To", ""),
        body=params.get("Body", ""),
        num_media=int(params.get("NumMedia", "0") or "0"),
        media_content_type_0=params.get("MediaContentType0") or None,
        media_url_0=params.get("MediaUrl0") or None,
    )

    logger.info(
        "webhook.whatsapp.received",
        message_sid=message.message_sid,
        from_number=message.from_number,
    )

    await orchestration_service.handle(message)

    return Response(content="OK", status_code=200, media_type="text/plain")


@router.post("/status")
async def status_callback(request: Request) -> Response:
    """Receive a Twilio delivery status callback.

    Parses the form body into a ``DeliveryStatusCallback`` DTO and logs
    delivery failures with ``phone_number`` and ``message_sid``.  Always
    returns HTTP 200.

    Requirements: 1.6.
    """
    body: bytes = getattr(request.state, "body", b"") or b""
    params = _parse_form_body(body)

    callback = DeliveryStatusCallback(
        message_sid=params.get("MessageSid", ""),
        message_status=params.get("MessageStatus", ""),
        to=params.get("To", ""),
        error_code=params.get("ErrorCode") or None,
        error_message=params.get("ErrorMessage") or None,
    )

    if callback.message_status in ("failed", "undelivered"):
        logger.error(
            "twilio.delivery_failure",
            phone_number=callback.to,
            message_sid=callback.message_sid,
            message_status=callback.message_status,
            error_code=callback.error_code,
            error_message=callback.error_message,
        )
    else:
        logger.info(
            "twilio.delivery_status",
            message_sid=callback.message_sid,
            message_status=callback.message_status,
        )

    return Response(content="OK", status_code=200, media_type="text/plain")
