"""Transport DTOs for inbound/outbound Twilio messages and health responses."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class InboundMessage(BaseModel):
    """DTO for an inbound Twilio WhatsApp message webhook payload."""

    message_sid: str = Field(..., description="Twilio MessageSid — used for idempotency")
    from_number: str = Field(
        ..., description="E.164 WhatsApp number (e.g. whatsapp:+447700900000)"
    )
    to_number: str = Field(..., description="The Twilio WhatsApp number that received the message")
    body: str = Field(..., description="The text body of the inbound message")
    num_media: int = Field(default=0, description="Number of media attachments")
    media_content_type_0: Optional[str] = Field(
        default=None, description="MIME type of the first media attachment (future voice note support)"
    )
    media_url_0: Optional[str] = Field(
        default=None, description="Twilio MediaUrl0 — URL of the first media attachment"
    )


class DeliveryStatusCallback(BaseModel):
    """DTO for a Twilio delivery status callback payload."""

    message_sid: str = Field(..., description="Twilio MessageSid")
    message_status: str = Field(
        ..., description="Delivery status: delivered | failed | undelivered | sent"
    )
    to: str = Field(..., description="Destination phone number")
    error_code: Optional[str] = Field(default=None, description="Twilio error code if failed")
    error_message: Optional[str] = Field(
        default=None, description="Human-readable error message if failed"
    )


class HealthResponse(BaseModel):
    """DTO for the health check endpoint response."""

    status: str = Field(..., description='"ok" | "degraded"')
    redis_connected: bool = Field(..., description="Whether Redis is reachable")
    version: str = Field(..., description="Application version string")
    timestamp: datetime = Field(..., description="UTC timestamp of the health check")
