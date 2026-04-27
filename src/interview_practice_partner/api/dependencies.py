"""FastAPI Depends() factories for service and repository injection."""

from __future__ import annotations

from typing import Annotated

import redis.asyncio
from fastapi import Depends, Request

from interview_practice_partner.audio.download_client import AudioDownloadClient
from interview_practice_partner.audio.tts_client import ElevenLabsClient
from interview_practice_partner.audio.whisper_client import GroqWhisperClient, WhisperClient
from interview_practice_partner.config import Settings
from interview_practice_partner.llm.client import LLMClient
from interview_practice_partner.llm.openai_client import OpenAIClient
from interview_practice_partner.llm.prompt_builder import PromptBuilder
from interview_practice_partner.repositories.idempotency import IdempotencyRepository
from interview_practice_partner.repositories.redis_session import RedisSessionRepository
from interview_practice_partner.services.feedback import FeedbackService
from interview_practice_partner.services.interview import InterviewService
from interview_practice_partner.services.messaging import TwilioMessagingService
from interview_practice_partner.services.orchestration import MessageOrchestrationService
from interview_practice_partner.services.session import SessionService
from interview_practice_partner.services.technical_round import TechnicalRoundService


# ---------------------------------------------------------------------------
# Low-level infrastructure dependencies
# ---------------------------------------------------------------------------


def get_settings(request: Request) -> Settings:
    """Return the ``Settings`` instance stored in ``app.state``."""
    return request.app.state.settings  # type: ignore[no-any-return]


def get_redis(request: Request) -> redis.asyncio.Redis:
    """Return the Redis client stored in ``app.state``."""
    return request.app.state.redis  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Repository dependencies
# ---------------------------------------------------------------------------


def get_session_repository(
    redis_client: Annotated[redis.asyncio.Redis, Depends(get_redis)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RedisSessionRepository:
    """Create a ``RedisSessionRepository`` for the current request."""
    return RedisSessionRepository(
        redis_client=redis_client,
        ttl_seconds=settings.redis_session_ttl_seconds,
    )


def get_idempotency_repository(
    redis_client: Annotated[redis.asyncio.Redis, Depends(get_redis)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> IdempotencyRepository:
    """Create an ``IdempotencyRepository`` for the current request."""
    return IdempotencyRepository(
        redis_client=redis_client,
        ttl_seconds=settings.redis_idempotency_ttl_seconds,
    )


# ---------------------------------------------------------------------------
# LLM dependencies
# ---------------------------------------------------------------------------


def get_llm_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LLMClient:
    """Create an ``OpenAIClient`` for the current request."""
    return OpenAIClient(settings=settings)


def get_prompt_builder() -> PromptBuilder:
    """Create a ``PromptBuilder`` instance."""
    return PromptBuilder()


# ---------------------------------------------------------------------------
# Audio client dependencies
# ---------------------------------------------------------------------------


def get_whisper_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> WhisperClient:
    """Create a ``GroqWhisperClient`` for the current request."""
    return GroqWhisperClient(settings=settings)


def get_tts_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ElevenLabsClient:
    """Create an ``ElevenLabsClient`` for the current request."""
    return ElevenLabsClient(settings=settings)


def get_audio_download_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AudioDownloadClient:
    """Create an ``AudioDownloadClient`` for the current request."""
    return AudioDownloadClient(settings=settings)


# ---------------------------------------------------------------------------
# Service dependencies
# ---------------------------------------------------------------------------


def get_messaging_service(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> TwilioMessagingService:
    """Create a ``TwilioMessagingService`` using the Twilio client from app state."""
    twilio_client = request.app.state.twilio_client
    return TwilioMessagingService(client=twilio_client, settings=settings)


def get_interview_service(
    llm_client: Annotated[LLMClient, Depends(get_llm_client)],
    prompt_builder: Annotated[PromptBuilder, Depends(get_prompt_builder)],
    whisper_client: Annotated[WhisperClient, Depends(get_whisper_client)],
    tts_client: Annotated[ElevenLabsClient, Depends(get_tts_client)],
    audio_download_client: Annotated[AudioDownloadClient, Depends(get_audio_download_client)],
) -> InterviewService:
    """Create an ``InterviewService`` for the current request."""
    # Create TechnicalRoundService
    technical_round_service = TechnicalRoundService(
        llm_client=llm_client,
        prompt_builder=prompt_builder,
    )
    
    return InterviewService(
        llm_client=llm_client,
        prompt_builder=prompt_builder,
        whisper_client=whisper_client,
        tts_client=tts_client,
        audio_download_client=audio_download_client,
        technical_round_service=technical_round_service,
    )


def get_feedback_service(
    llm_client: Annotated[LLMClient, Depends(get_llm_client)],
    prompt_builder: Annotated[PromptBuilder, Depends(get_prompt_builder)],
) -> FeedbackService:
    """Create a ``FeedbackService`` for the current request."""
    return FeedbackService(llm_client=llm_client, prompt_builder=prompt_builder)


def get_session_service(
    llm_client: Annotated[LLMClient, Depends(get_llm_client)],
    prompt_builder: Annotated[PromptBuilder, Depends(get_prompt_builder)],
    interview_service: Annotated[InterviewService, Depends(get_interview_service)],
    feedback_service: Annotated[FeedbackService, Depends(get_feedback_service)],
) -> SessionService:
    """Create a ``SessionService`` for the current request."""
    return SessionService(
        llm_client=llm_client,
        prompt_builder=prompt_builder,
        interview_service=interview_service,
        feedback_service=feedback_service,
    )


def get_orchestration_service(
    session_repository: Annotated[RedisSessionRepository, Depends(get_session_repository)],
    idempotency_repository: Annotated[IdempotencyRepository, Depends(get_idempotency_repository)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    messaging_service: Annotated[TwilioMessagingService, Depends(get_messaging_service)],
    tts_client: Annotated[ElevenLabsClient, Depends(get_tts_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> MessageOrchestrationService:
    """Create a ``MessageOrchestrationService`` for the current request."""
    return MessageOrchestrationService(
        session_repository=session_repository,
        idempotency_repository=idempotency_repository,
        session_service=session_service,
        messaging_service=messaging_service,
        tts_client=tts_client,
        settings=settings,
    )
