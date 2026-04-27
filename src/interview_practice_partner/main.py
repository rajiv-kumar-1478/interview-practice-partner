"""FastAPI application factory, lifespan, and middleware registration."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio
import structlog
import structlog.contextvars
from fastapi import FastAPI
from twilio.rest import Client as TwilioClient

from interview_practice_partner.api.middleware.request_logging import RequestLoggingMiddleware
from interview_practice_partner.api.middleware.twilio_signature import TwilioSignatureMiddleware
from interview_practice_partner.api.routers import health, webhook
from interview_practice_partner.api.routers.media import _MEDIA_DIR, router as media_router
from interview_practice_partner.config import Settings


def _configure_structlog(log_level: str) -> None:
    """Configure structlog for JSON output.

    Sets up structlog with:
    - Context variable support (for ``correlation_id`` propagation)
    - JSON renderer for structured log output
    - Standard processors (timestamps, log level, caller info)

    Args:
        log_level: The minimum log level string (e.g. ``"INFO"``).
    """
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


async def _media_cleanup_loop(ttl_seconds: int, interval_seconds: int = 300) -> None:
    """Periodically delete TTS audio files older than *ttl_seconds*.

    Runs in a background asyncio task for the lifetime of the application.
    Sleeps for *interval_seconds* between each cleanup pass (default 5 minutes).
    Exceptions are caught and logged so the loop never silently dies.

    Args:
        ttl_seconds: Files older than this many seconds are deleted.
        interval_seconds: How often to run the cleanup pass (default 300 s).

    Requirements: 4.2
    """
    logger = structlog.get_logger(__name__)
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            now = time.time()
            deleted = 0
            for file_path in _MEDIA_DIR.iterdir():
                try:
                    if file_path.is_file() and (now - file_path.stat().st_mtime) > ttl_seconds:
                        file_path.unlink()
                        deleted += 1
                except OSError as exc:
                    logger.warning("media_cleanup.file_error", path=str(file_path), error=str(exc))
            if deleted:
                logger.info("media_cleanup.complete", deleted=deleted, ttl_seconds=ttl_seconds)
        except asyncio.CancelledError:
            logger.info("media_cleanup.cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("media_cleanup.unexpected_error", error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager.

    On startup:
    - Configures structlog for JSON output
    - Opens a Redis async connection pool
    - Initialises the Twilio REST client
    - Stores all shared resources in ``app.state``
    - Starts a background task to clean up expired TTS audio files

    On shutdown:
    - Cancels the media cleanup background task
    - Closes the Redis connection pool

    Requirements: 12.1, 12.3, 4.2.
    """
    settings: Settings = app.state.settings

    # Configure structured logging
    _configure_structlog(settings.log_level)

    logger = structlog.get_logger(__name__)
    logger.info("application_startup", version=settings.app_version, environment=settings.environment)

    # Open Redis connection pool
    redis_client: redis.asyncio.Redis = redis.asyncio.from_url(
        settings.redis_url,
        max_connections=settings.redis_pool_size,
        decode_responses=False,
    )
    app.state.redis = redis_client
    logger.info("redis_connected", redis_url=settings.redis_url)

    # Initialise Twilio client
    twilio_client = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)
    app.state.twilio_client = twilio_client
    logger.info("twilio_client_initialised")

    # Start background media cleanup task
    cleanup_task = asyncio.create_task(
        _media_cleanup_loop(ttl_seconds=settings.media_ttl_seconds)
    )
    logger.info("media_cleanup_task_started", ttl_seconds=settings.media_ttl_seconds)

    yield

    # Shutdown: cancel cleanup task and close Redis connection pool
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    await redis_client.aclose()
    logger.info("application_shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional ``Settings`` instance.  If ``None``, a new
            ``Settings`` instance is created from environment variables.

    Returns:
        A fully configured ``FastAPI`` application instance.
    """
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title="Interview Practice Partner",
        version=settings.app_version,
        lifespan=lifespan,
    )

    # Store settings in app state so dependencies can access them
    app.state.settings = settings

    # Register middleware (order matters: outermost middleware runs first)
    # RequestLoggingMiddleware wraps everything so it captures total latency
    app.add_middleware(RequestLoggingMiddleware)
    # TwilioSignatureMiddleware validates signatures before routing
    app.add_middleware(TwilioSignatureMiddleware, auth_token=settings.twilio_auth_token)

    # Include routers
    app.include_router(webhook.router)
    app.include_router(health.router)
    app.include_router(media_router)

    return app


# Module-level app instance for ASGI servers (uvicorn, gunicorn).
# Settings are loaded from environment variables at import time.
# This will raise a ValidationError if required env vars are missing —
# that is intentional: the application cannot start without valid configuration.
try:
    app = create_app()
except Exception:  # noqa: BLE001
    # Allow the module to be imported in test environments where env vars
    # are not set; tests should call create_app() with explicit Settings.
    app = None  # type: ignore[assignment]
