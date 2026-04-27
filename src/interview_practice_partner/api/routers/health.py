"""Router for the GET /health endpoint used by load balancers and monitoring."""

from __future__ import annotations

from datetime import datetime, timezone

import redis.asyncio
import redis.exceptions
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from interview_practice_partner.api.schemas import HealthResponse

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> JSONResponse:
    """Check application health by pinging Redis.

    Returns HTTP 200 with ``status="ok"`` and ``redis_connected=True`` when
    Redis is reachable.  Returns HTTP 503 with ``status="degraded"`` and
    ``redis_connected=False`` when Redis is unreachable.

    Requirements: 12.2.
    """
    redis_client: redis.asyncio.Redis = request.app.state.redis
    settings = request.app.state.settings
    timestamp = datetime.now(tz=timezone.utc)

    try:
        await redis_client.ping()
        redis_connected = True
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, Exception):
        logger.warning("health_check.redis_unreachable")
        redis_connected = False

    if redis_connected:
        response_body = HealthResponse(
            status="ok",
            redis_connected=True,
            version=settings.app_version,
            timestamp=timestamp,
        )
        return JSONResponse(
            content=response_body.model_dump(mode="json"),
            status_code=200,
        )
    else:
        response_body = HealthResponse(
            status="degraded",
            redis_connected=False,
            version=settings.app_version,
            timestamp=timestamp,
        )
        return JSONResponse(
            content=response_body.model_dump(mode="json"),
            status_code=503,
        )
