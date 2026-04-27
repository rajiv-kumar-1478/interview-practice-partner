"""Unit tests for the GET /health endpoint.

Tests:
- Returns HTTP 200 with ``redis_connected=true`` when Redis is available
- Returns HTTP 503 with ``status="degraded"`` when Redis is unavailable

Requirements: 12.2
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import fakeredis.aioredis
import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from httpx import ASGITransport

from interview_practice_partner.api.routers.health import router as health_router
from interview_practice_partner.config import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings() -> Settings:
    """Return a minimal Settings instance suitable for tests."""
    return Settings(
        twilio_account_sid="ACtest",
        twilio_auth_token="test_auth_token",
        twilio_whatsapp_number="whatsapp:+14155238886",
        llm_api_key="test_llm_key",
        redis_url="redis://localhost:6379/0",
        app_version="1.0.0-test",
        groq_api_key="test_groq_key",
        elevenlabs_api_key="test_elevenlabs_key",
        elevenlabs_voice_id="test_voice_id",
    )


def _build_test_app(redis_client) -> FastAPI:
    """Build a minimal FastAPI app with the health router and a given Redis client.

    The app does NOT use the full lifespan (which would try to connect to a real
    Redis instance).  Instead, we inject the Redis client and Settings directly
    into ``app.state`` before each test.
    """
    app = FastAPI()
    app.include_router(health_router)

    # Inject shared state that the health router reads via ``request.app.state``
    app.state.redis = redis_client
    app.state.settings = _make_settings()

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Unit tests for GET /health.

    Requirements: 12.2
    """

    async def test_health_returns_200_when_redis_available(self) -> None:
        """GET /health returns HTTP 200 with redis_connected=true when Redis is reachable.

        A real fakeredis async client is used so that ``ping()`` succeeds without
        requiring a live Redis server.

        Requirements: 12.2
        """
        redis_client = fakeredis.aioredis.FakeRedis()
        app = _build_test_app(redis_client)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["redis_connected"] is True
        assert body["status"] == "ok"

    async def test_health_returns_503_when_redis_unavailable(self, mocker) -> None:
        """GET /health returns HTTP 503 with status="degraded" when Redis is unreachable.

        The Redis ``ping()`` method is patched to raise a ``ConnectionError`` so
        that the health router's exception handler is exercised.

        Requirements: 12.2
        """
        import redis.exceptions

        redis_client = fakeredis.aioredis.FakeRedis()
        # Patch ping to simulate a connection failure
        mocker.patch.object(
            redis_client,
            "ping",
            new=AsyncMock(side_effect=redis.exceptions.ConnectionError("unreachable")),
        )

        app = _build_test_app(redis_client)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "degraded"
        assert body["redis_connected"] is False

    async def test_health_response_includes_version_and_timestamp(self) -> None:
        """GET /health response body includes version and timestamp fields.

        Requirements: 12.2
        """
        redis_client = fakeredis.aioredis.FakeRedis()
        app = _build_test_app(redis_client)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert "version" in body
        assert "timestamp" in body
        assert body["version"] == "1.0.0-test"

    async def test_health_degraded_response_includes_version_and_timestamp(
        self, mocker
    ) -> None:
        """Degraded health response also includes version and timestamp fields.

        Requirements: 12.2
        """
        import redis.exceptions

        redis_client = fakeredis.aioredis.FakeRedis()
        mocker.patch.object(
            redis_client,
            "ping",
            new=AsyncMock(side_effect=redis.exceptions.TimeoutError("timeout")),
        )

        app = _build_test_app(redis_client)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 503
        body = response.json()
        assert "version" in body
        assert "timestamp" in body
        assert body["status"] == "degraded"
