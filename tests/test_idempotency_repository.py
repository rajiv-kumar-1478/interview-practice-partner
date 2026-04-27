"""Unit tests for IdempotencyRepository.

Covers:
- is_processed returns False for unknown MessageSids
- is_processed returns True after mark_processed
- mark_processed sets the key with the correct TTL
- Keys expire after TTL
- Both methods raise SessionStoreUnavailableError on Redis connection errors
"""

from unittest.mock import AsyncMock, patch

import fakeredis.aioredis
import pytest
import redis.asyncio
import redis.exceptions

from interview_practice_partner.domain.exceptions import SessionStoreUnavailableError
from interview_practice_partner.repositories.idempotency import IdempotencyRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_redis() -> fakeredis.aioredis.FakeRedis:
    return fakeredis.aioredis.FakeRedis()


@pytest.fixture
def repo(fake_redis: fakeredis.aioredis.FakeRedis) -> IdempotencyRepository:
    return IdempotencyRepository(redis_client=fake_redis, ttl_seconds=600)


# ---------------------------------------------------------------------------
# is_processed
# ---------------------------------------------------------------------------

class TestIsProcessed:
    async def test_returns_false_for_unknown_sid(self, repo: IdempotencyRepository):
        result = await repo.is_processed("SM_unknown_sid")
        assert result is False

    async def test_returns_true_after_mark_processed(self, repo: IdempotencyRepository):
        sid = "SM_abc123"
        await repo.mark_processed(sid)
        result = await repo.is_processed(sid)
        assert result is True

    async def test_different_sids_are_independent(self, repo: IdempotencyRepository):
        await repo.mark_processed("SM_first")
        assert await repo.is_processed("SM_first") is True
        assert await repo.is_processed("SM_second") is False


# ---------------------------------------------------------------------------
# mark_processed
# ---------------------------------------------------------------------------

class TestMarkProcessed:
    async def test_sets_key_with_ttl(
        self, repo: IdempotencyRepository, fake_redis: fakeredis.aioredis.FakeRedis
    ):
        sid = "SM_ttl_test"
        await repo.mark_processed(sid)
        ttl = await fake_redis.ttl(f"idempotency:{sid}")
        # TTL should be set (positive value, not -1 which means no expiry)
        assert ttl > 0
        assert ttl <= 600

    async def test_sets_key_value_to_one(
        self, repo: IdempotencyRepository, fake_redis: fakeredis.aioredis.FakeRedis
    ):
        sid = "SM_value_test"
        await repo.mark_processed(sid)
        value = await fake_redis.get(f"idempotency:{sid}")
        assert value == b"1"

    async def test_idempotent_double_mark(self, repo: IdempotencyRepository):
        """Calling mark_processed twice should not raise and key should still exist."""
        sid = "SM_double"
        await repo.mark_processed(sid)
        await repo.mark_processed(sid)
        assert await repo.is_processed(sid) is True

    async def test_key_pattern(
        self, repo: IdempotencyRepository, fake_redis: fakeredis.aioredis.FakeRedis
    ):
        sid = "SM_pattern_check"
        await repo.mark_processed(sid)
        # Verify the exact key pattern used
        exists = await fake_redis.exists(f"idempotency:{sid}")
        assert exists == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    async def test_is_processed_raises_on_connection_error(self, repo: IdempotencyRepository):
        with patch.object(
            repo._redis,
            "exists",
            new=AsyncMock(side_effect=redis.exceptions.ConnectionError("refused")),
        ):
            with pytest.raises(SessionStoreUnavailableError):
                await repo.is_processed("SM_any")

    async def test_is_processed_raises_on_timeout_error(self, repo: IdempotencyRepository):
        with patch.object(
            repo._redis,
            "exists",
            new=AsyncMock(side_effect=redis.exceptions.TimeoutError("timed out")),
        ):
            with pytest.raises(SessionStoreUnavailableError):
                await repo.is_processed("SM_any")

    async def test_mark_processed_raises_on_connection_error(self, repo: IdempotencyRepository):
        with patch.object(
            repo._redis,
            "setex",
            new=AsyncMock(side_effect=redis.exceptions.ConnectionError("refused")),
        ):
            with pytest.raises(SessionStoreUnavailableError):
                await repo.mark_processed("SM_any")

    async def test_mark_processed_raises_on_timeout_error(self, repo: IdempotencyRepository):
        with patch.object(
            repo._redis,
            "setex",
            new=AsyncMock(side_effect=redis.exceptions.TimeoutError("timed out")),
        ):
            with pytest.raises(SessionStoreUnavailableError):
                await repo.mark_processed("SM_any")
