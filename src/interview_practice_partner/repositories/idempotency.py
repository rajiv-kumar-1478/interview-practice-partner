"""IdempotencyRepository — Redis-backed MessageSid deduplication store."""

from __future__ import annotations

import redis.asyncio
import redis.exceptions

from interview_practice_partner.domain.exceptions import SessionStoreUnavailableError


class IdempotencyRepository:
    """Redis-backed idempotency store for Twilio MessageSid deduplication.

    Prevents duplicate processing of retried Twilio webhook requests by
    tracking already-processed MessageSids in Redis with a configurable TTL.

    Satisfies Requirement 1.7: idempotent handling of Twilio retry requests.
    """

    def __init__(self, redis_client: redis.asyncio.Redis, ttl_seconds: int) -> None:
        """Initialise the repository.

        Args:
            redis_client: An already-configured ``redis.asyncio.Redis`` instance.
            ttl_seconds: Time-to-live in seconds applied to each idempotency key.
        """
        self._redis = redis_client
        self._ttl = ttl_seconds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(message_sid: str) -> str:
        return f"idempotency:{message_sid}"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def is_processed(self, message_sid: str) -> bool:
        """Check whether a MessageSid has already been processed.

        Args:
            message_sid: The Twilio MessageSid to check.

        Returns:
            ``True`` if the MessageSid exists in Redis, ``False`` otherwise.

        Raises:
            SessionStoreUnavailableError: If Redis is unreachable or times out.
        """
        try:
            result = await self._redis.exists(self._key(message_sid))
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as exc:
            raise SessionStoreUnavailableError(
                f"Redis unavailable while checking idempotency for {message_sid}: {exc}"
            ) from exc

        return bool(result)

    async def mark_processed(self, message_sid: str) -> None:
        """Record a MessageSid as processed with a TTL.

        Args:
            message_sid: The Twilio MessageSid to mark as processed.

        Raises:
            SessionStoreUnavailableError: If Redis is unreachable or times out.
        """
        try:
            await self._redis.setex(self._key(message_sid), self._ttl, "1")
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as exc:
            raise SessionStoreUnavailableError(
                f"Redis unavailable while marking idempotency for {message_sid}: {exc}"
            ) from exc
