"""RedisSessionRepository — Redis-backed implementation of SessionRepository."""

from __future__ import annotations

import redis.asyncio
import redis.exceptions

from interview_practice_partner.domain.exceptions import SessionStoreUnavailableError
from interview_practice_partner.domain.models import SessionState
from interview_practice_partner.repositories.base import SessionRepository


class RedisSessionRepository(SessionRepository):
    """Redis-backed session repository.

    Serialises ``SessionState`` to/from JSON using Pydantic's
    ``.model_dump_json()`` / ``.model_validate_json()``.  Every ``save``
    refreshes the TTL so that active sessions never expire mid-conversation.

    Satisfies Requirements 2.1, 2.2, 2.3, 2.5, 2.6.
    """

    def __init__(self, redis_client: redis.asyncio.Redis, ttl_seconds: int) -> None:
        """Initialise the repository.

        Args:
            redis_client: An already-configured ``redis.asyncio.Redis`` instance.
            ttl_seconds: Time-to-live in seconds applied to every ``save``.
        """
        self._redis = redis_client
        self._ttl = ttl_seconds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(phone_number: str) -> str:
        return f"session:{phone_number}"

    # ------------------------------------------------------------------
    # SessionRepository interface
    # ------------------------------------------------------------------

    async def get(self, phone_number: str) -> SessionState | None:
        """Retrieve the session state for the given phone number.

        Returns:
            The persisted ``SessionState``, or ``None`` if no session exists.

        Raises:
            SessionStoreUnavailableError: If Redis is unreachable or times out.
        """
        try:
            raw: bytes | None = await self._redis.get(self._key(phone_number))
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as exc:
            raise SessionStoreUnavailableError(
                f"Redis unavailable while fetching session for {phone_number}: {exc}"
            ) from exc

        if raw is None:
            return None

        return SessionState.model_validate_json(raw)

    async def save(self, session: SessionState) -> None:
        """Persist (create or update) the given session state with a refreshed TTL.

        Args:
            session: The ``SessionState`` to store, keyed by its ``phone_number``.

        Raises:
            SessionStoreUnavailableError: If Redis is unreachable or times out.
        """
        try:
            await self._redis.setex(
                self._key(session.phone_number),
                self._ttl,
                session.model_dump_json(),
            )
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as exc:
            raise SessionStoreUnavailableError(
                f"Redis unavailable while saving session for {session.phone_number}: {exc}"
            ) from exc

    async def delete(self, phone_number: str) -> None:
        """Remove the session state for the given phone number.

        Args:
            phone_number: E.164-formatted WhatsApp phone number.

        Raises:
            SessionStoreUnavailableError: If Redis is unreachable or times out.
        """
        try:
            await self._redis.delete(self._key(phone_number))
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as exc:
            raise SessionStoreUnavailableError(
                f"Redis unavailable while deleting session for {phone_number}: {exc}"
            ) from exc
