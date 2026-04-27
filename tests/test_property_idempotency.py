# Feature: interview-practice-partner, Property 4: Idempotent Message Processing
"""Property-based tests for IdempotencyRepository idempotency invariants.

Processing the same MessageSid twice results in exactly one state mutation.

Validates: Requirements 1.7
"""

import fakeredis.aioredis
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from interview_practice_partner.repositories.idempotency import IdempotencyRepository


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Arbitrary MessageSid strings: alphanumeric plus hyphens and underscores,
# matching the character set used by Twilio MessageSids (e.g. "SMxxxxxxxx...")
_message_sid_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="-_",
    ),
    min_size=1,
    max_size=64,
)


# ---------------------------------------------------------------------------
# Property 4: Idempotent Message Processing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(message_sid=_message_sid_strategy)
@settings(max_examples=100)
async def test_property_4_mark_processed_once_is_processed_returns_true(
    message_sid: str,
) -> None:
    """Property 4 (part 1): After calling mark_processed once, is_processed returns True.

    For any arbitrary MessageSid, marking it as processed must cause
    is_processed to return True — the state mutation is observable.

    Validates: Requirements 1.7
    """
    redis_client = fakeredis.aioredis.FakeRedis()
    repo = IdempotencyRepository(redis_client=redis_client, ttl_seconds=600)

    # Initially the SID is unknown
    assert await repo.is_processed(message_sid) is False

    # After marking, it must be recognised as processed
    await repo.mark_processed(message_sid)
    assert await repo.is_processed(message_sid) is True


@pytest.mark.asyncio
@given(message_sid=_message_sid_strategy)
@settings(max_examples=100)
async def test_property_4_double_mark_does_not_raise_and_remains_processed(
    message_sid: str,
) -> None:
    """Property 4 (part 2): Calling mark_processed twice does not raise and is_processed is still True.

    Idempotency requires that a second call with the same SID is a no-op:
    no exception is raised and the processed state is preserved.

    Validates: Requirements 1.7
    """
    redis_client = fakeredis.aioredis.FakeRedis()
    repo = IdempotencyRepository(redis_client=redis_client, ttl_seconds=600)

    # First mark — establishes the state
    await repo.mark_processed(message_sid)

    # Second mark — must not raise
    await repo.mark_processed(message_sid)

    # State must still be True after the second call
    assert await repo.is_processed(message_sid) is True


@pytest.mark.asyncio
@given(
    sid_a=_message_sid_strategy,
    sid_b=_message_sid_strategy,
)
@settings(max_examples=100)
async def test_property_4_different_sid_is_not_affected(
    sid_a: str,
    sid_b: str,
) -> None:
    """Property 4 (part 3): A different SID is not affected by marking another SID.

    Marking sid_a must not cause sid_b to appear processed, unless sid_a == sid_b.

    Validates: Requirements 1.7
    """
    redis_client = fakeredis.aioredis.FakeRedis()
    repo = IdempotencyRepository(redis_client=redis_client, ttl_seconds=600)

    await repo.mark_processed(sid_a)

    if sid_a == sid_b:
        # Same SID — both lookups refer to the same key, must be True
        assert await repo.is_processed(sid_b) is True
    else:
        # Different SID — must remain unprocessed
        assert await repo.is_processed(sid_b) is False


@pytest.mark.asyncio
@given(
    message_sid=_message_sid_strategy,
    extra_calls=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=100)
async def test_property_4_state_mutation_count_is_exactly_one(
    message_sid: str,
    extra_calls: int,
) -> None:
    """Property 4 (part 4): State mutation count is exactly 1 regardless of submission count.

    No matter how many times mark_processed is called with the same SID,
    the key exists exactly once in Redis (is_processed returns True, not
    multiple entries), and the state is indistinguishable from a single call.

    Validates: Requirements 1.7
    """
    redis_client = fakeredis.aioredis.FakeRedis()
    repo = IdempotencyRepository(redis_client=redis_client, ttl_seconds=600)

    # Call mark_processed 1 + extra_calls times (at least once, up to 11 times)
    await repo.mark_processed(message_sid)
    for _ in range(extra_calls):
        await repo.mark_processed(message_sid)

    # Regardless of how many times it was submitted, is_processed must be True
    assert await repo.is_processed(message_sid) is True

    # The underlying Redis key must exist exactly once (count = 1, not 0 or >1)
    key = repo._key(message_sid)
    key_count = await redis_client.exists(key)
    assert key_count == 1, (
        f"Expected exactly 1 Redis key for SID '{message_sid}' after "
        f"{1 + extra_calls} mark_processed calls, got {key_count}"
    )
