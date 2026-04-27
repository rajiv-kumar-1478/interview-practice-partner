# Feature: voice-note-support, Property 5: preferred_mode Survives Redis Round-Trip
"""Property-based tests for SessionState.preferred_mode JSON serialisation round-trip.

Validates: Requirements 7.2
"""

import uuid
from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from interview_practice_partner.domain.models import SessionState

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_preferred_mode_strategy = st.sampled_from(["voice", "text"])

_session_id_strategy = st.uuids().map(str)

_phone_number_strategy = st.from_regex(r"\+[1-9]\d{6,14}", fullmatch=True)

_datetime_strategy = st.datetimes(timezones=st.just(timezone.utc))


def _st_session_state(preferred_mode_st=_preferred_mode_strategy):
    """Strategy that generates arbitrary SessionState instances.

    Generates both 'voice' and 'text' preferred_mode values alongside
    the other required fields, simulating the full range of sessions
    that would be stored in and retrieved from Redis.
    """
    return st.builds(
        SessionState,
        session_id=_session_id_strategy,
        phone_number=_phone_number_strategy,
        preferred_mode=preferred_mode_st,
        created_at=_datetime_strategy,
        updated_at=_datetime_strategy,
    )


# ---------------------------------------------------------------------------
# Property 5: preferred_mode Survives Redis Round-Trip
# ---------------------------------------------------------------------------


@given(session=_st_session_state())
@settings(max_examples=100)
def test_property_5_preferred_mode_survives_redis_round_trip(session: SessionState) -> None:
    """Property 5: preferred_mode Survives Redis Round-Trip.

    For any SessionState with preferred_mode set to either "voice" or "text",
    serialising the model to JSON (as done when writing to Redis) and
    deserialising it back (as done when reading from Redis) SHALL produce a
    SessionState with the same preferred_mode value.

    This simulates the exact Redis serialisation/deserialisation path used by
    RedisSessionRepository: model_dump_json() on write, model_validate_json()
    on read.

    **Validates: Requirements 7.2**
    """
    json_str = session.model_dump_json()
    restored = SessionState.model_validate_json(json_str)

    assert restored.preferred_mode == session.preferred_mode, (
        f"preferred_mode changed after JSON round-trip: "
        f"before={session.preferred_mode!r}, after={restored.preferred_mode!r}, "
        f"json={json_str!r}"
    )


@given(
    session=_st_session_state(),
)
@settings(max_examples=100)
def test_property_5_round_trip_preserves_preferred_mode_type(session: SessionState) -> None:
    """Property 5 (type invariant): preferred_mode is always a str after round-trip.

    After deserialisation, preferred_mode must be a plain Python str (not a
    Literal subtype or enum), and must be one of the two valid values.

    **Validates: Requirements 7.2**
    """
    json_str = session.model_dump_json()
    restored = SessionState.model_validate_json(json_str)

    assert isinstance(restored.preferred_mode, str), (
        f"Expected preferred_mode to be str, got {type(restored.preferred_mode)}"
    )
    assert restored.preferred_mode in ("voice", "text"), (
        f"Expected preferred_mode to be 'voice' or 'text', got {restored.preferred_mode!r}"
    )
