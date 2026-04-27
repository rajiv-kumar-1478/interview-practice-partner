# Feature: interview-practice-partner, Property 13: Role in Opening Message Starts Interview Directly
"""Property-based tests for the role-in-opening-message fast path.

Any opening message (INIT stage) that contains a clearly identifiable
supported role must transition the session directly to INTERVIEW stage
(skipping ROLE_SELECTION) and the reply must contain an interview question
(not a role clarification prompt).

Validates: Requirements 6.1
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from interview_practice_partner.domain.enums import Role, Stage
from interview_practice_partner.domain.models import SessionState
from interview_practice_partner.services.session import SessionService

# ---------------------------------------------------------------------------
# Supported roles and their representative keyword phrases
# ---------------------------------------------------------------------------

# Each entry is (Role enum value, list of message templates containing that role)
_ROLE_MESSAGE_TEMPLATES: dict[Role, list[str]] = {
    Role.SOFTWARE_ENGINEER: [
        "I want to practice for software engineer",
        "software engineer interview please",
        "I'm preparing for a software developer role",
        "help me with a developer interview",
        "I need to practice as an engineer",
        "swe interview practice",
        "coding interview practice",
        "programming interview",
    ],
    Role.SALES_REPRESENTATIVE: [
        "I want to practice for sales representative",
        "sales representative interview please",
        "I'm preparing for a sales rep role",
        "help me with a sales interview",
        "sales practice session",
        "account executive interview",
        "business development interview",
        "bdr interview practice",
    ],
    Role.RETAIL_ASSOCIATE: [
        "I want to practice for retail associate",
        "retail associate interview please",
        "I'm preparing for a retail role",
        "help me with a retail interview",
        "shop assistant interview practice",
        "store associate interview",
        "customer service interview",
        "cashier interview practice",
    ],
}

# Flatten all (role, message) pairs for use in strategies
_ALL_ROLE_MESSAGES: list[tuple[Role, str]] = [
    (role, msg)
    for role, messages in _ROLE_MESSAGE_TEMPLATES.items()
    for msg in messages
]

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_nonempty_text = st.text(min_size=1, max_size=200)

# Strategy that produces (role, opening_message) pairs where the message
# clearly contains a supported role keyword.
_role_message_strategy = st.sampled_from(_ALL_ROLE_MESSAGES)

# Strategy for optional prefix/suffix text to add around the role keyword
# (simulates realistic user messages with surrounding context)
_prefix_strategy = st.one_of(
    st.just(""),
    st.just("Hi! "),
    st.just("Hello, "),
    st.just("Hey there, "),
    st.just("Good morning, "),
)

_suffix_strategy = st.one_of(
    st.just(""),
    st.just(" please"),
    st.just(" thanks"),
    st.just(" thank you"),
    st.just(". Can you help?"),
)

# Strategy for a fresh INIT session (simulates a brand-new user)
_init_session_strategy = st.builds(
    SessionState,
    session_id=st.just(str(uuid.uuid4())),
    phone_number=st.from_regex(r"\+1[0-9]{10}", fullmatch=True),
    stage=st.just(Stage.INIT),
    role=st.just(Role.UNKNOWN),
    questions=st.just([]),
    responses=st.just([]),
    off_topic_count=st.just(0),
    consecutive_out_of_scope_count=st.just(0),
    clarification_turn_count=st.just(0),
    requested_short_session=st.just(False),
    is_complete=st.just(False),
    created_at=st.datetimes(),
    updated_at=st.datetimes(),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_INTERVIEW_QUESTION = "Tell me about a time you faced a challenging technical problem and how you resolved it."


def _make_session_service() -> SessionService:
    """Build a SessionService with all dependencies mocked.

    - LLM mock: returns a role-selection JSON response (not used in fast path)
    - Interview service mock: generate_question returns a sample interview question
    - Feedback service mock: not used in this test
    """
    mock_llm = AsyncMock()
    # Role-selection LLM response (only used if fast path is NOT taken)
    mock_llm.complete.return_value = '{"role": "unknown", "confidence": "low", "message": "Which role are you preparing for?"}'

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_role_selection_prompt.return_value = [
        {"role": "system", "content": "You are a role selection assistant."},
        {"role": "user", "content": "Which role?"},
    ]

    mock_interview_service = AsyncMock()
    mock_interview_service.generate_question.return_value = _SAMPLE_INTERVIEW_QUESTION

    mock_feedback_service = AsyncMock()
    mock_feedback_service.generate_feedback_report.return_value = (
        "Here is your feedback.",
        MagicMock(spec=SessionState),
    )

    return SessionService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
        interview_service=mock_interview_service,
        feedback_service=mock_feedback_service,
    )


# ---------------------------------------------------------------------------
# Property 13a: Opening message with a supported role transitions to INTERVIEW
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    session=_init_session_strategy,
    role_and_message=_role_message_strategy,
    prefix=_prefix_strategy,
    suffix=_suffix_strategy,
)
@settings(max_examples=100)
async def test_property_13_role_in_opening_message_transitions_to_interview(
    session: SessionState,
    role_and_message: tuple[Role, str],
    prefix: str,
    suffix: str,
) -> None:
    """Property 13: Role in Opening Message Starts Interview Directly.

    Any opening message (INIT stage) containing a clearly identifiable
    supported role must transition the session directly to INTERVIEW stage
    without going through ROLE_SELECTION.

    **Validates: Requirements 6.1**
    """
    expected_role, base_message = role_and_message
    opening_message = f"{prefix}{base_message}{suffix}"

    # Confirm we start from INIT
    assert session.stage == Stage.INIT, (
        f"Strategy produced a non-INIT session: {session.stage!r}"
    )

    service = _make_session_service()

    reply, updated_session = await service.transition(session, opening_message)

    # The session must transition directly to INTERVIEW (not ROLE_SELECTION)
    assert updated_session.stage == Stage.INTERVIEW, (
        f"Expected session to transition to INTERVIEW for opening message "
        f"{opening_message!r} (role: {expected_role.value}), "
        f"but got stage {updated_session.stage!r}"
    )

    # The session must NOT pass through ROLE_SELECTION
    assert updated_session.stage != Stage.ROLE_SELECTION, (
        f"Session incorrectly transitioned to ROLE_SELECTION for opening message "
        f"{opening_message!r} — the fast path should skip role selection."
    )


# ---------------------------------------------------------------------------
# Property 13b: Opening message with a supported role sets the correct role
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    session=_init_session_strategy,
    role_and_message=_role_message_strategy,
)
@settings(max_examples=100)
async def test_property_13_role_in_opening_message_sets_correct_role(
    session: SessionState,
    role_and_message: tuple[Role, str],
) -> None:
    """Property 13: Role in Opening Message Starts Interview Directly.

    The session role must be set to the detected supported role (not UNKNOWN)
    when the opening message contains a clearly identifiable role.

    **Validates: Requirements 6.1**
    """
    expected_role, opening_message = role_and_message

    assert session.stage == Stage.INIT

    service = _make_session_service()

    reply, updated_session = await service.transition(session, opening_message)

    # The role must be set (not UNKNOWN)
    assert updated_session.role != Role.UNKNOWN, (
        f"Expected role to be set for opening message {opening_message!r}, "
        f"but got Role.UNKNOWN"
    )

    # The role must match the expected role for this message
    assert updated_session.role == expected_role, (
        f"Expected role {expected_role.value!r} for opening message "
        f"{opening_message!r}, but got {updated_session.role.value!r}"
    )


# ---------------------------------------------------------------------------
# Property 13c: Reply contains an interview question (not a role clarification)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    session=_init_session_strategy,
    role_and_message=_role_message_strategy,
    prefix=_prefix_strategy,
    suffix=_suffix_strategy,
)
@settings(max_examples=100)
async def test_property_13_reply_contains_interview_question(
    session: SessionState,
    role_and_message: tuple[Role, str],
    prefix: str,
    suffix: str,
) -> None:
    """Property 13: Role in Opening Message Starts Interview Directly.

    The reply to an opening message containing a supported role must include
    the interview question text (not a role clarification prompt asking the
    user to specify their role).

    **Validates: Requirements 6.1**
    """
    expected_role, base_message = role_and_message
    opening_message = f"{prefix}{base_message}{suffix}"

    assert session.stage == Stage.INIT

    service = _make_session_service()

    reply, updated_session = await service.transition(session, opening_message)

    # The reply must be a non-empty string
    assert isinstance(reply, str), (
        f"Expected transition to return a str reply, got {type(reply)!r}"
    )
    assert len(reply.strip()) > 0, (
        "Expected a non-empty reply for an opening message with a role."
    )

    # The reply must contain the interview question text
    # (our mock returns _SAMPLE_INTERVIEW_QUESTION)
    assert _SAMPLE_INTERVIEW_QUESTION in reply, (
        f"Expected reply to contain the interview question for opening message "
        f"{opening_message!r}, but got: {reply!r}"
    )

    # The reply must NOT be a role clarification prompt
    # (i.e., it should not ask the user to specify a role)
    reply_lower = reply.lower()
    role_clarification_phrases = [
        "which role",
        "what role",
        "please specify",
        "please tell me the role",
        "what position",
        "which position",
    ]
    is_clarification = any(phrase in reply_lower for phrase in role_clarification_phrases)
    assert not is_clarification, (
        f"Reply looks like a role clarification prompt for opening message "
        f"{opening_message!r}: {reply!r}"
    )


# ---------------------------------------------------------------------------
# Property 13d: Fast path works for all three supported roles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(
    session=_init_session_strategy,
    role=st.sampled_from([
        Role.SOFTWARE_ENGINEER,
        Role.SALES_REPRESENTATIVE,
        Role.RETAIL_ASSOCIATE,
    ]),
)
@settings(max_examples=100)
async def test_property_13_fast_path_works_for_all_supported_roles(
    session: SessionState,
    role: Role,
) -> None:
    """Property 13: Role in Opening Message Starts Interview Directly.

    The fast path must work for all three supported roles: SOFTWARE_ENGINEER,
    SALES_REPRESENTATIVE, and RETAIL_ASSOCIATE.

    **Validates: Requirements 6.1**
    """
    assert session.stage == Stage.INIT

    # Use the canonical role name as the opening message keyword
    role_keyword = role.value.replace("_", " ")  # e.g. "software engineer"
    opening_message = f"I want to practice for {role_keyword}"

    service = _make_session_service()

    reply, updated_session = await service.transition(session, opening_message)

    # Must transition directly to INTERVIEW
    assert updated_session.stage == Stage.INTERVIEW, (
        f"Expected INTERVIEW stage for role {role.value!r} with message "
        f"{opening_message!r}, but got {updated_session.stage!r}"
    )

    # Role must be set correctly
    assert updated_session.role == role, (
        f"Expected role {role.value!r} but got {updated_session.role.value!r}"
    )

    # Reply must contain the interview question
    assert _SAMPLE_INTERVIEW_QUESTION in reply, (
        f"Expected reply to contain interview question for role {role.value!r}, "
        f"but got: {reply!r}"
    )

    # generate_question must have been called (fast path invokes it)
    service._interview.generate_question.assert_called()
