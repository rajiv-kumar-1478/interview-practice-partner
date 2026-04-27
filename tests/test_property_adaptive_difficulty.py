# Feature: interview-practice-partner, Property 20: Adaptive Difficulty Signal Is Propagated to Prompt
"""Property-based tests for adaptive difficulty signal propagation.

When response evaluation returns ``increase`` or ``decrease``, the next
question generation prompt must contain the corresponding difficulty
instruction.  This validates that the PromptBuilder correctly translates
the difficulty signal into actionable guidance for the LLM.

Validates: Requirements 11.2, 11.3
"""

from datetime import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from interview_practice_partner.domain.enums import QuestionType, Role, Stage
from interview_practice_partner.domain.models import Question, SessionState, UserResponse
from interview_practice_partner.llm.prompt_builder import PromptBuilder

# ---------------------------------------------------------------------------
# Expected instruction fragments for each difficulty signal
# ---------------------------------------------------------------------------

# These substrings are taken directly from PromptBuilder.build_question_generation_prompt
# and must appear in the system prompt when the corresponding signal is passed.
_INCREASE_FRAGMENT = "Increase"   # "Increase* the difficulty and depth"
_DECREASE_FRAGMENT = "Decrease"   # "Decrease* the difficulty of this question"
_MAINTAIN_FRAGMENT = "Maintain"   # "Maintain the current difficulty level"

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_nonempty_text = st.text(min_size=1, max_size=200)
_any_text = st.text(min_size=0, max_size=200)

_question_strategy = st.builds(
    Question,
    question_id=_nonempty_text,
    text=_any_text,
    question_type=st.sampled_from(list(QuestionType)),
    asked_at=st.datetimes(),
    skipped=st.booleans(),
)

_user_response_strategy = st.builds(
    UserResponse,
    response_id=_nonempty_text,
    question_id=_nonempty_text,
    text=_any_text,
    word_count=st.integers(min_value=0, max_value=1000),
    is_off_topic=st.booleans(),
    received_at=st.datetimes(),
)

_session_strategy = st.builds(
    SessionState,
    session_id=_nonempty_text,
    phone_number=_nonempty_text,
    stage=st.just(Stage.INTERVIEW),
    role=st.sampled_from([r for r in Role if r != Role.UNKNOWN]),
    questions=st.lists(_question_strategy, min_size=0, max_size=8),
    responses=st.lists(_user_response_strategy, min_size=0, max_size=8),
    off_topic_count=st.integers(min_value=0, max_value=5),
    consecutive_out_of_scope_count=st.integers(min_value=0, max_value=5),
    clarification_turn_count=st.integers(min_value=0, max_value=3),
    requested_short_session=st.booleans(),
    is_complete=st.just(False),
    created_at=st.datetimes(),
    updated_at=st.datetimes(),
)

_question_type_strategy = st.sampled_from(list(QuestionType))


def _get_system_content(messages: list[dict[str, str]]) -> str:
    """Extract the system message content from a messages list."""
    for msg in messages:
        if msg.get("role") == "system":
            return msg.get("content", "")
    return ""


# ---------------------------------------------------------------------------
# Property 20a: "increase" signal → increase instruction present
# ---------------------------------------------------------------------------


@given(session=_session_strategy, question_type=_question_type_strategy)
@settings(max_examples=100)
def test_property_20_increase_signal_propagated_to_prompt(
    session: SessionState,
    question_type: QuestionType,
) -> None:
    """Property 20: Adaptive Difficulty Signal Is Propagated to Prompt.

    When ``difficulty_signal="increase"`` is passed to
    ``build_question_generation_prompt``, the resulting system prompt must
    contain an instruction to increase difficulty, and must NOT contain an
    instruction to decrease difficulty.

    Validates: Requirements 11.2, 11.3
    """
    # Feature: interview-practice-partner, Property 20: Adaptive Difficulty Signal Is Propagated to Prompt
    builder = PromptBuilder()
    messages = builder.build_question_generation_prompt(
        session=session,
        question_type=question_type,
        difficulty_signal="increase",
    )

    system_content = _get_system_content(messages)

    assert _INCREASE_FRAGMENT in system_content, (
        f"Expected increase instruction ('{_INCREASE_FRAGMENT}') in system prompt "
        f"when difficulty_signal='increase', but it was absent.\n"
        f"System prompt (first 400 chars): {system_content[:400]}"
    )
    assert _DECREASE_FRAGMENT not in system_content, (
        f"Decrease instruction ('{_DECREASE_FRAGMENT}') must NOT appear in system "
        f"prompt when difficulty_signal='increase'.\n"
        f"System prompt (first 400 chars): {system_content[:400]}"
    )


# ---------------------------------------------------------------------------
# Property 20b: "decrease" signal → decrease instruction present
# ---------------------------------------------------------------------------


@given(session=_session_strategy, question_type=_question_type_strategy)
@settings(max_examples=100)
def test_property_20_decrease_signal_propagated_to_prompt(
    session: SessionState,
    question_type: QuestionType,
) -> None:
    """Property 20: Adaptive Difficulty Signal Is Propagated to Prompt.

    When ``difficulty_signal="decrease"`` is passed to
    ``build_question_generation_prompt``, the resulting system prompt must
    contain an instruction to decrease difficulty, and must NOT contain an
    instruction to increase difficulty.

    Validates: Requirements 11.2, 11.3
    """
    # Feature: interview-practice-partner, Property 20: Adaptive Difficulty Signal Is Propagated to Prompt
    builder = PromptBuilder()
    messages = builder.build_question_generation_prompt(
        session=session,
        question_type=question_type,
        difficulty_signal="decrease",
    )

    system_content = _get_system_content(messages)

    assert _DECREASE_FRAGMENT in system_content, (
        f"Expected decrease instruction ('{_DECREASE_FRAGMENT}') in system prompt "
        f"when difficulty_signal='decrease', but it was absent.\n"
        f"System prompt (first 400 chars): {system_content[:400]}"
    )
    assert _INCREASE_FRAGMENT not in system_content, (
        f"Increase instruction ('{_INCREASE_FRAGMENT}') must NOT appear in system "
        f"prompt when difficulty_signal='decrease'.\n"
        f"System prompt (first 400 chars): {system_content[:400]}"
    )


# ---------------------------------------------------------------------------
# Property 20c: "maintain" / None signal → maintain instruction, no increase/decrease
# ---------------------------------------------------------------------------


@given(
    session=_session_strategy,
    question_type=_question_type_strategy,
    signal=st.one_of(st.just("maintain"), st.none()),
)
@settings(max_examples=100)
def test_property_20_maintain_or_none_signal_propagated_to_prompt(
    session: SessionState,
    question_type: QuestionType,
    signal: str | None,
) -> None:
    """Property 20: Adaptive Difficulty Signal Is Propagated to Prompt.

    When ``difficulty_signal`` is ``"maintain"`` or ``None``, the resulting
    system prompt must contain the maintain instruction and must NOT contain
    increase or decrease instructions.

    Validates: Requirements 11.2, 11.3
    """
    # Feature: interview-practice-partner, Property 20: Adaptive Difficulty Signal Is Propagated to Prompt
    builder = PromptBuilder()
    messages = builder.build_question_generation_prompt(
        session=session,
        question_type=question_type,
        difficulty_signal=signal,
    )

    system_content = _get_system_content(messages)

    assert _MAINTAIN_FRAGMENT in system_content, (
        f"Expected maintain instruction ('{_MAINTAIN_FRAGMENT}') in system prompt "
        f"when difficulty_signal={signal!r}, but it was absent.\n"
        f"System prompt (first 400 chars): {system_content[:400]}"
    )
    assert _INCREASE_FRAGMENT not in system_content, (
        f"Increase instruction ('{_INCREASE_FRAGMENT}') must NOT appear when "
        f"difficulty_signal={signal!r}.\n"
        f"System prompt (first 400 chars): {system_content[:400]}"
    )
    assert _DECREASE_FRAGMENT not in system_content, (
        f"Decrease instruction ('{_DECREASE_FRAGMENT}') must NOT appear when "
        f"difficulty_signal={signal!r}.\n"
        f"System prompt (first 400 chars): {system_content[:400]}"
    )


# ---------------------------------------------------------------------------
# Property 20d: exactly one difficulty instruction appears for any valid signal
# ---------------------------------------------------------------------------


@given(
    session=_session_strategy,
    question_type=_question_type_strategy,
    signal=st.one_of(
        st.just("increase"),
        st.just("decrease"),
        st.just("maintain"),
        st.none(),
    ),
)
@settings(max_examples=100)
def test_property_20_exactly_one_difficulty_instruction_present(
    session: SessionState,
    question_type: QuestionType,
    signal: str | None,
) -> None:
    """Property 20: Adaptive Difficulty Signal Is Propagated to Prompt.

    For any valid difficulty signal (including None), the system prompt
    contains exactly one difficulty instruction — the one corresponding to
    the signal.  The other two instructions must be absent.

    Validates: Requirements 11.2, 11.3
    """
    # Feature: interview-practice-partner, Property 20: Adaptive Difficulty Signal Is Propagated to Prompt
    builder = PromptBuilder()
    messages = builder.build_question_generation_prompt(
        session=session,
        question_type=question_type,
        difficulty_signal=signal,
    )

    system_content = _get_system_content(messages)

    has_increase = _INCREASE_FRAGMENT in system_content
    has_decrease = _DECREASE_FRAGMENT in system_content
    has_maintain = _MAINTAIN_FRAGMENT in system_content

    active_instructions = sum([has_increase, has_decrease, has_maintain])

    assert active_instructions == 1, (
        f"Expected exactly one difficulty instruction in the system prompt for "
        f"difficulty_signal={signal!r}, but found {active_instructions} "
        f"(increase={has_increase}, decrease={has_decrease}, maintain={has_maintain}).\n"
        f"System prompt (first 400 chars): {system_content[:400]}"
    )

    # Verify the correct instruction is the one present
    if signal == "increase":
        assert has_increase, "Expected 'increase' instruction to be the active one."
    elif signal == "decrease":
        assert has_decrease, "Expected 'decrease' instruction to be the active one."
    else:
        assert has_maintain, "Expected 'maintain' instruction to be the active one."
