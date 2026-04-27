# Feature: interview-practice-partner, Property 12: Outbound Messages Contain No Forbidden Formatting
"""Property-based tests for PromptBuilder outbound message formatting invariants.

No HTML tags, markdown headers, or fenced code blocks in any LLM-generated
text passed through PromptBuilder, regardless of the inputs provided.

Validates: Requirements 4.7, 10.2
"""

import re
from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from interview_practice_partner.domain.enums import QuestionType, Role, Stage
from interview_practice_partner.domain.models import Question, SessionState, UserResponse
from interview_practice_partner.llm.prompt_builder import PromptBuilder

# ---------------------------------------------------------------------------
# Forbidden formatting patterns
# ---------------------------------------------------------------------------

HTML_TAG_PATTERN = re.compile(r"<[a-zA-Z][^>]*>")
MARKDOWN_HEADER_PATTERN = re.compile(r"^#{1,6}\s", re.MULTILINE)
FENCED_CODE_BLOCK_PATTERN = re.compile(r"```")


def assert_no_forbidden_formatting(messages: list[dict[str, str]]) -> None:
    """Assert that no message contains HTML tags, markdown headers, or fenced code blocks."""
    for msg in messages:
        content = msg.get("content", "")
        assert not HTML_TAG_PATTERN.search(content), (
            f"HTML tag found in {msg['role']} message: {content[:200]}"
        )
        assert not MARKDOWN_HEADER_PATTERN.search(content), (
            f"Markdown header found in {msg['role']} message: {content[:200]}"
        )
        assert not FENCED_CODE_BLOCK_PATTERN.search(content), (
            f"Fenced code block found in {msg['role']} message: {content[:200]}"
        )


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_text_strategy = st.text(min_size=0, max_size=500)
_nonempty_text_strategy = st.text(min_size=1, max_size=200)

_question_strategy = st.builds(
    Question,
    question_id=_nonempty_text_strategy,
    text=_text_strategy,
    question_type=st.sampled_from(list(QuestionType)),
    asked_at=st.datetimes(),
    skipped=st.booleans(),
)

_user_response_strategy = st.builds(
    UserResponse,
    response_id=_nonempty_text_strategy,
    question_id=_nonempty_text_strategy,
    text=_text_strategy,
    word_count=st.integers(min_value=0, max_value=1000),
    is_off_topic=st.booleans(),
    received_at=st.datetimes(),
)

_session_state_strategy = st.builds(
    SessionState,
    session_id=_nonempty_text_strategy,
    phone_number=_nonempty_text_strategy,
    stage=st.sampled_from(list(Stage)),
    role=st.sampled_from(list(Role)),
    questions=st.lists(_question_strategy, min_size=0, max_size=5),
    responses=st.lists(_user_response_strategy, min_size=0, max_size=5),
    off_topic_count=st.integers(min_value=0, max_value=10),
    consecutive_out_of_scope_count=st.integers(min_value=0, max_value=10),
    clarification_turn_count=st.integers(min_value=0, max_value=5),
    requested_short_session=st.booleans(),
    is_complete=st.booleans(),
    created_at=st.datetimes(),
    updated_at=st.datetimes(),
)

_difficulty_signal_strategy = st.one_of(
    st.none(),
    st.just("increase"),
    st.just("maintain"),
    st.just("decrease"),
)


# ---------------------------------------------------------------------------
# Property 12: build_role_selection_prompt
# ---------------------------------------------------------------------------


@given(
    user_message=_text_strategy,
    clarification_turn_count=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=100)
def test_property_12_role_selection_prompt_no_forbidden_formatting(
    user_message: str,
    clarification_turn_count: int,
) -> None:
    """Property 12: Outbound Messages Contain No Forbidden Formatting.

    For any user_message and clarification_turn_count, the messages produced
    by build_role_selection_prompt contain no HTML tags, markdown headers, or
    fenced code blocks.

    Validates: Requirements 4.7, 10.2
    """
    # Feature: interview-practice-partner, Property 12: Outbound Messages Contain No Forbidden Formatting
    builder = PromptBuilder()
    messages = builder.build_role_selection_prompt(
        user_message=user_message,
        clarification_turn_count=clarification_turn_count,
    )
    assert_no_forbidden_formatting(messages)


# ---------------------------------------------------------------------------
# Property 12: build_question_generation_prompt
# ---------------------------------------------------------------------------


@given(
    session=_session_state_strategy,
    question_type=st.sampled_from(list(QuestionType)),
    difficulty_signal=_difficulty_signal_strategy,
)
@settings(max_examples=100)
def test_property_12_question_generation_prompt_no_forbidden_formatting(
    session: SessionState,
    question_type: QuestionType,
    difficulty_signal: str | None,
) -> None:
    """Property 12: Outbound Messages Contain No Forbidden Formatting.

    For any SessionState, QuestionType, and difficulty_signal, the messages
    produced by build_question_generation_prompt contain no HTML tags,
    markdown headers, or fenced code blocks.

    Validates: Requirements 4.7, 10.2
    """
    # Feature: interview-practice-partner, Property 12: Outbound Messages Contain No Forbidden Formatting
    builder = PromptBuilder()
    messages = builder.build_question_generation_prompt(
        session=session,
        question_type=question_type,
        difficulty_signal=difficulty_signal,
    )
    assert_no_forbidden_formatting(messages)


# ---------------------------------------------------------------------------
# Property 12: build_response_evaluation_prompt
# ---------------------------------------------------------------------------


@given(
    question=_question_strategy,
    response=_user_response_strategy,
    session=_session_state_strategy,
)
@settings(max_examples=100)
def test_property_12_response_evaluation_prompt_no_forbidden_formatting(
    question: Question,
    response: UserResponse,
    session: SessionState,
) -> None:
    """Property 12: Outbound Messages Contain No Forbidden Formatting.

    For any Question, UserResponse, and SessionState, the messages produced
    by build_response_evaluation_prompt contain no HTML tags, markdown headers,
    or fenced code blocks.

    Validates: Requirements 4.7, 10.2
    """
    # Feature: interview-practice-partner, Property 12: Outbound Messages Contain No Forbidden Formatting
    builder = PromptBuilder()
    messages = builder.build_response_evaluation_prompt(
        question=question,
        response=response,
        session=session,
    )
    assert_no_forbidden_formatting(messages)


# ---------------------------------------------------------------------------
# Property 12: build_feedback_prompt
# ---------------------------------------------------------------------------


@given(
    session=st.builds(
        SessionState,
        session_id=_nonempty_text_strategy,
        phone_number=_nonempty_text_strategy,
        stage=st.sampled_from(list(Stage)),
        role=st.sampled_from(list(Role)),
        questions=st.lists(_question_strategy, min_size=0, max_size=5),
        responses=st.lists(_user_response_strategy, min_size=0, max_size=5),
        off_topic_count=st.integers(min_value=0, max_value=10),
        consecutive_out_of_scope_count=st.integers(min_value=0, max_value=10),
        clarification_turn_count=st.integers(min_value=0, max_value=5),
        requested_short_session=st.booleans(),
        is_complete=st.booleans(),
        created_at=st.datetimes(),
        updated_at=st.datetimes(),
    ),
)
@settings(max_examples=100)
def test_property_12_feedback_prompt_no_forbidden_formatting(
    session: SessionState,
) -> None:
    """Property 12: Outbound Messages Contain No Forbidden Formatting.

    For any SessionState (with arbitrary role, questions, responses, and
    off_topic_count), the messages produced by build_feedback_prompt contain
    no HTML tags, markdown headers, or fenced code blocks.

    Validates: Requirements 4.7, 10.2
    """
    # Feature: interview-practice-partner, Property 12: Outbound Messages Contain No Forbidden Formatting
    builder = PromptBuilder()
    messages = builder.build_feedback_prompt(session=session)
    assert_no_forbidden_formatting(messages)
