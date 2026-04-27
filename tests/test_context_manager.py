"""Unit tests for ContextWindowManager.

Covers:
- build_context returns correct structure: system → summary → Q&A pairs → user message
- Sliding window returns only the last N pairs when session is within window
- Rolling summary is prepended when session exceeds window
- Token budget trimming removes oldest turns first
- maybe_summarise calls LLM and stores result in session.context_summary
- maybe_summarise does nothing when session is within window size
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from interview_practice_partner.domain.enums import QuestionType, Role, Stage
from interview_practice_partner.domain.models import Question, SessionState, UserResponse
from interview_practice_partner.llm.context_manager import ContextWindowManager, _count_tokens

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
PHONE = "+15550001234"


def make_session(
    questions: list[Question] | None = None,
    responses: list[UserResponse] | None = None,
    context_summary: str | None = None,
) -> SessionState:
    return SessionState(
        session_id="sess-001",
        phone_number=PHONE,
        stage=Stage.INTERVIEW,
        role=Role.SOFTWARE_ENGINEER,
        questions=questions or [],
        responses=responses or [],
        context_summary=context_summary,
        created_at=NOW,
        updated_at=NOW,
    )


def make_question(idx: int, text: str | None = None) -> Question:
    return Question(
        question_id=f"q-{idx:03d}",
        text=text or f"Question number {idx}?",
        question_type=QuestionType.BEHAVIOURAL,
        asked_at=NOW,
    )


def make_response(question_id: str, text: str | None = None) -> UserResponse:
    return UserResponse(
        response_id=f"r-{question_id}",
        question_id=question_id,
        text=text or f"Response to {question_id}.",
        word_count=5,
        received_at=NOW,
    )


def make_settings(window_size: int = 6, max_tokens: int = 8000) -> MagicMock:
    settings = MagicMock()
    settings.llm_context_window_size = window_size
    settings.llm_max_context_tokens = max_tokens
    return settings


def make_llm_client(summary_response: str = "This is a summary.") -> AsyncMock:
    client = AsyncMock()
    client.complete = AsyncMock(return_value=summary_response)
    return client


# ---------------------------------------------------------------------------
# Token counting helper
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_empty_string_returns_one(self):
        assert _count_tokens("") == 1

    def test_four_chars_returns_one(self):
        assert _count_tokens("abcd") == 1

    def test_eight_chars_returns_two(self):
        assert _count_tokens("abcdefgh") == 2

    def test_long_text(self):
        text = "a" * 400
        assert _count_tokens(text) == 100


# ---------------------------------------------------------------------------
# build_context — structure
# ---------------------------------------------------------------------------


class TestBuildContextStructure:
    def test_first_message_is_system_prompt(self):
        manager = ContextWindowManager(make_llm_client(), make_settings())
        session = make_session()
        messages = manager.build_context(session, "System prompt text.", "Hello")
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt text."

    def test_last_message_is_current_user_message(self):
        manager = ContextWindowManager(make_llm_client(), make_settings())
        session = make_session()
        messages = manager.build_context(session, "System.", "Current user message")
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Current user message"

    def test_empty_session_returns_system_and_user_only(self):
        manager = ContextWindowManager(make_llm_client(), make_settings())
        session = make_session()
        messages = manager.build_context(session, "System.", "Hello")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_no_summary_when_context_summary_is_none(self):
        manager = ContextWindowManager(make_llm_client(), make_settings())
        session = make_session(context_summary=None)
        messages = manager.build_context(session, "System.", "Hello")
        # Only system + user (no summary)
        roles = [m["role"] for m in messages]
        # There should be no second system message for summary
        assert roles.count("system") == 1

    def test_summary_message_inserted_when_context_summary_present(self):
        manager = ContextWindowManager(make_llm_client(), make_settings())
        session = make_session(context_summary="Earlier the candidate discussed X.")
        messages = manager.build_context(session, "System.", "Hello")
        # Should have: system, system (summary), user
        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) == 2
        assert "Earlier the candidate discussed X." in system_messages[1]["content"]

    def test_summary_message_comes_before_qa_pairs(self):
        manager = ContextWindowManager(make_llm_client(), make_settings())
        q = make_question(1)
        r = make_response(q.question_id)
        session = make_session(
            questions=[q],
            responses=[r],
            context_summary="Summary of earlier turns.",
        )
        messages = manager.build_context(session, "System.", "Hello")
        # Order: system, system(summary), assistant(Q), user(A), user(current)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "system"
        assert "Summary of earlier turns." in messages[1]["content"]
        assert messages[2]["role"] == "assistant"

    def test_qa_pairs_appear_as_assistant_then_user(self):
        manager = ContextWindowManager(make_llm_client(), make_settings())
        q = make_question(1, "Tell me about yourself.")
        r = make_response(q.question_id, "I am a software engineer.")
        session = make_session(questions=[q], responses=[r])
        messages = manager.build_context(session, "System.", "Next message")
        # system, assistant(Q), user(A), user(current)
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Tell me about yourself."
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "I am a software engineer."

    def test_question_without_response_included_as_assistant_only(self):
        manager = ContextWindowManager(make_llm_client(), make_settings())
        q = make_question(1, "What is your greatest strength?")
        session = make_session(questions=[q], responses=[])
        messages = manager.build_context(session, "System.", "My answer")
        # system, assistant(Q), user(current)
        assert len(messages) == 3
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "What is your greatest strength?"


# ---------------------------------------------------------------------------
# build_context — sliding window
# ---------------------------------------------------------------------------


class TestBuildContextSlidingWindow:
    def test_within_window_all_pairs_included(self):
        manager = ContextWindowManager(make_llm_client(), make_settings(window_size=6))
        questions = [make_question(i) for i in range(1, 4)]
        responses = [make_response(q.question_id) for q in questions]
        session = make_session(questions=questions, responses=responses)
        messages = manager.build_context(session, "System.", "Hello")
        # system + 3*(assistant+user) + user(current) = 1 + 6 + 1 = 8
        assert len(messages) == 8

    def test_exceeds_window_only_last_n_pairs_included(self):
        window_size = 3
        manager = ContextWindowManager(make_llm_client(), make_settings(window_size=window_size))
        questions = [make_question(i) for i in range(1, 8)]  # 7 questions
        responses = [make_response(q.question_id) for q in questions]
        session = make_session(questions=questions, responses=responses)
        messages = manager.build_context(session, "System.", "Hello")
        # system + 3*(assistant+user) + user(current) = 1 + 6 + 1 = 8
        assert len(messages) == 8

    def test_exceeds_window_most_recent_pairs_are_included(self):
        window_size = 2
        manager = ContextWindowManager(make_llm_client(), make_settings(window_size=window_size))
        questions = [make_question(i, f"Question {i}?") for i in range(1, 5)]
        responses = [make_response(q.question_id, f"Answer {i}.") for i, q in enumerate(questions, 1)]
        session = make_session(questions=questions, responses=responses)
        messages = manager.build_context(session, "System.", "Hello")
        # Only the last 2 Q&A pairs should appear
        contents = [m["content"] for m in messages]
        assert "Question 3?" in contents
        assert "Question 4?" in contents
        assert "Question 1?" not in contents
        assert "Question 2?" not in contents

    def test_window_size_one_includes_only_last_pair(self):
        manager = ContextWindowManager(make_llm_client(), make_settings(window_size=1))
        questions = [make_question(i, f"Q{i}?") for i in range(1, 4)]
        responses = [make_response(q.question_id, f"A{i}.") for i, q in enumerate(questions, 1)]
        session = make_session(questions=questions, responses=responses)
        messages = manager.build_context(session, "System.", "Hello")
        contents = [m["content"] for m in messages]
        assert "Q3?" in contents
        assert "Q1?" not in contents
        assert "Q2?" not in contents


# ---------------------------------------------------------------------------
# build_context — token budget trimming
# ---------------------------------------------------------------------------


class TestBuildContextTokenBudget:
    def test_trim_oldest_turns_when_budget_exceeded(self):
        # Use a very small token budget to force trimming
        # Each message content is ~20 chars = ~5 tokens
        # System prompt: ~10 tokens, user message: ~5 tokens
        # Budget: 30 tokens → only ~15 tokens left for Q&A pairs (~3 messages)
        manager = ContextWindowManager(
            make_llm_client(),
            make_settings(window_size=10, max_tokens=30),
        )
        # Create 5 Q&A pairs with short content
        questions = [make_question(i, f"Q{i}?") for i in range(1, 6)]
        responses = [make_response(q.question_id, f"A{i}.") for i, q in enumerate(questions, 1)]
        session = make_session(questions=questions, responses=responses)
        messages = manager.build_context(session, "System.", "Hello")
        # The oldest turns should have been trimmed
        contents = [m["content"] for m in messages]
        # Q5 (most recent) should still be present
        assert "Q5?" in contents

    def test_zero_budget_for_qa_returns_only_system_and_user(self):
        # Budget so small that even the system + user messages consume it all
        manager = ContextWindowManager(
            make_llm_client(),
            make_settings(window_size=6, max_tokens=1),
        )
        questions = [make_question(1, "Tell me about yourself.")]
        responses = [make_response(questions[0].question_id, "I am a developer.")]
        session = make_session(questions=questions, responses=responses)
        messages = manager.build_context(session, "System.", "Hello")
        # No Q&A pairs should be included
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Hello"
        # No assistant messages
        assert not any(m["role"] == "assistant" for m in messages)

    def test_all_pairs_fit_within_budget_no_trimming(self):
        # Large budget — nothing should be trimmed
        manager = ContextWindowManager(
            make_llm_client(),
            make_settings(window_size=6, max_tokens=8000),
        )
        questions = [make_question(i) for i in range(1, 4)]
        responses = [make_response(q.question_id) for q in questions]
        session = make_session(questions=questions, responses=responses)
        messages = manager.build_context(session, "System.", "Hello")
        # All 3 Q&A pairs should be present
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_messages) == 3


# ---------------------------------------------------------------------------
# maybe_summarise
# ---------------------------------------------------------------------------


class TestMaybeSummarise:
    @pytest.mark.asyncio
    async def test_does_nothing_when_within_window(self):
        llm_client = make_llm_client()
        manager = ContextWindowManager(llm_client, make_settings(window_size=6))
        questions = [make_question(i) for i in range(1, 4)]  # 3 pairs, window=6
        responses = [make_response(q.question_id) for q in questions]
        session = make_session(questions=questions, responses=responses)

        await manager.maybe_summarise(session)

        # LLM should NOT have been called
        llm_client.complete.assert_not_called()
        assert session.context_summary is None

    @pytest.mark.asyncio
    async def test_calls_llm_when_exceeds_window(self):
        llm_client = make_llm_client(summary_response="Summary of older turns.")
        manager = ContextWindowManager(llm_client, make_settings(window_size=3))
        questions = [make_question(i) for i in range(1, 6)]  # 5 pairs, window=3
        responses = [make_response(q.question_id) for q in questions]
        session = make_session(questions=questions, responses=responses)

        await manager.maybe_summarise(session)

        llm_client.complete.assert_called_once()
        assert session.context_summary == "Summary of older turns."

    @pytest.mark.asyncio
    async def test_stores_summary_in_session(self):
        llm_client = make_llm_client(summary_response="  Trimmed summary.  ")
        manager = ContextWindowManager(llm_client, make_settings(window_size=2))
        questions = [make_question(i) for i in range(1, 5)]  # 4 pairs, window=2
        responses = [make_response(q.question_id) for q in questions]
        session = make_session(questions=questions, responses=responses)

        await manager.maybe_summarise(session)

        # Summary should be stripped
        assert session.context_summary == "Trimmed summary."

    @pytest.mark.asyncio
    async def test_replaces_existing_summary(self):
        llm_client = make_llm_client(summary_response="New summary.")
        manager = ContextWindowManager(llm_client, make_settings(window_size=2))
        questions = [make_question(i) for i in range(1, 5)]
        responses = [make_response(q.question_id) for q in questions]
        session = make_session(
            questions=questions,
            responses=responses,
            context_summary="Old summary.",
        )

        await manager.maybe_summarise(session)

        assert session.context_summary == "New summary."

    @pytest.mark.asyncio
    async def test_summarises_only_older_turns(self):
        """The LLM should only receive the turns outside the window."""
        llm_client = make_llm_client(summary_response="Summary.")
        window_size = 2
        manager = ContextWindowManager(llm_client, make_settings(window_size=window_size))
        questions = [make_question(i, f"Question {i}?") for i in range(1, 5)]
        responses = [make_response(q.question_id, f"Answer {i}.") for i, q in enumerate(questions, 1)]
        session = make_session(questions=questions, responses=responses)

        await manager.maybe_summarise(session)

        # The summarisation prompt should contain Q1 and Q2 (older turns)
        # but NOT Q3 and Q4 (within window)
        call_args = llm_client.complete.call_args
        messages_sent = call_args[1]["messages"] if call_args[1] else call_args[0][0]
        all_content = " ".join(m["content"] for m in messages_sent)
        assert "Question 1?" in all_content
        assert "Question 2?" in all_content
        assert "Question 3?" not in all_content
        assert "Question 4?" not in all_content

    @pytest.mark.asyncio
    async def test_does_not_raise_on_llm_failure(self):
        """If the LLM call fails, maybe_summarise should not raise."""
        llm_client = AsyncMock()
        llm_client.complete = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        manager = ContextWindowManager(llm_client, make_settings(window_size=2))
        questions = [make_question(i) for i in range(1, 5)]
        responses = [make_response(q.question_id) for q in questions]
        session = make_session(questions=questions, responses=responses, context_summary="Old summary.")

        # Should not raise
        await manager.maybe_summarise(session)

        # Old summary should be retained
        assert session.context_summary == "Old summary."

    @pytest.mark.asyncio
    async def test_exactly_at_window_size_does_not_summarise(self):
        """When total pairs == window_size, no summarisation should occur."""
        llm_client = make_llm_client()
        window_size = 4
        manager = ContextWindowManager(llm_client, make_settings(window_size=window_size))
        questions = [make_question(i) for i in range(1, window_size + 1)]
        responses = [make_response(q.question_id) for q in questions]
        session = make_session(questions=questions, responses=responses)

        await manager.maybe_summarise(session)

        llm_client.complete.assert_not_called()
        assert session.context_summary is None

    @pytest.mark.asyncio
    async def test_one_over_window_size_triggers_summarise(self):
        """When total pairs == window_size + 1, summarisation should occur."""
        llm_client = make_llm_client(summary_response="Summary.")
        window_size = 4
        manager = ContextWindowManager(llm_client, make_settings(window_size=window_size))
        questions = [make_question(i) for i in range(1, window_size + 2)]
        responses = [make_response(q.question_id) for q in questions]
        session = make_session(questions=questions, responses=responses)

        await manager.maybe_summarise(session)

        llm_client.complete.assert_called_once()
        assert session.context_summary == "Summary."
