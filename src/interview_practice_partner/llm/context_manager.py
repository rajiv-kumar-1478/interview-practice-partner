"""ContextWindowManager — sliding window + rolling summary for LLM context management.

Manages the conversation history passed to the LLM on each call.  Implements:
- Sliding window: the last N Q&A pairs are always included verbatim.
- Rolling summary: when the session exceeds the window size, older turns are
  summarised by an LLM call and stored in ``session.context_summary``.
- Token budget: trims the oldest verbatim turns first if the budget is exceeded.

LLM context structure per call:
  1. System Prompt (stage-specific)
  2. Context Summary (if session > window)
  3. Recent Q&A pairs (sliding window)
  4. Current user message
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from interview_practice_partner.domain.models import SessionState

if TYPE_CHECKING:
    from interview_practice_partner.config import Settings
    from interview_practice_partner.llm.client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counting helper
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Approximate token count using the heuristic: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


def _count_message_tokens(message: dict[str, str]) -> int:
    """Approximate token count for a single chat message dict."""
    return _count_tokens(message.get("content", ""))


# ---------------------------------------------------------------------------
# Summarisation prompt
# ---------------------------------------------------------------------------

_SUMMARISATION_SYSTEM_PROMPT = (
    "You are a concise summariser for an AI interview practice session. "
    "You will be given a series of interview question-and-answer pairs from an "
    "ongoing mock interview. Your task is to produce a brief, factual summary "
    "of the conversation so far that preserves the key topics covered, the "
    "candidate's main points, and any notable strengths or weaknesses observed. "
    "The summary will be used as context for the next part of the interview. "
    "Write in plain text only. Keep the summary under 300 words."
)


def _build_summarisation_prompt(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build a prompt asking the LLM to summarise a list of Q&A turns."""
    # Format the turns as a readable transcript
    lines: list[str] = []
    for msg in turns:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"Candidate: {content}")
        elif role == "assistant":
            lines.append(f"Interviewer: {content}")
        else:
            lines.append(content)

    transcript = "\n".join(lines)

    return [
        {"role": "system", "content": _SUMMARISATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Please summarise the following interview conversation:\n\n"
                f"{transcript}\n\n"
                "Provide a concise summary that captures the key topics and the "
                "candidate's responses."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# ContextWindowManager
# ---------------------------------------------------------------------------

class ContextWindowManager:
    """Manages the conversation history passed to the LLM on each call.

    Args:
        llm_client: The ``LLMClient`` instance used for summarisation calls.
        settings: Application ``Settings`` providing ``llm_context_window_size``
            and ``llm_max_context_tokens``.
    """

    def __init__(self, llm_client: "LLMClient", settings: "Settings") -> None:
        self._llm_client = llm_client
        self._window_size: int = settings.llm_context_window_size
        self._max_context_tokens: int = settings.llm_max_context_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def maybe_summarise(self, session: SessionState) -> None:
        """Summarise older turns if the session exceeds the window size.

        When the total number of Q&A pairs in the session exceeds
        ``llm_context_window_size``, the turns that fall outside the window
        are summarised by an LLM call.  The resulting summary is stored in
        ``session.context_summary``, replacing any previous summary.

        The session object is mutated in place; the caller is responsible for
        persisting the updated session.

        Args:
            session: The current ``SessionState``.
        """
        qa_pairs = self._build_qa_pairs(session)
        total_pairs = len(qa_pairs)

        if total_pairs <= self._window_size:
            # Nothing to summarise yet
            return

        # Turns that fall outside the sliding window
        older_turns_pairs = qa_pairs[: total_pairs - self._window_size]

        # Flatten the older pairs into a flat message list for summarisation
        older_messages: list[dict[str, str]] = []
        for q_msg, a_msg in older_turns_pairs:
            older_messages.append(q_msg)
            if a_msg is not None:
                older_messages.append(a_msg)

        if not older_messages:
            return

        summarisation_prompt = _build_summarisation_prompt(older_messages)

        try:
            summary = await self._llm_client.complete(
                messages=summarisation_prompt,
                temperature=0.3,
                max_tokens=512,
            )
            session.context_summary = summary.strip()
            logger.debug(
                "Context summary updated for session %s (length=%d)",
                session.session_id,
                len(session.context_summary),
            )
        except Exception:
            logger.warning(
                "Failed to generate context summary for session %s; retaining previous summary",
                session.session_id,
                exc_info=True,
            )

    def build_context(
        self,
        session: SessionState,
        system_prompt: str,
        current_user_message: str,
    ) -> list[dict[str, str]]:
        """Assemble the full context list to pass to the LLM.

        The assembled context follows this structure:
          1. System Prompt (stage-specific)
          2. Context Summary (if session > window)
          3. Recent Q&A pairs (sliding window, oldest verbatim turns trimmed
             first if token budget is exceeded)
          4. Current user message

        Args:
            session: The current ``SessionState``.
            system_prompt: The stage-specific system prompt text.
            current_user_message: The user's current inbound message text.

        Returns:
            A ``list[dict[str, str]]`` of chat messages ready to pass to
            ``LLMClient.complete``.
        """
        messages: list[dict[str, str]] = []

        # 1. System prompt
        system_message: dict[str, str] = {"role": "system", "content": system_prompt}
        messages.append(system_message)

        # 2. Context summary (if present)
        summary_message: dict[str, str] | None = None
        if session.context_summary:
            summary_message = {
                "role": "system",
                "content": (
                    "Summary of the conversation so far:\n"
                    f"{session.context_summary}"
                ),
            }
            messages.append(summary_message)

        # 3. Recent Q&A pairs (sliding window)
        qa_pairs = self._build_qa_pairs(session)
        recent_pairs = qa_pairs[-self._window_size :] if qa_pairs else []

        # Flatten recent pairs into a message list
        recent_messages: list[dict[str, str]] = []
        for q_msg, a_msg in recent_pairs:
            recent_messages.append(q_msg)
            if a_msg is not None:
                recent_messages.append(a_msg)

        # 4. Current user message
        user_message: dict[str, str] = {"role": "user", "content": current_user_message}

        # Enforce token budget: trim oldest verbatim turns first
        recent_messages = self._trim_to_budget(
            system_message=system_message,
            summary_message=summary_message,
            recent_messages=recent_messages,
            user_message=user_message,
        )

        messages.extend(recent_messages)
        messages.append(user_message)

        return messages

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_qa_pairs(
        self, session: SessionState
    ) -> list[tuple[dict[str, str], dict[str, str] | None]]:
        """Build a list of (question_message, response_message | None) pairs.

        Each pair represents one Q&A turn in the session.  The question is
        represented as an ``assistant`` message (the interviewer asked it) and
        the response as a ``user`` message (the candidate answered it).

        Returns:
            A list of ``(assistant_msg, user_msg | None)`` tuples, ordered
            chronologically.
        """
        response_map = {r.question_id: r for r in session.responses}
        pairs: list[tuple[dict[str, str], dict[str, str] | None]] = []

        for question in session.questions:
            q_msg: dict[str, str] = {
                "role": "assistant",
                "content": question.text,
            }
            response = response_map.get(question.question_id)
            if response is not None:
                a_msg: dict[str, str] | None = {
                    "role": "user",
                    "content": response.text,
                }
            else:
                a_msg = None

            pairs.append((q_msg, a_msg))

        return pairs

    def _trim_to_budget(
        self,
        system_message: dict[str, str],
        summary_message: dict[str, str] | None,
        recent_messages: list[dict[str, str]],
        user_message: dict[str, str],
    ) -> list[dict[str, str]]:
        """Trim the oldest verbatim turns until the total token count fits within budget.

        The system prompt, context summary, and current user message are never
        trimmed.  Only the ``recent_messages`` (verbatim Q&A pairs) are
        candidates for removal, starting from the oldest.

        Args:
            system_message: The system prompt message (never trimmed).
            summary_message: The context summary message, if any (never trimmed).
            recent_messages: The flattened list of recent Q&A messages.
            user_message: The current user message (never trimmed).

        Returns:
            The (possibly trimmed) ``recent_messages`` list.
        """
        # Calculate fixed token cost (system + summary + user)
        fixed_tokens = _count_message_tokens(system_message)
        if summary_message is not None:
            fixed_tokens += _count_message_tokens(summary_message)
        fixed_tokens += _count_message_tokens(user_message)

        available_tokens = self._max_context_tokens - fixed_tokens

        if available_tokens <= 0:
            # No budget left for any verbatim turns
            return []

        # Calculate total tokens for recent messages
        total_recent_tokens = sum(_count_message_tokens(m) for m in recent_messages)

        if total_recent_tokens <= available_tokens:
            # All recent messages fit within budget
            return recent_messages

        # Trim oldest messages first until we fit within budget
        trimmed = list(recent_messages)
        while trimmed and total_recent_tokens > available_tokens:
            removed = trimmed.pop(0)
            total_recent_tokens -= _count_message_tokens(removed)

        return trimmed
