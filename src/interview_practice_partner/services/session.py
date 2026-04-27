"""SessionService — owns the interview session state machine and stage transitions.

The service is the single authority on which stage a session is in and what
triggers a transition.  It delegates content generation to ``InterviewService``
and ``FeedbackService`` and returns a plain-text reply string that the
orchestration layer can forward to the user via WhatsApp.

Stage transition table (from design):
┌──────────────────┬──────────────────────────────────────────────┬──────────────────────┐
│ Current Stage    │ Trigger                                      │ Next Stage           │
├──────────────────┼──────────────────────────────────────────────┼──────────────────────┤
│ INIT             │ Any first message                            │ ROLE_SELECTION       │
│ ROLE_SELECTION   │ Role confirmed                               │ INTERVIEW            │
│ ROLE_SELECTION   │ 2 clarification turns with no role           │ INTERVIEW (default)  │
│ INTERVIEW        │ ≥5 questions answered AND no pending follow-up│ FEEDBACK            │
│ INTERVIEW        │ User requests short session AND ≥3 answered  │ FEEDBACK             │
│ INTERVIEW        │ User changes role                            │ ROLE_SELECTION (new) │
│ FEEDBACK         │ Feedback fully delivered                     │ COMPLETE             │
│ COMPLETE         │ New message received                         │ INIT (new session)   │
└──────────────────┴──────────────────────────────────────────────┴──────────────────────┘
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Protocol

import structlog

from interview_practice_partner.domain.enums import Role, Stage
from interview_practice_partner.domain.exceptions import InvalidSessionStateError
from interview_practice_partner.domain.models import SessionState
from interview_practice_partner.llm.client import LLMClient
from interview_practice_partner.llm.prompt_builder import PromptBuilder

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Dependency protocols — avoids circular imports while keeping type safety
# ---------------------------------------------------------------------------


class InterviewServiceProtocol(Protocol):
    """Minimal interface that SessionService needs from InterviewService."""

    async def generate_question(self, session: SessionState) -> str:
        """Generate the next interview question text for the given session."""
        ...

    async def handle_response(
        self,
        session: SessionState,
        user_message: str,
        num_media: int = 0,
        media_content_type: Optional[str] = None,
        media_url: Optional[str] = None,
    ) -> tuple[str, SessionState]:
        """Evaluate the user's response and return (reply_text, updated_session)."""
        ...


class FeedbackServiceProtocol(Protocol):
    """Minimal interface that SessionService needs from FeedbackService."""

    async def generate_feedback_report(self, session: SessionState) -> tuple[str, SessionState]:
        """Generate the feedback report and return (reply_text, updated_session)."""
        ...


# ---------------------------------------------------------------------------
# Short-session / role-change keyword detection helpers
# ---------------------------------------------------------------------------

_SHORT_SESSION_KEYWORDS = frozenset(
    [
        "short",
        "quick",
        "brief",
        "fast",
        "fewer questions",
        "less questions",
        "shorter",
        "mini",
        "rapid",
    ]
)

_ROLE_CHANGE_PHRASES = frozenset(
    [
        "change role",
        "switch role",
        "different role",
        "change my role",
        "switch to",
        "i want to be",
        "i'd like to practice as",
        "i'd like to practise as",
        "start over",
        "restart",
        "let's start fresh",
        "start fresh",
    ]
)

_ROLE_KEYWORDS: dict[Role, list[str]] = {
    Role.SOFTWARE_ENGINEER: [
        "software engineer",
        "software developer",
        "developer",
        "engineer",
        "swe",
        "coding",
        "programming",
    ],
    Role.SALES_REPRESENTATIVE: [
        "sales representative",
        "sales rep",
        "sales",
        "account executive",
        "ae",
        "business development",
        "bdr",
        "sdr",
    ],
    Role.RETAIL_ASSOCIATE: [
        "retail associate",
        "retail",
        "shop assistant",
        "store associate",
        "customer service",
        "cashier",
    ],
}


def _detect_role_in_message(message: str) -> Role:
    """Return the first ``Role`` whose keywords appear in *message*, or ``Role.UNKNOWN``."""
    lower = message.lower()
    for role, keywords in _ROLE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return role
    return Role.UNKNOWN


def _is_short_session_request(message: str) -> bool:
    """Return True if the message contains a short-session request keyword."""
    lower = message.lower()
    return any(kw in lower for kw in _SHORT_SESSION_KEYWORDS)


def _is_role_change_request(message: str) -> bool:
    """Return True if the message looks like a mid-session role-change request."""
    lower = message.lower()
    return any(phrase in lower for phrase in _ROLE_CHANGE_PHRASES)


def _count_answered_questions(session: SessionState) -> int:
    """Return the number of non-skipped questions that have a recorded response."""
    answered_ids = {r.question_id for r in session.responses}
    return sum(
        1
        for q in session.questions
        if not q.skipped and q.question_id in answered_ids
    )


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# SessionService
# ---------------------------------------------------------------------------


class SessionService:
    """Owns the session state machine and coordinates content generation.

    The service is intentionally stateless — all mutable state lives in the
    ``SessionState`` object that is passed in and returned from ``transition``.

    Args:
        llm_client: An ``LLMClient`` implementation used for role extraction
            during ``ROLE_SELECTION``.
        prompt_builder: A ``PromptBuilder`` instance for constructing prompts.
        interview_service: An ``InterviewService``-compatible object for
            question generation and response evaluation.
        feedback_service: A ``FeedbackService``-compatible object for report
            generation.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_builder: PromptBuilder,
        interview_service: InterviewServiceProtocol,
        feedback_service: FeedbackServiceProtocol,
    ) -> None:
        self._llm = llm_client
        self._prompt_builder = prompt_builder
        self._interview = interview_service
        self._feedback = feedback_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def transition(
        self,
        session: SessionState,
        user_message: str,
        num_media: int = 0,
        media_content_type: Optional[str] = None,
        media_url: Optional[str] = None,
    ) -> tuple[str, SessionState]:
        """Apply the appropriate state machine transition for *user_message*.

        Determines the current stage, applies the correct transition logic,
        updates the ``SessionState`` in-place (via a copy), and returns the
        agent's reply text together with the updated session.

        Args:
            session: The current ``SessionState`` loaded from the repository.
            user_message: The raw text body of the inbound WhatsApp message.
            num_media: Number of media attachments (default 0).
            media_content_type: MIME type of the first attachment (default None).
            media_url: Twilio media URL of the first attachment (default None).

        Returns:
            A ``(reply_text, updated_session)`` tuple.  The caller is
            responsible for persisting the updated session.

        Raises:
            InvalidSessionStateError: If the session is in an unrecognised stage.
        """
        log = logger.bind(
            session_id=session.session_id,
            stage=session.stage.value,
            phone_number=session.phone_number,
        )
        log.info("session_transition_start", user_message_length=len(user_message))

        stage = session.stage

        if stage == Stage.INIT:
            reply, session = await self._handle_init(session, user_message)

        elif stage == Stage.ROLE_SELECTION:
            reply, session = await self._handle_role_selection(session, user_message)

        elif stage == Stage.INTERVIEW:
            reply, session = await self._handle_interview(
                session, user_message,
                num_media=num_media,
                media_content_type=media_content_type,
                media_url=media_url,
            )

        elif stage == Stage.FEEDBACK:
            reply, session = await self._handle_feedback(session, user_message)

        elif stage == Stage.COMPLETE:
            reply, session = await self._handle_complete(session, user_message)

        else:
            raise InvalidSessionStateError(
                f"Unrecognised session stage: {stage!r}"
            )

        session.updated_at = _now()
        log.info(
            "session_transition_complete",
            new_stage=session.stage.value,
            reply_length=len(reply),
        )
        return reply, session

    # ------------------------------------------------------------------
    # Stage handlers
    # ------------------------------------------------------------------

    async def _handle_init(
        self, session: SessionState, user_message: str
    ) -> tuple[str, SessionState]:
        """INIT → ROLE_SELECTION on any first message.

        If the opening message already contains a recognisable role we skip
        straight to INTERVIEW (Property 13 / Requirement 6.1).
        """
        session.stage = Stage.ROLE_SELECTION
        session.clarification_turn_count = 0

        # Fast-path: role present in opening message → go straight to INTERVIEW
        detected_role = _detect_role_in_message(user_message)
        if detected_role != Role.UNKNOWN:
            session.role = detected_role
            session.stage = Stage.INTERVIEW
            logger.info(
                "fast_path_role_detected",
                role=detected_role.value,
                session_id=session.session_id,
            )
            question_text = await self._interview.generate_question(session)
            role_display = detected_role.value.replace("_", " ").title()
            reply = (
                f"Great, let's practise for a *{role_display}* interview!\n\n"
                f"{question_text}"
            )
            return reply, session

        # Normal path: ask for role
        messages = self._prompt_builder.build_role_selection_prompt(
            user_message=user_message,
            clarification_turn_count=0,
        )
        raw = await self._llm.complete(messages)
        reply = self._extract_message_from_role_response(raw)
        return reply, session

    async def _handle_role_selection(
        self, session: SessionState, user_message: str
    ) -> tuple[str, SessionState]:
        """ROLE_SELECTION → INTERVIEW (role confirmed) or stay (clarification).

        After 2 clarification turns with no confirmed role, transition to
        INTERVIEW with a default general format (Requirement 5.2).
        """
        # Increment clarification turn counter before processing
        session.clarification_turn_count += 1

        # Check for timeout: 2 turns without a role → default general format
        if session.clarification_turn_count >= 2:
            # Try one last time to detect a role in this message
            detected_role = _detect_role_in_message(user_message)
            if detected_role != Role.UNKNOWN:
                # Role found at timeout — use it and transition to INTERVIEW
                session.role = detected_role
                session.stage = Stage.INTERVIEW
                logger.info(
                    "clarification_timeout_role_detected",
                    role=detected_role.value,
                    session_id=session.session_id,
                )
                question_text = await self._interview.generate_question(session)
                role_display = detected_role.value.replace("_", " ").title()
                reply = (
                    f"Great, let's practise for a *{role_display}* interview!\n\n"
                    f"{question_text}"
                )
                return reply, session
            else:
                # Timeout: proceed with general format
                session.role = Role.UNKNOWN
                session.stage = Stage.INTERVIEW
                logger.info(
                    "clarification_timeout_default_format",
                    session_id=session.session_id,
                    clarification_turn_count=session.clarification_turn_count,
                )
                question_text = await self._interview.generate_question(session)
                reply = (
                    "No worries! I'll proceed with a general interview format.\n\n"
                    f"{question_text}"
                )
                return reply, session

        # Try to extract role via LLM
        messages = self._prompt_builder.build_role_selection_prompt(
            user_message=user_message,
            clarification_turn_count=session.clarification_turn_count,
        )
        raw = await self._llm.complete(messages)
        role, reply = self._parse_role_selection_response(raw)

        if role != Role.UNKNOWN:
            # Role confirmed → transition to INTERVIEW
            session.role = role
            session.stage = Stage.INTERVIEW
            logger.info(
                "role_confirmed",
                role=role.value,
                session_id=session.session_id,
            )
            question_text = await self._interview.generate_question(session)
            role_display = role.value.replace("_", " ").title()
            reply = (
                f"Perfect! Let's begin your *{role_display}* mock interview.\n\n"
                f"{question_text}"
            )
        # else: stay in ROLE_SELECTION, reply is the clarification message from LLM

        return reply, session

    async def _handle_interview(
        self, session: SessionState, user_message: str,
        num_media: int = 0,
        media_content_type: Optional[str] = None,
        media_url: Optional[str] = None,
    ) -> tuple[str, SessionState]:
        """INTERVIEW stage — evaluate response, check transitions.

        Handles:
        - Role change mid-session → new session, ROLE_SELECTION
        - Short session request → FEEDBACK if ≥3 questions answered
        - ≥5 questions answered with no pending follow-up → FEEDBACK
        - Otherwise → continue interview (next question or follow-up)
        """
        # Check for role change request
        if _is_role_change_request(user_message):
            new_role = _detect_role_in_message(user_message)
            logger.info(
                "role_change_requested",
                session_id=session.session_id,
                detected_role=new_role.value,
            )
            # Create a fresh SessionState for the new session
            fresh_session = self._create_fresh_session(
                phone_number=session.phone_number,
                role=new_role if new_role != Role.UNKNOWN else Role.UNKNOWN,
            )
            fresh_session.stage = Stage.ROLE_SELECTION

            if new_role != Role.UNKNOWN:
                role_display = new_role.value.replace("_", " ").title()
                reply = (
                    f"Sure! I'll restart the session for a *{role_display}* interview. "
                    "Let me confirm — would you like to practise for a "
                    f"*{role_display}* role?"
                )
            else:
                reply = (
                    "Of course! Let's start fresh. "
                    "Which role would you like to practise for? "
                    "I support Software Engineer, Sales Representative, and Retail Associate."
                )
            return reply, fresh_session

        # Check for short session request
        if _is_short_session_request(user_message):
            session.requested_short_session = True
            logger.info(
                "short_session_requested",
                session_id=session.session_id,
            )

        # Delegate response evaluation and reply generation to InterviewService
        reply, session = await self._interview.handle_response(
            session, user_message,
            num_media=num_media,
            media_content_type=media_content_type,
            media_url=media_url,
        )

        # After handling the response, check transition conditions
        answered_count = _count_answered_questions(session)

        # Short session: ≥3 answered → FEEDBACK
        if session.requested_short_session and answered_count >= 3:
            session.stage = Stage.FEEDBACK
            logger.info(
                "short_session_complete",
                session_id=session.session_id,
                answered_count=answered_count,
            )
            feedback_reply, session = await self._feedback.generate_feedback_report(session)
            return feedback_reply, session

        # Standard session: ≥5 answered → FEEDBACK
        if answered_count >= 5:
            # Only transition if there's no pending follow-up (InterviewService
            # signals a pending follow-up by returning a follow-up question as
            # the reply; we check the session's last question type instead)
            has_pending_follow_up = self._has_pending_follow_up(session)
            if not has_pending_follow_up:
                session.stage = Stage.FEEDBACK
                logger.info(
                    "standard_session_complete",
                    session_id=session.session_id,
                    answered_count=answered_count,
                )
                feedback_reply, session = await self._feedback.generate_feedback_report(session)
                return feedback_reply, session

        return reply, session

    async def _handle_feedback(
        self, session: SessionState, user_message: str
    ) -> tuple[str, SessionState]:
        """FEEDBACK → COMPLETE once feedback is fully delivered.

        The user may ask for elaboration on specific feedback points.
        Once the feedback has been delivered (tracked by ``is_complete``),
        any further message transitions to COMPLETE.
        """
        if session.is_complete:
            # Feedback already delivered — mark COMPLETE
            session.stage = Stage.COMPLETE
            session.completed_at = _now()
            reply = (
                "Your session is now complete. "
                "Feel free to message me again whenever you'd like to practise more!"
            )
            return reply, session

        # Deliver (or re-deliver) the feedback report
        if session.feedback_report is None:
            # Generate the report if not yet done
            reply, session = await self._feedback.generate_feedback_report(session)
        else:
            # Feedback report exists — handle elaboration request or mark complete
            reply, session = await self._handle_feedback_elaboration(session, user_message)

        return reply, session

    async def _handle_feedback_elaboration(
        self, session: SessionState, user_message: str
    ) -> tuple[str, SessionState]:
        """Handle elaboration requests during the FEEDBACK stage.

        If the user asks for more detail on a specific feedback point, the LLM
        provides elaboration.  Otherwise, mark the session as complete.
        """
        elaboration_keywords = frozenset(
            ["elaborate", "explain", "more detail", "tell me more", "why", "how", "expand"]
        )
        lower = user_message.lower()
        wants_elaboration = any(kw in lower for kw in elaboration_keywords)

        if wants_elaboration and session.feedback_report is not None:
            # Build a simple elaboration prompt
            report = session.feedback_report
            strengths_text = "\n".join(f"- {s}" for s in report.strengths)
            improvements_text = "\n".join(f"- {i}" for i in report.improvements)
            recommendations_text = "\n".join(
                f"- {r}" for r in report.actionable_recommendations
            )
            system_content = (
                "You are an interview coach. The candidate has asked for more detail "
                "about their feedback. Provide a helpful, encouraging elaboration "
                "based on the feedback summary below.\n\n"
                f"Strengths:\n{strengths_text}\n\n"
                f"Areas for improvement:\n{improvements_text}\n\n"
                f"Recommendations:\n{recommendations_text}\n\n"
                "Use plain text only. No HTML, no markdown headers, no code blocks."
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_message},
            ]
            reply = await self._llm.complete(messages)
        else:
            # No elaboration requested — mark session complete
            session.is_complete = True
            session.completed_at = _now()
            session.stage = Stage.COMPLETE
            reply = (
                "That wraps up your mock interview session! "
                "Great work — keep practising and you'll do brilliantly. "
                "Message me any time to start a new session."
            )

        return reply, session

    async def _handle_complete(
        self, session: SessionState, user_message: str
    ) -> tuple[str, SessionState]:
        """COMPLETE → INIT (new session) on any new message."""
        logger.info(
            "new_session_after_complete",
            old_session_id=session.session_id,
            phone_number=session.phone_number,
        )
        # Create a brand-new session for this phone number
        new_session = self._create_fresh_session(phone_number=session.phone_number)
        new_session.stage = Stage.ROLE_SELECTION

        # Check if the new message already contains a role (fast-path)
        detected_role = _detect_role_in_message(user_message)
        if detected_role != Role.UNKNOWN:
            new_session.role = detected_role
            new_session.stage = Stage.INTERVIEW
            question_text = await self._interview.generate_question(new_session)
            role_display = detected_role.value.replace("_", " ").title()
            reply = (
                f"Welcome back! Let's start a new *{role_display}* interview.\n\n"
                f"{question_text}"
            )
        else:
            messages = self._prompt_builder.build_role_selection_prompt(
                user_message=user_message,
                clarification_turn_count=0,
            )
            raw = await self._llm.complete(messages)
            reply = self._extract_message_from_role_response(raw)

        return reply, new_session

    # ------------------------------------------------------------------
    # Counter management helpers
    # ------------------------------------------------------------------

    def increment_off_topic_count(self, session: SessionState) -> SessionState:
        """Increment ``off_topic_count`` and ``consecutive_out_of_scope_count``."""
        session.off_topic_count += 1
        session.consecutive_out_of_scope_count += 1
        return session

    def reset_consecutive_out_of_scope_count(self, session: SessionState) -> SessionState:
        """Reset ``consecutive_out_of_scope_count`` to 0 (on an on-topic response)."""
        session.consecutive_out_of_scope_count = 0
        return session

    def increment_clarification_turn_count(self, session: SessionState) -> SessionState:
        """Increment ``clarification_turn_count`` by 1."""
        session.clarification_turn_count += 1
        return session

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_fresh_session(
        phone_number: str,
        role: Role = Role.UNKNOWN,
    ) -> SessionState:
        """Create a brand-new ``SessionState`` for *phone_number*."""
        now = _now()
        return SessionState(
            session_id=str(uuid.uuid4()),
            phone_number=phone_number,
            stage=Stage.INIT,
            role=role,
            created_at=now,
            updated_at=now,
        )

    @staticmethod
    def _has_pending_follow_up(session: SessionState) -> bool:
        """Return True if the most recent question in the session is a follow-up.

        A follow-up question is pending when the last question added to the
        session has ``question_type == QuestionType.FOLLOW_UP`` and has not
        yet received a response.
        """
        from interview_practice_partner.domain.enums import QuestionType

        if not session.questions:
            return False
        last_question = session.questions[-1]
        if last_question.question_type != QuestionType.FOLLOW_UP:
            return False
        answered_ids = {r.question_id for r in session.responses}
        return last_question.question_id not in answered_ids

    @staticmethod
    def _extract_message_from_role_response(raw: str) -> str:
        """Extract the ``message`` field from a role-selection LLM JSON response.

        Falls back to returning the raw string if JSON parsing fails.
        Handles cases where the LLM adds extra text before/after the JSON.
        """
        try:
            # Try to parse the entire response as JSON first
            data = json.loads(raw)
            return str(data.get("message", raw))
        except (json.JSONDecodeError, AttributeError):
            # If that fails, try to find JSON within the response
            import re
            json_match = re.search(r'\{[^{}]*"message"[^{}]*\}', raw, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    return str(data.get("message", raw))
                except (json.JSONDecodeError, AttributeError):
                    pass
            # If all parsing fails, return the raw string
            return raw

    @staticmethod
    def _parse_role_selection_response(raw: str) -> tuple[Role, str]:
        """Parse a role-selection LLM JSON response.

        Returns:
            A ``(role, message)`` tuple.  ``role`` is ``Role.UNKNOWN`` if the
            LLM could not identify a role with high confidence.
        """
        try:
            # Try to parse the entire response as JSON first
            data = json.loads(raw)
        except (json.JSONDecodeError, AttributeError):
            # If that fails, try to find JSON within the response
            import re
            json_match = re.search(r'\{[^{}]*"role"[^{}]*\}', raw, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                except (json.JSONDecodeError, AttributeError):
                    # If JSON parsing fails, try to detect a role from the raw text
                    detected = _detect_role_in_message(raw)
                    return detected, raw
            else:
                # If JSON parsing fails, try to detect a role from the raw text
                detected = _detect_role_in_message(raw)
                return detected, raw
        
        # Successfully parsed JSON - extract fields
        role_str = data.get("role", "unknown")
        confidence = data.get("confidence", "low")
        message = str(data.get("message", raw))

        # Only accept the role if confidence is high
        if confidence == "high":
            try:
                role = Role(role_str)
            except ValueError:
                role = Role.UNKNOWN
        else:
            role = Role.UNKNOWN

        return role, message
            return detected, raw
