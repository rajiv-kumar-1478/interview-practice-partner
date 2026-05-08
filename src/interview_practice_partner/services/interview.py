"""InterviewService — question generation, response evaluation, and follow-up logic.

This service is responsible for:
- Generating interview questions (with type variety and deduplication)
- Evaluating user responses via LLM
- Handling short responses (< 15 words) with elaboration prompts
- Handling off-topic responses with redirect messages
- Handling skip requests
- Handling voice note inputs (unsupported)
- Orchestrating the main response flow via ``handle_response``
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Literal, TYPE_CHECKING

import structlog

from interview_practice_partner.audio.download_client import AudioDownloadClient
from interview_practice_partner.audio.tts_client import ElevenLabsClient
from interview_practice_partner.audio.whisper_client import WhisperClient
from interview_practice_partner.domain.enums import InterviewRoundType, QuestionType, ProblemDifficulty, DesignPhase
from interview_practice_partner.domain.exceptions import TranscriptionError
from interview_practice_partner.domain.models import Question, SessionState, UserResponse
from interview_practice_partner.llm.client import LLMClient
from interview_practice_partner.llm.prompt_builder import PromptBuilder

if TYPE_CHECKING:
    from interview_practice_partner.services.technical_round import TechnicalRoundService

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Question type cycle — enforces variety across a session
# ---------------------------------------------------------------------------

# The cycle order for question types (excluding FOLLOW_UP which is generated
# on-demand based on response evaluation).
_QUESTION_TYPE_CYCLE: list[QuestionType] = [
    QuestionType.BEHAVIOURAL,
    QuestionType.SITUATIONAL,
    QuestionType.TECHNICAL,
]

# ---------------------------------------------------------------------------
# Skip keyword detection
# ---------------------------------------------------------------------------

_SKIP_KEYWORDS = frozenset(
    [
        "skip",
        "next question",
        "pass",
        "move on",
        "next",
        "skip this",
        "skip that",
        "skip it",
        "leave this",
        "leave this question",
        "different question",
        "another question",
        "change question",
        "change the question",
        "ask another",
        "ask a different",
        "ask dsa",
        "ask me dsa",
        "next one",
        "i don't know",
        "i dont know",
        "i do not know",
        "don't know the answer",
        "dont know the answer",
        "no idea",
        "have no idea",
        "give me solution",
        "give me the solution",
        "give me the answer",
        "give me answer",
        "tell me the answer",
        "tell me answer",
        "what's the answer",
        "what is the answer",
        "can't answer",
        "cannot answer",
        "cant answer",
        "unable to answer",
    ]
)

# ---------------------------------------------------------------------------
# Repeat request detection
# ---------------------------------------------------------------------------

_REPEAT_KEYWORDS = frozenset(
    [
        "repeat",
        "repeat the question",
        "repeat question",
        "say again",
        "say that again",
        "explain again",
        "explain the question",
        "can you repeat",
        "could you repeat",
        "what was the question",
        "what's the question",
        "what is the question",
        "i don't understand",
        "i dont understand",
        "i can't understand",
        "i cant understand",
        "didn't understand",
        "didnt understand",
        "don't understand",
        "dont understand",
        "not clear",
        "unclear",
        "rephrase",
        "rephrase the question",
        "ask again",
        "please repeat",
        "please explain",
        "what did you ask",
        "what did you say",
    ]
)


def _is_repeat_request(message: str) -> bool:
    """Return True if the message is a request to repeat the current question."""
    lower = message.lower().strip()
    return any(kw in lower for kw in _REPEAT_KEYWORDS)

# ---------------------------------------------------------------------------
# Round type detection keywords
# ---------------------------------------------------------------------------

_DSA_KEYWORDS = frozenset([
    "dsa", "coding", "algorithms", "algorithm", "data structures",
    "leetcode", "hackerrank", "code", "programming",
])

_SYSTEM_DESIGN_KEYWORDS = frozenset([
    "system design", "design", "architecture", "scalability",
    "distributed systems", "system", "architect",
])

_BEHAVIORAL_KEYWORDS = frozenset([
    "behavioral", "behavioural", "soft skills", "experience",
    "tell me about", "describe a time",
])


def _detect_round_type(message: str) -> Optional[InterviewRoundType]:
    """Detect interview round type from user message keywords.
    
    Returns:
        InterviewRoundType if detected, None if ambiguous.
    """
    lower = message.lower().strip()
    
    # Check for DSA keywords
    dsa_match = any(kw in lower for kw in _DSA_KEYWORDS)
    
    # Check for System Design keywords
    design_match = any(kw in lower for kw in _SYSTEM_DESIGN_KEYWORDS)
    
    # Check for Behavioral keywords
    behavioral_match = any(kw in lower for kw in _BEHAVIORAL_KEYWORDS)
    
    # Count matches
    match_count = sum([dsa_match, design_match, behavioral_match])
    
    # If exactly one match, return that type
    if match_count == 1:
        if dsa_match:
            return InterviewRoundType.DSA_CODING
        elif design_match:
            return InterviewRoundType.SYSTEM_DESIGN
        elif behavioral_match:
            return InterviewRoundType.BEHAVIORAL
    
    # Ambiguous or no match
    return None

# ---------------------------------------------------------------------------
# Audio media type prefixes
# ---------------------------------------------------------------------------

_AUDIO_MEDIA_TYPES = frozenset(
    [
        "audio/",
        "application/ogg",
    ]
)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _is_skip_request(message: str) -> bool:
    """Return True if the message is a skip request."""
    lower = message.lower().strip()
    return any(kw in lower for kw in _SKIP_KEYWORDS)


def _is_dont_know(message: str) -> bool:
    """Return True if the message indicates the user doesn't know the answer."""
    lower = message.lower().strip()
    dont_know_phrases = frozenset([
        "i don't know", "i dont know", "i do not know",
        "don't know the answer", "dont know the answer",
        "no idea", "have no idea", "give me solution",
        "give me the solution", "give me the answer",
        "give me answer", "tell me the answer", "tell me answer",
        "what's the answer", "what is the answer",
    ])
    return any(phrase in lower for phrase in dont_know_phrases)


def _is_solution_request(message: str) -> bool:
    """Return True if the message is explicitly requesting the solution/answer.

    Distinct from a general skip — the user specifically wants to see the
    solution so we should provide it (Req 14.4).
    """
    lower = message.lower().strip()
    solution_phrases = frozenset([
        "give me solution", "give me the solution",
        "give me the answer", "give me answer",
        "tell me the answer", "tell me answer",
        "what's the answer", "what is the answer",
        "show me the solution", "show solution",
        "reveal solution", "reveal the solution",
        "what's the solution", "what is the solution",
    ])
    return any(phrase in lower for phrase in solution_phrases)


def _is_audio_media(media_content_type: Optional[str]) -> bool:
    """Return True if the media content type indicates an audio file."""
    if not media_content_type:
        return False
    lower = media_content_type.lower()
    return any(lower.startswith(prefix) for prefix in _AUDIO_MEDIA_TYPES)


def _count_words(text: str) -> int:
    """Return the number of words in *text*."""
    return len(text.split())


def _determine_next_question_type(session: SessionState) -> QuestionType:
    """Determine the next question type to maintain variety.

    Cycles through BEHAVIOURAL → SITUATIONAL → TECHNICAL, skipping types
    that have already been asked if we're still in the first cycle.  After
    all three types have been asked at least once, cycles back to the
    beginning.

    Args:
        session: The current ``SessionState``.

    Returns:
        The ``QuestionType`` to generate next.
    """
    # Count how many of each type have been asked (excluding follow-ups)
    type_counts: dict[QuestionType, int] = {qt: 0 for qt in _QUESTION_TYPE_CYCLE}
    for q in session.questions:
        if q.question_type in type_counts:
            type_counts[q.question_type] += 1

    # Find the type with the fewest questions asked (prioritise variety)
    min_count = min(type_counts.values())
    for qt in _QUESTION_TYPE_CYCLE:
        if type_counts[qt] == min_count:
            return qt

    # Fallback (should not be reached)
    return QuestionType.BEHAVIOURAL


# ---------------------------------------------------------------------------
# InterviewService
# ---------------------------------------------------------------------------


class InterviewService:
    """Generates questions, evaluates responses, and manages interview flow.

    This service is intentionally stateless — all mutable state lives in the
    ``SessionState`` object that is passed in and returned from each method.

    Args:
        llm_client: An ``LLMClient`` implementation for LLM calls.
        prompt_builder: A ``PromptBuilder`` instance for constructing prompts.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_builder: PromptBuilder,
        whisper_client: WhisperClient,
        tts_client: ElevenLabsClient,
        audio_download_client: AudioDownloadClient,
        technical_round_service: Optional[TechnicalRoundService] = None,
    ) -> None:
        self._llm = llm_client
        self._prompt_builder = prompt_builder
        self._whisper = whisper_client
        self._tts = tts_client
        self._audio_download = audio_download_client
        self._technical_round_service = technical_round_service

    # ------------------------------------------------------------------
    # Voice note detection
    # ------------------------------------------------------------------

    @staticmethod
    def is_voice_note(num_media: int, media_content_type: Optional[str]) -> bool:
        """Return True if the message contains an audio voice note attachment.

        Classifies as a voice note when all of the following are true:
        - ``num_media > 0`` (at least one media attachment is present)
        - ``media_content_type`` is not None
        - the content type starts with ``"audio/"`` or equals ``"application/ogg"``

        Args:
            num_media: Number of media attachments in the inbound message.
            media_content_type: The MIME type of the first media attachment,
                or ``None`` if no media is present.

        Returns:
            ``True`` if the message is a voice note, ``False`` otherwise.
        """
        if num_media <= 0 or media_content_type is None:
            return False
        lower = media_content_type.lower()
        return lower.startswith("audio/") or lower == "application/ogg"

    # ------------------------------------------------------------------
    # Mode command detection
    # ------------------------------------------------------------------

    @staticmethod
    def is_mode_command(body: str) -> Literal["voice", "text"] | None:
        """Return the requested mode if *body* is a mode command, else None.

        Strips whitespace and lowercases before comparison.
        Returns ``"voice"`` for ``"voice mode"``, ``"text"`` for ``"text mode"``,
        ``None`` otherwise.

        Args:
            body: The message body text to check.

        Returns:
            ``"voice"`` if the body is ``"voice mode"`` (case-insensitive),
            ``"text"`` if the body is ``"text mode"`` (case-insensitive),
            ``None`` otherwise.
        """
        normalized = body.strip().lower()
        if normalized == "voice mode":
            return "voice"
        elif normalized == "text mode":
            return "text"
        else:
            return None

    # ------------------------------------------------------------------
    # Question generation
    # ------------------------------------------------------------------

    async def generate_question(
        self,
        session: SessionState,
        difficulty_signal: Optional[str] = None,
        question_type: Optional[QuestionType] = None,
    ) -> str:
        """Generate the next interview question for the session.

        Enforces question type variety by cycling through BEHAVIOURAL,
        SITUATIONAL, and TECHNICAL types.  Passes all previously asked
        questions to the prompt to prevent repetition.

        Args:
            session: The current ``SessionState``.
            difficulty_signal: One of ``"increase"``, ``"maintain"``, or
                ``"decrease"`` (or ``None`` to omit the signal).
            question_type: Override the question type to generate.  If
                ``None``, the type is determined automatically to maintain
                variety.

        Returns:
            The generated question text as a plain string.
        """
        if question_type is None:
            question_type = _determine_next_question_type(session)

        log = logger.bind(
            session_id=session.session_id,
            question_type=question_type.value,
            difficulty_signal=difficulty_signal,
        )
        log.info("generating_question")

        messages = self._prompt_builder.build_question_generation_prompt(
            session=session,
            question_type=question_type,
            difficulty_signal=difficulty_signal,
        )
        question_text = await self._llm.complete(messages, temperature=0.8)

        # Strip any leading/trailing whitespace from the generated question
        question_text = question_text.strip()

        # Record the question in the session
        question = Question(
            question_id=str(uuid.uuid4()),
            text=question_text,
            question_type=question_type,
            asked_at=_now(),
        )
        session.questions.append(question)

        log.info("question_generated", question_id=question.question_id)
        return question_text

    # ------------------------------------------------------------------
    # Response evaluation
    # ------------------------------------------------------------------

    async def evaluate_response(
        self,
        session: SessionState,
        response: UserResponse,
    ) -> dict:
        """Evaluate a user's response via LLM and return the evaluation result.

        Calls the LLM with the response evaluation prompt and parses the JSON
        result.  The evaluation result contains:
        - ``is_off_topic`` (bool)
        - ``is_short`` (bool)
        - ``follow_up_warranted`` (bool)
        - ``follow_up_text`` (str | None)
        - ``difficulty_signal`` ("increase" | "maintain" | "decrease")

        The ``is_off_topic`` field is also set on the ``UserResponse`` object.

        Args:
            session: The current ``SessionState``.
            response: The ``UserResponse`` to evaluate.

        Returns:
            A dict with the evaluation fields described above.
        """
        # Find the question this response is for
        question = next(
            (q for q in session.questions if q.question_id == response.question_id),
            None,
        )
        if question is None:
            logger.warning(
                "evaluate_response_question_not_found",
                session_id=session.session_id,
                question_id=response.question_id,
            )
            # Return a safe default evaluation
            return {
                "is_off_topic": False,
                "is_short": response.word_count < 15,
                "follow_up_warranted": False,
                "follow_up_text": None,
                "difficulty_signal": "maintain",
            }

        log = logger.bind(
            session_id=session.session_id,
            question_id=question.question_id,
            response_id=response.response_id,
        )
        log.info("evaluating_response")

        messages = self._prompt_builder.build_response_evaluation_prompt(
            question=question,
            response=response,
            session=session,
        )

        raw = await self._llm.complete(messages, temperature=0.3)

        # Parse the JSON evaluation result
        evaluation = self._parse_evaluation_response(raw, response)

        # Set is_off_topic on the response object
        response.is_off_topic = evaluation.get("is_off_topic", False)

        log.info(
            "response_evaluated",
            is_off_topic=evaluation.get("is_off_topic"),
            is_short=evaluation.get("is_short"),
            follow_up_warranted=evaluation.get("follow_up_warranted"),
            difficulty_signal=evaluation.get("difficulty_signal"),
        )

        return evaluation

    @staticmethod
    def _parse_evaluation_response(raw: str, response: UserResponse) -> dict:
        """Parse the LLM evaluation JSON response.

        Falls back to safe defaults if JSON parsing fails.

        Args:
            raw: The raw LLM response string.
            response: The ``UserResponse`` being evaluated (used for fallback
                ``is_short`` calculation).

        Returns:
            A dict with evaluation fields.
        """
        defaults = {
            "is_off_topic": False,
            "is_short": response.word_count < 15,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        }

        try:
            data = json.loads(raw)
            return {
                "is_off_topic": bool(data.get("is_off_topic", defaults["is_off_topic"])),
                "is_short": bool(data.get("is_short", defaults["is_short"])),
                "follow_up_warranted": bool(
                    data.get("follow_up_warranted", defaults["follow_up_warranted"])
                ),
                "follow_up_text": data.get("follow_up_text") or None,
                "difficulty_signal": str(
                    data.get("difficulty_signal", defaults["difficulty_signal"])
                ),
            }
        except (json.JSONDecodeError, AttributeError, TypeError):
            logger.warning(
                "evaluation_json_parse_failed",
                raw_response=raw[:200],
            )
            return defaults

    # ------------------------------------------------------------------
    # Repeat question handling
    # ------------------------------------------------------------------

    def handle_repeat_request(self, session: SessionState) -> str:
        """Restate the current question verbatim.

        Finds the most recent unanswered question and returns it with a
        brief preamble. Does NOT advance the session.

        Args:
            session: The current ``SessionState``.

        Returns:
            The current question restated as a plain string.
        """
        answered_ids = {r.question_id for r in session.responses}
        current_question = None
        for q in reversed(session.questions):
            if not q.skipped and q.question_id not in answered_ids:
                current_question = q
                break

        if current_question is None:
            return (
                "I don't have an active question for you right now. "
                "Please send any message to continue."
            )

        logger.info(
            "repeat_request_handled",
            session_id=session.session_id,
            question_id=current_question.question_id,
        )
        return f"Of course! Here's the question again:\n\n{current_question.text}"

    # ------------------------------------------------------------------
    # Short response handling
    # ------------------------------------------------------------------

    def handle_short_response(self, session: SessionState) -> str:
        """Return an elaboration prompt when the response is too short.

        Does NOT advance the session — the current question remains active.

        Args:
            session: The current ``SessionState``.

        Returns:
            An elaboration prompt string.
        """
        logger.info(
            "short_response_detected",
            session_id=session.session_id,
        )
        return (
            "Your answer was a bit brief. Could you please elaborate a little more? "
            "Try to give a specific example or explain your reasoning in more detail."
        )

    # ------------------------------------------------------------------
    # Empty/whitespace submission handling
    # ------------------------------------------------------------------

    def handle_empty_submission(self, session: SessionState) -> str:
        """Return a prompt when the user submits an empty or whitespace-only message.

        For technical rounds (Req 14.5): prompts the user to provide their
        approach or skip the problem.  For other contexts, returns a generic
        prompt.

        Also increments ``consecutive_out_of_scope_count`` so that repeated
        empty submissions eventually trigger the skip/end offer (Req 14.7).

        Args:
            session: The current ``SessionState`` (mutated in-place).

        Returns:
            A prompt string asking the user to provide their approach or skip.
        """
        session.off_topic_count += 1
        session.consecutive_out_of_scope_count += 1

        log = logger.bind(
            session_id=session.session_id,
            consecutive_out_of_scope_count=session.consecutive_out_of_scope_count,
        )
        log.info("empty_submission_detected")

        round_type = session.interview_round_type
        is_technical = round_type in (
            InterviewRoundType.DSA_CODING,
            InterviewRoundType.SYSTEM_DESIGN,
        )

        # Req 14.7: After 3+ consecutive invalid responses, offer to skip or end
        if session.consecutive_out_of_scope_count >= 3:
            if is_technical:
                return (
                    "It looks like we've had a few empty submissions. "
                    "Would you like to:\n\n"
                    "1. *Skip* the current problem and try a different one\n"
                    "2. *End the session* and receive your feedback so far\n"
                    "3. *Continue* with the current problem\n\n"
                    "Please reply with *1*, *2*, or *3*."
                )

        # Req 14.5: Prompt to provide approach or skip
        if is_technical:
            return (
                "It looks like your message was empty. "
                "Please share your approach or solution, or reply *skip* to move on to the next problem."
            )

        return (
            "It looks like your message was empty. "
            "Please provide your answer or reply *skip* to move on."
        )

    # ------------------------------------------------------------------
    # Off-topic handling
    # ------------------------------------------------------------------

    def handle_off_topic(self, session: SessionState) -> str:
        """Return a redirect message and update off-topic counters.

        Increments ``off_topic_count`` and ``consecutive_out_of_scope_count``.
        After 3 or more consecutive out-of-scope inputs, offers to end the
        session or return to role selection.

        For technical rounds (Req 14.2, 14.6): redirects the user back to the
        current problem and asks them to re-read it.
        For technical rounds (Req 14.7): After 3+ consecutive off-topic/invalid
        responses, offers to skip the current problem or end the session.

        Args:
            session: The current ``SessionState`` (mutated in-place).

        Returns:
            A redirect message string.
        """
        session.off_topic_count += 1
        session.consecutive_out_of_scope_count += 1

        log = logger.bind(
            session_id=session.session_id,
            off_topic_count=session.off_topic_count,
            consecutive_out_of_scope_count=session.consecutive_out_of_scope_count,
        )
        log.info("off_topic_response_detected")

        if session.consecutive_out_of_scope_count >= 3:
            # Check if we're in a technical round
            round_type = session.interview_round_type
            is_technical = round_type in (
                InterviewRoundType.DSA_CODING,
                InterviewRoundType.SYSTEM_DESIGN,
            )

            if is_technical:
                # Req 14.7: Offer to skip the current problem or end the session
                return (
                    "It looks like we've gone off track a few times. "
                    "Would you like to:\n\n"
                    "1. *Skip* the current problem and try a different one\n"
                    "2. *End the session* and receive your feedback so far\n"
                    "3. *Continue* with the current problem\n\n"
                    "Please reply with *1*, *2*, or *3*."
                )
            else:
                # Behavioral round: offer to end session or return to role selection
                return (
                    "It looks like we've gone off track a few times. "
                    "Would you like to:\n"
                    "1. End the session and receive your feedback so far\n"
                    "2. Return to role selection and start a new session\n\n"
                    "Please reply with *1* or *2*, or send your answer to continue the interview."
                )

        # Check if we're in a technical round for a more specific redirect
        round_type = session.interview_round_type
        is_technical = round_type in (
            InterviewRoundType.DSA_CODING,
            InterviewRoundType.SYSTEM_DESIGN,
        )

        if is_technical:
            # Req 14.2 & 14.6: Ask user to re-read the problem and redirect back
            current_problem_text = self._get_current_problem_text(session)
            if current_problem_text:
                return (
                    "That response doesn't seem related to the current problem. "
                    "Please re-read the problem and try again:\n\n"
                    f"{current_problem_text}"
                )
            return (
                "That response doesn't seem related to the current problem. "
                "Please re-read the problem and try again, "
                "or reply *skip* to move on to a different problem."
            )

        # Standard redirect for behavioral rounds
        return (
            "That response doesn't seem to be related to the interview question. "
            "Let's keep focused on the interview — please try to answer the question asked."
        )

    @staticmethod
    def _get_current_problem_text(session: SessionState) -> Optional[str]:
        """Return the text of the current unanswered question, or None.

        Args:
            session: The current ``SessionState``.

        Returns:
            The question text string, or ``None`` if no active question found.
        """
        answered_ids = {r.question_id for r in session.responses}
        for q in reversed(session.questions):
            if not q.skipped and q.question_id not in answered_ids:
                return q.text
        return None

    # ------------------------------------------------------------------
    # Skip handling
    # ------------------------------------------------------------------

    async def handle_skip(
        self,
        session: SessionState,
        user_message: str = "",
    ) -> tuple[str, SessionState]:
        """Mark the current question as skipped and advance to the next question.

        For technical rounds (DSA/System Design):
        - Tracks consecutive skips; decreases difficulty after 2 consecutive skips (Req 15.3)
        - When user requests the solution, provides it with explanation (Req 14.4)
        - After 3 consecutive skips, offers to end the session (Req 14.7)

        For behavioral rounds:
        - Marks question as skipped and generates the next question.

        Args:
            session: The current ``SessionState`` (mutated in-place).
            user_message: The original user message (used to tailor the reply).

        Returns:
            A ``(reply_text, updated_session)`` tuple.
        """
        # Find the most recent unanswered, non-skipped question
        answered_ids = {r.question_id for r in session.responses}
        current_question = None
        for q in reversed(session.questions):
            if not q.skipped and q.question_id not in answered_ids:
                current_question = q
                break

        if current_question is not None:
            current_question.skipped = True
            logger.info(
                "question_skipped",
                session_id=session.session_id,
                question_id=current_question.question_id,
            )

        # Handle technical rounds differently
        round_type = session.interview_round_type
        is_technical = round_type in (
            InterviewRoundType.DSA_CODING,
            InterviewRoundType.SYSTEM_DESIGN,
        )

        if is_technical:
            return await self._handle_technical_skip(session, current_question, user_message)

        # Behavioral round: generate the next question
        next_question_text = await self.generate_question(session)

        # Tailor the acknowledgement based on why they skipped
        if _is_dont_know(user_message):
            preamble = (
                "No worries — that's a common one to get stuck on. "
                "In a real interview, it's fine to say you'd approach it by breaking "
                "it down or asking clarifying questions. Let's move on.\n\n"
            )
        else:
            preamble = "No problem — we'll skip that one.\n\n"

        reply = f"{preamble}{next_question_text}"
        return reply, session

    async def _handle_technical_skip(
        self,
        session: SessionState,
        current_question,
        user_message: str,
    ) -> tuple[str, SessionState]:
        """Handle skip requests during DSA/System Design rounds.

        Tracks consecutive skips, decreases difficulty after 2 consecutive
        skips (Req 15.3), provides solution when requested (Req 14.4), and
        offers to end the session after 3 consecutive skips (Req 14.7).

        Args:
            session: The current ``SessionState`` (mutated in-place).
            current_question: The question that was just skipped (may be None).
            user_message: The original user message.

        Returns:
            A ``(reply_text, updated_session)`` tuple.
        """
        # Increment consecutive skip counter
        session.consecutive_skips_count += 1

        log = logger.bind(
            session_id=session.session_id,
            consecutive_skips=session.consecutive_skips_count,
            round_type=session.interview_round_type.value if session.interview_round_type else None,
        )
        log.info("technical_skip_detected")

        # Req 14.7: After 3 consecutive skips, offer to end the session
        if session.consecutive_skips_count >= 3:
            log.info("consecutive_skip_limit_reached")
            return (
                "It looks like you've skipped a few problems in a row. "
                "Would you like to:\n\n"
                "1. *End the session* and receive your feedback so far\n"
                "2. *Continue* with a new problem\n\n"
                "Please reply with *1* to end or *2* to continue.",
                session,
            )

        # Req 15.3: Decrease difficulty after 2 consecutive skips
        if session.consecutive_skips_count >= 2:
            old_difficulty = session.problem_difficulty
            if old_difficulty == ProblemDifficulty.HARD:
                session.problem_difficulty = ProblemDifficulty.MEDIUM
            elif old_difficulty == ProblemDifficulty.MEDIUM:
                session.problem_difficulty = ProblemDifficulty.EASY
            # (already EASY — stays at EASY)

            if session.problem_difficulty != old_difficulty:
                session.difficulty_adjustment_history.append({
                    "from": old_difficulty.value,
                    "to": session.problem_difficulty.value,
                    "reason": "consecutive_skips",
                    "timestamp": _now().isoformat(),
                })
                log.info(
                    "difficulty_decreased_after_consecutive_skips",
                    old_difficulty=old_difficulty.value,
                    new_difficulty=session.problem_difficulty.value,
                )

        # Req 14.4: If user requested the solution, provide it via LLM
        solution_requested = _is_solution_request(user_message)

        if (solution_requested and
                current_question is not None and
                session.interview_round_type == InterviewRoundType.DSA_CODING and
                self._technical_round_service is not None):
            # Generate solution explanation via LLM
            solution_text = await self._get_problem_solution(session, current_question)
            preamble = (
                "No problem — here's the solution with explanation:\n\n"
                f"{solution_text}\n\n"
            )
        elif _is_dont_know(user_message):
            preamble = (
                "No worries — that's a tricky one. "
                "In a real interview, it helps to think out loud and break the problem down. "
                "Let's try a different problem.\n\n"
            )
        else:
            preamble = "No problem — we'll skip that one.\n\n"

        # Generate next technical problem/question
        if session.interview_round_type == InterviewRoundType.DSA_CODING:
            if self._technical_round_service is not None:
                next_problem = await self._technical_round_service.generate_coding_problem(
                    session, session.problem_difficulty
                )
                from interview_practice_partner.domain.models import CodingProblem
                next_question = Question(
                    question_id=next_problem.problem_id,
                    text=next_problem.text,
                    question_type=QuestionType.TECHNICAL,
                    asked_at=next_problem.asked_at,
                )
                session.questions.append(next_question)
                next_text = next_problem.text
                if next_problem.examples:
                    next_text += "\n\n*Examples:*\n" + "\n".join(next_problem.examples)
                if next_problem.constraints:
                    next_text += f"\n\n*Constraints:* {next_problem.constraints}"
            else:
                next_text = await self.generate_question(session)
        else:
            # System Design round
            if self._technical_round_service is not None:
                next_design_q = await self._technical_round_service.generate_system_design_question(
                    session
                )
                next_question = Question(
                    question_id=next_design_q.question_id,
                    text=next_design_q.text,
                    question_type=QuestionType.TECHNICAL,
                    asked_at=next_design_q.asked_at,
                )
                session.questions.append(next_question)
                next_text = f"{next_design_q.text}\n\n{next_design_q.description}"
            else:
                next_text = await self.generate_question(session)

        difficulty_note = ""
        if session.consecutive_skips_count >= 2:
            difficulty_note = (
                f"_(Difficulty adjusted to *{session.problem_difficulty.value.capitalize()}*)_\n\n"
            )

        reply = f"{preamble}{difficulty_note}{next_text}"
        return reply, session

    async def _get_problem_solution(
        self,
        session: SessionState,
        current_question,
    ) -> str:
        """Generate a solution explanation for the current DSA problem via LLM.

        Falls back to a generic message if the LLM call fails.

        Args:
            session: The current ``SessionState``.
            current_question: The ``Question`` that was skipped.

        Returns:
            A solution explanation string.
        """
        try:
            messages = self._prompt_builder.build_problem_solution_prompt(
                problem_text=current_question.text,
                difficulty=session.problem_difficulty,
            )
            solution = await self._llm.complete(messages, temperature=0.3)
            return solution.strip()
        except Exception:  # noqa: BLE001
            logger.warning(
                "problem_solution_generation_failed",
                session_id=session.session_id,
                question_id=current_question.question_id,
            )
            return (
                "I wasn't able to generate the full solution right now, "
                "but the key approach for this type of problem is to break it down "
                "into smaller subproblems and consider edge cases carefully."
            )

    # ------------------------------------------------------------------
    # Round type selection handling (Software Engineer only)
    # ------------------------------------------------------------------

    async def handle_round_type_selection(
        self,
        session: SessionState,
        user_message: str,
    ) -> tuple[str, SessionState]:
        """Handle round type selection for Software Engineer role.

        Detects round type from user message or prompts for selection.
        Sets session.interview_round_type and generates first question.

        Args:
            session: Current SessionState (mutated in-place).
            user_message: The user's message.

        Returns:
            (reply_text, updated_session) tuple.
        """
        log = logger.bind(
            session_id=session.session_id,
        )
        log.info("handle_round_type_selection_start")

        # Try to detect round type from message
        detected_type = _detect_round_type(user_message)

        if detected_type is not None:
            # Round type detected — set it and generate first question
            session.interview_round_type = detected_type
            log.info(
                "round_type_detected",
                round_type=detected_type.value,
            )

            # Generate first question based on round type
            if detected_type == InterviewRoundType.DSA_CODING:
                # Initialize DSA round
                session.problem_difficulty = ProblemDifficulty.MEDIUM
                log.info(
                    "technical_round_started",
                    session_id=session.session_id,
                    round_type=detected_type.value,
                    initial_difficulty=session.problem_difficulty.value,
                )
                if self._technical_round_service is None:
                    # Fallback to behavioral if technical service not available
                    session.interview_round_type = InterviewRoundType.BEHAVIORAL
                    question_text = await self.generate_question(session)
                    reply = (
                        "Technical rounds are not available right now. "
                        "Let's proceed with a behavioral interview instead.\n\n"
                        f"{question_text}"
                    )
                else:
                    # Generate first DSA problem
                    problem = await self._technical_round_service.generate_coding_problem(
                        session, session.problem_difficulty
                    )
                    # Store problem as a question
                    question = Question(
                        question_id=problem.problem_id,
                        text=problem.text,
                        question_type=QuestionType.TECHNICAL,
                        asked_at=problem.asked_at,
                    )
                    session.questions.append(question)
                    reply = (
                        "Great! Let's start with a *DSA/Coding Round*.\n\n"
                        f"{problem.text}\n\n"
                        f"*Examples:*\n" + "\n".join(problem.examples) + "\n\n"
                        f"*Constraints:* {problem.constraints}"
                    )

            elif detected_type == InterviewRoundType.SYSTEM_DESIGN:
                # Initialize System Design round
                log.info(
                    "technical_round_started",
                    session_id=session.session_id,
                    round_type=detected_type.value,
                    initial_difficulty=None,
                )
                if self._technical_round_service is None:
                    # Fallback to behavioral if technical service not available
                    session.interview_round_type = InterviewRoundType.BEHAVIORAL
                    question_text = await self.generate_question(session)
                    reply = (
                        "Technical rounds are not available right now. "
                        "Let's proceed with a behavioral interview instead.\n\n"
                        f"{question_text}"
                    )
                else:
                    # Generate first System Design question
                    # (design_phase is initialized inside generate_system_design_question)
                    design_question = await self._technical_round_service.generate_system_design_question(
                        session
                    )
                    # Store question
                    question = Question(
                        question_id=design_question.question_id,
                        text=design_question.text,
                        question_type=QuestionType.TECHNICAL,
                        asked_at=design_question.asked_at,
                    )
                    session.questions.append(question)
                    reply = (
                        "Great! Let's start with a *System Design Round*.\n\n"
                        f"{design_question.text}\n\n"
                        f"{design_question.description}"
                    )

            else:  # BEHAVIORAL
                session.interview_round_type = InterviewRoundType.BEHAVIORAL
                question_text = await self.generate_question(session)
                reply = (
                    "Great! Let's start with a *Behavioral Round*.\n\n"
                    f"{question_text}"
                )

            return reply, session

        # No clear round type detected — ask the user to choose
        messages = self._prompt_builder.build_round_type_selection_prompt(user_message)
        raw = await self._llm.complete(messages, temperature=0.3)

        # Parse the response
        try:
            data = json.loads(raw)
            reply_message = data.get("message", "")
            round_type_str = data.get("round_type_detected")

            if round_type_str:
                # LLM detected a round type
                try:
                    detected_type = InterviewRoundType(round_type_str)
                    session.interview_round_type = detected_type
                    log.info(
                        "round_type_detected_by_llm",
                        round_type=detected_type.value,
                    )
                    # Generate first question (recursive call with detected type)
                    return await self.handle_round_type_selection(session, detected_type.value)
                except ValueError:
                    # Invalid round type from LLM — ask user to clarify
                    pass

            # No round type detected — return the prompt message
            if not reply_message:
                reply_message = (
                    "Which interview round would you like to practice?\n\n"
                    "1. *DSA/Coding Round* - Practice algorithmic problem-solving\n"
                    "2. *System Design Round* - Practice architectural design\n"
                    "3. *Behavioral Round* - Practice soft skills and experience questions\n\n"
                    "Please reply with the number or name of the round type."
                )

            return reply_message, session

        except (json.JSONDecodeError, AttributeError, TypeError):
            # JSON parsing failed — return a default prompt
            log.warning("round_type_selection_json_parse_failed", raw_response=raw[:200])
            reply = (
                "Which interview round would you like to practice?\n\n"
                "1. *DSA/Coding Round* - Practice algorithmic problem-solving\n"
                "2. *System Design Round* - Practice architectural design\n"
                "3. *Behavioral Round* - Practice soft skills and experience questions\n\n"
                "Please reply with the number or name of the round type."
            )
            return reply, session

    # ------------------------------------------------------------------
    # Mode command handling
    # ------------------------------------------------------------------

    async def handle_mode_command(
        self,
        session: SessionState,
        mode: Literal["voice", "text"],
    ) -> tuple[str, SessionState]:
        """Handle an explicit mode command.

        Sets preferred_mode on session, returns a confirmation text message.
        Does NOT advance interview state.

        Args:
            session: Current SessionState (mutated in-place).
            mode: The requested mode ("voice" or "text").

        Returns:
            (confirmation_text, updated_session) tuple.
        """
        session.preferred_mode = mode

        logger.info(
            "mode_command_handled",
            session_id=session.session_id,
            mode=mode,
        )

        if mode == "voice":
            confirmation = (
                "Voice mode is now active. "
                "I'll send my replies as audio messages from now on."
            )
        else:  # mode == "text"
            confirmation = (
                "Text mode is now active. "
                "I'll send my replies as text messages from now on."
            )

        return confirmation, session

    # ------------------------------------------------------------------
    # Voice note handling
    # ------------------------------------------------------------------

    async def handle_voice_note(
        self,
        session: SessionState,
        media_url: str,
        media_content_type: str,
    ) -> tuple[str, SessionState]:
        """Download, transcribe, and process a voice note.

        Replaces the old stub implementation. Full flow:
        1. Download audio bytes via AudioDownloadClient
        2. Transcribe via GroqWhisperClient
        3. Handle empty transcription (ask user to resend)
        4. Delegate to handle_response() with transcribed text

        On TranscriptionError: returns a text fallback message, does not
        advance session state.

        Args:
            session: Current SessionState.
            media_url: Twilio media URL for the voice note.
            media_content_type: MIME type (used to derive file extension).

        Returns:
            (reply_text, updated_session) tuple.
        """
        log = logger.bind(
            session_id=session.session_id,
            media_url=media_url,
            media_content_type=media_content_type,
        )
        log.info("handle_voice_note_start")

        # Step 1: Download audio bytes
        try:
            audio_bytes = await self._audio_download.download(media_url)
        except TranscriptionError:
            log.info("voice_note_download_failed")
            return (
                "I couldn't process your voice note. Please resend it or type your answer.",
                session,
            )

        # Step 2: Derive filename from media_content_type
        if media_content_type and media_content_type.lower() == "audio/mpeg":
            filename = "voice_note.mp3"
        elif media_content_type and media_content_type.lower() == "audio/ogg":
            filename = "voice_note.ogg"
        else:
            filename = "voice_note.ogg"  # default

        # Step 3: Transcribe audio
        try:
            transcribed_text = await self._whisper.transcribe(audio_bytes, filename)
        except TranscriptionError:
            log.info("voice_note_transcription_failed")
            return (
                "I couldn't process your voice note. Please resend it or type your answer.",
                session,
            )

        # Step 4: Handle empty transcription
        if not transcribed_text.strip():
            log.info("voice_note_transcription_empty")
            return (
                "I couldn't make out what you said. Please resend your voice note or type your answer.",
                session,
            )

        # Step 5: Update preferred_mode and delegate to handle_response
        session.preferred_mode = "voice"
        log.info("voice_note_transcribed", transcribed_text_length=len(transcribed_text))
        
        # Delegate to handle_response with text-only path (no media parameters)
        return await self.handle_response(session, transcribed_text, _from_voice_note=True)

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    async def classify_intent(
        self,
        session: SessionState,
        user_message: str,
    ) -> str:
        """Classify the user's intent via LLM.

        Returns one of: "answer", "skip", "repeat", "out_of_scope".
        Falls back to "answer" if classification fails.

        Args:
            session: The current ``SessionState``.
            user_message: The raw text sent by the user.

        Returns:
            Intent string: "answer" | "skip" | "repeat" | "out_of_scope".
        """
        # Get the current question text for context
        answered_ids = {r.question_id for r in session.responses}
        current_question = None
        for q in reversed(session.questions):
            if not q.skipped and q.question_id not in answered_ids:
                current_question = q
                break

        current_question_text = current_question.text if current_question else "No active question."

        messages = self._prompt_builder.build_intent_classification_prompt(
            user_message=user_message,
            current_question=current_question_text,
        )

        try:
            raw = await self._llm.complete(messages, temperature=0.0)
            data = json.loads(raw)
            intent = data.get("intent", "answer")
            if intent not in ("answer", "skip", "repeat", "out_of_scope"):
                intent = "answer"
            logger.info(
                "intent_classified",
                session_id=session.session_id,
                intent=intent,
            )
            return intent
        except Exception:
            logger.warning(
                "intent_classification_failed",
                session_id=session.session_id,
                message_preview=user_message[:50],
            )
            return "answer"

    # ------------------------------------------------------------------
    # Main response handler
    # ------------------------------------------------------------------

    async def handle_response(
        self,
        session: SessionState,
        user_message: str,
        num_media: int = 0,
        media_content_type: Optional[str] = None,
        media_url: Optional[str] = None,
        _from_voice_note: bool = False,
    ) -> tuple[str, SessionState]:
        """Main entry point for handling a user's response during the interview.

        Orchestrates the full response handling flow:
        1. Check for mode command → call ``handle_mode_command``
        2. Check for voice note → call ``handle_voice_note``
        3. If text message: set ``session.preferred_mode = "text"`` silently
        3b. Check for empty/whitespace-only submission → call ``handle_empty_submission``
        4. Check for round type selection (Software Engineer only)
        5. Check for skip keyword → call ``handle_skip``
        6. Create a ``UserResponse`` and evaluate it via LLM
        7. If short (word_count < 15) → call ``handle_short_response``
        8. If off-topic → call ``handle_off_topic``
        9. Otherwise → record response, generate next question or follow-up

        Args:
            session: The current ``SessionState``.
            user_message: The raw text body of the inbound WhatsApp message.
            num_media: Number of media attachments in the message (default 0).
            media_content_type: The MIME type of the first media attachment,
                if any (default ``None``).
            media_url: Twilio media URL of first attachment (for voice notes).
            _from_voice_note: Internal flag indicating this call is from a transcribed
                voice note (should not switch to text mode).

        Returns:
            A ``(reply_text, updated_session)`` tuple.
        """
        log = logger.bind(
            session_id=session.session_id,
            message_length=len(user_message),
            num_media=num_media,
        )
        log.info("handle_response_start")

        # 1. Check for mode command first
        mode_command = self.is_mode_command(user_message)
        if mode_command is not None:
            log.info("mode_command_detected", mode=mode_command)
            return await self.handle_mode_command(session, mode_command)

        # 2. Check for voice note
        if self.is_voice_note(num_media, media_content_type):
            log.info("voice_note_detected")
            if media_url is None:
                log.error("voice_note_detected_but_no_media_url")
                return "I couldn't process your voice note. Please resend it or type your answer.", session
            return await self.handle_voice_note(session, media_url, media_content_type or "")

        # 3. If text message and not from voice note: set preferred_mode to "text" silently
        if not _from_voice_note:
            session.preferred_mode = "text"

        # 3b. Check for empty or whitespace-only submission (Req 14.5)
        # Only intercept for technical rounds; non-technical rounds fall through
        # to the existing short-response handling path.
        if not user_message.strip():
            round_type = session.interview_round_type
            is_technical = round_type in (
                InterviewRoundType.DSA_CODING,
                InterviewRoundType.SYSTEM_DESIGN,
            )
            if is_technical:
                log.info("empty_submission_detected")
                reply = self.handle_empty_submission(session)
                return reply, session

        # 4. Check for round type selection (Software Engineer only, no round type set yet)
        from interview_practice_partner.domain.enums import Role
        if (session.role == Role.SOFTWARE_ENGINEER and
                session.interview_round_type is None):
            # In ROUND_TYPE_SELECTION stage — prompt for or detect round type
            return await self.handle_round_type_selection(session, user_message)

        # 4b. Check for round type switching mid-session (INTERVIEW stage)
        if (session.role == Role.SOFTWARE_ENGINEER and
                session.interview_round_type is not None and
                len(session.questions) > 0):
            detected_new_type = _detect_round_type(user_message)
            if (detected_new_type is not None and
                    detected_new_type != session.interview_round_type):
                # User wants to switch round type — confirm and restart
                log.info(
                    "round_type_switch_requested",
                    current_type=session.interview_round_type.value,
                    new_type=detected_new_type.value,
                )
                old_type = session.interview_round_type
                # Reset session for new round type
                session.interview_round_type = None
                session.questions = []
                session.responses = []
                session.off_topic_count = 0
                session.consecutive_out_of_scope_count = 0
                session.consecutive_skips_count = 0
                session.difficulty_adjustment_history = []
                session.topics_covered = []
                session.design_aspects_covered = []
                session.design_phase = None
                session.problem_difficulty = ProblemDifficulty.MEDIUM
                # Now handle the new round type selection
                return await self.handle_round_type_selection(session, user_message)

        # 5. Classify intent via LLM — handles skip, repeat, out_of_scope, answer
        intent = await self.classify_intent(session, user_message)

        if intent == "skip":
            log.info("skip_intent_detected")
            return await self.handle_skip(session, user_message)

        if intent == "repeat":
            log.info("repeat_intent_detected")
            return self.handle_repeat_request(session), session

        if intent == "out_of_scope":
            log.info("out_of_scope_intent_detected")
            reply = self.handle_off_topic(session)
            return reply, session

        # 6. intent == "answer" — create response and evaluate
        word_count = _count_words(user_message)
        response = UserResponse(
            response_id=str(uuid.uuid4()),
            question_id=self._get_current_question_id(session),
            text=user_message,
            word_count=word_count,
            received_at=_now(),
        )

        # 7. Check for short response (word_count < 15)
        if word_count < 15:
            log.info("short_response_detected_pre_eval", word_count=word_count)
            reply = self.handle_short_response(session)
            return reply, session

        # 8. Route to appropriate evaluation based on round type
        round_type = session.interview_round_type

        if (round_type == InterviewRoundType.DSA_CODING and
                self._technical_round_service is not None):
            return await self._handle_dsa_response(session, response, log)

        if (round_type == InterviewRoundType.SYSTEM_DESIGN and
                self._technical_round_service is not None):
            return await self._handle_system_design_response(session, response, log)

        # BEHAVIORAL (or no round type set) — use existing behavioral evaluation
        evaluation = await self.evaluate_response(session, response)

        # Update response with evaluation results
        response.is_off_topic = evaluation.get("is_off_topic", False)

        # 9. Handle off-topic response (LLM evaluation may still catch subtle off-topic)
        if evaluation.get("is_off_topic", False):
            reply = self.handle_off_topic(session)
            return reply, session

        # 10. Normal response — record it and generate next question or follow-up
        session.responses.append(response)

        # Reset consecutive out-of-scope count on a valid on-topic response
        session.consecutive_out_of_scope_count = 0
        # Reset consecutive skips count on a valid answer
        session.consecutive_skips_count = 0

        difficulty_signal = evaluation.get("difficulty_signal", "maintain")
        follow_up_warranted = evaluation.get("follow_up_warranted", False)
        follow_up_text = evaluation.get("follow_up_text")

        if follow_up_warranted and follow_up_text:
            # Generate a follow-up question
            follow_up_question = Question(
                question_id=str(uuid.uuid4()),
                text=follow_up_text,
                question_type=QuestionType.FOLLOW_UP,
                asked_at=_now(),
            )
            session.questions.append(follow_up_question)
            log.info(
                "follow_up_generated",
                follow_up_question_id=follow_up_question.question_id,
            )
            reply = (
                "Good answer! Let me follow up on that:\n\n"
                f"{follow_up_text}"
            )
        else:
            # Generate the next main question
            next_question_text = await self.generate_question(
                session,
                difficulty_signal=difficulty_signal,
            )
            reply = (
                "Thank you for your answer.\n\n"
                f"{next_question_text}"
            )

        log.info("handle_response_complete", reply_length=len(reply))
        return reply, session

    # ------------------------------------------------------------------
    # Technical round response handlers
    # ------------------------------------------------------------------

    async def _handle_dsa_response(
        self,
        session: SessionState,
        response: UserResponse,
        log,
    ) -> tuple[str, SessionState]:
        """Handle a DSA/Coding round response by routing to TechnicalRoundService.

        Reconstructs a CodingProblem from the current question, calls
        evaluate_coding_solution, adjusts difficulty, and generates the
        next problem or follow-up.

        Args:
            session: The current ``SessionState``.
            response: The ``UserResponse`` to evaluate.
            log: Bound structlog logger.

        Returns:
            A ``(reply_text, updated_session)`` tuple.
        """
        from interview_practice_partner.domain.models import CodingProblem
        from interview_practice_partner.domain.enums import ProblemTopic

        # Reconstruct a CodingProblem from the current question
        current_question_id = response.question_id
        current_question = next(
            (q for q in session.questions if q.question_id == current_question_id),
            session.questions[-1] if session.questions else None,
        )

        if current_question is None:
            log.warning("dsa_response_no_current_question")
            return "I couldn't find the current problem. Please send any message to continue.", session

        problem = CodingProblem(
            problem_id=current_question.question_id,
            text=current_question.text,
            difficulty=session.problem_difficulty,
            topic=ProblemTopic.ARRAYS,  # Default; actual topic not stored in Question
            constraints="",
            examples=[],
            asked_at=current_question.asked_at,
        )

        log.info("routing_to_technical_round_service", round_type="dsa_coding")

        # Evaluate the coding solution
        technical_eval = await self._technical_round_service.evaluate_coding_solution(
            session, problem, response
        )

        # Record the response
        session.responses.append(response)
        session.consecutive_out_of_scope_count = 0
        # Reset consecutive skips count on a valid answer
        session.consecutive_skips_count = 0

        # Adjust difficulty based on evaluation
        new_difficulty = self._technical_round_service.adjust_difficulty(session, technical_eval)
        session.problem_difficulty = new_difficulty

        log.info(
            "dsa_solution_evaluated",
            correctness=technical_eval.correctness,
            difficulty_signal=technical_eval.difficulty_signal,
            new_difficulty=new_difficulty.value,
        )

        # Build reply from technical evaluation
        if technical_eval.follow_up_warranted and technical_eval.follow_up_text:
            # Ask follow-up question
            follow_up_question = Question(
                question_id=str(uuid.uuid4()),
                text=technical_eval.follow_up_text,
                question_type=QuestionType.FOLLOW_UP,
                asked_at=_now(),
            )
            session.questions.append(follow_up_question)
            reply = f"Good attempt! Let me follow up:\n\n{technical_eval.follow_up_text}"
        else:
            # Generate next DSA problem
            next_problem = await self._technical_round_service.generate_coding_problem(
                session, new_difficulty
            )
            next_question = Question(
                question_id=next_problem.problem_id,
                text=next_problem.text,
                question_type=QuestionType.TECHNICAL,
                asked_at=next_problem.asked_at,
            )
            session.questions.append(next_question)

            # Build feedback summary
            correctness = technical_eval.correctness or "partial"
            complexity_info = ""
            if technical_eval.complexity_analysis:
                ca = technical_eval.complexity_analysis
                complexity_info = (
                    f"\n*Time:* {ca.time_complexity} | *Space:* {ca.space_complexity}"
                )

            reply = (
                f"*Evaluation:* {correctness.capitalize()}{complexity_info}\n\n"
                f"Here's your next problem:\n\n"
                f"{next_problem.text}"
            )
            if next_problem.examples:
                reply += "\n\n*Examples:*\n" + "\n".join(next_problem.examples)
            if next_problem.constraints:
                reply += f"\n\n*Constraints:* {next_problem.constraints}"

        return reply, session

    async def _handle_system_design_response(
        self,
        session: SessionState,
        response: UserResponse,
        log,
    ) -> tuple[str, SessionState]:
        """Handle a System Design round response by routing to TechnicalRoundService.

        Reconstructs a SystemDesignQuestion from the current question, calls
        evaluate_system_design, and generates a follow-up or next question.

        Args:
            session: The current ``SessionState``.
            response: The ``UserResponse`` to evaluate.
            log: Bound structlog logger.

        Returns:
            A ``(reply_text, updated_session)`` tuple.
        """
        from interview_practice_partner.domain.models import SystemDesignQuestion

        # Reconstruct a SystemDesignQuestion from the current question
        current_question_id = response.question_id
        current_question = next(
            (q for q in session.questions if q.question_id == current_question_id),
            session.questions[-1] if session.questions else None,
        )

        if current_question is None:
            log.warning("system_design_response_no_current_question")
            return "I couldn't find the current question. Please send any message to continue.", session

        design_question = SystemDesignQuestion(
            question_id=current_question.question_id,
            text=current_question.text,
            system_name="System",  # Actual name not stored in Question
            description="",
            asked_at=current_question.asked_at,
        )

        log.info("routing_to_technical_round_service", round_type="system_design")

        # Evaluate the system design response
        technical_eval = await self._technical_round_service.evaluate_system_design(
            session, design_question, response
        )

        # Record the response
        session.responses.append(response)
        session.consecutive_out_of_scope_count = 0
        # Reset consecutive skips count on a valid answer
        session.consecutive_skips_count = 0

        log.info(
            "system_design_evaluated",
            follow_up_warranted=technical_eval.follow_up_warranted,
            aspects_count=len(technical_eval.design_aspects_evaluated),
        )

        # Build reply from technical evaluation
        if technical_eval.follow_up_warranted and technical_eval.follow_up_text:
            # Ask follow-up question
            follow_up_question = Question(
                question_id=str(uuid.uuid4()),
                text=technical_eval.follow_up_text,
                question_type=QuestionType.FOLLOW_UP,
                asked_at=_now(),
            )
            session.questions.append(follow_up_question)
            reply = f"Good thinking! Let me probe deeper:\n\n{technical_eval.follow_up_text}"
        else:
            # Generate next system design question
            next_design_question = await self._technical_round_service.generate_system_design_question(
                session
            )
            next_question = Question(
                question_id=next_design_question.question_id,
                text=next_design_question.text,
                question_type=QuestionType.TECHNICAL,
                asked_at=next_design_question.asked_at,
            )
            session.questions.append(next_question)

            # Build feedback summary
            strengths_text = ""
            if technical_eval.design_strengths:
                strengths_text = "\n*Strengths:* " + ", ".join(technical_eval.design_strengths[:2])

            reply = (
                f"Good response!{strengths_text}\n\n"
                f"Let's move on to the next design challenge:\n\n"
                f"{next_design_question.text}\n\n"
                f"{next_design_question.description}"
            )

        return reply, session

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_current_question_id(session: SessionState) -> str:
        """Return the question_id of the most recent unanswered question.

        If no unanswered question exists, returns an empty string (which
        will result in a response not linked to any question — this is a
        defensive fallback).

        Args:
            session: The current ``SessionState``.

        Returns:
            The ``question_id`` string of the current question, or ``""``
            if none is found.
        """
        answered_ids = {r.question_id for r in session.responses}
        for q in reversed(session.questions):
            if not q.skipped and q.question_id not in answered_ids:
                return q.question_id
        # Fallback: return the last question's ID if all are answered
        if session.questions:
            return session.questions[-1].question_id
        return ""
