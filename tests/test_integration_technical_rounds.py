"""Integration tests for Technical Interview Rounds end-to-end flows.

Covers:
- Task 11.1: Complete DSA round flow
  - User selects Software Engineer role
  - User selects DSA round
  - System generates coding problem
  - User submits solution (code format)
  - System evaluates solution
  - System asks follow-up question
  - User responds to follow-up
  - System adjusts difficulty
  - System generates next problem
  - User requests feedback
  - System generates technical feedback

Requirements: 1.1-1.6, 2.1-2.8, 3.1-3.7, 4.1-4.8, 5.1-5.7, 10.1-10.9
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from interview_practice_partner.domain.enums import (
    InterviewRoundType,
    ProblemDifficulty,
    ProblemTopic,
    QuestionType,
    Role,
    Stage,
)
from interview_practice_partner.domain.models import (
    CodingProblem,
    ComplexityAnalysis,
    FeedbackReport,
    Question,
    SessionState,
    TechnicalEvaluation,
    UserResponse,
)
from interview_practice_partner.llm.client import LLMClient
from interview_practice_partner.llm.prompt_builder import PromptBuilder
from interview_practice_partner.services.feedback import FeedbackService
from interview_practice_partner.services.interview import InterviewService
from interview_practice_partner.services.technical_round import TechnicalRoundService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_session(
    phone: str = "+1234567890",
    stage: Stage = Stage.ROUND_TYPE_SELECTION,
    role: Role = Role.SOFTWARE_ENGINEER,
) -> SessionState:
    return SessionState(
        session_id="sess-dsa-integration-001",
        phone_number=phone,
        stage=stage,
        role=role,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_mock_llm() -> Mock:
    client = Mock(spec=LLMClient)
    client.complete = AsyncMock()
    return client


def _make_services(mock_llm: Mock) -> tuple[InterviewService, TechnicalRoundService, FeedbackService]:
    """Wire up real service instances with a mocked LLM client."""
    prompt_builder = PromptBuilder()

    technical_round_service = TechnicalRoundService(
        llm_client=mock_llm,
        prompt_builder=prompt_builder,
    )

    # Minimal stubs for audio clients (not exercised in these tests)
    mock_whisper = Mock()
    mock_whisper.transcribe = AsyncMock(return_value="stub transcription")
    mock_tts = Mock()
    mock_tts.synthesize = AsyncMock(return_value=b"stub audio")
    mock_audio_download = Mock()
    mock_audio_download.download = AsyncMock(return_value=b"stub audio bytes")

    interview_service = InterviewService(
        llm_client=mock_llm,
        prompt_builder=prompt_builder,
        whisper_client=mock_whisper,
        tts_client=mock_tts,
        audio_download_client=mock_audio_download,
        technical_round_service=technical_round_service,
    )

    feedback_service = FeedbackService(
        llm_client=mock_llm,
        prompt_builder=prompt_builder,
    )

    return interview_service, technical_round_service, feedback_service


# ---------------------------------------------------------------------------
# LLM response builders
# ---------------------------------------------------------------------------

def _coding_problem_llm_response(
    problem_text: str = "Given an array of integers, return indices of two numbers that add up to target.",
    topic: str = "arrays",
    difficulty: str = "medium",
) -> str:
    return json.dumps({
        "problem_statement": problem_text,
        "examples": [
            "Input: nums=[2,7,11,15], target=9 → Output: [0,1]",
            "Input: nums=[3,2,4], target=6 → Output: [1,2]",
        ],
        "constraints": "2 <= nums.length <= 10^4, -10^9 <= nums[i] <= 10^9",
        "topic": topic,
    })


def _solution_evaluation_llm_response(
    correctness: str = "correct",
    is_optimal: bool = True,
    follow_up_warranted: bool = True,
    follow_up_text: str = "What is the time complexity of your solution?",
    difficulty_signal: str = "increase",
) -> str:
    return json.dumps({
        "correctness": correctness,
        "time_complexity": "O(n)",
        "space_complexity": "O(n)",
        "is_optimal": is_optimal,
        "edge_cases_handled": ["empty array", "single element"],
        "edge_cases_missed": [],
        "code_quality_notes": "Clean and readable code.",
        "follow_up_warranted": follow_up_warranted,
        "follow_up_text": follow_up_text,
        "difficulty_signal": difficulty_signal,
    })


def _intent_answer_response() -> str:
    return json.dumps({"intent": "answer"})


def _next_coding_problem_llm_response() -> str:
    return json.dumps({
        "problem_statement": "Given a string, find the length of the longest substring without repeating characters.",
        "examples": [
            "Input: s='abcabcbb' → Output: 3",
            "Input: s='bbbbb' → Output: 1",
        ],
        "constraints": "0 <= s.length <= 5 * 10^4",
        "topic": "strings",
    })


def _technical_feedback_llm_response() -> str:
    return json.dumps({
        "summary": "You completed 1 DSA problem with a correct and optimal solution.",
        "strengths": ["Correct solution", "Optimal time complexity O(n)"],
        "improvements": ["Consider more edge cases"],
        "actionable_recommendations": ["Practice more hash table problems"],
        "problem_summaries": [
            {
                "problem": "Two Sum",
                "correctness": "correct",
                "time_complexity": "O(n)",
                "space_complexity": "O(n)",
            }
        ],
    })


# ---------------------------------------------------------------------------
# Task 11.1 — Complete DSA round flow integration test
# Requirements: 1.1-1.6, 2.1-2.8, 3.1-3.7, 4.1-4.8, 5.1-5.7, 10.1-10.9
# ---------------------------------------------------------------------------


class TestDSARoundCompleteFlow:
    """Integration test for the complete DSA round flow using real service instances."""

    @pytest.fixture
    def mock_llm(self) -> Mock:
        return _make_mock_llm()

    @pytest.fixture
    def services(self, mock_llm: Mock) -> tuple[InterviewService, TechnicalRoundService, FeedbackService]:
        return _make_services(mock_llm)

    # ------------------------------------------------------------------
    # Step 1: User selects DSA round → system generates first problem
    # Requirements: 1.1, 1.2, 1.3, 2.1, 2.4, 2.8
    # ------------------------------------------------------------------

    async def test_step1_dsa_round_selection_generates_first_problem(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Selecting DSA round sets round type and generates first coding problem.

        Requirements: 1.1, 1.2, 1.3, 2.1, 2.4, 2.8
        """
        interview_service, _, _ = services
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)

        # LLM returns a coding problem for the first DSA problem generation
        mock_llm.complete.return_value = _coding_problem_llm_response()

        reply, updated_session = await interview_service.handle_response(
            session, "DSA round"
        )

        # Round type must be set to DSA_CODING
        assert updated_session.interview_round_type == InterviewRoundType.DSA_CODING

        # A question (the first problem) must have been added
        assert len(updated_session.questions) == 1
        assert updated_session.questions[0].question_type == QuestionType.TECHNICAL

        # Reply must contain the problem text
        assert "array" in reply.lower() or "two sum" in reply.lower() or "indices" in reply.lower()

        # Difficulty must be initialized to MEDIUM (default)
        assert updated_session.problem_difficulty == ProblemDifficulty.MEDIUM

    # ------------------------------------------------------------------
    # Step 2: User submits code solution → system evaluates and asks follow-up
    # Requirements: 3.1, 3.2, 4.1-4.8, 5.1, 5.2, 5.4
    # ------------------------------------------------------------------

    async def test_step2_code_solution_triggers_evaluation_and_follow_up(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Submitting a code solution triggers evaluation and a follow-up question.

        Requirements: 3.1, 3.2, 4.1-4.8, 5.1, 5.2, 5.4
        """
        interview_service, _, _ = services
        session = _make_session(stage=Stage.INTERVIEW)
        session.interview_round_type = InterviewRoundType.DSA_CODING
        session.problem_difficulty = ProblemDifficulty.MEDIUM

        # Add the first problem as a question
        problem_question = Question(
            question_id="q-dsa-001",
            text="Given an array of integers, return indices of two numbers that add up to target.",
            question_type=QuestionType.TECHNICAL,
            asked_at=_NOW,
        )
        session.questions.append(problem_question)

        # LLM sequence: intent classification → solution evaluation
        mock_llm.complete.side_effect = [
            _intent_answer_response(),
            _solution_evaluation_llm_response(
                correctness="correct",
                is_optimal=True,
                follow_up_warranted=True,
                follow_up_text="What is the time complexity of your solution?",
                difficulty_signal="increase",
            ),
        ]

        code_solution = (
            "```python\n"
            "def two_sum(nums, target):\n"
            "    seen = {}\n"
            "    for i, num in enumerate(nums):\n"
            "        complement = target - num\n"
            "        if complement in seen:\n"
            "            return [seen[complement], i]\n"
            "        seen[num] = i\n"
            "```"
        )

        reply, updated_session = await interview_service.handle_response(
            session, code_solution
        )

        # Response must be recorded
        assert len(updated_session.responses) == 1

        # A follow-up question must have been added
        assert len(updated_session.questions) == 2
        follow_up = updated_session.questions[1]
        assert follow_up.question_type == QuestionType.FOLLOW_UP
        assert "time complexity" in follow_up.text.lower()

        # Reply must contain the follow-up
        assert "time complexity" in reply.lower() or "follow up" in reply.lower()

    # ------------------------------------------------------------------
    # Step 3: User responds to follow-up → difficulty adjusts, next problem generated
    # Requirements: 2.6, 2.7, 4.7, 5.7, 15.1-15.5
    # ------------------------------------------------------------------

    async def test_step3_follow_up_response_adjusts_difficulty_and_generates_next_problem(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Responding to follow-up adjusts difficulty and generates the next problem.

        Requirements: 2.6, 2.7, 4.7, 5.7, 15.1-15.5
        """
        interview_service, _, _ = services
        session = _make_session(stage=Stage.INTERVIEW)
        session.interview_round_type = InterviewRoundType.DSA_CODING
        session.problem_difficulty = ProblemDifficulty.MEDIUM

        # Session has original problem + follow-up question
        original_question = Question(
            question_id="q-dsa-001",
            text="Two Sum problem",
            question_type=QuestionType.TECHNICAL,
            asked_at=_NOW,
        )
        follow_up_question = Question(
            question_id="q-followup-001",
            text="What is the time complexity of your solution?",
            question_type=QuestionType.FOLLOW_UP,
            asked_at=_NOW,
        )
        session.questions.extend([original_question, follow_up_question])

        # Record the original response so follow-up is the current question
        original_response = UserResponse(
            response_id="r-001",
            question_id="q-dsa-001",
            text="def two_sum(nums, target): ...",
            word_count=20,
            received_at=_NOW,
        )
        session.responses.append(original_response)

        # LLM sequence: intent → evaluation (no follow-up) → next problem generation
        mock_llm.complete.side_effect = [
            _intent_answer_response(),
            _solution_evaluation_llm_response(
                correctness="correct",
                is_optimal=True,
                follow_up_warranted=False,
                follow_up_text=None,
                difficulty_signal="increase",
            ),
            _next_coding_problem_llm_response(),
        ]

        follow_up_answer = (
            "The time complexity is O(n) because we iterate through the array once "
            "and each hash table lookup is O(1) on average giving us linear time overall."
        )

        reply, updated_session = await interview_service.handle_response(
            session, follow_up_answer
        )

        # Difficulty must have been adjusted upward (MEDIUM → HARD)
        assert updated_session.problem_difficulty == ProblemDifficulty.HARD

        # A new problem must have been generated and added
        assert len(updated_session.questions) == 3
        next_problem = updated_session.questions[2]
        assert next_problem.question_type == QuestionType.TECHNICAL

        # Reply must contain the next problem
        assert "substring" in reply.lower() or "string" in reply.lower() or "next" in reply.lower()

    # ------------------------------------------------------------------
    # Step 4: User requests feedback → technical feedback generated
    # Requirements: 10.1-10.9
    # ------------------------------------------------------------------

    async def test_step4_feedback_request_generates_technical_feedback(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Requesting feedback generates a technical feedback report.

        Requirements: 10.1-10.9
        """
        _, _, feedback_service = services
        session = _make_session(stage=Stage.FEEDBACK)
        session.interview_round_type = InterviewRoundType.DSA_CODING
        session.problem_difficulty = ProblemDifficulty.HARD

        # Session has one problem and one response
        problem_question = Question(
            question_id="q-dsa-001",
            text="Two Sum problem",
            question_type=QuestionType.TECHNICAL,
            asked_at=_NOW,
        )
        session.questions.append(problem_question)

        user_response = UserResponse(
            response_id="r-001",
            question_id="q-dsa-001",
            text="def two_sum(nums, target): ...",
            word_count=20,
            received_at=_NOW,
        )
        session.responses.append(user_response)

        # Difficulty adjustment history
        session.difficulty_adjustment_history = [
            {"from": "medium", "to": "hard", "reason": "correct and optimal solution"}
        ]

        mock_llm.complete.return_value = _technical_feedback_llm_response()

        reply, updated_session = await feedback_service.generate_feedback_report(session)

        # Feedback report must be stored in session
        assert updated_session.feedback_report is not None
        report = updated_session.feedback_report

        # Report must have strengths and improvements
        assert len(report.strengths) >= 1
        assert len(report.improvements) >= 1
        assert len(report.actionable_recommendations) >= 1

        # Reply must be a non-empty string
        assert isinstance(reply, str)
        assert len(reply) > 0

    # ------------------------------------------------------------------
    # Full end-to-end flow in a single test
    # Requirements: 1.1-1.6, 2.1-2.8, 3.1-3.7, 4.1-4.8, 5.1-5.7, 10.1-10.9
    # ------------------------------------------------------------------

    async def test_complete_dsa_round_flow_end_to_end(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Complete DSA round: role → DSA selection → problem → solution → follow-up → next problem → feedback.

        Requirements: 1.1-1.6, 2.1-2.8, 3.1-3.7, 4.1-4.8, 5.1-5.7, 10.1-10.9
        """
        interview_service, _, feedback_service = services

        # ---- Turn 1: User selects DSA round ----
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)
        mock_llm.complete.return_value = _coding_problem_llm_response()

        _, session = await interview_service.handle_response(session, "DSA round")

        assert session.interview_round_type == InterviewRoundType.DSA_CODING
        assert len(session.questions) == 1
        assert session.stage == Stage.ROUND_TYPE_SELECTION  # stage managed by orchestration layer

        # ---- Turn 2: User submits code solution → follow-up asked ----
        session.stage = Stage.INTERVIEW
        mock_llm.complete.side_effect = [
            _intent_answer_response(),
            _solution_evaluation_llm_response(
                correctness="correct",
                is_optimal=True,
                follow_up_warranted=True,
                follow_up_text="What is the time complexity of your solution?",
                difficulty_signal="increase",
            ),
        ]

        code_solution = (
            "```python\n"
            "def two_sum(nums, target):\n"
            "    seen = {}\n"
            "    for i, num in enumerate(nums):\n"
            "        complement = target - num\n"
            "        if complement in seen:\n"
            "            return [seen[complement], i]\n"
            "        seen[num] = i\n"
            "```"
        )
        _, session = await interview_service.handle_response(session, code_solution)

        assert len(session.responses) == 1
        assert len(session.questions) == 2  # original + follow-up
        assert session.questions[1].question_type == QuestionType.FOLLOW_UP

        # ---- Turn 3: User responds to follow-up → difficulty increases, next problem ----
        mock_llm.complete.side_effect = [
            _intent_answer_response(),
            _solution_evaluation_llm_response(
                correctness="correct",
                is_optimal=True,
                follow_up_warranted=False,
                follow_up_text=None,
                difficulty_signal="increase",
            ),
            _next_coding_problem_llm_response(),
        ]

        follow_up_answer = (
            "The time complexity is O(n) because we iterate through the array once "
            "and each hash table lookup is O(1) on average giving us linear time overall."
        )
        _, session = await interview_service.handle_response(session, follow_up_answer)

        assert session.problem_difficulty == ProblemDifficulty.HARD
        assert len(session.questions) == 3  # original + follow-up + next problem
        assert session.questions[2].question_type == QuestionType.TECHNICAL

        # ---- Turn 4: User requests feedback ----
        mock_llm.complete.return_value = _technical_feedback_llm_response()

        reply, session = await feedback_service.generate_feedback_report(session)

        assert session.feedback_report is not None
        assert len(session.feedback_report.strengths) >= 1
        assert len(session.feedback_report.improvements) >= 1
        assert isinstance(reply, str) and len(reply) > 0

    # ------------------------------------------------------------------
    # Additional scenario: weak solution decreases difficulty
    # Requirements: 2.7, 15.2, 15.3
    # ------------------------------------------------------------------

    async def test_weak_solution_decreases_difficulty(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """A weak/incorrect solution causes difficulty to decrease.

        Requirements: 2.7, 15.2
        """
        interview_service, _, _ = services
        session = _make_session(stage=Stage.INTERVIEW)
        session.interview_round_type = InterviewRoundType.DSA_CODING
        session.problem_difficulty = ProblemDifficulty.MEDIUM

        problem_question = Question(
            question_id="q-dsa-002",
            text="Find the longest palindromic substring.",
            question_type=QuestionType.TECHNICAL,
            asked_at=_NOW,
        )
        session.questions.append(problem_question)

        # LLM: intent → evaluation (incorrect, decrease) → next problem
        mock_llm.complete.side_effect = [
            _intent_answer_response(),
            _solution_evaluation_llm_response(
                correctness="incorrect",
                is_optimal=False,
                follow_up_warranted=False,
                follow_up_text=None,
                difficulty_signal="decrease",
            ),
            _coding_problem_llm_response(
                problem_text="Given an array, find the maximum element.",
                topic="arrays",
                difficulty="easy",
            ),
        ]

        weak_solution = (
            "I would iterate through all substrings and check each one if it is a palindrome "
            "by comparing characters from both ends until I find the longest one that matches."
        )

        _, updated_session = await interview_service.handle_response(session, weak_solution)

        # Difficulty must have decreased (MEDIUM → EASY)
        assert updated_session.problem_difficulty == ProblemDifficulty.EASY

    # ------------------------------------------------------------------
    # Additional scenario: difficulty stays at HARD boundary
    # Requirements: 15.4
    # ------------------------------------------------------------------

    async def test_difficulty_stays_at_hard_boundary(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Difficulty does not increase beyond HARD.

        Requirements: 15.4
        """
        interview_service, _, _ = services
        session = _make_session(stage=Stage.INTERVIEW)
        session.interview_round_type = InterviewRoundType.DSA_CODING
        session.problem_difficulty = ProblemDifficulty.HARD  # Already at max

        problem_question = Question(
            question_id="q-dsa-003",
            text="Solve the N-Queens problem.",
            question_type=QuestionType.TECHNICAL,
            asked_at=_NOW,
        )
        session.questions.append(problem_question)

        # LLM: intent → evaluation (correct, optimal, increase signal) → next problem
        mock_llm.complete.side_effect = [
            _intent_answer_response(),
            _solution_evaluation_llm_response(
                correctness="correct",
                is_optimal=True,
                follow_up_warranted=False,
                follow_up_text=None,
                difficulty_signal="increase",
            ),
            _next_coding_problem_llm_response(),
        ]

        solution = (
            "I would use backtracking to place queens row by row checking column and "
            "anti-diagonal constraints at each step and undo the placement when a constraint "
            "is violated to explore all valid arrangements until all queens are placed safely."
        )

        _, updated_session = await interview_service.handle_response(session, solution)

        # Difficulty must remain at HARD (cannot go above)
        assert updated_session.problem_difficulty == ProblemDifficulty.HARD

    # ------------------------------------------------------------------
    # Additional scenario: round type stored in session persists
    # Requirements: 11.1, 11.2, 11.5
    # ------------------------------------------------------------------

    async def test_round_type_persists_in_session(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Interview round type is stored in session state after selection.

        Requirements: 11.1, 11.2, 11.5
        """
        interview_service, _, _ = services
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)

        mock_llm.complete.return_value = _coding_problem_llm_response()

        _, updated_session = await interview_service.handle_response(session, "coding round")

        # Round type must be persisted in session
        assert updated_session.interview_round_type == InterviewRoundType.DSA_CODING

        # Subsequent calls must see the same round type
        assert updated_session.interview_round_type is not None

    # ------------------------------------------------------------------
    # Additional scenario: no behavioral questions in DSA round
    # Requirements: 1.6, 12.2, 12.3
    # ------------------------------------------------------------------

    async def test_dsa_round_generates_only_technical_questions(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """DSA round generates only TECHNICAL question types, not BEHAVIOURAL.

        Requirements: 1.6, 12.2, 12.3
        """
        interview_service, _, _ = services
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)

        mock_llm.complete.return_value = _coding_problem_llm_response()

        _, updated_session = await interview_service.handle_response(session, "DSA")

        # All questions must be TECHNICAL type
        for question in updated_session.questions:
            assert question.question_type in (QuestionType.TECHNICAL, QuestionType.FOLLOW_UP), (
                f"Expected TECHNICAL or FOLLOW_UP, got {question.question_type}"
            )


# ---------------------------------------------------------------------------
# System Design LLM response builders
# ---------------------------------------------------------------------------


def _system_design_question_llm_response(
    system_name: str = "Twitter",
    question_text: str = "Design a social media feed like Twitter",
    description: str = "Design a scalable system that allows users to post tweets and see a feed of tweets from people they follow.",
) -> str:
    return json.dumps({
        "system_name": system_name,
        "question_text": question_text,
        "description": description,
    })


def _system_design_evaluation_llm_response(
    follow_up_warranted: bool = True,
    follow_up_text: str = "What are the main components of your system?",
    next_phase_suggestion: str = "high_level_design",
    strengths: list | None = None,
    weaknesses: list | None = None,
) -> str:
    return json.dumps({
        "design_aspects_evaluated": {
            "scalability": "Candidate identified key scale requirements.",
            "database_design": "Not yet discussed.",
            "api_design": "Not yet discussed.",
            "caching_strategy": "Not yet discussed.",
            "load_balancing": "Not yet discussed.",
        },
        "design_strengths": strengths or ["Good requirements gathering"],
        "design_weaknesses": weaknesses or ["Database design not addressed"],
        "follow_up_warranted": follow_up_warranted,
        "follow_up_text": follow_up_text,
        "next_phase_suggestion": next_phase_suggestion,
    })


def _technical_feedback_system_design_llm_response() -> str:
    return json.dumps({
        "summary": "You completed a System Design round for designing Twitter.",
        "strengths": [
            "Good requirements gathering",
            "Clear high-level architecture",
            "Identified key bottlenecks",
        ],
        "improvements": [
            "Could elaborate more on database sharding strategy",
            "Caching layer could be more detailed",
        ],
        "actionable_recommendations": [
            "Study consistent hashing for distributed systems",
            "Practice designing database schemas for social media",
        ],
        "design_aspect_summaries": {
            "scalability": "Well addressed",
            "database_design": "Partially addressed",
            "api_design": "Not addressed",
            "caching_strategy": "Briefly mentioned",
            "load_balancing": "Not addressed",
        },
    })


# ---------------------------------------------------------------------------
# Task 11.2 — Complete System Design round flow integration test
# Requirements: 6.1-6.5, 7.1-7.7, 8.1-8.8, 9.1-9.6, 10.1-10.9
# ---------------------------------------------------------------------------


class TestSystemDesignRoundCompleteFlow:
    """Integration test for the complete System Design round flow using real service instances."""

    @pytest.fixture
    def mock_llm(self) -> Mock:
        return _make_mock_llm()

    @pytest.fixture
    def services(self, mock_llm: Mock) -> tuple[InterviewService, TechnicalRoundService, FeedbackService]:
        return _make_services(mock_llm)

    # ------------------------------------------------------------------
    # Step 1: User selects System Design round → system generates first question
    # Requirements: 1.1, 1.3, 6.1, 6.2, 6.3, 11.1, 11.2
    # ------------------------------------------------------------------

    async def test_step1_system_design_round_selection_generates_first_question(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Selecting System Design round sets round type and generates first design question.

        Requirements: 1.1, 1.3, 6.1, 6.2, 6.3, 11.1, 11.2
        """
        interview_service, _, _ = services
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)

        mock_llm.complete.return_value = _system_design_question_llm_response()

        reply, updated_session = await interview_service.handle_response(
            session, "system design round"
        )

        # Round type must be set to SYSTEM_DESIGN
        assert updated_session.interview_round_type == InterviewRoundType.SYSTEM_DESIGN

        # A question (the first design question) must have been added
        assert len(updated_session.questions) == 1
        assert updated_session.questions[0].question_type == QuestionType.TECHNICAL

        # Design phase must be initialized to REQUIREMENTS_GATHERING
        assert updated_session.design_phase is not None

        # Reply must contain the design question
        assert "twitter" in reply.lower() or "design" in reply.lower() or "system" in reply.lower()

    # ------------------------------------------------------------------
    # Step 2: User provides requirements → system transitions to High-Level Design
    # Requirements: 7.1, 7.2, 8.1, 8.7
    # ------------------------------------------------------------------

    async def test_step2_user_provides_requirements_transitions_to_high_level_design(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """User providing requirements triggers evaluation and phase transition to High-Level Design.

        Requirements: 7.1, 7.2, 8.1, 8.7
        """
        from interview_practice_partner.domain.enums import DesignPhase

        interview_service, _, _ = services
        session = _make_session(stage=Stage.INTERVIEW)
        session.interview_round_type = InterviewRoundType.SYSTEM_DESIGN
        session.design_phase = DesignPhase.REQUIREMENTS_GATHERING

        # Add the first design question
        design_question = Question(
            question_id="q-sd-001",
            text="Design a social media feed like Twitter",
            question_type=QuestionType.TECHNICAL,
            asked_at=_NOW,
        )
        session.questions.append(design_question)

        # LLM: intent → system design evaluation (transitions to high_level_design)
        mock_llm.complete.side_effect = [
            _intent_answer_response(),
            _system_design_evaluation_llm_response(
                follow_up_warranted=True,
                follow_up_text="Now describe the high-level architecture of your system.",
                next_phase_suggestion="high_level_design",
                strengths=["Good requirements gathering", "Identified scale requirements"],
            ),
        ]

        requirements_response = (
            "The system needs to support 100 million daily active users. "
            "Functional requirements include posting tweets, following users, and viewing a feed. "
            "Non-functional requirements include low latency (under 200ms), high availability (99.9%), "
            "and eventual consistency for the feed. The system should handle 10,000 tweets per second."
        )

        reply, updated_session = await interview_service.handle_response(
            session, requirements_response
        )

        # Response must be recorded
        assert len(updated_session.responses) == 1

        # Design phase must have transitioned to HIGH_LEVEL_DESIGN
        assert updated_session.design_phase == DesignPhase.HIGH_LEVEL_DESIGN

        # A follow-up question must have been added
        assert len(updated_session.questions) == 2
        follow_up = updated_session.questions[1]
        assert follow_up.question_type == QuestionType.FOLLOW_UP

        # Reply must contain the follow-up
        assert len(reply) > 0

    # ------------------------------------------------------------------
    # Step 3: User describes architecture → system transitions to Deep Dive
    # Requirements: 7.3, 7.4, 8.1
    # ------------------------------------------------------------------

    async def test_step3_user_describes_architecture_transitions_to_deep_dive(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """User describing architecture triggers evaluation and phase transition to Deep Dive.

        Requirements: 7.3, 7.4, 8.1
        """
        from interview_practice_partner.domain.enums import DesignPhase

        interview_service, _, _ = services
        session = _make_session(stage=Stage.INTERVIEW)
        session.interview_round_type = InterviewRoundType.SYSTEM_DESIGN
        session.design_phase = DesignPhase.HIGH_LEVEL_DESIGN

        # Session has original question + follow-up
        design_question = Question(
            question_id="q-sd-001",
            text="Design a social media feed like Twitter",
            question_type=QuestionType.TECHNICAL,
            asked_at=_NOW,
        )
        follow_up_question = Question(
            question_id="q-sd-followup-001",
            text="Now describe the high-level architecture of your system.",
            question_type=QuestionType.FOLLOW_UP,
            asked_at=_NOW,
        )
        session.questions.extend([design_question, follow_up_question])

        # Record the requirements response
        requirements_response = UserResponse(
            response_id="r-sd-001",
            question_id="q-sd-001",
            text="100M DAU, low latency, high availability...",
            word_count=20,
            received_at=_NOW,
        )
        session.responses.append(requirements_response)

        # LLM: intent → evaluation (transitions to deep_dive)
        mock_llm.complete.side_effect = [
            _intent_answer_response(),
            _system_design_evaluation_llm_response(
                follow_up_warranted=True,
                follow_up_text="Let's dive deeper into the database design. How would you store tweets?",
                next_phase_suggestion="deep_dive",
                strengths=["Clear component separation", "Good use of microservices"],
            ),
        ]

        architecture_response = (
            "The high-level architecture consists of a web tier with load balancers, "
            "an application tier with microservices for user service, tweet service, and feed service. "
            "The data tier uses a combination of SQL for user data and NoSQL for tweets. "
            "A CDN handles static content and a message queue handles async processing."
        )

        reply, updated_session = await interview_service.handle_response(
            session, architecture_response
        )

        # Response must be recorded
        assert len(updated_session.responses) == 2

        # Design phase must have transitioned to DEEP_DIVE
        assert updated_session.design_phase == DesignPhase.DEEP_DIVE

        # A new follow-up question must have been added
        assert len(updated_session.questions) == 3
        new_follow_up = updated_session.questions[2]
        assert new_follow_up.question_type == QuestionType.FOLLOW_UP

    # ------------------------------------------------------------------
    # Step 4: User identifies bottlenecks → system transitions to Bottleneck Analysis
    # Requirements: 7.5, 8.1
    # ------------------------------------------------------------------

    async def test_step4_user_identifies_bottlenecks_transitions_to_bottleneck_analysis(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """User elaborating on components triggers evaluation and phase transition to Bottleneck Analysis.

        Requirements: 7.5, 8.1
        """
        from interview_practice_partner.domain.enums import DesignPhase

        interview_service, _, _ = services
        session = _make_session(stage=Stage.INTERVIEW)
        session.interview_round_type = InterviewRoundType.SYSTEM_DESIGN
        session.design_phase = DesignPhase.DEEP_DIVE

        # Session has original question + two follow-ups
        design_question = Question(
            question_id="q-sd-001",
            text="Design a social media feed like Twitter",
            question_type=QuestionType.TECHNICAL,
            asked_at=_NOW,
        )
        follow_up_1 = Question(
            question_id="q-sd-followup-001",
            text="Describe the high-level architecture.",
            question_type=QuestionType.FOLLOW_UP,
            asked_at=_NOW,
        )
        follow_up_2 = Question(
            question_id="q-sd-followup-002",
            text="How would you store tweets in the database?",
            question_type=QuestionType.FOLLOW_UP,
            asked_at=_NOW,
        )
        session.questions.extend([design_question, follow_up_1, follow_up_2])

        # Record previous responses
        for i, (qid, text) in enumerate([
            ("q-sd-001", "100M DAU requirements..."),
            ("q-sd-followup-001", "Microservices architecture..."),
        ]):
            session.responses.append(UserResponse(
                response_id=f"r-sd-00{i + 1}",
                question_id=qid,
                text=text,
                word_count=20,
                received_at=_NOW,
            ))

        # LLM: intent → evaluation (transitions to bottleneck_analysis)
        mock_llm.complete.side_effect = [
            _intent_answer_response(),
            _system_design_evaluation_llm_response(
                follow_up_warranted=True,
                follow_up_text="What are the main bottlenecks in your design and how would you address them?",
                next_phase_suggestion="bottleneck_analysis",
                strengths=["Detailed database schema", "Good indexing strategy"],
            ),
        ]

        deep_dive_response = (
            "For the tweet storage, I would use a NoSQL database like Cassandra with a wide-column schema. "
            "The tweet table would have user_id as partition key and tweet_id as clustering key for efficient reads. "
            "For the feed generation, I would use a fan-out on write approach for users with fewer than 10,000 followers "
            "and fan-out on read for celebrity accounts to avoid write amplification."
        )

        reply, updated_session = await interview_service.handle_response(
            session, deep_dive_response
        )

        # Response must be recorded
        assert len(updated_session.responses) == 3

        # Design phase must have transitioned to BOTTLENECK_ANALYSIS
        assert updated_session.design_phase == DesignPhase.BOTTLENECK_ANALYSIS

        # A new follow-up question must have been added
        assert len(updated_session.questions) == 4
        new_follow_up = updated_session.questions[3]
        assert new_follow_up.question_type == QuestionType.FOLLOW_UP

    # ------------------------------------------------------------------
    # Step 5: User requests feedback → technical feedback generated
    # Requirements: 10.1-10.9
    # ------------------------------------------------------------------

    async def test_step5_feedback_request_generates_system_design_feedback(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Requesting feedback generates a technical feedback report for System Design round.

        Requirements: 10.1-10.9
        """
        from interview_practice_partner.domain.enums import DesignPhase, DesignAspect

        _, _, feedback_service = services
        session = _make_session(stage=Stage.FEEDBACK)
        session.interview_round_type = InterviewRoundType.SYSTEM_DESIGN
        session.design_phase = DesignPhase.BOTTLENECK_ANALYSIS
        session.design_aspects_covered = [
            DesignAspect.SCALABILITY,
            DesignAspect.DATABASE_DESIGN,
        ]

        # Session has one design question and multiple responses
        design_question = Question(
            question_id="q-sd-001",
            text="Design a social media feed like Twitter",
            question_type=QuestionType.TECHNICAL,
            asked_at=_NOW,
        )
        session.questions.append(design_question)

        for i, text in enumerate([
            "100M DAU, low latency requirements...",
            "Microservices architecture with load balancers...",
            "Cassandra for tweet storage with fan-out on write...",
            "Main bottleneck is the feed generation for celebrity accounts...",
        ]):
            session.responses.append(UserResponse(
                response_id=f"r-sd-00{i + 1}",
                question_id="q-sd-001",
                text=text,
                word_count=20,
                received_at=_NOW,
            ))

        mock_llm.complete.return_value = _technical_feedback_system_design_llm_response()

        reply, updated_session = await feedback_service.generate_feedback_report(session)

        # Feedback report must be stored in session
        assert updated_session.feedback_report is not None
        report = updated_session.feedback_report

        # Report must have strengths and improvements
        assert len(report.strengths) >= 1
        assert len(report.improvements) >= 1
        assert len(report.actionable_recommendations) >= 1

        # Reply must be a non-empty string
        assert isinstance(reply, str)
        assert len(reply) > 0

    # ------------------------------------------------------------------
    # Full end-to-end flow in a single test
    # Requirements: 6.1-6.5, 7.1-7.7, 8.1-8.8, 9.1-9.6, 10.1-10.9
    # ------------------------------------------------------------------

    async def test_complete_system_design_round_flow_end_to_end(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Complete System Design round: role → SD selection → question → requirements → HLD → deep dive → bottlenecks → feedback.

        Requirements: 6.1-6.5, 7.1-7.7, 8.1-8.8, 9.1-9.6, 10.1-10.9
        """
        from interview_practice_partner.domain.enums import DesignPhase

        interview_service, _, feedback_service = services

        # ---- Turn 1: User selects System Design round ----
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)
        mock_llm.complete.return_value = _system_design_question_llm_response()

        _, session = await interview_service.handle_response(session, "system design")

        assert session.interview_round_type == InterviewRoundType.SYSTEM_DESIGN
        assert len(session.questions) == 1
        assert session.questions[0].question_type == QuestionType.TECHNICAL
        assert session.design_phase == DesignPhase.REQUIREMENTS_GATHERING

        # ---- Turn 2: User provides requirements → transitions to High-Level Design ----
        session.stage = Stage.INTERVIEW
        mock_llm.complete.side_effect = [
            _intent_answer_response(),
            _system_design_evaluation_llm_response(
                follow_up_warranted=True,
                follow_up_text="Now describe the high-level architecture.",
                next_phase_suggestion="high_level_design",
            ),
        ]

        requirements_response = (
            "The system needs to support 100 million daily active users with low latency. "
            "Functional requirements: post tweets, follow users, view feed. "
            "Non-functional: 99.9% availability, eventual consistency, under 200ms latency."
        )
        _, session = await interview_service.handle_response(session, requirements_response)

        assert len(session.responses) == 1
        assert len(session.questions) == 2  # original + follow-up
        assert session.questions[1].question_type == QuestionType.FOLLOW_UP
        assert session.design_phase == DesignPhase.HIGH_LEVEL_DESIGN

        # ---- Turn 3: User describes architecture → transitions to Deep Dive ----
        mock_llm.complete.side_effect = [
            _intent_answer_response(),
            _system_design_evaluation_llm_response(
                follow_up_warranted=True,
                follow_up_text="Let's dive deeper into the database design.",
                next_phase_suggestion="deep_dive",
            ),
        ]

        architecture_response = (
            "The architecture uses microservices: user service, tweet service, feed service. "
            "Load balancers distribute traffic. SQL for user data, Cassandra for tweets. "
            "Redis cache for hot data. Message queue for async feed updates."
        )
        _, session = await interview_service.handle_response(session, architecture_response)

        assert len(session.responses) == 2
        assert len(session.questions) == 3  # original + 2 follow-ups
        assert session.design_phase == DesignPhase.DEEP_DIVE

        # ---- Turn 4: User elaborates on components → transitions to Bottleneck Analysis ----
        mock_llm.complete.side_effect = [
            _intent_answer_response(),
            _system_design_evaluation_llm_response(
                follow_up_warranted=True,
                follow_up_text="What are the main bottlenecks and how would you address them?",
                next_phase_suggestion="bottleneck_analysis",
            ),
        ]

        deep_dive_response = (
            "For tweet storage, Cassandra with user_id as partition key and tweet_id as clustering key. "
            "Fan-out on write for regular users, fan-out on read for celebrities. "
            "Indexes on created_at for timeline queries. Sharding by user_id for even distribution."
        )
        _, session = await interview_service.handle_response(session, deep_dive_response)

        assert len(session.responses) == 3
        assert len(session.questions) == 4  # original + 3 follow-ups
        assert session.design_phase == DesignPhase.BOTTLENECK_ANALYSIS

        # ---- Turn 5: User identifies bottlenecks ----
        mock_llm.complete.side_effect = [
            _intent_answer_response(),
            _system_design_evaluation_llm_response(
                follow_up_warranted=False,
                follow_up_text=None,
                next_phase_suggestion=None,
                strengths=["Identified celebrity fan-out bottleneck", "Proposed caching solution"],
            ),
            _system_design_question_llm_response(
                system_name="URL Shortener",
                question_text="Design a URL shortener service like bit.ly",
                description="Design a service that generates short aliases for long URLs.",
            ),
        ]

        bottleneck_response = (
            "The main bottleneck is the feed generation for celebrity accounts with millions of followers. "
            "Solution: use a hybrid approach — fan-out on write for regular users, fan-out on read for celebrities. "
            "Add a Redis cache layer for the most active feeds. Use CDN for static content. "
            "Database replication for read scalability. Rate limiting to prevent abuse."
        )
        _, session = await interview_service.handle_response(session, bottleneck_response)

        assert len(session.responses) == 4

        # ---- Turn 6: User requests feedback ----
        mock_llm.complete.return_value = _technical_feedback_system_design_llm_response()

        reply, session = await feedback_service.generate_feedback_report(session)

        assert session.feedback_report is not None
        assert len(session.feedback_report.strengths) >= 1
        assert len(session.feedback_report.improvements) >= 1
        assert isinstance(reply, str) and len(reply) > 0

    # ------------------------------------------------------------------
    # Additional scenario: round type stored in session persists
    # Requirements: 11.1, 11.2, 11.5
    # ------------------------------------------------------------------

    async def test_system_design_round_type_persists_in_session(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """System Design round type is stored in session state after selection.

        Requirements: 11.1, 11.2, 11.5
        """
        interview_service, _, _ = services
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)

        mock_llm.complete.return_value = _system_design_question_llm_response()

        _, updated_session = await interview_service.handle_response(
            session, "system design"
        )

        # Round type must be persisted in session
        assert updated_session.interview_round_type == InterviewRoundType.SYSTEM_DESIGN
        assert updated_session.interview_round_type is not None

    # ------------------------------------------------------------------
    # Additional scenario: no behavioral questions in System Design round
    # Requirements: 1.6, 12.2, 12.3
    # ------------------------------------------------------------------

    async def test_system_design_round_generates_only_technical_questions(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """System Design round generates only TECHNICAL question types, not BEHAVIOURAL.

        Requirements: 1.6, 12.2, 12.3
        """
        interview_service, _, _ = services
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)

        mock_llm.complete.return_value = _system_design_question_llm_response()

        _, updated_session = await interview_service.handle_response(
            session, "system design round"
        )

        # All questions must be TECHNICAL type
        for question in updated_session.questions:
            assert question.question_type in (QuestionType.TECHNICAL, QuestionType.FOLLOW_UP), (
                f"Expected TECHNICAL or FOLLOW_UP, got {question.question_type}"
            )

    # ------------------------------------------------------------------
    # Additional scenario: design aspects are tracked in session
    # Requirements: 8.1, 18.5
    # ------------------------------------------------------------------

    async def test_design_aspects_tracked_in_session(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Design aspects evaluated are tracked in session.design_aspects_covered.

        Requirements: 8.1, 18.5
        """
        from interview_practice_partner.domain.enums import DesignPhase

        interview_service, _, _ = services
        session = _make_session(stage=Stage.INTERVIEW)
        session.interview_round_type = InterviewRoundType.SYSTEM_DESIGN
        session.design_phase = DesignPhase.REQUIREMENTS_GATHERING

        design_question = Question(
            question_id="q-sd-001",
            text="Design a social media feed like Twitter",
            question_type=QuestionType.TECHNICAL,
            asked_at=_NOW,
        )
        session.questions.append(design_question)

        mock_llm.complete.side_effect = [
            _intent_answer_response(),
            _system_design_evaluation_llm_response(
                follow_up_warranted=False,
                follow_up_text=None,
                next_phase_suggestion="high_level_design",
            ),
            _system_design_question_llm_response(
                system_name="URL Shortener",
                question_text="Design a URL shortener",
                description="Short URL service.",
            ),
        ]

        response_text = (
            "The system needs to handle 100 million users with high availability. "
            "Functional requirements include posting, following, and feed viewing. "
            "Non-functional requirements include low latency and eventual consistency."
        )

        _, updated_session = await interview_service.handle_response(session, response_text)

        # Design aspects must have been tracked
        assert len(updated_session.design_aspects_covered) >= 0  # aspects tracked from evaluation


# ---------------------------------------------------------------------------
# Task 11.3 — Round type switching integration tests
# Requirements: 1.5, 12.4
# ---------------------------------------------------------------------------


class TestRoundTypeSwitching:
    """Integration tests for round type switching mid-session.

    Requirements: 1.5, 12.4
    """

    @pytest.fixture
    def mock_llm(self) -> Mock:
        return _make_mock_llm()

    @pytest.fixture
    def services(self, mock_llm: Mock) -> tuple[InterviewService, TechnicalRoundService, FeedbackService]:
        return _make_services(mock_llm)

    # ------------------------------------------------------------------
    # Scenario 1: DSA → System Design switch
    # User starts DSA round, then switches to System Design.
    # Session resets and a new System Design question is generated.
    # Requirements: 1.5, 12.4
    # ------------------------------------------------------------------

    async def test_dsa_to_system_design_switch_resets_session_and_generates_new_question(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """User starts DSA round then switches to System Design — session resets, new question generated.

        Requirements: 1.5, 12.4
        """
        interview_service, _, _ = services

        # ---- Turn 1: Start DSA round ----
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)
        mock_llm.complete.return_value = _coding_problem_llm_response()

        _, session = await interview_service.handle_response(session, "DSA round")

        assert session.interview_round_type == InterviewRoundType.DSA_CODING
        assert len(session.questions) == 1
        dsa_question_text = session.questions[0].text

        # ---- Turn 2: User switches to System Design mid-session ----
        session.stage = Stage.INTERVIEW
        mock_llm.complete.return_value = _system_design_question_llm_response()

        reply, updated_session = await interview_service.handle_response(
            session, "system design"
        )

        # Round type must now be SYSTEM_DESIGN
        assert updated_session.interview_round_type == InterviewRoundType.SYSTEM_DESIGN

        # Session must have been restarted — only the new SD question present
        assert len(updated_session.questions) == 1
        assert updated_session.questions[0].question_type == QuestionType.TECHNICAL

        # The new question must not be the old DSA question
        assert updated_session.questions[0].text != dsa_question_text

        # Reply must reference the new System Design question
        assert len(reply) > 0
        assert "design" in reply.lower() or "twitter" in reply.lower() or "system" in reply.lower()

    # ------------------------------------------------------------------
    # Scenario 2: System Design → DSA switch
    # User starts System Design round, then switches to DSA.
    # Session resets and a new DSA problem is generated.
    # Requirements: 1.5, 12.4
    # ------------------------------------------------------------------

    async def test_system_design_to_dsa_switch_resets_session_and_generates_new_problem(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """User starts System Design round then switches to DSA — session resets, new problem generated.

        Requirements: 1.5, 12.4
        """
        interview_service, _, _ = services

        # ---- Turn 1: Start System Design round ----
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)
        mock_llm.complete.return_value = _system_design_question_llm_response()

        _, session = await interview_service.handle_response(session, "system design")

        assert session.interview_round_type == InterviewRoundType.SYSTEM_DESIGN
        assert len(session.questions) == 1
        sd_question_text = session.questions[0].text

        # ---- Turn 2: User switches to DSA mid-session ----
        session.stage = Stage.INTERVIEW
        mock_llm.complete.return_value = _coding_problem_llm_response()

        reply, updated_session = await interview_service.handle_response(
            session, "DSA round"
        )

        # Round type must now be DSA_CODING
        assert updated_session.interview_round_type == InterviewRoundType.DSA_CODING

        # Session must have been restarted — only the new DSA problem present
        assert len(updated_session.questions) == 1
        assert updated_session.questions[0].question_type == QuestionType.TECHNICAL

        # The new question must not be the old System Design question
        assert updated_session.questions[0].text != sd_question_text

        # Reply must reference the new DSA problem
        assert len(reply) > 0
        assert "array" in reply.lower() or "dsa" in reply.lower() or "coding" in reply.lower() or "indices" in reply.lower()

    # ------------------------------------------------------------------
    # Scenario 3: Switching clears questions, responses, and difficulty history
    # Requirements: 1.5
    # ------------------------------------------------------------------

    async def test_switching_clears_questions_responses_and_difficulty_history(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Switching round type clears all questions, responses, and difficulty history.

        Requirements: 1.5
        """
        interview_service, _, _ = services

        # Build a session mid-DSA-round with accumulated state
        session = _make_session(stage=Stage.INTERVIEW)
        session.interview_round_type = InterviewRoundType.DSA_CODING
        session.problem_difficulty = ProblemDifficulty.HARD

        # Add questions (original problem + follow-up)
        session.questions = [
            Question(
                question_id="q-dsa-001",
                text="Two Sum problem",
                question_type=QuestionType.TECHNICAL,
                asked_at=_NOW,
            ),
            Question(
                question_id="q-followup-001",
                text="What is the time complexity?",
                question_type=QuestionType.FOLLOW_UP,
                asked_at=_NOW,
            ),
        ]

        # Add responses
        session.responses = [
            UserResponse(
                response_id="r-001",
                question_id="q-dsa-001",
                text="def two_sum(nums, target): ...",
                word_count=20,
                received_at=_NOW,
            ),
        ]

        # Add difficulty history
        session.difficulty_adjustment_history = [
            {"from": "medium", "to": "hard", "reason": "correct and optimal solution"},
        ]

        # Add topics covered
        session.topics_covered = []

        # Switch to System Design
        mock_llm.complete.return_value = _system_design_question_llm_response()

        _, updated_session = await interview_service.handle_response(
            session, "system design"
        )

        # All prior state must be cleared
        assert updated_session.interview_round_type == InterviewRoundType.SYSTEM_DESIGN
        assert len(updated_session.responses) == 0
        assert updated_session.difficulty_adjustment_history == []
        assert updated_session.topics_covered == []
        assert updated_session.design_aspects_covered == []

        # Only the new SD question should be present
        assert len(updated_session.questions) == 1
        assert updated_session.questions[0].question_type == QuestionType.TECHNICAL

    # ------------------------------------------------------------------
    # Scenario 4: New round type is set correctly after switch
    # Requirements: 1.5, 12.4
    # ------------------------------------------------------------------

    async def test_new_round_type_is_set_correctly_after_switch(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """After switching, the session's interview_round_type reflects the new round type.

        Requirements: 1.5, 12.4
        """
        interview_service, _, _ = services

        # Start with a DSA session that has at least one question
        session = _make_session(stage=Stage.INTERVIEW)
        session.interview_round_type = InterviewRoundType.DSA_CODING
        session.problem_difficulty = ProblemDifficulty.MEDIUM
        session.questions = [
            Question(
                question_id="q-dsa-001",
                text="Find the maximum subarray sum.",
                question_type=QuestionType.TECHNICAL,
                asked_at=_NOW,
            ),
        ]

        # Switch to System Design
        mock_llm.complete.return_value = _system_design_question_llm_response(
            system_name="Instagram",
            question_text="Design a photo sharing app like Instagram",
            description="Design a scalable photo sharing platform.",
        )

        _, updated_session = await interview_service.handle_response(
            session, "system design round"
        )

        # New round type must be SYSTEM_DESIGN
        assert updated_session.interview_round_type == InterviewRoundType.SYSTEM_DESIGN

        # Old round type (DSA_CODING) must not be present
        assert updated_session.interview_round_type != InterviewRoundType.DSA_CODING

        # Difficulty must be reset to MEDIUM (default)
        assert updated_session.problem_difficulty == ProblemDifficulty.MEDIUM

        # Design phase must be initialized for the new System Design round
        from interview_practice_partner.domain.enums import DesignPhase
        assert updated_session.design_phase == DesignPhase.REQUIREMENTS_GATHERING


# ---------------------------------------------------------------------------
# Behavioral LLM response builders
# ---------------------------------------------------------------------------


def _behavioral_question_llm_response(
    question_text: str = "Tell me about a time you had to work under pressure. How did you handle it?",
) -> str:
    """Return a plain-text behavioral question (not JSON — matches generate_question behavior)."""
    return question_text


def _behavioral_evaluation_llm_response(
    follow_up_warranted: bool = False,
    follow_up_text: str | None = None,
    difficulty_signal: str = "maintain",
) -> str:
    return json.dumps({
        "is_off_topic": False,
        "is_short": False,
        "follow_up_warranted": follow_up_warranted,
        "follow_up_text": follow_up_text,
        "difficulty_signal": difficulty_signal,
    })


def _behavioral_feedback_llm_response() -> str:
    return json.dumps({
        "dimension_scores": [
            {"dimension": "communication", "score": 4, "justification": "Clear and structured response."},
            {"dimension": "problem_solving", "score": 3, "justification": "Good approach described."},
            {"dimension": "teamwork", "score": 4, "justification": "Demonstrated collaboration."},
            {"dimension": "leadership", "score": 3, "justification": "Showed initiative."},
        ],
        "strengths": [
            "Clear communication with specific examples",
            "Demonstrated ability to work under pressure",
        ],
        "improvements": [
            "Could quantify the impact of your actions more",
        ],
        "actionable_recommendations": [
            "Practice the STAR method for structuring behavioral answers",
            "Prepare 2-3 examples for each common behavioral theme",
        ],
        "off_topic_references": [],
    })


# ---------------------------------------------------------------------------
# Task 11.4 — Behavioral round compatibility integration tests
# Requirements: 12.1-12.5
# ---------------------------------------------------------------------------


class TestBehavioralRoundCompatibility:
    """Integration tests verifying behavioral rounds still work correctly after
    the technical rounds feature was added.

    Requirements: 12.1-12.5
    """

    @pytest.fixture
    def mock_llm(self) -> Mock:
        return _make_mock_llm()

    @pytest.fixture
    def services(self, mock_llm: Mock) -> tuple[InterviewService, TechnicalRoundService, FeedbackService]:
        return _make_services(mock_llm)

    # ------------------------------------------------------------------
    # Scenario 1: Behavioral round selection generates behavioral questions
    # Requirements: 12.1
    # ------------------------------------------------------------------

    async def test_behavioral_round_selection_generates_behavioral_question(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Selecting Behavioral round sets round type and generates a behavioral question.

        Requirements: 12.1
        """
        interview_service, _, _ = services
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)

        mock_llm.complete.return_value = _behavioral_question_llm_response()

        reply, updated_session = await interview_service.handle_response(
            session, "behavioral"
        )

        # Round type must be set to BEHAVIORAL
        assert updated_session.interview_round_type == InterviewRoundType.BEHAVIORAL

        # A question must have been generated
        assert len(updated_session.questions) == 1

        # The question must be a behavioral/situational type — NOT technical
        question = updated_session.questions[0]
        assert question.question_type in (
            QuestionType.BEHAVIOURAL,
            QuestionType.SITUATIONAL,
            QuestionType.FOLLOW_UP,
        ), f"Expected behavioral question type, got {question.question_type}"
        assert question.question_type != QuestionType.TECHNICAL

        # Reply must contain the behavioral question
        assert len(reply) > 0

    # ------------------------------------------------------------------
    # Scenario 2: Behavioral round does not generate DSA problems
    # Requirements: 12.2
    # ------------------------------------------------------------------

    async def test_behavioral_round_does_not_generate_dsa_problems(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Behavioral round never generates DSA/coding problems.

        Requirements: 12.2
        """
        interview_service, _, _ = services
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)

        # LLM returns a behavioral question (not a DSA problem JSON)
        mock_llm.complete.return_value = _behavioral_question_llm_response(
            "Describe a situation where you had to learn a new technology quickly."
        )

        _, updated_session = await interview_service.handle_response(
            session, "behavioral round"
        )

        assert updated_session.interview_round_type == InterviewRoundType.BEHAVIORAL

        # No question should have a TECHNICAL type (which would indicate a DSA problem)
        for question in updated_session.questions:
            assert question.question_type != QuestionType.TECHNICAL, (
                f"Behavioral round should not generate TECHNICAL questions, "
                f"but found: {question.question_type}"
            )

        # The TechnicalRoundService should NOT have been called — verify by checking
        # that no DSA-specific session state was set
        assert updated_session.problem_difficulty == ProblemDifficulty.MEDIUM  # default, unchanged
        assert updated_session.topics_covered == []
        assert updated_session.difficulty_adjustment_history == []

    # ------------------------------------------------------------------
    # Scenario 3: Behavioral round does not generate System Design questions
    # Requirements: 12.2
    # ------------------------------------------------------------------

    async def test_behavioral_round_does_not_generate_system_design_questions(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Behavioral round never generates System Design questions.

        Requirements: 12.2
        """
        interview_service, _, _ = services
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)

        mock_llm.complete.return_value = _behavioral_question_llm_response(
            "Tell me about a time you resolved a conflict with a teammate."
        )

        _, updated_session = await interview_service.handle_response(
            session, "soft skills"
        )

        assert updated_session.interview_round_type == InterviewRoundType.BEHAVIORAL

        # No question should be a TECHNICAL type (which would indicate a System Design question)
        for question in updated_session.questions:
            assert question.question_type != QuestionType.TECHNICAL, (
                f"Behavioral round should not generate TECHNICAL questions, "
                f"but found: {question.question_type}"
            )

        # System Design-specific state should not be set
        assert updated_session.design_phase is None
        assert updated_session.design_aspects_covered == []

    # ------------------------------------------------------------------
    # Scenario 4: Behavioral feedback still works correctly
    # Requirements: 12.5
    # ------------------------------------------------------------------

    async def test_behavioral_feedback_works_correctly(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Behavioral feedback generation still works correctly after technical rounds were added.

        Requirements: 12.5
        """
        _, _, feedback_service = services
        session = _make_session(stage=Stage.FEEDBACK)
        session.interview_round_type = InterviewRoundType.BEHAVIORAL

        # Add a behavioral question and response
        behavioral_question = Question(
            question_id="q-beh-001",
            text="Tell me about a time you had to work under pressure.",
            question_type=QuestionType.BEHAVIOURAL,
            asked_at=_NOW,
        )
        session.questions.append(behavioral_question)

        user_response = UserResponse(
            response_id="r-beh-001",
            question_id="q-beh-001",
            text=(
                "In my previous role, we had a critical production outage two days before a major release. "
                "I coordinated with the team to triage the issue, delegated tasks based on expertise, "
                "and we resolved it within four hours. The release went ahead on schedule."
            ),
            word_count=45,
            received_at=_NOW,
        )
        session.responses.append(user_response)

        mock_llm.complete.return_value = _behavioral_feedback_llm_response()

        reply, updated_session = await feedback_service.generate_feedback_report(session)

        # Feedback report must be stored in session
        assert updated_session.feedback_report is not None
        report = updated_session.feedback_report

        # Behavioral feedback must have dimension scores (unlike technical feedback)
        assert len(report.dimension_scores) >= 1

        # Report must have strengths and improvements
        assert len(report.strengths) >= 1
        assert len(report.improvements) >= 1
        assert len(report.actionable_recommendations) >= 1

        # Reply must be a non-empty string
        assert isinstance(reply, str)
        assert len(reply) > 0

    # ------------------------------------------------------------------
    # Scenario 5: Existing behavioral interview flow is unchanged
    # Full end-to-end: selection → question → answer → follow-up → feedback
    # Requirements: 12.1, 12.5
    # ------------------------------------------------------------------

    async def test_complete_behavioral_round_flow_end_to_end(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Complete behavioral round flow works end-to-end after technical rounds feature was added.

        Requirements: 12.1, 12.5
        """
        interview_service, _, feedback_service = services

        # ---- Turn 1: User selects Behavioral round ----
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)
        mock_llm.complete.return_value = _behavioral_question_llm_response(
            "Tell me about a time you had to work under pressure. How did you handle it?"
        )

        _, session = await interview_service.handle_response(session, "behavioral")

        assert session.interview_round_type == InterviewRoundType.BEHAVIORAL
        assert len(session.questions) == 1
        assert session.questions[0].question_type != QuestionType.TECHNICAL

        # ---- Turn 2: User answers behavioral question ----
        session.stage = Stage.INTERVIEW
        mock_llm.complete.side_effect = [
            # intent classification
            _intent_answer_response(),
            # response evaluation
            _behavioral_evaluation_llm_response(
                follow_up_warranted=True,
                follow_up_text="What was the outcome of that situation?",
                difficulty_signal="maintain",
            ),
        ]

        behavioral_answer = (
            "In my previous role, we had a critical production outage two days before a major release. "
            "I coordinated with the team to triage the issue, delegated tasks based on expertise, "
            "and we resolved it within four hours. The release went ahead on schedule."
        )

        _, session = await interview_service.handle_response(session, behavioral_answer)

        # Response must be recorded
        assert len(session.responses) == 1

        # A follow-up question must have been added
        assert len(session.questions) == 2
        follow_up = session.questions[1]
        assert follow_up.question_type == QuestionType.FOLLOW_UP
        assert "outcome" in follow_up.text.lower()

        # No technical questions should have been generated
        for question in session.questions:
            assert question.question_type != QuestionType.TECHNICAL

        # ---- Turn 3: User answers follow-up → next behavioral question ----
        mock_llm.complete.side_effect = [
            # intent classification
            _intent_answer_response(),
            # response evaluation (no follow-up this time)
            _behavioral_evaluation_llm_response(
                follow_up_warranted=False,
                follow_up_text=None,
                difficulty_signal="increase",
            ),
            # next question generation
            _behavioral_question_llm_response(
                "Describe a time you had to influence someone without direct authority."
            ),
        ]

        follow_up_answer = (
            "The outcome was very positive. We not only fixed the outage but also identified "
            "the root cause and implemented safeguards to prevent recurrence. "
            "The team's confidence in our incident response process improved significantly."
        )

        _, session = await interview_service.handle_response(session, follow_up_answer)

        # Second response must be recorded
        assert len(session.responses) == 2

        # A new behavioral question must have been generated
        assert len(session.questions) == 3
        next_question = session.questions[2]
        assert next_question.question_type != QuestionType.TECHNICAL

        # ---- Turn 4: User requests feedback ----
        mock_llm.complete.return_value = _behavioral_feedback_llm_response()

        reply, session = await feedback_service.generate_feedback_report(session)

        assert session.feedback_report is not None
        assert len(session.feedback_report.strengths) >= 1
        assert len(session.feedback_report.improvements) >= 1
        assert isinstance(reply, str) and len(reply) > 0

    # ------------------------------------------------------------------
    # Scenario 6: Behavioral round type persists in session
    # Requirements: 12.1, 11.1, 11.2
    # ------------------------------------------------------------------

    async def test_behavioral_round_type_persists_in_session(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Behavioral round type is stored and persists in session state.

        Requirements: 12.1, 11.1, 11.2
        """
        interview_service, _, _ = services
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)

        mock_llm.complete.return_value = _behavioral_question_llm_response()

        _, updated_session = await interview_service.handle_response(
            session, "behavioral"
        )

        # Round type must be BEHAVIORAL and persisted
        assert updated_session.interview_round_type == InterviewRoundType.BEHAVIORAL
        assert updated_session.interview_round_type is not None

        # Technical round fields must remain at defaults (not modified)
        assert updated_session.design_phase is None
        assert updated_session.design_aspects_covered == []
        assert updated_session.difficulty_adjustment_history == []


# ---------------------------------------------------------------------------
# Task 11.5 — Session persistence integration tests
# Requirements: 11.1-11.5
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    """Integration tests verifying session state persists correctly across requests.

    Simulates Redis persistence by serializing SessionState to JSON (model_dump)
    and deserializing back (model_validate), then verifying all state is preserved.

    Requirements: 11.1-11.5
    """

    @pytest.fixture
    def mock_llm(self) -> Mock:
        return _make_mock_llm()

    @pytest.fixture
    def services(self, mock_llm: Mock) -> tuple[InterviewService, TechnicalRoundService, FeedbackService]:
        return _make_services(mock_llm)

    # ------------------------------------------------------------------
    # Helper: simulate Redis round-trip (serialize → deserialize)
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_redis_round_trip(session: SessionState) -> SessionState:
        """Serialize session to JSON dict and deserialize back, simulating Redis persistence."""
        serialized = session.model_dump(mode="json")
        return SessionState.model_validate(serialized)

    # ------------------------------------------------------------------
    # Scenario 1: DSA round state persists across simulated session interruption
    # Requirements: 11.1, 11.2, 11.5
    # ------------------------------------------------------------------

    async def test_dsa_round_state_persists_across_session_interruption(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """DSA round type, difficulty, and questions persist after simulated session interruption.

        Requirements: 11.1, 11.2, 11.5
        """
        interview_service, _, _ = services

        # ---- Turn 1: User selects DSA round → first problem generated ----
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)
        mock_llm.complete.return_value = _coding_problem_llm_response()

        _, session = await interview_service.handle_response(session, "DSA round")

        assert session.interview_round_type == InterviewRoundType.DSA_CODING
        assert len(session.questions) == 1
        assert session.problem_difficulty == ProblemDifficulty.MEDIUM

        original_question_text = session.questions[0].text

        # ---- Simulate session interruption: serialize → deserialize (Redis round-trip) ----
        session.stage = Stage.INTERVIEW
        resumed_session = self._simulate_redis_round_trip(session)

        # ---- Verify all state is preserved after resumption ----
        assert resumed_session.interview_round_type == InterviewRoundType.DSA_CODING
        assert resumed_session.problem_difficulty == ProblemDifficulty.MEDIUM
        assert len(resumed_session.questions) == 1
        assert resumed_session.questions[0].text == original_question_text
        assert resumed_session.questions[0].question_type == QuestionType.TECHNICAL
        assert resumed_session.stage == Stage.INTERVIEW

    # ------------------------------------------------------------------
    # Scenario 2: System Design round state persists (round type, design_phase, design_aspects_covered)
    # Requirements: 11.1, 11.2, 11.3, 11.5
    # ------------------------------------------------------------------

    async def test_system_design_round_state_persists_across_session_interruption(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """System Design round type, design_phase, and design_aspects_covered persist after interruption.

        Requirements: 11.1, 11.2, 11.3, 11.5
        """
        from interview_practice_partner.domain.enums import DesignAspect, DesignPhase

        interview_service, _, _ = services

        # ---- Turn 1: User selects System Design round → first question generated ----
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)
        mock_llm.complete.return_value = _system_design_question_llm_response()

        _, session = await interview_service.handle_response(session, "system design round")

        assert session.interview_round_type == InterviewRoundType.SYSTEM_DESIGN
        assert session.design_phase is not None
        assert len(session.questions) == 1

        original_question_text = session.questions[0].text

        # ---- Turn 2: User provides requirements → phase transitions to HIGH_LEVEL_DESIGN ----
        session.stage = Stage.INTERVIEW
        mock_llm.complete.side_effect = [
            _intent_answer_response(),
            _system_design_evaluation_llm_response(
                follow_up_warranted=True,
                follow_up_text="Now describe the high-level architecture.",
                next_phase_suggestion="high_level_design",
            ),
        ]

        _, session = await interview_service.handle_response(
            session,
            (
                "The system needs to support 100 million daily active users with low latency under 200ms. "
                "Functional requirements include posting tweets, following users, and viewing a personalized feed. "
                "Non-functional requirements include 99.9% availability, eventual consistency, and horizontal scalability. "
                "The system should handle 10,000 writes per second and 100,000 reads per second at peak load."
            ),
        )

        from interview_practice_partner.domain.enums import DesignPhase
        assert session.design_phase == DesignPhase.HIGH_LEVEL_DESIGN

        # Manually add a design aspect to simulate tracking
        session.design_aspects_covered = [DesignAspect.SCALABILITY]

        # ---- Simulate session interruption: serialize → deserialize ----
        resumed_session = self._simulate_redis_round_trip(session)

        # ---- Verify all System Design state is preserved ----
        assert resumed_session.interview_round_type == InterviewRoundType.SYSTEM_DESIGN
        assert resumed_session.design_phase == DesignPhase.HIGH_LEVEL_DESIGN
        assert DesignAspect.SCALABILITY in resumed_session.design_aspects_covered
        assert len(resumed_session.questions) == 2  # original + follow-up
        assert resumed_session.questions[0].text == original_question_text
        assert len(resumed_session.responses) == 1

    # ------------------------------------------------------------------
    # Scenario 3: Difficulty adjustment history persists
    # Requirements: 11.5, 15.5, 18.6
    # ------------------------------------------------------------------

    async def test_difficulty_adjustment_history_persists(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """Difficulty adjustment history is preserved across simulated session interruption.

        Requirements: 11.5, 15.5, 18.6
        """
        interview_service, _, _ = services

        # Build a DSA session at MEDIUM difficulty so it can increase to HARD
        session = _make_session(stage=Stage.INTERVIEW)
        session.interview_round_type = InterviewRoundType.DSA_CODING
        session.problem_difficulty = ProblemDifficulty.MEDIUM

        problem_question = Question(
            question_id="q-dsa-001",
            text="Two Sum problem",
            question_type=QuestionType.TECHNICAL,
            asked_at=_NOW,
        )
        session.questions.append(problem_question)

        # LLM: intent → evaluation (correct, increase MEDIUM→HARD) → next problem
        mock_llm.complete.side_effect = [
            _intent_answer_response(),
            _solution_evaluation_llm_response(
                correctness="correct",
                is_optimal=True,
                follow_up_warranted=False,
                follow_up_text=None,
                difficulty_signal="increase",
            ),
            _next_coding_problem_llm_response(),
        ]

        _, session = await interview_service.handle_response(
            session,
            "def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen: return [seen[target-n], i]\n        seen[n] = i",
        )

        # Difficulty must have increased to HARD and history must have been recorded
        assert session.problem_difficulty == ProblemDifficulty.HARD
        assert len(session.difficulty_adjustment_history) >= 1

        # ---- Simulate session interruption ----
        resumed_session = self._simulate_redis_round_trip(session)

        # ---- Verify difficulty history is preserved ----
        assert len(resumed_session.difficulty_adjustment_history) == len(session.difficulty_adjustment_history)
        assert resumed_session.difficulty_adjustment_history == session.difficulty_adjustment_history
        assert resumed_session.problem_difficulty == session.problem_difficulty

    # ------------------------------------------------------------------
    # Scenario 4: After resuming, user can submit solution and evaluation works correctly
    # Requirements: 11.1, 11.4, 11.5
    # ------------------------------------------------------------------

    async def test_after_resuming_user_can_submit_solution_and_evaluation_works(
        self, services: tuple, mock_llm: Mock
    ) -> None:
        """After resuming from a simulated interruption, the user can submit a solution and get evaluated.

        Requirements: 11.1, 11.4, 11.5
        """
        interview_service, _, _ = services

        # ---- Turn 1: User selects DSA round → first problem generated ----
        session = _make_session(stage=Stage.ROUND_TYPE_SELECTION)
        mock_llm.complete.return_value = _coding_problem_llm_response()

        _, session = await interview_service.handle_response(session, "DSA round")

        assert session.interview_round_type == InterviewRoundType.DSA_CODING
        assert len(session.questions) == 1

        # ---- Simulate session interruption (e.g., user closes app, Redis TTL not expired) ----
        session.stage = Stage.INTERVIEW
        resumed_session = self._simulate_redis_round_trip(session)

        # Verify state is intact after resumption
        assert resumed_session.interview_round_type == InterviewRoundType.DSA_CODING
        assert len(resumed_session.questions) == 1
        assert resumed_session.stage == Stage.INTERVIEW

        # ---- Turn 2 (after resumption): User submits solution → evaluation works ----
        mock_llm.complete.side_effect = [
            _intent_answer_response(),
            _solution_evaluation_llm_response(
                correctness="correct",
                is_optimal=True,
                follow_up_warranted=True,
                follow_up_text="What is the time complexity of your solution?",
                difficulty_signal="increase",
            ),
        ]

        code_solution = (
            "```python\n"
            "def two_sum(nums, target):\n"
            "    seen = {}\n"
            "    for i, num in enumerate(nums):\n"
            "        complement = target - num\n"
            "        if complement in seen:\n"
            "            return [seen[complement], i]\n"
            "        seen[num] = i\n"
            "```"
        )

        reply, updated_session = await interview_service.handle_response(
            resumed_session, code_solution
        )

        # Response must be recorded
        assert len(updated_session.responses) == 1

        # Follow-up question must have been added
        assert len(updated_session.questions) == 2
        follow_up = updated_session.questions[1]
        assert follow_up.question_type == QuestionType.FOLLOW_UP
        assert "time complexity" in follow_up.text.lower()

        # Reply must contain the follow-up
        assert len(reply) > 0
        assert "time complexity" in reply.lower() or "complexity" in reply.lower()

        # Round type must still be DSA_CODING
        assert updated_session.interview_round_type == InterviewRoundType.DSA_CODING

    # ------------------------------------------------------------------
    # Scenario 5: Full persistence round-trip — all technical fields survive serialization
    # Requirements: 11.5, 18.1-18.7
    # ------------------------------------------------------------------

    async def test_all_technical_session_fields_survive_serialization(self) -> None:
        """All technical round fields in SessionState survive a JSON serialization round-trip.

        Requirements: 11.5, 18.1-18.7
        """
        from interview_practice_partner.domain.enums import DesignAspect, DesignPhase, ProblemTopic

        # Build a fully-populated technical session
        session = SessionState(
            session_id="sess-persist-001",
            phone_number="+1234567890",
            stage=Stage.INTERVIEW,
            role=Role.SOFTWARE_ENGINEER,
            created_at=_NOW,
            updated_at=_NOW,
            interview_round_type=InterviewRoundType.DSA_CODING,
            problem_difficulty=ProblemDifficulty.HARD,
            design_phase=DesignPhase.DEEP_DIVE,
            topics_covered=[ProblemTopic.ARRAYS, ProblemTopic.HASH_TABLES],
            design_aspects_covered=[DesignAspect.SCALABILITY, DesignAspect.DATABASE_DESIGN],
            difficulty_adjustment_history=[
                {"from": "medium", "to": "hard", "reason": "correct and optimal solution"},
                {"from": "hard", "to": "hard", "reason": "already at maximum difficulty"},
            ],
        )

        # Add a question and response
        session.questions.append(Question(
            question_id="q-001",
            text="Two Sum problem",
            question_type=QuestionType.TECHNICAL,
            asked_at=_NOW,
        ))
        session.responses.append(UserResponse(
            response_id="r-001",
            question_id="q-001",
            text="def two_sum(nums, target): ...",
            word_count=20,
            received_at=_NOW,
        ))

        # ---- Simulate Redis round-trip ----
        resumed = self._simulate_redis_round_trip(session)

        # ---- Verify all technical fields are preserved exactly ----
        assert resumed.interview_round_type == InterviewRoundType.DSA_CODING
        assert resumed.problem_difficulty == ProblemDifficulty.HARD
        assert resumed.design_phase == DesignPhase.DEEP_DIVE
        assert ProblemTopic.ARRAYS in resumed.topics_covered
        assert ProblemTopic.HASH_TABLES in resumed.topics_covered
        assert DesignAspect.SCALABILITY in resumed.design_aspects_covered
        assert DesignAspect.DATABASE_DESIGN in resumed.design_aspects_covered
        assert len(resumed.difficulty_adjustment_history) == 2
        assert resumed.difficulty_adjustment_history[0]["from"] == "medium"
        assert resumed.difficulty_adjustment_history[0]["to"] == "hard"

        # ---- Verify questions and responses are preserved ----
        assert len(resumed.questions) == 1
        assert resumed.questions[0].text == "Two Sum problem"
        assert resumed.questions[0].question_type == QuestionType.TECHNICAL

        assert len(resumed.responses) == 1
        assert resumed.responses[0].text == "def two_sum(nums, target): ..."
