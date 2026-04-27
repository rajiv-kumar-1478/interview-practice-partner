"""TechnicalRoundService — handles DSA/Coding and System Design interview rounds.

This service is responsible for:
- Generating DSA/Coding problems with appropriate difficulty
- Generating System Design questions
- Evaluating technical solutions (correctness, complexity, edge cases)
- Managing adaptive difficulty adjustments
- Guiding users through system design phases
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog

from interview_practice_partner.domain.enums import (
    DesignPhase,
    ProblemDifficulty,
    ProblemTopic,
    SolutionFormat,
)
from interview_practice_partner.domain.models import (
    CodingProblem,
    ComplexityAnalysis,
    SessionState,
    SystemDesignQuestion,
    TechnicalEvaluation,
    UserResponse,
)
from interview_practice_partner.llm.client import LLMClient
from interview_practice_partner.llm.prompt_builder import PromptBuilder
from interview_practice_partner.services import code_parser

logger = structlog.get_logger(__name__)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class TechnicalRoundService:
    """Handles DSA/Coding and System Design interview rounds.

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
    ) -> None:
        self._llm = llm_client
        self._prompt_builder = prompt_builder

    # ------------------------------------------------------------------
    # DSA/Coding Round: Problem Generation
    # ------------------------------------------------------------------

    async def generate_coding_problem(
        self,
        session: SessionState,
        difficulty: ProblemDifficulty,
        topic: Optional[ProblemTopic] = None,
    ) -> CodingProblem:
        """Generate a coding problem with specified difficulty and optional topic.

        Calls the LLM with the coding problem generation prompt and parses the
        JSON response into a ``CodingProblem`` model. Ensures the problem is
        distinct from previously asked problems in the session.

        Args:
            session: The current ``SessionState``.
            difficulty: The desired problem difficulty (EASY, MEDIUM, HARD).
            topic: Optional specific topic to focus on (e.g., ARRAYS, TREES).

        Returns:
            A ``CodingProblem`` instance with problem statement, examples,
            constraints, and topic.
        """
        log = logger.bind(
            session_id=session.session_id,
            difficulty=difficulty.value,
            topic=topic.value if topic else None,
        )
        log.info("generating_coding_problem")

        # Get previously asked problems from session
        problems_asked = [
            CodingProblem(
                problem_id=q.question_id,
                text=q.text,
                difficulty=session.problem_difficulty,
                topic=ProblemTopic.ARRAYS,  # Default, actual topic stored separately
                constraints="",
                examples=[],
                asked_at=q.asked_at,
            )
            for q in session.questions
            if not q.skipped
        ]

        messages = self._prompt_builder.build_coding_problem_generation_prompt(
            session=session,
            difficulty=difficulty,
            topic=topic,
            problems_asked=problems_asked,
        )

        try:
            raw = await self._llm.complete(messages, temperature=0.8, max_tokens=1024)
            problem = self._parse_coding_problem_response(raw, difficulty)
            log.info(
                "coding_problem_generated",
                problem_id=problem.problem_id,
                topic=problem.topic.value,
            )
            return problem
        except Exception as exc:  # noqa: BLE001
            log.warning("coding_problem_generation_failed", error=str(exc))
            # Fallback to a safe default problem
            return self._build_fallback_coding_problem(difficulty, topic)

    def _parse_coding_problem_response(
        self,
        raw: str,
        difficulty: ProblemDifficulty,
    ) -> CodingProblem:
        """Parse the LLM JSON response into a ``CodingProblem``.

        Falls back to a minimal valid problem if JSON parsing fails.

        Args:
            raw: The raw LLM response string (expected to be JSON).
            difficulty: The requested difficulty level.

        Returns:
            A ``CodingProblem`` instance.
        """
        try:
            data = json.loads(raw)
            
            # Parse topic string to enum
            topic_str = data.get("topic", "arrays").lower()
            topic = self._parse_topic(topic_str)

            problem = CodingProblem(
                problem_id=str(uuid.uuid4()),
                text=data.get("problem_statement", ""),
                difficulty=difficulty,
                topic=topic,
                constraints=data.get("constraints", ""),
                examples=data.get("examples", []),
                asked_at=_now(),
            )
            return problem

        except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "coding_problem_json_parse_failed",
                error=str(exc),
                raw_response=raw[:200],
            )
            return self._build_fallback_coding_problem(difficulty, None)

    def _parse_topic(self, topic_str: str) -> ProblemTopic:
        """Parse a topic string to a ``ProblemTopic`` enum.

        Args:
            topic_str: The topic string from the LLM response.

        Returns:
            A ``ProblemTopic`` enum value, defaulting to ARRAYS if unrecognized.
        """
        topic_map = {
            "arrays": ProblemTopic.ARRAYS,
            "strings": ProblemTopic.STRINGS,
            "linked_lists": ProblemTopic.LINKED_LISTS,
            "trees": ProblemTopic.TREES,
            "graphs": ProblemTopic.GRAPHS,
            "dynamic_programming": ProblemTopic.DYNAMIC_PROGRAMMING,
            "sorting": ProblemTopic.SORTING,
            "searching": ProblemTopic.SEARCHING,
            "hash_tables": ProblemTopic.HASH_TABLES,
            "stacks_queues": ProblemTopic.STACKS_QUEUES,
        }
        return topic_map.get(topic_str.lower(), ProblemTopic.ARRAYS)

    def _build_fallback_coding_problem(
        self,
        difficulty: ProblemDifficulty,
        topic: Optional[ProblemTopic],
    ) -> CodingProblem:
        """Build a minimal valid ``CodingProblem`` as a fallback.

        Used when the LLM returns invalid JSON or the call fails entirely.

        Args:
            difficulty: The requested difficulty level.
            topic: The requested topic (or None).

        Returns:
            A minimal ``CodingProblem`` with safe defaults.
        """
        fallback_problems = {
            ProblemDifficulty.EASY: {
                "text": "Given an array of integers, return the indices of the two numbers that add up to a specific target.",
                "examples": [
                    "Input: nums = [2,7,11,15], target = 9\nOutput: [0,1]",
                    "Input: nums = [3,2,4], target = 6\nOutput: [1,2]",
                ],
                "constraints": "2 <= nums.length <= 10^4, -10^9 <= nums[i] <= 10^9",
            },
            ProblemDifficulty.MEDIUM: {
                "text": "Given a string, find the length of the longest substring without repeating characters.",
                "examples": [
                    'Input: s = "abcabcbb"\nOutput: 3',
                    'Input: s = "bbbbb"\nOutput: 1',
                ],
                "constraints": "0 <= s.length <= 5 * 10^4",
            },
            ProblemDifficulty.HARD: {
                "text": "Given two sorted arrays, find the median of the two sorted arrays.",
                "examples": [
                    "Input: nums1 = [1,3], nums2 = [2]\nOutput: 2.0",
                    "Input: nums1 = [1,2], nums2 = [3,4]\nOutput: 2.5",
                ],
                "constraints": "nums1.length + nums2.length >= 1",
            },
        }

        fallback = fallback_problems.get(difficulty, fallback_problems[ProblemDifficulty.MEDIUM])
        
        return CodingProblem(
            problem_id=str(uuid.uuid4()),
            text=fallback["text"],
            difficulty=difficulty,
            topic=topic or ProblemTopic.ARRAYS,
            constraints=fallback["constraints"],
            examples=fallback["examples"],
            asked_at=_now(),
        )

    # ------------------------------------------------------------------
    # DSA/Coding Round: Solution Evaluation
    # ------------------------------------------------------------------

    async def evaluate_coding_solution(
        self,
        session: SessionState,
        problem: CodingProblem,
        response: UserResponse,
    ) -> TechnicalEvaluation:
        """Evaluate a coding solution across multiple dimensions.

        Detects the solution format (code, pseudocode, or explanation), calls
        the LLM with the solution evaluation prompt, and parses the JSON
        response into a ``TechnicalEvaluation`` model.

        Args:
            session: The current ``SessionState``.
            problem: The ``CodingProblem`` that was asked.
            response: The ``UserResponse`` containing the solution.

        Returns:
            A ``TechnicalEvaluation`` instance with correctness, complexity
            analysis, edge cases, code quality, and follow-up information.
        """
        log = logger.bind(
            session_id=session.session_id,
            problem_id=problem.problem_id,
            response_id=response.response_id,
        )
        log.info("evaluating_coding_solution")

        # Detect solution format using code_parser
        solution_format = code_parser.parse_solution_format(response.text)
        log.info("solution_format_detected", format=solution_format.value)

        messages = self._prompt_builder.build_coding_solution_evaluation_prompt(
            problem=problem,
            response=response,
            solution_format=solution_format,
        )

        try:
            raw = await self._llm.complete(messages, temperature=0.3, max_tokens=1024)
            evaluation = self._parse_coding_evaluation_response(
                raw, problem, response, solution_format
            )
            log.info(
                "coding_solution_evaluated",
                correctness=evaluation.correctness,
                difficulty_signal=evaluation.difficulty_signal,
            )
            return evaluation
        except Exception as exc:  # noqa: BLE001
            log.warning("coding_solution_evaluation_failed", error=str(exc))
            # Fallback to a safe default evaluation
            return self._build_fallback_coding_evaluation(
                problem, response, solution_format
            )

    def _parse_coding_evaluation_response(
        self,
        raw: str,
        problem: CodingProblem,
        response: UserResponse,
        solution_format: SolutionFormat,
    ) -> TechnicalEvaluation:
        """Parse the LLM JSON response into a ``TechnicalEvaluation``.

        Falls back to safe defaults if JSON parsing fails.

        Args:
            raw: The raw LLM response string (expected to be JSON).
            problem: The coding problem being evaluated.
            response: The user's response.
            solution_format: The detected solution format.

        Returns:
            A ``TechnicalEvaluation`` instance.
        """
        try:
            data = json.loads(raw)

            # Parse complexity analysis
            complexity_analysis = ComplexityAnalysis(
                time_complexity=data.get("time_complexity", "O(?)"),
                space_complexity=data.get("space_complexity", "O(?)"),
                is_optimal=bool(data.get("is_optimal", False)),
                optimization_suggestions=data.get("optimization_suggestions"),
            )

            evaluation = TechnicalEvaluation(
                evaluation_id=str(uuid.uuid4()),
                question_id=problem.problem_id,
                response_id=response.response_id,
                correctness=data.get("correctness", "partial"),
                complexity_analysis=complexity_analysis,
                edge_cases_handled=data.get("edge_cases_handled", []),
                edge_cases_missed=data.get("edge_cases_missed", []),
                code_quality_notes=data.get("code_quality_notes"),
                solution_format=solution_format,
                follow_up_warranted=bool(data.get("follow_up_warranted", False)),
                follow_up_text=data.get("follow_up_text"),
                difficulty_signal=data.get("difficulty_signal", "maintain"),
                evaluated_at=_now(),
            )
            return evaluation

        except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "coding_evaluation_json_parse_failed",
                error=str(exc),
                raw_response=raw[:200],
            )
            return self._build_fallback_coding_evaluation(
                problem, response, solution_format
            )

    def _build_fallback_coding_evaluation(
        self,
        problem: CodingProblem,
        response: UserResponse,
        solution_format: SolutionFormat,
    ) -> TechnicalEvaluation:
        """Build a minimal valid ``TechnicalEvaluation`` as a fallback.

        Used when the LLM returns invalid JSON or the call fails entirely.

        Args:
            problem: The coding problem being evaluated.
            response: The user's response.
            solution_format: The detected solution format.

        Returns:
            A minimal ``TechnicalEvaluation`` with safe defaults.
        """
        complexity_analysis = ComplexityAnalysis(
            time_complexity="O(?)",
            space_complexity="O(?)",
            is_optimal=False,
            optimization_suggestions=None,
        )

        return TechnicalEvaluation(
            evaluation_id=str(uuid.uuid4()),
            question_id=problem.problem_id,
            response_id=response.response_id,
            correctness="partial",
            complexity_analysis=complexity_analysis,
            edge_cases_handled=[],
            edge_cases_missed=[],
            code_quality_notes="Unable to evaluate — evaluation encountered an error.",
            solution_format=solution_format,
            follow_up_warranted=False,
            follow_up_text=None,
            difficulty_signal="maintain",
            evaluated_at=_now(),
        )

    # ------------------------------------------------------------------
    # DSA/Coding Round: Solution Format Detection
    # ------------------------------------------------------------------

    def parse_solution_format(self, response_text: str) -> SolutionFormat:
        """Determine if response is code, pseudocode, or explanation.

        Uses heuristics to classify the solution format:
        - CODE: Contains code blocks (markdown fences or indentation patterns)
        - PSEUDOCODE: Contains algorithmic structure but lacks syntax
        - EXPLANATION: Conversational description without code structure

        Args:
            response_text: The user's response text.

        Returns:
            A ``SolutionFormat`` enum value.
        """
        text = response_text.strip().lower()

        # Check for markdown code fences
        if "```" in text:
            return SolutionFormat.CODE

        # Check for common programming keywords and syntax
        code_indicators = [
            "def ", "function ", "class ", "return ", "if ", "for ", "while ",
            "int ", "string ", "void ", "public ", "private ", "const ",
            "{", "}", "[", "]", "=>", "->", "==", "!=", "<=", ">=",
        ]
        code_indicator_count = sum(1 for indicator in code_indicators if indicator in text)

        # Check for pseudocode indicators
        pseudocode_indicators = [
            "step 1", "step 2", "algorithm", "procedure", "begin", "end",
            "initialize", "iterate", "loop", "set ", "get ",
        ]
        pseudocode_indicator_count = sum(
            1 for indicator in pseudocode_indicators if indicator in text
        )

        # Classification logic
        if code_indicator_count >= 3:
            return SolutionFormat.CODE
        elif pseudocode_indicator_count >= 2:
            return SolutionFormat.PSEUDOCODE
        elif code_indicator_count >= 1:
            # Some code-like structure but not enough for full code
            return SolutionFormat.PSEUDOCODE
        else:
            return SolutionFormat.EXPLANATION

    # ------------------------------------------------------------------
    # DSA/Coding Round: Difficulty Adjustment
    # ------------------------------------------------------------------

    def adjust_difficulty(
        self,
        session: SessionState,
        evaluation: TechnicalEvaluation,
    ) -> ProblemDifficulty:
        """Determine next problem difficulty based on performance.

        Increases difficulty on correct + optimal solutions, decreases on
        incorrect or skipped problems, and maintains otherwise. Respects
        difficulty boundaries (EASY/HARD).

        Args:
            session: The current ``SessionState``.
            evaluation: The ``TechnicalEvaluation`` for the last solution.

        Returns:
            The ``ProblemDifficulty`` for the next problem.
        """
        current_difficulty = session.problem_difficulty
        signal = evaluation.difficulty_signal

        log = logger.bind(
            session_id=session.session_id,
            current_difficulty=current_difficulty.value,
            signal=signal,
        )

        # Determine new difficulty
        if signal == "increase":
            if current_difficulty == ProblemDifficulty.EASY:
                new_difficulty = ProblemDifficulty.MEDIUM
            elif current_difficulty == ProblemDifficulty.MEDIUM:
                new_difficulty = ProblemDifficulty.HARD
            else:  # Already HARD
                new_difficulty = ProblemDifficulty.HARD
        elif signal == "decrease":
            if current_difficulty == ProblemDifficulty.HARD:
                new_difficulty = ProblemDifficulty.MEDIUM
            elif current_difficulty == ProblemDifficulty.MEDIUM:
                new_difficulty = ProblemDifficulty.EASY
            else:  # Already EASY
                new_difficulty = ProblemDifficulty.EASY
        else:  # maintain
            new_difficulty = current_difficulty

        # Record adjustment in history
        if new_difficulty != current_difficulty:
            adjustment_record = {
                "from": current_difficulty.value,
                "to": new_difficulty.value,
                "reason": signal,
                "timestamp": _now().isoformat(),
            }
            session.difficulty_adjustment_history.append(adjustment_record)
            log.info(
                "difficulty_adjusted",
                new_difficulty=new_difficulty.value,
                reason=signal,
            )
        else:
            log.info("difficulty_maintained", difficulty=current_difficulty.value)

        return new_difficulty

    # ------------------------------------------------------------------
    # System Design Round: Question Generation
    # ------------------------------------------------------------------

    async def generate_system_design_question(
        self,
        session: SessionState,
    ) -> SystemDesignQuestion:
        """Generate a system design question.

        Calls the LLM with the system design question generation prompt and
        parses the JSON response into a ``SystemDesignQuestion`` model. Ensures
        the question is distinct from previously asked questions in the session.
        Initializes the design phase to REQUIREMENTS_GATHERING.

        Args:
            session: The current ``SessionState``.

        Returns:
            A ``SystemDesignQuestion`` instance with system name, question text,
            and description.
        """
        log = logger.bind(session_id=session.session_id)
        log.info("generating_system_design_question")

        # Initialize design phase to REQUIREMENTS_GATHERING
        session.design_phase = DesignPhase.REQUIREMENTS_GATHERING

        # Get previously asked questions from session
        questions_asked = [
            SystemDesignQuestion(
                question_id=q.question_id,
                text=q.text,
                system_name="Unknown",  # Would need to be stored separately
                description="",
                asked_at=q.asked_at,
            )
            for q in session.questions
            if not q.skipped
        ]

        messages = self._prompt_builder.build_system_design_question_generation_prompt(
            session=session,
            questions_asked=questions_asked,
        )

        try:
            raw = await self._llm.complete(messages, temperature=0.8, max_tokens=1024)
            question = self._parse_system_design_question_response(raw)
            log.info(
                "system_design_question_generated",
                question_id=question.question_id,
                system_name=question.system_name,
            )
            return question
        except Exception as exc:  # noqa: BLE001
            log.warning("system_design_question_generation_failed", error=str(exc))
            # Fallback to a safe default question
            return self._build_fallback_system_design_question()

    def _parse_system_design_question_response(
        self,
        raw: str,
    ) -> SystemDesignQuestion:
        """Parse the LLM JSON response into a ``SystemDesignQuestion``.

        Falls back to a minimal valid question if JSON parsing fails.

        Args:
            raw: The raw LLM response string (expected to be JSON).

        Returns:
            A ``SystemDesignQuestion`` instance.
        """
        try:
            data = json.loads(raw)

            question = SystemDesignQuestion(
                question_id=str(uuid.uuid4()),
                text=data.get("question_text", ""),
                system_name=data.get("system_name", "Unknown System"),
                description=data.get("description", ""),
                asked_at=_now(),
            )
            return question

        except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "system_design_question_json_parse_failed",
                error=str(exc),
                raw_response=raw[:200],
            )
            return self._build_fallback_system_design_question()

    def _build_fallback_system_design_question(self) -> SystemDesignQuestion:
        """Build a minimal valid ``SystemDesignQuestion`` as a fallback.

        Used when the LLM returns invalid JSON or the call fails entirely.

        Returns:
            A minimal ``SystemDesignQuestion`` with safe defaults.
        """
        return SystemDesignQuestion(
            question_id=str(uuid.uuid4()),
            text="Design a URL shortener service like bit.ly",
            system_name="URL Shortener",
            description=(
                "Design a service that takes long URLs and generates short, "
                "unique aliases that redirect to the original URL."
            ),
            asked_at=_now(),
        )

    # ------------------------------------------------------------------
    # System Design Round: Evaluation
    # ------------------------------------------------------------------

    async def evaluate_system_design(
        self,
        session: SessionState,
        question: SystemDesignQuestion,
        response: UserResponse,
    ) -> TechnicalEvaluation:
        """Evaluate a system design response.

        Calls the LLM with the system design evaluation prompt (phase-aware)
        and parses the JSON response into a ``TechnicalEvaluation`` model.

        Args:
            session: The current ``SessionState``.
            question: The ``SystemDesignQuestion`` that was asked.
            response: The ``UserResponse`` containing the design explanation.

        Returns:
            A ``TechnicalEvaluation`` instance with design aspect evaluations,
            strengths, weaknesses, and follow-up information.
        """
        log = logger.bind(
            session_id=session.session_id,
            question_id=question.question_id,
            response_id=response.response_id,
            current_phase=session.design_phase.value if session.design_phase else None,
        )
        log.info("evaluating_system_design")

        current_phase = session.design_phase or DesignPhase.REQUIREMENTS_GATHERING

        messages = self._prompt_builder.build_system_design_evaluation_prompt(
            question=question,
            response=response,
            current_phase=current_phase,
        )

        try:
            raw = await self._llm.complete(messages, temperature=0.3, max_tokens=1024)
            evaluation = self._parse_system_design_evaluation_response(
                raw, question, response
            )
            log.info(
                "system_design_evaluated",
                aspects_count=len(evaluation.design_aspects_evaluated),
                follow_up_warranted=evaluation.follow_up_warranted,
            )
            return evaluation
        except Exception as exc:  # noqa: BLE001
            log.warning("system_design_evaluation_failed", error=str(exc))
            # Fallback to a safe default evaluation
            return self._build_fallback_system_design_evaluation(question, response)

    def _parse_system_design_evaluation_response(
        self,
        raw: str,
        question: SystemDesignQuestion,
        response: UserResponse,
    ) -> TechnicalEvaluation:
        """Parse the LLM JSON response into a ``TechnicalEvaluation``.

        Falls back to safe defaults if JSON parsing fails.

        Args:
            raw: The raw LLM response string (expected to be JSON).
            question: The system design question being evaluated.
            response: The user's response.

        Returns:
            A ``TechnicalEvaluation`` instance.
        """
        try:
            data = json.loads(raw)

            evaluation = TechnicalEvaluation(
                evaluation_id=str(uuid.uuid4()),
                question_id=question.question_id,
                response_id=response.response_id,
                design_aspects_evaluated=data.get("design_aspects_evaluated", {}),
                design_strengths=data.get("design_strengths", []),
                design_weaknesses=data.get("design_weaknesses", []),
                follow_up_warranted=bool(data.get("follow_up_warranted", False)),
                follow_up_text=data.get("follow_up_text"),
                difficulty_signal="maintain",  # Not used for system design
                evaluated_at=_now(),
            )
            return evaluation

        except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "system_design_evaluation_json_parse_failed",
                error=str(exc),
                raw_response=raw[:200],
            )
            return self._build_fallback_system_design_evaluation(question, response)

    def _build_fallback_system_design_evaluation(
        self,
        question: SystemDesignQuestion,
        response: UserResponse,
    ) -> TechnicalEvaluation:
        """Build a minimal valid ``TechnicalEvaluation`` as a fallback.

        Used when the LLM returns invalid JSON or the call fails entirely.

        Args:
            question: The system design question being evaluated.
            response: The user's response.

        Returns:
            A minimal ``TechnicalEvaluation`` with safe defaults.
        """
        return TechnicalEvaluation(
            evaluation_id=str(uuid.uuid4()),
            question_id=question.question_id,
            response_id=response.response_id,
            design_aspects_evaluated={},
            design_strengths=["You provided a response to the design question."],
            design_weaknesses=[],
            follow_up_warranted=False,
            follow_up_text=None,
            difficulty_signal="maintain",
            evaluated_at=_now(),
        )

    # ------------------------------------------------------------------
    # System Design Round: Phase Progression
    # ------------------------------------------------------------------

    def determine_next_design_phase(
        self,
        session: SessionState,
        response: UserResponse,
    ) -> DesignPhase:
        """Determine next system design phase based on conversation flow.

        Uses the current phase and allows natural progression through:
        REQUIREMENTS_GATHERING → HIGH_LEVEL_DESIGN → DEEP_DIVE → BOTTLENECK_ANALYSIS

        Args:
            session: The current ``SessionState``.
            response: The user's latest response.

        Returns:
            The next ``DesignPhase`` to guide the user through.
        """
        current_phase = session.design_phase or DesignPhase.REQUIREMENTS_GATHERING

        # Simple linear progression for now
        # In a more sophisticated implementation, this could use LLM evaluation
        # to determine if the user is ready to move to the next phase
        phase_order = [
            DesignPhase.REQUIREMENTS_GATHERING,
            DesignPhase.HIGH_LEVEL_DESIGN,
            DesignPhase.DEEP_DIVE,
            DesignPhase.BOTTLENECK_ANALYSIS,
        ]

        try:
            current_index = phase_order.index(current_phase)
            if current_index < len(phase_order) - 1:
                next_phase = phase_order[current_index + 1]
            else:
                next_phase = current_phase  # Stay at final phase
        except ValueError:
            # Current phase not in order, default to first phase
            next_phase = DesignPhase.REQUIREMENTS_GATHERING

        if next_phase != current_phase:
            logger.info(
                "design_phase_transition",
                session_id=session.session_id,
                from_phase=current_phase.value,
                to_phase=next_phase.value,
            )

        return next_phase
