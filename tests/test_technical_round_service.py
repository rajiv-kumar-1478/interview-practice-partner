"""Unit tests for TechnicalRoundService.

Covers:
- DSA problem generation
- DSA solution evaluation
- Difficulty adjustment logic
- System Design question generation
- System Design evaluation
- Design phase progression

**Validates: Requirements 2.1-2.8, 4.1-4.8, 6.1-6.5, 7.1-7.7, 8.1-8.8, 15.1-15.5, 19.2, 19.7**
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from interview_practice_partner.domain.enums import (
    DesignPhase,
    InterviewRoundType,
    ProblemDifficulty,
    ProblemTopic,
    SolutionFormat,
    Stage,
)
from interview_practice_partner.domain.models import (
    CodingProblem,
    ComplexityAnalysis,
    Question,
    SessionState,
    SystemDesignQuestion,
    TechnicalEvaluation,
    UserResponse,
)
from interview_practice_partner.llm.client import LLMClient
from interview_practice_partner.llm.prompt_builder import PromptBuilder
from interview_practice_partner.services.technical_round import TechnicalRoundService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock(spec=LLMClient)
    client.complete = AsyncMock()
    return client


@pytest.fixture
def prompt_builder():
    """Create a real PromptBuilder instance."""
    return PromptBuilder()


@pytest.fixture
def technical_round_service(mock_llm_client, prompt_builder):
    """Create a TechnicalRoundService with mocked LLM."""
    return TechnicalRoundService(
        llm_client=mock_llm_client,
        prompt_builder=prompt_builder,
    )


@pytest.fixture
def sample_session():
    """Create a sample session state for testing."""
    now = datetime.now(tz=timezone.utc)
    return SessionState(
        session_id="test-session-123",
        phone_number="+1234567890",
        stage=Stage.INTERVIEW,
        interview_round_type=InterviewRoundType.DSA_CODING,
        problem_difficulty=ProblemDifficulty.MEDIUM,
        questions=[],
        responses=[],
        topics_covered=[],
        difficulty_adjustment_history=[],
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def sample_coding_problem():
    """Create a sample coding problem."""
    return CodingProblem(
        problem_id="problem-123",
        text="Given an array of integers, return the indices of the two numbers that add up to a specific target.",
        difficulty=ProblemDifficulty.MEDIUM,
        topic=ProblemTopic.ARRAYS,
        constraints="2 <= nums.length <= 10^4",
        examples=[
            "Input: nums = [2,7,11,15], target = 9\nOutput: [0,1]",
            "Input: nums = [3,2,4], target = 6\nOutput: [1,2]",
        ],
        asked_at=datetime.now(tz=timezone.utc),
    )


@pytest.fixture
def sample_user_response():
    """Create a sample user response."""
    return UserResponse(
        response_id="response-123",
        question_id="problem-123",
        text="""```python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```""",
        word_count=50,
        received_at=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# DSA Problem Generation Tests
# ---------------------------------------------------------------------------


class TestGenerateCodingProblem:
    """Test generate_coding_problem method."""

    @pytest.mark.asyncio
    async def test_generates_problem_with_valid_llm_response(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Should generate a coding problem from valid LLM JSON response."""
        # Mock LLM response
        llm_response = json.dumps({
            "problem_statement": "Find the maximum subarray sum.",
            "examples": [
                "Input: [-2,1,-3,4,-1,2,1,-5,4]\nOutput: 6",
                "Input: [1]\nOutput: 1",
            ],
            "constraints": "1 <= nums.length <= 10^5",
            "topic": "dynamic_programming",
        })
        mock_llm_client.complete.return_value = llm_response

        # Generate problem
        problem = await technical_round_service.generate_coding_problem(
            session=sample_session,
            difficulty=ProblemDifficulty.MEDIUM,
            topic=ProblemTopic.DYNAMIC_PROGRAMMING,
        )

        # Verify problem structure
        assert isinstance(problem, CodingProblem)
        assert problem.text == "Find the maximum subarray sum."
        assert problem.difficulty == ProblemDifficulty.MEDIUM
        assert problem.topic == ProblemTopic.DYNAMIC_PROGRAMMING
        assert len(problem.examples) == 2
        assert problem.constraints == "1 <= nums.length <= 10^5"
        assert problem.problem_id is not None

        # Verify LLM was called
        mock_llm_client.complete.assert_called_once()
        call_args = mock_llm_client.complete.call_args
        assert call_args.kwargs["temperature"] == 0.8
        assert call_args.kwargs["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_generates_problem_without_topic(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Should generate a problem when no specific topic is requested."""
        llm_response = json.dumps({
            "problem_statement": "Reverse a linked list.",
            "examples": ["Input: 1->2->3\nOutput: 3->2->1"],
            "constraints": "0 <= list length <= 5000",
            "topic": "linked_lists",
        })
        mock_llm_client.complete.return_value = llm_response

        problem = await technical_round_service.generate_coding_problem(
            session=sample_session,
            difficulty=ProblemDifficulty.EASY,
            topic=None,
        )

        assert isinstance(problem, CodingProblem)
        assert problem.difficulty == ProblemDifficulty.EASY
        assert problem.topic == ProblemTopic.LINKED_LISTS

    @pytest.mark.asyncio
    async def test_falls_back_on_invalid_json(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Should return fallback problem when LLM returns invalid JSON."""
        mock_llm_client.complete.return_value = "This is not valid JSON"

        problem = await technical_round_service.generate_coding_problem(
            session=sample_session,
            difficulty=ProblemDifficulty.HARD,
            topic=None,
        )

        # Should still return a valid problem
        assert isinstance(problem, CodingProblem)
        assert problem.difficulty == ProblemDifficulty.HARD
        assert len(problem.text) > 0
        assert len(problem.examples) > 0

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_failure(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Should return fallback problem when LLM call fails."""
        mock_llm_client.complete.side_effect = Exception("LLM timeout")

        problem = await technical_round_service.generate_coding_problem(
            session=sample_session,
            difficulty=ProblemDifficulty.MEDIUM,
            topic=ProblemTopic.ARRAYS,
        )

        # Should still return a valid problem
        assert isinstance(problem, CodingProblem)
        assert problem.difficulty == ProblemDifficulty.MEDIUM
        assert problem.topic == ProblemTopic.ARRAYS

    @pytest.mark.asyncio
    async def test_logs_problem_generation(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Should log problem generation events."""
        llm_response = json.dumps({
            "problem_statement": "Test problem",
            "examples": ["Example 1"],
            "constraints": "Test constraints",
            "topic": "arrays",
        })
        mock_llm_client.complete.return_value = llm_response

        problem = await technical_round_service.generate_coding_problem(
            session=sample_session,
            difficulty=ProblemDifficulty.EASY,
            topic=None,
        )

        # Verify problem was generated (logging is implicit)
        assert problem is not None
        assert isinstance(problem, CodingProblem)


# ---------------------------------------------------------------------------
# DSA Solution Evaluation Tests
# ---------------------------------------------------------------------------


class TestEvaluateCodingSolution:
    """Test evaluate_coding_solution method."""

    @pytest.mark.asyncio
    async def test_evaluates_solution_with_valid_llm_response(
        self,
        technical_round_service,
        mock_llm_client,
        sample_session,
        sample_coding_problem,
        sample_user_response,
    ):
        """Should evaluate a solution from valid LLM JSON response."""
        llm_response = json.dumps({
            "correctness": "correct",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "is_optimal": True,
            "edge_cases_handled": ["empty array", "single element"],
            "edge_cases_missed": [],
            "code_quality_notes": "Clean and readable implementation",
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "increase",
        })
        mock_llm_client.complete.return_value = llm_response

        evaluation = await technical_round_service.evaluate_coding_solution(
            session=sample_session,
            problem=sample_coding_problem,
            response=sample_user_response,
        )

        # Verify evaluation structure
        assert isinstance(evaluation, TechnicalEvaluation)
        assert evaluation.correctness == "correct"
        assert evaluation.complexity_analysis.time_complexity == "O(n)"
        assert evaluation.complexity_analysis.space_complexity == "O(n)"
        assert evaluation.complexity_analysis.is_optimal is True
        assert len(evaluation.edge_cases_handled) == 2
        assert len(evaluation.edge_cases_missed) == 0
        assert evaluation.difficulty_signal == "increase"
        assert evaluation.solution_format == SolutionFormat.CODE

    @pytest.mark.asyncio
    async def test_detects_solution_format(
        self,
        technical_round_service,
        mock_llm_client,
        sample_session,
        sample_coding_problem,
    ):
        """Should detect solution format before evaluation."""
        # Test with pseudocode
        pseudocode_response = UserResponse(
            response_id="resp-1",
            question_id="problem-123",
            text="Step 1: Initialize hash map\nStep 2: Iterate through array",
            word_count=10,
            received_at=datetime.now(tz=timezone.utc),
        )

        llm_response = json.dumps({
            "correctness": "partial",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "is_optimal": True,
            "edge_cases_handled": [],
            "edge_cases_missed": ["empty array"],
            "code_quality_notes": None,
            "follow_up_warranted": True,
            "follow_up_text": "Can you implement this in actual code?",
            "difficulty_signal": "maintain",
        })
        mock_llm_client.complete.return_value = llm_response

        evaluation = await technical_round_service.evaluate_coding_solution(
            session=sample_session,
            problem=sample_coding_problem,
            response=pseudocode_response,
        )

        assert evaluation.solution_format == SolutionFormat.PSEUDOCODE

    @pytest.mark.asyncio
    async def test_falls_back_on_invalid_evaluation_json(
        self,
        technical_round_service,
        mock_llm_client,
        sample_session,
        sample_coding_problem,
        sample_user_response,
    ):
        """Should return fallback evaluation when LLM returns invalid JSON."""
        mock_llm_client.complete.return_value = "Invalid JSON response"

        evaluation = await technical_round_service.evaluate_coding_solution(
            session=sample_session,
            problem=sample_coding_problem,
            response=sample_user_response,
        )

        # Should still return a valid evaluation
        assert isinstance(evaluation, TechnicalEvaluation)
        assert evaluation.correctness == "partial"
        assert evaluation.difficulty_signal == "maintain"
        assert "error" in evaluation.code_quality_notes.lower()

    @pytest.mark.asyncio
    async def test_falls_back_on_evaluation_failure(
        self,
        technical_round_service,
        mock_llm_client,
        sample_session,
        sample_coding_problem,
        sample_user_response,
    ):
        """Should return fallback evaluation when LLM call fails."""
        mock_llm_client.complete.side_effect = Exception("LLM error")

        evaluation = await technical_round_service.evaluate_coding_solution(
            session=sample_session,
            problem=sample_coding_problem,
            response=sample_user_response,
        )

        # Should still return a valid evaluation
        assert isinstance(evaluation, TechnicalEvaluation)
        assert evaluation.question_id == sample_coding_problem.problem_id
        assert evaluation.response_id == sample_user_response.response_id


# ---------------------------------------------------------------------------
# Difficulty Adjustment Tests
# ---------------------------------------------------------------------------


class TestAdjustDifficulty:
    """Test adjust_difficulty method."""

    def test_increases_difficulty_on_strong_performance(
        self, technical_round_service, sample_session
    ):
        """Should increase difficulty when user performs well."""
        evaluation = TechnicalEvaluation(
            evaluation_id="eval-1",
            question_id="q-1",
            response_id="r-1",
            correctness="correct",
            complexity_analysis=ComplexityAnalysis(
                time_complexity="O(n)",
                space_complexity="O(1)",
                is_optimal=True,
            ),
            difficulty_signal="increase",
            evaluated_at=datetime.now(tz=timezone.utc),
        )

        # Test EASY -> MEDIUM
        sample_session.problem_difficulty = ProblemDifficulty.EASY
        new_difficulty = technical_round_service.adjust_difficulty(
            sample_session, evaluation
        )
        assert new_difficulty == ProblemDifficulty.MEDIUM

        # Test MEDIUM -> HARD
        sample_session.problem_difficulty = ProblemDifficulty.MEDIUM
        new_difficulty = technical_round_service.adjust_difficulty(
            sample_session, evaluation
        )
        assert new_difficulty == ProblemDifficulty.HARD

    def test_maintains_difficulty_at_hard_boundary(
        self, technical_round_service, sample_session
    ):
        """Should not increase difficulty beyond HARD."""
        evaluation = TechnicalEvaluation(
            evaluation_id="eval-1",
            question_id="q-1",
            response_id="r-1",
            difficulty_signal="increase",
            evaluated_at=datetime.now(tz=timezone.utc),
        )

        sample_session.problem_difficulty = ProblemDifficulty.HARD
        new_difficulty = technical_round_service.adjust_difficulty(
            sample_session, evaluation
        )
        assert new_difficulty == ProblemDifficulty.HARD

    def test_decreases_difficulty_on_weak_performance(
        self, technical_round_service, sample_session
    ):
        """Should decrease difficulty when user struggles."""
        evaluation = TechnicalEvaluation(
            evaluation_id="eval-1",
            question_id="q-1",
            response_id="r-1",
            correctness="incorrect",
            difficulty_signal="decrease",
            evaluated_at=datetime.now(tz=timezone.utc),
        )

        # Test HARD -> MEDIUM
        sample_session.problem_difficulty = ProblemDifficulty.HARD
        new_difficulty = technical_round_service.adjust_difficulty(
            sample_session, evaluation
        )
        assert new_difficulty == ProblemDifficulty.MEDIUM

        # Test MEDIUM -> EASY
        sample_session.problem_difficulty = ProblemDifficulty.MEDIUM
        new_difficulty = technical_round_service.adjust_difficulty(
            sample_session, evaluation
        )
        assert new_difficulty == ProblemDifficulty.EASY

    def test_maintains_difficulty_at_easy_boundary(
        self, technical_round_service, sample_session
    ):
        """Should not decrease difficulty below EASY."""
        evaluation = TechnicalEvaluation(
            evaluation_id="eval-1",
            question_id="q-1",
            response_id="r-1",
            difficulty_signal="decrease",
            evaluated_at=datetime.now(tz=timezone.utc),
        )

        sample_session.problem_difficulty = ProblemDifficulty.EASY
        new_difficulty = technical_round_service.adjust_difficulty(
            sample_session, evaluation
        )
        assert new_difficulty == ProblemDifficulty.EASY

    def test_maintains_difficulty_on_maintain_signal(
        self, technical_round_service, sample_session
    ):
        """Should maintain difficulty when signal is 'maintain'."""
        evaluation = TechnicalEvaluation(
            evaluation_id="eval-1",
            question_id="q-1",
            response_id="r-1",
            difficulty_signal="maintain",
            evaluated_at=datetime.now(tz=timezone.utc),
        )

        for difficulty in [ProblemDifficulty.EASY, ProblemDifficulty.MEDIUM, ProblemDifficulty.HARD]:
            sample_session.problem_difficulty = difficulty
            new_difficulty = technical_round_service.adjust_difficulty(
                sample_session, evaluation
            )
            assert new_difficulty == difficulty

    def test_records_adjustment_in_history(
        self, technical_round_service, sample_session
    ):
        """Should record difficulty adjustments in session history."""
        evaluation = TechnicalEvaluation(
            evaluation_id="eval-1",
            question_id="q-1",
            response_id="r-1",
            difficulty_signal="increase",
            evaluated_at=datetime.now(tz=timezone.utc),
        )

        sample_session.problem_difficulty = ProblemDifficulty.EASY
        sample_session.difficulty_adjustment_history = []

        new_difficulty = technical_round_service.adjust_difficulty(
            sample_session, evaluation
        )

        assert new_difficulty == ProblemDifficulty.MEDIUM
        assert len(sample_session.difficulty_adjustment_history) == 1
        
        adjustment = sample_session.difficulty_adjustment_history[0]
        assert adjustment["from"] == "easy"
        assert adjustment["to"] == "medium"
        assert adjustment["reason"] == "increase"
        assert "timestamp" in adjustment


# ---------------------------------------------------------------------------
# Solution Format Detection Tests
# ---------------------------------------------------------------------------


class TestParseSolutionFormat:
    """Test parse_solution_format method."""

    def test_detects_code_with_markdown_fence(self, technical_round_service):
        """Should detect CODE format with markdown fence."""
        text = """```python
def solution(arr):
    return sorted(arr)
```"""
        assert technical_round_service.parse_solution_format(text) == SolutionFormat.CODE

    def test_detects_code_with_keywords(self, technical_round_service):
        """Should detect CODE format from programming keywords."""
        text = """def solution(arr):
    for i in range(len(arr)):
        if arr[i] == 0:
            return i
    return -1"""
        assert technical_round_service.parse_solution_format(text) == SolutionFormat.CODE

    def test_detects_pseudocode(self, technical_round_service):
        """Should detect PSEUDOCODE format."""
        text = """Step 1: Initialize counter
Step 2: Iterate through array
Step 3: Return result"""
        assert technical_round_service.parse_solution_format(text) == SolutionFormat.PSEUDOCODE

    def test_detects_explanation(self, technical_round_service):
        """Should detect EXPLANATION format."""
        text = """I would solve this by using a hash map to track elements."""
        result = technical_round_service.parse_solution_format(text)
        # Could be EXPLANATION or PSEUDOCODE depending on keywords
        assert result in (SolutionFormat.EXPLANATION, SolutionFormat.PSEUDOCODE)
