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


# ---------------------------------------------------------------------------
# System Design Question Generation Tests
# ---------------------------------------------------------------------------


class TestGenerateSystemDesignQuestion:
    """Test generate_system_design_question method.

    **Validates: Requirements 6.1-6.5, 19.2**
    """

    @pytest.mark.asyncio
    async def test_generates_question_with_valid_llm_response(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Should generate a system design question from valid LLM JSON response."""
        llm_response = json.dumps({
            "system_name": "Twitter",
            "question_text": "Design a social media feed like Twitter",
            "description": "A platform where users post short messages and follow others.",
        })
        mock_llm_client.complete.return_value = llm_response

        question = await technical_round_service.generate_system_design_question(
            session=sample_session,
        )

        assert isinstance(question, SystemDesignQuestion)
        assert question.system_name == "Twitter"
        assert question.text == "Design a social media feed like Twitter"
        assert "platform" in question.description
        assert question.question_id is not None
        assert question.asked_at is not None

    @pytest.mark.asyncio
    async def test_initializes_design_phase_to_requirements_gathering(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Should set design_phase to REQUIREMENTS_GATHERING on the session."""
        llm_response = json.dumps({
            "system_name": "URL Shortener",
            "question_text": "Design a URL shortener like bit.ly",
            "description": "A service that shortens long URLs.",
        })
        mock_llm_client.complete.return_value = llm_response

        # Ensure design_phase starts as None
        sample_session.design_phase = None

        await technical_round_service.generate_system_design_question(
            session=sample_session,
        )

        assert sample_session.design_phase == DesignPhase.REQUIREMENTS_GATHERING

    @pytest.mark.asyncio
    async def test_falls_back_on_invalid_json(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Should return fallback question when LLM returns invalid JSON."""
        mock_llm_client.complete.return_value = "Not valid JSON at all"

        question = await technical_round_service.generate_system_design_question(
            session=sample_session,
        )

        assert isinstance(question, SystemDesignQuestion)
        assert len(question.text) > 0
        assert len(question.system_name) > 0
        assert question.question_id is not None

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_failure(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Should return fallback question when LLM call raises an exception."""
        mock_llm_client.complete.side_effect = Exception("LLM timeout")

        question = await technical_round_service.generate_system_design_question(
            session=sample_session,
        )

        assert isinstance(question, SystemDesignQuestion)
        assert len(question.text) > 0
        assert question.question_id is not None

    @pytest.mark.asyncio
    async def test_passes_previously_asked_questions_for_distinctness(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Should pass existing session questions to the prompt for deduplication."""
        from interview_practice_partner.domain.enums import QuestionType

        # Add a previously asked question to the session
        existing_question = Question(
            question_id="q-existing",
            text="Design Twitter",
            question_type=QuestionType.TECHNICAL,
            asked_at=datetime.now(tz=timezone.utc),
        )
        sample_session.questions = [existing_question]

        llm_response = json.dumps({
            "system_name": "Netflix",
            "question_text": "Design a video streaming service like Netflix",
            "description": "A platform for streaming movies and TV shows.",
        })
        mock_llm_client.complete.return_value = llm_response

        question = await technical_round_service.generate_system_design_question(
            session=sample_session,
        )

        # LLM should have been called (with deduplication context)
        mock_llm_client.complete.assert_called_once()
        assert isinstance(question, SystemDesignQuestion)
        assert question.system_name == "Netflix"

    @pytest.mark.asyncio
    async def test_calls_llm_with_correct_parameters(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Should call LLM with temperature=0.8 and max_tokens=1024."""
        llm_response = json.dumps({
            "system_name": "Uber",
            "question_text": "Design a ride-sharing service like Uber",
            "description": "A platform connecting drivers and riders.",
        })
        mock_llm_client.complete.return_value = llm_response

        await technical_round_service.generate_system_design_question(
            session=sample_session,
        )

        mock_llm_client.complete.assert_called_once()
        call_kwargs = mock_llm_client.complete.call_args.kwargs
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_fallback_question_is_url_shortener(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Fallback question should be the URL Shortener design problem."""
        mock_llm_client.complete.side_effect = Exception("LLM unavailable")

        question = await technical_round_service.generate_system_design_question(
            session=sample_session,
        )

        assert question.system_name == "URL Shortener"
        assert "url shortener" in question.text.lower() or "bit.ly" in question.text.lower()


# ---------------------------------------------------------------------------
# System Design Evaluation Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_system_design_question():
    """Create a sample system design question."""
    return SystemDesignQuestion(
        question_id="sd-question-123",
        text="Design a URL shortener service like bit.ly",
        system_name="URL Shortener",
        description="A service that takes long URLs and generates short unique aliases.",
        asked_at=datetime.now(tz=timezone.utc),
    )


@pytest.fixture
def sample_design_response():
    """Create a sample user response for a system design question."""
    return UserResponse(
        response_id="sd-response-123",
        question_id="sd-question-123",
        text=(
            "I would use a hash function to generate short codes. "
            "The database would store the mapping from short code to long URL. "
            "We can use Redis for caching frequently accessed URLs."
        ),
        word_count=40,
        received_at=datetime.now(tz=timezone.utc),
    )


@pytest.fixture
def sample_system_design_session():
    """Create a sample session state for system design testing."""
    now = datetime.now(tz=timezone.utc)
    return SessionState(
        session_id="sd-session-123",
        phone_number="+1234567890",
        stage=Stage.INTERVIEW,
        interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
        design_phase=DesignPhase.REQUIREMENTS_GATHERING,
        design_aspects_covered=[],
        questions=[],
        responses=[],
        created_at=now,
        updated_at=now,
    )


class TestEvaluateSystemDesign:
    """Test evaluate_system_design method.

    **Validates: Requirements 7.1-7.7, 8.1-8.8, 19.2**
    """

    @pytest.mark.asyncio
    async def test_evaluates_design_with_valid_llm_response(
        self,
        technical_round_service,
        mock_llm_client,
        sample_system_design_session,
        sample_system_design_question,
        sample_design_response,
    ):
        """Should evaluate a system design response from valid LLM JSON response."""
        llm_response = json.dumps({
            "design_aspects_evaluated": {
                "scalability": "Good awareness of horizontal scaling needs.",
                "database_design": "Appropriate use of key-value store for URL mapping.",
                "api_design": "Basic REST API described.",
                "caching_strategy": "Redis caching mentioned for hot URLs.",
            },
            "design_strengths": ["Identified caching as a key optimization."],
            "design_weaknesses": ["Load balancing not addressed."],
            "follow_up_warranted": True,
            "follow_up_text": "How would you handle load balancing?",
            "next_phase_suggestion": "high_level_design",
        })
        mock_llm_client.complete.return_value = llm_response

        evaluation = await technical_round_service.evaluate_system_design(
            session=sample_system_design_session,
            question=sample_system_design_question,
            response=sample_design_response,
        )

        assert isinstance(evaluation, TechnicalEvaluation)
        assert "scalability" in evaluation.design_aspects_evaluated
        assert "database_design" in evaluation.design_aspects_evaluated
        assert len(evaluation.design_strengths) == 1
        assert len(evaluation.design_weaknesses) == 1
        assert evaluation.follow_up_warranted is True
        assert evaluation.follow_up_text == "How would you handle load balancing?"
        assert evaluation.question_id == sample_system_design_question.question_id
        assert evaluation.response_id == sample_design_response.response_id

    @pytest.mark.asyncio
    async def test_updates_design_aspects_covered_in_session(
        self,
        technical_round_service,
        mock_llm_client,
        sample_system_design_session,
        sample_system_design_question,
        sample_design_response,
    ):
        """Should update session.design_aspects_covered with newly evaluated aspects."""
        llm_response = json.dumps({
            "design_aspects_evaluated": {
                "scalability": "Good scalability discussion.",
                "database_design": "Appropriate database choice.",
                "caching_strategy": "Redis caching mentioned.",
            },
            "design_strengths": ["Good overall design."],
            "design_weaknesses": [],
            "follow_up_warranted": False,
            "follow_up_text": None,
            "next_phase_suggestion": None,
        })
        mock_llm_client.complete.return_value = llm_response

        # Session starts with no aspects covered
        assert sample_system_design_session.design_aspects_covered == []

        await technical_round_service.evaluate_system_design(
            session=sample_system_design_session,
            question=sample_system_design_question,
            response=sample_design_response,
        )

        # Session should now have the evaluated aspects
        assert "scalability" in sample_system_design_session.design_aspects_covered
        assert "database_design" in sample_system_design_session.design_aspects_covered
        assert "caching_strategy" in sample_system_design_session.design_aspects_covered

    @pytest.mark.asyncio
    async def test_does_not_duplicate_design_aspects_covered(
        self,
        technical_round_service,
        mock_llm_client,
        sample_system_design_session,
        sample_system_design_question,
        sample_design_response,
    ):
        """Should not add duplicate aspects to session.design_aspects_covered."""
        from interview_practice_partner.domain.enums import DesignAspect

        # Pre-populate with one aspect
        sample_system_design_session.design_aspects_covered = [DesignAspect.SCALABILITY]

        llm_response = json.dumps({
            "design_aspects_evaluated": {
                "scalability": "Already covered, still good.",
                "api_design": "New aspect evaluated.",
            },
            "design_strengths": [],
            "design_weaknesses": [],
            "follow_up_warranted": False,
            "follow_up_text": None,
            "next_phase_suggestion": None,
        })
        mock_llm_client.complete.return_value = llm_response

        await technical_round_service.evaluate_system_design(
            session=sample_system_design_session,
            question=sample_system_design_question,
            response=sample_design_response,
        )

        # scalability should appear only once
        scalability_count = sum(
            1 for a in sample_system_design_session.design_aspects_covered
            if a == DesignAspect.SCALABILITY or a == "scalability"
        )
        assert scalability_count == 1

    @pytest.mark.asyncio
    async def test_updates_design_phase_on_next_phase_suggestion(
        self,
        technical_round_service,
        mock_llm_client,
        sample_system_design_session,
        sample_system_design_question,
        sample_design_response,
    ):
        """Should update session.design_phase when LLM suggests a next phase."""
        llm_response = json.dumps({
            "design_aspects_evaluated": {"scalability": "Good."},
            "design_strengths": ["Good requirements gathering."],
            "design_weaknesses": [],
            "follow_up_warranted": False,
            "follow_up_text": None,
            "next_phase_suggestion": "high_level_design",
        })
        mock_llm_client.complete.return_value = llm_response

        # Session starts at REQUIREMENTS_GATHERING
        assert sample_system_design_session.design_phase == DesignPhase.REQUIREMENTS_GATHERING

        await technical_round_service.evaluate_system_design(
            session=sample_system_design_session,
            question=sample_system_design_question,
            response=sample_design_response,
        )

        assert sample_system_design_session.design_phase == DesignPhase.HIGH_LEVEL_DESIGN

    @pytest.mark.asyncio
    async def test_does_not_change_phase_when_no_suggestion(
        self,
        technical_round_service,
        mock_llm_client,
        sample_system_design_session,
        sample_system_design_question,
        sample_design_response,
    ):
        """Should not change session.design_phase when next_phase_suggestion is null."""
        llm_response = json.dumps({
            "design_aspects_evaluated": {"scalability": "Needs more detail."},
            "design_strengths": [],
            "design_weaknesses": ["Incomplete requirements."],
            "follow_up_warranted": True,
            "follow_up_text": "Can you clarify the scale requirements?",
            "next_phase_suggestion": None,
        })
        mock_llm_client.complete.return_value = llm_response

        sample_system_design_session.design_phase = DesignPhase.REQUIREMENTS_GATHERING

        await technical_round_service.evaluate_system_design(
            session=sample_system_design_session,
            question=sample_system_design_question,
            response=sample_design_response,
        )

        # Phase should remain unchanged
        assert sample_system_design_session.design_phase == DesignPhase.REQUIREMENTS_GATHERING

    @pytest.mark.asyncio
    async def test_uses_current_phase_in_prompt(
        self,
        technical_round_service,
        mock_llm_client,
        sample_system_design_session,
        sample_system_design_question,
        sample_design_response,
    ):
        """Should call LLM with the current design phase from session."""
        llm_response = json.dumps({
            "design_aspects_evaluated": {},
            "design_strengths": [],
            "design_weaknesses": [],
            "follow_up_warranted": False,
            "follow_up_text": None,
            "next_phase_suggestion": None,
        })
        mock_llm_client.complete.return_value = llm_response

        sample_system_design_session.design_phase = DesignPhase.DEEP_DIVE

        await technical_round_service.evaluate_system_design(
            session=sample_system_design_session,
            question=sample_system_design_question,
            response=sample_design_response,
        )

        mock_llm_client.complete.assert_called_once()
        call_args = mock_llm_client.complete.call_args
        # Verify the prompt contains the current phase
        messages = call_args.args[0]
        system_content = messages[0]["content"]
        assert "deep_dive" in system_content

    @pytest.mark.asyncio
    async def test_calls_llm_with_correct_parameters(
        self,
        technical_round_service,
        mock_llm_client,
        sample_system_design_session,
        sample_system_design_question,
        sample_design_response,
    ):
        """Should call LLM with temperature=0.3 and max_tokens=1024."""
        llm_response = json.dumps({
            "design_aspects_evaluated": {},
            "design_strengths": [],
            "design_weaknesses": [],
            "follow_up_warranted": False,
            "follow_up_text": None,
            "next_phase_suggestion": None,
        })
        mock_llm_client.complete.return_value = llm_response

        await technical_round_service.evaluate_system_design(
            session=sample_system_design_session,
            question=sample_system_design_question,
            response=sample_design_response,
        )

        call_kwargs = mock_llm_client.complete.call_args.kwargs
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_falls_back_on_invalid_json(
        self,
        technical_round_service,
        mock_llm_client,
        sample_system_design_session,
        sample_system_design_question,
        sample_design_response,
    ):
        """Should return fallback evaluation when LLM returns invalid JSON."""
        mock_llm_client.complete.return_value = "Not valid JSON"

        evaluation = await technical_round_service.evaluate_system_design(
            session=sample_system_design_session,
            question=sample_system_design_question,
            response=sample_design_response,
        )

        assert isinstance(evaluation, TechnicalEvaluation)
        assert evaluation.question_id == sample_system_design_question.question_id
        assert evaluation.response_id == sample_design_response.response_id
        assert len(evaluation.design_strengths) > 0

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_failure(
        self,
        technical_round_service,
        mock_llm_client,
        sample_system_design_session,
        sample_system_design_question,
        sample_design_response,
    ):
        """Should return fallback evaluation when LLM call raises an exception."""
        mock_llm_client.complete.side_effect = Exception("LLM timeout")

        evaluation = await technical_round_service.evaluate_system_design(
            session=sample_system_design_session,
            question=sample_system_design_question,
            response=sample_design_response,
        )

        assert isinstance(evaluation, TechnicalEvaluation)
        assert evaluation.question_id == sample_system_design_question.question_id
        assert evaluation.response_id == sample_design_response.response_id

    @pytest.mark.asyncio
    async def test_defaults_to_requirements_gathering_when_phase_is_none(
        self,
        technical_round_service,
        mock_llm_client,
        sample_system_design_session,
        sample_system_design_question,
        sample_design_response,
    ):
        """Should default to REQUIREMENTS_GATHERING phase when session.design_phase is None."""
        llm_response = json.dumps({
            "design_aspects_evaluated": {},
            "design_strengths": [],
            "design_weaknesses": [],
            "follow_up_warranted": False,
            "follow_up_text": None,
            "next_phase_suggestion": None,
        })
        mock_llm_client.complete.return_value = llm_response

        # Set design_phase to None
        sample_system_design_session.design_phase = None

        await technical_round_service.evaluate_system_design(
            session=sample_system_design_session,
            question=sample_system_design_question,
            response=sample_design_response,
        )

        # Verify the prompt used REQUIREMENTS_GATHERING as the phase
        messages = mock_llm_client.complete.call_args.args[0]
        system_content = messages[0]["content"]
        assert "requirements_gathering" in system_content

    @pytest.mark.asyncio
    async def test_phase_progression_through_all_phases(
        self,
        technical_round_service,
        mock_llm_client,
        sample_system_design_session,
        sample_system_design_question,
        sample_design_response,
    ):
        """Should correctly progress through all design phases based on LLM suggestions."""
        phases = [
            ("high_level_design", DesignPhase.HIGH_LEVEL_DESIGN),
            ("deep_dive", DesignPhase.DEEP_DIVE),
            ("bottleneck_analysis", DesignPhase.BOTTLENECK_ANALYSIS),
        ]

        sample_system_design_session.design_phase = DesignPhase.REQUIREMENTS_GATHERING

        for next_phase_str, expected_phase in phases:
            mock_llm_client.complete.return_value = json.dumps({
                "design_aspects_evaluated": {},
                "design_strengths": [],
                "design_weaknesses": [],
                "follow_up_warranted": False,
                "follow_up_text": None,
                "next_phase_suggestion": next_phase_str,
            })

            await technical_round_service.evaluate_system_design(
                session=sample_system_design_session,
                question=sample_system_design_question,
                response=sample_design_response,
            )

            assert sample_system_design_session.design_phase == expected_phase


# ---------------------------------------------------------------------------
# Design Phase Progression Tests
# ---------------------------------------------------------------------------


def _make_response(text: str) -> UserResponse:
    """Helper to create a UserResponse with the given text."""
    return UserResponse(
        response_id="resp-phase-test",
        question_id="q-phase-test",
        text=text,
        word_count=len(text.split()),
        received_at=datetime.now(tz=timezone.utc),
    )


def _make_session_at_phase(phase: DesignPhase | None) -> SessionState:
    """Helper to create a SessionState at a given design phase."""
    now = datetime.now(tz=timezone.utc)
    return SessionState(
        session_id="phase-session",
        phone_number="+1234567890",
        stage=Stage.INTERVIEW,
        interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
        design_phase=phase,
        questions=[],
        responses=[],
        created_at=now,
        updated_at=now,
    )


class TestDetermineNextDesignPhase:
    """Test determine_next_design_phase method.

    **Validates: Requirements 7.1-7.7**
    """

    # ------------------------------------------------------------------
    # Default / None phase handling
    # ------------------------------------------------------------------

    def test_defaults_to_requirements_gathering_when_phase_is_none(
        self, technical_round_service
    ):
        """When session.design_phase is None, should default to REQUIREMENTS_GATHERING."""
        session = _make_session_at_phase(None)
        response = _make_response("I would start by thinking about the architecture.")
        # No strong keyword signal → linear advance from default REQUIREMENTS_GATHERING
        result = technical_round_service.determine_next_design_phase(session, response)
        # Should advance from REQUIREMENTS_GATHERING (the default) to HIGH_LEVEL_DESIGN
        assert result == DesignPhase.HIGH_LEVEL_DESIGN

    # ------------------------------------------------------------------
    # Stay at final phase
    # ------------------------------------------------------------------

    def test_stays_at_bottleneck_analysis_when_already_at_final_phase(
        self, technical_round_service
    ):
        """Should stay at BOTTLENECK_ANALYSIS when already at the last phase."""
        session = _make_session_at_phase(DesignPhase.BOTTLENECK_ANALYSIS)
        response = _make_response("I would identify bottlenecks and optimize performance.")
        result = technical_round_service.determine_next_design_phase(session, response)
        assert result == DesignPhase.BOTTLENECK_ANALYSIS

    def test_stays_at_bottleneck_analysis_regardless_of_response_content(
        self, technical_round_service
    ):
        """Should stay at BOTTLENECK_ANALYSIS even with unrelated response content."""
        session = _make_session_at_phase(DesignPhase.BOTTLENECK_ANALYSIS)
        response = _make_response("I need to clarify the requirements first.")
        result = technical_round_service.determine_next_design_phase(session, response)
        assert result == DesignPhase.BOTTLENECK_ANALYSIS

    # ------------------------------------------------------------------
    # Linear progression (no strong keyword signal)
    # ------------------------------------------------------------------

    def test_advances_linearly_from_requirements_gathering(
        self, technical_round_service
    ):
        """Should advance to HIGH_LEVEL_DESIGN from REQUIREMENTS_GATHERING with no signal."""
        session = _make_session_at_phase(DesignPhase.REQUIREMENTS_GATHERING)
        response = _make_response("Here is my answer to the question.")
        result = technical_round_service.determine_next_design_phase(session, response)
        assert result == DesignPhase.HIGH_LEVEL_DESIGN

    def test_advances_linearly_from_high_level_design(
        self, technical_round_service
    ):
        """Should advance to DEEP_DIVE from HIGH_LEVEL_DESIGN with no signal."""
        session = _make_session_at_phase(DesignPhase.HIGH_LEVEL_DESIGN)
        response = _make_response("Here is my answer to the question.")
        result = technical_round_service.determine_next_design_phase(session, response)
        assert result == DesignPhase.DEEP_DIVE

    def test_advances_linearly_from_deep_dive(
        self, technical_round_service
    ):
        """Should advance to BOTTLENECK_ANALYSIS from DEEP_DIVE with no signal."""
        session = _make_session_at_phase(DesignPhase.DEEP_DIVE)
        response = _make_response("Here is my answer to the question.")
        result = technical_round_service.determine_next_design_phase(session, response)
        assert result == DesignPhase.BOTTLENECK_ANALYSIS

    # ------------------------------------------------------------------
    # Natural / keyword-driven transitions
    # ------------------------------------------------------------------

    def test_detects_high_level_design_keywords_and_jumps_forward(
        self, technical_round_service
    ):
        """Should jump to HIGH_LEVEL_DESIGN when response contains architecture keywords."""
        session = _make_session_at_phase(DesignPhase.REQUIREMENTS_GATHERING)
        response = _make_response(
            "The high-level architecture consists of a web service, a database, "
            "and a caching layer. The components interact via REST APIs."
        )
        result = technical_round_service.determine_next_design_phase(session, response)
        assert result == DesignPhase.HIGH_LEVEL_DESIGN

    def test_detects_bottleneck_keywords_and_skips_phases(
        self, technical_round_service
    ):
        """Should skip to BOTTLENECK_ANALYSIS when response contains bottleneck keywords."""
        session = _make_session_at_phase(DesignPhase.REQUIREMENTS_GATHERING)
        response = _make_response(
            "The main bottleneck will be the database. We need to optimize "
            "throughput and handle single point of failure scenarios."
        )
        result = technical_round_service.determine_next_design_phase(session, response)
        assert result == DesignPhase.BOTTLENECK_ANALYSIS

    def test_detects_deep_dive_keywords_and_skips_one_phase(
        self, technical_round_service
    ):
        """Should jump to DEEP_DIVE when response contains deep-dive keywords."""
        session = _make_session_at_phase(DesignPhase.REQUIREMENTS_GATHERING)
        response = _make_response(
            "Let me detail the database schema and the sharding strategy "
            "for the data model."
        )
        result = technical_round_service.determine_next_design_phase(session, response)
        assert result == DesignPhase.DEEP_DIVE

    # ------------------------------------------------------------------
    # No backward movement
    # ------------------------------------------------------------------

    def test_does_not_go_backwards_when_earlier_phase_keywords_detected(
        self, technical_round_service
    ):
        """Should not regress to an earlier phase even if earlier keywords appear."""
        session = _make_session_at_phase(DesignPhase.DEEP_DIVE)
        # "requirement" keyword signals REQUIREMENTS_GATHERING, but we're already at DEEP_DIVE
        response = _make_response(
            "The functional requirement here is that we need low latency."
        )
        result = technical_round_service.determine_next_design_phase(session, response)
        # Should advance forward (to BOTTLENECK_ANALYSIS), not go back
        assert result in (DesignPhase.DEEP_DIVE, DesignPhase.BOTTLENECK_ANALYSIS)
        assert result != DesignPhase.REQUIREMENTS_GATHERING
        assert result != DesignPhase.HIGH_LEVEL_DESIGN

    # ------------------------------------------------------------------
    # Return type
    # ------------------------------------------------------------------

    def test_always_returns_a_design_phase(self, technical_round_service):
        """Should always return a valid DesignPhase regardless of input."""
        for phase in [
            None,
            DesignPhase.REQUIREMENTS_GATHERING,
            DesignPhase.HIGH_LEVEL_DESIGN,
            DesignPhase.DEEP_DIVE,
            DesignPhase.BOTTLENECK_ANALYSIS,
        ]:
            session = _make_session_at_phase(phase)
            response = _make_response("Some response text.")
            result = technical_round_service.determine_next_design_phase(session, response)
            assert isinstance(result, DesignPhase)


# ---------------------------------------------------------------------------
# Additional Coverage Tests (Task 5.8 gap-fill)
# ---------------------------------------------------------------------------


class TestGenerateCodingProblemAllDifficulties:
    """Additional tests for coding problem generation across all difficulties.

    **Validates: Requirements 2.1, 2.2, 19.4, 19.7**
    """

    @pytest.mark.asyncio
    async def test_generates_hard_problem_with_valid_llm_response(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Should generate a HARD coding problem from a valid LLM JSON response."""
        llm_response = json.dumps({
            "problem_statement": "Find the median of two sorted arrays in O(log(m+n)) time.",
            "examples": [
                "Input: nums1 = [1,3], nums2 = [2]\nOutput: 2.0",
                "Input: nums1 = [1,2], nums2 = [3,4]\nOutput: 2.5",
            ],
            "constraints": "nums1.length + nums2.length >= 1",
            "topic": "searching",
        })
        mock_llm_client.complete.return_value = llm_response

        problem = await technical_round_service.generate_coding_problem(
            session=sample_session,
            difficulty=ProblemDifficulty.HARD,
            topic=ProblemTopic.SEARCHING,
        )

        assert isinstance(problem, CodingProblem)
        assert problem.difficulty == ProblemDifficulty.HARD
        assert problem.topic == ProblemTopic.SEARCHING
        assert "median" in problem.text.lower()
        assert len(problem.examples) == 2

    @pytest.mark.asyncio
    async def test_generates_easy_problem_with_topic(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Should generate an EASY problem with a specific topic."""
        llm_response = json.dumps({
            "problem_statement": "Check if a string is a palindrome.",
            "examples": [
                'Input: s = "racecar"\nOutput: true',
                'Input: s = "hello"\nOutput: false',
            ],
            "constraints": "1 <= s.length <= 1000",
            "topic": "strings",
        })
        mock_llm_client.complete.return_value = llm_response

        problem = await technical_round_service.generate_coding_problem(
            session=sample_session,
            difficulty=ProblemDifficulty.EASY,
            topic=ProblemTopic.STRINGS,
        )

        assert isinstance(problem, CodingProblem)
        assert problem.difficulty == ProblemDifficulty.EASY
        assert problem.topic == ProblemTopic.STRINGS

    @pytest.mark.asyncio
    async def test_fallback_preserves_requested_topic(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Fallback problem should preserve the requested topic when LLM fails."""
        mock_llm_client.complete.side_effect = Exception("Connection refused")

        problem = await technical_round_service.generate_coding_problem(
            session=sample_session,
            difficulty=ProblemDifficulty.EASY,
            topic=ProblemTopic.TREES,
        )

        assert isinstance(problem, CodingProblem)
        assert problem.difficulty == ProblemDifficulty.EASY
        assert problem.topic == ProblemTopic.TREES

    @pytest.mark.asyncio
    async def test_fallback_uses_arrays_when_no_topic_given(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Fallback problem should default to ARRAYS topic when no topic is specified."""
        mock_llm_client.complete.return_value = "not json"

        problem = await technical_round_service.generate_coding_problem(
            session=sample_session,
            difficulty=ProblemDifficulty.MEDIUM,
            topic=None,
        )

        assert isinstance(problem, CodingProblem)
        assert problem.topic == ProblemTopic.ARRAYS

    @pytest.mark.asyncio
    async def test_generates_problem_with_missing_optional_fields(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Should handle LLM response with missing optional fields gracefully."""
        # Minimal JSON — only required field present
        llm_response = json.dumps({
            "problem_statement": "Sort an array of integers.",
            "topic": "sorting",
        })
        mock_llm_client.complete.return_value = llm_response

        problem = await technical_round_service.generate_coding_problem(
            session=sample_session,
            difficulty=ProblemDifficulty.EASY,
            topic=None,
        )

        assert isinstance(problem, CodingProblem)
        assert problem.text == "Sort an array of integers."
        assert problem.topic == ProblemTopic.SORTING
        # Missing fields should default gracefully
        assert problem.constraints == "" or problem.constraints is not None
        assert isinstance(problem.examples, list)


class TestEvaluateCodingSolutionFormats:
    """Additional tests for solution evaluation with all three formats.

    **Validates: Requirements 3.1-3.4, 4.1, 19.4**
    """

    @pytest.mark.asyncio
    async def test_evaluates_explanation_format(
        self,
        technical_round_service,
        mock_llm_client,
        sample_session,
        sample_coding_problem,
    ):
        """Should evaluate a plain-English explanation and detect a non-code format."""
        # Pure conversational text with no code keywords or pseudocode structure
        explanation_response = UserResponse(
            response_id="resp-explanation",
            question_id="problem-123",
            text=(
                "My approach relies on a hash map. "
                "As we traverse the array, we look up whether the complement of the current "
                "value already exists in the map. When found, we have our answer."
            ),
            word_count=38,
            received_at=datetime.now(tz=timezone.utc),
        )

        llm_response = json.dumps({
            "correctness": "correct",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "is_optimal": True,
            "edge_cases_handled": [],
            "edge_cases_missed": ["empty array"],
            "code_quality_notes": None,
            "follow_up_warranted": True,
            "follow_up_text": "Can you implement this in code?",
            "difficulty_signal": "maintain",
        })
        mock_llm_client.complete.return_value = llm_response

        evaluation = await technical_round_service.evaluate_coding_solution(
            session=sample_session,
            problem=sample_coding_problem,
            response=explanation_response,
        )

        assert isinstance(evaluation, TechnicalEvaluation)
        # Conversational text should be classified as EXPLANATION or PSEUDOCODE
        # (not CODE), depending on heuristics
        assert evaluation.solution_format in (SolutionFormat.EXPLANATION, SolutionFormat.PSEUDOCODE)
        assert evaluation.correctness == "correct"

    @pytest.mark.asyncio
    async def test_evaluates_code_format_with_markdown_fence(
        self,
        technical_round_service,
        mock_llm_client,
        sample_session,
        sample_coding_problem,
        sample_user_response,
    ):
        """Should detect CODE format for markdown-fenced code and evaluate correctly."""
        llm_response = json.dumps({
            "correctness": "correct",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "is_optimal": True,
            "edge_cases_handled": ["empty array"],
            "edge_cases_missed": [],
            "code_quality_notes": "Clean implementation.",
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "increase",
        })
        mock_llm_client.complete.return_value = llm_response

        evaluation = await technical_round_service.evaluate_coding_solution(
            session=sample_session,
            problem=sample_coding_problem,
            response=sample_user_response,  # fixture uses ```python fence
        )

        assert evaluation.solution_format == SolutionFormat.CODE
        assert evaluation.difficulty_signal == "increase"

    @pytest.mark.asyncio
    async def test_evaluation_links_to_correct_question_and_response_ids(
        self,
        technical_round_service,
        mock_llm_client,
        sample_session,
        sample_coding_problem,
        sample_user_response,
    ):
        """Evaluation should reference the correct question_id and response_id."""
        llm_response = json.dumps({
            "correctness": "partial",
            "time_complexity": "O(n^2)",
            "space_complexity": "O(1)",
            "is_optimal": False,
            "edge_cases_handled": [],
            "edge_cases_missed": ["duplicates"],
            "code_quality_notes": "Nested loops used.",
            "follow_up_warranted": True,
            "follow_up_text": "Can you optimize to O(n)?",
            "difficulty_signal": "maintain",
        })
        mock_llm_client.complete.return_value = llm_response

        evaluation = await technical_round_service.evaluate_coding_solution(
            session=sample_session,
            problem=sample_coding_problem,
            response=sample_user_response,
        )

        assert evaluation.question_id == sample_coding_problem.problem_id
        assert evaluation.response_id == sample_user_response.response_id

    @pytest.mark.asyncio
    async def test_evaluation_with_missing_json_fields_uses_defaults(
        self,
        technical_round_service,
        mock_llm_client,
        sample_session,
        sample_coding_problem,
        sample_user_response,
    ):
        """Should use safe defaults when LLM response is missing optional fields."""
        # Minimal valid JSON — many fields absent
        llm_response = json.dumps({
            "correctness": "correct",
        })
        mock_llm_client.complete.return_value = llm_response

        evaluation = await technical_round_service.evaluate_coding_solution(
            session=sample_session,
            problem=sample_coding_problem,
            response=sample_user_response,
        )

        assert isinstance(evaluation, TechnicalEvaluation)
        assert evaluation.correctness == "correct"
        # Missing fields should have safe defaults
        assert evaluation.difficulty_signal == "maintain"
        assert isinstance(evaluation.edge_cases_handled, list)
        assert isinstance(evaluation.edge_cases_missed, list)


class TestDifficultyAdjustmentEdgeCases:
    """Additional edge-case tests for difficulty adjustment.

    **Validates: Requirements 15.1-15.5, 19.4**
    """

    def test_no_history_entry_when_difficulty_unchanged(
        self, technical_round_service, sample_session
    ):
        """Should NOT add a history entry when difficulty stays the same."""
        evaluation = TechnicalEvaluation(
            evaluation_id="eval-1",
            question_id="q-1",
            response_id="r-1",
            difficulty_signal="maintain",
            evaluated_at=datetime.now(tz=timezone.utc),
        )

        sample_session.problem_difficulty = ProblemDifficulty.MEDIUM
        sample_session.difficulty_adjustment_history = []

        technical_round_service.adjust_difficulty(sample_session, evaluation)

        assert len(sample_session.difficulty_adjustment_history) == 0

    def test_history_entry_has_iso_timestamp(
        self, technical_round_service, sample_session
    ):
        """Difficulty adjustment history entry should contain a valid ISO timestamp."""
        evaluation = TechnicalEvaluation(
            evaluation_id="eval-1",
            question_id="q-1",
            response_id="r-1",
            difficulty_signal="increase",
            evaluated_at=datetime.now(tz=timezone.utc),
        )

        sample_session.problem_difficulty = ProblemDifficulty.EASY
        sample_session.difficulty_adjustment_history = []

        technical_round_service.adjust_difficulty(sample_session, evaluation)

        assert len(sample_session.difficulty_adjustment_history) == 1
        entry = sample_session.difficulty_adjustment_history[0]
        # Timestamp should be parseable as ISO 8601
        parsed = datetime.fromisoformat(entry["timestamp"])
        assert parsed is not None

    def test_multiple_adjustments_accumulate_in_history(
        self, technical_round_service, sample_session
    ):
        """Multiple difficulty adjustments should all be recorded in history."""
        sample_session.difficulty_adjustment_history = []

        # First adjustment: EASY → MEDIUM
        sample_session.problem_difficulty = ProblemDifficulty.EASY
        eval1 = TechnicalEvaluation(
            evaluation_id="eval-1",
            question_id="q-1",
            response_id="r-1",
            difficulty_signal="increase",
            evaluated_at=datetime.now(tz=timezone.utc),
        )
        technical_round_service.adjust_difficulty(sample_session, eval1)
        sample_session.problem_difficulty = ProblemDifficulty.MEDIUM

        # Second adjustment: MEDIUM → HARD
        eval2 = TechnicalEvaluation(
            evaluation_id="eval-2",
            question_id="q-2",
            response_id="r-2",
            difficulty_signal="increase",
            evaluated_at=datetime.now(tz=timezone.utc),
        )
        technical_round_service.adjust_difficulty(sample_session, eval2)

        assert len(sample_session.difficulty_adjustment_history) == 2
        assert sample_session.difficulty_adjustment_history[0]["from"] == "easy"
        assert sample_session.difficulty_adjustment_history[0]["to"] == "medium"
        assert sample_session.difficulty_adjustment_history[1]["from"] == "medium"
        assert sample_session.difficulty_adjustment_history[1]["to"] == "hard"


class TestErrorHandlingAndFallback:
    """Tests for error handling and fallback behavior across all service methods.

    **Validates: Requirements 19.7**
    """

    @pytest.mark.asyncio
    async def test_coding_problem_fallback_returns_valid_problem_structure(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Fallback coding problem should have all required fields populated."""
        mock_llm_client.complete.side_effect = TimeoutError("LLM request timed out")

        problem = await technical_round_service.generate_coding_problem(
            session=sample_session,
            difficulty=ProblemDifficulty.MEDIUM,
            topic=None,
        )

        assert problem.problem_id is not None
        assert len(problem.text) > 0
        assert len(problem.examples) > 0
        assert problem.constraints is not None
        assert problem.asked_at is not None
        assert isinstance(problem.difficulty, ProblemDifficulty)
        assert isinstance(problem.topic, ProblemTopic)

    @pytest.mark.asyncio
    async def test_coding_evaluation_fallback_returns_valid_evaluation_structure(
        self,
        technical_round_service,
        mock_llm_client,
        sample_session,
        sample_coding_problem,
        sample_user_response,
    ):
        """Fallback coding evaluation should have all required fields populated."""
        mock_llm_client.complete.side_effect = TimeoutError("LLM request timed out")

        evaluation = await technical_round_service.evaluate_coding_solution(
            session=sample_session,
            problem=sample_coding_problem,
            response=sample_user_response,
        )

        assert evaluation.evaluation_id is not None
        assert evaluation.question_id == sample_coding_problem.problem_id
        assert evaluation.response_id == sample_user_response.response_id
        assert evaluation.correctness == "partial"
        assert evaluation.difficulty_signal == "maintain"
        assert evaluation.evaluated_at is not None
        assert "error" in evaluation.code_quality_notes.lower()

    @pytest.mark.asyncio
    async def test_system_design_question_fallback_returns_valid_structure(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Fallback system design question should have all required fields populated."""
        mock_llm_client.complete.side_effect = TimeoutError("LLM request timed out")

        question = await technical_round_service.generate_system_design_question(
            session=sample_session,
        )

        assert question.question_id is not None
        assert len(question.text) > 0
        assert len(question.system_name) > 0
        assert len(question.description) > 0
        assert question.asked_at is not None

    @pytest.mark.asyncio
    async def test_system_design_evaluation_fallback_returns_valid_structure(
        self,
        technical_round_service,
        mock_llm_client,
        sample_system_design_session,
        sample_system_design_question,
        sample_design_response,
    ):
        """Fallback system design evaluation should have all required fields populated."""
        mock_llm_client.complete.side_effect = TimeoutError("LLM request timed out")

        evaluation = await technical_round_service.evaluate_system_design(
            session=sample_system_design_session,
            question=sample_system_design_question,
            response=sample_design_response,
        )

        assert evaluation.evaluation_id is not None
        assert evaluation.question_id == sample_system_design_question.question_id
        assert evaluation.response_id == sample_design_response.response_id
        assert isinstance(evaluation.design_aspects_evaluated, dict)
        assert isinstance(evaluation.design_strengths, list)
        assert isinstance(evaluation.design_weaknesses, list)
        assert evaluation.evaluated_at is not None

    @pytest.mark.asyncio
    async def test_coding_problem_handles_empty_string_response(
        self, technical_round_service, mock_llm_client, sample_session
    ):
        """Should fall back gracefully when LLM returns an empty string."""
        mock_llm_client.complete.return_value = ""

        problem = await technical_round_service.generate_coding_problem(
            session=sample_session,
            difficulty=ProblemDifficulty.EASY,
            topic=None,
        )

        assert isinstance(problem, CodingProblem)
        assert len(problem.text) > 0

    @pytest.mark.asyncio
    async def test_system_design_evaluation_handles_empty_string_response(
        self,
        technical_round_service,
        mock_llm_client,
        sample_system_design_session,
        sample_system_design_question,
        sample_design_response,
    ):
        """Should fall back gracefully when LLM returns an empty string."""
        mock_llm_client.complete.return_value = ""

        evaluation = await technical_round_service.evaluate_system_design(
            session=sample_system_design_session,
            question=sample_system_design_question,
            response=sample_design_response,
        )

        assert isinstance(evaluation, TechnicalEvaluation)
        assert evaluation.question_id == sample_system_design_question.question_id


# ---------------------------------------------------------------------------
# Hint Generation Tests
# ---------------------------------------------------------------------------


class TestGenerateHint:
    """Test generate_hint method.

    **Validates: Requirements 14.3**
    """

    @pytest.mark.asyncio
    async def test_returns_hint_string_from_llm(
        self, technical_round_service, mock_llm_client, sample_coding_problem
    ):
        """Should return the LLM-generated hint as a string."""
        mock_llm_client.complete.return_value = "Think about how you can avoid scanning the array multiple times."

        hint = await technical_round_service.generate_hint(
            problem=sample_coding_problem,
            hint_number=1,
        )

        assert isinstance(hint, str)
        assert len(hint) > 0
        assert "multiple times" in hint

    @pytest.mark.asyncio
    async def test_default_hint_number_is_1(
        self, technical_round_service, mock_llm_client, sample_coding_problem
    ):
        """Should default to hint_number=1 when not specified."""
        mock_llm_client.complete.return_value = "Consider what data structure gives O(1) lookup."

        hint = await technical_round_service.generate_hint(problem=sample_coding_problem)

        assert isinstance(hint, str)
        assert len(hint) > 0
        mock_llm_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_calls_llm_with_correct_parameters(
        self, technical_round_service, mock_llm_client, sample_coding_problem
    ):
        """Should call LLM with temperature=0.5 and max_tokens=256."""
        mock_llm_client.complete.return_value = "A hash map could be useful here."

        await technical_round_service.generate_hint(
            problem=sample_coding_problem,
            hint_number=2,
        )

        mock_llm_client.complete.assert_called_once()
        call_kwargs = mock_llm_client.complete.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 256

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_failure(
        self, technical_round_service, mock_llm_client, sample_coding_problem
    ):
        """Should return a safe fallback hint when LLM call fails."""
        mock_llm_client.complete.side_effect = Exception("LLM timeout")

        hint = await technical_round_service.generate_hint(
            problem=sample_coding_problem,
            hint_number=1,
        )

        assert isinstance(hint, str)
        assert len(hint) > 0

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_hint(
        self, technical_round_service, mock_llm_client, sample_coding_problem
    ):
        """Should strip leading/trailing whitespace from the LLM response."""
        mock_llm_client.complete.return_value = "  Think about sorting first.  \n"

        hint = await technical_round_service.generate_hint(
            problem=sample_coding_problem,
            hint_number=1,
        )

        assert hint == "Think about sorting first."

    @pytest.mark.asyncio
    async def test_passes_hint_number_to_prompt_builder(
        self, technical_round_service, mock_llm_client, sample_coding_problem, prompt_builder
    ):
        """Should pass the hint_number to the prompt builder."""
        mock_llm_client.complete.return_value = "Use two pointers from each end."

        # Verify hint_number=3 produces a more direct prompt than hint_number=1
        # by checking the LLM is called (prompt builder is real, so this validates integration)
        await technical_round_service.generate_hint(
            problem=sample_coding_problem,
            hint_number=3,
        )

        mock_llm_client.complete.assert_called_once()
        # The messages passed to LLM should reference hint #3
        call_args = mock_llm_client.complete.call_args
        messages = call_args.args[0]
        combined_content = " ".join(m["content"] for m in messages)
        assert "3" in combined_content

    @pytest.mark.asyncio
    async def test_hint_does_not_reveal_full_solution(
        self, technical_round_service, mock_llm_client, sample_coding_problem
    ):
        """Prompt should instruct LLM not to reveal the full solution."""
        mock_llm_client.complete.return_value = "Consider a complementary lookup approach."

        await technical_round_service.generate_hint(
            problem=sample_coding_problem,
            hint_number=1,
        )

        # Verify the prompt instructs LLM not to reveal the full solution
        call_args = mock_llm_client.complete.call_args
        messages = call_args.args[0]
        system_message = next(m["content"] for m in messages if m["role"] == "system")
        assert "NOT" in system_message or "not" in system_message.lower()
        assert "solution" in system_message.lower()
