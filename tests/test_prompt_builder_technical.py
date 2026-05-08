"""Unit tests for PromptBuilder technical round prompt methods.

Covers:
- DSA problem generation prompt with different difficulties and topics
- DSA solution evaluation prompt with different solution formats
- System Design question generation prompt
- System Design evaluation prompt for each design phase
- Technical feedback prompt construction for DSA and System Design rounds
- JSON structure requirements in all technical prompts
"""

import re
from datetime import datetime, timezone

import pytest

from interview_practice_partner.domain.enums import (
    DesignAspect,
    DesignPhase,
    InterviewRoundType,
    ProblemDifficulty,
    ProblemTopic,
    Role,
    SolutionFormat,
    Stage,
)
from interview_practice_partner.domain.models import (
    CodingProblem,
    Question,
    SessionState,
    SystemDesignQuestion,
    UserResponse,
)
from interview_practice_partner.llm.prompt_builder import PromptBuilder

# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

NOW = datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
UUID1 = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
UUID2 = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
UUID3 = "cccccccc-cccc-cccc-cccc-cccccccccccc"
UUID4 = "dddddddd-dddd-dddd-dddd-dddddddddddd"
PHONE = "+15550001234"

HTML_TAG_PATTERN = re.compile(r"<[a-zA-Z][^>]*>")
MARKDOWN_HEADER_PATTERN = re.compile(r"^#{1,6}\s", re.MULTILINE)
FENCED_CODE_BLOCK_PATTERN = re.compile(r"```")


def assert_no_forbidden_formatting(messages: list[dict]) -> None:
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


def make_session(
    role: Role = Role.SOFTWARE_ENGINEER,
    stage: Stage = Stage.INTERVIEW,
    interview_round_type: InterviewRoundType | None = None,
    problem_difficulty: ProblemDifficulty = ProblemDifficulty.MEDIUM,
    design_phase: DesignPhase | None = None,
    questions: list | None = None,
    responses: list | None = None,
    topics_covered: list | None = None,
    design_aspects_covered: list | None = None,
    difficulty_adjustment_history: list | None = None,
) -> SessionState:
    return SessionState(
        session_id=UUID1,
        phone_number=PHONE,
        stage=stage,
        role=role,
        interview_round_type=interview_round_type,
        problem_difficulty=problem_difficulty,
        design_phase=design_phase,
        questions=questions or [],
        responses=responses or [],
        topics_covered=topics_covered or [],
        design_aspects_covered=design_aspects_covered or [],
        difficulty_adjustment_history=difficulty_adjustment_history or [],
        created_at=NOW,
        updated_at=NOW,
    )


def make_coding_problem(
    problem_id: str = UUID2,
    text: str = "Given an array of integers, return indices of the two numbers that add up to target.",
    difficulty: ProblemDifficulty = ProblemDifficulty.MEDIUM,
    topic: ProblemTopic = ProblemTopic.ARRAYS,
    constraints: str = "2 <= nums.length <= 10^4",
    examples: list[str] | None = None,
) -> CodingProblem:
    return CodingProblem(
        problem_id=problem_id,
        text=text,
        difficulty=difficulty,
        topic=topic,
        constraints=constraints,
        examples=examples or ["Input: nums=[2,7,11,15], target=9 -> Output: [0,1]"],
        asked_at=NOW,
    )


def make_system_design_question(
    question_id: str = UUID3,
    text: str = "Design a social media feed like Twitter",
    system_name: str = "Twitter",
    description: str = "A platform for short messages and social connections.",
) -> SystemDesignQuestion:
    return SystemDesignQuestion(
        question_id=question_id,
        text=text,
        system_name=system_name,
        description=description,
        asked_at=NOW,
    )


def make_user_response(
    response_id: str = UUID4,
    question_id: str = UUID2,
    text: str = "I would use a hash map to store the complement of each number.",
    word_count: int = 15,
) -> UserResponse:
    return UserResponse(
        response_id=response_id,
        question_id=question_id,
        text=text,
        word_count=word_count,
        received_at=NOW,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder()


# ===========================================================================
# build_coding_problem_generation_prompt
# ===========================================================================


class TestBuildCodingProblemGenerationPrompt:
    def test_returns_non_empty_list(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.MEDIUM, None, []
        )
        assert len(messages) > 0

    def test_first_message_is_system(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.EASY, None, []
        )
        assert messages[0]["role"] == "system"

    def test_last_message_is_user(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.HARD, None, []
        )
        assert messages[-1]["role"] == "user"

    def test_easy_difficulty_included_in_prompt(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.EASY, None, []
        )
        system_content = messages[0]["content"]
        assert "easy" in system_content.lower()

    def test_medium_difficulty_included_in_prompt(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.MEDIUM, None, []
        )
        system_content = messages[0]["content"]
        assert "medium" in system_content.lower()

    def test_hard_difficulty_included_in_prompt(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.HARD, None, []
        )
        system_content = messages[0]["content"]
        assert "hard" in system_content.lower()

    def test_topic_included_when_specified(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.MEDIUM, ProblemTopic.TREES, []
        )
        system_content = messages[0]["content"]
        assert "trees" in system_content.lower()

    def test_no_topic_uses_generic_instruction(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.MEDIUM, None, []
        )
        system_content = messages[0]["content"]
        assert "appropriate algorithmic topic" in system_content.lower()

    def test_all_topics_produce_valid_prompts(self, builder: PromptBuilder):
        session = make_session()
        for topic in ProblemTopic:
            messages = builder.build_coding_problem_generation_prompt(
                session, ProblemDifficulty.MEDIUM, topic, []
            )
            assert len(messages) >= 2
            assert messages[0]["role"] == "system"

    def test_previously_asked_problems_included(self, builder: PromptBuilder):
        session = make_session()
        problem = make_coding_problem(text="Find the maximum subarray sum.")
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.MEDIUM, None, [problem]
        )
        system_content = messages[0]["content"]
        assert "Find the maximum subarray sum." in system_content

    def test_no_problems_asked_message_when_empty(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.MEDIUM, None, []
        )
        system_content = messages[0]["content"]
        assert "No problems have been asked yet" in system_content

    def test_json_output_requested(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.MEDIUM, None, []
        )
        system_content = messages[0]["content"]
        assert "JSON" in system_content

    def test_json_fields_problem_statement_in_prompt(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.MEDIUM, None, []
        )
        system_content = messages[0]["content"]
        assert "problem_statement" in system_content

    def test_json_fields_examples_in_prompt(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.MEDIUM, None, []
        )
        system_content = messages[0]["content"]
        assert "examples" in system_content

    def test_json_fields_constraints_in_prompt(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.MEDIUM, None, []
        )
        system_content = messages[0]["content"]
        assert "constraints" in system_content

    def test_json_fields_topic_in_prompt(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.MEDIUM, None, []
        )
        system_content = messages[0]["content"]
        assert '"topic"' in system_content

    def test_user_message_mentions_difficulty(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.HARD, None, []
        )
        user_content = messages[-1]["content"]
        assert "hard" in user_content.lower()

    def test_user_message_mentions_topic_when_specified(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.MEDIUM, ProblemTopic.GRAPHS, []
        )
        user_content = messages[-1]["content"]
        assert "graphs" in user_content.lower()

    def test_no_forbidden_formatting(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_coding_problem_generation_prompt(
            session, ProblemDifficulty.MEDIUM, ProblemTopic.ARRAYS, []
        )
        assert_no_forbidden_formatting(messages)


# ===========================================================================
# build_coding_solution_evaluation_prompt
# ===========================================================================


class TestBuildCodingSolutionEvaluationPrompt:
    def test_returns_non_empty_list(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.CODE
        )
        assert len(messages) > 0

    def test_first_message_is_system(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.CODE
        )
        assert messages[0]["role"] == "system"

    def test_last_message_is_user(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.PSEUDOCODE
        )
        assert messages[-1]["role"] == "user"

    def test_problem_text_included_in_system_prompt(self, builder: PromptBuilder):
        problem = make_coding_problem(text="Find the longest palindromic substring.")
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.CODE
        )
        system_content = messages[0]["content"]
        assert "Find the longest palindromic substring." in system_content

    def test_problem_constraints_included_in_system_prompt(self, builder: PromptBuilder):
        problem = make_coding_problem(constraints="1 <= s.length <= 1000")
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.CODE
        )
        system_content = messages[0]["content"]
        assert "1 <= s.length <= 1000" in system_content

    def test_problem_examples_included_in_system_prompt(self, builder: PromptBuilder):
        problem = make_coding_problem(examples=["Input: [1,2,3] -> Output: 6"])
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.CODE
        )
        system_content = messages[0]["content"]
        assert "Input: [1,2,3] -> Output: 6" in system_content

    def test_response_text_included_in_system_prompt(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response(text="def two_sum(nums, target): pass")
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.CODE
        )
        system_content = messages[0]["content"]
        assert "def two_sum(nums, target): pass" in system_content

    def test_code_format_note_in_system_prompt(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.CODE
        )
        system_content = messages[0]["content"]
        assert "actual code" in system_content.lower()

    def test_pseudocode_format_note_in_system_prompt(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.PSEUDOCODE
        )
        system_content = messages[0]["content"]
        assert "pseudocode" in system_content.lower()

    def test_explanation_format_note_in_system_prompt(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.EXPLANATION
        )
        system_content = messages[0]["content"]
        assert "plain explanation" in system_content.lower()

    def test_json_output_requested(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.CODE
        )
        system_content = messages[0]["content"]
        assert "JSON" in system_content

    def test_json_field_correctness_in_prompt(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.CODE
        )
        system_content = messages[0]["content"]
        assert "correctness" in system_content

    def test_json_field_time_complexity_in_prompt(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.CODE
        )
        system_content = messages[0]["content"]
        assert "time_complexity" in system_content

    def test_json_field_space_complexity_in_prompt(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.CODE
        )
        system_content = messages[0]["content"]
        assert "space_complexity" in system_content

    def test_json_field_is_optimal_in_prompt(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.CODE
        )
        system_content = messages[0]["content"]
        assert "is_optimal" in system_content

    def test_json_field_edge_cases_handled_in_prompt(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.CODE
        )
        system_content = messages[0]["content"]
        assert "edge_cases_handled" in system_content

    def test_json_field_edge_cases_missed_in_prompt(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.CODE
        )
        system_content = messages[0]["content"]
        assert "edge_cases_missed" in system_content

    def test_json_field_follow_up_warranted_in_prompt(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.CODE
        )
        system_content = messages[0]["content"]
        assert "follow_up_warranted" in system_content

    def test_json_field_difficulty_signal_in_prompt(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.CODE
        )
        system_content = messages[0]["content"]
        assert "difficulty_signal" in system_content

    def test_all_solution_formats_produce_valid_prompts(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response()
        for fmt in SolutionFormat:
            messages = builder.build_coding_solution_evaluation_prompt(problem, response, fmt)
            assert len(messages) >= 2
            assert messages[0]["role"] == "system"

    def test_no_forbidden_formatting(self, builder: PromptBuilder):
        problem = make_coding_problem()
        response = make_user_response()
        messages = builder.build_coding_solution_evaluation_prompt(
            problem, response, SolutionFormat.CODE
        )
        assert_no_forbidden_formatting(messages)


# ===========================================================================
# build_system_design_question_generation_prompt
# ===========================================================================


class TestBuildSystemDesignQuestionGenerationPrompt:
    def test_returns_non_empty_list(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_system_design_question_generation_prompt(session, [])
        assert len(messages) > 0

    def test_first_message_is_system(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_system_design_question_generation_prompt(session, [])
        assert messages[0]["role"] == "system"

    def test_last_message_is_user(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_system_design_question_generation_prompt(session, [])
        assert messages[-1]["role"] == "user"

    def test_system_prompt_mentions_common_design_systems(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_system_design_question_generation_prompt(session, [])
        system_content = messages[0]["content"]
        # At least some of the common systems should be mentioned
        common_systems = ["Twitter", "URL shortener", "Instagram", "Netflix", "Uber", "WhatsApp"]
        mentioned = [s for s in common_systems if s.lower() in system_content.lower()]
        assert len(mentioned) >= 3

    def test_previously_asked_questions_included(self, builder: PromptBuilder):
        session = make_session()
        question = make_system_design_question(
            system_name="Twitter",
            text="Design a social media feed like Twitter",
        )
        messages = builder.build_system_design_question_generation_prompt(session, [question])
        system_content = messages[0]["content"]
        assert "Twitter" in system_content

    def test_no_questions_asked_message_when_empty(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_system_design_question_generation_prompt(session, [])
        system_content = messages[0]["content"]
        assert "No questions have been asked yet" in system_content

    def test_json_output_requested(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_system_design_question_generation_prompt(session, [])
        system_content = messages[0]["content"]
        assert "JSON" in system_content

    def test_json_field_system_name_in_prompt(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_system_design_question_generation_prompt(session, [])
        system_content = messages[0]["content"]
        assert "system_name" in system_content

    def test_json_field_question_text_in_prompt(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_system_design_question_generation_prompt(session, [])
        system_content = messages[0]["content"]
        assert "question_text" in system_content

    def test_json_field_description_in_prompt(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_system_design_question_generation_prompt(session, [])
        system_content = messages[0]["content"]
        assert "description" in system_content

    def test_prompt_mentions_software_engineer_level(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_system_design_question_generation_prompt(session, [])
        system_content = messages[0]["content"]
        assert "software engineer" in system_content.lower()

    def test_no_forbidden_formatting(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_system_design_question_generation_prompt(session, [])
        assert_no_forbidden_formatting(messages)


# ===========================================================================
# build_system_design_evaluation_prompt
# ===========================================================================


class TestBuildSystemDesignEvaluationPrompt:
    def test_returns_non_empty_list(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.REQUIREMENTS_GATHERING
        )
        assert len(messages) > 0

    def test_first_message_is_system(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.HIGH_LEVEL_DESIGN
        )
        assert messages[0]["role"] == "system"

    def test_last_message_is_user(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.DEEP_DIVE
        )
        assert messages[-1]["role"] == "user"

    def test_question_text_included_in_system_prompt(self, builder: PromptBuilder):
        question = make_system_design_question(text="Design a URL shortener like bit.ly")
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.REQUIREMENTS_GATHERING
        )
        system_content = messages[0]["content"]
        assert "Design a URL shortener like bit.ly" in system_content

    def test_question_description_included_in_system_prompt(self, builder: PromptBuilder):
        question = make_system_design_question(description="A service that shortens long URLs.")
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.REQUIREMENTS_GATHERING
        )
        system_content = messages[0]["content"]
        assert "A service that shortens long URLs." in system_content

    def test_response_text_included_in_system_prompt(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response(text="I would use a distributed key-value store.")
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.HIGH_LEVEL_DESIGN
        )
        system_content = messages[0]["content"]
        assert "I would use a distributed key-value store." in system_content

    def test_requirements_gathering_phase_guidance(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.REQUIREMENTS_GATHERING
        )
        system_content = messages[0]["content"]
        assert "requirements" in system_content.lower()

    def test_high_level_design_phase_guidance(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.HIGH_LEVEL_DESIGN
        )
        system_content = messages[0]["content"]
        assert "components" in system_content.lower() or "high-level" in system_content.lower()

    def test_deep_dive_phase_guidance(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.DEEP_DIVE
        )
        system_content = messages[0]["content"]
        assert "detail" in system_content.lower() or "elaborate" in system_content.lower()

    def test_bottleneck_analysis_phase_guidance(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.BOTTLENECK_ANALYSIS
        )
        system_content = messages[0]["content"]
        assert "bottleneck" in system_content.lower() or "scalability" in system_content.lower()

    def test_all_design_phases_produce_valid_prompts(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        for phase in DesignPhase:
            messages = builder.build_system_design_evaluation_prompt(question, response, phase)
            assert len(messages) >= 2
            assert messages[0]["role"] == "system"

    def test_json_output_requested(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.HIGH_LEVEL_DESIGN
        )
        system_content = messages[0]["content"]
        assert "JSON" in system_content

    def test_json_field_design_aspects_evaluated_in_prompt(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.HIGH_LEVEL_DESIGN
        )
        system_content = messages[0]["content"]
        assert "design_aspects_evaluated" in system_content

    def test_json_field_scalability_in_prompt(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.HIGH_LEVEL_DESIGN
        )
        system_content = messages[0]["content"]
        assert "scalability" in system_content

    def test_json_field_database_design_in_prompt(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.HIGH_LEVEL_DESIGN
        )
        system_content = messages[0]["content"]
        assert "database_design" in system_content

    def test_json_field_api_design_in_prompt(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.HIGH_LEVEL_DESIGN
        )
        system_content = messages[0]["content"]
        assert "api_design" in system_content

    def test_json_field_caching_strategy_in_prompt(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.HIGH_LEVEL_DESIGN
        )
        system_content = messages[0]["content"]
        assert "caching_strategy" in system_content

    def test_json_field_load_balancing_in_prompt(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.HIGH_LEVEL_DESIGN
        )
        system_content = messages[0]["content"]
        assert "load_balancing" in system_content

    def test_json_field_design_strengths_in_prompt(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.DEEP_DIVE
        )
        system_content = messages[0]["content"]
        assert "design_strengths" in system_content

    def test_json_field_design_weaknesses_in_prompt(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.DEEP_DIVE
        )
        system_content = messages[0]["content"]
        assert "design_weaknesses" in system_content

    def test_json_field_follow_up_warranted_in_prompt(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.BOTTLENECK_ANALYSIS
        )
        system_content = messages[0]["content"]
        assert "follow_up_warranted" in system_content

    def test_json_field_next_phase_suggestion_in_prompt(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.REQUIREMENTS_GATHERING
        )
        system_content = messages[0]["content"]
        assert "next_phase_suggestion" in system_content

    def test_current_phase_value_in_system_prompt(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.DEEP_DIVE
        )
        system_content = messages[0]["content"]
        assert "deep_dive" in system_content.lower()

    def test_no_forbidden_formatting(self, builder: PromptBuilder):
        question = make_system_design_question()
        response = make_user_response()
        messages = builder.build_system_design_evaluation_prompt(
            question, response, DesignPhase.HIGH_LEVEL_DESIGN
        )
        assert_no_forbidden_formatting(messages)


# ===========================================================================
# build_technical_feedback_prompt
# ===========================================================================


class TestBuildTechnicalFeedbackPrompt:
    # --- DSA round feedback ---

    def test_dsa_round_returns_non_empty_list(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)
        messages = builder.build_technical_feedback_prompt(session)
        assert len(messages) > 0

    def test_dsa_round_first_message_is_system(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)
        messages = builder.build_technical_feedback_prompt(session)
        assert messages[0]["role"] == "system"

    def test_dsa_round_last_message_is_user(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)
        messages = builder.build_technical_feedback_prompt(session)
        assert messages[-1]["role"] == "user"

    def test_dsa_round_type_mentioned_in_system_prompt(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)
        messages = builder.build_technical_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "dsa_coding" in system_content.lower()

    def test_dsa_round_json_output_requested(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)
        messages = builder.build_technical_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "JSON" in system_content

    def test_dsa_round_json_field_strengths_in_prompt(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)
        messages = builder.build_technical_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "strengths" in system_content

    def test_dsa_round_json_field_improvements_in_prompt(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)
        messages = builder.build_technical_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "improvements" in system_content

    def test_dsa_round_json_field_actionable_recommendations_in_prompt(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)
        messages = builder.build_technical_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "actionable_recommendations" in system_content

    def test_dsa_round_json_field_complexity_summary_in_prompt(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)
        messages = builder.build_technical_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "complexity_summary" in system_content

    def test_dsa_round_json_field_problem_solving_approach_in_prompt(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)
        messages = builder.build_technical_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "problem_solving_approach" in system_content

    def test_dsa_round_difficulty_in_transcript(self, builder: PromptBuilder):
        session = make_session(
            interview_round_type=InterviewRoundType.DSA_CODING,
            problem_difficulty=ProblemDifficulty.HARD,
        )
        messages = builder.build_technical_feedback_prompt(session)
        user_content = messages[-1]["content"]
        assert "hard" in user_content.lower()

    def test_dsa_round_topics_covered_in_transcript(self, builder: PromptBuilder):
        session = make_session(
            interview_round_type=InterviewRoundType.DSA_CODING,
            topics_covered=[ProblemTopic.ARRAYS, ProblemTopic.TREES],
        )
        messages = builder.build_technical_feedback_prompt(session)
        user_content = messages[-1]["content"]
        assert "arrays" in user_content.lower()
        assert "trees" in user_content.lower()

    def test_dsa_round_difficulty_adjustment_history_in_transcript(self, builder: PromptBuilder):
        session = make_session(
            interview_round_type=InterviewRoundType.DSA_CODING,
            difficulty_adjustment_history=[
                {"from": "medium", "to": "hard", "reason": "correct optimal solution"}
            ],
        )
        messages = builder.build_technical_feedback_prompt(session)
        user_content = messages[-1]["content"]
        assert "medium" in user_content.lower()
        assert "hard" in user_content.lower()

    # --- System Design round feedback ---

    def test_system_design_round_returns_non_empty_list(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.SYSTEM_DESIGN)
        messages = builder.build_technical_feedback_prompt(session)
        assert len(messages) > 0

    def test_system_design_round_first_message_is_system(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.SYSTEM_DESIGN)
        messages = builder.build_technical_feedback_prompt(session)
        assert messages[0]["role"] == "system"

    def test_system_design_round_type_mentioned_in_system_prompt(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.SYSTEM_DESIGN)
        messages = builder.build_technical_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "system_design" in system_content.lower()

    def test_system_design_round_json_output_requested(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.SYSTEM_DESIGN)
        messages = builder.build_technical_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "JSON" in system_content

    def test_system_design_round_json_field_strengths_in_prompt(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.SYSTEM_DESIGN)
        messages = builder.build_technical_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "strengths" in system_content

    def test_system_design_round_json_field_improvements_in_prompt(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.SYSTEM_DESIGN)
        messages = builder.build_technical_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "improvements" in system_content

    def test_system_design_round_json_field_design_thinking_in_prompt(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.SYSTEM_DESIGN)
        messages = builder.build_technical_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "design_thinking" in system_content

    def test_system_design_round_json_field_scalability_awareness_in_prompt(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.SYSTEM_DESIGN)
        messages = builder.build_technical_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "scalability_awareness" in system_content

    def test_system_design_round_design_aspects_in_transcript(self, builder: PromptBuilder):
        session = make_session(
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
            design_aspects_covered=[DesignAspect.SCALABILITY, DesignAspect.API_DESIGN],
        )
        messages = builder.build_technical_feedback_prompt(session)
        user_content = messages[-1]["content"]
        assert "scalability" in user_content.lower()
        assert "api_design" in user_content.lower()

    # --- No round type fallback ---

    def test_no_round_type_falls_back_to_behavioral_feedback(self, builder: PromptBuilder):
        session = make_session(interview_round_type=None)
        messages = builder.build_technical_feedback_prompt(session)
        # Should fall back to behavioral feedback — still returns a valid messages list
        assert len(messages) > 0
        assert messages[0]["role"] == "system"

    # --- Formatting ---

    def test_dsa_round_no_forbidden_formatting(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)
        messages = builder.build_technical_feedback_prompt(session)
        assert_no_forbidden_formatting(messages)

    def test_system_design_round_no_forbidden_formatting(self, builder: PromptBuilder):
        session = make_session(interview_round_type=InterviewRoundType.SYSTEM_DESIGN)
        messages = builder.build_technical_feedback_prompt(session)
        assert_no_forbidden_formatting(messages)
