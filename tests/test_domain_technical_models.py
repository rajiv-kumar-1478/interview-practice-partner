"""Unit tests for technical interview round domain models and enums.

Covers:
- New enum membership and string values (InterviewRoundType, ProblemDifficulty,
  ProblemTopic, SolutionFormat, DesignPhase, DesignAspect)
- Enum serialization and deserialization
- CodingProblem construction and validation
- SystemDesignQuestion construction and validation
- ComplexityAnalysis construction and validation
- TechnicalEvaluation construction, defaults, and validation
- Requirements: 18.7, 19.4
"""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from interview_practice_partner.domain.enums import (
    DesignAspect,
    DesignPhase,
    InterviewRoundType,
    ProblemDifficulty,
    ProblemTopic,
    SolutionFormat,
)
from interview_practice_partner.domain.models import (
    CodingProblem,
    ComplexityAnalysis,
    SystemDesignQuestion,
    TechnicalEvaluation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
UUID1 = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
UUID2 = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
UUID3 = "cccccccc-cccc-cccc-cccc-cccccccccccc"


def make_coding_problem(**overrides) -> CodingProblem:
    defaults = dict(
        problem_id=UUID1,
        text="Given an array of integers, return indices of the two numbers that add up to target.",
        difficulty=ProblemDifficulty.MEDIUM,
        topic=ProblemTopic.ARRAYS,
        constraints="2 <= nums.length <= 10^4",
        examples=["Input: nums=[2,7,11,15], target=9 → Output: [0,1]"],
        asked_at=NOW,
    )
    defaults.update(overrides)
    return CodingProblem(**defaults)


def make_system_design_question(**overrides) -> SystemDesignQuestion:
    defaults = dict(
        question_id=UUID1,
        text="Design a URL shortener like bit.ly.",
        system_name="URL Shortener",
        description="A service that converts long URLs into short, shareable links.",
        asked_at=NOW,
    )
    defaults.update(overrides)
    return SystemDesignQuestion(**defaults)


def make_complexity_analysis(**overrides) -> ComplexityAnalysis:
    defaults = dict(
        time_complexity="O(n)",
        space_complexity="O(1)",
        is_optimal=True,
    )
    defaults.update(overrides)
    return ComplexityAnalysis(**defaults)


def make_technical_evaluation(**overrides) -> TechnicalEvaluation:
    defaults = dict(
        evaluation_id=UUID1,
        question_id=UUID2,
        response_id=UUID3,
        evaluated_at=NOW,
    )
    defaults.update(overrides)
    return TechnicalEvaluation(**defaults)


# ===========================================================================
# New enum tests
# ===========================================================================


class TestInterviewRoundTypeEnum:
    def test_all_members_exist(self):
        members = {m.name for m in InterviewRoundType}
        assert members == {"DSA_CODING", "SYSTEM_DESIGN", "BEHAVIORAL"}

    def test_string_values(self):
        assert InterviewRoundType.DSA_CODING == "dsa_coding"
        assert InterviewRoundType.SYSTEM_DESIGN == "system_design"
        assert InterviewRoundType.BEHAVIORAL == "behavioral"

    def test_is_str_subclass(self):
        assert isinstance(InterviewRoundType.DSA_CODING, str)

    def test_serializes_to_string_value(self):
        data = {"round": InterviewRoundType.DSA_CODING}
        serialized = json.dumps(data)
        assert "dsa_coding" in serialized

    def test_deserializes_from_string_value(self):
        assert InterviewRoundType("dsa_coding") == InterviewRoundType.DSA_CODING
        assert InterviewRoundType("system_design") == InterviewRoundType.SYSTEM_DESIGN
        assert InterviewRoundType("behavioral") == InterviewRoundType.BEHAVIORAL

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            InterviewRoundType("unknown_round")


class TestProblemDifficultyEnum:
    def test_all_members_exist(self):
        members = {m.name for m in ProblemDifficulty}
        assert members == {"EASY", "MEDIUM", "HARD"}

    def test_string_values(self):
        assert ProblemDifficulty.EASY == "easy"
        assert ProblemDifficulty.MEDIUM == "medium"
        assert ProblemDifficulty.HARD == "hard"

    def test_is_str_subclass(self):
        assert isinstance(ProblemDifficulty.MEDIUM, str)

    def test_deserializes_from_string_value(self):
        assert ProblemDifficulty("easy") == ProblemDifficulty.EASY
        assert ProblemDifficulty("medium") == ProblemDifficulty.MEDIUM
        assert ProblemDifficulty("hard") == ProblemDifficulty.HARD

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ProblemDifficulty("extreme")


class TestProblemTopicEnum:
    def test_all_required_members_exist(self):
        """Requirements 2.3: must cover at minimum these topics."""
        required = {
            "ARRAYS", "STRINGS", "LINKED_LISTS", "TREES", "GRAPHS",
            "DYNAMIC_PROGRAMMING", "SORTING", "SEARCHING",
        }
        members = {m.name for m in ProblemTopic}
        assert required.issubset(members)

    def test_all_members_exist(self):
        members = {m.name for m in ProblemTopic}
        assert members == {
            "ARRAYS", "STRINGS", "LINKED_LISTS", "TREES", "GRAPHS",
            "DYNAMIC_PROGRAMMING", "SORTING", "SEARCHING",
            "HASH_TABLES", "STACKS_QUEUES",
        }

    def test_string_values(self):
        assert ProblemTopic.ARRAYS == "arrays"
        assert ProblemTopic.DYNAMIC_PROGRAMMING == "dynamic_programming"
        assert ProblemTopic.STACKS_QUEUES == "stacks_queues"

    def test_is_str_subclass(self):
        assert isinstance(ProblemTopic.TREES, str)

    def test_deserializes_from_string_value(self):
        assert ProblemTopic("arrays") == ProblemTopic.ARRAYS
        assert ProblemTopic("graphs") == ProblemTopic.GRAPHS


class TestSolutionFormatEnum:
    def test_all_members_exist(self):
        members = {m.name for m in SolutionFormat}
        assert members == {"CODE", "PSEUDOCODE", "EXPLANATION"}

    def test_string_values(self):
        assert SolutionFormat.CODE == "code"
        assert SolutionFormat.PSEUDOCODE == "pseudocode"
        assert SolutionFormat.EXPLANATION == "explanation"

    def test_is_str_subclass(self):
        assert isinstance(SolutionFormat.CODE, str)

    def test_deserializes_from_string_value(self):
        assert SolutionFormat("code") == SolutionFormat.CODE
        assert SolutionFormat("pseudocode") == SolutionFormat.PSEUDOCODE
        assert SolutionFormat("explanation") == SolutionFormat.EXPLANATION

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            SolutionFormat("diagram")


class TestDesignPhaseEnum:
    def test_all_members_exist(self):
        members = {m.name for m in DesignPhase}
        assert members == {
            "REQUIREMENTS_GATHERING", "HIGH_LEVEL_DESIGN",
            "DEEP_DIVE", "BOTTLENECK_ANALYSIS",
        }

    def test_string_values(self):
        assert DesignPhase.REQUIREMENTS_GATHERING == "requirements_gathering"
        assert DesignPhase.HIGH_LEVEL_DESIGN == "high_level_design"
        assert DesignPhase.DEEP_DIVE == "deep_dive"
        assert DesignPhase.BOTTLENECK_ANALYSIS == "bottleneck_analysis"

    def test_is_str_subclass(self):
        assert isinstance(DesignPhase.DEEP_DIVE, str)

    def test_deserializes_from_string_value(self):
        assert DesignPhase("deep_dive") == DesignPhase.DEEP_DIVE
        assert DesignPhase("bottleneck_analysis") == DesignPhase.BOTTLENECK_ANALYSIS


class TestDesignAspectEnum:
    def test_all_members_exist(self):
        members = {m.name for m in DesignAspect}
        assert members == {
            "SCALABILITY", "DATABASE_DESIGN", "API_DESIGN",
            "CACHING_STRATEGY", "LOAD_BALANCING",
        }

    def test_string_values(self):
        assert DesignAspect.SCALABILITY == "scalability"
        assert DesignAspect.DATABASE_DESIGN == "database_design"
        assert DesignAspect.API_DESIGN == "api_design"
        assert DesignAspect.CACHING_STRATEGY == "caching_strategy"
        assert DesignAspect.LOAD_BALANCING == "load_balancing"

    def test_is_str_subclass(self):
        assert isinstance(DesignAspect.SCALABILITY, str)

    def test_deserializes_from_string_value(self):
        assert DesignAspect("scalability") == DesignAspect.SCALABILITY
        assert DesignAspect("load_balancing") == DesignAspect.LOAD_BALANCING


# ===========================================================================
# CodingProblem tests
# ===========================================================================


class TestCodingProblem:
    def test_construction_with_required_fields(self):
        problem = make_coding_problem()
        assert problem.problem_id == UUID1
        assert problem.difficulty == ProblemDifficulty.MEDIUM
        assert problem.topic == ProblemTopic.ARRAYS
        assert problem.asked_at == NOW

    def test_text_is_stored(self):
        problem = make_coding_problem(text="Find the maximum subarray sum.")
        assert problem.text == "Find the maximum subarray sum."

    def test_constraints_is_stored(self):
        problem = make_coding_problem(constraints="1 <= n <= 10^5")
        assert problem.constraints == "1 <= n <= 10^5"

    def test_examples_list_is_stored(self):
        examples = ["Input: [1,2,3] → Output: 6", "Input: [-1,0,1] → Output: 1"]
        problem = make_coding_problem(examples=examples)
        assert len(problem.examples) == 2
        assert problem.examples[0] == "Input: [1,2,3] → Output: 6"

    def test_examples_can_be_empty_list(self):
        problem = make_coding_problem(examples=[])
        assert problem.examples == []

    def test_difficulty_easy(self):
        problem = make_coding_problem(difficulty=ProblemDifficulty.EASY)
        assert problem.difficulty == ProblemDifficulty.EASY

    def test_difficulty_hard(self):
        problem = make_coding_problem(difficulty=ProblemDifficulty.HARD)
        assert problem.difficulty == ProblemDifficulty.HARD

    def test_all_topics_accepted(self):
        for topic in ProblemTopic:
            problem = make_coding_problem(topic=topic)
            assert problem.topic == topic

    def test_missing_problem_id_raises(self):
        with pytest.raises(ValidationError):
            CodingProblem(
                text="Some problem",
                difficulty=ProblemDifficulty.MEDIUM,
                topic=ProblemTopic.ARRAYS,
                constraints="n >= 1",
                examples=[],
                asked_at=NOW,
            )

    def test_missing_text_raises(self):
        with pytest.raises(ValidationError):
            CodingProblem(
                problem_id=UUID1,
                difficulty=ProblemDifficulty.MEDIUM,
                topic=ProblemTopic.ARRAYS,
                constraints="n >= 1",
                examples=[],
                asked_at=NOW,
            )

    def test_missing_asked_at_raises(self):
        with pytest.raises(ValidationError):
            CodingProblem(
                problem_id=UUID1,
                text="Some problem",
                difficulty=ProblemDifficulty.MEDIUM,
                topic=ProblemTopic.ARRAYS,
                constraints="n >= 1",
                examples=[],
            )

    def test_json_round_trip(self):
        """Requirement 18.7: CodingProblem is JSON-serializable."""
        original = make_coding_problem()
        json_str = original.model_dump_json()
        restored = CodingProblem.model_validate_json(json_str)
        assert restored.problem_id == original.problem_id
        assert restored.difficulty == ProblemDifficulty.MEDIUM
        assert restored.topic == ProblemTopic.ARRAYS
        assert restored.examples == original.examples

    def test_json_round_trip_preserves_enum_types(self):
        original = make_coding_problem(
            difficulty=ProblemDifficulty.HARD,
            topic=ProblemTopic.DYNAMIC_PROGRAMMING,
        )
        restored = CodingProblem.model_validate_json(original.model_dump_json())
        assert restored.difficulty == ProblemDifficulty.HARD
        assert restored.topic == ProblemTopic.DYNAMIC_PROGRAMMING


# ===========================================================================
# SystemDesignQuestion tests
# ===========================================================================


class TestSystemDesignQuestion:
    def test_construction_with_required_fields(self):
        question = make_system_design_question()
        assert question.question_id == UUID1
        assert question.text == "Design a URL shortener like bit.ly."
        assert question.system_name == "URL Shortener"
        assert question.asked_at == NOW

    def test_description_is_stored(self):
        question = make_system_design_question(
            description="Design a system that handles 100M daily active users."
        )
        assert "100M" in question.description

    def test_various_system_names(self):
        for system_name in ["Twitter", "Instagram", "Netflix", "Uber", "WhatsApp"]:
            question = make_system_design_question(system_name=system_name)
            assert question.system_name == system_name

    def test_missing_question_id_raises(self):
        with pytest.raises(ValidationError):
            SystemDesignQuestion(
                text="Design Twitter.",
                system_name="Twitter",
                description="A social media platform.",
                asked_at=NOW,
            )

    def test_missing_system_name_raises(self):
        with pytest.raises(ValidationError):
            SystemDesignQuestion(
                question_id=UUID1,
                text="Design something.",
                description="A system.",
                asked_at=NOW,
            )

    def test_missing_asked_at_raises(self):
        with pytest.raises(ValidationError):
            SystemDesignQuestion(
                question_id=UUID1,
                text="Design Twitter.",
                system_name="Twitter",
                description="A social media platform.",
            )

    def test_json_round_trip(self):
        """Requirement 18.7: SystemDesignQuestion is JSON-serializable."""
        original = make_system_design_question()
        restored = SystemDesignQuestion.model_validate_json(original.model_dump_json())
        assert restored.question_id == original.question_id
        assert restored.system_name == original.system_name
        assert restored.description == original.description


# ===========================================================================
# ComplexityAnalysis tests
# ===========================================================================


class TestComplexityAnalysis:
    def test_construction_with_required_fields(self):
        analysis = make_complexity_analysis()
        assert analysis.time_complexity == "O(n)"
        assert analysis.space_complexity == "O(1)"
        assert analysis.is_optimal is True

    def test_optimization_suggestions_defaults_to_none(self):
        analysis = make_complexity_analysis()
        assert analysis.optimization_suggestions is None

    def test_optimization_suggestions_can_be_set(self):
        analysis = make_complexity_analysis(
            optimization_suggestions="Use a hash map to reduce time complexity to O(n)."
        )
        assert analysis.optimization_suggestions is not None
        assert "hash map" in analysis.optimization_suggestions

    def test_is_optimal_false(self):
        analysis = make_complexity_analysis(
            time_complexity="O(n^2)",
            space_complexity="O(n)",
            is_optimal=False,
            optimization_suggestions="Use sorting to achieve O(n log n).",
        )
        assert analysis.is_optimal is False
        assert analysis.time_complexity == "O(n^2)"

    def test_various_big_o_notations(self):
        for notation in ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(2^n)"]:
            analysis = make_complexity_analysis(time_complexity=notation)
            assert analysis.time_complexity == notation

    def test_missing_time_complexity_raises(self):
        with pytest.raises(ValidationError):
            ComplexityAnalysis(
                space_complexity="O(1)",
                is_optimal=True,
            )

    def test_missing_space_complexity_raises(self):
        with pytest.raises(ValidationError):
            ComplexityAnalysis(
                time_complexity="O(n)",
                is_optimal=True,
            )

    def test_missing_is_optimal_raises(self):
        with pytest.raises(ValidationError):
            ComplexityAnalysis(
                time_complexity="O(n)",
                space_complexity="O(1)",
            )

    def test_json_round_trip(self):
        """Requirement 18.7: ComplexityAnalysis is JSON-serializable."""
        original = make_complexity_analysis(
            time_complexity="O(n log n)",
            space_complexity="O(n)",
            is_optimal=False,
            optimization_suggestions="Use in-place sort.",
        )
        restored = ComplexityAnalysis.model_validate_json(original.model_dump_json())
        assert restored.time_complexity == "O(n log n)"
        assert restored.space_complexity == "O(n)"
        assert restored.is_optimal is False
        assert restored.optimization_suggestions == "Use in-place sort."

    def test_json_round_trip_with_none_suggestions(self):
        original = make_complexity_analysis()
        restored = ComplexityAnalysis.model_validate_json(original.model_dump_json())
        assert restored.optimization_suggestions is None


# ===========================================================================
# TechnicalEvaluation tests
# ===========================================================================


class TestTechnicalEvaluationDefaults:
    def test_construction_with_required_fields(self):
        evaluation = make_technical_evaluation()
        assert evaluation.evaluation_id == UUID1
        assert evaluation.question_id == UUID2
        assert evaluation.response_id == UUID3
        assert evaluation.evaluated_at == NOW

    def test_correctness_defaults_to_none(self):
        evaluation = make_technical_evaluation()
        assert evaluation.correctness is None

    def test_complexity_analysis_defaults_to_none(self):
        evaluation = make_technical_evaluation()
        assert evaluation.complexity_analysis is None

    def test_edge_cases_handled_defaults_to_empty_list(self):
        evaluation = make_technical_evaluation()
        assert evaluation.edge_cases_handled == []

    def test_edge_cases_missed_defaults_to_empty_list(self):
        evaluation = make_technical_evaluation()
        assert evaluation.edge_cases_missed == []

    def test_code_quality_notes_defaults_to_none(self):
        evaluation = make_technical_evaluation()
        assert evaluation.code_quality_notes is None

    def test_solution_format_defaults_to_none(self):
        evaluation = make_technical_evaluation()
        assert evaluation.solution_format is None

    def test_design_aspects_evaluated_defaults_to_empty_dict(self):
        evaluation = make_technical_evaluation()
        assert evaluation.design_aspects_evaluated == {}

    def test_design_strengths_defaults_to_empty_list(self):
        evaluation = make_technical_evaluation()
        assert evaluation.design_strengths == []

    def test_design_weaknesses_defaults_to_empty_list(self):
        evaluation = make_technical_evaluation()
        assert evaluation.design_weaknesses == []

    def test_follow_up_warranted_defaults_to_false(self):
        evaluation = make_technical_evaluation()
        assert evaluation.follow_up_warranted is False

    def test_follow_up_text_defaults_to_none(self):
        evaluation = make_technical_evaluation()
        assert evaluation.follow_up_text is None

    def test_difficulty_signal_defaults_to_maintain(self):
        evaluation = make_technical_evaluation()
        assert evaluation.difficulty_signal == "maintain"


class TestTechnicalEvaluationDSAFields:
    def test_correctness_can_be_set_to_correct(self):
        evaluation = make_technical_evaluation(correctness="correct")
        assert evaluation.correctness == "correct"

    def test_correctness_can_be_set_to_incorrect(self):
        evaluation = make_technical_evaluation(correctness="incorrect")
        assert evaluation.correctness == "incorrect"

    def test_correctness_can_be_set_to_partial(self):
        evaluation = make_technical_evaluation(correctness="partial")
        assert evaluation.correctness == "partial"

    def test_complexity_analysis_can_be_set(self):
        analysis = make_complexity_analysis(time_complexity="O(n)", space_complexity="O(1)")
        evaluation = make_technical_evaluation(complexity_analysis=analysis)
        assert evaluation.complexity_analysis is not None
        assert evaluation.complexity_analysis.time_complexity == "O(n)"

    def test_edge_cases_handled_can_be_populated(self):
        evaluation = make_technical_evaluation(
            edge_cases_handled=["empty array", "single element", "all negatives"]
        )
        assert len(evaluation.edge_cases_handled) == 3
        assert "empty array" in evaluation.edge_cases_handled

    def test_edge_cases_missed_can_be_populated(self):
        evaluation = make_technical_evaluation(
            edge_cases_missed=["integer overflow", "duplicate values"]
        )
        assert len(evaluation.edge_cases_missed) == 2

    def test_code_quality_notes_can_be_set(self):
        evaluation = make_technical_evaluation(
            code_quality_notes="Good variable naming, but missing comments."
        )
        assert evaluation.code_quality_notes == "Good variable naming, but missing comments."

    def test_solution_format_can_be_set(self):
        for fmt in SolutionFormat:
            evaluation = make_technical_evaluation(solution_format=fmt)
            assert evaluation.solution_format == fmt

    def test_difficulty_signal_increase(self):
        evaluation = make_technical_evaluation(difficulty_signal="increase")
        assert evaluation.difficulty_signal == "increase"

    def test_difficulty_signal_decrease(self):
        evaluation = make_technical_evaluation(difficulty_signal="decrease")
        assert evaluation.difficulty_signal == "decrease"


class TestTechnicalEvaluationSystemDesignFields:
    def test_design_aspects_evaluated_can_be_populated(self):
        aspects = {
            DesignAspect.SCALABILITY: "Handles 10M users with horizontal scaling.",
            DesignAspect.DATABASE_DESIGN: "Uses PostgreSQL with read replicas.",
        }
        evaluation = make_technical_evaluation(design_aspects_evaluated=aspects)
        assert len(evaluation.design_aspects_evaluated) == 2
        assert DesignAspect.SCALABILITY in evaluation.design_aspects_evaluated

    def test_design_strengths_can_be_populated(self):
        evaluation = make_technical_evaluation(
            design_strengths=["Good use of caching", "Clear API design"]
        )
        assert len(evaluation.design_strengths) == 2

    def test_design_weaknesses_can_be_populated(self):
        evaluation = make_technical_evaluation(
            design_weaknesses=["No mention of load balancing"]
        )
        assert len(evaluation.design_weaknesses) == 1

    def test_follow_up_warranted_can_be_true(self):
        evaluation = make_technical_evaluation(
            follow_up_warranted=True,
            follow_up_text="Can you optimize this further?",
        )
        assert evaluation.follow_up_warranted is True
        assert evaluation.follow_up_text == "Can you optimize this further?"


class TestTechnicalEvaluationValidation:
    def test_missing_evaluation_id_raises(self):
        with pytest.raises(ValidationError):
            TechnicalEvaluation(
                question_id=UUID2,
                response_id=UUID3,
                evaluated_at=NOW,
            )

    def test_missing_question_id_raises(self):
        with pytest.raises(ValidationError):
            TechnicalEvaluation(
                evaluation_id=UUID1,
                response_id=UUID3,
                evaluated_at=NOW,
            )

    def test_missing_response_id_raises(self):
        with pytest.raises(ValidationError):
            TechnicalEvaluation(
                evaluation_id=UUID1,
                question_id=UUID2,
                evaluated_at=NOW,
            )

    def test_missing_evaluated_at_raises(self):
        with pytest.raises(ValidationError):
            TechnicalEvaluation(
                evaluation_id=UUID1,
                question_id=UUID2,
                response_id=UUID3,
            )

    def test_default_lists_are_independent_instances(self):
        """Ensure default_factory is used so lists are not shared between instances."""
        e1 = make_technical_evaluation()
        e2 = make_technical_evaluation()
        e1.edge_cases_handled.append("empty input")
        assert len(e2.edge_cases_handled) == 0

        e1.design_strengths.append("Good scalability")
        assert len(e2.design_strengths) == 0

        e1.design_weaknesses.append("Missing caching")
        assert len(e2.design_weaknesses) == 0


class TestTechnicalEvaluationJsonSerialization:
    def test_minimal_evaluation_round_trips_json(self):
        """Requirement 18.7: TechnicalEvaluation is JSON-serializable."""
        original = make_technical_evaluation()
        restored = TechnicalEvaluation.model_validate_json(original.model_dump_json())
        assert restored.evaluation_id == original.evaluation_id
        assert restored.question_id == original.question_id
        assert restored.response_id == original.response_id
        assert restored.follow_up_warranted is False
        assert restored.difficulty_signal == "maintain"

    def test_full_dsa_evaluation_round_trips_json(self):
        """Requirement 18.7: Full DSA evaluation with nested ComplexityAnalysis serializes."""
        analysis = make_complexity_analysis(
            time_complexity="O(n log n)",
            space_complexity="O(n)",
            is_optimal=False,
            optimization_suggestions="Use heap instead of sort.",
        )
        original = make_technical_evaluation(
            correctness="correct",
            complexity_analysis=analysis,
            edge_cases_handled=["empty array"],
            edge_cases_missed=["overflow"],
            code_quality_notes="Clean code.",
            solution_format=SolutionFormat.CODE,
            follow_up_warranted=True,
            follow_up_text="Can you optimize?",
            difficulty_signal="increase",
        )
        restored = TechnicalEvaluation.model_validate_json(original.model_dump_json())
        assert restored.correctness == "correct"
        assert restored.complexity_analysis is not None
        assert restored.complexity_analysis.time_complexity == "O(n log n)"
        assert restored.complexity_analysis.is_optimal is False
        assert restored.edge_cases_handled == ["empty array"]
        assert restored.edge_cases_missed == ["overflow"]
        assert restored.solution_format == SolutionFormat.CODE
        assert restored.follow_up_warranted is True
        assert restored.difficulty_signal == "increase"

    def test_full_system_design_evaluation_round_trips_json(self):
        """Requirement 18.7: Full System Design evaluation with DesignAspect keys serializes."""
        aspects = {
            DesignAspect.SCALABILITY: "Handles 10M users.",
            DesignAspect.API_DESIGN: "RESTful and well-documented.",
        }
        original = make_technical_evaluation(
            design_aspects_evaluated=aspects,
            design_strengths=["Good caching strategy"],
            design_weaknesses=["No mention of CDN"],
            follow_up_warranted=True,
            follow_up_text="What about load balancing?",
        )
        restored = TechnicalEvaluation.model_validate_json(original.model_dump_json())
        assert len(restored.design_aspects_evaluated) == 2
        assert DesignAspect.SCALABILITY in restored.design_aspects_evaluated
        assert restored.design_strengths == ["Good caching strategy"]
        assert restored.design_weaknesses == ["No mention of CDN"]
        assert restored.follow_up_text == "What about load balancing?"

    def test_none_optional_fields_serialize_as_null(self):
        original = make_technical_evaluation()
        data = json.loads(original.model_dump_json())
        assert data["correctness"] is None
        assert data["complexity_analysis"] is None
        assert data["code_quality_notes"] is None
        assert data["solution_format"] is None
        assert data["follow_up_text"] is None
