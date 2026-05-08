"""Unit tests for domain models and enums.

Covers:
- Enum membership and string values (Stage, Role, EvaluationDimension, QuestionType)
- SessionState default field values and construction
- SessionState.preferred_mode field (Requirements 7.1, 7.2, 7.4)
- FeedbackReport construction with required and optional fields
- Question default skipped=False
- UserResponse default is_off_topic=False
- DimensionScore construction and score constraints
- TranscriptionError and TTSError domain exceptions (Requirements 7.4)
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from interview_practice_partner.domain.enums import (
    DesignAspect,
    EvaluationDimension,
    QuestionType,
    Role,
    Stage,
)
from interview_practice_partner.domain.exceptions import TranscriptionError, TTSError
from interview_practice_partner.domain.models import (
    DimensionScore,
    FeedbackReport,
    Question,
    SessionState,
    UserResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
UUID1 = "11111111-1111-1111-1111-111111111111"
UUID2 = "22222222-2222-2222-2222-222222222222"
UUID3 = "33333333-3333-3333-3333-333333333333"
PHONE = "+15550001234"


def make_dimension_score(
    dimension: EvaluationDimension = EvaluationDimension.COMMUNICATION_CLARITY,
    assessment: str = "Good clarity",
    score: int = 4,
) -> DimensionScore:
    return DimensionScore(dimension=dimension, qualitative_assessment=assessment, score=score)


def make_feedback_report(**overrides) -> FeedbackReport:
    defaults = dict(
        report_id=UUID1,
        session_id=UUID2,
        dimension_scores=[make_dimension_score()],
        strengths=["Clear communication"],
        improvements=["Be more concise"],
        actionable_recommendations=["Practice STAR method"],
        generated_at=NOW,
    )
    defaults.update(overrides)
    return FeedbackReport(**defaults)


def make_session_state(**overrides) -> SessionState:
    defaults = dict(
        session_id=UUID1,
        phone_number=PHONE,
        created_at=NOW,
        updated_at=NOW,
    )
    defaults.update(overrides)
    return SessionState(**defaults)


# ===========================================================================
# Enum tests
# ===========================================================================


class TestStageEnum:
    def test_all_members_exist(self):
        members = {m.name for m in Stage}
        assert members == {"INIT", "ROLE_SELECTION", "ROUND_TYPE_SELECTION", "INTERVIEW", "FEEDBACK", "COMPLETE"}

    def test_string_values(self):
        assert Stage.INIT == "INIT"
        assert Stage.ROLE_SELECTION == "ROLE_SELECTION"
        assert Stage.INTERVIEW == "INTERVIEW"
        assert Stage.FEEDBACK == "FEEDBACK"
        assert Stage.COMPLETE == "COMPLETE"

    def test_is_str_subclass(self):
        assert isinstance(Stage.INIT, str)


class TestRoleEnum:
    def test_all_members_exist(self):
        members = {m.name for m in Role}
        assert members == {
            "SOFTWARE_ENGINEER",
            "SALES_REPRESENTATIVE",
            "RETAIL_ASSOCIATE",
            "UNKNOWN",
        }

    def test_string_values(self):
        assert Role.SOFTWARE_ENGINEER == "software_engineer"
        assert Role.SALES_REPRESENTATIVE == "sales_representative"
        assert Role.RETAIL_ASSOCIATE == "retail_associate"
        assert Role.UNKNOWN == "unknown"

    def test_is_str_subclass(self):
        assert isinstance(Role.UNKNOWN, str)


class TestEvaluationDimensionEnum:
    def test_all_members_exist(self):
        members = {m.name for m in EvaluationDimension}
        assert members == {
            "COMMUNICATION_CLARITY",
            "RELEVANCE",
            "TECHNICAL_KNOWLEDGE",
            "CONFIDENCE",
        }

    def test_string_values(self):
        assert EvaluationDimension.COMMUNICATION_CLARITY == "communication_clarity"
        assert EvaluationDimension.RELEVANCE == "relevance"
        assert EvaluationDimension.TECHNICAL_KNOWLEDGE == "technical_knowledge"
        assert EvaluationDimension.CONFIDENCE == "confidence"

    def test_is_str_subclass(self):
        assert isinstance(EvaluationDimension.CONFIDENCE, str)


class TestQuestionTypeEnum:
    def test_all_members_exist(self):
        members = {m.name for m in QuestionType}
        assert members == {"BEHAVIOURAL", "SITUATIONAL", "TECHNICAL", "FOLLOW_UP"}

    def test_string_values(self):
        assert QuestionType.BEHAVIOURAL == "behavioural"
        assert QuestionType.SITUATIONAL == "situational"
        assert QuestionType.TECHNICAL == "technical"
        assert QuestionType.FOLLOW_UP == "follow_up"

    def test_is_str_subclass(self):
        assert isinstance(QuestionType.BEHAVIOURAL, str)


# ===========================================================================
# Question tests
# ===========================================================================


class TestQuestion:
    def test_construction_with_required_fields(self):
        q = Question(
            question_id=UUID1,
            text="Tell me about yourself.",
            question_type=QuestionType.BEHAVIOURAL,
            asked_at=NOW,
        )
        assert q.question_id == UUID1
        assert q.text == "Tell me about yourself."
        assert q.question_type == QuestionType.BEHAVIOURAL
        assert q.asked_at == NOW

    def test_skipped_defaults_to_false(self):
        q = Question(
            question_id=UUID1,
            text="Describe a challenge.",
            question_type=QuestionType.SITUATIONAL,
            asked_at=NOW,
        )
        assert q.skipped is False

    def test_skipped_can_be_set_to_true(self):
        q = Question(
            question_id=UUID1,
            text="Describe a challenge.",
            question_type=QuestionType.SITUATIONAL,
            asked_at=NOW,
            skipped=True,
        )
        assert q.skipped is True

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            Question(
                question_id=UUID1,
                question_type=QuestionType.BEHAVIOURAL,
                asked_at=NOW,
                # text is missing
            )


# ===========================================================================
# UserResponse tests
# ===========================================================================


class TestUserResponse:
    def test_construction_with_required_fields(self):
        r = UserResponse(
            response_id=UUID1,
            question_id=UUID2,
            text="I worked on a distributed system.",
            word_count=7,
            received_at=NOW,
        )
        assert r.response_id == UUID1
        assert r.question_id == UUID2
        assert r.text == "I worked on a distributed system."
        assert r.word_count == 7
        assert r.received_at == NOW

    def test_is_off_topic_defaults_to_false(self):
        r = UserResponse(
            response_id=UUID1,
            question_id=UUID2,
            text="Some answer.",
            word_count=2,
            received_at=NOW,
        )
        assert r.is_off_topic is False

    def test_is_off_topic_can_be_set_to_true(self):
        r = UserResponse(
            response_id=UUID1,
            question_id=UUID2,
            text="Unrelated answer.",
            word_count=2,
            received_at=NOW,
            is_off_topic=True,
        )
        assert r.is_off_topic is True

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            UserResponse(
                response_id=UUID1,
                question_id=UUID2,
                # text is missing
                word_count=2,
                received_at=NOW,
            )


# ===========================================================================
# DimensionScore tests
# ===========================================================================


class TestDimensionScore:
    def test_construction(self):
        ds = DimensionScore(
            dimension=EvaluationDimension.CONFIDENCE,
            qualitative_assessment="Spoke confidently throughout.",
            score=5,
        )
        assert ds.dimension == EvaluationDimension.CONFIDENCE
        assert ds.qualitative_assessment == "Spoke confidently throughout."
        assert ds.score == 5

    def test_score_minimum_boundary(self):
        ds = DimensionScore(
            dimension=EvaluationDimension.RELEVANCE,
            qualitative_assessment="Barely relevant.",
            score=1,
        )
        assert ds.score == 1

    def test_score_maximum_boundary(self):
        ds = DimensionScore(
            dimension=EvaluationDimension.TECHNICAL_KNOWLEDGE,
            qualitative_assessment="Excellent technical depth.",
            score=5,
        )
        assert ds.score == 5

    def test_score_below_minimum_raises(self):
        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=EvaluationDimension.RELEVANCE,
                qualitative_assessment="Too low.",
                score=0,
            )

    def test_score_above_maximum_raises(self):
        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=EvaluationDimension.RELEVANCE,
                qualitative_assessment="Too high.",
                score=6,
            )


# ===========================================================================
# FeedbackReport tests
# ===========================================================================


class TestFeedbackReport:
    def test_construction_with_all_required_fields(self):
        report = make_feedback_report()
        assert report.report_id == UUID1
        assert report.session_id == UUID2
        assert len(report.dimension_scores) == 1
        assert report.strengths == ["Clear communication"]
        assert report.improvements == ["Be more concise"]
        assert report.actionable_recommendations == ["Practice STAR method"]
        assert report.generated_at == NOW

    def test_off_topic_references_defaults_to_empty_list(self):
        report = make_feedback_report()
        assert report.off_topic_references == []

    def test_off_topic_references_can_be_populated(self):
        report = make_feedback_report(
            off_topic_references=["Response 1 was about cooking", "Response 3 was unrelated"]
        )
        assert len(report.off_topic_references) == 2
        assert "Response 1 was about cooking" in report.off_topic_references

    def test_multiple_dimension_scores(self):
        scores = [
            make_dimension_score(EvaluationDimension.COMMUNICATION_CLARITY, "Clear", 4),
            make_dimension_score(EvaluationDimension.CONFIDENCE, "Confident", 5),
            make_dimension_score(EvaluationDimension.RELEVANCE, "On topic", 3),
        ]
        report = make_feedback_report(dimension_scores=scores)
        assert len(report.dimension_scores) == 3

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            FeedbackReport(
                # report_id missing
                session_id=UUID2,
                dimension_scores=[make_dimension_score()],
                strengths=["Good"],
                improvements=["Improve X"],
                actionable_recommendations=["Do Y"],
                generated_at=NOW,
            )

    def test_empty_strengths_raises(self):
        with pytest.raises(ValidationError):
            make_feedback_report(strengths=[])

    def test_empty_improvements_raises(self):
        with pytest.raises(ValidationError):
            make_feedback_report(improvements=[])

    def test_empty_actionable_recommendations_raises(self):
        with pytest.raises(ValidationError):
            make_feedback_report(actionable_recommendations=[])


# ===========================================================================
# SessionState tests
# ===========================================================================


class TestSessionState:
    def test_construction_with_required_fields(self):
        session = make_session_state()
        assert session.session_id == UUID1
        assert session.phone_number == PHONE
        assert session.created_at == NOW
        assert session.updated_at == NOW

    def test_stage_defaults_to_init(self):
        session = make_session_state()
        assert session.stage == Stage.INIT

    def test_role_defaults_to_unknown(self):
        session = make_session_state()
        assert session.role == Role.UNKNOWN

    def test_questions_defaults_to_empty_list(self):
        session = make_session_state()
        assert session.questions == []

    def test_responses_defaults_to_empty_list(self):
        session = make_session_state()
        assert session.responses == []

    def test_off_topic_count_defaults_to_zero(self):
        session = make_session_state()
        assert session.off_topic_count == 0

    def test_consecutive_out_of_scope_count_defaults_to_zero(self):
        session = make_session_state()
        assert session.consecutive_out_of_scope_count == 0

    def test_clarification_turn_count_defaults_to_zero(self):
        session = make_session_state()
        assert session.clarification_turn_count == 0

    def test_requested_short_session_defaults_to_false(self):
        session = make_session_state()
        assert session.requested_short_session is False

    def test_is_complete_defaults_to_false(self):
        session = make_session_state()
        assert session.is_complete is False

    def test_feedback_report_defaults_to_none(self):
        session = make_session_state()
        assert session.feedback_report is None

    def test_completed_at_defaults_to_none(self):
        session = make_session_state()
        assert session.completed_at is None

    def test_context_summary_defaults_to_none(self):
        session = make_session_state()
        assert session.context_summary is None

    def test_stage_can_be_set(self):
        session = make_session_state(stage=Stage.INTERVIEW)
        assert session.stage == Stage.INTERVIEW

    def test_role_can_be_set(self):
        session = make_session_state(role=Role.SOFTWARE_ENGINEER)
        assert session.role == Role.SOFTWARE_ENGINEER

    def test_with_feedback_report(self):
        report = make_feedback_report(report_id=UUID3, session_id=UUID1)
        session = make_session_state(feedback_report=report, is_complete=True)
        assert session.feedback_report is not None
        assert session.feedback_report.report_id == UUID3
        assert session.is_complete is True

    def test_with_questions_and_responses(self):
        question = Question(
            question_id=UUID2,
            text="Tell me about yourself.",
            question_type=QuestionType.BEHAVIOURAL,
            asked_at=NOW,
        )
        response = UserResponse(
            response_id=UUID3,
            question_id=UUID2,
            text="I am a software engineer.",
            word_count=5,
            received_at=NOW,
        )
        session = make_session_state(questions=[question], responses=[response])
        assert len(session.questions) == 1
        assert len(session.responses) == 1

    def test_missing_session_id_raises(self):
        with pytest.raises(ValidationError):
            SessionState(
                # session_id missing
                phone_number=PHONE,
                created_at=NOW,
                updated_at=NOW,
            )

    def test_missing_phone_number_raises(self):
        with pytest.raises(ValidationError):
            SessionState(
                session_id=UUID1,
                # phone_number missing
                created_at=NOW,
                updated_at=NOW,
            )

    def test_missing_created_at_raises(self):
        with pytest.raises(ValidationError):
            SessionState(
                session_id=UUID1,
                phone_number=PHONE,
                # created_at missing
                updated_at=NOW,
            )

    def test_missing_updated_at_raises(self):
        with pytest.raises(ValidationError):
            SessionState(
                session_id=UUID1,
                phone_number=PHONE,
                created_at=NOW,
                # updated_at missing
            )

    def test_default_lists_are_independent_instances(self):
        """Ensure default_factory is used so lists are not shared between instances."""
        s1 = make_session_state()
        s2 = make_session_state()
        s1.questions.append(
            Question(
                question_id=UUID2,
                text="Q?",
                question_type=QuestionType.TECHNICAL,
                asked_at=NOW,
            )
        )
        assert len(s2.questions) == 0


# ===========================================================================
# SessionState.preferred_mode tests  (Requirements 7.1, 7.2, 7.4)
# ===========================================================================


class TestSessionStatePreferredMode:
    """Tests for the preferred_mode field added in the voice-note-support feature."""

    def test_preferred_mode_defaults_to_text(self):
        """Requirement 7.1 / 7.3: new sessions default to 'text' mode."""
        session = make_session_state()
        assert session.preferred_mode == "text"

    def test_preferred_mode_can_be_set_to_voice(self):
        """Requirement 7.1: the field accepts 'voice' as a valid value."""
        session = make_session_state(preferred_mode="voice")
        assert session.preferred_mode == "voice"

    def test_preferred_mode_can_be_set_to_text_explicitly(self):
        """Requirement 7.1: the field accepts 'text' as a valid value."""
        session = make_session_state(preferred_mode="text")
        assert session.preferred_mode == "text"

    def test_preferred_mode_invalid_value_raises(self):
        """Requirement 7.1: only 'voice' and 'text' are valid values."""
        with pytest.raises(ValidationError):
            make_session_state(preferred_mode="audio")  # type: ignore[arg-type]

    def test_preferred_mode_voice_round_trips_json(self):
        """Requirement 7.2: preferred_mode='voice' survives model_dump_json / model_validate_json."""
        original = make_session_state(preferred_mode="voice")
        json_str = original.model_dump_json()
        restored = SessionState.model_validate_json(json_str)
        assert restored.preferred_mode == "voice"

    def test_preferred_mode_text_round_trips_json(self):
        """Requirement 7.2: preferred_mode='text' survives model_dump_json / model_validate_json."""
        original = make_session_state(preferred_mode="text")
        json_str = original.model_dump_json()
        restored = SessionState.model_validate_json(json_str)
        assert restored.preferred_mode == "text"

    def test_preferred_mode_absent_in_json_defaults_to_text(self):
        """Requirement 7.2 / 7.3: existing Redis payloads without preferred_mode deserialise
        correctly — Pydantic applies the default value of 'text'."""
        session = make_session_state()
        data = session.model_dump()
        # Simulate an old Redis payload that has no preferred_mode key
        data.pop("preferred_mode", None)
        # Re-serialise to JSON without the field
        import json
        json_str = json.dumps(data, default=str)
        restored = SessionState.model_validate_json(json_str)
        assert restored.preferred_mode == "text"

    def test_round_trip_preserves_all_other_fields(self):
        """Requirement 7.2: JSON round-trip does not corrupt other SessionState fields."""
        original = make_session_state(preferred_mode="voice", stage=Stage.INTERVIEW)
        restored = SessionState.model_validate_json(original.model_dump_json())
        assert restored.session_id == original.session_id
        assert restored.phone_number == original.phone_number
        assert restored.stage == Stage.INTERVIEW
        assert restored.preferred_mode == "voice"


# ===========================================================================
# SessionState technical round fields tests (Requirements 11.1, 11.2, 18.1-18.7)
# ===========================================================================


class TestSessionStateTechnicalRoundFields:
    """Tests for technical round fields added in the technical-interview-rounds feature."""

    def test_interview_round_type_defaults_to_none(self):
        """Requirement 18.1: interview_round_type defaults to None."""
        session = make_session_state()
        assert session.interview_round_type is None

    def test_interview_round_type_can_be_set(self):
        """Requirement 18.1: interview_round_type can be set to InterviewRoundType enum."""
        from interview_practice_partner.domain.enums import InterviewRoundType

        session = make_session_state(interview_round_type=InterviewRoundType.DSA_CODING)
        assert session.interview_round_type == InterviewRoundType.DSA_CODING

    def test_problem_difficulty_defaults_to_medium(self):
        """Requirement 18.2: problem_difficulty defaults to MEDIUM."""
        from interview_practice_partner.domain.enums import ProblemDifficulty

        session = make_session_state()
        assert session.problem_difficulty == ProblemDifficulty.MEDIUM

    def test_problem_difficulty_can_be_set(self):
        """Requirement 18.2: problem_difficulty can be set to ProblemDifficulty enum."""
        from interview_practice_partner.domain.enums import ProblemDifficulty

        session = make_session_state(problem_difficulty=ProblemDifficulty.HARD)
        assert session.problem_difficulty == ProblemDifficulty.HARD

    def test_design_phase_defaults_to_none(self):
        """Requirement 18.3: design_phase defaults to None."""
        session = make_session_state()
        assert session.design_phase is None

    def test_design_phase_can_be_set(self):
        """Requirement 18.3: design_phase can be set to DesignPhase enum."""
        from interview_practice_partner.domain.enums import DesignPhase

        session = make_session_state(design_phase=DesignPhase.HIGH_LEVEL_DESIGN)
        assert session.design_phase == DesignPhase.HIGH_LEVEL_DESIGN

    def test_topics_covered_defaults_to_empty_list(self):
        """Requirement 18.4: topics_covered defaults to empty list."""
        session = make_session_state()
        assert session.topics_covered == []

    def test_topics_covered_can_be_populated(self):
        """Requirement 18.4: topics_covered can contain ProblemTopic enums."""
        from interview_practice_partner.domain.enums import ProblemTopic

        session = make_session_state(
            topics_covered=[ProblemTopic.ARRAYS, ProblemTopic.TREES]
        )
        assert len(session.topics_covered) == 2
        assert ProblemTopic.ARRAYS in session.topics_covered
        assert ProblemTopic.TREES in session.topics_covered

    def test_design_aspects_covered_defaults_to_empty_list(self):
        """Requirement 18.5: design_aspects_covered defaults to empty list."""
        session = make_session_state()
        assert session.design_aspects_covered == []

    def test_design_aspects_covered_can_be_populated(self):
        """Requirement 18.5: design_aspects_covered can contain DesignAspect enums."""
        from interview_practice_partner.domain.enums import DesignAspect

        session = make_session_state(
            design_aspects_covered=[DesignAspect.SCALABILITY, DesignAspect.DATABASE_DESIGN]
        )
        assert len(session.design_aspects_covered) == 2
        assert DesignAspect.SCALABILITY in session.design_aspects_covered
        assert DesignAspect.DATABASE_DESIGN in session.design_aspects_covered

    def test_difficulty_adjustment_history_defaults_to_empty_list(self):
        """Requirement 18.6: difficulty_adjustment_history defaults to empty list."""
        session = make_session_state()
        assert session.difficulty_adjustment_history == []

    def test_difficulty_adjustment_history_can_be_populated(self):
        """Requirement 18.6: difficulty_adjustment_history can contain dicts."""
        history = [
            {"from": "medium", "to": "hard", "reason": "optimal solution"},
            {"from": "hard", "to": "medium", "reason": "struggled with problem"},
        ]
        session = make_session_state(difficulty_adjustment_history=history)
        assert len(session.difficulty_adjustment_history) == 2
        assert session.difficulty_adjustment_history[0]["from"] == "medium"

    def test_technical_fields_round_trip_json(self):
        """Requirement 18.7: All technical fields are JSON-serializable."""
        from interview_practice_partner.domain.enums import (
            DesignAspect,
            DesignPhase,
            InterviewRoundType,
            ProblemDifficulty,
            ProblemTopic,
        )

        original = make_session_state(
            interview_round_type=InterviewRoundType.DSA_CODING,
            problem_difficulty=ProblemDifficulty.HARD,
            design_phase=DesignPhase.DEEP_DIVE,
            topics_covered=[ProblemTopic.ARRAYS, ProblemTopic.GRAPHS],
            design_aspects_covered=[DesignAspect.SCALABILITY],
            difficulty_adjustment_history=[{"from": "medium", "to": "hard"}],
        )
        json_str = original.model_dump_json()
        restored = SessionState.model_validate_json(json_str)

        assert restored.interview_round_type == InterviewRoundType.DSA_CODING
        assert restored.problem_difficulty == ProblemDifficulty.HARD
        assert restored.design_phase == DesignPhase.DEEP_DIVE
        assert len(restored.topics_covered) == 2
        assert ProblemTopic.ARRAYS in restored.topics_covered
        assert len(restored.design_aspects_covered) == 1
        assert len(restored.difficulty_adjustment_history) == 1

    def test_technical_fields_absent_in_json_use_defaults(self):
        """Requirement 18.7: Existing Redis payloads without technical fields deserialize correctly."""
        session = make_session_state()
        data = session.model_dump()
        # Simulate an old Redis payload without technical fields
        data.pop("interview_round_type", None)
        data.pop("problem_difficulty", None)
        data.pop("design_phase", None)
        data.pop("topics_covered", None)
        data.pop("design_aspects_covered", None)
        data.pop("difficulty_adjustment_history", None)

        import json
        json_str = json.dumps(data, default=str)
        restored = SessionState.model_validate_json(json_str)

        # Verify defaults are applied
        assert restored.interview_round_type is None
        from interview_practice_partner.domain.enums import ProblemDifficulty
        assert restored.problem_difficulty == ProblemDifficulty.MEDIUM
        assert restored.design_phase is None
        assert restored.topics_covered == []
        assert restored.design_aspects_covered == []
        assert restored.difficulty_adjustment_history == []

    def test_default_lists_are_independent_instances(self):
        """Ensure default_factory is used so lists are not shared between instances."""
        from interview_practice_partner.domain.enums import ProblemTopic

        s1 = make_session_state()
        s2 = make_session_state()
        s1.topics_covered.append(ProblemTopic.ARRAYS)
        assert len(s2.topics_covered) == 0

        s1.design_aspects_covered.append(DesignAspect.SCALABILITY)
        assert len(s2.design_aspects_covered) == 0

        s1.difficulty_adjustment_history.append({"test": "data"})
        assert len(s2.difficulty_adjustment_history) == 0


# ===========================================================================
# TranscriptionError and TTSError tests  (Requirements 7.4 / 10.1, 10.2)
# ===========================================================================


class TestTranscriptionError:
    """TranscriptionError must be an Exception subclass (Requirement 10.1)."""

    def test_is_exception_subclass(self):
        assert issubclass(TranscriptionError, Exception)

    def test_can_be_raised_and_caught_as_exception(self):
        with pytest.raises(Exception):
            raise TranscriptionError()

    def test_can_be_raised_and_caught_as_transcription_error(self):
        with pytest.raises(TranscriptionError):
            raise TranscriptionError()

    def test_default_message(self):
        err = TranscriptionError()
        assert str(err) == "Voice note transcription failed"

    def test_custom_message(self):
        err = TranscriptionError("Groq API returned 503")
        assert str(err) == "Groq API returned 503"

    def test_not_caught_as_tts_error(self):
        """TranscriptionError and TTSError are distinct exception types."""
        with pytest.raises(TranscriptionError):
            try:
                raise TranscriptionError("transcription failed")
            except TTSError:
                pass  # should NOT be caught here


class TestTTSError:
    """TTSError must be an Exception subclass (Requirement 10.2)."""

    def test_is_exception_subclass(self):
        assert issubclass(TTSError, Exception)

    def test_can_be_raised_and_caught_as_exception(self):
        with pytest.raises(Exception):
            raise TTSError()

    def test_can_be_raised_and_caught_as_tts_error(self):
        with pytest.raises(TTSError):
            raise TTSError()

    def test_default_message(self):
        err = TTSError()
        assert str(err) == "Text-to-speech synthesis failed"

    def test_custom_message(self):
        err = TTSError("ElevenLabs API returned 429")
        assert str(err) == "ElevenLabs API returned 429"

    def test_not_caught_as_transcription_error(self):
        """TTSError and TranscriptionError are distinct exception types."""
        with pytest.raises(TTSError):
            try:
                raise TTSError("tts failed")
            except TranscriptionError:
                pass  # should NOT be caught here
