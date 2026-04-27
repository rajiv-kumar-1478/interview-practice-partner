"""Unit tests for PromptBuilder.

Covers:
- Each stage produces a non-empty messages list with a system message
- Role selection prompt handles clarification turns
- Question generation prompt includes previously asked questions
- Question generation prompt includes difficulty signal when provided
- Response evaluation prompt structure and JSON schema instruction
- Feedback prompt includes session transcript and off-topic notes
- No HTML tags or markdown headers in any prompt
"""

import re
from datetime import datetime, timezone

import pytest

from interview_practice_partner.domain.enums import (
    EvaluationDimension,
    QuestionType,
    Role,
    Stage,
)
from interview_practice_partner.domain.models import (
    DimensionScore,
    FeedbackReport,
    Question,
    SessionState,
    UserResponse,
)
from interview_practice_partner.llm.prompt_builder import PromptBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
UUID1 = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
UUID2 = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
UUID3 = "cccccccc-cccc-cccc-cccc-cccccccccccc"
PHONE = "+15550001234"

HTML_TAG_PATTERN = re.compile(r"<[a-zA-Z][^>]*>")
MARKDOWN_HEADER_PATTERN = re.compile(r"^#{1,6}\s", re.MULTILINE)
FENCED_CODE_BLOCK_PATTERN = re.compile(r"```")


def make_session(
    role: Role = Role.SOFTWARE_ENGINEER,
    stage: Stage = Stage.INTERVIEW,
    questions: list[Question] | None = None,
    responses: list[UserResponse] | None = None,
    off_topic_count: int = 0,
) -> SessionState:
    return SessionState(
        session_id=UUID1,
        phone_number=PHONE,
        stage=stage,
        role=role,
        questions=questions or [],
        responses=responses or [],
        off_topic_count=off_topic_count,
        created_at=NOW,
        updated_at=NOW,
    )


def make_question(
    question_id: str = UUID2,
    text: str = "Tell me about a challenging project.",
    question_type: QuestionType = QuestionType.BEHAVIOURAL,
    skipped: bool = False,
) -> Question:
    return Question(
        question_id=question_id,
        text=text,
        question_type=question_type,
        asked_at=NOW,
        skipped=skipped,
    )


def make_response(
    response_id: str = UUID3,
    question_id: str = UUID2,
    text: str = "I led a team to migrate a monolith to microservices.",
    word_count: int = 10,
    is_off_topic: bool = False,
) -> UserResponse:
    return UserResponse(
        response_id=response_id,
        question_id=question_id,
        text=text,
        word_count=word_count,
        is_off_topic=is_off_topic,
        received_at=NOW,
    )


def assert_no_forbidden_formatting(messages: list[dict[str, str]]) -> None:
    """Assert that no message contains HTML tags, markdown headers, or fenced code blocks."""
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder()


# ===========================================================================
# build_role_selection_prompt
# ===========================================================================


class TestBuildRoleSelectionPrompt:
    def test_returns_non_empty_list(self, builder: PromptBuilder):
        messages = builder.build_role_selection_prompt("I want to practise for a software engineer role.")
        assert len(messages) > 0

    def test_first_message_is_system(self, builder: PromptBuilder):
        messages = builder.build_role_selection_prompt("Hello")
        assert messages[0]["role"] == "system"

    def test_last_message_is_user(self, builder: PromptBuilder):
        messages = builder.build_role_selection_prompt("I want to be a sales rep.")
        assert messages[-1]["role"] == "user"

    def test_user_message_is_preserved(self, builder: PromptBuilder):
        user_msg = "I want to practise for a retail associate position."
        messages = builder.build_role_selection_prompt(user_msg)
        assert messages[-1]["content"] == user_msg

    def test_system_prompt_mentions_supported_roles(self, builder: PromptBuilder):
        messages = builder.build_role_selection_prompt("Hello")
        system_content = messages[0]["content"]
        assert "Software Engineer" in system_content
        assert "Sales Representative" in system_content
        assert "Retail Associate" in system_content

    def test_system_prompt_requests_json_output(self, builder: PromptBuilder):
        messages = builder.build_role_selection_prompt("Hello")
        system_content = messages[0]["content"]
        assert "JSON" in system_content

    def test_clarification_turn_count_zero_asks_for_clarification(self, builder: PromptBuilder):
        messages = builder.build_role_selection_prompt("I'm not sure", clarification_turn_count=0)
        system_content = messages[0]["content"]
        # Should NOT mention defaulting to general format yet
        assert "general interview format" not in system_content.lower() or "clarify" in system_content.lower()

    def test_clarification_turn_count_two_triggers_fallback(self, builder: PromptBuilder):
        messages = builder.build_role_selection_prompt("Still not sure", clarification_turn_count=2)
        system_content = messages[0]["content"]
        assert "general interview format" in system_content.lower()

    def test_no_forbidden_formatting(self, builder: PromptBuilder):
        messages = builder.build_role_selection_prompt("I want to be a software engineer.")
        assert_no_forbidden_formatting(messages)


# ===========================================================================
# build_question_generation_prompt
# ===========================================================================


class TestBuildQuestionGenerationPrompt:
    def test_returns_non_empty_list(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_question_generation_prompt(session, QuestionType.BEHAVIOURAL)
        assert len(messages) > 0

    def test_first_message_is_system(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_question_generation_prompt(session, QuestionType.TECHNICAL)
        assert messages[0]["role"] == "system"

    def test_last_message_is_user(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_question_generation_prompt(session, QuestionType.SITUATIONAL)
        assert messages[-1]["role"] == "user"

    def test_system_prompt_includes_role_name(self, builder: PromptBuilder):
        session = make_session(role=Role.SALES_REPRESENTATIVE)
        messages = builder.build_question_generation_prompt(session, QuestionType.BEHAVIOURAL)
        system_content = messages[0]["content"]
        assert "Sales Representative" in system_content

    def test_system_prompt_includes_question_type(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_question_generation_prompt(session, QuestionType.TECHNICAL)
        system_content = messages[0]["content"]
        assert "technical" in system_content.lower()

    def test_system_prompt_includes_previously_asked_questions(self, builder: PromptBuilder):
        q1 = make_question(UUID2, "Tell me about a challenging project.")
        q2 = make_question(UUID3, "Describe your experience with Python.", QuestionType.TECHNICAL)
        session = make_session(questions=[q1, q2])
        messages = builder.build_question_generation_prompt(session, QuestionType.BEHAVIOURAL)
        system_content = messages[0]["content"]
        assert "Tell me about a challenging project." in system_content
        assert "Describe your experience with Python." in system_content

    def test_skipped_questions_excluded_from_deduplication_list(self, builder: PromptBuilder):
        q1 = make_question(UUID2, "Tell me about yourself.", skipped=True)
        session = make_session(questions=[q1])
        messages = builder.build_question_generation_prompt(session, QuestionType.BEHAVIOURAL)
        system_content = messages[0]["content"]
        # Skipped questions should not appear in the "already asked" list
        assert "Tell me about yourself." not in system_content

    def test_no_questions_asked_yet_message(self, builder: PromptBuilder):
        session = make_session(questions=[])
        messages = builder.build_question_generation_prompt(session, QuestionType.BEHAVIOURAL)
        system_content = messages[0]["content"]
        assert "No questions have been asked yet" in system_content

    def test_difficulty_signal_increase(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_question_generation_prompt(
            session, QuestionType.TECHNICAL, difficulty_signal="increase"
        )
        system_content = messages[0]["content"]
        assert "increase" in system_content.lower() or "Increase" in system_content
        assert "strong knowledge" in system_content.lower() or "deeper" in system_content.lower()

    def test_difficulty_signal_decrease(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_question_generation_prompt(
            session, QuestionType.BEHAVIOURAL, difficulty_signal="decrease"
        )
        system_content = messages[0]["content"]
        assert "decrease" in system_content.lower() or "Decrease" in system_content
        assert "foundational" in system_content.lower() or "weaker" in system_content.lower()

    def test_difficulty_signal_maintain(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_question_generation_prompt(
            session, QuestionType.SITUATIONAL, difficulty_signal="maintain"
        )
        system_content = messages[0]["content"]
        assert "maintain" in system_content.lower() or "Maintain" in system_content

    def test_no_difficulty_signal_defaults_to_maintain(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_question_generation_prompt(
            session, QuestionType.SITUATIONAL, difficulty_signal=None
        )
        system_content = messages[0]["content"]
        assert "maintain" in system_content.lower() or "Maintain" in system_content

    def test_no_forbidden_formatting(self, builder: PromptBuilder):
        session = make_session(questions=[make_question()])
        messages = builder.build_question_generation_prompt(session, QuestionType.TECHNICAL)
        assert_no_forbidden_formatting(messages)

    def test_all_question_types_produce_valid_prompts(self, builder: PromptBuilder):
        session = make_session()
        for q_type in QuestionType:
            messages = builder.build_question_generation_prompt(session, q_type)
            assert len(messages) >= 2
            assert messages[0]["role"] == "system"


# ===========================================================================
# build_response_evaluation_prompt
# ===========================================================================


class TestBuildResponseEvaluationPrompt:
    def test_returns_non_empty_list(self, builder: PromptBuilder):
        session = make_session()
        question = make_question()
        response = make_response()
        messages = builder.build_response_evaluation_prompt(question, response, session)
        assert len(messages) > 0

    def test_first_message_is_system(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_response_evaluation_prompt(
            make_question(), make_response(), session
        )
        assert messages[0]["role"] == "system"

    def test_last_message_is_user(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_response_evaluation_prompt(
            make_question(), make_response(), session
        )
        assert messages[-1]["role"] == "user"

    def test_system_prompt_mentions_is_off_topic(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_response_evaluation_prompt(
            make_question(), make_response(), session
        )
        system_content = messages[0]["content"]
        assert "is_off_topic" in system_content

    def test_system_prompt_mentions_is_short(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_response_evaluation_prompt(
            make_question(), make_response(), session
        )
        system_content = messages[0]["content"]
        assert "is_short" in system_content

    def test_system_prompt_mentions_follow_up_warranted(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_response_evaluation_prompt(
            make_question(), make_response(), session
        )
        system_content = messages[0]["content"]
        assert "follow_up_warranted" in system_content

    def test_system_prompt_mentions_follow_up_text(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_response_evaluation_prompt(
            make_question(), make_response(), session
        )
        system_content = messages[0]["content"]
        assert "follow_up_text" in system_content

    def test_system_prompt_mentions_difficulty_signal(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_response_evaluation_prompt(
            make_question(), make_response(), session
        )
        system_content = messages[0]["content"]
        assert "difficulty_signal" in system_content

    def test_system_prompt_mentions_increase_maintain_decrease(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_response_evaluation_prompt(
            make_question(), make_response(), session
        )
        system_content = messages[0]["content"]
        assert "increase" in system_content
        assert "maintain" in system_content
        assert "decrease" in system_content

    def test_user_message_includes_question_text(self, builder: PromptBuilder):
        session = make_session()
        question = make_question(text="What is your greatest strength?")
        messages = builder.build_response_evaluation_prompt(
            question, make_response(), session
        )
        user_content = messages[-1]["content"]
        assert "What is your greatest strength?" in user_content

    def test_user_message_includes_response_text(self, builder: PromptBuilder):
        session = make_session()
        response = make_response(text="My greatest strength is problem solving.")
        messages = builder.build_response_evaluation_prompt(
            make_question(), response, session
        )
        user_content = messages[-1]["content"]
        assert "My greatest strength is problem solving." in user_content

    def test_system_prompt_includes_role_context(self, builder: PromptBuilder):
        session = make_session(role=Role.RETAIL_ASSOCIATE)
        messages = builder.build_response_evaluation_prompt(
            make_question(), make_response(), session
        )
        system_content = messages[0]["content"]
        assert "Retail Associate" in system_content

    def test_system_prompt_requests_json_only(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_response_evaluation_prompt(
            make_question(), make_response(), session
        )
        system_content = messages[0]["content"]
        assert "JSON" in system_content

    def test_system_prompt_mentions_15_words_threshold(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_response_evaluation_prompt(
            make_question(), make_response(), session
        )
        system_content = messages[0]["content"]
        assert "15" in system_content

    def test_no_forbidden_formatting(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_response_evaluation_prompt(
            make_question(), make_response(), session
        )
        assert_no_forbidden_formatting(messages)


# ===========================================================================
# build_feedback_prompt
# ===========================================================================


class TestBuildFeedbackPrompt:
    def test_returns_non_empty_list(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_feedback_prompt(session)
        assert len(messages) > 0

    def test_first_message_is_system(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_feedback_prompt(session)
        assert messages[0]["role"] == "system"

    def test_last_message_is_user(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_feedback_prompt(session)
        assert messages[-1]["role"] == "user"

    def test_system_prompt_mentions_all_evaluation_dimensions(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "communication_clarity" in system_content
        assert "relevance" in system_content
        assert "technical_knowledge" in system_content
        assert "confidence" in system_content

    def test_system_prompt_mentions_dimension_scores(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "dimension_scores" in system_content

    def test_system_prompt_mentions_strengths(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "strengths" in system_content

    def test_system_prompt_mentions_improvements(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "improvements" in system_content

    def test_system_prompt_mentions_actionable_recommendations(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "actionable_recommendations" in system_content

    def test_system_prompt_mentions_off_topic_references(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "off_topic_references" in system_content

    def test_user_message_includes_transcript_with_question_text(self, builder: PromptBuilder):
        question = make_question(text="Describe a time you handled a difficult customer.")
        response = make_response(question_id=UUID2, text="I once helped an angry customer by listening carefully.")
        session = make_session(questions=[question], responses=[response])
        messages = builder.build_feedback_prompt(session)
        user_content = messages[-1]["content"]
        assert "Describe a time you handled a difficult customer." in user_content

    def test_user_message_includes_transcript_with_response_text(self, builder: PromptBuilder):
        question = make_question()
        response = make_response(text="I led a cross-functional team to deliver the project on time.")
        session = make_session(questions=[question], responses=[response])
        messages = builder.build_feedback_prompt(session)
        user_content = messages[-1]["content"]
        assert "I led a cross-functional team to deliver the project on time." in user_content

    def test_skipped_question_marked_in_transcript(self, builder: PromptBuilder):
        question = make_question(skipped=True)
        session = make_session(questions=[question])
        messages = builder.build_feedback_prompt(session)
        user_content = messages[-1]["content"]
        assert "SKIPPED" in user_content

    def test_off_topic_response_flagged_in_transcript(self, builder: PromptBuilder):
        question = make_question()
        response = make_response(is_off_topic=True, text="I like cooking pasta.")
        session = make_session(questions=[question], responses=[response], off_topic_count=1)
        messages = builder.build_feedback_prompt(session)
        user_content = messages[-1]["content"]
        assert "OFF-TOPIC" in user_content

    def test_high_off_topic_count_triggers_focus_note(self, builder: PromptBuilder):
        session = make_session(off_topic_count=3)
        messages = builder.build_feedback_prompt(session)
        system_content = messages[0]["content"]
        # Should mention focus/relevance when off_topic_count > 2
        assert "focus" in system_content.lower() or "relevance" in system_content.lower()

    def test_zero_off_topic_count_no_off_topic_note(self, builder: PromptBuilder):
        session = make_session(off_topic_count=0)
        messages = builder.build_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "did not go off-topic" in system_content.lower()

    def test_system_prompt_includes_role_name(self, builder: PromptBuilder):
        session = make_session(role=Role.SALES_REPRESENTATIVE)
        messages = builder.build_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "Sales Representative" in system_content

    def test_system_prompt_requests_json_only(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "JSON" in system_content

    def test_system_prompt_requires_at_least_one_strength(self, builder: PromptBuilder):
        session = make_session()
        messages = builder.build_feedback_prompt(session)
        system_content = messages[0]["content"]
        assert "at least one" in system_content.lower()

    def test_no_forbidden_formatting(self, builder: PromptBuilder):
        question = make_question()
        response = make_response()
        session = make_session(questions=[question], responses=[response])
        messages = builder.build_feedback_prompt(session)
        assert_no_forbidden_formatting(messages)

    def test_multiple_questions_all_appear_in_transcript(self, builder: PromptBuilder):
        q1 = make_question(UUID2, "Tell me about yourself.", QuestionType.BEHAVIOURAL)
        q2 = make_question(UUID3, "How do you handle pressure?", QuestionType.SITUATIONAL)
        r1 = make_response(UUID3, UUID2, "I am a software engineer with 5 years of experience.", 10)
        session = make_session(questions=[q1, q2], responses=[r1])
        messages = builder.build_feedback_prompt(session)
        user_content = messages[-1]["content"]
        assert "Tell me about yourself." in user_content
        assert "How do you handle pressure?" in user_content


# ===========================================================================
# build_round_type_selection_prompt
# ===========================================================================


class TestBuildRoundTypeSelectionPrompt:
    def test_returns_non_empty_list(self, builder: PromptBuilder):
        messages = builder.build_round_type_selection_prompt("I want to practice DSA")
        assert len(messages) > 0

    def test_first_message_is_system(self, builder: PromptBuilder):
        messages = builder.build_round_type_selection_prompt("coding round")
        assert messages[0]["role"] == "system"

    def test_last_message_is_user(self, builder: PromptBuilder):
        messages = builder.build_round_type_selection_prompt("system design")
        assert messages[-1]["role"] == "user"

    def test_user_message_is_preserved(self, builder: PromptBuilder):
        user_msg = "I want to practice DSA problems"
        messages = builder.build_round_type_selection_prompt(user_msg)
        assert messages[-1]["content"] == user_msg

    def test_system_prompt_presents_three_options(self, builder: PromptBuilder):
        messages = builder.build_round_type_selection_prompt("Hello")
        system_content = messages[0]["content"]
        assert "DSA/Coding Round" in system_content
        assert "System Design Round" in system_content
        assert "Behavioral Round" in system_content

    def test_system_prompt_describes_dsa_option(self, builder: PromptBuilder):
        messages = builder.build_round_type_selection_prompt("Hello")
        system_content = messages[0]["content"]
        assert "algorithmic problem-solving" in system_content.lower()

    def test_system_prompt_describes_system_design_option(self, builder: PromptBuilder):
        messages = builder.build_round_type_selection_prompt("Hello")
        system_content = messages[0]["content"]
        assert "architectural design" in system_content.lower()

    def test_system_prompt_describes_behavioral_option(self, builder: PromptBuilder):
        messages = builder.build_round_type_selection_prompt("Hello")
        system_content = messages[0]["content"]
        assert "soft skills" in system_content.lower() or "experience questions" in system_content.lower()

    def test_system_prompt_requests_json_output(self, builder: PromptBuilder):
        messages = builder.build_round_type_selection_prompt("Hello")
        system_content = messages[0]["content"]
        assert "JSON" in system_content

    def test_system_prompt_specifies_message_field(self, builder: PromptBuilder):
        messages = builder.build_round_type_selection_prompt("Hello")
        system_content = messages[0]["content"]
        assert '"message"' in system_content

    def test_system_prompt_specifies_round_type_detected_field(self, builder: PromptBuilder):
        messages = builder.build_round_type_selection_prompt("Hello")
        system_content = messages[0]["content"]
        assert '"round_type_detected"' in system_content

    def test_system_prompt_lists_valid_round_types(self, builder: PromptBuilder):
        messages = builder.build_round_type_selection_prompt("Hello")
        system_content = messages[0]["content"]
        assert "dsa_coding" in system_content
        assert "system_design" in system_content
        assert "behavioral" in system_content

    def test_system_prompt_mentions_keyword_detection(self, builder: PromptBuilder):
        messages = builder.build_round_type_selection_prompt("Hello")
        system_content = messages[0]["content"]
        assert "keywords" in system_content.lower()

    def test_system_prompt_provides_dsa_keyword_examples(self, builder: PromptBuilder):
        messages = builder.build_round_type_selection_prompt("Hello")
        system_content = messages[0]["content"]
        assert "DSA" in system_content or "coding" in system_content.lower()
        assert "algorithms" in system_content.lower()

    def test_system_prompt_provides_system_design_keyword_examples(self, builder: PromptBuilder):
        messages = builder.build_round_type_selection_prompt("Hello")
        system_content = messages[0]["content"]
        assert "system design" in system_content.lower() or "design" in system_content.lower()
        assert "architecture" in system_content.lower()

    def test_system_prompt_provides_behavioral_keyword_examples(self, builder: PromptBuilder):
        messages = builder.build_round_type_selection_prompt("Hello")
        system_content = messages[0]["content"]
        assert "behavioral" in system_content.lower()
        assert "soft skills" in system_content.lower()

    def test_system_prompt_mentions_software_engineer_role(self, builder: PromptBuilder):
        messages = builder.build_round_type_selection_prompt("Hello")
        system_content = messages[0]["content"]
        assert "Software Engineer" in system_content

    def test_no_forbidden_formatting(self, builder: PromptBuilder):
        messages = builder.build_round_type_selection_prompt("I want to practice coding")
        assert_no_forbidden_formatting(messages)
