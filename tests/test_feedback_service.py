"""Unit tests for FeedbackService.

Covers:
- generate_feedback_report parses LLM JSON into a valid FeedbackReport
- All four evaluation dimensions are present in the report
- Off-topic references are populated when off_topic_count > 0
- Fallback report is used when LLM returns invalid JSON
- Focus/relevance note is added when off_topic_count > 2
- Formatted reply is WhatsApp-friendly plain text

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 7.2
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

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
from interview_practice_partner.services.feedback import FeedbackService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
PHONE = "+15550001234"

_ALL_DIMENSIONS = list(EvaluationDimension)


def make_session(**overrides) -> SessionState:
    defaults = dict(
        session_id=str(uuid.uuid4()),
        phone_number=PHONE,
        stage=Stage.FEEDBACK,
        role=Role.SOFTWARE_ENGINEER,
        created_at=NOW,
        updated_at=NOW,
    )
    defaults.update(overrides)
    return SessionState(**defaults)


def make_question(
    question_id: str | None = None,
    text: str = "Tell me about yourself.",
    question_type: QuestionType = QuestionType.BEHAVIOURAL,
    skipped: bool = False,
) -> Question:
    return Question(
        question_id=question_id or str(uuid.uuid4()),
        text=text,
        question_type=question_type,
        asked_at=NOW,
        skipped=skipped,
    )


def make_response(
    question_id: str,
    text: str = "I have five years of experience working on distributed systems.",
    is_off_topic: bool = False,
) -> UserResponse:
    return UserResponse(
        response_id=str(uuid.uuid4()),
        question_id=question_id,
        text=text,
        word_count=len(text.split()),
        is_off_topic=is_off_topic,
        received_at=NOW,
    )


def _make_full_dimension_scores_json() -> list[dict]:
    """Return a list of dimension score dicts covering all four dimensions."""
    return [
        {
            "dimension": dim.value,
            "qualitative_assessment": f"Good performance on {dim.value}.",
            "score": 4,
        }
        for dim in EvaluationDimension
    ]


def _make_valid_feedback_json(
    off_topic_references: list[str] | None = None,
    improvements: list[str] | None = None,
    actionable_recommendations: list[str] | None = None,
) -> str:
    """Build a valid LLM feedback JSON response."""
    return json.dumps({
        "dimension_scores": _make_full_dimension_scores_json(),
        "strengths": ["Clear communication throughout the session."],
        "improvements": improvements or ["Work on providing more specific examples."],
        "actionable_recommendations": actionable_recommendations or [
            "Use the STAR method to structure your answers."
        ],
        "off_topic_references": off_topic_references or [],
    })


def make_service(
    llm_response: str = "",
) -> tuple[FeedbackService, AsyncMock, MagicMock]:
    """Build a FeedbackService with mocked dependencies."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = llm_response

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_feedback_prompt.return_value = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]

    service = FeedbackService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
    )
    return service, mock_llm, mock_prompt_builder


# ===========================================================================
# generate_feedback_report — happy path
# ===========================================================================


class TestGenerateFeedbackReportHappyPath:
    @pytest.mark.asyncio
    async def test_returns_tuple_of_reply_and_session(self):
        """generate_feedback_report returns (str, SessionState)."""
        service, _, _ = make_service(llm_response=_make_valid_feedback_json())
        session = make_session()

        result = await service.generate_feedback_report(session)

        assert isinstance(result, tuple)
        assert len(result) == 2
        reply_text, updated_session = result
        assert isinstance(reply_text, str)
        assert isinstance(updated_session, SessionState)

    @pytest.mark.asyncio
    async def test_stores_report_in_session(self):
        """generate_feedback_report stores FeedbackReport in session.feedback_report."""
        service, _, _ = make_service(llm_response=_make_valid_feedback_json())
        session = make_session()

        _, updated_session = await service.generate_feedback_report(session)

        assert updated_session.feedback_report is not None
        assert isinstance(updated_session.feedback_report, FeedbackReport)

    @pytest.mark.asyncio
    async def test_report_has_correct_session_id(self):
        """FeedbackReport.session_id matches the session's session_id."""
        service, _, _ = make_service(llm_response=_make_valid_feedback_json())
        session = make_session()

        _, updated_session = await service.generate_feedback_report(session)

        assert updated_session.feedback_report.session_id == session.session_id

    @pytest.mark.asyncio
    async def test_calls_prompt_builder_with_session(self):
        """generate_feedback_report calls PromptBuilder.build_feedback_prompt."""
        service, _, mock_pb = make_service(llm_response=_make_valid_feedback_json())
        session = make_session()

        await service.generate_feedback_report(session)

        mock_pb.build_feedback_prompt.assert_called_once_with(session=session)

    @pytest.mark.asyncio
    async def test_calls_llm_complete(self):
        """generate_feedback_report calls LLMClient.complete."""
        service, mock_llm, _ = make_service(llm_response=_make_valid_feedback_json())
        session = make_session()

        await service.generate_feedback_report(session)

        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_reply_text_is_non_empty_string(self):
        """Reply text is a non-empty string."""
        service, _, _ = make_service(llm_response=_make_valid_feedback_json())
        session = make_session()

        reply_text, _ = await service.generate_feedback_report(session)

        assert len(reply_text) > 0

    @pytest.mark.asyncio
    async def test_reply_text_contains_no_html_tags(self):
        """Reply text contains no HTML tags."""
        service, _, _ = make_service(llm_response=_make_valid_feedback_json())
        session = make_session()

        reply_text, _ = await service.generate_feedback_report(session)

        assert "<b>" not in reply_text
        assert "<br>" not in reply_text
        assert "<p>" not in reply_text
        assert "</" not in reply_text

    @pytest.mark.asyncio
    async def test_reply_text_contains_no_markdown_headers(self):
        """Reply text contains no markdown headers (lines starting with #)."""
        service, _, _ = make_service(llm_response=_make_valid_feedback_json())
        session = make_session()

        reply_text, _ = await service.generate_feedback_report(session)

        for line in reply_text.splitlines():
            assert not line.startswith("#"), f"Markdown header found: {line!r}"


# ===========================================================================
# All four EvaluationDimension values must be present
# ===========================================================================


class TestAllDimensionsPresent:
    @pytest.mark.asyncio
    async def test_all_four_dimensions_present_from_llm(self):
        """Report includes DimensionScore for all four EvaluationDimension values."""
        service, _, _ = make_service(llm_response=_make_valid_feedback_json())
        session = make_session()

        _, updated_session = await service.generate_feedback_report(session)

        report = updated_session.feedback_report
        present = {ds.dimension for ds in report.dimension_scores}
        assert present == set(EvaluationDimension)

    @pytest.mark.asyncio
    async def test_missing_dimensions_are_filled_in(self):
        """When LLM omits some dimensions, they are filled with safe defaults."""
        # Only include two dimensions in the LLM response
        partial_json = json.dumps({
            "dimension_scores": [
                {
                    "dimension": EvaluationDimension.COMMUNICATION_CLARITY.value,
                    "qualitative_assessment": "Good clarity.",
                    "score": 4,
                },
                {
                    "dimension": EvaluationDimension.CONFIDENCE.value,
                    "qualitative_assessment": "Confident delivery.",
                    "score": 4,
                },
            ],
            "strengths": ["Good communication."],
            "improvements": ["Work on technical depth."],
            "actionable_recommendations": ["Study system design."],
            "off_topic_references": [],
        })
        service, _, _ = make_service(llm_response=partial_json)
        session = make_session()

        _, updated_session = await service.generate_feedback_report(session)

        report = updated_session.feedback_report
        present = {ds.dimension for ds in report.dimension_scores}
        assert present == set(EvaluationDimension)

    @pytest.mark.asyncio
    async def test_empty_dimension_scores_filled_with_all_four(self):
        """When LLM returns empty dimension_scores, all four are added."""
        empty_dims_json = json.dumps({
            "dimension_scores": [],
            "strengths": ["Completed the session."],
            "improvements": ["Keep practising."],
            "actionable_recommendations": ["Practise daily."],
            "off_topic_references": [],
        })
        service, _, _ = make_service(llm_response=empty_dims_json)
        session = make_session()

        _, updated_session = await service.generate_feedback_report(session)

        report = updated_session.feedback_report
        assert len(report.dimension_scores) == 4
        present = {ds.dimension for ds in report.dimension_scores}
        assert present == set(EvaluationDimension)


# ===========================================================================
# Minimum list entries
# ===========================================================================


class TestMinimumListEntries:
    @pytest.mark.asyncio
    async def test_at_least_one_strength(self):
        """Report always has at least one strength."""
        service, _, _ = make_service(llm_response=_make_valid_feedback_json())
        session = make_session()

        _, updated_session = await service.generate_feedback_report(session)

        assert len(updated_session.feedback_report.strengths) >= 1

    @pytest.mark.asyncio
    async def test_at_least_one_improvement(self):
        """Report always has at least one improvement."""
        service, _, _ = make_service(llm_response=_make_valid_feedback_json())
        session = make_session()

        _, updated_session = await service.generate_feedback_report(session)

        assert len(updated_session.feedback_report.improvements) >= 1

    @pytest.mark.asyncio
    async def test_at_least_one_actionable_recommendation(self):
        """Report always has at least one actionable recommendation."""
        service, _, _ = make_service(llm_response=_make_valid_feedback_json())
        session = make_session()

        _, updated_session = await service.generate_feedback_report(session)

        assert len(updated_session.feedback_report.actionable_recommendations) >= 1

    @pytest.mark.asyncio
    async def test_empty_strengths_filled_with_default(self):
        """When LLM returns empty strengths, a default is added."""
        no_strengths_json = json.dumps({
            "dimension_scores": _make_full_dimension_scores_json(),
            "strengths": [],
            "improvements": ["Keep practising."],
            "actionable_recommendations": ["Practise daily."],
            "off_topic_references": [],
        })
        service, _, _ = make_service(llm_response=no_strengths_json)
        session = make_session()

        _, updated_session = await service.generate_feedback_report(session)

        assert len(updated_session.feedback_report.strengths) >= 1

    @pytest.mark.asyncio
    async def test_empty_improvements_filled_with_default(self):
        """When LLM returns empty improvements, a default is added."""
        no_improvements_json = json.dumps({
            "dimension_scores": _make_full_dimension_scores_json(),
            "strengths": ["Good communication."],
            "improvements": [],
            "actionable_recommendations": ["Practise daily."],
            "off_topic_references": [],
        })
        service, _, _ = make_service(llm_response=no_improvements_json)
        session = make_session()

        _, updated_session = await service.generate_feedback_report(session)

        assert len(updated_session.feedback_report.improvements) >= 1

    @pytest.mark.asyncio
    async def test_empty_recommendations_filled_with_default(self):
        """When LLM returns empty actionable_recommendations, a default is added."""
        no_recs_json = json.dumps({
            "dimension_scores": _make_full_dimension_scores_json(),
            "strengths": ["Good communication."],
            "improvements": ["Work on depth."],
            "actionable_recommendations": [],
            "off_topic_references": [],
        })
        service, _, _ = make_service(llm_response=no_recs_json)
        session = make_session()

        _, updated_session = await service.generate_feedback_report(session)

        assert len(updated_session.feedback_report.actionable_recommendations) >= 1


# ===========================================================================
# Off-topic references
# ===========================================================================


class TestOffTopicReferences:
    @pytest.mark.asyncio
    async def test_off_topic_references_populated_when_off_topic_count_gt_0(self):
        """off_topic_references is populated when off_topic_count > 0."""
        session = make_session(off_topic_count=1)
        q = make_question()
        session.questions.append(q)
        off_topic_resp = make_response(
            question_id=q.question_id,
            text="I like cooking pasta on weekends.",
            is_off_topic=True,
        )
        session.responses.append(off_topic_resp)

        # LLM returns empty off_topic_references — service should fill them in
        service, _, _ = make_service(llm_response=_make_valid_feedback_json(off_topic_references=[]))
        _, updated_session = await service.generate_feedback_report(session)

        assert len(updated_session.feedback_report.off_topic_references) > 0

    @pytest.mark.asyncio
    async def test_off_topic_references_contain_actual_response_text(self):
        """off_topic_references contains the actual off-topic response text."""
        session = make_session(off_topic_count=1)
        q = make_question()
        session.questions.append(q)
        off_topic_text = "I like cooking pasta on weekends."
        off_topic_resp = make_response(
            question_id=q.question_id,
            text=off_topic_text,
            is_off_topic=True,
        )
        session.responses.append(off_topic_resp)

        service, _, _ = make_service(llm_response=_make_valid_feedback_json(off_topic_references=[]))
        _, updated_session = await service.generate_feedback_report(session)

        assert off_topic_text in updated_session.feedback_report.off_topic_references

    @pytest.mark.asyncio
    async def test_off_topic_references_empty_when_no_off_topic(self):
        """off_topic_references is empty when off_topic_count == 0."""
        session = make_session(off_topic_count=0)
        service, _, _ = make_service(llm_response=_make_valid_feedback_json(off_topic_references=[]))
        _, updated_session = await service.generate_feedback_report(session)

        assert updated_session.feedback_report.off_topic_references == []

    @pytest.mark.asyncio
    async def test_llm_provided_off_topic_references_are_kept(self):
        """When LLM provides off_topic_references, they are preserved."""
        session = make_session(off_topic_count=1)
        llm_refs = ["The candidate discussed cooking instead of answering the question."]
        service, _, _ = make_service(
            llm_response=_make_valid_feedback_json(off_topic_references=llm_refs)
        )
        _, updated_session = await service.generate_feedback_report(session)

        assert updated_session.feedback_report.off_topic_references == llm_refs

    @pytest.mark.asyncio
    async def test_off_topic_count_gt_0_no_flagged_responses_adds_generic_note(self):
        """When off_topic_count > 0 but no responses are flagged, a generic note is added."""
        # off_topic_count=1 but no responses have is_off_topic=True
        session = make_session(off_topic_count=1)
        service, _, _ = make_service(llm_response=_make_valid_feedback_json(off_topic_references=[]))
        _, updated_session = await service.generate_feedback_report(session)

        assert len(updated_session.feedback_report.off_topic_references) > 0


# ===========================================================================
# Focus/relevance note when off_topic_count > 2
# ===========================================================================


class TestFocusRelevanceNote:
    @pytest.mark.asyncio
    async def test_focus_note_added_to_improvements_when_off_topic_count_gt_2(self):
        """When off_topic_count > 2, improvements references focus/relevance."""
        session = make_session(off_topic_count=3)
        # LLM returns improvements without any focus/relevance mention
        service, _, _ = make_service(
            llm_response=_make_valid_feedback_json(
                improvements=["Work on providing more specific examples."]
            )
        )
        _, updated_session = await service.generate_feedback_report(session)

        report = updated_session.feedback_report
        combined = " ".join(report.improvements + report.actionable_recommendations).lower()
        assert any(
            kw in combined
            for kw in ["focus", "relevance", "relevant", "on-topic", "off-topic", "topic"]
        )

    @pytest.mark.asyncio
    async def test_no_focus_note_when_off_topic_count_le_2(self):
        """When off_topic_count <= 2, no focus/relevance note is forcibly added."""
        session = make_session(off_topic_count=2)
        service, _, _ = make_service(
            llm_response=_make_valid_feedback_json(
                improvements=["Work on providing more specific examples."]
            )
        )
        _, updated_session = await service.generate_feedback_report(session)

        # The improvements list should not have been extended with a focus note
        # (it may still contain focus keywords if the LLM included them, but
        # we're testing that the service doesn't forcibly add one)
        report = updated_session.feedback_report
        # Count improvements — should be exactly what the LLM returned
        assert len(report.improvements) == 1

    @pytest.mark.asyncio
    async def test_no_duplicate_focus_note_when_llm_already_mentions_focus(self):
        """When LLM already mentions focus/relevance, no duplicate note is added."""
        session = make_session(off_topic_count=4)
        service, _, _ = make_service(
            llm_response=_make_valid_feedback_json(
                improvements=["Work on staying focused and relevant to the question."]
            )
        )
        _, updated_session = await service.generate_feedback_report(session)

        # The improvements list should not have grown (LLM already covered it)
        report = updated_session.feedback_report
        assert len(report.improvements) == 1

    @pytest.mark.asyncio
    async def test_focus_note_not_added_when_recommendations_already_mention_focus(self):
        """When actionable_recommendations already mentions focus, no note added to improvements."""
        session = make_session(off_topic_count=5)
        service, _, _ = make_service(
            llm_response=_make_valid_feedback_json(
                improvements=["Work on technical depth."],
                actionable_recommendations=["Practise staying on topic and relevant."],
            )
        )
        _, updated_session = await service.generate_feedback_report(session)

        report = updated_session.feedback_report
        # improvements should not have grown since recommendations already cover it
        assert len(report.improvements) == 1


# ===========================================================================
# Fallback on invalid JSON
# ===========================================================================


class TestFallbackOnInvalidJson:
    @pytest.mark.asyncio
    async def test_fallback_report_on_invalid_json(self):
        """When LLM returns invalid JSON, a fallback FeedbackReport is used."""
        service, _, _ = make_service(llm_response="This is not valid JSON at all!")
        session = make_session()

        _, updated_session = await service.generate_feedback_report(session)

        report = updated_session.feedback_report
        assert report is not None
        assert isinstance(report, FeedbackReport)

    @pytest.mark.asyncio
    async def test_fallback_report_has_all_four_dimensions(self):
        """Fallback report has all four EvaluationDimension values."""
        service, _, _ = make_service(llm_response="Not JSON")
        session = make_session()

        _, updated_session = await service.generate_feedback_report(session)

        present = {ds.dimension for ds in updated_session.feedback_report.dimension_scores}
        assert present == set(EvaluationDimension)

    @pytest.mark.asyncio
    async def test_fallback_report_has_minimum_list_entries(self):
        """Fallback report has at least one strength, improvement, and recommendation."""
        service, _, _ = make_service(llm_response="{}")
        session = make_session()

        _, updated_session = await service.generate_feedback_report(session)

        report = updated_session.feedback_report
        assert len(report.strengths) >= 1
        assert len(report.improvements) >= 1
        assert len(report.actionable_recommendations) >= 1

    @pytest.mark.asyncio
    async def test_fallback_report_on_llm_exception(self):
        """When LLM raises an exception, a fallback FeedbackReport is used."""
        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = RuntimeError("LLM unavailable")

        mock_pb = MagicMock()
        mock_pb.build_feedback_prompt.return_value = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

        service = FeedbackService(llm_client=mock_llm, prompt_builder=mock_pb)
        session = make_session()

        _, updated_session = await service.generate_feedback_report(session)

        assert updated_session.feedback_report is not None
        assert len(updated_session.feedback_report.strengths) >= 1


# ===========================================================================
# Reply text formatting
# ===========================================================================


class TestReplyTextFormatting:
    @pytest.mark.asyncio
    async def test_reply_contains_strengths_section(self):
        """Reply text contains a strengths section."""
        service, _, _ = make_service(llm_response=_make_valid_feedback_json())
        session = make_session()

        reply_text, _ = await service.generate_feedback_report(session)

        assert "strength" in reply_text.lower()

    @pytest.mark.asyncio
    async def test_reply_contains_improvements_section(self):
        """Reply text contains an improvements section."""
        service, _, _ = make_service(llm_response=_make_valid_feedback_json())
        session = make_session()

        reply_text, _ = await service.generate_feedback_report(session)

        assert "improvement" in reply_text.lower()

    @pytest.mark.asyncio
    async def test_reply_contains_recommendations_section(self):
        """Reply text contains a recommendations section."""
        service, _, _ = make_service(llm_response=_make_valid_feedback_json())
        session = make_session()

        reply_text, _ = await service.generate_feedback_report(session)

        assert "recommendation" in reply_text.lower()

    @pytest.mark.asyncio
    async def test_reply_contains_off_topic_section_when_present(self):
        """Reply text contains off-topic section when off_topic_references is non-empty."""
        session = make_session(off_topic_count=1)
        q = make_question()
        session.questions.append(q)
        session.responses.append(
            make_response(
                question_id=q.question_id,
                text="I like cooking pasta.",
                is_off_topic=True,
            )
        )
        service, _, _ = make_service(llm_response=_make_valid_feedback_json(off_topic_references=[]))
        reply_text, _ = await service.generate_feedback_report(session)

        assert "off-topic" in reply_text.lower() or "off topic" in reply_text.lower()

    @pytest.mark.asyncio
    async def test_reply_does_not_contain_off_topic_section_when_none(self):
        """Reply text does not contain off-topic section when there are no off-topic responses."""
        session = make_session(off_topic_count=0)
        service, _, _ = make_service(llm_response=_make_valid_feedback_json())
        reply_text, _ = await service.generate_feedback_report(session)

        # The off-topic section header should not appear
        assert "Responses That Were Off-Topic" not in reply_text


# ===========================================================================
# Technical feedback — DSA round
# ===========================================================================


def make_technical_service(
    llm_response: str = "",
) -> tuple[FeedbackService, AsyncMock, MagicMock]:
    """Build a FeedbackService with mocked dependencies for technical rounds."""
    from interview_practice_partner.domain.enums import InterviewRoundType

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = llm_response

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_technical_feedback_prompt.return_value = [
        {"role": "system", "content": "technical system prompt"},
        {"role": "user", "content": "technical user prompt"},
    ]

    service = FeedbackService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
    )
    return service, mock_llm, mock_prompt_builder


def _make_dsa_feedback_json(
    strengths: list[str] | None = None,
    improvements: list[str] | None = None,
    recommendations: list[str] | None = None,
    complexity_summary: str | None = None,
    problem_solving_approach: str | None = None,
) -> str:
    return json.dumps({
        "strengths": strengths or ["Good use of hash maps for O(1) lookups."],
        "improvements": improvements or ["Consider edge cases for empty inputs."],
        "actionable_recommendations": recommendations or ["Practice dynamic programming problems."],
        "complexity_summary": complexity_summary,
        "problem_solving_approach": problem_solving_approach,
    })


def _make_system_design_feedback_json(
    strengths: list[str] | None = None,
    improvements: list[str] | None = None,
    recommendations: list[str] | None = None,
    design_thinking: str | None = None,
    scalability_awareness: str | None = None,
) -> str:
    return json.dumps({
        "strengths": strengths or ["Clear component separation."],
        "improvements": improvements or ["Consider caching strategies."],
        "actionable_recommendations": recommendations or ["Study distributed systems patterns."],
        "design_thinking": design_thinking,
        "scalability_awareness": scalability_awareness,
    })


class TestTechnicalFeedbackDSARound:
    """Tests for DSA round technical feedback generation."""

    @pytest.mark.asyncio
    async def test_dsa_round_calls_technical_feedback_prompt(self):
        """For DSA rounds, build_technical_feedback_prompt is called instead of build_feedback_prompt."""
        from interview_practice_partner.domain.enums import InterviewRoundType

        service, _, mock_pb = make_technical_service(
            llm_response=_make_dsa_feedback_json()
        )
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)

        await service.generate_feedback_report(session)

        mock_pb.build_technical_feedback_prompt.assert_called_once_with(session=session)
        mock_pb.build_feedback_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_dsa_round_returns_feedback_report(self):
        """DSA round feedback generation returns a valid FeedbackReport."""
        from interview_practice_partner.domain.enums import InterviewRoundType

        service, _, _ = make_technical_service(llm_response=_make_dsa_feedback_json())
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)

        reply_text, updated_session = await service.generate_feedback_report(session)

        assert isinstance(reply_text, str)
        assert updated_session.feedback_report is not None
        assert isinstance(updated_session.feedback_report, FeedbackReport)

    @pytest.mark.asyncio
    async def test_dsa_round_includes_complexity_summary_in_improvements(self):
        """complexity_summary from LLM is folded into improvements."""
        from interview_practice_partner.domain.enums import InterviewRoundType

        service, _, _ = make_technical_service(
            llm_response=_make_dsa_feedback_json(
                complexity_summary="Candidate consistently achieved O(n log n) solutions."
            )
        )
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)

        _, updated_session = await service.generate_feedback_report(session)

        improvements_text = " ".join(updated_session.feedback_report.improvements)
        assert "Complexity analysis" in improvements_text
        assert "O(n log n)" in improvements_text

    @pytest.mark.asyncio
    async def test_dsa_round_includes_problem_solving_approach_in_strengths(self):
        """problem_solving_approach from LLM is folded into strengths."""
        from interview_practice_partner.domain.enums import InterviewRoundType

        service, _, _ = make_technical_service(
            llm_response=_make_dsa_feedback_json(
                problem_solving_approach="Methodically breaks down problems before coding."
            )
        )
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)

        _, updated_session = await service.generate_feedback_report(session)

        strengths_text = " ".join(updated_session.feedback_report.strengths)
        assert "Problem-solving approach" in strengths_text
        assert "Methodically" in strengths_text

    @pytest.mark.asyncio
    async def test_dsa_round_reply_contains_dsa_label(self):
        """DSA round reply text contains 'DSA/Coding Round' label."""
        from interview_practice_partner.domain.enums import InterviewRoundType

        service, _, _ = make_technical_service(llm_response=_make_dsa_feedback_json())
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)

        reply_text, _ = await service.generate_feedback_report(session)

        assert "DSA/Coding Round" in reply_text

    @pytest.mark.asyncio
    async def test_dsa_round_reply_contains_no_html_tags(self):
        """DSA round reply text contains no HTML tags."""
        from interview_practice_partner.domain.enums import InterviewRoundType

        service, _, _ = make_technical_service(llm_response=_make_dsa_feedback_json())
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)

        reply_text, _ = await service.generate_feedback_report(session)

        assert "<b>" not in reply_text
        assert "<br>" not in reply_text
        assert "</" not in reply_text

    @pytest.mark.asyncio
    async def test_dsa_round_reply_contains_no_markdown_headers(self):
        """DSA round reply text contains no markdown headers."""
        from interview_practice_partner.domain.enums import InterviewRoundType

        service, _, _ = make_technical_service(llm_response=_make_dsa_feedback_json())
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)

        reply_text, _ = await service.generate_feedback_report(session)

        for line in reply_text.splitlines():
            assert not line.startswith("#"), f"Markdown header found: {line!r}"

    @pytest.mark.asyncio
    async def test_dsa_round_reply_shows_topics_covered(self):
        """DSA round reply includes topics covered when present in session."""
        from interview_practice_partner.domain.enums import InterviewRoundType, ProblemTopic

        service, _, _ = make_technical_service(llm_response=_make_dsa_feedback_json())
        session = make_session(
            interview_round_type=InterviewRoundType.DSA_CODING,
            topics_covered=[ProblemTopic.ARRAYS, ProblemTopic.TREES],
        )

        reply_text, _ = await service.generate_feedback_report(session)

        assert "Arrays" in reply_text
        assert "Trees" in reply_text

    @pytest.mark.asyncio
    async def test_dsa_round_reply_shows_difficulty_progression(self):
        """DSA round reply includes difficulty adjustment history when present."""
        from interview_practice_partner.domain.enums import InterviewRoundType

        service, _, _ = make_technical_service(llm_response=_make_dsa_feedback_json())
        session = make_session(
            interview_round_type=InterviewRoundType.DSA_CODING,
            difficulty_adjustment_history=[
                {"from": "medium", "to": "hard", "reason": "Optimal solution submitted"}
            ],
        )

        reply_text, _ = await service.generate_feedback_report(session)

        assert "Medium" in reply_text
        assert "Hard" in reply_text

    @pytest.mark.asyncio
    async def test_dsa_round_fallback_on_invalid_json(self):
        """DSA round falls back gracefully on invalid LLM JSON."""
        from interview_practice_partner.domain.enums import InterviewRoundType

        service, _, _ = make_technical_service(llm_response="not valid json")
        session = make_session(interview_round_type=InterviewRoundType.DSA_CODING)

        reply_text, updated_session = await service.generate_feedback_report(session)

        assert updated_session.feedback_report is not None
        assert len(updated_session.feedback_report.strengths) >= 1
        assert len(updated_session.feedback_report.improvements) >= 1


# ===========================================================================
# Technical feedback — System Design round
# ===========================================================================


class TestTechnicalFeedbackSystemDesignRound:
    """Tests for System Design round technical feedback generation."""

    @pytest.mark.asyncio
    async def test_system_design_round_calls_technical_feedback_prompt(self):
        """For System Design rounds, build_technical_feedback_prompt is called."""
        from interview_practice_partner.domain.enums import InterviewRoundType

        service, _, mock_pb = make_technical_service(
            llm_response=_make_system_design_feedback_json()
        )
        session = make_session(interview_round_type=InterviewRoundType.SYSTEM_DESIGN)

        await service.generate_feedback_report(session)

        mock_pb.build_technical_feedback_prompt.assert_called_once_with(session=session)
        mock_pb.build_feedback_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_system_design_round_includes_design_thinking_in_strengths(self):
        """design_thinking from LLM is folded into strengths."""
        from interview_practice_partner.domain.enums import InterviewRoundType

        service, _, _ = make_technical_service(
            llm_response=_make_system_design_feedback_json(
                design_thinking="Strong understanding of microservices architecture."
            )
        )
        session = make_session(interview_round_type=InterviewRoundType.SYSTEM_DESIGN)

        _, updated_session = await service.generate_feedback_report(session)

        strengths_text = " ".join(updated_session.feedback_report.strengths)
        assert "Design thinking" in strengths_text
        assert "microservices" in strengths_text

    @pytest.mark.asyncio
    async def test_system_design_round_includes_scalability_awareness_in_improvements(self):
        """scalability_awareness from LLM is folded into improvements."""
        from interview_practice_partner.domain.enums import InterviewRoundType

        service, _, _ = make_technical_service(
            llm_response=_make_system_design_feedback_json(
                scalability_awareness="Needs more focus on horizontal scaling strategies."
            )
        )
        session = make_session(interview_round_type=InterviewRoundType.SYSTEM_DESIGN)

        _, updated_session = await service.generate_feedback_report(session)

        improvements_text = " ".join(updated_session.feedback_report.improvements)
        assert "Scalability awareness" in improvements_text
        assert "horizontal scaling" in improvements_text

    @pytest.mark.asyncio
    async def test_system_design_round_reply_contains_system_design_label(self):
        """System Design round reply text contains 'System Design Round' label."""
        from interview_practice_partner.domain.enums import InterviewRoundType

        service, _, _ = make_technical_service(
            llm_response=_make_system_design_feedback_json()
        )
        session = make_session(interview_round_type=InterviewRoundType.SYSTEM_DESIGN)

        reply_text, _ = await service.generate_feedback_report(session)

        assert "System Design Round" in reply_text

    @pytest.mark.asyncio
    async def test_system_design_round_reply_shows_design_aspects_covered(self):
        """System Design round reply includes design aspects covered when present."""
        from interview_practice_partner.domain.enums import DesignAspect, InterviewRoundType

        service, _, _ = make_technical_service(
            llm_response=_make_system_design_feedback_json()
        )
        session = make_session(
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
            design_aspects_covered=[DesignAspect.SCALABILITY, DesignAspect.API_DESIGN],
        )

        reply_text, _ = await service.generate_feedback_report(session)

        assert "Scalability" in reply_text
        assert "Api Design" in reply_text

    @pytest.mark.asyncio
    async def test_system_design_round_fallback_on_llm_exception(self):
        """System Design round falls back gracefully when LLM raises an exception."""
        from interview_practice_partner.domain.enums import InterviewRoundType

        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = RuntimeError("LLM unavailable")

        mock_pb = MagicMock()
        mock_pb.build_technical_feedback_prompt.return_value = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

        service = FeedbackService(llm_client=mock_llm, prompt_builder=mock_pb)
        session = make_session(interview_round_type=InterviewRoundType.SYSTEM_DESIGN)

        reply_text, updated_session = await service.generate_feedback_report(session)

        assert updated_session.feedback_report is not None
        assert len(updated_session.feedback_report.strengths) >= 1


# ===========================================================================
# Behavioral round — unchanged behaviour
# ===========================================================================


class TestBehavioralRoundUnchanged:
    """Verify behavioral round still uses the original feedback path."""

    @pytest.mark.asyncio
    async def test_behavioral_round_calls_build_feedback_prompt(self):
        """For behavioral rounds (no round type), build_feedback_prompt is called."""
        service, _, mock_pb = make_service(llm_response=_make_valid_feedback_json())
        session = make_session()  # interview_round_type=None → behavioral

        await service.generate_feedback_report(session)

        mock_pb.build_feedback_prompt.assert_called_once_with(session=session)

    @pytest.mark.asyncio
    async def test_explicit_behavioral_round_type_calls_build_feedback_prompt(self):
        """For explicit BEHAVIORAL round type, build_feedback_prompt is called."""
        from interview_practice_partner.domain.enums import InterviewRoundType

        service, _, mock_pb = make_service(llm_response=_make_valid_feedback_json())
        session = make_session(interview_round_type=InterviewRoundType.BEHAVIORAL)

        await service.generate_feedback_report(session)

        mock_pb.build_feedback_prompt.assert_called_once_with(session=session)

    @pytest.mark.asyncio
    async def test_behavioral_round_still_has_all_four_dimensions(self):
        """Behavioral round still enforces all four EvaluationDimension values."""
        from interview_practice_partner.domain.enums import InterviewRoundType

        service, _, _ = make_service(llm_response=_make_valid_feedback_json())
        session = make_session(interview_round_type=InterviewRoundType.BEHAVIORAL)

        _, updated_session = await service.generate_feedback_report(session)

        present = {ds.dimension for ds in updated_session.feedback_report.dimension_scores}
        assert present == set(EvaluationDimension)
