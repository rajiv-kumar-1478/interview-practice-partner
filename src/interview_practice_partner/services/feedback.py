"""FeedbackService — generates structured post-interview FeedbackReport via LLM."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import structlog

from interview_practice_partner.domain.enums import EvaluationDimension
from interview_practice_partner.domain.models import (
    DimensionScore,
    FeedbackReport,
    SessionState,
)
from interview_practice_partner.llm.client import LLMClient
from interview_practice_partner.llm.prompt_builder import PromptBuilder

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Focus/relevance keywords used to detect whether the LLM already addressed
# the off-topic pattern in its output.
# ---------------------------------------------------------------------------

_FOCUS_KEYWORDS = frozenset(
    [
        "focus",
        "relevance",
        "relevant",
        "on-topic",
        "on topic",
        "stay on",
        "off-topic",
        "off topic",
        "topic",
        "distract",
    ]
)

# ---------------------------------------------------------------------------
# Safe defaults for fallback FeedbackReport construction
# ---------------------------------------------------------------------------

_FALLBACK_DIMENSION_SCORES: list[dict] = [
    {
        "dimension": EvaluationDimension.COMMUNICATION_CLARITY,
        "qualitative_assessment": "Unable to assess — feedback generation encountered an error.",
        "score": 3,
    },
    {
        "dimension": EvaluationDimension.RELEVANCE,
        "qualitative_assessment": "Unable to assess — feedback generation encountered an error.",
        "score": 3,
    },
    {
        "dimension": EvaluationDimension.TECHNICAL_KNOWLEDGE,
        "qualitative_assessment": "Unable to assess — feedback generation encountered an error.",
        "score": 3,
    },
    {
        "dimension": EvaluationDimension.CONFIDENCE,
        "qualitative_assessment": "Unable to assess — feedback generation encountered an error.",
        "score": 3,
    },
]


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _contains_focus_reference(texts: list[str]) -> bool:
    """Return True if any string in *texts* contains a focus/relevance keyword."""
    combined = " ".join(texts).lower()
    return any(kw in combined for kw in _FOCUS_KEYWORDS)


class FeedbackService:
    """Generates a structured ``FeedbackReport`` for a completed interview session.

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
    # Public API
    # ------------------------------------------------------------------

    async def generate_feedback_report(
        self,
        session: SessionState,
    ) -> tuple[str, SessionState]:
        """Generate a structured ``FeedbackReport`` for the completed session.

        Calls ``PromptBuilder.build_feedback_prompt`` with the full session
        transcript (including off-topic flags and skipped questions), sends the
        prompt to the LLM, and parses the JSON response into a ``FeedbackReport``.

        Post-processing guarantees:
        - All four ``EvaluationDimension`` values are present in
          ``dimension_scores`` (missing ones are filled with safe defaults).
        - At least one ``strengths`` entry, one ``improvements`` entry, and one
          ``actionable_recommendations`` entry are present.
        - When ``off_topic_count > 0``, specific off-topic responses are
          referenced in ``off_topic_references``.
        - When ``off_topic_count > 2``, ``improvements`` or
          ``actionable_recommendations`` contains a focus/relevance note.

        Args:
            session: The completed ``SessionState`` containing all questions,
                responses, and evaluation metadata.

        Returns:
            A ``(reply_text, updated_session)`` tuple where ``reply_text`` is a
            WhatsApp-friendly plain-text feedback message and ``updated_session``
            has ``feedback_report`` populated.
        """
        log = logger.bind(
            session_id=session.session_id,
            off_topic_count=session.off_topic_count,
            question_count=len(session.questions),
            response_count=len(session.responses),
        )
        log.info("generating_feedback_report")

        messages = self._prompt_builder.build_feedback_prompt(session=session)

        try:
            raw = await self._llm.complete(messages, temperature=0.4, max_tokens=2048)
            report = self._parse_feedback_response(raw, session)
        except Exception as exc:  # noqa: BLE001
            log.warning("feedback_llm_call_failed", error=str(exc))
            report = self._build_fallback_report(session)

        # Post-process: enforce structural invariants
        report = self._ensure_all_dimensions(report)
        report = self._ensure_minimum_lists(report)
        report = self._ensure_off_topic_references(report, session)
        report = self._ensure_focus_relevance_note(report, session)

        # Store the report in the session
        session.feedback_report = report

        log.info(
            "feedback_report_generated",
            report_id=report.report_id,
            dimension_count=len(report.dimension_scores),
            strengths_count=len(report.strengths),
            improvements_count=len(report.improvements),
            recommendations_count=len(report.actionable_recommendations),
            off_topic_refs_count=len(report.off_topic_references),
        )

        reply_text = self._format_feedback_message(report, session)
        return reply_text, session

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_feedback_response(
        self,
        raw: str,
        session: SessionState,
    ) -> FeedbackReport:
        """Parse the LLM JSON response into a ``FeedbackReport``.

        Falls back to a minimal valid ``FeedbackReport`` with safe defaults if
        JSON parsing or Pydantic validation fails.

        Args:
            raw: The raw LLM response string (expected to be JSON).
            session: The current ``SessionState`` (used for ``session_id``).

        Returns:
            A ``FeedbackReport`` instance.
        """
        try:
            data = json.loads(raw)

            # Parse dimension_scores
            dimension_scores: list[DimensionScore] = []
            for ds_data in data.get("dimension_scores", []):
                try:
                    dimension_scores.append(DimensionScore(**ds_data))
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "feedback_dimension_score_parse_failed",
                        dimension_data=str(ds_data)[:200],
                    )

            # Build the report — use safe defaults for missing list fields
            strengths = data.get("strengths") or []
            improvements = data.get("improvements") or []
            actionable_recommendations = data.get("actionable_recommendations") or []
            off_topic_references = data.get("off_topic_references") or []

            report = FeedbackReport(
                report_id=str(uuid.uuid4()),
                session_id=session.session_id,
                dimension_scores=dimension_scores,
                strengths=strengths if strengths else ["Session completed."],
                improvements=improvements if improvements else ["Continue practising."],
                actionable_recommendations=(
                    actionable_recommendations
                    if actionable_recommendations
                    else ["Keep practising mock interviews regularly."]
                ),
                off_topic_references=off_topic_references,
                generated_at=_now(),
            )
            return report

        except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "feedback_json_parse_failed",
                error=str(exc),
                raw_response=raw[:200],
            )
            return self._build_fallback_report(session)

    # ------------------------------------------------------------------
    # Fallback report
    # ------------------------------------------------------------------

    def _build_fallback_report(self, session: SessionState) -> FeedbackReport:
        """Build a minimal valid ``FeedbackReport`` with safe defaults.

        Used when the LLM returns invalid JSON or the call fails entirely.

        Args:
            session: The current ``SessionState``.

        Returns:
            A minimal ``FeedbackReport`` with all required fields populated.
        """
        dimension_scores = [
            DimensionScore(**ds) for ds in _FALLBACK_DIMENSION_SCORES
        ]
        return FeedbackReport(
            report_id=str(uuid.uuid4()),
            session_id=session.session_id,
            dimension_scores=dimension_scores,
            strengths=["You completed the mock interview session."],
            improvements=["Continue practising to improve your interview skills."],
            actionable_recommendations=[
                "Practise answering common interview questions out loud regularly."
            ],
            off_topic_references=[],
            generated_at=_now(),
        )

    # ------------------------------------------------------------------
    # Post-processing: structural invariant enforcement
    # ------------------------------------------------------------------

    def _ensure_all_dimensions(self, report: FeedbackReport) -> FeedbackReport:
        """Ensure all four ``EvaluationDimension`` values are in ``dimension_scores``.

        If any dimension is missing, a safe default ``DimensionScore`` is appended.

        Args:
            report: The ``FeedbackReport`` to check and fix.

        Returns:
            The updated ``FeedbackReport``.
        """
        present_dimensions = {ds.dimension for ds in report.dimension_scores}
        missing = [d for d in EvaluationDimension if d not in present_dimensions]

        if not missing:
            return report

        extra_scores = [
            DimensionScore(
                dimension=dim,
                qualitative_assessment=(
                    "Insufficient data to assess this dimension for this session."
                ),
                score=3,
            )
            for dim in missing
        ]

        # Rebuild with all dimensions (Pydantic models are immutable by default)
        return report.model_copy(
            update={"dimension_scores": report.dimension_scores + extra_scores}
        )

    def _ensure_minimum_lists(self, report: FeedbackReport) -> FeedbackReport:
        """Ensure ``strengths``, ``improvements``, and ``actionable_recommendations``
        each have at least one entry.

        Args:
            report: The ``FeedbackReport`` to check and fix.

        Returns:
            The updated ``FeedbackReport``.
        """
        updates: dict = {}

        if not report.strengths:
            updates["strengths"] = ["You completed the mock interview session."]

        if not report.improvements:
            updates["improvements"] = [
                "Continue practising to improve your interview skills."
            ]

        if not report.actionable_recommendations:
            updates["actionable_recommendations"] = [
                "Practise answering common interview questions out loud regularly."
            ]

        if updates:
            return report.model_copy(update=updates)
        return report

    def _ensure_off_topic_references(
        self,
        report: FeedbackReport,
        session: SessionState,
    ) -> FeedbackReport:
        """Populate ``off_topic_references`` when ``off_topic_count > 0``.

        If the LLM already populated ``off_topic_references``, this is a no-op.
        Otherwise, the specific off-topic responses are extracted from the session
        and added to the report.

        Args:
            report: The ``FeedbackReport`` to check and fix.
            session: The ``SessionState`` containing response data.

        Returns:
            The updated ``FeedbackReport``.
        """
        if session.off_topic_count == 0:
            return report

        # If the LLM already provided references, keep them
        if report.off_topic_references:
            return report

        # Extract off-topic responses from the session
        off_topic_texts = [
            r.text for r in session.responses if r.is_off_topic
        ]

        if not off_topic_texts:
            # off_topic_count > 0 but no responses flagged — add a generic note
            off_topic_texts = [
                f"One or more responses were flagged as off-topic during this session "
                f"({session.off_topic_count} total)."
            ]

        return report.model_copy(update={"off_topic_references": off_topic_texts})

    def _ensure_focus_relevance_note(
        self,
        report: FeedbackReport,
        session: SessionState,
    ) -> FeedbackReport:
        """When ``off_topic_count > 2``, ensure ``improvements`` or
        ``actionable_recommendations`` references focus/relevance.

        If neither list already contains a focus/relevance reference, a note is
        appended to ``improvements``.

        Args:
            report: The ``FeedbackReport`` to check and fix.
            session: The ``SessionState`` containing off-topic metadata.

        Returns:
            The updated ``FeedbackReport``.
        """
        if session.off_topic_count <= 2:
            return report

        # Check if focus/relevance is already mentioned
        if _contains_focus_reference(report.improvements) or _contains_focus_reference(
            report.actionable_recommendations
        ):
            return report

        # Append a focus/relevance note to improvements
        focus_note = (
            f"Focus and relevance: you went off-topic {session.off_topic_count} time(s) "
            "during this session. In a real interview, staying focused on the question "
            "asked is essential — practise listening carefully and structuring your "
            "answers to directly address what the interviewer is asking."
        )
        updated_improvements = list(report.improvements) + [focus_note]
        return report.model_copy(update={"improvements": updated_improvements})

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_feedback_message(
        self,
        report: FeedbackReport,
        session: SessionState,
    ) -> str:
        """Format the ``FeedbackReport`` as a WhatsApp-friendly plain-text message.

        Uses plain text with asterisks for bold emphasis and line breaks for
        structure. No HTML tags, no markdown headers, no code blocks.

        Args:
            report: The ``FeedbackReport`` to format.
            session: The ``SessionState`` (used for role display name).

        Returns:
            A formatted plain-text string suitable for WhatsApp delivery.
        """
        lines: list[str] = []

        lines.append("*Your Interview Feedback*")
        lines.append("")
        lines.append(
            "Well done on completing your mock interview! Here is your personalised feedback."
        )
        lines.append("")

        # Dimension scores
        lines.append("*Performance by Dimension*")
        lines.append("")
        for ds in report.dimension_scores:
            dimension_label = ds.dimension.value.replace("_", " ").title()
            lines.append(f"*{dimension_label}*")
            lines.append(ds.qualitative_assessment)
            lines.append("")

        # Strengths
        lines.append("*Strengths*")
        for strength in report.strengths:
            lines.append(f"- {strength}")
        lines.append("")

        # Areas for improvement
        lines.append("*Areas for Improvement*")
        for improvement in report.improvements:
            lines.append(f"- {improvement}")
        lines.append("")

        # Actionable recommendations
        lines.append("*Actionable Recommendations*")
        for rec in report.actionable_recommendations:
            lines.append(f"- {rec}")
        lines.append("")

        # Off-topic references (only if present)
        if report.off_topic_references:
            lines.append("*Responses That Were Off-Topic*")
            for ref in report.off_topic_references:
                lines.append(f"- {ref}")
            lines.append("")

        lines.append(
            "Keep practising — every session brings you closer to interview success!"
        )

        return "\n".join(lines)
