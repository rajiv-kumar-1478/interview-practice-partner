"""Domain models: SessionState, Question, UserResponse, FeedbackReport, DimensionScore."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from interview_practice_partner.domain.enums import (
    DesignAspect,
    DesignPhase,
    EvaluationDimension,
    InterviewRoundType,
    ProblemDifficulty,
    ProblemTopic,
    QuestionType,
    Role,
    SolutionFormat,
    Stage,
)


class Question(BaseModel):
    question_id: str = Field(..., description="UUID identifying this question")
    text: str
    question_type: QuestionType
    asked_at: datetime
    skipped: bool = False


class UserResponse(BaseModel):
    response_id: str = Field(..., description="UUID identifying this response")
    question_id: str = Field(..., description="FK to Question.question_id")
    text: str
    word_count: int
    is_off_topic: bool = False
    received_at: datetime


class DimensionScore(BaseModel):
    dimension: EvaluationDimension
    qualitative_assessment: str
    score: int = Field(..., ge=1, le=5, description="Internal score 1–5; not exposed to user")


class FeedbackReport(BaseModel):
    report_id: str = Field(..., description="UUID identifying this report")
    session_id: str
    dimension_scores: list[DimensionScore]
    strengths: list[str] = Field(..., min_length=1, description="At least one strength")
    improvements: list[str] = Field(..., min_length=1, description="At least one improvement")
    actionable_recommendations: list[str] = Field(
        ..., min_length=1, description="At least one actionable recommendation"
    )
    off_topic_references: list[str] = Field(
        default_factory=list, description="Specific responses flagged as off-topic"
    )
    generated_at: datetime


class SessionState(BaseModel):
    session_id: str = Field(..., description="UUID identifying this session")
    phone_number: str = Field(..., description="E.164 format — primary key in Redis")
    stage: Stage = Stage.INIT
    role: Role = Role.UNKNOWN
    questions: list[Question] = Field(default_factory=list)
    responses: list[UserResponse] = Field(default_factory=list)
    off_topic_count: int = 0
    consecutive_out_of_scope_count: int = 0
    clarification_turn_count: int = 0
    requested_short_session: bool = False
    feedback_report: Optional[FeedbackReport] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    is_complete: bool = False
    context_summary: Optional[str] = None
    preferred_mode: Literal["voice", "text"] = "text"
    
    # Technical round fields
    interview_round_type: Optional[InterviewRoundType] = None
    problem_difficulty: ProblemDifficulty = ProblemDifficulty.MEDIUM
    design_phase: Optional[DesignPhase] = None
    topics_covered: list[ProblemTopic] = Field(default_factory=list)
    design_aspects_covered: list[DesignAspect] = Field(default_factory=list)
    difficulty_adjustment_history: list[dict] = Field(default_factory=list)


class CodingProblem(BaseModel):
    problem_id: str = Field(..., description="UUID identifying this problem")
    text: str
    difficulty: ProblemDifficulty
    topic: ProblemTopic
    constraints: str
    examples: list[str]
    asked_at: datetime


class SystemDesignQuestion(BaseModel):
    question_id: str = Field(..., description="UUID identifying this question")
    text: str
    system_name: str = Field(..., description="e.g., 'Twitter', 'URL Shortener'")
    description: str
    asked_at: datetime


class ComplexityAnalysis(BaseModel):
    time_complexity: str = Field(..., description="Big-O notation")
    space_complexity: str = Field(..., description="Big-O notation")
    is_optimal: bool
    optimization_suggestions: Optional[str] = None


class TechnicalEvaluation(BaseModel):
    evaluation_id: str = Field(..., description="UUID identifying this evaluation")
    question_id: str
    response_id: str
    
    # DSA-specific fields
    correctness: Optional[str] = None  # "correct", "incorrect", "partial"
    complexity_analysis: Optional[ComplexityAnalysis] = None
    edge_cases_handled: list[str] = Field(default_factory=list)
    edge_cases_missed: list[str] = Field(default_factory=list)
    code_quality_notes: Optional[str] = None
    solution_format: Optional[SolutionFormat] = None
    
    # System Design-specific fields
    design_aspects_evaluated: dict[DesignAspect, str] = Field(default_factory=dict)
    design_strengths: list[str] = Field(default_factory=list)
    design_weaknesses: list[str] = Field(default_factory=list)
    
    # Common fields
    follow_up_warranted: bool = False
    follow_up_text: Optional[str] = None
    difficulty_signal: str = "maintain"  # "increase", "maintain", "decrease"
    evaluated_at: datetime
