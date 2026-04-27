"""Domain enums: Stage, Role, EvaluationDimension, QuestionType, InterviewRoundType, ProblemDifficulty, ProblemTopic, SolutionFormat, DesignPhase, DesignAspect."""

from enum import Enum


class Stage(str, Enum):
    INIT = "INIT"
    ROLE_SELECTION = "ROLE_SELECTION"
    INTERVIEW = "INTERVIEW"
    FEEDBACK = "FEEDBACK"
    COMPLETE = "COMPLETE"


class Role(str, Enum):
    SOFTWARE_ENGINEER = "software_engineer"
    SALES_REPRESENTATIVE = "sales_representative"
    RETAIL_ASSOCIATE = "retail_associate"
    UNKNOWN = "unknown"


class EvaluationDimension(str, Enum):
    COMMUNICATION_CLARITY = "communication_clarity"
    RELEVANCE = "relevance"
    TECHNICAL_KNOWLEDGE = "technical_knowledge"
    CONFIDENCE = "confidence"


class QuestionType(str, Enum):
    BEHAVIOURAL = "behavioural"
    SITUATIONAL = "situational"
    TECHNICAL = "technical"
    FOLLOW_UP = "follow_up"


class InterviewRoundType(str, Enum):
    DSA_CODING = "dsa_coding"
    SYSTEM_DESIGN = "system_design"
    BEHAVIORAL = "behavioral"


class ProblemDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ProblemTopic(str, Enum):
    ARRAYS = "arrays"
    STRINGS = "strings"
    LINKED_LISTS = "linked_lists"
    TREES = "trees"
    GRAPHS = "graphs"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    SORTING = "sorting"
    SEARCHING = "searching"
    HASH_TABLES = "hash_tables"
    STACKS_QUEUES = "stacks_queues"


class SolutionFormat(str, Enum):
    CODE = "code"
    PSEUDOCODE = "pseudocode"
    EXPLANATION = "explanation"


class DesignPhase(str, Enum):
    REQUIREMENTS_GATHERING = "requirements_gathering"
    HIGH_LEVEL_DESIGN = "high_level_design"
    DEEP_DIVE = "deep_dive"
    BOTTLENECK_ANALYSIS = "bottleneck_analysis"


class DesignAspect(str, Enum):
    SCALABILITY = "scalability"
    DATABASE_DESIGN = "database_design"
    API_DESIGN = "api_design"
    CACHING_STRATEGY = "caching_strategy"
    LOAD_BALANCING = "load_balancing"
