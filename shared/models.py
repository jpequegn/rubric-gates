"""Shared data models for rubric-gates.

Core Pydantic models used across the scorecard, gate, and registry projects.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# --- Enums ---


class Dimension(str, Enum):
    """Scoring dimensions for code quality evaluation."""

    CORRECTNESS = "correctness"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"
    TESTABILITY = "testability"


class ScoringMethod(str, Enum):
    """Method used to produce a dimension score."""

    AST_PARSE = "ast_parse"
    RULE_BASED = "rule_based"
    LLM_JUDGE = "llm_judge"
    HYBRID = "hybrid"


class GateTier(str, Enum):
    """Gate classification tiers."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class ToolTier(str, Enum):
    """Tool lifecycle graduation tiers."""

    T0 = "T0"  # Personal
    T1 = "T1"  # Shared (registered)
    T2 = "T2"  # Team (owned)
    T3 = "T3"  # Critical (production)


# --- Scorecard Models ---


class DimensionScore(BaseModel):
    """Score for a single quality dimension."""

    dimension: Dimension
    score: float = Field(ge=0.0, le=1.0)
    method: ScoringMethod
    details: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoreResult(BaseModel):
    """Complete scoring result for a code generation event."""

    timestamp: datetime = Field(default_factory=datetime.now)
    user: str
    skill_used: str = ""
    files_touched: list[str] = Field(default_factory=list)
    dimension_scores: list[DimensionScore] = Field(default_factory=list)
    composite_score: float = Field(ge=0.0, le=1.0, default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- Gate Models ---


class PatternFinding(BaseModel):
    """A single finding from a critical pattern detector."""

    pattern: str
    severity: str  # "critical", "high", "medium"
    line_number: int = 0
    line_content: str = ""
    description: str = ""
    remediation: str = ""


class GateResult(BaseModel):
    """Result of gate tier evaluation."""

    tier: GateTier
    score_result: ScoreResult
    critical_patterns_found: list[str] = Field(default_factory=list)
    pattern_findings: list[PatternFinding] = Field(default_factory=list)
    advisory_messages: list[str] = Field(default_factory=list)
    blocked: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class OverrideRecord(BaseModel):
    """Audit record for a gate override."""

    id: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    user: str
    filename: str = ""
    gate_result: GateResult
    justification: str = ""
    override_type: str = "proceed"  # "proceed" | "escalate"
    reviewed_by: str = ""
    review_outcome: str = "pending"  # "approved" | "rejected" | "pending"


# --- Registry Models ---


class GraduationEvent(BaseModel):
    """Record of a tool's tier promotion."""

    from_tier: ToolTier
    to_tier: ToolTier
    date: datetime = Field(default_factory=datetime.now)
    reason: str = ""
    approved_by: str = ""


class ScorecardSummary(BaseModel):
    """Summary of a tool's scorecard history."""

    latest_composite: float = Field(ge=0.0, le=1.0, default=0.0)
    latest_scores: dict[str, float] = Field(default_factory=dict)
    trend: str = "stable"  # "improving", "stable", "declining"
    total_scores: int = 0
    red_flags: int = 0


class ToolRegistryEntry(BaseModel):
    """A registered tool in the catalog."""

    name: str
    slug: str = ""
    description: str = ""
    tier: ToolTier = ToolTier.T0
    created_by: str = ""
    created_date: datetime = Field(default_factory=datetime.now)
    tech_owner: str | None = None
    users: list[str] = Field(default_factory=list)
    source_path: str = ""
    repository: str | None = None
    scorecard: ScorecardSummary = Field(default_factory=ScorecardSummary)
    graduation_history: list[GraduationEvent] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
