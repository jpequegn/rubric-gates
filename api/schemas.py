"""Request and response schemas for the REST API.

Thin wrappers around shared models to define API-specific fields
(e.g. optional inputs, response envelopes).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from shared.models import GateResult, GateTier, ScoreResult, ScorecardSummary, ToolTier


# --- Requests ---


class ScoreRequest(BaseModel):
    """Request body for POST /score."""

    code: str = Field(..., min_length=1, description="Source code to score.")
    filename: str = Field(default="untitled.py", description="Filename for context.")
    user: str = Field(default="", description="User who generated the code.")
    skill_used: str = Field(default="", description="Claude Code skill used.")


class GateRequest(BaseModel):
    """Request body for POST /gate."""

    code: str = Field(..., min_length=1, description="Source code to evaluate.")
    filename: str = Field(default="untitled.py", description="Filename for context.")
    user: str = Field(default="", description="User who generated the code.")
    profile: str = Field(default="", description="Threshold profile to use.")


class ToolScoreRequest(BaseModel):
    """Request body for POST /tools/{slug}/score."""

    code: str = Field(..., min_length=1, description="Source code to score.")
    filename: str = Field(default="untitled.py", description="Filename for context.")
    user: str = Field(default="", description="User who generated the code.")


# --- Responses ---


class ScoreResponse(BaseModel):
    """Response for POST /score."""

    composite_score: float
    dimensions: dict[str, float] = Field(default_factory=dict)
    result: ScoreResult


class GateResponse(BaseModel):
    """Response for POST /gate."""

    tier: GateTier
    blocked: bool
    findings_count: int
    advisory_messages: list[str] = Field(default_factory=list)
    result: GateResult


class ToolSummary(BaseModel):
    """Abbreviated tool info for GET /tools list."""

    name: str
    slug: str
    tier: ToolTier
    latest_composite: float = 0.0
    tags: list[str] = Field(default_factory=list)


class ToolDetailResponse(BaseModel):
    """Response for GET /tools/{slug}."""

    name: str
    slug: str
    description: str
    tier: ToolTier
    created_by: str
    tech_owner: str | None
    users: list[str] = Field(default_factory=list)
    scorecard: ScorecardSummary
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolScoreResponse(BaseModel):
    """Response for POST /tools/{slug}/score."""

    slug: str
    score: ScoreResult
    updated_scorecard: ScorecardSummary


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    version: str = ""


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
