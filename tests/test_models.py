"""Tests for shared data models."""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from shared.models import (
    Dimension,
    DimensionScore,
    GateResult,
    GateTier,
    GraduationEvent,
    PatternFinding,
    ScoreResult,
    ScorecardSummary,
    ScoringMethod,
    ToolRegistryEntry,
    ToolTier,
)


# --- DimensionScore ---


class TestDimensionScore:
    def test_create_valid(self):
        score = DimensionScore(
            dimension=Dimension.CORRECTNESS,
            score=0.85,
            method=ScoringMethod.AST_PARSE,
            details="Syntax valid, no bare excepts",
        )
        assert score.dimension == Dimension.CORRECTNESS
        assert score.score == 0.85
        assert score.method == ScoringMethod.AST_PARSE
        assert score.details == "Syntax valid, no bare excepts"

    def test_score_bounds_low(self):
        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=Dimension.SECURITY,
                score=-0.1,
                method=ScoringMethod.RULE_BASED,
            )

    def test_score_bounds_high(self):
        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=Dimension.SECURITY,
                score=1.1,
                method=ScoringMethod.RULE_BASED,
            )

    def test_score_boundary_zero(self):
        score = DimensionScore(
            dimension=Dimension.SECURITY,
            score=0.0,
            method=ScoringMethod.RULE_BASED,
        )
        assert score.score == 0.0

    def test_score_boundary_one(self):
        score = DimensionScore(
            dimension=Dimension.SECURITY,
            score=1.0,
            method=ScoringMethod.RULE_BASED,
        )
        assert score.score == 1.0

    def test_default_metadata(self):
        score = DimensionScore(
            dimension=Dimension.DOCUMENTATION,
            score=0.5,
            method=ScoringMethod.LLM_JUDGE,
        )
        assert score.metadata == {}
        assert score.details == ""

    def test_with_metadata(self):
        score = DimensionScore(
            dimension=Dimension.TESTABILITY,
            score=0.7,
            method=ScoringMethod.HYBRID,
            metadata={"model": "claude-sonnet", "latency_ms": 230},
        )
        assert score.metadata["model"] == "claude-sonnet"

    def test_json_roundtrip(self):
        score = DimensionScore(
            dimension=Dimension.MAINTAINABILITY,
            score=0.65,
            method=ScoringMethod.AST_PARSE,
            details="Cyclomatic complexity: 12",
        )
        json_str = score.model_dump_json()
        restored = DimensionScore.model_validate_json(json_str)
        assert restored == score

    def test_all_dimensions(self):
        for dim in Dimension:
            score = DimensionScore(dimension=dim, score=0.5, method=ScoringMethod.RULE_BASED)
            assert score.dimension == dim

    def test_all_methods(self):
        for method in ScoringMethod:
            score = DimensionScore(dimension=Dimension.CORRECTNESS, score=0.5, method=method)
            assert score.method == method


# --- ScoreResult ---


class TestScoreResult:
    def test_create_minimal(self):
        result = ScoreResult(user="jane.doe")
        assert result.user == "jane.doe"
        assert result.skill_used == ""
        assert result.files_touched == []
        assert result.dimension_scores == []
        assert result.composite_score == 0.0

    def test_create_full(self):
        scores = [
            DimensionScore(
                dimension=Dimension.CORRECTNESS,
                score=0.9,
                method=ScoringMethod.AST_PARSE,
            ),
            DimensionScore(
                dimension=Dimension.SECURITY,
                score=0.8,
                method=ScoringMethod.RULE_BASED,
            ),
        ]
        result = ScoreResult(
            user="bob.smith",
            skill_used="scaffold-api",
            files_touched=["main.py", "utils.py"],
            dimension_scores=scores,
            composite_score=0.85,
            metadata={"session_id": "abc123"},
        )
        assert len(result.dimension_scores) == 2
        assert result.composite_score == 0.85
        assert result.metadata["session_id"] == "abc123"

    def test_timestamp_auto_set(self):
        before = datetime.now()
        result = ScoreResult(user="test")
        after = datetime.now()
        assert before <= result.timestamp <= after

    def test_composite_score_bounds(self):
        with pytest.raises(ValidationError):
            ScoreResult(user="test", composite_score=1.5)

    def test_json_roundtrip(self):
        result = ScoreResult(
            user="jane.doe",
            skill_used="generate-tool",
            files_touched=["app.py"],
            dimension_scores=[
                DimensionScore(
                    dimension=Dimension.CORRECTNESS,
                    score=0.9,
                    method=ScoringMethod.AST_PARSE,
                )
            ],
            composite_score=0.9,
        )
        json_str = result.model_dump_json()
        restored = ScoreResult.model_validate_json(json_str)
        assert restored.user == result.user
        assert restored.composite_score == result.composite_score
        assert len(restored.dimension_scores) == 1

    def test_dict_roundtrip(self):
        result = ScoreResult(user="test", composite_score=0.75)
        d = result.model_dump()
        assert isinstance(d, dict)
        restored = ScoreResult.model_validate(d)
        assert restored.user == "test"


# --- GateTier ---


class TestGateTier:
    def test_values(self):
        assert GateTier.GREEN == "green"
        assert GateTier.YELLOW == "yellow"
        assert GateTier.RED == "red"

    def test_all_tiers(self):
        assert len(GateTier) == 3


# --- PatternFinding ---


class TestPatternFinding:
    def test_create(self):
        finding = PatternFinding(
            pattern="hardcoded_credentials",
            severity="critical",
            line_number=23,
            line_content='API_KEY = "sk-abc123"',
            description="Hardcoded API key found",
            remediation="Use environment variables instead",
        )
        assert finding.pattern == "hardcoded_credentials"
        assert finding.severity == "critical"
        assert finding.line_number == 23

    def test_minimal(self):
        finding = PatternFinding(pattern="sql_injection", severity="critical")
        assert finding.line_number == 0
        assert finding.line_content == ""


# --- GateResult ---


class TestGateResult:
    def test_green_result(self):
        score = ScoreResult(user="test", composite_score=0.85)
        result = GateResult(
            tier=GateTier.GREEN,
            score_result=score,
            blocked=False,
        )
        assert result.tier == GateTier.GREEN
        assert not result.blocked
        assert result.critical_patterns_found == []

    def test_red_result_with_findings(self):
        score = ScoreResult(user="test", composite_score=0.4)
        result = GateResult(
            tier=GateTier.RED,
            score_result=score,
            critical_patterns_found=["hardcoded_credentials", "sql_injection"],
            pattern_findings=[
                PatternFinding(
                    pattern="hardcoded_credentials",
                    severity="critical",
                    line_number=10,
                )
            ],
            advisory_messages=["Remove hardcoded API key on line 10"],
            blocked=True,
        )
        assert result.tier == GateTier.RED
        assert result.blocked
        assert len(result.critical_patterns_found) == 2
        assert len(result.pattern_findings) == 1

    def test_yellow_result(self):
        score = ScoreResult(user="test", composite_score=0.6)
        result = GateResult(
            tier=GateTier.YELLOW,
            score_result=score,
            advisory_messages=["Consider adding tests", "Function too complex"],
            blocked=False,
        )
        assert result.tier == GateTier.YELLOW
        assert not result.blocked
        assert len(result.advisory_messages) == 2

    def test_json_roundtrip(self):
        score = ScoreResult(user="test", composite_score=0.5)
        result = GateResult(
            tier=GateTier.RED,
            score_result=score,
            blocked=True,
            critical_patterns_found=["eval_usage"],
        )
        json_str = result.model_dump_json()
        restored = GateResult.model_validate_json(json_str)
        assert restored.tier == GateTier.RED
        assert restored.blocked
        assert restored.critical_patterns_found == ["eval_usage"]


# --- ToolTier ---


class TestToolTier:
    def test_values(self):
        assert ToolTier.T0 == "T0"
        assert ToolTier.T1 == "T1"
        assert ToolTier.T2 == "T2"
        assert ToolTier.T3 == "T3"

    def test_ordering(self):
        tiers = [ToolTier.T3, ToolTier.T0, ToolTier.T2, ToolTier.T1]
        sorted_tiers = sorted(tiers, key=lambda t: t.value)
        assert sorted_tiers == [ToolTier.T0, ToolTier.T1, ToolTier.T2, ToolTier.T3]


# --- GraduationEvent ---


class TestGraduationEvent:
    def test_create(self):
        event = GraduationEvent(
            from_tier=ToolTier.T0,
            to_tier=ToolTier.T1,
            reason="Second user started using tool",
            approved_by="auto",
        )
        assert event.from_tier == ToolTier.T0
        assert event.to_tier == ToolTier.T1

    def test_timestamp_auto_set(self):
        before = datetime.now()
        event = GraduationEvent(from_tier=ToolTier.T1, to_tier=ToolTier.T2)
        after = datetime.now()
        assert before <= event.date <= after

    def test_json_roundtrip(self):
        event = GraduationEvent(
            from_tier=ToolTier.T1,
            to_tier=ToolTier.T2,
            reason="Daily usage by 3+ users",
            approved_by="john.doe",
        )
        json_str = event.model_dump_json()
        restored = GraduationEvent.model_validate_json(json_str)
        assert restored.from_tier == ToolTier.T1
        assert restored.approved_by == "john.doe"


# --- ScorecardSummary ---


class TestScorecardSummary:
    def test_defaults(self):
        summary = ScorecardSummary()
        assert summary.latest_composite == 0.0
        assert summary.trend == "stable"
        assert summary.total_scores == 0
        assert summary.red_flags == 0

    def test_with_scores(self):
        summary = ScorecardSummary(
            latest_composite=0.72,
            latest_scores={"correctness": 0.85, "security": 0.90, "maintainability": 0.55},
            trend="improving",
            total_scores=23,
            red_flags=0,
        )
        assert summary.latest_composite == 0.72
        assert summary.latest_scores["correctness"] == 0.85


# --- ToolRegistryEntry ---


class TestToolRegistryEntry:
    def test_create_minimal(self):
        entry = ToolRegistryEntry(name="Expense Categorizer")
        assert entry.name == "Expense Categorizer"
        assert entry.tier == ToolTier.T0
        assert entry.tech_owner is None
        assert entry.users == []
        assert entry.graduation_history == []

    def test_create_full(self):
        entry = ToolRegistryEntry(
            name="Expense Categorizer",
            slug="expense-categorizer",
            description="Categorizes expense reports by department",
            tier=ToolTier.T1,
            created_by="jane.doe",
            tech_owner=None,
            users=["jane.doe", "bob.smith"],
            source_path="/Users/jane/tools/expense-categorizer/",
            scorecard=ScorecardSummary(latest_composite=0.72, trend="improving"),
            graduation_history=[
                GraduationEvent(
                    from_tier=ToolTier.T0,
                    to_tier=ToolTier.T1,
                    reason="Second user started using tool",
                    approved_by="auto",
                )
            ],
            tags=["finance", "automation"],
        )
        assert entry.tier == ToolTier.T1
        assert len(entry.users) == 2
        assert len(entry.graduation_history) == 1
        assert entry.scorecard.trend == "improving"
        assert "finance" in entry.tags

    def test_json_roundtrip(self):
        entry = ToolRegistryEntry(
            name="Data Processor",
            slug="data-processor",
            tier=ToolTier.T2,
            created_by="alice",
            tech_owner="bob",
            users=["alice", "charlie"],
        )
        json_str = entry.model_dump_json()
        restored = ToolRegistryEntry.model_validate_json(json_str)
        assert restored.name == "Data Processor"
        assert restored.tier == ToolTier.T2
        assert restored.tech_owner == "bob"

    def test_json_parseable_from_string(self):
        """Verify models can be loaded from raw JSON strings (e.g., JSONL files)."""
        raw = json.dumps(
            {
                "name": "Test Tool",
                "slug": "test-tool",
                "tier": "T0",
                "created_by": "user",
                "created_date": "2026-01-15T10:00:00",
                "users": [],
                "scorecard": {"latest_composite": 0.5, "trend": "stable"},
                "graduation_history": [],
                "tags": [],
                "metadata": {},
            }
        )
        entry = ToolRegistryEntry.model_validate_json(raw)
        assert entry.name == "Test Tool"
        assert entry.tier == ToolTier.T0

    def test_metadata_extensibility(self):
        entry = ToolRegistryEntry(
            name="Custom Tool",
            metadata={"custom_field": "value", "priority": 1, "nested": {"a": "b"}},
        )
        assert entry.metadata["custom_field"] == "value"
        assert entry.metadata["nested"]["a"] == "b"
