"""Tests for the monitoring dashboard CLI."""

from datetime import datetime, timedelta

from scorecard.dashboard.cli import (
    _format_table,
    parse_since,
    view_by_skill,
    view_by_user,
    view_dimensions,
    view_overview,
    view_red_flags,
)
from shared.models import Dimension, DimensionScore, ScoreResult, ScoringMethod


# --- Helpers ---


def _make_result(
    user: str = "alice",
    score: float = 0.8,
    skill: str = "",
    dims: list[tuple[Dimension, float]] | None = None,
    files: list[str] | None = None,
    timestamp: datetime | None = None,
) -> ScoreResult:
    """Build a ScoreResult for testing."""
    dim_scores = []
    if dims:
        for dim, val in dims:
            dim_scores.append(
                DimensionScore(dimension=dim, score=val, method=ScoringMethod.RULE_BASED)
            )
    return ScoreResult(
        user=user,
        composite_score=score,
        dimension_scores=dim_scores,
        skill_used=skill,
        files_touched=files or [],
        timestamp=timestamp or datetime.now(),
    )


# --- parse_since ---


class TestParseSince:
    def test_days(self):
        result = parse_since("7d")
        expected = datetime.now() - timedelta(days=7)
        assert abs((result - expected).total_seconds()) < 2

    def test_hours(self):
        result = parse_since("24h")
        expected = datetime.now() - timedelta(hours=24)
        assert abs((result - expected).total_seconds()) < 2

    def test_weeks(self):
        result = parse_since("2w")
        expected = datetime.now() - timedelta(weeks=2)
        assert abs((result - expected).total_seconds()) < 2

    def test_iso_date(self):
        result = parse_since("2025-01-15")
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15

    def test_strips_whitespace(self):
        result = parse_since("  7d  ")
        expected = datetime.now() - timedelta(days=7)
        assert abs((result - expected).total_seconds()) < 2

    def test_case_insensitive(self):
        result = parse_since("7D")
        expected = datetime.now() - timedelta(days=7)
        assert abs((result - expected).total_seconds()) < 2


# --- _format_table ---


class TestFormatTable:
    def test_basic_table(self):
        headers = ["Name", "Score"]
        rows = [["alice", "0.82"], ["bob", "0.65"]]
        result = _format_table(headers, rows)
        assert "| Name" in result
        assert "| alice" in result
        assert "| bob" in result
        assert "|-" in result

    def test_empty_rows(self):
        result = _format_table(["Name"], [])
        assert "(no data)" in result

    def test_column_width_adapts(self):
        headers = ["X"]
        rows = [["long_value_here"]]
        result = _format_table(headers, rows)
        # The column should be wide enough for the value
        assert "long_value_here" in result

    def test_multiple_columns(self):
        headers = ["A", "B", "C"]
        rows = [["1", "2", "3"]]
        result = _format_table(headers, rows)
        lines = result.strip().split("\n")
        assert len(lines) == 3  # header + separator + 1 row


# --- view_overview ---


class TestViewOverview:
    def test_empty(self):
        result = view_overview([], "all time")
        assert "No scores recorded" in result

    def test_basic_stats(self):
        results = [
            _make_result(score=0.8),
            _make_result(score=0.6),
            _make_result(score=0.3),
        ]
        output = view_overview(results, "since 7d")
        assert "since 7d" in output
        assert "Files scored:** 3" in output
        assert "0.57" in output  # avg of 0.8, 0.6, 0.3
        assert "red flags" in output

    def test_all_green(self):
        results = [_make_result(score=0.9), _make_result(score=0.8)]
        output = view_overview(results, "all time")
        assert "red flags):** 0" in output

    def test_all_red(self):
        results = [_make_result(score=0.2), _make_result(score=0.3)]
        output = view_overview(results, "all time")
        assert "red flags):** 2" in output

    def test_percentage_calculation(self):
        results = [_make_result(score=0.8)] * 4 + [_make_result(score=0.3)]
        output = view_overview(results, "all time")
        assert "80%" in output  # 4/5 >= 0.7


# --- view_by_user ---


class TestViewByUser:
    def test_empty(self):
        result = view_by_user([])
        assert "No scores recorded" in result

    def test_single_user(self):
        results = [_make_result(user="alice", score=0.8)]
        output = view_by_user(results)
        assert "alice" in output
        assert "0.80" in output

    def test_multiple_users(self):
        results = [
            _make_result(user="alice", score=0.9),
            _make_result(user="bob", score=0.4),
            _make_result(user="bob", score=0.3),
        ]
        output = view_by_user(results)
        assert "alice" in output
        assert "bob" in output

    def test_sorted_by_red_flags(self):
        results = [
            _make_result(user="alice", score=0.9),
            _make_result(user="bob", score=0.3),
            _make_result(user="bob", score=0.4),
        ]
        output = view_by_user(results)
        lines = output.strip().split("\n")
        # bob has 2 red flags, should appear first in table body
        data_lines = [
            line
            for line in lines
            if line.startswith("|") and "User" not in line and "---" not in line
        ]
        assert "bob" in data_lines[0]


# --- view_by_skill ---


class TestViewBySkill:
    def test_empty(self):
        result = view_by_skill([])
        assert "No scores recorded" in result

    def test_single_skill(self):
        results = [
            _make_result(
                skill="scaffold-api",
                score=0.7,
                dims=[(Dimension.CORRECTNESS, 0.8), (Dimension.SECURITY, 0.5)],
            ),
        ]
        output = view_by_skill(results)
        assert "scaffold-api" in output
        assert "security" in output.lower() or "Security" in output

    def test_no_skill(self):
        results = [_make_result(skill="", score=0.8)]
        output = view_by_skill(results)
        assert "(none)" in output

    def test_worst_dimension_shown(self):
        results = [
            _make_result(
                skill="gen-tests",
                score=0.6,
                dims=[
                    (Dimension.CORRECTNESS, 0.9),
                    (Dimension.SECURITY, 0.3),
                ],
            ),
        ]
        output = view_by_skill(results)
        assert "security" in output


# --- view_red_flags ---


class TestViewRedFlags:
    def test_no_flags(self):
        results = [_make_result(score=0.8)]
        output = view_red_flags(results)
        assert "All clear" in output

    def test_flags_shown(self):
        results = [
            _make_result(
                user="bob",
                score=0.3,
                files=["bad_code.py"],
                dims=[(Dimension.SECURITY, 0.1)],
            ),
        ]
        output = view_red_flags(results)
        assert "bob" in output
        assert "0.30" in output
        assert "bad_code.py" in output

    def test_custom_threshold(self):
        results = [_make_result(score=0.6)]
        # Default threshold 0.5 — should not flag
        output_default = view_red_flags(results, threshold=0.5)
        assert "All clear" in output_default

        # Higher threshold — should flag
        output_high = view_red_flags(results, threshold=0.7)
        assert "1 scores below" in output_high

    def test_sorted_by_score(self):
        results = [
            _make_result(user="alice", score=0.4),
            _make_result(user="bob", score=0.2),
        ]
        output = view_red_flags(results)
        lines = output.strip().split("\n")
        data_lines = [
            line
            for line in lines
            if line.startswith("|") and "File" not in line and "---" not in line
        ]
        # bob (0.2) should appear before alice (0.4)
        assert "bob" in data_lines[0]

    def test_empty_results(self):
        output = view_red_flags([])
        assert "All clear" in output


# --- view_dimensions ---


class TestViewDimensions:
    def test_empty(self):
        result = view_dimensions([])
        assert "No scores recorded" in result

    def test_single_dimension(self):
        results = [
            _make_result(dims=[(Dimension.CORRECTNESS, 0.9)]),
        ]
        output = view_dimensions(results)
        assert "correctness" in output
        assert "0.90" in output
        assert "Strong" in output

    def test_assessment_levels(self):
        results = [
            _make_result(
                dims=[
                    (Dimension.CORRECTNESS, 0.9),  # Strong
                    (Dimension.SECURITY, 0.7),  # Adequate
                    (Dimension.MAINTAINABILITY, 0.45),  # Needs work
                    (Dimension.DOCUMENTATION, 0.2),  # Critical
                ]
            ),
        ]
        output = view_dimensions(results)
        assert "Strong" in output
        assert "Adequate" in output
        assert "Needs work" in output
        assert "Critical" in output

    def test_sorted_worst_first(self):
        results = [
            _make_result(
                dims=[
                    (Dimension.CORRECTNESS, 0.9),
                    (Dimension.SECURITY, 0.3),
                ]
            ),
        ]
        output = view_dimensions(results)
        lines = output.strip().split("\n")
        data_lines = [
            line
            for line in lines
            if line.startswith("|") and "Dimension" not in line and "---" not in line
        ]
        # security (0.3) should appear before correctness (0.9)
        assert "security" in data_lines[0]

    def test_low_count(self):
        results = [
            _make_result(dims=[(Dimension.CORRECTNESS, 0.3)]),
            _make_result(dims=[(Dimension.CORRECTNESS, 0.8)]),
        ]
        output = view_dimensions(results)
        # One score < 0.5
        assert "1" in output

    def test_multiple_results_averaged(self):
        results = [
            _make_result(dims=[(Dimension.CORRECTNESS, 0.6)]),
            _make_result(dims=[(Dimension.CORRECTNESS, 0.8)]),
        ]
        output = view_dimensions(results)
        assert "0.70" in output  # avg of 0.6 and 0.8
