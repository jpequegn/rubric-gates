"""Tests for the composite scorer and rubric engine."""

import time
from unittest.mock import patch

import pytest

from scorecard.engine import RubricEngine
from shared.config import DimensionConfig, RubricGatesConfig, ScorecardConfig
from shared.models import Dimension


@pytest.fixture()
def default_engine():
    """Engine with default config (all dimensions enabled, LLM disabled)."""
    return RubricEngine()


@pytest.fixture()
def custom_config():
    """Config with specific weights and some dimensions disabled."""
    return RubricGatesConfig(
        scorecard=ScorecardConfig(
            dimensions={
                "correctness": DimensionConfig(weight=0.5, enabled=True),
                "security": DimensionConfig(weight=0.3, enabled=True),
                "maintainability": DimensionConfig(weight=0.2, enabled=True),
                "documentation": DimensionConfig(weight=0.0, enabled=False),
                "testability": DimensionConfig(weight=0.0, enabled=False),
            }
        )
    )


# --- Engine Initialization ---


class TestEngineInit:
    def test_default_loads_all_scorers(self, default_engine):
        assert Dimension.CORRECTNESS in default_engine._scorers
        assert Dimension.SECURITY in default_engine._scorers
        assert Dimension.MAINTAINABILITY in default_engine._scorers
        assert Dimension.DOCUMENTATION in default_engine._scorers
        assert Dimension.TESTABILITY in default_engine._scorers

    def test_disabled_dimensions_excluded(self, custom_config):
        engine = RubricEngine(config=custom_config)
        assert Dimension.CORRECTNESS in engine._scorers
        assert Dimension.SECURITY in engine._scorers
        assert Dimension.MAINTAINABILITY in engine._scorers
        assert Dimension.DOCUMENTATION not in engine._scorers
        assert Dimension.TESTABILITY not in engine._scorers

    def test_loads_config_from_defaults(self):
        engine = RubricEngine()
        assert engine.config is not None
        assert engine.scorecard_config is not None


# --- Scoring ---


class TestScoring:
    def test_score_clean_code(self, default_engine):
        code = """\
\"\"\"Math utilities.\"\"\"


def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b


def multiply(a: int, b: int) -> int:
    \"\"\"Multiply two numbers.\"\"\"
    return a * b
"""
        result = default_engine.score(code, "math_utils.py", user="alice")

        assert result.user == "alice"
        assert result.files_touched == ["math_utils.py"]
        assert 0.0 <= result.composite_score <= 1.0
        assert len(result.dimension_scores) == 5  # All 5 scorers
        assert result.composite_score > 0.5  # Clean code should score well

    def test_score_bad_code(self, default_engine):
        code = "def broken( return"
        result = default_engine.score(code, "bad.py", user="bob")

        assert result.composite_score < 0.5
        assert result.user == "bob"

    def test_score_empty_code(self, default_engine):
        result = default_engine.score("", "empty.py", user="test")
        assert result.composite_score == 0.0

    def test_score_with_metadata(self, default_engine):
        code = "x = 1"
        result = default_engine.score(
            code,
            "test.py",
            user="alice",
            metadata={"session_id": "abc123"},
        )
        assert result.metadata["session_id"] == "abc123"

    def test_score_with_skill(self, default_engine):
        code = "x = 1"
        result = default_engine.score(
            code,
            "test.py",
            user="alice",
            skill_used="scaffold-api",
        )
        assert result.skill_used == "scaffold-api"

    def test_score_auto_user(self, default_engine):
        code = "x = 1"
        result = default_engine.score(code, "test.py")
        assert result.user != ""  # Should auto-detect

    def test_score_no_filename(self, default_engine):
        code = "x = 1\nprint(x)"
        result = default_engine.score(code, user="test")
        assert result.files_touched == []
        assert result.composite_score > 0.0


# --- Composite Score Calculation ---


class TestCompositeScore:
    def test_weights_are_applied(self, custom_config):
        engine = RubricEngine(config=custom_config)
        code = """\
def add(a: int, b: int) -> int:
    return a + b
"""
        result = engine.score(code, "utils.py", user="test")

        # Should only have 3 dimension scores (doc + test disabled)
        assert len(result.dimension_scores) == 3
        dims = {ds.dimension for ds in result.dimension_scores}
        assert Dimension.CORRECTNESS in dims
        assert Dimension.SECURITY in dims
        assert Dimension.MAINTAINABILITY in dims

    def test_all_perfect_scores_give_1(self, custom_config):
        engine = RubricEngine(config=custom_config)
        # Simple, clean code should score near 1.0 for rule-based scorers
        code = """\
def add(a: int, b: int) -> int:
    return a + b
"""
        result = engine.score(code, "utils.py", user="test")
        # Correctness + security + maintainability should all be high
        for ds in result.dimension_scores:
            assert ds.score >= 0.8, f"{ds.dimension.value} scored {ds.score}"

    def test_composite_between_0_and_1(self, default_engine):
        code = "x = 1\nprint(x)"
        result = default_engine.score(code, "test.py", user="test")
        assert 0.0 <= result.composite_score <= 1.0


# --- Error Handling ---


class TestErrorHandling:
    def test_scorer_failure_graceful(self, default_engine):
        """If a scorer raises, the engine should still return a result."""
        code = "x = 1"

        with patch.object(
            default_engine._scorers[Dimension.CORRECTNESS],
            "score",
            side_effect=RuntimeError("scorer broke"),
        ):
            result = default_engine.score(code, "test.py", user="test")

        assert result.composite_score >= 0.0
        # Should have a zero-score entry for correctness
        correctness = next(
            ds for ds in result.dimension_scores if ds.dimension == Dimension.CORRECTNESS
        )
        assert correctness.score == 0.0
        assert "Scorer error" in correctness.details
        assert "scorer_errors" in result.metadata

    def test_all_scorers_fail(self, default_engine):
        """If all scorers fail, composite should be 0.0."""
        code = "x = 1"

        for scorer in default_engine._scorers.values():
            with patch.object(scorer, "score", side_effect=RuntimeError("broken")):
                pass

        # Patch all at once
        patches = []
        for scorer in default_engine._scorers.values():
            p = patch.object(scorer, "score", side_effect=RuntimeError("broken"))
            patches.append(p)

        for p in patches:
            p.start()
        try:
            result = default_engine.score(code, "test.py", user="test")
            assert result.composite_score == 0.0
            assert len(result.metadata["scorer_errors"]) == 5
        finally:
            for p in patches:
                p.stop()


# --- File Scoring ---


class TestFileScoring:
    def test_score_file(self, default_engine, tmp_path):
        f = tmp_path / "utils.py"
        f.write_text("def add(a, b):\n    return a + b\n")

        result = default_engine.score_file(f, user="alice")
        assert result.files_touched == ["utils.py"]
        assert result.composite_score > 0.0

    def test_score_files(self, default_engine, tmp_path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("x = 1\n")
        f2.write_text("y = 2\n")

        results = default_engine.score_files([f1, f2], user="alice")
        assert len(results) == 2
        assert results[0].files_touched == ["a.py"]
        assert results[1].files_touched == ["b.py"]

    def test_score_files_passes_project_files(self, default_engine, tmp_path):
        code_file = tmp_path / "utils.py"
        test_file = tmp_path / "test_utils.py"
        code_file.write_text("def add(a, b):\n    return a + b\n")
        test_file.write_text("def test_add():\n    assert add(1, 2) == 3\n")

        results = default_engine.score_files([code_file, test_file], user="test")
        assert len(results) == 2


# --- Performance ---


class TestPerformance:
    def test_rule_based_under_100ms(self, default_engine):
        """Rule-based scorers should complete quickly."""
        code = """\
import os
from pathlib import Path

def process(filename: str) -> str:
    path = Path(filename)
    if not path.exists():
        return ""
    with open(path) as f:
        return f.read()

class FileHandler:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def list_files(self) -> list[str]:
        return [f.name for f in self.base_dir.iterdir()]
"""
        start = time.perf_counter()
        result = default_engine.score(code, "handler.py", user="test")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result.composite_score > 0.0
        assert elapsed_ms < 100, f"Scoring took {elapsed_ms:.1f}ms (should be <100ms)"


# --- Dimension Coverage ---


class TestDimensionCoverage:
    def test_all_dimensions_scored(self, default_engine):
        code = "def foo():\n    return 1"
        result = default_engine.score(code, "test.py", user="test")

        scored_dims = {ds.dimension for ds in result.dimension_scores}
        assert Dimension.CORRECTNESS in scored_dims
        assert Dimension.SECURITY in scored_dims
        assert Dimension.MAINTAINABILITY in scored_dims
        assert Dimension.DOCUMENTATION in scored_dims
        assert Dimension.TESTABILITY in scored_dims

    def test_each_dimension_has_details(self, default_engine):
        code = "def foo():\n    return 1"
        result = default_engine.score(code, "test.py", user="test")

        for ds in result.dimension_scores:
            assert ds.details != "", f"{ds.dimension.value} has empty details"

    def test_each_dimension_has_method(self, default_engine):
        code = "def foo():\n    return 1"
        result = default_engine.score(code, "test.py", user="test")

        for ds in result.dimension_scores:
            assert ds.method is not None
