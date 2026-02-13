"""Tests for Claude Code post-edit hook."""

from unittest.mock import MagicMock, patch

from scorecard.hooks.post_edit import format_score_line, run_scoring
from shared.models import Dimension, DimensionScore, ScoreResult, ScoringMethod


# --- format_score_line ---


class TestFormatScoreLine:
    def test_basic_format(self):
        result = ScoreResult(
            user="alice",
            composite_score=0.82,
            dimension_scores=[
                DimensionScore(
                    dimension=Dimension.CORRECTNESS, score=0.9, method=ScoringMethod.AST_PARSE
                ),
                DimensionScore(
                    dimension=Dimension.SECURITY, score=0.8, method=ScoringMethod.RULE_BASED
                ),
                DimensionScore(
                    dimension=Dimension.MAINTAINABILITY, score=0.7, method=ScoringMethod.RULE_BASED
                ),
                DimensionScore(
                    dimension=Dimension.DOCUMENTATION, score=0.8, method=ScoringMethod.RULE_BASED
                ),
                DimensionScore(
                    dimension=Dimension.TESTABILITY, score=0.9, method=ScoringMethod.RULE_BASED
                ),
            ],
        )
        line = format_score_line(result)
        assert "Score: 0.82" in line
        assert "C:0.9" in line
        assert "S:0.8" in line
        assert "M:0.7" in line
        assert "D:0.8" in line
        assert "T:0.9" in line
        assert "[" in line and "]" in line

    def test_empty_dimensions(self):
        result = ScoreResult(user="alice", composite_score=0.0)
        line = format_score_line(result)
        assert "Score: 0.00" in line

    def test_single_dimension(self):
        result = ScoreResult(
            user="alice",
            composite_score=0.5,
            dimension_scores=[
                DimensionScore(
                    dimension=Dimension.CORRECTNESS, score=0.5, method=ScoringMethod.AST_PARSE
                ),
            ],
        )
        line = format_score_line(result)
        assert "C:0.5" in line


# --- run_scoring ---


class TestRunScoring:
    def test_score_python_file(self, tmp_path):
        f = tmp_path / "utils.py"
        f.write_text("def add(a, b):\n    return a + b\n")

        result = run_scoring(str(f), user="test", quiet=True, store=False)
        assert result is not None
        assert result.composite_score > 0.0
        assert result.user == "test"

    def test_nonexistent_file(self):
        result = run_scoring("/nonexistent/file.py", quiet=True, store=False)
        assert result is None

    def test_non_python_file(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Hello")
        result = run_scoring(str(f), quiet=True, store=False)
        assert result is None

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        result = run_scoring(str(f), quiet=True, store=False)
        assert result is None

    def test_whitespace_only_file(self, tmp_path):
        f = tmp_path / "blank.py"
        f.write_text("   \n\n  ")
        result = run_scoring(str(f), quiet=True, store=False)
        assert result is None

    def test_output_to_stderr(self, tmp_path, capsys):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")

        run_scoring(str(f), user="test", quiet=False, store=False)
        captured = capsys.readouterr()
        assert "rubric-gates: Score:" in captured.err

    def test_quiet_suppresses_output(self, tmp_path, capsys):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")

        run_scoring(str(f), user="test", quiet=True, store=False)
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_stores_result(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")

        with patch("scorecard.hooks.post_edit.create_storage") as mock_storage:
            mock_backend = MagicMock()
            mock_storage.return_value = mock_backend

            run_scoring(str(f), user="test", quiet=True, store=True)
            mock_backend.append.assert_called_once()

    def test_no_store_skips_storage(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")

        with patch("scorecard.hooks.post_edit.create_storage") as mock_storage:
            run_scoring(str(f), user="test", quiet=True, store=False)
            mock_storage.assert_not_called()

    def test_storage_failure_graceful(self, tmp_path, capsys):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")

        with patch("scorecard.hooks.post_edit.create_storage") as mock_storage:
            mock_storage.side_effect = RuntimeError("storage broken")

            result = run_scoring(str(f), user="test", quiet=False, store=True)
            # Should still return a result despite storage failure
            assert result is not None
            captured = capsys.readouterr()
            assert "storage error" in captured.err

    def test_skill_passed_through(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")

        result = run_scoring(
            str(f),
            user="alice",
            skill_used="scaffold-api",
            quiet=True,
            store=False,
        )
        assert result is not None
        assert result.skill_used == "scaffold-api"

    def test_full_mode(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    return 1\n")

        result = run_scoring(str(f), user="test", full=True, quiet=True, store=False)
        assert result is not None
        assert result.composite_score > 0.0

    def test_scoring_error_graceful(self, tmp_path, capsys):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")

        with patch("scorecard.hooks.post_edit.RubricEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.score.side_effect = RuntimeError("engine broke")
            mock_engine_cls.return_value = mock_engine

            result = run_scoring(str(f), user="test", quiet=False, store=False)
            assert result is None
            captured = capsys.readouterr()
            assert "scoring error" in captured.err


# --- Quick mode config ---


class TestQuickMode:
    def test_quick_mode_is_default(self, tmp_path):
        """Quick mode (no --full) should still score all dimensions."""
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    return 1\n")

        result = run_scoring(str(f), user="test", full=False, quiet=True, store=False)
        assert result is not None
        dims = {ds.dimension for ds in result.dimension_scores}
        assert Dimension.CORRECTNESS in dims
        assert Dimension.SECURITY in dims

    def test_quick_mode_fast(self, tmp_path):
        """Quick mode should be fast (<100ms)."""
        import time

        f = tmp_path / "test.py"
        f.write_text("def foo():\n    return 1\n")

        start = time.perf_counter()
        run_scoring(str(f), user="test", full=False, quiet=True, store=False)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 200  # Allow some margin for test overhead
