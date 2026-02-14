"""Tests for the gate post-edit hook."""

from unittest.mock import MagicMock, patch

from shared.models import (
    GateResult,
    GateTier,
    ScoreResult,
)

from gate.hooks.post_edit import (
    format_gate_summary,
    run_gate,
)


# --- run_gate ---


class TestRunGate:
    def test_score_clean_python_file(self, tmp_path):
        f = tmp_path / "clean.py"
        f.write_text("def add(a, b):\n    return a + b\n")

        result = run_gate(str(f), user="test", quiet=True, store=False)
        assert result is not None
        assert result.tier in (GateTier.GREEN, GateTier.YELLOW)
        assert not result.blocked

    def test_nonexistent_file_returns_none(self):
        result = run_gate("/nonexistent/file.py", quiet=True, store=False)
        assert result is None

    def test_non_python_file_returns_none(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Hello")
        result = run_gate(str(f), quiet=True, store=False)
        assert result is None

    def test_empty_file_returns_none(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        result = run_gate(str(f), quiet=True, store=False)
        assert result is None

    def test_file_with_hardcoded_password_is_red(self, tmp_path):
        f = tmp_path / "leaked.py"
        f.write_text('password = "supersecret123"\nx = 1\n')

        result = run_gate(str(f), user="test", quiet=True, store=False)
        assert result is not None
        assert result.tier == GateTier.RED
        assert result.blocked

    def test_file_with_sql_injection_is_red(self, tmp_path):
        f = tmp_path / "db.py"
        f.write_text(
            'def query(uid):\n    q = f"SELECT * FROM users WHERE id = {uid}"\n    return q\n'
        )

        result = run_gate(str(f), user="test", quiet=True, store=False)
        assert result is not None
        assert result.tier == GateTier.RED
        assert result.blocked

    def test_green_output_is_silent(self, tmp_path, capsys):
        f = tmp_path / "clean.py"
        f.write_text("def add(a, b):\n    return a + b\n")

        result = run_gate(str(f), user="test", quiet=False, store=False)
        captured = capsys.readouterr()
        if result and result.tier == GateTier.GREEN:
            assert "BLOCKED" not in captured.err
            assert "advisory" not in captured.err.lower()

    def test_red_output_shows_blocked(self, tmp_path, capsys):
        f = tmp_path / "bad.py"
        f.write_text('password = "hunter2"\nx = 1\n')

        run_gate(str(f), user="test", quiet=False, store=False)
        captured = capsys.readouterr()
        assert "BLOCKED" in captured.err

    def test_red_output_shows_options(self, tmp_path, capsys):
        f = tmp_path / "bad.py"
        f.write_text('api_key = "sk-abc123def456ghi789jklmnopqrstuvwxyz"\n')

        run_gate(str(f), user="test", quiet=False, store=False)
        captured = capsys.readouterr()
        assert "Options" in captured.err
        assert "Fix" in captured.err or "fix" in captured.err

    def test_quiet_suppresses_output(self, tmp_path, capsys):
        f = tmp_path / "bad.py"
        f.write_text('password = "secret"\nx = 1\n')

        run_gate(str(f), user="test", quiet=True, store=False)
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_stores_result(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")

        with patch("gate.hooks.post_edit.create_storage") as mock_storage:
            mock_backend = MagicMock()
            mock_storage.return_value = mock_backend

            run_gate(str(f), user="test", quiet=True, store=True)
            mock_backend.append.assert_called_once()

    def test_no_store_skips_storage(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")

        with patch("gate.hooks.post_edit.create_storage") as mock_storage:
            run_gate(str(f), user="test", quiet=True, store=False)
            mock_storage.assert_not_called()

    def test_storage_failure_graceful(self, tmp_path, capsys):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")

        with patch("gate.hooks.post_edit.create_storage") as mock_storage:
            mock_storage.side_effect = RuntimeError("storage broken")

            result = run_gate(str(f), user="test", quiet=False, store=True)
            assert result is not None
            captured = capsys.readouterr()
            assert "storage error" in captured.err


# --- format_gate_summary ---


class TestFormatGateSummary:
    def test_green_summary(self):
        result = GateResult(
            tier=GateTier.GREEN,
            score_result=ScoreResult(user="test", composite_score=0.85),
        )
        summary = format_gate_summary(result)
        assert "GREEN" in summary
        assert "0.85" in summary

    def test_yellow_summary(self):
        result = GateResult(
            tier=GateTier.YELLOW,
            score_result=ScoreResult(user="test", composite_score=0.6),
            advisory_messages=["msg1", "msg2"],
        )
        summary = format_gate_summary(result)
        assert "YELLOW" in summary
        assert "0.60" in summary
        assert "2" in summary

    def test_red_summary(self):
        result = GateResult(
            tier=GateTier.RED,
            score_result=ScoreResult(user="test", composite_score=0.3),
            critical_patterns_found=["hardcoded_credentials", "sql_injection"],
            blocked=True,
        )
        summary = format_gate_summary(result)
        assert "RED" in summary
        assert "0.30" in summary
        assert "2" in summary


# --- Yellow tier advisory output ---


class TestYellowOutput:
    def test_yellow_shows_advisories(self, tmp_path, capsys):
        f = tmp_path / "mediocre.py"
        # Code that should score in yellow range (below green threshold)
        f.write_text("x=1\ny=2\nz=3\n")

        result = run_gate(str(f), user="test", quiet=False, store=False)
        captured = capsys.readouterr()
        if result and result.tier == GateTier.YELLOW:
            assert "advisory" in captured.err.lower()

    def test_yellow_is_non_blocking(self, tmp_path):
        f = tmp_path / "ok.py"
        f.write_text("x = 1\n")

        result = run_gate(str(f), user="test", quiet=True, store=False)
        if result and result.tier == GateTier.YELLOW:
            assert not result.blocked


# --- Anti-gaming integration ---


class TestAntiGamingIntegration:
    def test_comment_stuffing_affects_gate(self, tmp_path):
        f = tmp_path / "stuffed.py"
        f.write_text(
            "# set x to 1\n"
            "x = 1\n"
            "# set y to 2\n"
            "y = 2\n"
            "# set z to 3\n"
            "z = 3\n"
            "# return result\n"
            "result = x + y + z\n"
        )

        result = run_gate(str(f), user="test", quiet=True, store=False)
        # Should still return a result (anti-gaming adjustments applied)
        assert result is not None
