"""Tests for the unified CLI orchestrator."""

from unittest.mock import MagicMock, patch

from cli import (
    _build_parser,
    _format_tier,
    _print_gate_result,
    _print_score_result,
    cmd_check,
    cmd_report,
    cmd_score,
    cmd_status,
    main,
)
from shared.models import (
    Dimension,
    DimensionScore,
    GateResult,
    GateTier,
    ScoreResult,
    ScoringMethod,
    ToolRegistryEntry,
    ToolTier,
)


# --- Helper ---


def _sample_code():
    return "def hello():\n    return 'world'\n"


def _write_py_file(tmp_path, code=None):
    """Write a sample Python file and return its path."""
    f = tmp_path / "sample.py"
    f.write_text(code or _sample_code())
    return f


def _make_score_result(composite=0.75, user="testuser"):
    return ScoreResult(
        user=user,
        composite_score=composite,
        files_touched=["sample.py"],
        dimension_scores=[
            DimensionScore(
                dimension=Dimension.CORRECTNESS,
                score=0.8,
                method=ScoringMethod.AST_PARSE,
            ),
            DimensionScore(
                dimension=Dimension.SECURITY,
                score=0.7,
                method=ScoringMethod.RULE_BASED,
            ),
        ],
    )


def _make_gate_result(tier=GateTier.GREEN, blocked=False):
    return GateResult(
        tier=tier,
        score_result=_make_score_result(),
        blocked=blocked,
        advisory_messages=["Looks good"] if tier == GateTier.GREEN else ["Needs work"],
        critical_patterns_found=["sql_injection"] if tier == GateTier.RED else [],
    )


# --- Parser tests ---


class TestBuildParser:
    def test_has_all_commands(self):
        parser = _build_parser()
        # Parse each subcommand to verify they exist
        args = parser.parse_args(["score", "test.py"])
        assert args.command == "score"
        assert args.file == "test.py"

    def test_score_args(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "score",
                "test.py",
                "--user",
                "alice",
                "--skill",
                "codegen",
                "--store",
                "--tool",
                "my-tool",
            ]
        )
        assert args.user == "alice"
        assert args.skill == "codegen"
        assert args.store is True
        assert args.tool == "my-tool"

    def test_check_args(self):
        parser = _build_parser()
        args = parser.parse_args(["check", "test.py", "--user", "bob"])
        assert args.command == "check"
        assert args.user == "bob"

    def test_status_args(self):
        parser = _build_parser()
        args = parser.parse_args(["status", "my-tool"])
        assert args.command == "status"
        assert args.slug == "my-tool"

    def test_report_args(self):
        parser = _build_parser()
        args = parser.parse_args(["report", "--since", "7d", "--user", "alice"])
        assert args.command == "report"
        assert args.since == "7d"
        assert args.user == "alice"

    def test_config_arg(self):
        parser = _build_parser()
        args = parser.parse_args(["--config", "/path/to/config.yaml", "check", "test.py"])
        assert args.config == "/path/to/config.yaml"


# --- Formatter tests ---


class TestFormatTier:
    def test_green(self):
        assert "GREEN" in _format_tier(GateTier.GREEN)
        assert "merge" in _format_tier(GateTier.GREEN).lower()

    def test_yellow(self):
        assert "YELLOW" in _format_tier(GateTier.YELLOW)

    def test_red(self):
        assert "RED" in _format_tier(GateTier.RED)
        assert "Blocked" in _format_tier(GateTier.RED)


class TestPrintScoreResult:
    def test_includes_file_and_score(self):
        result = _make_score_result(composite=0.75)
        output = _print_score_result(result)
        assert "sample.py" in output
        assert "0.75" in output

    def test_includes_dimensions(self):
        result = _make_score_result()
        output = _print_score_result(result)
        assert "correctness" in output
        assert "security" in output

    def test_handles_no_files(self):
        result = ScoreResult(user="test", files_touched=[])
        output = _print_score_result(result)
        assert "(none)" in output


class TestPrintGateResult:
    def test_green_output(self):
        gate = _make_gate_result(GateTier.GREEN)
        output = _print_gate_result(gate)
        assert "GREEN" in output

    def test_red_shows_blocked(self):
        gate = _make_gate_result(GateTier.RED, blocked=True)
        output = _print_gate_result(gate)
        assert "BLOCKED" in output
        assert "sql_injection" in output

    def test_shows_advisories(self):
        gate = _make_gate_result(GateTier.YELLOW)
        output = _print_gate_result(gate)
        assert "Needs work" in output


# --- Command tests ---


class TestCmdScore:
    @patch("cli.TierEvaluator")
    @patch("cli.RubricEngine")
    def test_runs_pipeline(self, mock_engine_cls, mock_eval_cls, tmp_path, capsys):
        f = _write_py_file(tmp_path)
        mock_engine = mock_engine_cls.return_value
        mock_engine.score.return_value = _make_score_result()
        mock_eval = mock_eval_cls.return_value
        mock_eval.evaluate.return_value = _make_gate_result(GateTier.GREEN)

        parser = _build_parser()
        args = parser.parse_args(["score", str(f)])
        args.config = None
        result = cmd_score(args)

        assert result == 0
        mock_engine.score.assert_called_once()
        mock_eval.evaluate.assert_called_once()

    @patch("cli.TierEvaluator")
    @patch("cli.RubricEngine")
    def test_blocked_returns_1(self, mock_engine_cls, mock_eval_cls, tmp_path):
        f = _write_py_file(tmp_path)
        mock_engine_cls.return_value.score.return_value = _make_score_result()
        mock_eval_cls.return_value.evaluate.return_value = _make_gate_result(
            GateTier.RED, blocked=True
        )

        parser = _build_parser()
        args = parser.parse_args(["score", str(f)])
        args.config = None
        result = cmd_score(args)
        assert result == 1

    @patch("cli.create_storage")
    @patch("cli.TierEvaluator")
    @patch("cli.RubricEngine")
    def test_store_flag(self, mock_engine_cls, mock_eval_cls, mock_storage_fn, tmp_path):
        f = _write_py_file(tmp_path)
        mock_engine_cls.return_value.score.return_value = _make_score_result()
        mock_eval_cls.return_value.evaluate.return_value = _make_gate_result()
        mock_storage = mock_storage_fn.return_value

        parser = _build_parser()
        args = parser.parse_args(["score", str(f), "--store"])
        args.config = None
        cmd_score(args)

        mock_storage.append.assert_called_once()

    @patch("cli.TierEvaluator")
    @patch("cli.RubricEngine")
    def test_tool_flag_updates_registry(self, mock_engine_cls, mock_eval_cls, tmp_path):
        f = _write_py_file(tmp_path)
        mock_engine_cls.return_value.score.return_value = _make_score_result()
        mock_eval_cls.return_value.evaluate.return_value = _make_gate_result()

        mock_catalog = MagicMock()
        with patch("registry.catalog.catalog.ToolCatalog", return_value=mock_catalog):
            parser = _build_parser()
            args = parser.parse_args(["score", str(f), "--tool", "my-tool"])
            args.config = None
            cmd_score(args)

        mock_catalog.update_scorecard.assert_called_once()


class TestCmdCheck:
    @patch("cli.TierEvaluator")
    @patch("cli.RubricEngine")
    def test_outputs_score_and_tier(self, mock_engine_cls, mock_eval_cls, tmp_path, capsys):
        f = _write_py_file(tmp_path)
        mock_engine_cls.return_value.score.return_value = _make_score_result(composite=0.82)
        mock_eval_cls.return_value.evaluate.return_value = _make_gate_result(GateTier.GREEN)

        parser = _build_parser()
        args = parser.parse_args(["check", str(f)])
        args.config = None
        result = cmd_check(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "0.82" in captured.out
        assert "GREEN" in captured.out

    @patch("cli.TierEvaluator")
    @patch("cli.RubricEngine")
    def test_blocked_returns_1(self, mock_engine_cls, mock_eval_cls, tmp_path):
        f = _write_py_file(tmp_path)
        mock_engine_cls.return_value.score.return_value = _make_score_result()
        mock_eval_cls.return_value.evaluate.return_value = _make_gate_result(
            GateTier.RED, blocked=True
        )

        parser = _build_parser()
        args = parser.parse_args(["check", str(f)])
        args.config = None
        assert cmd_check(args) == 1

    @patch("cli.TierEvaluator")
    @patch("cli.RubricEngine")
    def test_shows_advisories(self, mock_engine_cls, mock_eval_cls, tmp_path, capsys):
        f = _write_py_file(tmp_path)
        mock_engine_cls.return_value.score.return_value = _make_score_result()
        gate = _make_gate_result(GateTier.YELLOW)
        mock_eval_cls.return_value.evaluate.return_value = gate

        parser = _build_parser()
        args = parser.parse_args(["check", str(f)])
        args.config = None
        cmd_check(args)

        captured = capsys.readouterr()
        assert "Needs work" in captured.out


class TestCmdStatus:
    def test_found(self, tmp_path, capsys):
        tool = ToolRegistryEntry(
            name="My Tool",
            slug="my-tool",
            tier=ToolTier.T1,
            tech_owner="alice",
            users=["alice", "bob"],
            tags=["util"],
        )
        mock_catalog = MagicMock()
        mock_catalog.get.return_value = tool

        with patch("registry.catalog.catalog.ToolCatalog", return_value=mock_catalog):
            parser = _build_parser()
            args = parser.parse_args(["status", "my-tool"])
            args.config = None
            result = cmd_status(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "My Tool" in captured.out
        assert "T1" in captured.out
        assert "alice" in captured.out

    def test_not_found(self, capsys):
        mock_catalog = MagicMock()
        mock_catalog.get.return_value = None

        with patch("registry.catalog.catalog.ToolCatalog", return_value=mock_catalog):
            parser = _build_parser()
            args = parser.parse_args(["status", "nonexistent"])
            args.config = None
            result = cmd_status(args)

        assert result == 1


class TestCmdReport:
    @patch("cli.create_storage")
    def test_generates_report(self, mock_storage_fn, capsys):
        mock_storage = mock_storage_fn.return_value
        mock_storage.query.return_value = [
            _make_score_result(composite=0.75, user="alice"),
            _make_score_result(composite=0.55, user="bob"),
        ]

        parser = _build_parser()
        args = parser.parse_args(["report"])
        args.config = None
        result = cmd_report(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Overview" in captured.out
        assert "By User" in captured.out

    @patch("cli.create_storage")
    def test_with_since_filter(self, mock_storage_fn, capsys):
        mock_storage = mock_storage_fn.return_value
        mock_storage.query.return_value = []

        parser = _build_parser()
        args = parser.parse_args(["report", "--since", "7d"])
        args.config = None
        cmd_report(args)

        # Verify filters were passed
        call_args = mock_storage.query.call_args
        filters = call_args[0][0] if call_args[0] else call_args[1].get("filters")
        if filters:
            assert filters.start_date is not None

    @patch("cli.create_storage")
    def test_empty_report(self, mock_storage_fn, capsys):
        mock_storage_fn.return_value.query.return_value = []

        parser = _build_parser()
        args = parser.parse_args(["report"])
        args.config = None
        cmd_report(args)

        captured = capsys.readouterr()
        assert "No scores recorded" in captured.out


# --- Main entry point tests ---


class TestMain:
    def test_no_command_returns_1(self, capsys):
        result = main([])
        assert result == 1

    def test_unknown_command_returns_1(self):
        # argparse will raise SystemExit for unknown commands
        import pytest

        with pytest.raises(SystemExit):
            main(["unknown"])

    @patch("cli.cmd_check")
    def test_dispatches_check(self, mock_cmd):
        mock_cmd.return_value = 0
        result = main(["check", "test.py"])
        assert result == 0
        mock_cmd.assert_called_once()

    @patch("cli.cmd_score")
    def test_dispatches_score(self, mock_cmd):
        mock_cmd.return_value = 0
        result = main(["score", "test.py"])
        assert result == 0
        mock_cmd.assert_called_once()

    @patch("cli.cmd_status")
    def test_dispatches_status(self, mock_cmd):
        mock_cmd.return_value = 0
        result = main(["status", "my-tool"])
        assert result == 0
        mock_cmd.assert_called_once()

    @patch("cli.cmd_report")
    def test_dispatches_report(self, mock_cmd):
        mock_cmd.return_value = 0
        result = main(["report"])
        assert result == 0
        mock_cmd.assert_called_once()
