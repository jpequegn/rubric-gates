"""Tests for configurable pattern detection rules via YAML (issue #59)."""

from __future__ import annotations

import pytest
import yaml

from gate.patterns.base import get_all_detectors, get_configured_detectors
from gate.patterns.custom import CustomPatternDetector, load_custom_detectors
from gate.tiers.evaluator import TierEvaluator
from shared.config import GateConfig, PatternRuleConfig, PatternsConfig, load_config
from shared.models import GateTier, ScoreResult


# --- PatternRuleConfig validation ---


class TestPatternRuleConfig:
    def test_valid_rule(self):
        rule = PatternRuleConfig(
            name="test_pattern",
            pattern=r"TODO",
            severity="medium",
            message="TODO found",
        )
        assert rule.name == "test_pattern"
        assert rule.severity == "medium"

    def test_default_severity(self):
        rule = PatternRuleConfig(name="test", pattern=r"test")
        assert rule.severity == "medium"

    def test_all_severities(self):
        for sev in ("critical", "high", "medium", "low"):
            rule = PatternRuleConfig(name="test", pattern=r"test", severity=sev)
            assert rule.severity == sev


# --- CustomPatternDetector ---


class TestCustomPatternDetector:
    def test_basic_detection(self):
        rule = PatternRuleConfig(
            name="todo_check",
            pattern=r"TODO",
            severity="low",
            message="TODO comment found",
        )
        detector = CustomPatternDetector(rule)
        findings = detector.detect("x = 1\n# TODO: fix this\ny = 2", "test.py")
        # Comments are skipped
        assert len(findings) == 0

    def test_detects_in_code_lines(self):
        rule = PatternRuleConfig(
            name="print_check",
            pattern=r"\bprint\s*\(",
            severity="low",
            message="Print statement found",
        )
        detector = CustomPatternDetector(rule)
        findings = detector.detect('print("hello")\nx = 1', "test.py")
        assert len(findings) == 1
        assert findings[0].pattern == "custom:print_check"
        assert findings[0].severity == "low"
        assert findings[0].line_number == 1

    def test_skips_comments(self):
        rule = PatternRuleConfig(name="test", pattern=r"secret", severity="high")
        detector = CustomPatternDetector(rule)
        findings = detector.detect("# secret stuff here\nx = 1", "test.py")
        assert len(findings) == 0

    def test_multiple_matches(self):
        rule = PatternRuleConfig(
            name="internal_url",
            pattern=r"https://internal\.corp\.com",
            severity="high",
            message="Internal URL detected",
        )
        detector = CustomPatternDetector(rule)
        code = (
            'url1 = "https://internal.corp.com/api"\nx = 1\nurl2 = "https://internal.corp.com/v2"\n'
        )
        findings = detector.detect(code, "test.py")
        assert len(findings) == 2
        assert findings[0].line_number == 1
        assert findings[1].line_number == 3

    def test_custom_message_and_remediation(self):
        rule = PatternRuleConfig(
            name="test",
            pattern=r"DEBUG",
            severity="medium",
            message="Debug flag found",
            remediation="Remove before production",
        )
        detector = CustomPatternDetector(rule)
        findings = detector.detect("DEBUG = True", "test.py")
        assert findings[0].description == "Debug flag found"
        assert findings[0].remediation == "Remove before production"

    def test_default_message(self):
        rule = PatternRuleConfig(name="mycheck", pattern=r"FIXME")
        detector = CustomPatternDetector(rule)
        findings = detector.detect("FIXME this", "test.py")
        assert "mycheck" in findings[0].description

    def test_invalid_severity_raises(self):
        rule = PatternRuleConfig(name="test", pattern=r"test", severity="unknown")
        with pytest.raises(ValueError, match="Invalid severity"):
            CustomPatternDetector(rule)

    def test_invalid_regex_raises(self):
        rule = PatternRuleConfig(name="test", pattern=r"[invalid", severity="medium")
        with pytest.raises(ValueError, match="Invalid regex"):
            CustomPatternDetector(rule)

    def test_no_match_returns_empty(self):
        rule = PatternRuleConfig(name="test", pattern=r"NONEXISTENT_PATTERN_XYZ")
        detector = CustomPatternDetector(rule)
        findings = detector.detect("x = 1\ny = 2", "test.py")
        assert findings == []

    def test_name_prefix(self):
        rule = PatternRuleConfig(name="my_rule", pattern=r"test")
        detector = CustomPatternDetector(rule)
        assert detector.name == "custom:my_rule"


# --- load_custom_detectors ---


class TestLoadCustomDetectors:
    def test_empty_list(self):
        detectors = load_custom_detectors([])
        assert detectors == []

    def test_multiple_rules(self):
        rules = [
            PatternRuleConfig(name="rule1", pattern=r"TODO", severity="low"),
            PatternRuleConfig(name="rule2", pattern=r"HACK", severity="medium"),
        ]
        detectors = load_custom_detectors(rules)
        assert len(detectors) == 2
        assert detectors[0].name == "custom:rule1"
        assert detectors[1].name == "custom:rule2"

    def test_invalid_rule_raises(self):
        rules = [PatternRuleConfig(name="bad", pattern=r"[broken", severity="medium")]
        with pytest.raises(ValueError):
            load_custom_detectors(rules)


# --- PatternsConfig ---


class TestPatternsConfig:
    def test_default_empty(self):
        config = PatternsConfig()
        assert config.custom == []
        assert config.disabled == []

    def test_with_custom_rules(self):
        config = PatternsConfig(
            custom=[
                PatternRuleConfig(name="test", pattern=r"TODO", severity="low"),
            ]
        )
        assert len(config.custom) == 1

    def test_with_disabled(self):
        config = PatternsConfig(disabled=["hardcoded_credentials"])
        assert "hardcoded_credentials" in config.disabled


# --- get_configured_detectors ---


class TestGetConfiguredDetectors:
    def test_default_config_returns_builtins(self):
        config = PatternsConfig()
        detectors = get_configured_detectors(config)
        builtin = get_all_detectors()
        assert len(detectors) == len(builtin)

    def test_custom_patterns_added(self):
        config = PatternsConfig(
            custom=[
                PatternRuleConfig(name="todo", pattern=r"TODO", severity="low"),
            ]
        )
        detectors = get_configured_detectors(config)
        builtin_count = len(get_all_detectors())
        assert len(detectors) == builtin_count + 1

        custom_names = [d.name for d in detectors if d.name.startswith("custom:")]
        assert "custom:todo" in custom_names

    def test_disable_builtin(self):
        config = PatternsConfig(disabled=["hardcoded_credentials"])
        detectors = get_configured_detectors(config)
        names = [d.name for d in detectors]
        assert "hardcoded_credentials" not in names

    def test_disable_multiple(self):
        builtin = get_all_detectors()
        builtin_names = [d.name for d in builtin]

        # Disable first two built-in detectors
        to_disable = builtin_names[:2]
        config = PatternsConfig(disabled=to_disable)
        detectors = get_configured_detectors(config)

        remaining_names = [d.name for d in detectors]
        for disabled_name in to_disable:
            assert disabled_name not in remaining_names
        assert len(detectors) == len(builtin) - 2

    def test_disable_custom_by_name(self):
        config = PatternsConfig(
            custom=[
                PatternRuleConfig(name="my_rule", pattern=r"TODO"),
            ],
            disabled=["my_rule"],
        )
        detectors = get_configured_detectors(config)
        names = [d.name for d in detectors]
        assert "custom:my_rule" not in names

    def test_disable_nonexistent_is_noop(self):
        builtin_count = len(get_all_detectors())
        config = PatternsConfig(disabled=["nonexistent_detector_xyz"])
        detectors = get_configured_detectors(config)
        assert len(detectors) == builtin_count

    def test_custom_and_disable_together(self):
        builtin_count = len(get_all_detectors())
        config = PatternsConfig(
            custom=[
                PatternRuleConfig(name="check1", pattern=r"CHECK1", severity="high"),
                PatternRuleConfig(name="check2", pattern=r"CHECK2", severity="low"),
            ],
            disabled=["hardcoded_credentials"],
        )
        detectors = get_configured_detectors(config)
        # builtin - 1 disabled + 2 custom
        assert len(detectors) == builtin_count - 1 + 2


# --- YAML config loading ---


class TestYAMLConfigLoading:
    def test_load_config_with_patterns(self, tmp_path):
        config_data = {
            "gate": {
                "patterns": {
                    "custom": [
                        {
                            "name": "internal_url",
                            "pattern": r"https://internal\.corp\.com",
                            "severity": "high",
                            "message": "Internal URL should use config",
                        },
                        {
                            "name": "debug_flag",
                            "pattern": r"DEBUG\s*=\s*True",
                            "severity": "medium",
                            "message": "Debug flag left enabled",
                        },
                    ],
                    "disabled": ["hardcoded_credentials"],
                }
            }
        }
        config_file = tmp_path / ".rubric-gates.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_path=config_file)
        assert len(config.gate.patterns.custom) == 2
        assert config.gate.patterns.custom[0].name == "internal_url"
        assert config.gate.patterns.custom[1].name == "debug_flag"
        assert "hardcoded_credentials" in config.gate.patterns.disabled

    def test_load_config_without_patterns(self, tmp_path):
        config_data = {"gate": {"thresholds": {"green": {"min_composite": 0.8}}}}
        config_file = tmp_path / ".rubric-gates.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_path=config_file)
        assert config.gate.patterns.custom == []
        assert config.gate.patterns.disabled == []

    def test_empty_config_has_defaults(self):
        config = GateConfig()
        assert config.patterns.custom == []
        assert config.patterns.disabled == []


# --- TierEvaluator integration ---


class TestTierEvaluatorWithCustomPatterns:
    def test_custom_critical_pattern_triggers_red(self):
        gate_config = GateConfig(
            patterns=PatternsConfig(
                custom=[
                    PatternRuleConfig(
                        name="forbidden_import",
                        pattern=r"import\s+os",
                        severity="critical",
                        message="os module import forbidden in this project",
                    ),
                ]
            )
        )
        evaluator = TierEvaluator(config=gate_config)
        score = ScoreResult(user="alice", composite_score=0.9)
        code = "import os\nprint(os.getcwd())"

        result = evaluator.evaluate(score, code, "test.py")
        assert result.tier == GateTier.RED
        assert result.blocked
        custom_findings = [
            f for f in result.pattern_findings if f.pattern == "custom:forbidden_import"
        ]
        assert len(custom_findings) > 0

    def test_custom_high_pattern_triggers_yellow(self):
        gate_config = GateConfig(
            patterns=PatternsConfig(
                custom=[
                    PatternRuleConfig(
                        name="todo_check",
                        pattern=r"TODO",
                        severity="high",
                        message="TODO found",
                    ),
                ]
            )
        )
        evaluator = TierEvaluator(config=gate_config)
        score = ScoreResult(user="alice", composite_score=0.9)
        code = "x = 1  # TODO: fix later"

        result = evaluator.evaluate(score, code, "test.py")
        assert result.tier in (GateTier.YELLOW, GateTier.RED)

    def test_disabled_builtin_not_triggered(self):
        gate_config = GateConfig(
            patterns=PatternsConfig(
                disabled=["hardcoded_credentials"],
            )
        )
        evaluator = TierEvaluator(config=gate_config)
        score = ScoreResult(user="alice", composite_score=0.9)
        code = 'password = "admin123"'

        result = evaluator.evaluate(score, code, "test.py")
        cred_findings = [f for f in result.pattern_findings if f.pattern == "hardcoded_credentials"]
        assert len(cred_findings) == 0

    def test_default_config_works(self):
        evaluator = TierEvaluator()
        score = ScoreResult(user="alice", composite_score=0.9)
        result = evaluator.evaluate(score, "x = 1", "test.py")
        assert result.tier == GateTier.GREEN
