"""Tests for per-org threshold profiles (issue #60)."""

from __future__ import annotations

import pytest
import yaml

from gate.tiers.evaluator import TierEvaluator
from shared.config import (
    DimensionThresholdOverride,
    GateConfig,
    ThresholdProfile,
    load_config,
)
from shared.models import (
    Dimension,
    DimensionScore,
    GateTier,
    ScoreResult,
    ScoringMethod,
)


# --- Helpers ---


def _score(composite: float, **dim_scores: float) -> ScoreResult:
    """Create a ScoreResult with given composite and optional dimension scores."""
    dimensions = []
    for dim_name, score in dim_scores.items():
        dimensions.append(
            DimensionScore(
                dimension=Dimension(dim_name),
                score=score,
                method=ScoringMethod.RULE_BASED,
            )
        )
    return ScoreResult(
        user="alice",
        composite_score=composite,
        dimension_scores=dimensions,
    )


# --- ThresholdProfile model ---


class TestThresholdProfile:
    def test_default_values(self):
        profile = ThresholdProfile()
        assert profile.green == 0.7
        assert profile.yellow == 0.5
        assert profile.red_security == 0.3
        assert profile.dimensions == {}

    def test_custom_values(self):
        profile = ThresholdProfile(green=0.8, yellow=0.6, red_security=0.4)
        assert profile.green == 0.8
        assert profile.yellow == 0.6
        assert profile.red_security == 0.4

    def test_with_dimension_overrides(self):
        profile = ThresholdProfile(
            dimensions={
                "security": DimensionThresholdOverride(green=0.9, yellow=0.6, red=0.5),
            }
        )
        assert profile.dimensions["security"].green == 0.9
        assert profile.dimensions["security"].red == 0.5


class TestDimensionThresholdOverride:
    def test_all_none_defaults(self):
        override = DimensionThresholdOverride()
        assert override.green is None
        assert override.yellow is None
        assert override.red is None

    def test_partial_override(self):
        override = DimensionThresholdOverride(red=0.5)
        assert override.green is None
        assert override.red == 0.5


# --- GateConfig.get_profile ---


class TestGetProfile:
    def test_default_profile(self):
        config = GateConfig()
        profile = config.get_profile()
        assert profile.green == 0.7
        assert profile.yellow == 0.5
        assert profile.red_security == 0.3

    def test_explicit_default_name(self):
        config = GateConfig()
        profile = config.get_profile("default")
        assert profile.green == 0.7

    def test_named_profile(self):
        config = GateConfig(
            profiles={
                "strict": ThresholdProfile(green=0.8, yellow=0.6, red_security=0.4),
            }
        )
        profile = config.get_profile("strict")
        assert profile.green == 0.8
        assert profile.yellow == 0.6
        assert profile.red_security == 0.4

    def test_active_profile_from_config(self):
        config = GateConfig(
            profiles={
                "relaxed": ThresholdProfile(green=0.6, yellow=0.4, red_security=0.2),
            },
            active_profile="relaxed",
        )
        profile = config.get_profile()
        assert profile.green == 0.6

    def test_explicit_name_overrides_active(self):
        config = GateConfig(
            profiles={
                "strict": ThresholdProfile(green=0.8),
                "relaxed": ThresholdProfile(green=0.6),
            },
            active_profile="relaxed",
        )
        profile = config.get_profile("strict")
        assert profile.green == 0.8

    def test_unknown_profile_raises(self):
        config = GateConfig()
        with pytest.raises(ValueError, match="Unknown profile"):
            config.get_profile("nonexistent")

    def test_unknown_profile_lists_available(self):
        config = GateConfig(
            profiles={
                "strict": ThresholdProfile(),
                "relaxed": ThresholdProfile(),
            }
        )
        with pytest.raises(ValueError, match="relaxed"):
            config.get_profile("typo")


# --- TierEvaluator with profiles ---


class TestEvaluatorWithProfiles:
    def test_default_profile_green(self):
        evaluator = TierEvaluator()
        result = evaluator.evaluate(_score(0.8), "x = 1", "test.py")
        assert result.tier == GateTier.GREEN

    def test_strict_profile_yellow(self):
        """Score 0.75 is green with defaults but yellow with strict profile."""
        config = GateConfig(profiles={"strict": ThresholdProfile(green=0.8)})
        evaluator = TierEvaluator(config=config, profile="strict")
        result = evaluator.evaluate(_score(0.75), "x = 1", "test.py")
        assert result.tier == GateTier.YELLOW

    def test_relaxed_profile_green(self):
        """Score 0.55 is yellow with defaults but green with relaxed profile."""
        config = GateConfig(profiles={"relaxed": ThresholdProfile(green=0.5)})
        evaluator = TierEvaluator(config=config, profile="relaxed")
        result = evaluator.evaluate(_score(0.55), "x = 1", "test.py")
        assert result.tier == GateTier.GREEN

    def test_strict_security_red_threshold(self):
        """Strict profile with higher red_security threshold."""
        config = GateConfig(profiles={"strict": ThresholdProfile(red_security=0.5)})
        evaluator = TierEvaluator(config=config, profile="strict")
        result = evaluator.evaluate(_score(0.8, security=0.45), "x = 1", "test.py")
        assert result.tier == GateTier.RED
        assert result.blocked

    def test_per_dimension_yellow_override(self):
        """Profile overrides per-dimension yellow threshold."""
        config = GateConfig(
            profiles={
                "strict": ThresholdProfile(
                    dimensions={
                        "security": DimensionThresholdOverride(yellow=0.8),
                    }
                )
            }
        )
        evaluator = TierEvaluator(config=config, profile="strict")
        result = evaluator.evaluate(_score(0.9, security=0.7), "x = 1", "test.py")
        assert result.tier == GateTier.YELLOW

    def test_per_dimension_red_override(self):
        """Profile sets per-dimension red threshold."""
        config = GateConfig(
            profiles={
                "strict": ThresholdProfile(
                    dimensions={
                        "correctness": DimensionThresholdOverride(red=0.5),
                    }
                )
            }
        )
        evaluator = TierEvaluator(config=config, profile="strict")
        result = evaluator.evaluate(_score(0.9, correctness=0.4), "x = 1", "test.py")
        assert result.tier == GateTier.RED
        assert result.blocked

    def test_active_profile_used_by_default(self):
        config = GateConfig(
            profiles={"strict": ThresholdProfile(green=0.9)},
            active_profile="strict",
        )
        evaluator = TierEvaluator(config=config)
        result = evaluator.evaluate(_score(0.85), "x = 1", "test.py")
        assert result.tier == GateTier.YELLOW

    def test_profile_does_not_affect_pattern_detection(self):
        """Profiles affect thresholds, not pattern detection."""
        config = GateConfig(
            profiles={"relaxed": ThresholdProfile(green=0.3, yellow=0.2, red_security=0.1)}
        )
        evaluator = TierEvaluator(config=config, profile="relaxed")
        code = 'password = "admin123"'
        result = evaluator.evaluate(_score(0.9), code, "test.py")
        # Hardcoded credentials should still be detected
        assert len(result.pattern_findings) > 0


# --- YAML config loading ---


class TestYAMLProfileLoading:
    def test_load_profiles_from_yaml(self, tmp_path):
        config_data = {
            "gate": {
                "profiles": {
                    "strict": {
                        "green": 0.8,
                        "yellow": 0.6,
                        "red_security": 0.4,
                        "dimensions": {
                            "security": {"green": 0.9, "red": 0.5},
                        },
                    },
                    "relaxed": {
                        "green": 0.6,
                        "yellow": 0.4,
                        "red_security": 0.2,
                    },
                },
                "active_profile": "strict",
            }
        }
        config_file = tmp_path / ".rubric-gates.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_path=config_file)
        assert len(config.gate.profiles) == 2
        assert config.gate.active_profile == "strict"

        strict = config.gate.get_profile("strict")
        assert strict.green == 0.8
        assert strict.dimensions["security"].red == 0.5

        relaxed = config.gate.get_profile("relaxed")
        assert relaxed.green == 0.6

    def test_load_without_profiles(self, tmp_path):
        config_data = {"gate": {"thresholds": {"green": {"min_composite": 0.8}}}}
        config_file = tmp_path / ".rubric-gates.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_path=config_file)
        assert config.gate.profiles == {}
        assert config.gate.active_profile == ""

        profile = config.gate.get_profile()
        assert profile.green == 0.8  # Inherits from thresholds

    def test_multiple_dimension_overrides(self, tmp_path):
        config_data = {
            "gate": {
                "profiles": {
                    "custom": {
                        "green": 0.75,
                        "dimensions": {
                            "security": {"yellow": 0.7, "red": 0.4},
                            "correctness": {"yellow": 0.6},
                            "maintainability": {"yellow": 0.5},
                        },
                    }
                }
            }
        }
        config_file = tmp_path / ".rubric-gates.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_path=config_file)
        profile = config.gate.get_profile("custom")
        assert len(profile.dimensions) == 3
        assert profile.dimensions["security"].yellow == 0.7
        assert profile.dimensions["correctness"].yellow == 0.6


# --- CLI integration ---


class TestCLIProfileFlag:
    def test_parser_has_profile_flag(self):
        from cli import _build_parser

        parser = _build_parser()

        # score subcommand
        args = parser.parse_args(["score", "test.py", "--profile", "strict"])
        assert args.profile == "strict"

        # check subcommand
        args = parser.parse_args(["check", "test.py", "--profile", "relaxed"])
        assert args.profile == "relaxed"

    def test_parser_default_empty_profile(self):
        from cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["score", "test.py"])
        assert args.profile == ""
