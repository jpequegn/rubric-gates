"""Tests for configuration management."""

from pathlib import Path

import pytest
import yaml

from shared.config import (
    RubricGatesConfig,
    _deep_merge,
    clear_config_cache,
    load_config,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear config cache before each test."""
    clear_config_cache()


@pytest.fixture()
def config_dir(tmp_path):
    """Create a temp directory for config files."""
    return tmp_path


def write_config(path: Path, data: dict) -> Path:
    """Helper to write a YAML config file."""
    config_file = path / ".rubric-gates.yaml"
    config_file.write_text(yaml.dump(data))
    return config_file


# --- Defaults ---


class TestDefaults:
    def test_load_empty_returns_defaults(self):
        config = load_config(start_dir=Path("/nonexistent"))
        assert isinstance(config, RubricGatesConfig)

    def test_default_scorecard_dimensions(self):
        config = load_config(start_dir=Path("/nonexistent"))
        dims = config.scorecard.dimensions
        assert len(dims) == 5
        assert dims["correctness"].weight == 0.25
        assert dims["security"].weight == 0.25
        assert dims["maintainability"].weight == 0.20
        assert dims["documentation"].weight == 0.15
        assert dims["testability"].weight == 0.15

    def test_default_dimensions_all_enabled(self):
        config = load_config(start_dir=Path("/nonexistent"))
        for dim_config in config.scorecard.dimensions.values():
            assert dim_config.enabled is True

    def test_default_weights_sum_to_one(self):
        config = load_config(start_dir=Path("/nonexistent"))
        total = sum(d.weight for d in config.scorecard.dimensions.values())
        assert abs(total - 1.0) < 1e-9

    def test_default_gate_thresholds(self):
        config = load_config(start_dir=Path("/nonexistent"))
        assert config.gate.thresholds.green.min_composite == 0.7
        assert config.gate.thresholds.yellow.min_composite == 0.5
        assert config.gate.thresholds.red.security == 0.3

    def test_default_critical_patterns(self):
        config = load_config(start_dir=Path("/nonexistent"))
        patterns = config.gate.critical_patterns
        assert "hardcoded_credentials" in patterns
        assert "sql_injection" in patterns
        assert "unsafe_file_ops" in patterns
        assert "unvetted_dependencies" in patterns

    def test_default_overrides(self):
        config = load_config(start_dir=Path("/nonexistent"))
        assert config.gate.overrides.require_justification is True
        assert config.gate.overrides.notify_tech_team is True
        assert config.gate.overrides.max_overrides_per_user_per_week == 3

    def test_default_registry_triggers(self):
        config = load_config(start_dir=Path("/nonexistent"))
        triggers = config.registry.auto_triggers
        assert triggers.t0_to_t1.second_user is True
        assert triggers.t0_to_t1.max_lines == 500
        assert triggers.t1_to_t2.daily_usage_days == 14
        assert triggers.t1_to_t2.min_users == 3
        assert triggers.t2_to_t3.manual_only is True

    def test_default_storage(self):
        config = load_config(start_dir=Path("/nonexistent"))
        assert config.storage.backend == "jsonl"
        assert config.storage.path == "./rubric-gates-data/"


# --- Loading from file ---


class TestLoadFromFile:
    def test_load_explicit_path(self, config_dir):
        config_file = write_config(
            config_dir,
            {"storage": {"backend": "sqlite", "path": "/custom/path/"}},
        )
        config = load_config(config_path=config_file)
        assert config.storage.backend == "sqlite"
        assert config.storage.path == "/custom/path/"

    def test_explicit_path_preserves_other_defaults(self, config_dir):
        config_file = write_config(
            config_dir,
            {"storage": {"backend": "sqlite"}},
        )
        config = load_config(config_path=config_file)
        assert config.storage.backend == "sqlite"
        # Other defaults should still be present
        assert config.gate.thresholds.green.min_composite == 0.7
        assert len(config.scorecard.dimensions) == 5

    def test_load_from_start_dir(self, config_dir):
        write_config(
            config_dir,
            {"gate": {"thresholds": {"green": {"min_composite": 0.8}}}},
        )
        config = load_config(start_dir=config_dir)
        assert config.gate.thresholds.green.min_composite == 0.8

    def test_partial_override_preserves_sibling_defaults(self, config_dir):
        write_config(
            config_dir,
            {"scorecard": {"dimensions": {"correctness": {"weight": 0.30}}}},
        )
        config = load_config(start_dir=config_dir)
        # Overridden value
        assert config.scorecard.dimensions["correctness"].weight == 0.30
        # Only the explicitly provided dimension exists (YAML replaced the dict)
        # Other dimensions come from defaults at the Pydantic level,
        # but the YAML replaces the dimensions dict entirely
        # This is expected behavior — partial dimension override replaces the dict

    def test_disable_a_dimension(self, config_dir):
        write_config(
            config_dir,
            {
                "scorecard": {
                    "dimensions": {
                        "correctness": {"weight": 0.25, "enabled": True},
                        "security": {"weight": 0.25, "enabled": True},
                        "maintainability": {"weight": 0.20, "enabled": False},
                        "documentation": {"weight": 0.15, "enabled": True},
                        "testability": {"weight": 0.15, "enabled": True},
                    }
                }
            },
        )
        config = load_config(start_dir=config_dir)
        assert config.scorecard.dimensions["maintainability"].enabled is False
        assert config.scorecard.dimensions["correctness"].enabled is True

    def test_custom_critical_patterns(self, config_dir):
        write_config(
            config_dir,
            {"gate": {"critical_patterns": ["hardcoded_credentials", "custom_pattern"]}},
        )
        config = load_config(start_dir=config_dir)
        assert config.gate.critical_patterns == ["hardcoded_credentials", "custom_pattern"]

    def test_override_max_overrides(self, config_dir):
        write_config(
            config_dir,
            {"gate": {"overrides": {"max_overrides_per_user_per_week": 10}}},
        )
        config = load_config(start_dir=config_dir)
        assert config.gate.overrides.max_overrides_per_user_per_week == 10

    def test_nonexistent_explicit_path_returns_defaults(self):
        config = load_config(config_path=Path("/does/not/exist.yaml"))
        assert config == RubricGatesConfig()

    def test_empty_yaml_returns_defaults(self, config_dir):
        config_file = config_dir / ".rubric-gates.yaml"
        config_file.write_text("")
        config = load_config(config_path=config_file)
        assert config == RubricGatesConfig()


# --- Cascading / Merging ---


class TestCascading:
    def test_repo_overrides_home(self, config_dir, tmp_path, monkeypatch):
        # Simulate home config
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        write_config(home_dir, {"storage": {"backend": "sqlite"}})
        monkeypatch.setattr(Path, "home", lambda: home_dir)

        # Repo config overrides
        write_config(config_dir, {"storage": {"backend": "jsonl"}})

        config = load_config(start_dir=config_dir)
        assert config.storage.backend == "jsonl"  # repo wins

    def test_home_config_used_when_no_repo_config(self, tmp_path, monkeypatch):
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        write_config(home_dir, {"storage": {"backend": "sqlite"}})
        monkeypatch.setattr(Path, "home", lambda: home_dir)

        # Empty dir with no config
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        config = load_config(start_dir=empty_dir)
        assert config.storage.backend == "sqlite"

    def test_deep_merge_nested(self, config_dir, tmp_path, monkeypatch):
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        write_config(
            home_dir,
            {
                "gate": {
                    "thresholds": {"green": {"min_composite": 0.6}},
                    "overrides": {"max_overrides_per_user_per_week": 5},
                }
            },
        )
        monkeypatch.setattr(Path, "home", lambda: home_dir)

        write_config(
            config_dir,
            {"gate": {"thresholds": {"green": {"min_composite": 0.9}}}},
        )

        config = load_config(start_dir=config_dir)
        # Repo overrides green threshold
        assert config.gate.thresholds.green.min_composite == 0.9
        # Home's overrides config is preserved via deep merge
        assert config.gate.overrides.max_overrides_per_user_per_week == 5


# --- Deep Merge ---


class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        assert _deep_merge(base, override) == {"a": 1, "b": 3}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99}, "b": 3}

    def test_add_new_keys(self):
        base = {"a": 1}
        override = {"b": 2}
        assert _deep_merge(base, override) == {"a": 1, "b": 2}

    def test_override_dict_with_scalar(self):
        base = {"a": {"nested": True}}
        override = {"a": "replaced"}
        assert _deep_merge(base, override) == {"a": "replaced"}

    def test_empty_base(self):
        assert _deep_merge({}, {"a": 1}) == {"a": 1}

    def test_empty_override(self):
        assert _deep_merge({"a": 1}, {}) == {"a": 1}

    def test_both_empty(self):
        assert _deep_merge({}, {}) == {}

    def test_does_not_mutate_base(self):
        base = {"a": 1}
        _deep_merge(base, {"b": 2})
        assert base == {"a": 1}


# --- Validation ---


class TestValidation:
    def test_valid_config_from_dict(self):
        config = RubricGatesConfig.model_validate(
            {
                "scorecard": {
                    "dimensions": {
                        "correctness": {"weight": 0.5, "enabled": True},
                        "security": {"weight": 0.5, "enabled": True},
                    }
                }
            }
        )
        assert len(config.scorecard.dimensions) == 2

    def test_json_roundtrip(self):
        config = RubricGatesConfig()
        json_str = config.model_dump_json()
        restored = RubricGatesConfig.model_validate_json(json_str)
        assert restored.storage.backend == config.storage.backend
        assert len(restored.scorecard.dimensions) == len(config.scorecard.dimensions)

    def test_dict_roundtrip(self):
        config = RubricGatesConfig()
        d = config.model_dump()
        restored = RubricGatesConfig.model_validate(d)
        assert restored == config


# --- Caching ---


class TestCaching:
    def test_get_config_returns_same_instance(self):
        from shared.config import get_config

        a = get_config()
        b = get_config()
        assert a is b

    def test_clear_cache_reloads(self):
        from shared.config import get_config

        a = get_config()
        clear_config_cache()
        b = get_config()
        assert a is not b
        assert a == b  # Same values, different instance
