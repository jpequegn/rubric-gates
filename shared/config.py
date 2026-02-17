"""Configuration management for rubric-gates.

Loads YAML config with cascading precedence: repo root → user home → defaults.
"""

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# --- Config Schema ---

CONFIG_FILENAME = ".rubric-gates.yaml"


class DimensionConfig(BaseModel):
    """Configuration for a single scoring dimension."""

    weight: float = 0.0
    enabled: bool = True


class ScorecardConfig(BaseModel):
    """Configuration for the scorecard project."""

    dimensions: dict[str, DimensionConfig] = Field(
        default_factory=lambda: {
            "correctness": DimensionConfig(weight=0.25, enabled=True),
            "security": DimensionConfig(weight=0.25, enabled=True),
            "maintainability": DimensionConfig(weight=0.20, enabled=True),
            "documentation": DimensionConfig(weight=0.15, enabled=True),
            "testability": DimensionConfig(weight=0.15, enabled=True),
        }
    )


class GreenThreshold(BaseModel):
    min_composite: float = 0.7


class YellowThreshold(BaseModel):
    min_composite: float = 0.5


class RedThreshold(BaseModel):
    security: float = 0.3


class GateThresholds(BaseModel):
    green: GreenThreshold = Field(default_factory=GreenThreshold)
    yellow: YellowThreshold = Field(default_factory=YellowThreshold)
    red: RedThreshold = Field(default_factory=RedThreshold)


class OverridesConfig(BaseModel):
    require_justification: bool = True
    notify_tech_team: bool = True
    max_overrides_per_user_per_week: int = 3


class PatternRuleConfig(BaseModel):
    """A single custom pattern detection rule."""

    name: str
    pattern: str
    severity: str = "medium"  # critical, high, medium, low
    message: str = ""
    remediation: str = ""


class PatternsConfig(BaseModel):
    """Configuration for pattern detection rules."""

    custom: list[PatternRuleConfig] = Field(default_factory=list)
    disabled: list[str] = Field(default_factory=list)


class GateConfig(BaseModel):
    """Configuration for the gate project."""

    thresholds: GateThresholds = Field(default_factory=GateThresholds)
    critical_patterns: list[str] = Field(
        default_factory=lambda: [
            "hardcoded_credentials",
            "sql_injection",
            "unsafe_file_ops",
            "unvetted_dependencies",
        ]
    )
    patterns: PatternsConfig = Field(default_factory=PatternsConfig)
    overrides: OverridesConfig = Field(default_factory=OverridesConfig)


class T0ToT1Triggers(BaseModel):
    second_user: bool = True
    max_lines: int = 500


class T1ToT2Triggers(BaseModel):
    daily_usage_days: int = 14
    min_users: int = 3


class T2ToT3Triggers(BaseModel):
    manual_only: bool = True


class AutoTriggers(BaseModel):
    t0_to_t1: T0ToT1Triggers = Field(default_factory=T0ToT1Triggers)
    t1_to_t2: T1ToT2Triggers = Field(default_factory=T1ToT2Triggers)
    t2_to_t3: T2ToT3Triggers = Field(default_factory=T2ToT3Triggers)


class RegistryConfig(BaseModel):
    """Configuration for the registry project."""

    auto_triggers: AutoTriggers = Field(default_factory=AutoTriggers)


class StorageConfig(BaseModel):
    """Configuration for the storage backend."""

    backend: str = "jsonl"
    path: str = "./rubric-gates-data/"


class RubricGatesConfig(BaseModel):
    """Top-level configuration for rubric-gates."""

    scorecard: ScorecardConfig = Field(default_factory=ScorecardConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


# --- Config Loading ---


def _find_config_files(start_dir: Path | None = None) -> list[Path]:
    """Find config files in cascading order: defaults (lowest) → user home → repo root (highest).

    Returns paths in precedence order (lowest first, highest last) so that
    later entries override earlier ones when merged.
    """
    candidates: list[Path] = []

    # User home (lower precedence)
    home_config = Path.home() / CONFIG_FILENAME
    if home_config.is_file():
        candidates.append(home_config)

    # Repo root / start directory (higher precedence)
    search_dir = start_dir or Path.cwd()
    repo_config = search_dir / CONFIG_FILENAME
    if repo_config.is_file():
        candidates.append(repo_config)

    return candidates


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_path: Path | None = None,
    start_dir: Path | None = None,
) -> RubricGatesConfig:
    """Load rubric-gates configuration with cascading precedence.

    Priority (highest to lowest):
    1. Explicit config_path (if provided)
    2. Repo root / start_dir .rubric-gates.yaml
    3. User home .rubric-gates.yaml
    4. Built-in defaults

    Args:
        config_path: Explicit path to a config file (overrides discovery).
        start_dir: Directory to search for config files (defaults to cwd).

    Returns:
        Validated RubricGatesConfig.
    """
    merged: dict[str, Any] = {}

    if config_path is not None:
        # Explicit path — use only this file + defaults
        if config_path.is_file():
            merged = _load_yaml(config_path)
    else:
        # Cascading discovery
        for path in _find_config_files(start_dir):
            file_data = _load_yaml(path)
            merged = _deep_merge(merged, file_data)

    return RubricGatesConfig.model_validate(merged)


@functools.lru_cache(maxsize=1)
def get_config() -> RubricGatesConfig:
    """Get the cached global configuration. Loaded once per session."""
    return load_config()


def clear_config_cache() -> None:
    """Clear the cached configuration. Useful for testing."""
    get_config.cache_clear()
