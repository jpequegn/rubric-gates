"""Model versioning and registry for trained checkpoints.

Tracks trained model versions with metadata (training config hash,
dataset hash, evaluation scores). Supports promotion through stages
(dev -> staging -> production), comparison between versions, and
rollback.

Storage: YAML manifest per version in the registry directory.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# --- Models ---


class ModelVersion(BaseModel):
    """A versioned model checkpoint with metadata."""

    version_id: str
    phase: str  # "distillation", "grpo", "skill_optimizer"
    checkpoint_path: str
    created_at: datetime = Field(default_factory=datetime.now)
    stage: str = "dev"  # "dev", "staging", "production"
    training_config_hash: str = ""
    dataset_hash: str = ""
    model_name: str = ""
    eval_metrics: dict[str, float] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_version: str = ""  # Version this was derived from
    description: str = ""


class ComparisonReport(BaseModel):
    """Comparison between two model versions."""

    version_a: str
    version_b: str
    metric_diffs: dict[str, float] = Field(default_factory=dict)
    better_version: str = ""
    summary: str = ""


VALID_STAGES = ("dev", "staging", "production")
STAGE_ORDER = {stage: i for i, stage in enumerate(VALID_STAGES)}


# --- Registry ---


class ModelRegistry:
    """YAML-backed registry for trained model versions.

    Each version gets a YAML manifest file in the registry directory.

    Args:
        registry_dir: Directory to store version manifests.
    """

    def __init__(self, registry_dir: str | Path = "model-registry") -> None:
        self._dir = Path(registry_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _version_path(self, version_id: str) -> Path:
        return self._dir / f"{version_id}.yaml"

    def _save(self, version: ModelVersion) -> None:
        data = version.model_dump(mode="json")
        path = self._version_path(version.version_id)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _load(self, version_id: str) -> ModelVersion | None:
        path = self._version_path(version_id)
        if not path.exists():
            return None
        with open(path) as f:
            data = yaml.safe_load(f)
        if not data:
            return None
        return ModelVersion.model_validate(data)

    def register(
        self,
        checkpoint_path: str,
        phase: str,
        eval_metrics: dict[str, float] | None = None,
        training_config: dict[str, Any] | None = None,
        dataset_info: dict[str, Any] | None = None,
        model_name: str = "",
        tags: list[str] | None = None,
        description: str = "",
        parent_version: str = "",
    ) -> str:
        """Register a new model version.

        Args:
            checkpoint_path: Path to the checkpoint directory.
            phase: Training phase (distillation, grpo, skill_optimizer).
            eval_metrics: Evaluation metrics for this version.
            training_config: Training configuration dict (hashed for ID).
            dataset_info: Dataset information dict (hashed for tracking).
            model_name: Base model name.
            tags: Optional tags.
            description: Human-readable description.
            parent_version: Version ID this was derived from.

        Returns:
            The generated version_id.

        Raises:
            ValueError: If phase is not recognized.
        """
        valid_phases = ("distillation", "grpo", "skill_optimizer")
        if phase not in valid_phases:
            raise ValueError(f"Unknown phase '{phase}'. Valid: {', '.join(valid_phases)}")

        config_hash = _hash_dict(training_config) if training_config else ""
        ds_hash = _hash_dict(dataset_info) if dataset_info else ""

        version_id = _generate_version_id(phase, config_hash)

        # Ensure uniqueness
        if self._version_path(version_id).exists():
            version_id = _generate_version_id(phase, config_hash + str(datetime.now()))

        version = ModelVersion(
            version_id=version_id,
            phase=phase,
            checkpoint_path=checkpoint_path,
            stage="dev",
            training_config_hash=config_hash,
            dataset_hash=ds_hash,
            model_name=model_name,
            eval_metrics=eval_metrics or {},
            tags=tags or [],
            description=description,
            parent_version=parent_version,
            metadata={
                "training_config": training_config or {},
                "dataset_info": dataset_info or {},
            },
        )

        self._save(version)
        return version_id

    def get(self, version_id: str) -> ModelVersion | None:
        """Get a model version by ID."""
        return self._load(version_id)

    def list(
        self,
        phase: str | None = None,
        stage: str | None = None,
        tag: str | None = None,
    ) -> list[ModelVersion]:
        """List model versions with optional filters.

        Args:
            phase: Filter by training phase.
            stage: Filter by promotion stage.
            tag: Filter by tag.

        Returns:
            List of matching versions, sorted by creation date (newest first).
        """
        versions: list[ModelVersion] = []

        for path in self._dir.glob("*.yaml"):
            with open(path) as f:
                data = yaml.safe_load(f)
            if not data:
                continue

            version = ModelVersion.model_validate(data)

            if phase is not None and version.phase != phase:
                continue
            if stage is not None and version.stage != stage:
                continue
            if tag is not None and tag not in version.tags:
                continue

            versions.append(version)

        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def compare(self, version_a_id: str, version_b_id: str) -> ComparisonReport:
        """Compare two model versions by their evaluation metrics.

        Args:
            version_a_id: First version ID.
            version_b_id: Second version ID.

        Returns:
            ComparisonReport with metric differences.

        Raises:
            KeyError: If either version not found.
        """
        a = self._load(version_a_id)
        b = self._load(version_b_id)

        if a is None:
            raise KeyError(f"Version '{version_a_id}' not found.")
        if b is None:
            raise KeyError(f"Version '{version_b_id}' not found.")

        # Compute diffs: positive means B is better
        all_keys = set(a.eval_metrics.keys()) | set(b.eval_metrics.keys())
        diffs: dict[str, float] = {}
        a_wins = 0
        b_wins = 0

        for key in sorted(all_keys):
            val_a = a.eval_metrics.get(key, 0.0)
            val_b = b.eval_metrics.get(key, 0.0)
            diff = round(val_b - val_a, 6)
            diffs[key] = diff
            if diff > 0:
                b_wins += 1
            elif diff < 0:
                a_wins += 1

        if a_wins > b_wins:
            better = version_a_id
        elif b_wins > a_wins:
            better = version_b_id
        else:
            better = ""

        summary_parts = []
        for key, diff in diffs.items():
            direction = "+" if diff >= 0 else ""
            summary_parts.append(f"{key}: {direction}{diff:.4f}")

        return ComparisonReport(
            version_a=version_a_id,
            version_b=version_b_id,
            metric_diffs=diffs,
            better_version=better,
            summary="; ".join(summary_parts),
        )

    def promote(self, version_id: str, stage: str) -> ModelVersion:
        """Promote a model version to a new stage.

        Args:
            version_id: Version to promote.
            stage: Target stage (dev, staging, production).

        Returns:
            Updated ModelVersion.

        Raises:
            KeyError: If version not found.
            ValueError: If stage is invalid or not an advancement.
        """
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Valid: {', '.join(VALID_STAGES)}")

        version = self._load(version_id)
        if version is None:
            raise KeyError(f"Version '{version_id}' not found.")

        current_order = STAGE_ORDER.get(version.stage, 0)
        target_order = STAGE_ORDER[stage]

        if target_order <= current_order:
            raise ValueError(
                f"Cannot promote from '{version.stage}' to '{stage}'. "
                "Must advance to a higher stage."
            )

        version.stage = stage
        self._save(version)
        return version

    def rollback(self, to_version_id: str) -> ModelVersion:
        """Rollback: promote a previous version to production.

        Demotes the current production model (if any) back to staging,
        and sets the target version to production.

        Args:
            to_version_id: Version to roll back to.

        Returns:
            The newly promoted version.

        Raises:
            KeyError: If version not found.
        """
        target = self._load(to_version_id)
        if target is None:
            raise KeyError(f"Version '{to_version_id}' not found.")

        # Demote current production models for same phase
        for version in self.list(phase=target.phase, stage="production"):
            if version.version_id != to_version_id:
                version.stage = "staging"
                self._save(version)

        target.stage = "production"
        self._save(target)
        return target

    def delete(self, version_id: str) -> bool:
        """Delete a version from the registry.

        Returns True if deleted, False if not found.
        """
        path = self._version_path(version_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def get_production(self, phase: str) -> ModelVersion | None:
        """Get the current production model for a phase.

        Args:
            phase: Training phase.

        Returns:
            The production ModelVersion, or None.
        """
        versions = self.list(phase=phase, stage="production")
        return versions[0] if versions else None


# --- Helpers ---


def _hash_dict(d: dict[str, Any]) -> str:
    """Create a short hash of a dictionary for version tracking."""
    serialized = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def _generate_version_id(phase: str, seed: str) -> str:
    """Generate a version ID from phase and seed string."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    short_hash = hashlib.sha256((seed + timestamp).encode()).hexdigest()[:8]
    return f"{phase}-{timestamp}-{short_hash}"
