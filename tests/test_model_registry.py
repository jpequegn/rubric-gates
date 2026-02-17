"""Tests for model versioning and registry (issue #64).

All tests use a temporary directory for YAML storage — no side effects.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from training.model_registry import (
    STAGE_ORDER,
    VALID_STAGES,
    ComparisonReport,
    ModelRegistry,
    ModelVersion,
    _generate_version_id,
    _hash_dict,
)


# --- ModelVersion model ---


class TestModelVersion:
    def test_defaults(self):
        v = ModelVersion(
            version_id="v1",
            phase="grpo",
            checkpoint_path="/checkpoints/v1",
        )
        assert v.version_id == "v1"
        assert v.phase == "grpo"
        assert v.stage == "dev"
        assert v.eval_metrics == {}
        assert v.tags == []
        assert v.metadata == {}
        assert v.parent_version == ""
        assert v.description == ""

    def test_with_all_fields(self):
        v = ModelVersion(
            version_id="v2",
            phase="distillation",
            checkpoint_path="/checkpoints/v2",
            stage="staging",
            training_config_hash="abc123",
            dataset_hash="def456",
            model_name="Qwen/Qwen2.5-Coder-1.5B",
            eval_metrics={"accuracy": 0.95, "loss": 0.1},
            tags=["experiment", "baseline"],
            metadata={"lr": 1e-4},
            parent_version="v1",
            description="Improved baseline",
        )
        assert v.stage == "staging"
        assert v.model_name == "Qwen/Qwen2.5-Coder-1.5B"
        assert v.eval_metrics["accuracy"] == 0.95
        assert "experiment" in v.tags

    def test_created_at_auto(self):
        v = ModelVersion(version_id="v1", phase="grpo", checkpoint_path="/ckpt")
        assert v.created_at is not None


# --- ComparisonReport model ---


class TestComparisonReport:
    def test_defaults(self):
        r = ComparisonReport(version_a="v1", version_b="v2")
        assert r.metric_diffs == {}
        assert r.better_version == ""
        assert r.summary == ""


# --- Helper functions ---


class TestHelpers:
    def test_hash_dict_deterministic(self):
        d = {"lr": 1e-4, "epochs": 3}
        assert _hash_dict(d) == _hash_dict(d)

    def test_hash_dict_different_for_different_inputs(self):
        assert _hash_dict({"a": 1}) != _hash_dict({"a": 2})

    def test_hash_dict_order_independent(self):
        assert _hash_dict({"a": 1, "b": 2}) == _hash_dict({"b": 2, "a": 1})

    def test_hash_dict_length(self):
        h = _hash_dict({"x": "y"})
        assert len(h) == 12

    def test_generate_version_id_format(self):
        vid = _generate_version_id("grpo", "seed123")
        assert vid.startswith("grpo-")
        parts = vid.split("-")
        # grpo-YYYYMMDD-HHMMSS-hash
        assert len(parts) == 4

    def test_generate_version_id_different_phases(self):
        v1 = _generate_version_id("grpo", "seed")
        time.sleep(0.01)
        v2 = _generate_version_id("distillation", "seed")
        assert v1 != v2
        assert v1.startswith("grpo-")
        assert v2.startswith("distillation-")


# --- Constants ---


class TestConstants:
    def test_valid_stages(self):
        assert VALID_STAGES == ("dev", "staging", "production")

    def test_stage_order(self):
        assert STAGE_ORDER["dev"] < STAGE_ORDER["staging"]
        assert STAGE_ORDER["staging"] < STAGE_ORDER["production"]


# --- ModelRegistry ---


class TestRegistryRegister:
    def test_register_basic(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(
            checkpoint_path="/ckpt/v1",
            phase="grpo",
        )
        assert vid.startswith("grpo-")
        # YAML file created
        assert (tmp_path / f"{vid}.yaml").exists()

    def test_register_with_metrics(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(
            checkpoint_path="/ckpt/v1",
            phase="distillation",
            eval_metrics={"accuracy": 0.9, "loss": 0.15},
        )
        version = registry.get(vid)
        assert version is not None
        assert version.eval_metrics["accuracy"] == 0.9

    def test_register_with_config_and_dataset(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(
            checkpoint_path="/ckpt/v1",
            phase="grpo",
            training_config={"lr": 1e-4, "epochs": 3},
            dataset_info={"name": "code-reviews", "size": 1000},
        )
        version = registry.get(vid)
        assert version is not None
        assert version.training_config_hash != ""
        assert version.dataset_hash != ""
        assert version.metadata["training_config"]["lr"] == 1e-4

    def test_register_with_tags_and_description(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(
            checkpoint_path="/ckpt/v1",
            phase="skill_optimizer",
            tags=["baseline", "v1"],
            description="First baseline model",
        )
        version = registry.get(vid)
        assert version is not None
        assert "baseline" in version.tags
        assert version.description == "First baseline model"

    def test_register_with_parent_version(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        v1 = registry.register(checkpoint_path="/ckpt/v1", phase="grpo")
        v2 = registry.register(checkpoint_path="/ckpt/v2", phase="grpo", parent_version=v1)
        version = registry.get(v2)
        assert version is not None
        assert version.parent_version == v1

    def test_register_invalid_phase(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        with pytest.raises(ValueError, match="Unknown phase"):
            registry.register(checkpoint_path="/ckpt", phase="invalid")

    def test_register_starts_at_dev(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(checkpoint_path="/ckpt", phase="grpo")
        version = registry.get(vid)
        assert version is not None
        assert version.stage == "dev"


# --- Get ---


class TestRegistryGet:
    def test_get_existing(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(checkpoint_path="/ckpt", phase="grpo")
        version = registry.get(vid)
        assert version is not None
        assert version.version_id == vid

    def test_get_nonexistent(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        assert registry.get("nonexistent") is None


# --- List ---


class TestRegistryList:
    def test_list_empty(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        assert registry.list() == []

    def test_list_all(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        registry.register(checkpoint_path="/ckpt/1", phase="grpo")
        registry.register(checkpoint_path="/ckpt/2", phase="distillation")
        versions = registry.list()
        assert len(versions) == 2

    def test_list_filter_by_phase(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        registry.register(checkpoint_path="/ckpt/1", phase="grpo")
        registry.register(checkpoint_path="/ckpt/2", phase="distillation")
        registry.register(checkpoint_path="/ckpt/3", phase="grpo")
        grpo_versions = registry.list(phase="grpo")
        assert len(grpo_versions) == 2
        assert all(v.phase == "grpo" for v in grpo_versions)

    def test_list_filter_by_stage(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(checkpoint_path="/ckpt/1", phase="grpo")
        registry.register(checkpoint_path="/ckpt/2", phase="grpo")
        registry.promote(vid, "staging")
        staging = registry.list(stage="staging")
        assert len(staging) == 1
        assert staging[0].version_id == vid

    def test_list_filter_by_tag(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        registry.register(checkpoint_path="/ckpt/1", phase="grpo", tags=["baseline"])
        registry.register(checkpoint_path="/ckpt/2", phase="grpo", tags=["experiment"])
        baseline = registry.list(tag="baseline")
        assert len(baseline) == 1

    def test_list_sorted_newest_first(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        v1 = registry.register(checkpoint_path="/ckpt/1", phase="grpo")
        time.sleep(0.01)
        v2 = registry.register(checkpoint_path="/ckpt/2", phase="grpo")
        versions = registry.list()
        assert versions[0].version_id == v2
        assert versions[1].version_id == v1


# --- Compare ---


class TestRegistryCompare:
    def test_compare_basic(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        v1 = registry.register(
            checkpoint_path="/ckpt/1",
            phase="grpo",
            eval_metrics={"accuracy": 0.8, "loss": 0.3},
        )
        v2 = registry.register(
            checkpoint_path="/ckpt/2",
            phase="grpo",
            eval_metrics={"accuracy": 0.9, "loss": 0.2},
        )
        report = registry.compare(v1, v2)
        assert report.version_a == v1
        assert report.version_b == v2
        # B is better on both metrics (higher accuracy, lower loss treated as higher)
        assert report.metric_diffs["accuracy"] == pytest.approx(0.1)
        assert report.metric_diffs["loss"] == pytest.approx(-0.1)

    def test_compare_determines_better(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        v1 = registry.register(
            checkpoint_path="/ckpt/1",
            phase="grpo",
            eval_metrics={"accuracy": 0.8, "f1": 0.7},
        )
        v2 = registry.register(
            checkpoint_path="/ckpt/2",
            phase="grpo",
            eval_metrics={"accuracy": 0.9, "f1": 0.85},
        )
        report = registry.compare(v1, v2)
        assert report.better_version == v2

    def test_compare_tie(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        v1 = registry.register(
            checkpoint_path="/ckpt/1",
            phase="grpo",
            eval_metrics={"a": 0.9, "b": 0.7},
        )
        v2 = registry.register(
            checkpoint_path="/ckpt/2",
            phase="grpo",
            eval_metrics={"a": 0.7, "b": 0.9},
        )
        report = registry.compare(v1, v2)
        assert report.better_version == ""

    def test_compare_missing_metrics(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        v1 = registry.register(
            checkpoint_path="/ckpt/1",
            phase="grpo",
            eval_metrics={"accuracy": 0.8},
        )
        v2 = registry.register(
            checkpoint_path="/ckpt/2",
            phase="grpo",
            eval_metrics={"accuracy": 0.9, "f1": 0.85},
        )
        report = registry.compare(v1, v2)
        # Missing metric defaults to 0.0
        assert "f1" in report.metric_diffs

    def test_compare_nonexistent_version(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        v1 = registry.register(checkpoint_path="/ckpt", phase="grpo")
        with pytest.raises(KeyError):
            registry.compare(v1, "nonexistent")

    def test_compare_summary(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        v1 = registry.register(
            checkpoint_path="/ckpt/1",
            phase="grpo",
            eval_metrics={"accuracy": 0.8},
        )
        v2 = registry.register(
            checkpoint_path="/ckpt/2",
            phase="grpo",
            eval_metrics={"accuracy": 0.9},
        )
        report = registry.compare(v1, v2)
        assert "accuracy" in report.summary


# --- Promote ---


class TestRegistryPromote:
    def test_promote_dev_to_staging(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(checkpoint_path="/ckpt", phase="grpo")
        version = registry.promote(vid, "staging")
        assert version.stage == "staging"

    def test_promote_staging_to_production(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(checkpoint_path="/ckpt", phase="grpo")
        registry.promote(vid, "staging")
        version = registry.promote(vid, "production")
        assert version.stage == "production"

    def test_promote_dev_to_production(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(checkpoint_path="/ckpt", phase="grpo")
        version = registry.promote(vid, "production")
        assert version.stage == "production"

    def test_promote_invalid_stage(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(checkpoint_path="/ckpt", phase="grpo")
        with pytest.raises(ValueError, match="Invalid stage"):
            registry.promote(vid, "invalid")

    def test_promote_cannot_go_backward(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(checkpoint_path="/ckpt", phase="grpo")
        registry.promote(vid, "staging")
        with pytest.raises(ValueError, match="Must advance"):
            registry.promote(vid, "dev")

    def test_promote_cannot_stay_same(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(checkpoint_path="/ckpt", phase="grpo")
        with pytest.raises(ValueError, match="Must advance"):
            registry.promote(vid, "dev")

    def test_promote_nonexistent(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        with pytest.raises(KeyError):
            registry.promote("nonexistent", "staging")

    def test_promote_persists(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(checkpoint_path="/ckpt", phase="grpo")
        registry.promote(vid, "staging")
        # Reload from disk
        registry2 = ModelRegistry(tmp_path)
        version = registry2.get(vid)
        assert version is not None
        assert version.stage == "staging"


# --- Rollback ---


class TestRegistryRollback:
    def test_rollback_basic(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        v1 = registry.register(checkpoint_path="/ckpt/1", phase="grpo")
        v2 = registry.register(checkpoint_path="/ckpt/2", phase="grpo")
        registry.promote(v1, "production")
        registry.promote(v2, "staging")
        # Rollback to v2
        result = registry.rollback(v2)
        assert result.stage == "production"
        # v1 should be demoted to staging
        v1_after = registry.get(v1)
        assert v1_after is not None
        assert v1_after.stage == "staging"

    def test_rollback_no_current_production(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(checkpoint_path="/ckpt", phase="grpo")
        # No current production, just promote target
        result = registry.rollback(vid)
        assert result.stage == "production"

    def test_rollback_nonexistent(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        with pytest.raises(KeyError):
            registry.rollback("nonexistent")

    def test_rollback_only_affects_same_phase(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        v_grpo = registry.register(checkpoint_path="/ckpt/1", phase="grpo")
        v_dist = registry.register(checkpoint_path="/ckpt/2", phase="distillation")
        registry.promote(v_grpo, "production")
        registry.promote(v_dist, "production")
        # Register and rollback a new grpo version
        v_grpo2 = registry.register(checkpoint_path="/ckpt/3", phase="grpo")
        registry.rollback(v_grpo2)
        # Distillation production should be unaffected
        v_dist_after = registry.get(v_dist)
        assert v_dist_after is not None
        assert v_dist_after.stage == "production"


# --- Delete ---


class TestRegistryDelete:
    def test_delete_existing(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(checkpoint_path="/ckpt", phase="grpo")
        assert registry.delete(vid) is True
        assert registry.get(vid) is None

    def test_delete_nonexistent(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        assert registry.delete("nonexistent") is False

    def test_delete_removes_yaml(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(checkpoint_path="/ckpt", phase="grpo")
        yaml_path = tmp_path / f"{vid}.yaml"
        assert yaml_path.exists()
        registry.delete(vid)
        assert not yaml_path.exists()


# --- get_production ---


class TestGetProduction:
    def test_get_production_exists(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(checkpoint_path="/ckpt", phase="grpo")
        registry.promote(vid, "production")
        prod = registry.get_production("grpo")
        assert prod is not None
        assert prod.version_id == vid

    def test_get_production_none(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        registry.register(checkpoint_path="/ckpt", phase="grpo")
        assert registry.get_production("grpo") is None

    def test_get_production_phase_specific(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(checkpoint_path="/ckpt", phase="grpo")
        registry.promote(vid, "production")
        assert registry.get_production("distillation") is None


# --- YAML persistence ---


class TestYAMLPersistence:
    def test_roundtrip(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        vid = registry.register(
            checkpoint_path="/ckpt/v1",
            phase="grpo",
            eval_metrics={"accuracy": 0.95},
            model_name="test-model",
            tags=["baseline"],
            description="Test version",
        )
        # Create a new registry instance reading from same directory
        registry2 = ModelRegistry(tmp_path)
        version = registry2.get(vid)
        assert version is not None
        assert version.checkpoint_path == "/ckpt/v1"
        assert version.eval_metrics["accuracy"] == 0.95
        assert version.model_name == "test-model"
        assert "baseline" in version.tags

    def test_registry_creates_directory(self, tmp_path: Path):
        new_dir = tmp_path / "new-registry"
        assert not new_dir.exists()
        ModelRegistry(new_dir)
        assert new_dir.exists()


# --- Integration-style tests ---


class TestRegistryWorkflow:
    def test_full_promotion_lifecycle(self, tmp_path: Path):
        """Register, evaluate, promote through stages."""
        registry = ModelRegistry(tmp_path)

        # Register initial version
        v1 = registry.register(
            checkpoint_path="/ckpt/v1",
            phase="grpo",
            eval_metrics={"accuracy": 0.8, "loss": 0.3},
            training_config={"lr": 1e-4},
            model_name="Qwen/Qwen2.5-Coder-1.5B",
            description="Initial training run",
        )

        # Promote to staging after review
        registry.promote(v1, "staging")
        assert registry.get(v1).stage == "staging"

        # Train improved version
        v2 = registry.register(
            checkpoint_path="/ckpt/v2",
            phase="grpo",
            eval_metrics={"accuracy": 0.9, "loss": 0.2},
            training_config={"lr": 5e-5},
            parent_version=v1,
            description="Lower LR, more epochs",
        )

        # Compare versions — compare counts positive diffs as B-wins
        # accuracy: +0.1 (B wins), loss: -0.1 (A wins) → tie
        report = registry.compare(v1, v2)
        assert report.metric_diffs["accuracy"] == pytest.approx(0.1)
        assert report.metric_diffs["loss"] == pytest.approx(-0.1)
        assert report.better_version == ""  # Tie: 1 metric each

        # Promote v2 to production
        registry.promote(v2, "production")
        prod = registry.get_production("grpo")
        assert prod is not None
        assert prod.version_id == v2

    def test_rollback_workflow(self, tmp_path: Path):
        """Deploy, detect regression, rollback."""
        registry = ModelRegistry(tmp_path)

        v1 = registry.register(
            checkpoint_path="/ckpt/v1",
            phase="distillation",
            eval_metrics={"mae": 0.05},
        )
        registry.promote(v1, "production")

        v2 = registry.register(
            checkpoint_path="/ckpt/v2",
            phase="distillation",
            eval_metrics={"mae": 0.15},  # Worse!
            parent_version=v1,
        )

        # Compare shows v1 is better (lower mae = diff is positive for v2)
        report = registry.compare(v1, v2)
        assert report.metric_diffs["mae"] == pytest.approx(0.1)

        # Rollback to v1
        registry.rollback(v1)
        prod = registry.get_production("distillation")
        assert prod is not None
        assert prod.version_id == v1
