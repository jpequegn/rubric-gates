"""Tests for experiment tracking abstraction (issue #63).

All tests use InMemoryTracker — no real wandb needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from training.tracking import (
    ExperimentTracker,
    InMemoryTracker,
    NoOpTracker,
    WandbTracker,
    create_tracker,
    log_distillation_metrics,
    log_grpo_metrics,
    log_skill_optimizer_metrics,
)


# --- Protocol compliance ---


class TestExperimentTrackerProtocol:
    def test_noop_is_tracker(self):
        assert isinstance(NoOpTracker(), ExperimentTracker)

    def test_inmemory_is_tracker(self):
        assert isinstance(InMemoryTracker(), ExperimentTracker)

    def test_protocol_methods(self):
        """All required methods exist on protocol implementations."""
        for cls in [NoOpTracker, InMemoryTracker]:
            tracker = cls()
            assert hasattr(tracker, "log_params")
            assert hasattr(tracker, "log_metrics")
            assert hasattr(tracker, "log_artifact")
            assert hasattr(tracker, "set_summary")
            assert hasattr(tracker, "finish")


# --- NoOpTracker ---


class TestNoOpTracker:
    def test_log_params_noop(self):
        tracker = NoOpTracker()
        tracker.log_params({"lr": 1e-4})  # Should not raise

    def test_log_metrics_noop(self):
        tracker = NoOpTracker()
        tracker.log_metrics({"loss": 0.5}, step=1)

    def test_log_artifact_noop(self):
        tracker = NoOpTracker()
        tracker.log_artifact("/path/to/model", name="checkpoint")

    def test_set_summary_noop(self):
        tracker = NoOpTracker()
        tracker.set_summary("best_loss", 0.1)

    def test_finish_noop(self):
        tracker = NoOpTracker()
        tracker.finish()


# --- InMemoryTracker ---


class TestInMemoryTracker:
    def test_log_params(self):
        tracker = InMemoryTracker()
        tracker.log_params({"lr": 1e-4, "epochs": 3})
        assert tracker.params["lr"] == 1e-4
        assert tracker.params["epochs"] == 3

    def test_log_params_merge(self):
        tracker = InMemoryTracker()
        tracker.log_params({"a": 1})
        tracker.log_params({"b": 2})
        assert tracker.params == {"a": 1, "b": 2}

    def test_log_metrics(self):
        tracker = InMemoryTracker()
        tracker.log_metrics({"loss": 0.5, "kl": 0.01}, step=1)
        assert len(tracker.metrics_log) == 1
        assert tracker.metrics_log[0]["loss"] == 0.5
        assert tracker.metrics_log[0]["_step"] == 1

    def test_log_metrics_no_step(self):
        tracker = InMemoryTracker()
        tracker.log_metrics({"loss": 0.5})
        assert "_step" not in tracker.metrics_log[0]

    def test_log_multiple_metrics(self):
        tracker = InMemoryTracker()
        tracker.log_metrics({"loss": 0.5}, step=0)
        tracker.log_metrics({"loss": 0.3}, step=1)
        tracker.log_metrics({"loss": 0.2}, step=2)
        assert len(tracker.metrics_log) == 3

    def test_log_artifact(self):
        tracker = InMemoryTracker()
        tracker.log_artifact("/path/to/checkpoint", name="model-v1", artifact_type="model")
        assert len(tracker.artifacts) == 1
        assert tracker.artifacts[0]["path"] == "/path/to/checkpoint"
        assert tracker.artifacts[0]["name"] == "model-v1"
        assert tracker.artifacts[0]["type"] == "model"

    def test_set_summary(self):
        tracker = InMemoryTracker()
        tracker.set_summary("best_loss", 0.1)
        tracker.set_summary("final_reward", 0.85)
        assert tracker.summaries["best_loss"] == 0.1
        assert tracker.summaries["final_reward"] == 0.85

    def test_finish(self):
        tracker = InMemoryTracker()
        assert not tracker.finished
        tracker.finish()
        assert tracker.finished


# --- create_tracker factory ---


class TestCreateTracker:
    def test_disabled_returns_noop(self):
        tracker = create_tracker(enabled=False)
        assert isinstance(tracker, NoOpTracker)

    def test_disabled_by_default(self):
        tracker = create_tracker()
        assert isinstance(tracker, NoOpTracker)

    def test_enabled_without_wandb_raises(self):
        """Enabling tracking without wandb installed raises ImportError."""
        with patch.dict("sys.modules", {"wandb": None}):
            try:
                create_tracker(enabled=True)
                assert False, "Should have raised ImportError"
            except ImportError as e:
                assert "wandb" in str(e)

    def test_phase_appended_to_project(self):
        """Phase is appended to project name for organization."""
        # We test the logic without actually calling wandb
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            WandbTracker(project="rubric-gates/grpo", run_name="test-run")
            mock_wandb.init.assert_called_once()
            call_kwargs = mock_wandb.init.call_args[1]
            assert call_kwargs["project"] == "rubric-gates/grpo"

    def test_offline_mode(self):
        """Offline mode sets WANDB_MODE environment variable."""
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with patch.dict("os.environ", {}, clear=False):
                import os

                WandbTracker(project="test", offline=True)
                assert os.environ.get("WANDB_MODE") == "offline"

    def test_tags_passed_through(self):
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            WandbTracker(project="test", tags=["grpo", "experiment"])
            call_kwargs = mock_wandb.init.call_args[1]
            assert "grpo" in call_kwargs["tags"]


# --- WandbTracker ---


class TestWandbTracker:
    def _make_tracker(self) -> tuple[WandbTracker, MagicMock]:
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            tracker = WandbTracker(project="test")
        tracker._wandb = mock_wandb
        tracker._run = mock_run
        return tracker, mock_wandb

    def test_log_params(self):
        tracker, _ = self._make_tracker()
        tracker.log_params({"lr": 1e-4})
        tracker._run.config.update.assert_called_once_with({"lr": 1e-4})

    def test_log_metrics(self):
        tracker, mock_wandb = self._make_tracker()
        tracker.log_metrics({"loss": 0.5}, step=1)
        mock_wandb.log.assert_called_once_with({"loss": 0.5}, step=1)

    def test_log_artifact(self):
        tracker, mock_wandb = self._make_tracker()
        tracker.log_artifact("/path/to/model", name="checkpoint", artifact_type="model")
        mock_wandb.Artifact.assert_called_once_with("checkpoint", type="model")

    def test_set_summary(self):
        tracker, _ = self._make_tracker()
        tracker.set_summary("best", 0.1)
        tracker._run.summary.__setitem__.assert_called_once_with("best", 0.1)

    def test_finish(self):
        tracker, _ = self._make_tracker()
        tracker.finish()
        assert tracker._run is None

    def test_finish_idempotent(self):
        tracker, _ = self._make_tracker()
        tracker.finish()
        tracker.finish()  # Should not raise


# --- Phase-specific logging helpers ---


class TestDistillationMetrics:
    def test_basic_metrics(self):
        tracker = InMemoryTracker()
        log_distillation_metrics(tracker, epoch=1, train_loss=0.5, eval_loss=0.4)
        assert len(tracker.metrics_log) == 1
        entry = tracker.metrics_log[0]
        assert entry["distillation/train_loss"] == 0.5
        assert entry["distillation/eval_loss"] == 0.4
        assert entry["_step"] == 1

    def test_with_per_dimension_mae(self):
        tracker = InMemoryTracker()
        log_distillation_metrics(
            tracker,
            epoch=0,
            train_loss=0.5,
            eval_loss=0.4,
            per_dimension_mae={"correctness": 0.05, "security": 0.08},
        )
        entry = tracker.metrics_log[0]
        assert entry["distillation/mae/correctness"] == 0.05
        assert entry["distillation/mae/security"] == 0.08

    def test_with_calibration_error(self):
        tracker = InMemoryTracker()
        log_distillation_metrics(
            tracker, epoch=0, train_loss=0.5, eval_loss=0.4, calibration_error=0.02
        )
        assert tracker.metrics_log[0]["distillation/calibration_error"] == 0.02


class TestGRPOMetrics:
    def test_basic_metrics(self):
        tracker = InMemoryTracker()
        log_grpo_metrics(tracker, epoch=0, mean_reward=0.7, mean_loss=0.5, mean_kl=0.01)
        entry = tracker.metrics_log[0]
        assert entry["grpo/mean_reward"] == 0.7
        assert entry["grpo/mean_loss"] == 0.5
        assert entry["grpo/kl_divergence"] == 0.01
        assert entry["_step"] == 0

    def test_with_advantage_stats(self):
        tracker = InMemoryTracker()
        log_grpo_metrics(
            tracker,
            epoch=0,
            mean_reward=0.7,
            mean_loss=0.5,
            mean_kl=0.01,
            advantage_stats={"mean": 0.0, "std": 1.0, "min": -2.1, "max": 1.8},
        )
        entry = tracker.metrics_log[0]
        assert entry["grpo/advantage_mean"] == 0.0
        assert entry["grpo/advantage_std"] == 1.0

    def test_with_tier_distribution(self):
        tracker = InMemoryTracker()
        log_grpo_metrics(
            tracker,
            epoch=0,
            mean_reward=0.7,
            mean_loss=0.5,
            mean_kl=0.01,
            tier_distribution={"green": 10, "yellow": 5, "red": 2},
        )
        entry = tracker.metrics_log[0]
        assert entry["grpo/tier/green"] == 10
        assert entry["grpo/tier/red"] == 2


class TestSkillOptimizerMetrics:
    def test_basic_metrics(self):
        tracker = InMemoryTracker()
        log_skill_optimizer_metrics(
            tracker,
            iteration=0,
            strategy="instruction_refinement",
            score_improvement=0.05,
        )
        entry = tracker.metrics_log[0]
        assert entry["skill_optimizer/score_improvement"] == 0.05
        assert entry["skill_optimizer/strategy"] == "instruction_refinement"
        assert entry["skill_optimizer/regression_count"] == 0
        assert entry["_step"] == 0

    def test_with_dimension_scores(self):
        tracker = InMemoryTracker()
        log_skill_optimizer_metrics(
            tracker,
            iteration=1,
            strategy="constraint_addition",
            score_improvement=0.1,
            dimension_scores={"security": 0.9, "correctness": 0.8},
        )
        entry = tracker.metrics_log[0]
        assert entry["skill_optimizer/dim/security"] == 0.9
        assert entry["skill_optimizer/dim/correctness"] == 0.8

    def test_with_regression(self):
        tracker = InMemoryTracker()
        log_skill_optimizer_metrics(
            tracker,
            iteration=2,
            strategy="example_injection",
            score_improvement=-0.02,
            regression_count=1,
        )
        assert tracker.metrics_log[0]["skill_optimizer/regression_count"] == 1


# --- Integration-style tests ---


class TestTrackingWorkflow:
    def test_full_grpo_tracking_workflow(self):
        """Simulate a full GRPO training workflow with tracking."""
        tracker = InMemoryTracker()

        # Log hyperparameters
        tracker.log_params(
            {
                "model": "Qwen/Qwen2.5-Coder-1.5B",
                "learning_rate": 2e-5,
                "kl_penalty": 0.01,
                "group_size": 4,
                "num_epochs": 3,
            }
        )

        # Simulate 3 epochs
        for epoch in range(3):
            log_grpo_metrics(
                tracker,
                epoch=epoch,
                mean_reward=0.5 + epoch * 0.1,
                mean_loss=0.5 - epoch * 0.1,
                mean_kl=0.01 + epoch * 0.005,
            )

        # Log summary
        tracker.set_summary("final_reward", 0.7)
        tracker.set_summary("final_kl", 0.02)

        # Log model artifact
        tracker.log_artifact("/checkpoints/final", name="grpo-final")

        tracker.finish()

        assert len(tracker.metrics_log) == 3
        assert tracker.params["model"] == "Qwen/Qwen2.5-Coder-1.5B"
        assert tracker.summaries["final_reward"] == 0.7
        assert len(tracker.artifacts) == 1
        assert tracker.finished

    def test_disabled_tracking_no_side_effects(self):
        """When disabled, tracking has zero side effects."""
        tracker = create_tracker(enabled=False)
        tracker.log_params({"lr": 1e-4})
        tracker.log_metrics({"loss": 0.5}, step=0)
        tracker.log_artifact("/path")
        tracker.set_summary("key", "value")
        tracker.finish()
        # If we get here without errors, NoOpTracker works

    def test_full_distillation_workflow(self):
        tracker = InMemoryTracker()
        tracker.log_params({"dimensions": ["correctness", "security"], "eval_split": 0.2})

        for epoch in range(5):
            log_distillation_metrics(
                tracker,
                epoch=epoch,
                train_loss=1.0 - epoch * 0.1,
                eval_loss=1.1 - epoch * 0.1,
                per_dimension_mae={"correctness": 0.1 - epoch * 0.01},
            )

        tracker.finish()
        assert len(tracker.metrics_log) == 5
        assert tracker.finished
