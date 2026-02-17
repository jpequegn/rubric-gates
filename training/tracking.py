"""Experiment tracking for rubric-gates training.

Protocol-based abstraction for experiment tracking. wandb is the default
implementation but is lazy-imported — no dependency unless tracking is
enabled.

Usage:
    tracker = create_tracker(enabled=True, project="rubric-gates/grpo")
    tracker.log_params({"lr": 1e-4, "epochs": 3})
    tracker.log_metrics({"loss": 0.5, "kl": 0.01}, step=1)
    tracker.finish()
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExperimentTracker(Protocol):
    """Protocol for experiment tracking backends."""

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters for the run."""
        ...

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics at an optional step."""
        ...

    def log_artifact(self, path: str, name: str = "", artifact_type: str = "model") -> None:
        """Log a file or directory as an artifact."""
        ...

    def set_summary(self, key: str, value: Any) -> None:
        """Set a summary metric for the run."""
        ...

    def finish(self) -> None:
        """Finalize the tracking run."""
        ...


class NoOpTracker:
    """Tracker that does nothing. Used when tracking is disabled."""

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        pass

    def log_artifact(self, path: str, name: str = "", artifact_type: str = "model") -> None:
        pass

    def set_summary(self, key: str, value: Any) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbTracker:
    """Experiment tracker using Weights & Biases.

    Lazy-imports wandb on init so there's no dependency unless tracking
    is actually enabled.

    Args:
        project: wandb project name (e.g., "rubric-gates/grpo").
        run_name: Optional name for the run.
        config: Initial config/hyperparameters to log.
        offline: If True, log locally and sync later.
        tags: Optional tags for the run.
    """

    def __init__(
        self,
        project: str = "rubric-gates",
        run_name: str = "",
        config: dict[str, Any] | None = None,
        offline: bool = False,
        tags: list[str] | None = None,
    ) -> None:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb is required for experiment tracking. Install with: pip install wandb"
            ) from e

        if offline:
            import os

            os.environ["WANDB_MODE"] = "offline"

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            name=run_name or None,
            config=config or {},
            tags=tags or [],
            reinit=True,
        )

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        if self._run is not None:
            self._run.config.update(params)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics at an optional step."""
        if self._run is not None:
            self._wandb.log(metrics, step=step)

    def log_artifact(self, path: str, name: str = "", artifact_type: str = "model") -> None:
        """Log a file or directory as a wandb artifact."""
        if self._run is not None:
            artifact_name = name or path.split("/")[-1]
            artifact = self._wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_dir(path)
            self._run.log_artifact(artifact)

    def set_summary(self, key: str, value: Any) -> None:
        """Set a summary metric."""
        if self._run is not None:
            self._run.summary[key] = value

    def finish(self) -> None:
        """Finish the wandb run."""
        if self._run is not None:
            self._run.finish()
            self._run = None


class InMemoryTracker:
    """Tracker that stores everything in memory. Useful for testing.

    All logged params, metrics, artifacts, and summaries are accessible
    via public attributes for assertions.
    """

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.metrics_log: list[dict[str, Any]] = []
        self.artifacts: list[dict[str, str]] = []
        self.summaries: dict[str, Any] = {}
        self.finished: bool = False

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        entry = dict(metrics)
        if step is not None:
            entry["_step"] = step
        self.metrics_log.append(entry)

    def log_artifact(self, path: str, name: str = "", artifact_type: str = "model") -> None:
        self.artifacts.append({"path": path, "name": name, "type": artifact_type})

    def set_summary(self, key: str, value: Any) -> None:
        self.summaries[key] = value

    def finish(self) -> None:
        self.finished = True


def create_tracker(
    enabled: bool = False,
    project: str = "rubric-gates",
    run_name: str = "",
    phase: str = "",
    config: dict[str, Any] | None = None,
    offline: bool = False,
    tags: list[str] | None = None,
) -> ExperimentTracker:
    """Factory to create an experiment tracker.

    Args:
        enabled: Whether tracking is enabled. If False, returns NoOpTracker.
        project: wandb project name.
        run_name: Optional run name.
        phase: Training phase (distillation, grpo, skill_optimizer).
            Appended to project name: "rubric-gates/grpo".
        config: Initial hyperparameters.
        offline: If True, log offline for later sync.
        tags: Optional tags for the run.

    Returns:
        An ExperimentTracker implementation.
    """
    if not enabled:
        return NoOpTracker()

    full_project = f"{project}/{phase}" if phase else project
    all_tags = list(tags or [])
    if phase:
        all_tags.append(phase)

    return WandbTracker(
        project=full_project,
        run_name=run_name,
        config=config,
        offline=offline,
        tags=all_tags,
    )


# --- Phase-specific logging helpers ---


def log_distillation_metrics(
    tracker: ExperimentTracker,
    epoch: int,
    train_loss: float,
    eval_loss: float,
    per_dimension_mae: dict[str, float] | None = None,
    calibration_error: float | None = None,
) -> None:
    """Log metrics for a distillation training epoch.

    Args:
        tracker: The experiment tracker.
        epoch: Current epoch number.
        train_loss: Training loss for the epoch.
        eval_loss: Evaluation loss for the epoch.
        per_dimension_mae: MAE per scoring dimension.
        calibration_error: Calibration error metric.
    """
    metrics: dict[str, Any] = {
        "distillation/train_loss": train_loss,
        "distillation/eval_loss": eval_loss,
    }
    if per_dimension_mae:
        for dim, mae in per_dimension_mae.items():
            metrics[f"distillation/mae/{dim}"] = mae
    if calibration_error is not None:
        metrics["distillation/calibration_error"] = calibration_error
    tracker.log_metrics(metrics, step=epoch)


def log_grpo_metrics(
    tracker: ExperimentTracker,
    epoch: int,
    mean_reward: float,
    mean_loss: float,
    mean_kl: float,
    advantage_stats: dict[str, float] | None = None,
    tier_distribution: dict[str, int] | None = None,
) -> None:
    """Log metrics for a GRPO training epoch.

    Args:
        tracker: The experiment tracker.
        epoch: Current epoch number.
        mean_reward: Mean reward for the epoch.
        mean_loss: Mean policy gradient loss.
        mean_kl: Mean KL divergence from reference model.
        advantage_stats: Stats on advantage values (mean, std, min, max).
        tier_distribution: Count of green/yellow/red gate tiers.
    """
    metrics: dict[str, Any] = {
        "grpo/mean_reward": mean_reward,
        "grpo/mean_loss": mean_loss,
        "grpo/kl_divergence": mean_kl,
    }
    if advantage_stats:
        for key, val in advantage_stats.items():
            metrics[f"grpo/advantage_{key}"] = val
    if tier_distribution:
        for tier, count in tier_distribution.items():
            metrics[f"grpo/tier/{tier}"] = count
    tracker.log_metrics(metrics, step=epoch)


def log_skill_optimizer_metrics(
    tracker: ExperimentTracker,
    iteration: int,
    strategy: str,
    score_improvement: float,
    dimension_scores: dict[str, float] | None = None,
    regression_count: int = 0,
) -> None:
    """Log metrics for a skill optimizer iteration.

    Args:
        tracker: The experiment tracker.
        iteration: Optimization iteration number.
        strategy: Mutation strategy used.
        score_improvement: Score improvement from this iteration.
        dimension_scores: Per-dimension scores after mutation.
        regression_count: Number of dimensions that regressed.
    """
    metrics: dict[str, Any] = {
        "skill_optimizer/score_improvement": score_improvement,
        "skill_optimizer/strategy": strategy,
        "skill_optimizer/regression_count": regression_count,
    }
    if dimension_scores:
        for dim, score in dimension_scores.items():
            metrics[f"skill_optimizer/dim/{dim}"] = score
    tracker.log_metrics(metrics, step=iteration)
