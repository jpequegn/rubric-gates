"""Fast local scorer using a distilled model.

Replaces LLM-as-judge calls with local inference for dimension scoring.
Loads a model exported by ScorerDistillationTrainer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.models import Dimension


@dataclass
class CalibrationMetrics:
    """Calibration metrics for the distilled scorer."""

    expected_calibration_error: float = 0.0
    temperature: float = 1.0
    platt_a: float = 1.0
    platt_b: float = 0.0


class DistilledScorer:
    """Fast local model that predicts rubric scores.

    Loads a model exported by ScorerDistillationTrainer and provides
    fast inference for dimension scoring.
    """

    def __init__(self, model_path: str) -> None:
        """Load trained model from checkpoint.

        Args:
            model_path: Path to exported model directory.

        Raises:
            FileNotFoundError: If model path doesn't exist.
        """
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._model: Any = None
        self._tokenizer: Any = None
        self._config: dict[str, Any] = {}
        self._dimensions: list[str] = [d.value for d in Dimension]
        self._prompt_template: str = "Score this code for {dimension} (0.0-1.0):\n{code}"
        self._calibration = CalibrationMetrics()

        self._load_config()

    def _load_config(self) -> None:
        """Load distillation config from the model directory."""
        config_path = self._model_path / "distill_config.json"
        if config_path.exists():
            self._config = json.loads(config_path.read_text())
            self._dimensions = self._config.get("dimensions", self._dimensions)
            self._prompt_template = self._config.get("prompt_template", self._prompt_template)

        # Load calibration if available
        cal_path = self._model_path / "calibration.json"
        if cal_path.exists():
            cal_data = json.loads(cal_path.read_text())
            self._calibration = CalibrationMetrics(
                expected_calibration_error=cal_data.get("expected_calibration_error", 0.0),
                temperature=cal_data.get("temperature", 1.0),
                platt_a=cal_data.get("platt_a", 1.0),
                platt_b=cal_data.get("platt_b", 0.0),
            )

    def load_model(self) -> None:
        """Load the model and tokenizer into memory.

        Raises:
            ImportError: If transformers not installed.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Inference requires transformers. Install with: uv sync --extra training"
            ) from e

        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_path))
        self._model = AutoModelForCausalLM.from_pretrained(str(self._model_path))
        self._model.eval()

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready for inference."""
        return self._model is not None and self._tokenizer is not None

    @property
    def dimensions(self) -> list[str]:
        """Available scoring dimensions."""
        return list(self._dimensions)

    @property
    def calibration(self) -> CalibrationMetrics:
        """Current calibration metrics."""
        return self._calibration

    def score(self, code: str, dimension: str) -> float:
        """Predict score for a single dimension.

        Args:
            code: Code to evaluate.
            dimension: Dimension name (e.g. "correctness").

        Returns:
            Predicted score 0.0-1.0.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If dimension not recognized.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if dimension not in self._dimensions:
            raise ValueError(f"Unknown dimension '{dimension}'. Available: {self._dimensions}")

        prompt = self._prompt_template.format(dimension=dimension, code=code)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        try:
            import torch

            with torch.no_grad():
                outputs = self._model(**inputs)
            raw_score = outputs.logits[:, -1, 0].sigmoid().item()
            return self._calibrate(max(0.0, min(1.0, raw_score)))
        except Exception:
            return 0.5

    def score_all(self, code: str) -> dict[str, float]:
        """Predict all dimension scores at once.

        Args:
            code: Code to evaluate.

        Returns:
            Dict mapping dimension name to predicted score.

        Raises:
            RuntimeError: If model not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        return {dim: self.score(code, dim) for dim in self._dimensions}

    def _calibrate(self, raw_score: float) -> float:
        """Apply Platt scaling calibration to a raw score.

        Args:
            raw_score: Raw model output 0.0-1.0.

        Returns:
            Calibrated score 0.0-1.0.
        """
        scaled = raw_score / self._calibration.temperature
        calibrated = self._calibration.platt_a * scaled + self._calibration.platt_b
        return max(0.0, min(1.0, calibrated))

    def save_calibration(self, metrics: CalibrationMetrics) -> None:
        """Save calibration parameters.

        Args:
            metrics: Calibration metrics to save.
        """
        self._calibration = metrics
        cal_path = self._model_path / "calibration.json"
        cal_path.write_text(
            json.dumps(
                {
                    "expected_calibration_error": metrics.expected_calibration_error,
                    "temperature": metrics.temperature,
                    "platt_a": metrics.platt_a,
                    "platt_b": metrics.platt_b,
                },
                indent=2,
            )
        )

    def get_metadata(self) -> dict[str, Any]:
        """Get model metadata including config and calibration."""
        meta_path = self._model_path / "rubric_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta["calibration"] = {
            "ece": self._calibration.expected_calibration_error,
            "temperature": self._calibration.temperature,
        }
        meta["dimensions"] = self._dimensions
        return meta
