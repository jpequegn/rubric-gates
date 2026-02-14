"""Tests for scorer distillation trainer."""

from unittest.mock import MagicMock, patch

import pytest

from shared.models import Dimension
from training.base import TrainingConfig
from training.data import TrainingExample
from training.distillation import (
    DistillationConfig,
    EvalResult,
    ScorerDistillationTrainer,
    TrainingResult,
)


class TestDistillationConfig:
    def test_defaults(self):
        config = DistillationConfig()
        assert len(config.dimensions) == len(Dimension)
        assert config.min_training_examples == 100
        assert config.synthetic_augment_to == 2000
        assert config.eval_split == 0.2
        assert config.max_score_error == 0.1

    def test_custom_dimensions(self):
        config = DistillationConfig(dimensions=["correctness", "security"])
        assert config.dimensions == ["correctness", "security"]

    def test_custom_template(self):
        config = DistillationConfig(prompt_template="Rate {dimension}:\n{code}")
        assert "Rate" in config.prompt_template


class TestTrainingResult:
    def test_defaults(self):
        result = TrainingResult()
        assert result.epochs_completed == 0
        assert result.final_train_loss == 0.0
        assert result.best_eval_loss == float("inf")
        assert result.metrics == {}


class TestEvalResult:
    def test_defaults(self):
        result = EvalResult()
        assert result.mae == 0.0
        assert result.mse == 0.0
        assert result.per_dimension_mae == {}
        assert result.num_examples == 0


class TestScorerDistillationTrainer:
    @patch("training.base.detect_gpu")
    def test_init(self, mock_detect):
        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)
        config = TrainingConfig()
        trainer = ScorerDistillationTrainer(config)

        assert trainer.distill_config is not None
        assert trainer._train_examples == []
        assert trainer._eval_examples == []

    @patch("training.base.detect_gpu")
    def test_init_custom_distill_config(self, mock_detect):
        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)
        distill = DistillationConfig(min_training_examples=50)
        trainer = ScorerDistillationTrainer(TrainingConfig(), distill)

        assert trainer.distill_config.min_training_examples == 50


class TestPrepareData:
    @patch("training.base.detect_gpu")
    def test_synthetic_only(self, mock_detect):
        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)
        trainer = ScorerDistillationTrainer(TrainingConfig())

        train, eval_ = trainer.prepare_data(synthetic_only=True)

        assert len(train) > 0
        assert len(eval_) > 0
        assert len(train) + len(eval_) == 2000

    @patch("training.base.detect_gpu")
    def test_with_storage(self, mock_detect):
        from datetime import datetime, timezone

        from shared.models import DimensionScore, ScoreResult, ScoringMethod
        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)

        mock_storage = MagicMock()
        mock_storage.query.return_value = [
            ScoreResult(
                composite_score=0.75,
                dimension_scores=[
                    DimensionScore(
                        dimension=Dimension.CORRECTNESS,
                        score=0.8,
                        method=ScoringMethod.RULE_BASED,
                    ),
                ],
                user="alice",
                timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            ),
        ]

        trainer = ScorerDistillationTrainer(TrainingConfig())
        train, eval_ = trainer.prepare_data(storage=mock_storage)

        # Should augment with synthetic to reach 2000
        assert len(train) + len(eval_) == 2000

    @patch("training.base.detect_gpu")
    def test_insufficient_data_raises(self, mock_detect):
        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)
        distill = DistillationConfig(min_training_examples=5000, synthetic_augment_to=100)
        trainer = ScorerDistillationTrainer(TrainingConfig(), distill)

        with pytest.raises(ValueError, match="Insufficient data"):
            trainer.prepare_data(synthetic_only=True)

    @patch("training.base.detect_gpu")
    def test_split_ratio(self, mock_detect):
        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)
        distill = DistillationConfig(eval_split=0.3, synthetic_augment_to=100)
        trainer = ScorerDistillationTrainer(TrainingConfig(), distill)

        train, eval_ = trainer.prepare_data(synthetic_only=True)

        assert len(train) == 70
        assert len(eval_) == 30


class TestFormatPrompt:
    @patch("training.base.detect_gpu")
    def test_default_template(self, mock_detect):
        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)
        trainer = ScorerDistillationTrainer(TrainingConfig())

        prompt = trainer.format_prompt("x = 1", "correctness")
        assert "correctness" in prompt
        assert "x = 1" in prompt
        assert "0.0-1.0" in prompt

    @patch("training.base.detect_gpu")
    def test_custom_template(self, mock_detect):
        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)
        distill = DistillationConfig(prompt_template="Rate {dimension}: {code}")
        trainer = ScorerDistillationTrainer(TrainingConfig(), distill)

        prompt = trainer.format_prompt("x = 1", "security")
        assert prompt == "Rate security: x = 1"


class TestFormatTrainingPairs:
    @patch("training.base.detect_gpu")
    def test_generates_pairs(self, mock_detect):
        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)
        trainer = ScorerDistillationTrainer(TrainingConfig())

        examples = [
            TrainingExample(
                code="x = 1",
                composite_score=0.7,
                dimension_scores={"correctness": 0.8, "security": 0.6},
            ),
        ]

        pairs = trainer.format_training_pairs(examples)

        # 2 dimensions with scores = 2 pairs
        assert len(pairs) == 2
        assert all("prompt" in p and "completion" in p for p in pairs)

    @patch("training.base.detect_gpu")
    def test_skips_missing_dimensions(self, mock_detect):
        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)
        trainer = ScorerDistillationTrainer(TrainingConfig())

        examples = [
            TrainingExample(
                code="x = 1",
                composite_score=0.5,
                dimension_scores={"correctness": 0.8},
            ),
        ]

        pairs = trainer.format_training_pairs(examples)
        assert len(pairs) == 1

    @patch("training.base.detect_gpu")
    def test_completion_format(self, mock_detect):
        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)
        trainer = ScorerDistillationTrainer(TrainingConfig())

        examples = [
            TrainingExample(
                code="x = 1",
                composite_score=0.5,
                dimension_scores={"correctness": 0.823},
            ),
        ]

        pairs = trainer.format_training_pairs(examples)
        assert pairs[0]["completion"] == "0.823"


class TestTrain:
    @patch("training.base.detect_gpu")
    def test_train_raises_without_data(self, mock_detect):
        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)
        trainer = ScorerDistillationTrainer(TrainingConfig())

        with pytest.raises(RuntimeError, match="No training data"):
            trainer.train()

    @patch("training.base.detect_gpu")
    def test_train_returns_result(self, mock_detect):
        import sys

        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)
        trainer = ScorerDistillationTrainer(TrainingConfig())
        trainer.prepare_data(synthetic_only=True)

        mock_torch = MagicMock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = trainer.train()

        assert isinstance(result, TrainingResult)
        assert result.epochs_completed == 3
        assert result.metrics["train_examples"] > 0
        assert result.metrics["eval_examples"] > 0


class TestEvaluate:
    @patch("training.base.detect_gpu")
    def test_evaluate_empty(self, mock_detect):
        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)
        trainer = ScorerDistillationTrainer(TrainingConfig())

        result = trainer.evaluate()
        assert result.mae == 0.0
        assert result.num_examples == 0

    @patch("training.base.detect_gpu")
    def test_evaluate_with_data(self, mock_detect):
        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)
        trainer = ScorerDistillationTrainer(TrainingConfig())
        trainer.prepare_data(synthetic_only=True)

        result = trainer.evaluate()

        assert isinstance(result, EvalResult)
        assert result.num_examples > 0
        assert result.mae >= 0.0
        assert result.mse >= 0.0
        assert len(result.per_dimension_mae) > 0

    @patch("training.base.detect_gpu")
    def test_evaluate_custom_examples(self, mock_detect):
        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)
        trainer = ScorerDistillationTrainer(TrainingConfig())

        examples = [
            TrainingExample(
                code="x = 1",
                composite_score=0.5,
                dimension_scores={"correctness": 0.5},
            ),
        ]

        result = trainer.evaluate(examples)
        assert result.num_examples == 1


class TestExport:
    @patch("training.base.detect_gpu")
    def test_export_raises_without_model(self, mock_detect):
        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)
        trainer = ScorerDistillationTrainer(TrainingConfig())

        with pytest.raises(RuntimeError, match="No model loaded"):
            trainer.export("/tmp/test-export")

    @patch("training.base.detect_gpu")
    def test_export_saves_config(self, mock_detect, tmp_path):
        import json

        from training.gpu import GPUInfo

        mock_detect.return_value = GPUInfo(available=False)
        trainer = ScorerDistillationTrainer(TrainingConfig())
        trainer.model = MagicMock()
        trainer.tokenizer = MagicMock()

        export_path = tmp_path / "exported"
        trainer.export(str(export_path))

        assert (export_path / "distill_config.json").exists()
        assert (export_path / "rubric_meta.json").exists()

        config = json.loads((export_path / "distill_config.json").read_text())
        assert "dimensions" in config
        assert "prompt_template" in config

        meta = json.loads((export_path / "rubric_meta.json").read_text())
        assert meta["type"] == "distilled_scorer"
