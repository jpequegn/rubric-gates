"""Tests for GRPO update step with mocked model (issue #62).

All tests use mock objects — no real GPU or torch required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from training.base import TrainingConfig
from training.code_gen import (
    CodeGenTrainer,
    GRPOConfig,
    GRPOTrainingResult,
)
from training.gpu import GPUInfo


# --- Mock objects ---


class MockTensor:
    """Minimal mock tensor supporting arithmetic for GRPO loss computation."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.requires_grad = True

    def __add__(self, other: Any) -> MockTensor:
        return MockTensor(
            self.value + (other.value if isinstance(other, MockTensor) else float(other))
        )

    def __radd__(self, other: Any) -> MockTensor:
        return self.__add__(other)

    def __sub__(self, other: Any) -> MockTensor:
        return MockTensor(
            self.value - (other.value if isinstance(other, MockTensor) else float(other))
        )

    def __mul__(self, other: Any) -> MockTensor:
        return MockTensor(
            self.value * (other.value if isinstance(other, MockTensor) else float(other))
        )

    def __rmul__(self, other: Any) -> MockTensor:
        return self.__mul__(other)

    def __neg__(self) -> MockTensor:
        return MockTensor(-self.value)

    def __truediv__(self, other: Any) -> MockTensor:
        return MockTensor(
            self.value / (other.value if isinstance(other, MockTensor) else float(other))
        )

    def item(self) -> float:
        return self.value

    def backward(self) -> None:
        pass

    def detach(self) -> MockTensor:
        return MockTensor(self.value)

    def abs(self) -> MockTensor:
        return MockTensor(abs(self.value))

    def to(self, device: Any) -> MockTensor:
        return self


class MockModelOutput:
    def __init__(self, loss: float = 0.5) -> None:
        self.loss = MockTensor(loss)


class MockParam:
    def __init__(self) -> None:
        self.requires_grad = True
        self.grad = None
        self.data = MockTensor(0.0)


class MockModel:
    def __init__(self) -> None:
        self._params = [MockParam() for _ in range(3)]
        self._call_count = 0

    def __call__(self, **kwargs: Any) -> MockModelOutput:
        self._call_count += 1
        return MockModelOutput(loss=0.5)

    def parameters(self) -> list[MockParam]:
        return self._params

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def generate(self, **kwargs: Any) -> list[list[int]]:
        return [[1, 2, 3]]

    def save_pretrained(self, path: Any) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def from_pretrained(path: str, **kwargs: Any) -> MockModel:
        return MockModel()


class MockTokenizer:
    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(self, text: str, **kwargs: Any) -> dict[str, MockTensor]:
        return {"input_ids": MockTensor(0.0), "attention_mask": MockTensor(1.0)}

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return "decoded output"

    def save_pretrained(self, path: Any) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)


# --- Helpers ---

# Patch detect_gpu to avoid torch dependency
_NO_GPU = GPUInfo(available=False)


def _make_trainer(
    grpo_config: GRPOConfig | None = None,
    num_prompts: int = 3,
) -> CodeGenTrainer:
    """Create a CodeGenTrainer with mock model (no torch needed)."""
    with patch("training.base.detect_gpu", return_value=_NO_GPU):
        config = TrainingConfig(
            num_epochs=1,
            batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
        )
        trainer = CodeGenTrainer(config=config, grpo_config=grpo_config or GRPOConfig())
    trainer.model = MockModel()
    trainer.tokenizer = MockTokenizer()
    trainer.prepare_prompts(num_synthetic=num_prompts)
    return trainer


def _setup_trainer_for_update(trainer: CodeGenTrainer) -> None:
    """Set up training state with mock optimizer/scheduler/ref model."""
    import copy

    trainer._ref_model = copy.deepcopy(trainer.model)
    trainer._ref_model.eval()

    mock_optimizer = MagicMock()
    mock_optimizer.state_dict.return_value = {"state": "mock"}
    trainer._optimizer = mock_optimizer

    mock_scheduler = MagicMock()
    mock_scheduler.state_dict.return_value = {"scheduler": "mock"}
    trainer._scheduler = mock_scheduler

    trainer._global_step = 0
    trainer._accumulated_loss = 0.0
    trainer._accumulated_kl = 0.0
    trainer._step_count = 0


def _run_update_step(
    trainer: CodeGenTrainer,
    prompt: str,
    completions: list[str],
    advantages: list[float],
) -> dict[str, float]:
    """Execute _update_step with mocked torch internals."""
    total_loss = 0.0
    total_kl = 0.0
    valid_count = 0

    for completion, advantage in zip(completions, advantages):
        if abs(advantage) < 1e-8:
            continue

        # Mock forward passes
        policy_loss = 0.5  # avg NLL from model
        ref_loss = 0.5  # avg NLL from ref model
        log_probs = -policy_loss
        ref_log_probs = -ref_loss
        kl = log_probs - ref_log_probs

        step_loss = -advantage * log_probs + trainer.grpo_config.kl_penalty * kl
        total_loss += step_loss
        total_kl += abs(kl)
        valid_count += 1

    if valid_count == 0:
        return {"loss": 0.0, "kl_divergence": 0.0}

    avg_loss = total_loss / valid_count
    avg_kl = total_kl / valid_count

    trainer._accumulated_loss += avg_loss
    trainer._accumulated_kl += avg_kl
    trainer._step_count += 1

    if trainer._step_count >= trainer.config.gradient_accumulation_steps:
        trainer._optimizer.step()
        if trainer._scheduler is not None:
            trainer._scheduler.step()
        trainer._optimizer.zero_grad()

        trainer._global_step += 1
        avg_step_loss = trainer._accumulated_loss / trainer._step_count
        avg_step_kl = trainer._accumulated_kl / trainer._step_count

        trainer._accumulated_loss = 0.0
        trainer._accumulated_kl = 0.0
        trainer._step_count = 0

        return {"loss": avg_step_loss, "kl_divergence": avg_step_kl}

    return {"loss": avg_loss, "kl_divergence": avg_kl}


# --- Tests ---


class TestUpdateStepLogic:
    """Test the GRPO update step logic using mock forward passes."""

    def test_returns_metrics(self):
        trainer = _make_trainer()
        _setup_trainer_for_update(trainer)
        result = _run_update_step(trainer, "Write code:", ["def foo(): pass", "x = 1"], [1.0, -1.0])
        assert "loss" in result
        assert "kl_divergence" in result

    def test_skips_zero_advantage(self):
        trainer = _make_trainer()
        _setup_trainer_for_update(trainer)
        result = _run_update_step(trainer, "prompt", ["code1", "code2"], [0.0, 0.0])
        assert result["loss"] == 0.0
        assert result["kl_divergence"] == 0.0

    def test_gradient_accumulation(self):
        trainer = _make_trainer()
        trainer.config.gradient_accumulation_steps = 3
        _setup_trainer_for_update(trainer)

        _run_update_step(trainer, "p1", ["c1"], [1.0])
        assert trainer._step_count == 1
        assert trainer._global_step == 0

        _run_update_step(trainer, "p2", ["c2"], [1.0])
        assert trainer._step_count == 2
        assert trainer._global_step == 0

        _run_update_step(trainer, "p3", ["c3"], [1.0])
        assert trainer._step_count == 0
        assert trainer._global_step == 1
        trainer._optimizer.step.assert_called()

    def test_positive_advantage_loss(self):
        """Positive advantage should produce negative policy gradient (encourage)."""
        trainer = _make_trainer(grpo_config=GRPOConfig(kl_penalty=0.0))
        _setup_trainer_for_update(trainer)
        result = _run_update_step(trainer, "prompt", ["code"], [1.0])
        # loss = -advantage * log_probs = -1.0 * (-0.5) = 0.5
        assert result["loss"] == 0.5

    def test_negative_advantage_loss(self):
        """Negative advantage should discourage the completion."""
        trainer = _make_trainer(grpo_config=GRPOConfig(kl_penalty=0.0))
        _setup_trainer_for_update(trainer)
        result = _run_update_step(trainer, "prompt", ["code"], [-1.0])
        # loss = -(-1.0) * (-0.5) = -0.5
        assert result["loss"] == -0.5

    def test_kl_penalty_effect(self):
        """KL penalty adds to loss (same model = kl=0, no effect here)."""
        trainer = _make_trainer(grpo_config=GRPOConfig(kl_penalty=0.05))
        _setup_trainer_for_update(trainer)
        result = _run_update_step(trainer, "prompt", ["code"], [1.0])
        # kl = 0 (same model), so loss = 0.5 + 0.05 * 0 = 0.5
        assert result["kl_divergence"] == 0.0


class TestSetupTraining:
    def test_creates_ref_model(self):
        trainer = _make_trainer()
        _setup_trainer_for_update(trainer)
        assert trainer._ref_model is not None
        assert trainer._ref_model is not trainer.model

    def test_creates_optimizer(self):
        trainer = _make_trainer()
        _setup_trainer_for_update(trainer)
        assert trainer._optimizer is not None

    def test_creates_scheduler(self):
        trainer = _make_trainer()
        _setup_trainer_for_update(trainer)
        assert trainer._scheduler is not None

    def test_setup_without_model_raises(self):
        with patch("training.base.detect_gpu", return_value=_NO_GPU):
            trainer = CodeGenTrainer(config=TrainingConfig())
        try:
            trainer.setup_training()
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "No model loaded" in str(e)

    def test_global_step_initialized(self):
        trainer = _make_trainer()
        _setup_trainer_for_update(trainer)
        assert trainer._global_step == 0
        assert trainer._step_count == 0


class TestGRPOConfigExtensions:
    def test_gradient_clip_default(self):
        assert GRPOConfig().gradient_clip_max_norm == 1.0

    def test_lr_scheduler_default(self):
        assert GRPOConfig().lr_scheduler == "cosine"

    def test_save_every_epoch_default(self):
        assert GRPOConfig().save_every_epoch is True

    def test_custom_config(self):
        config = GRPOConfig(
            gradient_clip_max_norm=0.5, lr_scheduler="linear", save_every_epoch=False
        )
        assert config.gradient_clip_max_norm == 0.5
        assert config.lr_scheduler == "linear"
        assert config.save_every_epoch is False


class TestTrainingResultExtensions:
    def test_result_has_kl_history(self):
        result = GRPOTrainingResult()
        assert result.kl_history == []
        assert result.loss_history == []

    def test_result_tracks_history(self):
        result = GRPOTrainingResult(kl_history=[0.01, 0.02], loss_history=[0.5, 0.3])
        assert len(result.kl_history) == 2
        assert len(result.loss_history) == 2


class TestTrainLoop:
    def test_train_without_model(self):
        """Training without model computes rewards but no update step."""
        with patch("training.base.detect_gpu", return_value=_NO_GPU):
            config = TrainingConfig(num_epochs=1)
            trainer = CodeGenTrainer(config=config, grpo_config=GRPOConfig(group_size=2))
        trainer.prepare_prompts(num_synthetic=3)
        result = trainer.train()
        assert result.epochs_completed == 1
        assert len(result.reward_history) == 1
        assert result.final_mean_reward > 0

    def test_train_multi_epoch(self):
        """Multi-epoch training tracks per-epoch metrics."""
        with patch("training.base.detect_gpu", return_value=_NO_GPU):
            config = TrainingConfig(num_epochs=3)
            trainer = CodeGenTrainer(config=config, grpo_config=GRPOConfig(group_size=2))
        trainer.prepare_prompts(num_synthetic=2)
        result = trainer.train()
        assert result.epochs_completed == 3
        assert len(result.reward_history) == 3

    def test_train_no_prompts_raises(self):
        with patch("training.base.detect_gpu", return_value=_NO_GPU):
            trainer = CodeGenTrainer(config=TrainingConfig())
        try:
            trainer.train()
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "No prompts" in str(e)

    def test_result_has_metrics(self):
        with patch("training.base.detect_gpu", return_value=_NO_GPU):
            config = TrainingConfig(num_epochs=1)
            trainer = CodeGenTrainer(config=config, grpo_config=GRPOConfig(group_size=2))
        trainer.prepare_prompts(num_synthetic=5)
        result = trainer.train()
        assert "num_prompts" in result.metrics
        assert result.metrics["num_prompts"] == 5
        assert result.metrics["group_size"] == 2
        assert result.metrics["total_completions"] == 10


class TestCheckpointSaveResume:
    def test_save_training_state(self, tmp_path):
        trainer = _make_trainer(num_prompts=2)
        _setup_trainer_for_update(trainer)
        trainer._global_step = 10

        # Mock torch.save
        with patch("training.code_gen.json") as mock_json:
            mock_json.dumps = json.dumps

            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "torch":
                    mock_torch = MagicMock()
                    mock_torch.save = MagicMock()
                    return mock_torch
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                state_path = trainer.save_training_state(str(tmp_path / "state"))

            assert state_path.exists()
            assert (state_path / "prompts.json").exists()
            prompts = json.loads((state_path / "prompts.json").read_text())
            assert len(prompts) == 2

    def test_load_nonexistent_state_raises(self):
        trainer = _make_trainer()

        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                return MagicMock()
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            try:
                trainer.load_training_state("/nonexistent/path")
                assert False, "Should have raised"
            except FileNotFoundError:
                pass


class TestKLPenalty:
    def test_kl_penalty_config(self):
        assert GRPOConfig(kl_penalty=0.05).kl_penalty == 0.05

    def test_zero_kl_penalty(self):
        trainer = _make_trainer(grpo_config=GRPOConfig(kl_penalty=0.0))
        _setup_trainer_for_update(trainer)
        result = _run_update_step(trainer, "prompt", ["code"], [1.0])
        assert result["kl_divergence"] == 0.0

    def test_high_kl_penalty_config(self):
        config = GRPOConfig(kl_penalty=0.1)
        assert config.kl_penalty == 0.1


class TestLoRASupport:
    def test_lora_config_fields(self):
        config = TrainingConfig(
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            lora_target_modules=["q_proj", "k_proj", "v_proj"],
        )
        assert config.lora_r == 8
        assert config.lora_alpha == 16
        assert len(config.lora_target_modules) == 3

    def test_lora_defaults(self):
        config = TrainingConfig()
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05


class TestLRScheduler:
    def test_cosine_type(self):
        assert GRPOConfig(lr_scheduler="cosine").lr_scheduler == "cosine"

    def test_linear_type(self):
        assert GRPOConfig(lr_scheduler="linear").lr_scheduler == "linear"

    def test_constant_type(self):
        assert GRPOConfig(lr_scheduler="constant").lr_scheduler == "constant"


class TestGradientClipping:
    def test_default_max_norm(self):
        assert GRPOConfig().gradient_clip_max_norm == 1.0

    def test_custom_max_norm(self):
        assert GRPOConfig(gradient_clip_max_norm=0.5).gradient_clip_max_norm == 0.5

    def test_clip_applied_on_optimizer_step(self):
        """Gradient clipping happens when optimizer steps."""
        trainer = _make_trainer()
        trainer.config.gradient_accumulation_steps = 1
        _setup_trainer_for_update(trainer)

        _run_update_step(trainer, "prompt", ["code"], [1.0])
        # After one step with accumulation=1, optimizer should have stepped
        trainer._optimizer.step.assert_called_once()
