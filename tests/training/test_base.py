"""Tests for base trainer."""

import json
from unittest.mock import MagicMock, patch

import pytest

from training.base import RubricTrainer, TrainingConfig
from training.gpu import GPUInfo


class TestTrainingConfig:
    def test_defaults(self):
        config = TrainingConfig()
        assert config.model_name == "Qwen/Qwen2.5-Coder-1.5B"
        assert config.output_dir == "checkpoints"
        assert config.learning_rate == 2e-5
        assert config.num_epochs == 3
        assert config.batch_size == 0
        assert config.quantize is True
        assert config.lora_r == 16
        assert config.seed == 42

    def test_custom_values(self):
        config = TrainingConfig(
            model_name="custom/model",
            learning_rate=1e-4,
            num_epochs=5,
            batch_size=4,
        )
        assert config.model_name == "custom/model"
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 5
        assert config.batch_size == 4

    def test_lora_target_modules_default(self):
        config = TrainingConfig()
        assert config.lora_target_modules == ["q_proj", "v_proj"]

    def test_lora_target_modules_custom(self):
        config = TrainingConfig(lora_target_modules=["q_proj", "k_proj", "v_proj"])
        assert len(config.lora_target_modules) == 3


class TestRubricTrainer:
    @patch("training.base.detect_gpu")
    def test_init_auto_batch_size_no_gpu(self, mock_detect):
        mock_detect.return_value = GPUInfo(available=False)
        config = TrainingConfig()
        trainer = RubricTrainer(config)

        assert config.batch_size == 1
        assert trainer.model is None
        assert trainer.tokenizer is None

    @patch("training.base.detect_gpu")
    def test_init_auto_batch_size_gpu(self, mock_detect):
        mock_detect.return_value = GPUInfo(available=True, vram_gb=16.0)
        config = TrainingConfig()
        RubricTrainer(config)

        assert config.batch_size == 4

    @patch("training.base.detect_gpu")
    def test_init_explicit_batch_size(self, mock_detect):
        mock_detect.return_value = GPUInfo(available=True, vram_gb=24.0)
        config = TrainingConfig(batch_size=2)
        RubricTrainer(config)

        assert config.batch_size == 2

    @patch("training.base.detect_gpu")
    def test_recommended_model_size(self, mock_detect):
        mock_detect.return_value = GPUInfo(available=True, vram_gb=16.0, compute_capability="8.0")
        config = TrainingConfig()
        trainer = RubricTrainer(config)

        assert trainer.recommended_model_size == "3-7B"

    @patch("training.base.detect_gpu")
    def test_recommended_model_size_no_gpu(self, mock_detect):
        mock_detect.return_value = GPUInfo(available=False)
        config = TrainingConfig()
        trainer = RubricTrainer(config)

        assert trainer.recommended_model_size == "cpu-only"


class TestSaveCheckpoint:
    @patch("training.base.detect_gpu")
    def test_save_raises_without_model(self, mock_detect):
        mock_detect.return_value = GPUInfo(available=False)
        trainer = RubricTrainer(TrainingConfig())

        with pytest.raises(RuntimeError, match="No model loaded"):
            trainer.save_checkpoint()

    @patch("training.base.detect_gpu")
    def test_save_checkpoint(self, mock_detect, tmp_path):
        mock_detect.return_value = GPUInfo(available=False)
        config = TrainingConfig(output_dir=str(tmp_path / "out"))
        trainer = RubricTrainer(config)

        # Mock model and tokenizer
        trainer.model = MagicMock()
        trainer.tokenizer = MagicMock()

        path = trainer.save_checkpoint()

        assert path.exists()
        trainer.model.save_pretrained.assert_called_once_with(path)
        trainer.tokenizer.save_pretrained.assert_called_once_with(path)

        # Check metadata
        meta = json.loads((path / "rubric_meta.json").read_text())
        assert meta["model_name"] == config.model_name
        assert meta["quantize"] is True

    @patch("training.base.detect_gpu")
    def test_save_checkpoint_custom_path(self, mock_detect, tmp_path):
        mock_detect.return_value = GPUInfo(available=False)
        trainer = RubricTrainer(TrainingConfig())
        trainer.model = MagicMock()
        trainer.tokenizer = MagicMock()

        custom_path = tmp_path / "my_checkpoint"
        path = trainer.save_checkpoint(str(custom_path))

        assert path == custom_path
        assert path.exists()

    @patch("training.base.detect_gpu")
    def test_save_checkpoint_with_metadata(self, mock_detect, tmp_path):
        mock_detect.return_value = GPUInfo(available=False)
        trainer = RubricTrainer(TrainingConfig(output_dir=str(tmp_path)))
        trainer.model = MagicMock()
        trainer.tokenizer = MagicMock()

        trainer.save_checkpoint(metadata={"epoch": 3, "loss": 0.15})

        path = tmp_path / "checkpoint-latest" / "rubric_meta.json"
        meta = json.loads(path.read_text())
        assert meta["epoch"] == 3
        assert meta["loss"] == 0.15


class TestLoadCheckpoint:
    @patch("training.base.detect_gpu")
    def test_load_checkpoint_not_found(self, mock_detect):
        mock_detect.return_value = GPUInfo(available=False)
        trainer = RubricTrainer(TrainingConfig())

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            trainer.load_checkpoint("/nonexistent/path")

    @patch("training.base.detect_gpu")
    def test_load_checkpoint(self, mock_detect, tmp_path):
        mock_detect.return_value = GPUInfo(available=False)
        trainer = RubricTrainer(TrainingConfig())

        # Create checkpoint directory
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            model, tokenizer = trainer.load_checkpoint(str(checkpoint))

        assert model is mock_model
        assert tokenizer is mock_tokenizer
        assert trainer.model is mock_model
        assert trainer.tokenizer is mock_tokenizer


class TestGetCheckpointMetadata:
    @patch("training.base.detect_gpu")
    def test_no_metadata_file(self, mock_detect, tmp_path):
        mock_detect.return_value = GPUInfo(available=False)
        trainer = RubricTrainer(TrainingConfig())

        meta = trainer.get_checkpoint_metadata(str(tmp_path))
        assert meta == {}

    @patch("training.base.detect_gpu")
    def test_reads_metadata(self, mock_detect, tmp_path):
        mock_detect.return_value = GPUInfo(available=False)
        trainer = RubricTrainer(TrainingConfig())

        meta_path = tmp_path / "rubric_meta.json"
        meta_path.write_text(json.dumps({"model_name": "test", "epoch": 5}))

        meta = trainer.get_checkpoint_metadata(str(tmp_path))
        assert meta["model_name"] == "test"
        assert meta["epoch"] == 5


class TestLoadModel:
    @patch("training.base.detect_gpu")
    def test_load_model_import_error(self, mock_detect):
        mock_detect.return_value = GPUInfo(available=False)
        trainer = RubricTrainer(TrainingConfig())

        # transformers isn't installed in test environment
        # This should raise ImportError or succeed if installed
        # We test the import guard by mocking
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError):
                trainer.load_model()
