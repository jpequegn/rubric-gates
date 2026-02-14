"""Base trainer for rubric-guided model training.

Provides model loading with optional quantization, checkpoint management,
and a training loop skeleton that subclasses extend for specific methods
(distillation, GRPO, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.gpu import GPUInfo, detect_gpu, recommend_batch_size, recommend_model_size


@dataclass
class TrainingConfig:
    """Configuration for a training run."""

    model_name: str = "Qwen/Qwen2.5-Coder-1.5B"
    output_dir: str = "checkpoints"
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 0  # 0 = auto-detect from GPU
    max_seq_length: int = 512
    quantize: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 100
    seed: int = 42


class RubricTrainer:
    """Base class for all P4 training phases.

    Handles model loading with optional 4-bit quantization for local GPU,
    checkpoint save/load, and provides a training loop skeleton.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.gpu_info: GPUInfo = detect_gpu()
        self.model: Any = None
        self.tokenizer: Any = None
        self._checkpoint_dir = Path(config.output_dir)

        # Auto-detect batch size if not set
        if config.batch_size == 0:
            config.batch_size = recommend_batch_size(self.gpu_info)

    @property
    def recommended_model_size(self) -> str:
        """Return recommended model size string for current hardware."""
        return recommend_model_size(self.gpu_info)

    def load_model(self) -> tuple[Any, Any]:
        """Load model and tokenizer with optional 4-bit quantization.

        Returns:
            Tuple of (model, tokenizer).

        Raises:
            ImportError: If transformers/torch not installed.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Training requires transformers. Install with: uv sync --extra training"
            ) from e

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {}

        if self.config.quantize and self.gpu_info.available:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype="float16",
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                pass  # Fall back to full precision

        model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **model_kwargs)

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def save_checkpoint(
        self, path: str | None = None, metadata: dict[str, Any] | None = None
    ) -> Path:
        """Save model checkpoint and metadata.

        Args:
            path: Directory to save to. Defaults to output_dir/checkpoint-latest.
            metadata: Extra metadata to save alongside the checkpoint.

        Returns:
            Path to the saved checkpoint directory.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        checkpoint_path = Path(path) if path else self._checkpoint_dir / "checkpoint-latest"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(checkpoint_path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(checkpoint_path)

        # Save training metadata
        meta = {
            "model_name": self.config.model_name,
            "quantize": self.config.quantize,
            "gpu_info": {
                "available": self.gpu_info.available,
                "device_name": self.gpu_info.device_name,
                "vram_gb": self.gpu_info.vram_gb,
            },
        }
        if metadata:
            meta.update(metadata)

        meta_path = checkpoint_path / "rubric_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        return checkpoint_path

    def load_checkpoint(self, path: str) -> tuple[Any, Any]:
        """Load model and tokenizer from a checkpoint.

        Args:
            path: Directory containing the checkpoint.

        Returns:
            Tuple of (model, tokenizer).

        Raises:
            FileNotFoundError: If checkpoint path doesn't exist.
            ImportError: If transformers not installed.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Training requires transformers. Install with: uv sync --extra training"
            ) from e

        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint_path))

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def get_checkpoint_metadata(self, path: str) -> dict[str, Any]:
        """Read metadata from a saved checkpoint.

        Args:
            path: Directory containing the checkpoint.

        Returns:
            Metadata dict, or empty dict if no metadata file found.
        """
        meta_path = Path(path) / "rubric_meta.json"
        if not meta_path.exists():
            return {}
        return json.loads(meta_path.read_text())
