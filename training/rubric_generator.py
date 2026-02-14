"""Code generator using a rubric-trained model.

Loads a model exported by CodeGenTrainer and generates code
that has internalized rubric quality standards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GenerationResult:
    """Result of a code generation request."""

    code: str = ""
    score: float | None = None
    gate_tier: str | None = None
    metadata: dict[str, Any] | None = None


class RubricCodeGenerator:
    """Generate code using a rubric-trained model.

    Loads a model exported by CodeGenTrainer and provides fast local
    code generation that inherently favors high-quality output.
    """

    def __init__(self, model_path: str) -> None:
        """Load generator config from checkpoint.

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
        self._max_completion_length: int = 1024
        self._model_name: str = ""

        self._load_config()

    def _load_config(self) -> None:
        """Load generator config from the model directory."""
        config_path = self._model_path / "generator_config.json"
        if config_path.exists():
            self._config = json.loads(config_path.read_text())
            self._max_completion_length = self._config.get("max_completion_length", 1024)
            self._model_name = self._config.get("model_name", "")

    def load_model(self) -> None:
        """Load the model and tokenizer into memory.

        Raises:
            ImportError: If transformers not installed.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Generation requires transformers. Install with: uv sync --extra training"
            ) from e

        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_path))
        self._model = AutoModelForCausalLM.from_pretrained(str(self._model_path))
        self._model.eval()

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready for generation."""
        return self._model is not None and self._tokenizer is not None

    @property
    def model_name(self) -> str:
        """Original model name used for training."""
        return self._model_name

    def generate(
        self,
        prompt: str,
        max_length: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        num_candidates: int = 1,
    ) -> str:
        """Generate code for a given task description.

        Args:
            prompt: Task description or code prompt.
            max_length: Maximum tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
            top_p: Nucleus sampling threshold.
            num_candidates: Generate N candidates and return the longest.

        Returns:
            Generated code string.

        Raises:
            RuntimeError: If model not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        max_tokens = max_length or self._max_completion_length

        best_code = ""
        for _ in range(num_candidates):
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01),
                top_p=top_p,
            )
            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            code = text[len(prompt) :]  # Strip prompt prefix
            if len(code) > len(best_code):
                best_code = code

        return best_code.strip()

    def generate_and_score(
        self,
        prompt: str,
        rubric_engine: Any = None,
        gate_evaluator: Any = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate code and score it with the rubric engine.

        Args:
            prompt: Task description.
            rubric_engine: Optional RubricEngine for scoring.
            gate_evaluator: Optional GateEvaluator for tier classification.
            **kwargs: Passed to generate().

        Returns:
            GenerationResult with code, score, and tier.

        Raises:
            RuntimeError: If model not loaded.
        """
        code = self.generate(prompt, **kwargs)

        result = GenerationResult(code=code)

        if rubric_engine is not None:
            score_result = rubric_engine.score(code=code, filename="generated.py")
            result.score = score_result.composite_score

            if gate_evaluator is not None:
                gate_result = gate_evaluator.evaluate(score_result, code, "generated.py")
                result.gate_tier = gate_result.tier.value

        return result

    def get_metadata(self) -> dict[str, Any]:
        """Get model metadata."""
        meta_path = self._model_path / "rubric_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta["max_completion_length"] = self._max_completion_length
        meta["model_name"] = self._model_name
        return meta
