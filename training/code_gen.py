"""GRPO trainer for rubric-rewarded code generation.

Trains a code model using Group Relative Policy Optimization with
rubric scores as the reward signal, so it learns to generate code
that inherently passes quality gates.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from shared.models import GateTier

from training.base import RubricTrainer, TrainingConfig


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    group_size: int = 4  # Completions per prompt
    max_completion_length: int = 1024
    kl_penalty: float = 0.01
    length_normalization: bool = True
    diversity_penalty: float = 0.05
    red_tier_multiplier: float = 0.5
    green_tier_bonus: float = 1.2
    seed: int = 42


@dataclass
class RewardInfo:
    """Reward computation details for a single completion."""

    raw_score: float = 0.0
    gate_tier: str = "yellow"
    adjusted_score: float = 0.0
    length_penalty: float = 0.0
    diversity_penalty: float = 0.0
    final_reward: float = 0.0


@dataclass
class GRPOTrainingResult:
    """Result of a GRPO training run."""

    epochs_completed: int = 0
    final_mean_reward: float = 0.0
    final_kl_divergence: float = 0.0
    reward_history: list[float] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class GRPOEvalResult:
    """Evaluation comparing trained model vs base model."""

    base_avg_score: float = 0.0
    trained_avg_score: float = 0.0
    improvement: float = 0.0
    base_green_rate: float = 0.0
    trained_green_rate: float = 0.0
    base_red_rate: float = 0.0
    trained_red_rate: float = 0.0
    num_prompts: int = 0


# --- Prompt templates for code generation ---

_PROMPT_TEMPLATES = [
    "Write a Python function that {task}.",
    "Create a Python script that {task}.",
    "Build a Python class that {task}.",
    "Implement a Python tool that {task}.",
]

_TASK_DESCRIPTIONS = [
    "calculates the total price with tax for a list of items",
    "validates email addresses using regex",
    "reads a CSV file and returns summary statistics",
    "monitors a directory for new files and logs changes",
    "categorizes expenses by department from a spreadsheet",
    "parses JSON configuration and validates required fields",
    "implements a retry mechanism with exponential backoff",
    "creates a simple key-value cache with TTL expiration",
    "filters and sorts a list of records by multiple criteria",
    "generates a formatted report from database query results",
    "handles file uploads with size and type validation",
    "implements a rate limiter using a sliding window",
    "merges multiple dictionaries with conflict resolution",
    "creates a simple task queue with priority ordering",
    "validates and sanitizes user input for a web form",
    "implements a binary search on a sorted collection",
    "creates a thread-safe counter with locking",
    "parses command-line arguments with validation",
    "implements a simple pub-sub event system",
    "creates a data pipeline with transform stages",
]


class CodeGenTrainer(RubricTrainer):
    """Train code generation model with GRPO using rubric rewards.

    Uses Group Relative Policy Optimization: for each prompt, generate
    multiple completions, score them with the rubric engine, normalize
    scores within the group, and update the model to favor high-scoring
    completions.
    """

    def __init__(
        self,
        config: TrainingConfig,
        grpo_config: GRPOConfig | None = None,
        rubric_engine: Any = None,
        gate_evaluator: Any = None,
    ) -> None:
        super().__init__(config)
        self.grpo_config = grpo_config or GRPOConfig()
        self.rubric_engine = rubric_engine
        self.gate_evaluator = gate_evaluator
        self._prompts: list[str] = []

    def prepare_prompts(
        self,
        custom_prompts: list[str] | None = None,
        num_synthetic: int = 200,
    ) -> list[str]:
        """Generate or load training prompts.

        Args:
            custom_prompts: User-provided prompts. If given, synthetic
                generation is skipped.
            num_synthetic: Number of synthetic prompts to generate.

        Returns:
            List of code generation prompts.
        """
        if custom_prompts:
            self._prompts = list(custom_prompts)
            return self._prompts

        rng = random.Random(self.grpo_config.seed)
        prompts: list[str] = []

        for _ in range(num_synthetic):
            template = rng.choice(_PROMPT_TEMPLATES)
            task = rng.choice(_TASK_DESCRIPTIONS)
            prompts.append(template.format(task=task))

        self._prompts = prompts
        return prompts

    def compute_reward(
        self,
        prompt: str,
        completion: str,
        group_completions: list[str] | None = None,
    ) -> RewardInfo:
        """Score a completion using rubric engine + gate evaluation.

        Args:
            prompt: The original prompt.
            completion: Generated code to score.
            group_completions: Other completions in the group (for diversity).

        Returns:
            RewardInfo with detailed scoring breakdown.
        """
        info = RewardInfo()

        # Get rubric score
        if self.rubric_engine is not None:
            score_result = self.rubric_engine.score(code=completion, filename="generated.py")
            info.raw_score = score_result.composite_score
        else:
            # Fallback: simple heuristic scoring
            info.raw_score = _heuristic_score(completion)

        # Gate tier adjustment
        if self.gate_evaluator is not None:
            gate_result = self.gate_evaluator.evaluate(score_result, completion, "generated.py")
            info.gate_tier = gate_result.tier.value
        else:
            info.gate_tier = _heuristic_tier(info.raw_score)

        info.adjusted_score = info.raw_score
        if info.gate_tier == GateTier.RED.value:
            info.adjusted_score *= self.grpo_config.red_tier_multiplier
        elif info.gate_tier == GateTier.GREEN.value:
            info.adjusted_score *= self.grpo_config.green_tier_bonus

        # Length normalization
        if self.grpo_config.length_normalization:
            length = len(completion)
            if length > self.grpo_config.max_completion_length:
                excess = (length - self.grpo_config.max_completion_length) / 1000
                info.length_penalty = min(0.2, excess * 0.1)
            info.adjusted_score -= info.length_penalty

        # Diversity penalty
        if group_completions and self.grpo_config.diversity_penalty > 0:
            similarity = _group_similarity(completion, group_completions)
            info.diversity_penalty = similarity * self.grpo_config.diversity_penalty
            info.adjusted_score -= info.diversity_penalty

        info.final_reward = max(0.0, min(1.5, info.adjusted_score))
        return info

    def compute_group_advantages(self, rewards: list[float]) -> list[float]:
        """Normalize rewards within a group to compute advantages.

        GRPO normalizes rewards relative to the group mean, so the model
        learns from relative quality differences.

        Args:
            rewards: Raw reward values for each completion in the group.

        Returns:
            Advantage values (mean-centered, std-normalized).
        """
        if not rewards:
            return []

        mean = sum(rewards) / len(rewards)

        if len(rewards) == 1:
            return [0.0]

        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = variance**0.5

        if std < 1e-8:
            return [0.0] * len(rewards)

        return [(r - mean) / std for r in rewards]

    def train(self) -> GRPOTrainingResult:
        """Run GRPO training loop.

        Requires prepare_prompts() and load_model() first (or mock model).

        Returns:
            GRPOTrainingResult with reward history and metrics.

        Raises:
            RuntimeError: If no prompts prepared.
        """
        if not self._prompts:
            raise RuntimeError("No prompts prepared. Call prepare_prompts() first.")

        result = GRPOTrainingResult()
        rng = random.Random(self.grpo_config.seed)

        for epoch in range(self.config.num_epochs):
            epoch_rewards: list[float] = []
            shuffled = list(self._prompts)
            rng.shuffle(shuffled)

            for prompt in shuffled:
                # Generate group of completions
                completions = self._generate_group(prompt)

                # Compute rewards
                rewards = []
                for comp in completions:
                    reward_info = self.compute_reward(prompt, comp, group_completions=completions)
                    rewards.append(reward_info.final_reward)

                # Compute advantages
                advantages = self.compute_group_advantages(rewards)

                # Update model (if loaded)
                if self.model is not None:
                    self._update_step(prompt, completions, advantages)

                epoch_rewards.extend(rewards)

            mean_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
            result.reward_history.append(round(mean_reward, 4))
            result.final_mean_reward = round(mean_reward, 4)

        result.epochs_completed = self.config.num_epochs
        result.metrics = {
            "num_prompts": len(self._prompts),
            "group_size": self.grpo_config.group_size,
            "total_completions": len(self._prompts) * self.grpo_config.group_size,
        }

        return result

    def evaluate(
        self,
        test_prompts: list[str] | None = None,
        num_samples: int = 50,
    ) -> GRPOEvalResult:
        """Evaluate trained model vs base on rubric scores.

        Args:
            test_prompts: Prompts to evaluate on.
            num_samples: Number of prompts if generating.

        Returns:
            GRPOEvalResult with comparison metrics.
        """
        prompts = test_prompts or self._prompts[:num_samples]
        if not prompts:
            return GRPOEvalResult()

        trained_scores: list[float] = []
        trained_tiers: list[str] = []

        for prompt in prompts:
            completions = self._generate_group(prompt)
            if completions:
                # Take best completion
                best_reward = None
                best_tier = "yellow"
                for comp in completions:
                    info = self.compute_reward(prompt, comp)
                    if best_reward is None or info.final_reward > best_reward:
                        best_reward = info.final_reward
                        best_tier = info.gate_tier
                trained_scores.append(best_reward or 0.0)
                trained_tiers.append(best_tier)

        avg_score = sum(trained_scores) / len(trained_scores) if trained_scores else 0.0
        green_rate = trained_tiers.count("green") / len(trained_tiers) if trained_tiers else 0.0
        red_rate = trained_tiers.count("red") / len(trained_tiers) if trained_tiers else 0.0

        return GRPOEvalResult(
            trained_avg_score=round(avg_score, 4),
            trained_green_rate=round(green_rate, 4),
            trained_red_rate=round(red_rate, 4),
            num_prompts=len(prompts),
        )

    def export(self, path: str) -> None:
        """Export trained model for use as RubricCodeGenerator.

        Args:
            path: Directory to export to.

        Raises:
            RuntimeError: If no model loaded.
        """
        import json
        from pathlib import Path as P

        self.save_checkpoint(
            path=path,
            metadata={
                "type": "rubric_code_generator",
                "grpo_config": {
                    "group_size": self.grpo_config.group_size,
                    "kl_penalty": self.grpo_config.kl_penalty,
                    "max_completion_length": self.grpo_config.max_completion_length,
                },
            },
        )

        # Save prompt template info
        config_path = P(path) / "generator_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "max_completion_length": self.grpo_config.max_completion_length,
                    "model_name": self.config.model_name,
                },
                indent=2,
            )
        )

    def _generate_group(self, prompt: str) -> list[str]:
        """Generate a group of completions for a prompt.

        If model is loaded, uses the model. Otherwise returns
        placeholder completions for testing.
        """
        if self.model is not None and self.tokenizer is not None:
            return self._model_generate(prompt)

        # Placeholder for testing without a model
        rng = random.Random(hash(prompt) % (2**31))
        return [_synthetic_completion(rng) for _ in range(self.grpo_config.group_size)]

    def _model_generate(self, prompt: str) -> list[str]:
        """Generate completions using the loaded model."""
        completions = []
        for _ in range(self.grpo_config.group_size):
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.grpo_config.max_completion_length,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
            )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            completions.append(text[len(prompt) :])  # Strip prompt prefix
        return completions

    def _update_step(
        self,
        prompt: str,
        completions: list[str],
        advantages: list[float],
    ) -> None:
        """Perform a single GRPO update step (placeholder for real training)."""
        pass


# --- Helper functions ---


def _heuristic_score(code: str) -> float:
    """Simple heuristic scoring when rubric engine not available."""
    score = 0.3  # Base score

    if '"""' in code or "'''" in code:
        score += 0.15  # Has docstrings
    if "def " in code:
        score += 0.1  # Has functions
    if "class " in code:
        score += 0.1  # Has classes
    if "-> " in code or ": " in code:
        score += 0.1  # Has type hints
    if "try:" in code:
        score += 0.05  # Error handling
    if len(code.strip().splitlines()) > 3:
        score += 0.1  # Non-trivial length

    # Penalties
    if "password" in code.lower() and "=" in code:
        score -= 0.2
    if "eval(" in code or "exec(" in code:
        score -= 0.15
    if "SELECT" in code and "f'" in code:
        score -= 0.2  # SQL injection risk

    return max(0.0, min(1.0, score))


def _heuristic_tier(score: float) -> str:
    """Map score to gate tier heuristically."""
    if score >= 0.7:
        return GateTier.GREEN.value
    elif score >= 0.4:
        return GateTier.YELLOW.value
    else:
        return GateTier.RED.value


def _group_similarity(completion: str, others: list[str]) -> float:
    """Estimate similarity of a completion to others in the group."""
    if not others:
        return 0.0

    comp_set = set(completion.split())
    if not comp_set:
        return 0.0

    similarities = []
    for other in others:
        if other == completion:
            continue
        other_set = set(other.split())
        if not other_set:
            continue
        overlap = len(comp_set & other_set) / max(len(comp_set | other_set), 1)
        similarities.append(overlap)

    return sum(similarities) / len(similarities) if similarities else 0.0


def _synthetic_completion(rng: random.Random) -> str:
    """Generate a synthetic code completion for testing."""
    templates = [
        'def process(data):\n    """Process input data."""\n    result = []\n    for item in data:\n        if item:\n            result.append(item)\n    return result\n',
        'def calculate(values: list[float]) -> float:\n    """Calculate the sum of values."""\n    return sum(values)\n',
        'class Handler:\n    """Handle requests."""\n\n    def __init__(self) -> None:\n        self.count = 0\n\n    def handle(self, request: dict) -> dict:\n        self.count += 1\n        return {"status": "ok", "count": self.count}\n',
        "x = 1\ny = 2\nprint(x + y)\n",
        'password = "secret"\ndef check(pw):\n    return pw == password\n',
    ]
    return rng.choice(templates)
