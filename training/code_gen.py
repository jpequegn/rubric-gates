"""GRPO trainer for rubric-rewarded code generation.

Trains a code model using Group Relative Policy Optimization with
rubric scores as the reward signal, so it learns to generate code
that inherently passes quality gates.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
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
    gradient_clip_max_norm: float = 1.0
    lr_scheduler: str = "cosine"  # "cosine" | "linear" | "constant"
    save_every_epoch: bool = True
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
    kl_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
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
        self._ref_model: Any = None
        self._optimizer: Any = None
        self._scheduler: Any = None
        self._global_step: int = 0
        self._accumulated_loss: float = 0.0
        self._accumulated_kl: float = 0.0
        self._step_count: int = 0

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
            epoch_losses: list[float] = []
            epoch_kls: list[float] = []
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
                    step_metrics = self._update_step(prompt, completions, advantages)
                    epoch_losses.append(step_metrics.get("loss", 0.0))
                    epoch_kls.append(step_metrics.get("kl_divergence", 0.0))

                epoch_rewards.extend(rewards)

            mean_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
            mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            mean_kl = sum(epoch_kls) / len(epoch_kls) if epoch_kls else 0.0

            result.reward_history.append(round(mean_reward, 4))
            result.loss_history.append(round(mean_loss, 4))
            result.kl_history.append(round(mean_kl, 4))
            result.final_mean_reward = round(mean_reward, 4)
            result.final_kl_divergence = round(mean_kl, 4)

            # Checkpoint after each epoch
            if self.grpo_config.save_every_epoch and self.model is not None:
                epoch_path = str(Path(self.config.output_dir) / f"checkpoint-epoch-{epoch}")
                self.save_checkpoint(
                    path=epoch_path,
                    metadata={
                        "epoch": epoch,
                        "mean_reward": mean_reward,
                        "mean_loss": mean_loss,
                        "mean_kl": mean_kl,
                        "global_step": self._global_step,
                    },
                )

        result.epochs_completed = self.config.num_epochs
        result.metrics = {
            "num_prompts": len(self._prompts),
            "group_size": self.grpo_config.group_size,
            "total_completions": len(self._prompts) * self.grpo_config.group_size,
            "global_steps": self._global_step,
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

    def setup_training(self) -> None:
        """Initialize optimizer, scheduler, and reference model for GRPO.

        Must be called after load_model(). Creates a frozen reference copy
        for KL divergence computation, sets up AdamW optimizer, and
        configures learning rate scheduling.

        Raises:
            RuntimeError: If no model loaded.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        import copy

        import torch

        # Freeze reference model (for KL penalty)
        self._ref_model = copy.deepcopy(self.model)
        self._ref_model.eval()
        for param in self._ref_model.parameters():
            param.requires_grad = False

        # Set up optimizer — only train parameters that require grad
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        # Learning rate scheduler
        total_steps = (
            len(self._prompts) * self.config.num_epochs // self.config.gradient_accumulation_steps
        )
        total_steps = max(total_steps, 1)
        self._scheduler = _create_scheduler(
            self._optimizer,
            self.grpo_config.lr_scheduler,
            total_steps,
            self.config.warmup_ratio,
        )

        self._global_step = 0
        self._accumulated_loss = 0.0
        self._accumulated_kl = 0.0
        self._step_count = 0

    def _update_step(
        self,
        prompt: str,
        completions: list[str],
        advantages: list[float],
    ) -> dict[str, float]:
        """Perform a single GRPO update step.

        Computes policy gradient loss weighted by advantages, adds KL
        divergence penalty against the reference model, accumulates
        gradients, and steps the optimizer when the accumulation window
        is full.

        Args:
            prompt: The input prompt.
            completions: Generated completions for this prompt.
            advantages: Normalized advantage values per completion.

        Returns:
            Dict with step metrics (loss, kl_divergence).
        """
        import torch

        if self._optimizer is None:
            self.setup_training()

        self.model.train()
        device = next(self.model.parameters()).device

        total_loss = torch.tensor(0.0, device=device)
        total_kl = torch.tensor(0.0, device=device)
        valid_count = 0

        for completion, advantage in zip(completions, advantages):
            if abs(advantage) < 1e-8:
                continue

            text = prompt + completion
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass through policy model
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            log_probs = -outputs.loss  # avg negative log-likelihood

            # Forward pass through reference model (no grad)
            with torch.no_grad():
                ref_outputs = self._ref_model(**inputs, labels=inputs["input_ids"])
                ref_log_probs = -ref_outputs.loss

            # KL divergence: E[log(pi/pi_ref)] = log_probs - ref_log_probs
            kl = log_probs - ref_log_probs

            # GRPO policy gradient: -advantage * log_probs + kl_penalty * kl
            step_loss = -advantage * log_probs + self.grpo_config.kl_penalty * kl

            total_loss = total_loss + step_loss
            total_kl = total_kl + kl.detach().abs()
            valid_count += 1

        if valid_count == 0:
            return {"loss": 0.0, "kl_divergence": 0.0}

        # Average over group
        avg_loss = total_loss / valid_count
        avg_kl = (total_kl / valid_count).item()

        # Scale for gradient accumulation
        scaled_loss = avg_loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()

        self._accumulated_loss += avg_loss.item()
        self._accumulated_kl += avg_kl
        self._step_count += 1

        # Step optimizer when accumulation window is full
        if self._step_count >= self.config.gradient_accumulation_steps:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.grpo_config.gradient_clip_max_norm,
            )

            self._optimizer.step()
            if self._scheduler is not None:
                self._scheduler.step()
            self._optimizer.zero_grad()

            self._global_step += 1
            avg_step_loss = self._accumulated_loss / self._step_count
            avg_step_kl = self._accumulated_kl / self._step_count

            self._accumulated_loss = 0.0
            self._accumulated_kl = 0.0
            self._step_count = 0

            return {"loss": avg_step_loss, "kl_divergence": avg_step_kl}

        return {"loss": avg_loss.item(), "kl_divergence": avg_kl}

    def save_training_state(self, path: str) -> Path:
        """Save full training state for resumption.

        Saves model, optimizer, scheduler, and training progress.

        Args:
            path: Directory to save state to.

        Returns:
            Path to the saved state directory.
        """
        import torch

        state_path = Path(path)
        state_path.mkdir(parents=True, exist_ok=True)

        # Save model checkpoint
        self.save_checkpoint(path=path)

        # Save optimizer and scheduler state
        training_state = {
            "global_step": self._global_step,
            "optimizer_state_dict": (self._optimizer.state_dict() if self._optimizer else None),
            "scheduler_state_dict": (self._scheduler.state_dict() if self._scheduler else None),
            "grpo_config": {
                "group_size": self.grpo_config.group_size,
                "kl_penalty": self.grpo_config.kl_penalty,
                "gradient_clip_max_norm": self.grpo_config.gradient_clip_max_norm,
                "lr_scheduler": self.grpo_config.lr_scheduler,
            },
        }
        torch.save(training_state, state_path / "training_state.pt")

        # Save prompts
        if self._prompts:
            (state_path / "prompts.json").write_text(json.dumps(self._prompts))

        return state_path

    def load_training_state(self, path: str) -> None:
        """Resume training from a saved state.

        Loads model, optimizer, scheduler, and training progress.

        Args:
            path: Directory containing the saved state.

        Raises:
            FileNotFoundError: If state files not found.
        """
        import torch

        state_path = Path(path)
        if not state_path.exists():
            raise FileNotFoundError(f"Training state not found: {path}")

        # Load model checkpoint
        self.load_checkpoint(path)

        # Set up training (creates optimizer, scheduler, ref model)
        prompts_file = state_path / "prompts.json"
        if prompts_file.exists():
            self._prompts = json.loads(prompts_file.read_text())

        self.setup_training()

        # Restore optimizer/scheduler state
        state_file = state_path / "training_state.pt"
        if state_file.exists():
            training_state = torch.load(state_file, weights_only=False)
            self._global_step = training_state.get("global_step", 0)
            if training_state.get("optimizer_state_dict") and self._optimizer:
                self._optimizer.load_state_dict(training_state["optimizer_state_dict"])
            if training_state.get("scheduler_state_dict") and self._scheduler:
                self._scheduler.load_state_dict(training_state["scheduler_state_dict"])


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


def _create_scheduler(
    optimizer: Any,
    scheduler_type: str,
    total_steps: int,
    warmup_ratio: float,
) -> Any:
    """Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: One of "cosine", "linear", "constant".
        total_steps: Total number of optimizer steps.
        warmup_ratio: Fraction of steps for warmup.

    Returns:
        A PyTorch LR scheduler.
    """
    import torch

    warmup_steps = int(total_steps * warmup_ratio)

    if scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)

    if scheduler_type == "linear":

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return max(1e-8, step / max(warmup_steps, 1))
            remaining = max(total_steps - step, 0)
            total_decay = max(total_steps - warmup_steps, 1)
            return max(0.0, remaining / total_decay)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Default: cosine
    def cosine_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / max(warmup_steps, 1))
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lambda)


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
