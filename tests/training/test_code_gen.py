"""Tests for GRPO code generation trainer."""

from unittest.mock import MagicMock, patch

import pytest

from training.base import TrainingConfig
from training.code_gen import (
    CodeGenTrainer,
    GRPOConfig,
    GRPOEvalResult,
    GRPOTrainingResult,
    RewardInfo,
    _group_similarity,
    _heuristic_score,
    _heuristic_tier,
)
from training.gpu import GPUInfo


def _make_trainer(**kwargs):
    with patch("training.base.detect_gpu") as mock_detect:
        mock_detect.return_value = GPUInfo(available=False)
        return CodeGenTrainer(TrainingConfig(), **kwargs)


class TestGRPOConfig:
    def test_defaults(self):
        config = GRPOConfig()
        assert config.group_size == 4
        assert config.max_completion_length == 1024
        assert config.kl_penalty == 0.01
        assert config.length_normalization is True
        assert config.diversity_penalty == 0.05
        assert config.red_tier_multiplier == 0.5
        assert config.green_tier_bonus == 1.2

    def test_custom(self):
        config = GRPOConfig(group_size=8, kl_penalty=0.05)
        assert config.group_size == 8
        assert config.kl_penalty == 0.05


class TestRewardInfo:
    def test_defaults(self):
        info = RewardInfo()
        assert info.raw_score == 0.0
        assert info.gate_tier == "yellow"
        assert info.final_reward == 0.0


class TestGRPOTrainingResult:
    def test_defaults(self):
        result = GRPOTrainingResult()
        assert result.epochs_completed == 0
        assert result.reward_history == []
        assert result.metrics == {}


class TestGRPOEvalResult:
    def test_defaults(self):
        result = GRPOEvalResult()
        assert result.base_avg_score == 0.0
        assert result.trained_avg_score == 0.0
        assert result.num_prompts == 0


class TestCodeGenTrainerInit:
    def test_default_config(self):
        trainer = _make_trainer()
        assert trainer.grpo_config is not None
        assert trainer.rubric_engine is None
        assert trainer.gate_evaluator is None
        assert trainer._prompts == []

    def test_custom_grpo_config(self):
        grpo = GRPOConfig(group_size=8)
        trainer = _make_trainer(grpo_config=grpo)
        assert trainer.grpo_config.group_size == 8


class TestPreparePrompts:
    def test_synthetic_prompts(self):
        trainer = _make_trainer()
        prompts = trainer.prepare_prompts(num_synthetic=50)
        assert len(prompts) == 50
        assert all(isinstance(p, str) for p in prompts)

    def test_custom_prompts(self):
        trainer = _make_trainer()
        custom = ["Write a sort function.", "Implement a stack."]
        prompts = trainer.prepare_prompts(custom_prompts=custom)
        assert prompts == custom

    def test_custom_overrides_synthetic(self):
        trainer = _make_trainer()
        custom = ["Custom prompt"]
        prompts = trainer.prepare_prompts(custom_prompts=custom, num_synthetic=100)
        assert len(prompts) == 1

    def test_prompts_are_diverse(self):
        trainer = _make_trainer()
        prompts = trainer.prepare_prompts(num_synthetic=20)
        unique = set(prompts)
        # Should have some variety (at least 5 unique out of 20)
        assert len(unique) >= 5

    def test_reproducible_with_seed(self):
        t1 = _make_trainer(grpo_config=GRPOConfig(seed=42))
        p1 = t1.prepare_prompts(num_synthetic=10)

        t2 = _make_trainer(grpo_config=GRPOConfig(seed=42))
        p2 = t2.prepare_prompts(num_synthetic=10)

        assert p1 == p2


class TestComputeReward:
    def test_heuristic_fallback(self):
        trainer = _make_trainer()
        code = 'def foo():\n    """Docstring."""\n    return 1\n'
        info = trainer.compute_reward("prompt", code)

        assert isinstance(info, RewardInfo)
        assert info.raw_score > 0
        assert info.final_reward > 0

    def test_red_tier_penalty(self):
        trainer = _make_trainer()
        bad_code = 'password = "admin123"\n'
        info = trainer.compute_reward("prompt", bad_code)

        assert info.gate_tier == "red"
        assert info.adjusted_score < info.raw_score

    def test_green_tier_bonus(self):
        trainer = _make_trainer()
        good_code = 'def process(data: list) -> list:\n    """Process data."""\n    try:\n        return [x for x in data if x]\n    except Exception:\n        return []\n'
        info = trainer.compute_reward("prompt", good_code)

        assert info.gate_tier == "green"
        assert info.adjusted_score > info.raw_score

    def test_with_rubric_engine(self):
        mock_engine = MagicMock()
        mock_score = MagicMock()
        mock_score.composite_score = 0.85
        mock_engine.score.return_value = mock_score

        trainer = _make_trainer()
        trainer.rubric_engine = mock_engine

        info = trainer.compute_reward("prompt", "code")
        assert info.raw_score == 0.85

    def test_with_gate_evaluator(self):
        mock_engine = MagicMock()
        mock_score = MagicMock()
        mock_score.composite_score = 0.9
        mock_engine.score.return_value = mock_score

        mock_gate = MagicMock()
        from shared.models import GateTier

        mock_gate_result = MagicMock()
        mock_gate_result.tier = GateTier.GREEN
        mock_gate.evaluate.return_value = mock_gate_result

        trainer = _make_trainer()
        trainer.rubric_engine = mock_engine
        trainer.gate_evaluator = mock_gate

        info = trainer.compute_reward("prompt", "code")
        assert info.gate_tier == "green"

    def test_length_penalty(self):
        trainer = _make_trainer(grpo_config=GRPOConfig(max_completion_length=10))
        long_code = "x = 1\n" * 100
        info = trainer.compute_reward("prompt", long_code)
        assert info.length_penalty > 0

    def test_diversity_penalty(self):
        trainer = _make_trainer()
        comp = "def foo(): pass"
        others = ["def foo(): return 1", "def foo(): return 2"]
        info = trainer.compute_reward("prompt", comp, group_completions=others)
        assert info.diversity_penalty > 0


class TestComputeGroupAdvantages:
    def test_empty(self):
        trainer = _make_trainer()
        assert trainer.compute_group_advantages([]) == []

    def test_single(self):
        trainer = _make_trainer()
        assert trainer.compute_group_advantages([0.5]) == [0.0]

    def test_equal_rewards(self):
        trainer = _make_trainer()
        advantages = trainer.compute_group_advantages([0.5, 0.5, 0.5])
        assert all(a == 0.0 for a in advantages)

    def test_varied_rewards(self):
        trainer = _make_trainer()
        advantages = trainer.compute_group_advantages([0.2, 0.5, 0.8])

        # Highest reward should have highest advantage
        assert advantages[2] > advantages[1] > advantages[0]
        # Should be roughly mean-centered
        assert abs(sum(advantages) / len(advantages)) < 0.01

    def test_two_rewards(self):
        trainer = _make_trainer()
        advantages = trainer.compute_group_advantages([0.3, 0.7])
        assert advantages[0] < 0
        assert advantages[1] > 0


class TestTrain:
    def test_raises_without_prompts(self):
        trainer = _make_trainer()
        with pytest.raises(RuntimeError, match="No prompts prepared"):
            trainer.train()

    def test_train_returns_result(self):
        trainer = _make_trainer()
        trainer.prepare_prompts(num_synthetic=5)
        result = trainer.train()

        assert isinstance(result, GRPOTrainingResult)
        assert result.epochs_completed == 3
        assert len(result.reward_history) == 3
        assert result.metrics["num_prompts"] == 5
        assert result.metrics["group_size"] == 4

    def test_reward_history_populated(self):
        trainer = _make_trainer()
        trainer.prepare_prompts(num_synthetic=3)
        result = trainer.train()

        assert all(isinstance(r, float) for r in result.reward_history)
        assert all(r >= 0 for r in result.reward_history)


class TestEvaluate:
    def test_empty(self):
        trainer = _make_trainer()
        result = trainer.evaluate()
        assert result.num_prompts == 0

    def test_with_prompts(self):
        trainer = _make_trainer()
        trainer.prepare_prompts(num_synthetic=10)
        result = trainer.evaluate(num_samples=5)

        assert isinstance(result, GRPOEvalResult)
        assert result.num_prompts == 5
        assert result.trained_avg_score >= 0
        assert 0.0 <= result.trained_green_rate <= 1.0
        assert 0.0 <= result.trained_red_rate <= 1.0

    def test_custom_prompts(self):
        trainer = _make_trainer()
        result = trainer.evaluate(test_prompts=["Write a function."])
        assert result.num_prompts == 1


class TestExport:
    def test_raises_without_model(self):
        trainer = _make_trainer()
        with pytest.raises(RuntimeError, match="No model loaded"):
            trainer.export("/tmp/test")

    def test_exports_config(self, tmp_path):
        import json

        trainer = _make_trainer()
        trainer.model = MagicMock()
        trainer.tokenizer = MagicMock()

        export_path = tmp_path / "exported"
        trainer.export(str(export_path))

        assert (export_path / "generator_config.json").exists()
        assert (export_path / "rubric_meta.json").exists()

        config = json.loads((export_path / "generator_config.json").read_text())
        assert "max_completion_length" in config
        assert "model_name" in config


class TestHeuristicScore:
    def test_base_score(self):
        score = _heuristic_score("")
        assert score == 0.3

    def test_good_code(self):
        code = 'def foo(x: int) -> int:\n    """Docstring."""\n    try:\n        return x\n    except Exception:\n        return 0\n'
        score = _heuristic_score(code)
        assert score > 0.5

    def test_bad_code(self):
        code = 'password = "admin"\neval(input())\n'
        score = _heuristic_score(code)
        assert score < 0.3

    def test_clamped(self):
        score = _heuristic_score("x")
        assert 0.0 <= score <= 1.0


class TestHeuristicTier:
    def test_green(self):
        assert _heuristic_tier(0.8) == "green"

    def test_yellow(self):
        assert _heuristic_tier(0.5) == "yellow"

    def test_red(self):
        assert _heuristic_tier(0.2) == "red"


class TestGroupSimilarity:
    def test_empty_others(self):
        assert _group_similarity("hello world", []) == 0.0

    def test_identical(self):
        sim = _group_similarity("def foo(): pass", ["def foo(): pass"])
        # Self should be filtered out; if same string, overlap is 1.0
        # but the function skips identical strings
        assert sim == 0.0

    def test_different(self):
        sim = _group_similarity("def foo(): pass", ["class Bar: x = 1"])
        assert 0.0 <= sim <= 1.0

    def test_partial_overlap(self):
        sim = _group_similarity("def foo(): return 1", ["def bar(): return 2"])
        assert sim > 0.0  # Some common tokens
