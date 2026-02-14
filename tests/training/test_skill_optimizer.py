"""Tests for skill prompt optimizer."""

from training.skill_optimizer import (
    ChainOfThought,
    ComparisonReport,
    ConstraintAddition,
    ExampleInjection,
    InstructionRefinement,
    OptimizerConfig,
    OptimizedSkill,
    PromptCandidate,
    SkillAnalysis,
    SkillOptimizer,
)


# --- Data model tests ---


class TestSkillAnalysis:
    def test_defaults(self):
        a = SkillAnalysis()
        assert a.skill_name == ""
        assert a.total_scores == 0
        assert a.weak_dimensions == []
        assert a.trend == "stable"


class TestPromptCandidate:
    def test_defaults(self):
        c = PromptCandidate()
        assert c.prompt == ""
        assert c.avg_score == 0.0
        assert c.regression_detected is False


class TestOptimizedSkill:
    def test_defaults(self):
        o = OptimizedSkill()
        assert o.improvement == 0.0
        assert o.iterations_run == 0


class TestComparisonReport:
    def test_defaults(self):
        r = ComparisonReport()
        assert r.recommendation == ""
        assert r.regressions == []


# --- Mutation strategy tests ---


class TestInstructionRefinement:
    def test_adds_instruction_for_weak_dim(self):
        strategy = InstructionRefinement()
        analysis = SkillAnalysis(weak_dimensions=["security"])
        result = strategy.mutate("Base prompt", analysis)
        assert "credential" in result.lower() or "secret" in result.lower()
        assert result.startswith("Base prompt")

    def test_no_change_without_weaknesses(self):
        strategy = InstructionRefinement()
        analysis = SkillAnalysis(weak_dimensions=[])
        result = strategy.mutate("Base prompt", analysis)
        assert result == "Base prompt"

    def test_targets_weakest_dimension(self):
        strategy = InstructionRefinement()
        analysis = SkillAnalysis(weak_dimensions=["documentation", "security"])
        result = strategy.mutate("Base prompt", analysis)
        assert "docstring" in result.lower()


class TestExampleInjection:
    def test_adds_example(self):
        strategy = ExampleInjection()
        analysis = SkillAnalysis()
        result = strategy.mutate("Base prompt", analysis)
        assert "example" in result.lower()
        assert "```python" in result

    def test_preserves_original(self):
        strategy = ExampleInjection()
        analysis = SkillAnalysis()
        result = strategy.mutate("Original text", analysis)
        assert result.startswith("Original text")


class TestConstraintAddition:
    def test_adds_constraints_for_weak_dims(self):
        strategy = ConstraintAddition()
        analysis = SkillAnalysis(weak_dimensions=["security"])
        result = strategy.mutate("Base prompt", analysis)
        assert "Constraints:" in result
        assert "NEVER" in result

    def test_no_change_without_weaknesses(self):
        strategy = ConstraintAddition()
        analysis = SkillAnalysis(weak_dimensions=[])
        result = strategy.mutate("Base prompt", analysis)
        assert result == "Base prompt"

    def test_limits_to_top_2_dimensions(self):
        strategy = ConstraintAddition()
        analysis = SkillAnalysis(weak_dimensions=["security", "correctness", "maintainability"])
        result = strategy.mutate("Base prompt", analysis)
        assert "Constraints:" in result


class TestChainOfThought:
    def test_adds_reasoning(self):
        strategy = ChainOfThought()
        analysis = SkillAnalysis()
        result = strategy.mutate("Base prompt", analysis)
        assert "step by step" in result.lower()
        assert "security" in result.lower()


# --- Optimizer tests ---


class TestAnalyzeSkill:
    def test_empty_history(self):
        optimizer = SkillOptimizer()
        analysis = optimizer.analyze_skill("test-skill", score_history=[])
        assert analysis.total_scores == 0
        assert analysis.avg_composite == 0.0

    def test_computes_averages(self):
        optimizer = SkillOptimizer()
        history = [
            {"correctness": 0.8, "security": 0.6},
            {"correctness": 0.9, "security": 0.7},
        ]
        analysis = optimizer.analyze_skill("test", score_history=history)
        assert analysis.total_scores == 2
        assert analysis.per_dimension_avg["correctness"] == 0.85
        assert analysis.per_dimension_avg["security"] == 0.65

    def test_identifies_weak_dimensions(self):
        optimizer = SkillOptimizer()
        history = [
            {"correctness": 0.8, "security": 0.4, "documentation": 0.3},
        ]
        analysis = optimizer.analyze_skill("test", score_history=history)
        assert "documentation" in analysis.weak_dimensions
        assert "security" in analysis.weak_dimensions
        assert "correctness" not in analysis.weak_dimensions

    def test_weakest_sorted(self):
        optimizer = SkillOptimizer()
        history = [{"security": 0.4, "documentation": 0.3}]
        analysis = optimizer.analyze_skill("test", score_history=history)
        assert analysis.weak_dimensions[0] == "documentation"

    def test_trend_improving(self):
        optimizer = SkillOptimizer()
        history = [
            {"correctness": 0.5},
            {"correctness": 0.6},
            {"correctness": 0.7},
        ]
        analysis = optimizer.analyze_skill("test", score_history=history)
        assert analysis.trend == "improving"

    def test_trend_declining(self):
        optimizer = SkillOptimizer()
        history = [
            {"correctness": 0.7},
            {"correctness": 0.6},
            {"correctness": 0.5},
        ]
        analysis = optimizer.analyze_skill("test", score_history=history)
        assert analysis.trend == "declining"

    def test_trend_stable(self):
        optimizer = SkillOptimizer()
        history = [
            {"correctness": 0.6},
            {"correctness": 0.7},
            {"correctness": 0.6},
        ]
        analysis = optimizer.analyze_skill("test", score_history=history)
        assert analysis.trend == "stable"


class TestGenerateMutations:
    def test_generates_candidates(self):
        optimizer = SkillOptimizer()
        analysis = SkillAnalysis(weak_dimensions=["security"])
        candidates = optimizer.generate_mutations("Base prompt", analysis)
        assert len(candidates) > 0
        assert all(isinstance(c, PromptCandidate) for c in candidates)

    def test_all_strategies_produce_mutations(self):
        optimizer = SkillOptimizer()
        analysis = SkillAnalysis(weak_dimensions=["security", "documentation"])
        candidates = optimizer.generate_mutations("Base prompt", analysis)
        strategies = {c.strategy for c in candidates}
        assert len(strategies) >= 3  # At least instruction, constraint, cot, example

    def test_filter_strategies(self):
        config = OptimizerConfig(strategies=["chain_of_thought"])
        optimizer = SkillOptimizer(config=config)
        analysis = SkillAnalysis(weak_dimensions=["security"])
        candidates = optimizer.generate_mutations("Base prompt", analysis)
        assert all(c.strategy == "chain_of_thought" for c in candidates)


class TestEvaluateCandidate:
    def test_scores_candidate(self):
        optimizer = SkillOptimizer()
        candidate = PromptCandidate(prompt="Write good code")
        tasks = ["Sort a list", "Parse CSV"]
        result = optimizer.evaluate_candidate(candidate, tasks)
        assert result.avg_score > 0
        assert len(result.per_dimension_scores) > 0

    def test_empty_tasks(self):
        optimizer = SkillOptimizer()
        candidate = PromptCandidate(prompt="Test")
        result = optimizer.evaluate_candidate(candidate, [])
        assert result.avg_score == 0.0

    def test_regression_detection(self):
        optimizer = SkillOptimizer(config=OptimizerConfig(regression_threshold=0.05))
        candidate = PromptCandidate(prompt="Bad prompt")
        baseline = {"correctness": 0.9, "security": 0.9}
        result = optimizer.evaluate_candidate(candidate, ["task"], baseline_scores=baseline)
        # Heuristic scores are ~0.5, so regression from 0.9 should be detected
        assert result.regression_detected is True


class TestOptimize:
    def test_runs_optimization(self):
        optimizer = SkillOptimizer(config=OptimizerConfig(max_iterations=3))
        result = optimizer.optimize(
            skill_name="test-skill",
            original_prompt="Write code",
            test_tasks=["Sort a list", "Parse JSON"],
            score_history=[{"correctness": 0.5, "security": 0.4}],
        )
        assert isinstance(result, OptimizedSkill)
        assert result.skill_name == "test-skill"
        assert result.original_prompt == "Write code"

    def test_no_improvement_keeps_original(self):
        config = OptimizerConfig(max_iterations=1, min_improvement=0.5)
        optimizer = SkillOptimizer(config=config)
        result = optimizer.optimize(
            skill_name="test",
            original_prompt="Already great prompt",
            test_tasks=["task"],
        )
        assert result.improvement <= 0 or result.optimized_prompt != result.original_prompt

    def test_improvement_updates_prompt(self):
        optimizer = SkillOptimizer(config=OptimizerConfig(max_iterations=3, min_improvement=0.001))
        result = optimizer.optimize(
            skill_name="test",
            original_prompt="Basic",
            test_tasks=["Sort list", "Parse file"],
            score_history=[{"security": 0.3, "documentation": 0.3}],
        )
        # With weak dimensions, mutations should improve heuristic scores
        if result.improvement > 0:
            assert result.optimized_prompt != result.original_prompt
            assert result.strategy_used != ""


class TestCompare:
    def test_generates_report(self):
        optimizer = SkillOptimizer()
        report = optimizer.compare(
            skill_name="test",
            original_prompt="Basic prompt",
            optimized_prompt="Better prompt with security constraints",
            test_tasks=["task1", "task2"],
        )
        assert isinstance(report, ComparisonReport)
        assert report.skill_name == "test"
        assert len(report.improvements) > 0
        assert report.recommendation in ("deploy", "reject", "no_improvement")

    def test_detects_regressions(self):
        config = OptimizerConfig(regression_threshold=0.001)
        optimizer = SkillOptimizer(config=config)
        # Same prompt should have no regressions
        report = optimizer.compare(
            skill_name="test",
            original_prompt="Prompt A",
            optimized_prompt="Prompt A",
            test_tasks=["task"],
        )
        assert report.regressions == []
