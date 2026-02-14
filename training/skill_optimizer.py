"""Skill prompt optimizer using rubric reward signals.

Optimizes Claude Code skill prompts by treating rubric scores as
an objective function, generating prompt mutations targeting weak
dimensions, and selecting the best-performing variant.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from shared.models import Dimension


# --- Mutation Strategies ---


class MutationStrategy:
    """Base class for prompt mutation strategies."""

    name: str = "base"

    def mutate(self, prompt: str, analysis: SkillAnalysis) -> str:
        raise NotImplementedError


class InstructionRefinement(MutationStrategy):
    """Add quality/structure instructions targeting weak dimensions."""

    name = "instruction_refinement"

    _DIM_INSTRUCTIONS = {
        "correctness": (
            "\n\nIMPORTANT: Ensure all code is correct and handles edge cases. "
            "Validate inputs and return types."
        ),
        "security": (
            "\n\nIMPORTANT: Never hardcode credentials or secrets. "
            "Validate and sanitize all inputs. Avoid eval(), exec(), and SQL injection."
        ),
        "maintainability": (
            "\n\nIMPORTANT: Write clean, well-structured code with small focused functions. "
            "Follow naming conventions and keep functions under 20 lines."
        ),
        "documentation": (
            "\n\nIMPORTANT: Add docstrings to all public functions and classes. "
            "Include parameter descriptions and return types."
        ),
        "testability": (
            "\n\nIMPORTANT: Design code for testability with dependency injection. "
            "Avoid global state and hard-coded dependencies."
        ),
    }

    def mutate(self, prompt: str, analysis: SkillAnalysis) -> str:
        if not analysis.weak_dimensions:
            return prompt
        # Add instruction for weakest dimension
        weakest = analysis.weak_dimensions[0]
        instruction = self._DIM_INSTRUCTIONS.get(weakest, "")
        return prompt + instruction


class ExampleInjection(MutationStrategy):
    """Add few-shot examples of high-scoring code."""

    name = "example_injection"

    def mutate(self, prompt: str, analysis: SkillAnalysis) -> str:
        example = (
            "\n\nHere is an example of high-quality code:\n"
            "```python\n"
            "def process_items(items: list[dict]) -> list[dict]:\n"
            '    """Process and validate items.\n\n'
            "    Args:\n"
            "        items: List of item dicts with 'name' and 'value' keys.\n\n"
            "    Returns:\n"
            "        Filtered and transformed items.\n"
            '    """\n'
            "    return [\n"
            "        {k: v.strip() if isinstance(v, str) else v for k, v in item.items()}\n"
            "        for item in items\n"
            '        if item.get("name") and item.get("value") is not None\n'
            "    ]\n"
            "```\n"
            "Follow similar patterns for quality, documentation, and structure."
        )
        return prompt + example


class ConstraintAddition(MutationStrategy):
    """Add explicit constraints targeting violations."""

    name = "constraint_addition"

    _DIM_CONSTRAINTS = {
        "security": [
            "NEVER hardcode passwords, API keys, or secrets.",
            "NEVER use eval() or exec() on user input.",
            "ALWAYS parameterize SQL queries.",
            "ALWAYS validate and sanitize external input.",
        ],
        "correctness": [
            "ALWAYS handle edge cases (empty input, None values).",
            "ALWAYS include proper error handling with specific exceptions.",
            "NEVER silently swallow exceptions with bare except clauses.",
        ],
        "maintainability": [
            "Keep functions under 20 lines.",
            "Use descriptive variable and function names.",
            "Avoid deeply nested code (max 3 levels).",
        ],
        "documentation": [
            "Add docstrings to ALL public functions and classes.",
            "Include type hints for all function parameters and returns.",
        ],
        "testability": [
            "Use dependency injection instead of hard-coded dependencies.",
            "Avoid global mutable state.",
            "Design functions to be pure where possible.",
        ],
    }

    def mutate(self, prompt: str, analysis: SkillAnalysis) -> str:
        constraints: list[str] = []
        for dim in analysis.weak_dimensions[:2]:
            constraints.extend(self._DIM_CONSTRAINTS.get(dim, []))
        if not constraints:
            return prompt
        block = "\n\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)
        return prompt + block


class ChainOfThought(MutationStrategy):
    """Add reasoning prompts for broad quality improvement."""

    name = "chain_of_thought"

    def mutate(self, prompt: str, analysis: SkillAnalysis) -> str:
        cot = (
            "\n\nBefore writing code, think step by step:\n"
            "1. What are the security implications?\n"
            "2. How will this be tested?\n"
            "3. Is the code well-documented?\n"
            "4. Are edge cases handled?\n"
            "5. Is the code maintainable and readable?"
        )
        return prompt + cot


# All available strategies
ALL_STRATEGIES: list[MutationStrategy] = [
    InstructionRefinement(),
    ExampleInjection(),
    ConstraintAddition(),
    ChainOfThought(),
]


# --- Data Models ---


@dataclass
class SkillAnalysis:
    """Analysis of a skill's scoring history."""

    skill_name: str = ""
    total_scores: int = 0
    avg_composite: float = 0.0
    per_dimension_avg: dict[str, float] = field(default_factory=dict)
    weak_dimensions: list[str] = field(default_factory=list)
    trend: str = "stable"  # improving, stable, declining
    score_history: list[float] = field(default_factory=list)


@dataclass
class PromptCandidate:
    """A prompt mutation candidate."""

    prompt: str = ""
    strategy: str = ""
    avg_score: float = 0.0
    per_dimension_scores: dict[str, float] = field(default_factory=dict)
    regression_detected: bool = False


@dataclass
class OptimizedSkill:
    """Result of skill optimization."""

    skill_name: str = ""
    original_prompt: str = ""
    optimized_prompt: str = ""
    original_avg_score: float = 0.0
    optimized_avg_score: float = 0.0
    improvement: float = 0.0
    strategy_used: str = ""
    per_dimension_improvement: dict[str, float] = field(default_factory=dict)
    iterations_run: int = 0


@dataclass
class ComparisonReport:
    """Side-by-side comparison of original vs optimized."""

    skill_name: str = ""
    original_scores: dict[str, float] = field(default_factory=dict)
    optimized_scores: dict[str, float] = field(default_factory=dict)
    improvements: dict[str, float] = field(default_factory=dict)
    regressions: list[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class OptimizerConfig:
    """Configuration for skill prompt optimization."""

    max_iterations: int = 10
    min_improvement: float = 0.02
    regression_threshold: float = 0.1
    strategies: list[str] | None = None  # None = all
    seed: int = 42


# --- Optimizer ---


class SkillOptimizer:
    """Optimize Claude Code skill prompts using rubric rewards."""

    def __init__(
        self,
        rubric_engine: Any = None,
        config: OptimizerConfig | None = None,
        storage: Any = None,
    ) -> None:
        self.rubric_engine = rubric_engine
        self.config = config or OptimizerConfig()
        self.storage = storage

        # Filter strategies if configured
        if self.config.strategies:
            self._strategies = [s for s in ALL_STRATEGIES if s.name in self.config.strategies]
        else:
            self._strategies = list(ALL_STRATEGIES)

    def analyze_skill(
        self,
        skill_name: str,
        score_history: list[dict[str, float]] | None = None,
    ) -> SkillAnalysis:
        """Analyze a skill's scoring history and identify weaknesses.

        Args:
            skill_name: Name of the skill to analyze.
            score_history: Optional pre-loaded scores. Each dict maps
                dimension name to score. If not provided, queries storage.

        Returns:
            SkillAnalysis with dimension averages and weak spots.
        """
        if score_history is None:
            score_history = self._load_score_history(skill_name)

        analysis = SkillAnalysis(
            skill_name=skill_name,
            total_scores=len(score_history),
        )

        if not score_history:
            return analysis

        # Compute per-dimension averages
        dim_totals: dict[str, list[float]] = {}
        composites: list[float] = []

        for scores in score_history:
            entry_scores = []
            for dim, score in scores.items():
                dim_totals.setdefault(dim, []).append(score)
                entry_scores.append(score)
            if entry_scores:
                composites.append(sum(entry_scores) / len(entry_scores))

        analysis.per_dimension_avg = {
            dim: round(sum(vals) / len(vals), 4) for dim, vals in dim_totals.items() if vals
        }
        analysis.avg_composite = round(sum(composites) / len(composites), 4) if composites else 0.0
        analysis.score_history = [round(c, 4) for c in composites]

        # Identify weak dimensions (below 0.6)
        analysis.weak_dimensions = sorted(
            [dim for dim, avg in analysis.per_dimension_avg.items() if avg < 0.6],
            key=lambda d: analysis.per_dimension_avg[d],
        )

        # Trend detection
        if len(composites) >= 3:
            recent = composites[-3:]
            if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                analysis.trend = "improving"
            elif all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
                analysis.trend = "declining"

        return analysis

    def generate_mutations(self, prompt: str, analysis: SkillAnalysis) -> list[PromptCandidate]:
        """Generate prompt variations targeting weak dimensions.

        Args:
            prompt: Current skill prompt.
            analysis: Analysis of the skill's scoring history.

        Returns:
            List of candidate prompts, one per strategy.
        """
        candidates: list[PromptCandidate] = []
        for strategy in self._strategies:
            mutated = strategy.mutate(prompt, analysis)
            if mutated != prompt:
                candidates.append(PromptCandidate(prompt=mutated, strategy=strategy.name))
        return candidates

    def evaluate_candidate(
        self,
        candidate: PromptCandidate,
        test_tasks: list[str],
        baseline_scores: dict[str, float] | None = None,
    ) -> PromptCandidate:
        """Evaluate a candidate prompt on test tasks.

        Args:
            candidate: The prompt candidate to evaluate.
            test_tasks: Test task descriptions.
            baseline_scores: Original per-dimension averages for regression detection.

        Returns:
            Updated candidate with scores and regression flag.
        """
        if not test_tasks:
            return candidate

        dim_scores: dict[str, list[float]] = {}
        all_composites: list[float] = []

        for task in test_tasks:
            scores = self._score_prompt(candidate.prompt, task)
            for dim, score in scores.items():
                dim_scores.setdefault(dim, []).append(score)
            if scores:
                all_composites.append(sum(scores.values()) / len(scores))

        candidate.per_dimension_scores = {
            dim: round(sum(vals) / len(vals), 4) for dim, vals in dim_scores.items() if vals
        }
        candidate.avg_score = (
            round(sum(all_composites) / len(all_composites), 4) if all_composites else 0.0
        )

        # Regression detection
        if baseline_scores:
            for dim, baseline in baseline_scores.items():
                new_score = candidate.per_dimension_scores.get(dim, 0.0)
                if baseline - new_score > self.config.regression_threshold:
                    candidate.regression_detected = True
                    break

        return candidate

    def optimize(
        self,
        skill_name: str,
        original_prompt: str,
        test_tasks: list[str],
        score_history: list[dict[str, float]] | None = None,
    ) -> OptimizedSkill:
        """Run full optimization loop for a skill.

        Args:
            skill_name: Name of the skill.
            original_prompt: Current prompt text.
            test_tasks: Tasks to evaluate candidates on.
            score_history: Optional pre-loaded score history.

        Returns:
            OptimizedSkill with best prompt and improvement metrics.
        """
        analysis = self.analyze_skill(skill_name, score_history)

        # Evaluate baseline
        baseline_candidate = PromptCandidate(prompt=original_prompt, strategy="original")
        baseline_candidate = self.evaluate_candidate(baseline_candidate, test_tasks)
        baseline_scores = baseline_candidate.per_dimension_scores

        best = OptimizedSkill(
            skill_name=skill_name,
            original_prompt=original_prompt,
            optimized_prompt=original_prompt,
            original_avg_score=baseline_candidate.avg_score,
            optimized_avg_score=baseline_candidate.avg_score,
        )

        current_prompt = original_prompt
        current_score = baseline_candidate.avg_score

        for iteration in range(self.config.max_iterations):
            candidates = self.generate_mutations(current_prompt, analysis)
            if not candidates:
                break

            # Evaluate all candidates
            for candidate in candidates:
                self.evaluate_candidate(candidate, test_tasks, baseline_scores)

            # Filter out regressions
            valid = [c for c in candidates if not c.regression_detected]
            if not valid:
                break

            # Select best
            best_candidate = max(valid, key=lambda c: c.avg_score)

            if best_candidate.avg_score > current_score + self.config.min_improvement:
                current_prompt = best_candidate.prompt
                current_score = best_candidate.avg_score
                best.optimized_prompt = current_prompt
                best.optimized_avg_score = current_score
                best.strategy_used = best_candidate.strategy

                # Update analysis for next iteration
                analysis = self.analyze_skill(
                    skill_name,
                    [best_candidate.per_dimension_scores],
                )
            else:
                break  # Converged

            best.iterations_run = iteration + 1

        best.improvement = round(best.optimized_avg_score - best.original_avg_score, 4)
        best.per_dimension_improvement = (
            {
                dim: round(
                    best_candidate.per_dimension_scores.get(dim, 0.0)
                    - baseline_scores.get(dim, 0.0),
                    4,
                )
                for dim in set(
                    list(baseline_scores.keys()) + list(best_candidate.per_dimension_scores.keys())
                )
            }
            if best.improvement > 0
            else {}
        )

        return best

    def compare(
        self,
        skill_name: str,
        original_prompt: str,
        optimized_prompt: str,
        test_tasks: list[str],
    ) -> ComparisonReport:
        """Side-by-side comparison of original vs optimized.

        Args:
            skill_name: Skill being compared.
            original_prompt: Original prompt.
            optimized_prompt: Optimized prompt.
            test_tasks: Tasks to evaluate on.

        Returns:
            ComparisonReport with per-dimension improvements.
        """
        orig = self.evaluate_candidate(
            PromptCandidate(prompt=original_prompt, strategy="original"),
            test_tasks,
        )
        opt = self.evaluate_candidate(
            PromptCandidate(prompt=optimized_prompt, strategy="optimized"),
            test_tasks,
        )

        improvements = {}
        regressions = []
        all_dims = set(
            list(orig.per_dimension_scores.keys()) + list(opt.per_dimension_scores.keys())
        )

        for dim in all_dims:
            orig_score = orig.per_dimension_scores.get(dim, 0.0)
            opt_score = opt.per_dimension_scores.get(dim, 0.0)
            diff = round(opt_score - orig_score, 4)
            improvements[dim] = diff
            if diff < -self.config.regression_threshold:
                regressions.append(dim)

        recommendation = "deploy"
        if regressions:
            recommendation = "reject"
        elif opt.avg_score <= orig.avg_score:
            recommendation = "no_improvement"

        return ComparisonReport(
            skill_name=skill_name,
            original_scores=orig.per_dimension_scores,
            optimized_scores=opt.per_dimension_scores,
            improvements=improvements,
            regressions=regressions,
            recommendation=recommendation,
        )

    def _load_score_history(self, skill_name: str) -> list[dict[str, float]]:
        """Load score history from storage."""
        if self.storage is None:
            return []
        results = self.storage.query()
        history: list[dict[str, float]] = []
        for r in results:
            if r.skill_used == skill_name:
                scores = {ds.dimension.value: ds.score for ds in r.dimension_scores}
                history.append(scores)
        return history

    def _score_prompt(self, prompt: str, task: str) -> dict[str, float]:
        """Score a prompt on a task using the rubric engine.

        Falls back to heuristic scoring if no engine available.
        """
        if self.rubric_engine is not None:
            full_prompt = f"{prompt}\n\nTask: {task}"
            result = self.rubric_engine.score(code=full_prompt, filename="skill.py")
            return {ds.dimension.value: ds.score for ds in result.dimension_scores}

        # Heuristic fallback
        rng = random.Random(hash((prompt, task)) % (2**31))
        base = 0.5
        # Longer prompts with keywords tend to score higher
        if "docstring" in prompt.lower() or "documentation" in prompt.lower():
            base += 0.05
        if "security" in prompt.lower() or "credential" in prompt.lower():
            base += 0.05
        if "test" in prompt.lower() or "testab" in prompt.lower():
            base += 0.05
        if "error" in prompt.lower() or "exception" in prompt.lower():
            base += 0.03
        if "example" in prompt.lower():
            base += 0.03

        return {dim.value: max(0.0, min(1.0, base + rng.uniform(-0.1, 0.1))) for dim in Dimension}
