"""Batch skill optimization runner.

Identifies skills needing optimization, runs optimization in batch,
and manages deployment with backup and rollback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from training.skill_optimizer import (
    ComparisonReport,
    OptimizedSkill,
    SkillOptimizer,
)


@dataclass
class SkillCandidate:
    """A skill identified as a candidate for optimization."""

    name: str = ""
    avg_score: float = 0.0
    total_scores: int = 0
    weak_dimensions: list[str] = field(default_factory=list)


@dataclass
class BatchReport:
    """Report from a batch optimization run."""

    total_skills: int = 0
    optimized: int = 0
    no_improvement: int = 0
    rejected: int = 0
    results: list[OptimizedSkill] = field(default_factory=list)
    comparisons: list[ComparisonReport] = field(default_factory=list)


class SkillOptimizationRunner:
    """Batch optimizer for the skill ecosystem."""

    def __init__(
        self,
        optimizer: SkillOptimizer,
        skills_dir: str | Path | None = None,
        backup_dir: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self._skills_dir = Path(skills_dir) if skills_dir else None
        self._backup_dir = Path(backup_dir) if backup_dir else None
        self._deployed: dict[str, OptimizedSkill] = {}

    def identify_candidates(
        self,
        skills: dict[str, str] | None = None,
        score_histories: dict[str, list[dict[str, float]]] | None = None,
        min_score_history: int = 5,
        max_avg_score: float = 0.7,
    ) -> list[SkillCandidate]:
        """Find skills with enough data and low scores.

        Args:
            skills: Dict mapping skill name to prompt text.
            score_histories: Dict mapping skill name to score history.
            min_score_history: Minimum number of scores required.
            max_avg_score: Only include skills below this average.

        Returns:
            Sorted list of candidates (worst scores first).
        """
        score_histories = score_histories or {}
        candidates: list[SkillCandidate] = []

        for name, history in score_histories.items():
            if len(history) < min_score_history:
                continue

            analysis = self.optimizer.analyze_skill(name, history)

            if analysis.avg_composite <= max_avg_score:
                candidates.append(
                    SkillCandidate(
                        name=name,
                        avg_score=analysis.avg_composite,
                        total_scores=analysis.total_scores,
                        weak_dimensions=analysis.weak_dimensions,
                    )
                )

        return sorted(candidates, key=lambda c: c.avg_score)

    def run_batch(
        self,
        skills: dict[str, str],
        test_tasks: list[str],
        score_histories: dict[str, list[dict[str, float]]] | None = None,
    ) -> BatchReport:
        """Optimize multiple skills and generate improvement report.

        Args:
            skills: Dict mapping skill name to prompt text.
            test_tasks: Tasks to evaluate candidates on.
            score_histories: Optional score histories per skill.

        Returns:
            BatchReport with results for all skills.
        """
        report = BatchReport(total_skills=len(skills))
        score_histories = score_histories or {}

        for name, prompt in skills.items():
            history = score_histories.get(name)
            result = self.optimizer.optimize(
                skill_name=name,
                original_prompt=prompt,
                test_tasks=test_tasks,
                score_history=history,
            )
            report.results.append(result)

            if result.improvement > 0:
                comparison = self.optimizer.compare(
                    name, prompt, result.optimized_prompt, test_tasks
                )
                report.comparisons.append(comparison)

                if comparison.recommendation == "deploy":
                    report.optimized += 1
                else:
                    report.rejected += 1
            else:
                report.no_improvement += 1

        return report

    def deploy(
        self,
        skill_name: str,
        optimized: OptimizedSkill,
        backup: bool = True,
    ) -> Path | None:
        """Deploy optimized prompt (with backup).

        Args:
            skill_name: Name of the skill.
            optimized: Optimization result to deploy.
            backup: Whether to save a backup of the original.

        Returns:
            Path to backup file, or None if no backup.
        """
        backup_path = None

        if backup and self._backup_dir:
            self._backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = self._backup_dir / f"{skill_name}.backup.json"
            backup_path.write_text(
                json.dumps(
                    {
                        "skill_name": skill_name,
                        "original_prompt": optimized.original_prompt,
                        "original_avg_score": optimized.original_avg_score,
                    },
                    indent=2,
                )
            )

        if self._skills_dir:
            skill_path = self._skills_dir / f"{skill_name}.txt"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text(optimized.optimized_prompt)

        self._deployed[skill_name] = optimized
        return backup_path

    def rollback(self, skill_name: str) -> bool:
        """Rollback a deployed optimization.

        Args:
            skill_name: Name of the skill to rollback.

        Returns:
            True if rollback succeeded, False if no backup found.
        """
        if self._backup_dir:
            backup_path = self._backup_dir / f"{skill_name}.backup.json"
            if backup_path.exists():
                data = json.loads(backup_path.read_text())
                original_prompt = data.get("original_prompt", "")

                if self._skills_dir:
                    skill_path = self._skills_dir / f"{skill_name}.txt"
                    skill_path.write_text(original_prompt)

                self._deployed.pop(skill_name, None)
                return True

        return False

    def list_deployed(self) -> dict[str, float]:
        """List all deployed optimizations and their improvements.

        Returns:
            Dict mapping skill name to improvement delta.
        """
        return {name: opt.improvement for name, opt in self._deployed.items()}

    def generate_report(self, batch_report: BatchReport) -> str:
        """Generate a human-readable markdown report.

        Args:
            batch_report: The batch report to format.

        Returns:
            Markdown-formatted report string.
        """
        lines = ["## Skill Optimization Report", ""]
        lines.append(f"**Skills analyzed:** {batch_report.total_skills}")
        lines.append(f"**Optimized:** {batch_report.optimized}")
        lines.append(f"**No improvement:** {batch_report.no_improvement}")
        lines.append(f"**Rejected (regressions):** {batch_report.rejected}")
        lines.append("")

        if batch_report.results:
            lines.append("### Results")
            lines.append("")
            lines.append("| Skill | Original | Optimized | Improvement | Strategy |")
            lines.append("|-------|----------|-----------|-------------|----------|")

            for result in batch_report.results:
                imp = f"+{result.improvement:.4f}" if result.improvement > 0 else "—"
                strategy = result.strategy_used or "—"
                lines.append(
                    f"| {result.skill_name} | {result.original_avg_score:.4f} | "
                    f"{result.optimized_avg_score:.4f} | {imp} | {strategy} |"
                )
            lines.append("")

        if batch_report.comparisons:
            lines.append("### Dimension Comparisons")
            lines.append("")
            for comp in batch_report.comparisons:
                lines.append(f"**{comp.skill_name}** — {comp.recommendation}")
                if comp.regressions:
                    lines.append(f"  Regressions: {', '.join(comp.regressions)}")
                for dim, imp in sorted(comp.improvements.items()):
                    sign = "+" if imp > 0 else ""
                    lines.append(f"  - {dim}: {sign}{imp:.4f}")
                lines.append("")

        return "\n".join(lines)
