"""Composite scorer and rubric engine.

Orchestrates all 5 dimension scorers, computes weighted composite score,
and returns a complete ScoreResult. Main entry point for downstream
consumers (hooks, CLI, dashboard).
"""

from __future__ import annotations

import getpass
from datetime import datetime
from pathlib import Path
from typing import Any

from scorecard.dimensions.correctness import CorrectnessScorer
from scorecard.dimensions.documentation import DocumentationScorer
from scorecard.dimensions.maintainability import MaintainabilityScorer
from scorecard.dimensions.security import SecurityScorer
from scorecard.dimensions.testability import TestabilityScorer
from shared.config import RubricGatesConfig, load_config
from shared.models import Dimension, DimensionScore, ScoreResult, ScoringMethod


class RubricEngine:
    """Orchestrates dimension scorers and computes composite scores.

    Loads scorers based on config (only enabled ones), runs them,
    and produces a weighted composite ScoreResult.
    """

    def __init__(self, config: RubricGatesConfig | None = None):
        if config is None:
            config = load_config()
        self.config = config
        self.scorecard_config = config.scorecard
        self._scorers = self._load_scorers()

    def _load_scorers(self) -> dict[Dimension, Any]:
        """Load enabled dimension scorers based on config."""
        scorers: dict[Dimension, Any] = {}
        dims = self.scorecard_config.dimensions

        if dims.get("correctness") and dims["correctness"].enabled:
            scorers[Dimension.CORRECTNESS] = CorrectnessScorer()

        if dims.get("security") and dims["security"].enabled:
            scorers[Dimension.SECURITY] = SecurityScorer()

        if dims.get("maintainability") and dims["maintainability"].enabled:
            scorers[Dimension.MAINTAINABILITY] = MaintainabilityScorer()

        if dims.get("documentation") and dims["documentation"].enabled:
            # Use LLM=False by default in engine — opt-in via config/context
            scorers[Dimension.DOCUMENTATION] = DocumentationScorer(use_llm=False)

        if dims.get("testability") and dims["testability"].enabled:
            scorers[Dimension.TESTABILITY] = TestabilityScorer(use_llm=False)

        return scorers

    def _get_weight(self, dimension: Dimension) -> float:
        """Get the configured weight for a dimension."""
        dim_config = self.scorecard_config.dimensions.get(dimension.value)
        if dim_config is None:
            return 0.0
        return dim_config.weight

    def score(
        self,
        code: str,
        filename: str = "",
        *,
        user: str = "",
        skill_used: str = "",
        project_files: list[str] | None = None,
        test_code: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ScoreResult:
        """Run all enabled dimension scorers and return composite result.

        Args:
            code: Source code to score.
            filename: Name of the file being scored.
            user: User who generated the code.
            skill_used: Claude Code skill used.
            project_files: List of project files (for testability check).
            test_code: Corresponding test code (for testability check).
            metadata: Additional metadata to attach to the result.
        """
        if not user:
            try:
                user = getpass.getuser()
            except Exception:
                user = "unknown"

        dimension_scores: list[DimensionScore] = []
        scorer_errors: list[str] = []

        for dimension, scorer in self._scorers.items():
            try:
                if dimension == Dimension.TESTABILITY:
                    dim_score = scorer.score(
                        code,
                        filename,
                        project_files=project_files,
                        test_code=test_code,
                    )
                else:
                    dim_score = scorer.score(code, filename)
                dimension_scores.append(dim_score)
            except Exception as e:
                scorer_errors.append(f"{dimension.value}: {e}")
                # Add a zero score for the failed dimension
                dimension_scores.append(
                    DimensionScore(
                        dimension=dimension,
                        score=0.0,
                        method=ScoringMethod.RULE_BASED,
                        details=f"Scorer error: {e}",
                    )
                )

        # Compute weighted composite score
        composite = self._compute_composite(dimension_scores)

        result_metadata = metadata.copy() if metadata else {}
        if scorer_errors:
            result_metadata["scorer_errors"] = scorer_errors

        return ScoreResult(
            timestamp=datetime.now(),
            user=user,
            skill_used=skill_used,
            files_touched=[filename] if filename else [],
            dimension_scores=dimension_scores,
            composite_score=composite,
            source_code=code if code else None,
            metadata=result_metadata,
        )

    def _compute_composite(self, scores: list[DimensionScore]) -> float:
        """Compute weighted composite score from dimension scores."""
        total_weight = 0.0
        weighted_sum = 0.0

        for ds in scores:
            weight = self._get_weight(ds.dimension)
            weighted_sum += ds.score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0
        return min(max(weighted_sum / total_weight, 0.0), 1.0)

    def score_file(
        self,
        file_path: str | Path,
        *,
        user: str = "",
        skill_used: str = "",
        project_files: list[str] | None = None,
    ) -> ScoreResult:
        """Score a file from disk.

        Args:
            file_path: Path to the Python file to score.
            user: User who generated the code.
            skill_used: Claude Code skill used.
            project_files: List of project files for testability.
        """
        path = Path(file_path)
        code = path.read_text()
        return self.score(
            code,
            filename=path.name,
            user=user,
            skill_used=skill_used,
            project_files=project_files,
        )

    def score_files(
        self,
        file_paths: list[str | Path],
        *,
        user: str = "",
        skill_used: str = "",
    ) -> list[ScoreResult]:
        """Score multiple files.

        Args:
            file_paths: Paths to Python files.
            user: User who generated the code.
            skill_used: Claude Code skill used.
        """
        str_paths = [str(p) for p in file_paths]
        results = []
        for fp in file_paths:
            results.append(
                self.score_file(
                    fp,
                    user=user,
                    skill_used=skill_used,
                    project_files=str_paths,
                )
            )
        return results
