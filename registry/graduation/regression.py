"""Regression detector for tool quality regressions.

Monitors post-graduation score trends and generates demotion suggestions
when sustained quality drops are detected. Human approval is always
required before demotion — this detector only *suggests*.

Detection criteria:
  - Average composite score below tier minimum for N consecutive scores
  - Any dimension below red threshold for N consecutive scores
  - Red gate tier count exceeds threshold in a rolling window

A configurable grace period after promotion prevents false positives.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from shared.config import RegressionConfig
from shared.models import (
    DemotionSuggestion,
    ScoreResult,
    ToolRegistryEntry,
    ToolTier,
)

from registry.graduation.tiers import demotion_target

# Minimum composite score expected per tier
_TIER_MIN_COMPOSITE: dict[ToolTier, float] = {
    ToolTier.T0: 0.0,
    ToolTier.T1: 0.5,
    ToolTier.T2: 0.7,
    ToolTier.T3: 0.8,
}

# Red threshold for security dimension
_DEFAULT_RED_THRESHOLD = 0.3


class RegressionDetector:
    """Detects sustained quality regressions for graduated tools.

    Args:
        config: Regression detection configuration.
    """

    def __init__(self, config: RegressionConfig | None = None) -> None:
        if config is None:
            config = RegressionConfig()
        self._config = config

    def check(
        self,
        tool: ToolRegistryEntry,
        scores: list[ScoreResult],
        now: datetime | None = None,
    ) -> DemotionSuggestion | None:
        """Check a tool for quality regression.

        Args:
            tool: The tool registry entry.
            scores: Recent score results for this tool, ordered by timestamp.
            now: Current time (for grace period calculation). Defaults to now.

        Returns:
            A DemotionSuggestion if regression detected, None otherwise.
        """
        if now is None:
            now = datetime.now()

        # T0 tools cannot be demoted further
        target = demotion_target(tool.tier)
        if target is None:
            return None

        # Check grace period after most recent promotion
        if self._in_grace_period(tool, now):
            return None

        # Sort by timestamp
        sorted_scores = sorted(scores, key=lambda s: s.timestamp)

        # Check 1: Consecutive low composite scores
        suggestion = self._check_low_composite(tool, sorted_scores, target)
        if suggestion is not None:
            return suggestion

        # Check 2: Consecutive low dimension scores (any dimension below red)
        suggestion = self._check_low_dimension(tool, sorted_scores, target)
        if suggestion is not None:
            return suggestion

        # Check 3: Red tier count in rolling window
        suggestion = self._check_red_tier_count(tool, sorted_scores, target)
        if suggestion is not None:
            return suggestion

        return None

    def _in_grace_period(self, tool: ToolRegistryEntry, now: datetime) -> bool:
        """Check if the tool is within the post-promotion grace period."""
        if not tool.graduation_history:
            return False

        last_graduation = tool.graduation_history[-1]
        grace_end = last_graduation.date + timedelta(days=self._config.grace_period_days)
        return now < grace_end

    def _check_low_composite(
        self,
        tool: ToolRegistryEntry,
        scores: list[ScoreResult],
        target: ToolTier,
    ) -> DemotionSuggestion | None:
        """Check for N consecutive scores below the tier minimum."""
        n = self._config.consecutive_scores
        if len(scores) < n:
            return None

        tier_min = _TIER_MIN_COMPOSITE.get(tool.tier, 0.0)
        recent = scores[-n:]
        avg = sum(s.composite_score for s in recent) / n

        if all(s.composite_score < tier_min for s in recent):
            return DemotionSuggestion(
                tool_slug=tool.slug,
                current_tier=tool.tier,
                suggested_tier=target,
                reason=(
                    f"Composite score below {tier_min:.2f} for "
                    f"{n} consecutive scores (avg: {avg:.2f})"
                ),
                evidence={
                    "consecutive_scores": n,
                    "tier_minimum": tier_min,
                    "average_composite": round(avg, 4),
                    "scores": [round(s.composite_score, 4) for s in recent],
                },
            )
        return None

    def _check_low_dimension(
        self,
        tool: ToolRegistryEntry,
        scores: list[ScoreResult],
        target: ToolTier,
    ) -> DemotionSuggestion | None:
        """Check for N consecutive scores with any dimension below red threshold."""
        n = self._config.consecutive_scores
        if len(scores) < n:
            return None

        recent = scores[-n:]

        # Collect dimensions that are consistently below red threshold
        failing_dims: dict[str, list[float]] = {}
        for score in recent:
            for ds in score.dimension_scores:
                if ds.score < _DEFAULT_RED_THRESHOLD:
                    failing_dims.setdefault(ds.dimension.value, []).append(ds.score)

        # A dimension fails if it's below red for all N recent scores
        for dim_name, dim_scores in failing_dims.items():
            if len(dim_scores) >= n:
                avg = sum(dim_scores) / len(dim_scores)
                return DemotionSuggestion(
                    tool_slug=tool.slug,
                    current_tier=tool.tier,
                    suggested_tier=target,
                    reason=(
                        f"Dimension '{dim_name}' below red threshold "
                        f"{_DEFAULT_RED_THRESHOLD:.2f} for {n} consecutive scores "
                        f"(avg: {avg:.2f})"
                    ),
                    evidence={
                        "dimension": dim_name,
                        "red_threshold": _DEFAULT_RED_THRESHOLD,
                        "consecutive_scores": n,
                        "average_dimension_score": round(avg, 4),
                        "scores": [round(s, 4) for s in dim_scores[:n]],
                    },
                )
        return None

    def _check_red_tier_count(
        self,
        tool: ToolRegistryEntry,
        scores: list[ScoreResult],
        target: ToolTier,
    ) -> DemotionSuggestion | None:
        """Check if red gate tier count exceeds threshold in rolling window."""
        window = self._config.rolling_window
        threshold = self._config.red_tier_threshold

        if len(scores) < window:
            return None

        recent = scores[-window:]

        # Count scores that would be classified as red (composite < 0.3 or security < 0.3)
        red_count = 0
        for score in recent:
            if score.composite_score < _DEFAULT_RED_THRESHOLD:
                red_count += 1
                continue
            for ds in score.dimension_scores:
                if ds.dimension.value == "security" and ds.score < _DEFAULT_RED_THRESHOLD:
                    red_count += 1
                    break

        if red_count >= threshold:
            return DemotionSuggestion(
                tool_slug=tool.slug,
                current_tier=tool.tier,
                suggested_tier=target,
                reason=(
                    f"{red_count} red-tier scores in last {window} evaluations "
                    f"(threshold: {threshold})"
                ),
                evidence={
                    "red_count": red_count,
                    "rolling_window": window,
                    "threshold": threshold,
                },
            )
        return None
