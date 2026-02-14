"""Automatic graduation trigger engine.

Monitors scorecard data and tool metadata to suggest tier promotions
when predefined thresholds are crossed.

Trigger rules:
  T0→T1: second user, >500 LOC, 10+ scores, rising complexity trend
  T1→T2: 14+ daily scores by 3+ users, external API usage,
         PII handling, manual nomination
  T2→T3: manual nomination only (requires tech team approval)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shared.config import RegistryConfig, load_config
from shared.models import ScoreResult, ToolRegistryEntry, ToolTier
from shared.storage import StorageBackend

from registry.catalog.catalog import ToolCatalog
from registry.graduation.rubrics import GraduationResult, GraduationRubric


@dataclass
class GraduationSuggestion:
    """A suggestion to graduate a tool to a higher tier."""

    tool_slug: str
    current_tier: ToolTier
    suggested_tier: ToolTier
    trigger_reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    graduation_result: GraduationResult | None = None


class GraduationTriggerEngine:
    """Scans tools for graduation triggers and produces suggestions.

    Args:
        catalog: Tool catalog to read tool metadata.
        storage: Score storage backend for history queries.
        config: Registry configuration with trigger thresholds.
    """

    def __init__(
        self,
        catalog: ToolCatalog,
        storage: StorageBackend,
        config: RegistryConfig | None = None,
    ) -> None:
        self._catalog = catalog
        self._storage = storage
        if config is None:
            config = load_config().registry
        self._config = config
        self._triggers = config.auto_triggers
        self._rubric = GraduationRubric()

    def check_triggers(self) -> list[GraduationSuggestion]:
        """Scan all tools for graduation triggers.

        Returns:
            List of graduation suggestions for tools that crossed thresholds.
        """
        suggestions: list[GraduationSuggestion] = []
        tools = self._catalog.list()

        for tool in tools:
            suggestion = self.check_tool(tool.slug)
            if suggestion is not None:
                suggestions.append(suggestion)

        return suggestions

    def check_tool(self, slug: str) -> GraduationSuggestion | None:
        """Check a specific tool for graduation triggers.

        Returns:
            A GraduationSuggestion if a trigger fired, None otherwise.
        """
        tool = self._catalog.get(slug)
        if tool is None:
            return None

        if tool.tier == ToolTier.T0:
            return self._check_t0_triggers(tool)
        elif tool.tier == ToolTier.T1:
            return self._check_t1_triggers(tool)
        # T2→T3 is manual only
        return None

    def _check_t0_triggers(self, tool: ToolRegistryEntry) -> GraduationSuggestion | None:
        """Check T0→T1 triggers."""
        triggers = self._triggers.t0_to_t1

        # Trigger: second distinct user
        if triggers.second_user and len(tool.users) >= 2:
            return self._make_suggestion(
                tool,
                ToolTier.T1,
                "Second user detected",
                {"user_count": len(tool.users), "users": tool.users},
            )

        # Trigger: total scores >= 10 (active development)
        scores = self._get_tool_scores(tool)
        if len(scores) >= 10:
            return self._make_suggestion(
                tool,
                ToolTier.T1,
                "10+ scoring events (active development)",
                {"total_scores": len(scores)},
            )

        # Trigger: lines of code exceed threshold
        if triggers.max_lines > 0:
            loc = tool.metadata.get("lines_of_code", 0)
            if loc > triggers.max_lines:
                return self._make_suggestion(
                    tool,
                    ToolTier.T1,
                    f"Lines of code exceed {triggers.max_lines}",
                    {"lines_of_code": loc, "threshold": triggers.max_lines},
                )

        # Trigger: rising complexity trend (3 consecutive rising scores)
        if self._has_rising_trend(scores):
            return self._make_suggestion(
                tool,
                ToolTier.T1,
                "Rising complexity trend (3+ consecutive increases)",
                {"trend_length": 3},
            )

        return None

    def _check_t1_triggers(self, tool: ToolRegistryEntry) -> GraduationSuggestion | None:
        """Check T1→T2 triggers."""
        triggers = self._triggers.t1_to_t2

        # Trigger: daily usage by 3+ users for 14+ days
        scores = self._get_tool_scores(tool)
        distinct_users = {s.user for s in scores}
        distinct_days = {s.timestamp.date() for s in scores}

        if (
            len(distinct_users) >= triggers.min_users
            and len(distinct_days) >= triggers.daily_usage_days
        ):
            return self._make_suggestion(
                tool,
                ToolTier.T2,
                f"{len(distinct_users)} users over {len(distinct_days)} days",
                {
                    "distinct_users": len(distinct_users),
                    "distinct_days": len(distinct_days),
                    "min_users": triggers.min_users,
                    "min_days": triggers.daily_usage_days,
                },
            )

        # Trigger: handles PII (from metadata or security flags)
        if tool.metadata.get("handles_pii", False):
            return self._make_suggestion(
                tool,
                ToolTier.T2,
                "Tool handles PII data",
                {"pii_detected": True},
            )

        # Trigger: external API integration
        if tool.metadata.get("external_apis", False):
            return self._make_suggestion(
                tool,
                ToolTier.T2,
                "Tool integrates with external APIs",
                {"external_apis": True},
            )

        # Trigger: manual nomination
        if tool.metadata.get("nominated_for_t2", False):
            return self._make_suggestion(
                tool,
                ToolTier.T2,
                "Manual nomination for T2",
                {"nominated_by": tool.metadata.get("nominated_by", "unknown")},
            )

        return None

    def _get_tool_scores(self, tool: ToolRegistryEntry) -> list[ScoreResult]:
        """Get all scores for a tool from storage."""
        all_scores = self._storage.query()
        # Match scores by source path or files touched
        if not tool.source_path:
            return []
        return [
            s
            for s in all_scores
            if any(tool.source_path in f for f in s.files_touched)
            or tool.source_path in s.metadata.get("source_path", "")
        ]

    @staticmethod
    def _has_rising_trend(scores: list[ScoreResult], window: int = 3) -> bool:
        """Check if the last N scores show a rising trend."""
        if len(scores) < window:
            return False

        recent = sorted(scores, key=lambda s: s.timestamp)[-window:]
        for i in range(1, len(recent)):
            if recent[i].composite_score <= recent[i - 1].composite_score:
                return False
        return True

    def _make_suggestion(
        self,
        tool: ToolRegistryEntry,
        target_tier: ToolTier,
        reason: str,
        evidence: dict[str, Any],
    ) -> GraduationSuggestion:
        """Create a graduation suggestion with rubric evaluation."""
        grad_result = self._rubric.evaluate(tool, target_tier)
        return GraduationSuggestion(
            tool_slug=tool.slug,
            current_tier=tool.tier,
            suggested_tier=target_tier,
            trigger_reason=reason,
            evidence=evidence,
            graduation_result=grad_result,
        )
