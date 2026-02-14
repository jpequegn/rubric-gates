"""Tool catalog — persistent YAML-based registry for T1+ tools.

Provides CRUD operations, tier promotion with graduation history,
scorecard updates, and search/filter capabilities.

Storage: each tool gets a YAML file in the catalog data directory,
keyed by slug (e.g. ``data/expense-categorizer.yaml``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from shared.models import (
    GraduationEvent,
    ScorecardSummary,
    ScoreResult,
    ToolRegistryEntry,
    ToolTier,
)


def _slugify(name: str) -> str:
    """Convert a tool name to a URL-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug


class ToolCatalog:
    """YAML-backed tool catalog with CRUD, promotion, and search.

    Args:
        data_dir: Directory to store tool YAML files.
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path("./registry/catalog/data")
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _tool_path(self, slug: str) -> Path:
        """Get the YAML file path for a tool slug."""
        return self._data_dir / f"{slug}.yaml"

    def _save(self, tool: ToolRegistryEntry) -> None:
        """Save a tool entry to YAML."""
        data = tool.model_dump(mode="json")
        path = self._tool_path(tool.slug)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _load(self, slug: str) -> ToolRegistryEntry | None:
        """Load a tool entry from YAML."""
        path = self._tool_path(slug)
        if not path.exists():
            return None
        with open(path) as f:
            data = yaml.safe_load(f)
        if not data:
            return None
        return ToolRegistryEntry.model_validate(data)

    def register(self, tool: ToolRegistryEntry) -> str:
        """Register a new tool. Returns the slug.

        If no slug is set, one is generated from the name.

        Raises:
            ValueError: If a tool with the same slug already exists.
        """
        if not tool.slug:
            tool = tool.model_copy(update={"slug": _slugify(tool.name)})

        if self._tool_path(tool.slug).exists():
            raise ValueError(f"Tool '{tool.slug}' already exists.")

        self._save(tool)
        return tool.slug

    def get(self, slug: str) -> ToolRegistryEntry | None:
        """Get a tool by slug. Returns None if not found."""
        return self._load(slug)

    def list(
        self,
        *,
        tier: ToolTier | None = None,
        owner: str | None = None,
        tag: str | None = None,
    ) -> list[ToolRegistryEntry]:
        """List tools with optional filters.

        Args:
            tier: Filter by tier.
            owner: Filter by tech owner.
            tag: Filter by tag.

        Returns:
            List of matching tools, sorted by name.
        """
        tools: list[ToolRegistryEntry] = []

        for path in sorted(self._data_dir.glob("*.yaml")):
            with open(path) as f:
                data = yaml.safe_load(f)
            if not data:
                continue

            tool = ToolRegistryEntry.model_validate(data)

            if tier is not None and tool.tier != tier:
                continue
            if owner is not None and tool.tech_owner != owner:
                continue
            if tag is not None and tag not in tool.tags:
                continue

            tools.append(tool)

        return sorted(tools, key=lambda t: t.name.lower())

    def update(self, slug: str, updates: dict[str, Any]) -> ToolRegistryEntry:
        """Update tool metadata.

        Args:
            slug: Tool slug.
            updates: Dictionary of fields to update.

        Returns:
            Updated tool entry.

        Raises:
            KeyError: If tool not found.
        """
        tool = self._load(slug)
        if tool is None:
            raise KeyError(f"Tool '{slug}' not found.")

        tool = tool.model_copy(update=updates)
        self._save(tool)
        return tool

    def delete(self, slug: str) -> bool:
        """Delete a tool from the catalog.

        Returns True if deleted, False if not found.
        """
        path = self._tool_path(slug)
        if not path.exists():
            return False
        path.unlink()
        return True

    def promote(
        self,
        slug: str,
        target_tier: ToolTier,
        reason: str = "",
        approved_by: str = "",
    ) -> ToolRegistryEntry:
        """Promote a tool to a higher tier. Records graduation history.

        Args:
            slug: Tool slug.
            target_tier: The tier to promote to.
            reason: Reason for promotion.
            approved_by: Who approved the promotion.

        Returns:
            Updated tool entry.

        Raises:
            KeyError: If tool not found.
            ValueError: If transition is invalid.
        """
        tool = self._load(slug)
        if tool is None:
            raise KeyError(f"Tool '{slug}' not found.")

        from registry.graduation.tiers import is_valid_transition

        if not is_valid_transition(tool.tier, target_tier):
            raise ValueError(
                f"Cannot promote from {tool.tier.value} to {target_tier.value}. "
                "Must be one step up."
            )

        event = GraduationEvent(
            from_tier=tool.tier,
            to_tier=target_tier,
            reason=reason,
            approved_by=approved_by,
        )

        new_history = list(tool.graduation_history) + [event]
        tool = tool.model_copy(
            update={
                "tier": target_tier,
                "graduation_history": new_history,
            }
        )
        self._save(tool)
        return tool

    def update_scorecard(self, slug: str, score_result: ScoreResult) -> None:
        """Update a tool's scorecard summary with the latest score.

        Args:
            slug: Tool slug.
            score_result: The latest score result.

        Raises:
            KeyError: If tool not found.
        """
        tool = self._load(slug)
        if tool is None:
            raise KeyError(f"Tool '{slug}' not found.")

        # Build latest_scores from dimension scores
        latest_scores = {ds.dimension.value: ds.score for ds in score_result.dimension_scores}

        # Determine trend
        prev = tool.scorecard.latest_composite
        new = score_result.composite_score
        if new > prev + 0.05:
            trend = "improving"
        elif new < prev - 0.05:
            trend = "declining"
        else:
            trend = "stable"

        new_scorecard = ScorecardSummary(
            latest_composite=new,
            latest_scores=latest_scores,
            trend=trend,
            total_scores=tool.scorecard.total_scores + 1,
            red_flags=tool.scorecard.red_flags,
        )

        tool = tool.model_copy(update={"scorecard": new_scorecard})
        self._save(tool)

    def search(self, query: str) -> list[ToolRegistryEntry]:
        """Search tools by name or description.

        Args:
            query: Search string (case-insensitive substring match).

        Returns:
            List of matching tools.
        """
        query_lower = query.lower()
        results: list[ToolRegistryEntry] = []

        for path in sorted(self._data_dir.glob("*.yaml")):
            with open(path) as f:
                data = yaml.safe_load(f)
            if not data:
                continue

            tool = ToolRegistryEntry.model_validate(data)
            if query_lower in tool.name.lower() or query_lower in tool.description.lower():
                results.append(tool)

        return sorted(results, key=lambda t: t.name.lower())
