"""Registry dashboard views for tool inventory, graduation, ownership, and risk.

Outputs Markdown tables for sharing in Slack/docs. Four views:
  1. inventory  — tool list grouped by tier
  2. pipeline   — graduation suggestions and active workflows
  3. ownership  — tech owner coverage and load
  4. heatmap    — tools × dimensions risk matrix
"""

from __future__ import annotations

from shared.models import ToolTier

from registry.catalog.catalog import ToolCatalog
from registry.graduation.triggers import GraduationSuggestion
from registry.workflows.engine import GraduationWorkflow


# --- Inventory View ---


def render_inventory(catalog: ToolCatalog) -> str:
    """Render tool inventory grouped by tier (T3 first).

    Highlights tools without owners at T2+ and declining scores.
    """
    tools = catalog.list()
    if not tools:
        return "No tools registered."

    lines = ["## Tool Inventory", ""]
    lines.append("| Tool | Tier | Owner | Users | Score | Trend |")
    lines.append("|------|------|-------|-------|-------|-------|")

    # Sort: T3 first, then T2, T1, T0
    tier_order = {ToolTier.T3: 0, ToolTier.T2: 1, ToolTier.T1: 2, ToolTier.T0: 3}
    tools_sorted = sorted(tools, key=lambda t: (tier_order.get(t.tier, 99), t.name.lower()))

    warnings: list[str] = []

    for tool in tools_sorted:
        owner = tool.tech_owner or "-"
        user_count = len(tool.users)
        score = f"{tool.scorecard.latest_composite:.2f}"
        trend = tool.scorecard.trend

        # Highlight warnings
        marker = ""
        if tool.tier in (ToolTier.T2, ToolTier.T3) and not tool.tech_owner:
            marker = " ⚠"
            warnings.append(f"- **{tool.name}** ({tool.tier.value}): no tech owner assigned")
        if tool.scorecard.trend == "declining":
            marker += " ↓"
            warnings.append(f"- **{tool.name}**: score is declining")

        lines.append(
            f"| {tool.name}{marker} | {tool.tier.value} | {owner} | {user_count} | {score} | {trend} |"
        )

    if warnings:
        lines.append("")
        lines.append("### Warnings")
        lines.append("")
        lines.extend(warnings)

    return "\n".join(lines)


# --- Pipeline View ---


def render_pipeline(
    suggestions: list[GraduationSuggestion],
    workflow: GraduationWorkflow,
) -> str:
    """Render graduation pipeline: suggestions and active workflows."""
    lines = ["## Graduation Pipeline", ""]

    # Pending suggestions
    if suggestions:
        lines.append("### Pending Suggestions")
        lines.append("")
        lines.append("| Tool | Current | Suggested | Trigger |")
        lines.append("|------|---------|-----------|---------|")
        for s in suggestions:
            lines.append(
                f"| {s.tool_slug} | {s.current_tier.value} | "
                f"{s.suggested_tier.value} | {s.trigger_reason} |"
            )
        lines.append("")

    # Active workflows
    pending = workflow.list_pending()
    if pending:
        lines.append("### Active Workflows")
        lines.append("")
        lines.append("| Tool | From | To | Nominated By | Ready | Readiness |")
        lines.append("|------|------|----|-------------|-------|-----------|")
        for w in pending:
            ready = "Yes" if w.ready else "No"
            readiness = f"{w.overall_readiness:.0%}"
            lines.append(
                f"| {w.tool_slug} | {w.from_tier.value} | {w.to_tier.value} | "
                f"{w.nominated_by} | {ready} | {readiness} |"
            )

            # Action items count
            unmet = sum(1 for a in w.action_items if not a.met)
            if unmet > 0:
                lines.append(f"|  | | | | {unmet} action item(s) remaining | |")
        lines.append("")

    if not suggestions and not pending:
        lines.append("No pending suggestions or active workflows.")

    return "\n".join(lines)


# --- Ownership View ---


def render_ownership(catalog: ToolCatalog) -> str:
    """Render ownership coverage for T2+ tools."""
    tools = catalog.list()
    lines = ["## Ownership Coverage", ""]

    # T2+ tools
    t2_plus = [t for t in tools if t.tier in (ToolTier.T2, ToolTier.T3)]
    if not t2_plus:
        lines.append("No T2+ tools registered.")
        return "\n".join(lines)

    owned = [t for t in t2_plus if t.tech_owner]
    unowned = [t for t in t2_plus if not t.tech_owner]

    lines.append(f"**Coverage:** {len(owned)}/{len(t2_plus)} T2+ tools have assigned owners")
    lines.append("")

    # Owner load
    owner_tools: dict[str, list[str]] = {}
    for t in t2_plus:
        if t.tech_owner:
            owner_tools.setdefault(t.tech_owner, []).append(t.name)

    if owner_tools:
        lines.append("### Owner Load")
        lines.append("")
        lines.append("| Owner | Tools | Count |")
        lines.append("|-------|-------|-------|")
        for owner, tool_names in sorted(owner_tools.items(), key=lambda x: -len(x[1])):
            lines.append(f"| {owner} | {', '.join(tool_names)} | {len(tool_names)} |")
        lines.append("")

    # Unowned tools
    if unowned:
        lines.append("### Unowned T2+ Tools")
        lines.append("")
        for t in unowned:
            lines.append(f"- **{t.name}** ({t.tier.value})")
        lines.append("")

    return "\n".join(lines)


# --- Heatmap View ---

_SCORE_THRESHOLDS = {"green": 0.7, "yellow": 0.5}


def _score_indicator(score: float) -> str:
    """Return a text indicator for a score value."""
    if score >= _SCORE_THRESHOLDS["green"]:
        return f"{score:.2f} [G]"
    elif score >= _SCORE_THRESHOLDS["yellow"]:
        return f"{score:.2f} [Y]"
    else:
        return f"{score:.2f} [R]"


def render_heatmap(catalog: ToolCatalog) -> str:
    """Render a risk heatmap: tools × dimensions."""
    tools = catalog.list()
    lines = ["## Risk Heatmap", ""]

    if not tools:
        lines.append("No tools registered.")
        return "\n".join(lines)

    # Collect all dimension names
    all_dims: set[str] = set()
    for t in tools:
        all_dims.update(t.scorecard.latest_scores.keys())
    dims = sorted(all_dims)

    if not dims:
        lines.append("No dimension scores available.")
        return "\n".join(lines)

    lines.append("Legend: [G] >= 0.70, [Y] >= 0.50, [R] < 0.50")
    lines.append("")

    # Header
    header = "| Tool | " + " | ".join(dims) + " |"
    separator = "|------|" + "|".join("------" for _ in dims) + "|"
    lines.append(header)
    lines.append(separator)

    # Rows
    tier_order = {ToolTier.T3: 0, ToolTier.T2: 1, ToolTier.T1: 2, ToolTier.T0: 3}
    for t in sorted(tools, key=lambda t: (tier_order.get(t.tier, 99), t.name.lower())):
        cells = []
        for d in dims:
            score = t.scorecard.latest_scores.get(d, 0.0)
            cells.append(_score_indicator(score))
        lines.append(f"| {t.name} | " + " | ".join(cells) + " |")

    # Systemic weaknesses
    lines.append("")
    dim_averages: dict[str, list[float]] = {d: [] for d in dims}
    for t in tools:
        for d in dims:
            dim_averages[d].append(t.scorecard.latest_scores.get(d, 0.0))

    weak_dims = []
    for d, scores in dim_averages.items():
        avg = sum(scores) / len(scores) if scores else 0
        if avg < _SCORE_THRESHOLDS["yellow"]:
            weak_dims.append((d, avg))

    if weak_dims:
        lines.append("### Systemic Weaknesses")
        lines.append("")
        for d, avg in sorted(weak_dims, key=lambda x: x[1]):
            lines.append(f"- **{d}**: average {avg:.2f} across all tools")

    return "\n".join(lines)
