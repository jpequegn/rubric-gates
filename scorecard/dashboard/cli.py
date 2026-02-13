"""CLI monitoring dashboard for rubric-gates scores.

Read-only dashboard that surfaces quality trends across AI-generated code.
Outputs Markdown tables for easy sharing in Slack/docs.

Usage:
    python -m scorecard.dashboard overview
    python -m scorecard.dashboard users [--since 7d]
    python -m scorecard.dashboard skills [--since 7d]
    python -m scorecard.dashboard red-flags [--since 7d] [--user alice]
    python -m scorecard.dashboard dimensions [--since 7d]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta

from shared.models import ScoreResult
from shared.storage import QueryFilters, create_storage


# --- Time Parsing ---


def parse_since(since: str) -> datetime:
    """Parse a relative time string like '7d', '30d', '24h' into a datetime."""
    since = since.strip().lower()
    now = datetime.now()
    if since.endswith("d"):
        days = int(since[:-1])
        return now - timedelta(days=days)
    elif since.endswith("h"):
        hours = int(since[:-1])
        return now - timedelta(hours=hours)
    elif since.endswith("w"):
        weeks = int(since[:-1])
        return now - timedelta(weeks=weeks)
    else:
        # Try parsing as ISO date
        return datetime.fromisoformat(since)


def _build_filters(args: argparse.Namespace) -> QueryFilters:
    """Build QueryFilters from CLI args."""
    filters = QueryFilters()
    if hasattr(args, "since") and args.since:
        filters.start_date = parse_since(args.since)
    if hasattr(args, "user") and args.user:
        filters.user = args.user
    if hasattr(args, "skill") and args.skill:
        filters.skill = args.skill
    return filters


# --- Formatters ---


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format data as a Markdown table."""
    if not rows:
        return "(no data)\n"

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    # Build table
    lines = []
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    separator = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    lines.append(header_line)
    lines.append(separator)
    for row in rows:
        padded = [cell.ljust(widths[i]) if i < len(widths) else cell for i, cell in enumerate(row)]
        lines.append("| " + " | ".join(padded) + " |")

    return "\n".join(lines) + "\n"


def _dim_scores_map(result: ScoreResult) -> dict[str, float]:
    """Extract dimension scores as a dict."""
    return {ds.dimension.value: ds.score for ds in result.dimension_scores}


# --- Views ---


def view_overview(results: list[ScoreResult], since_label: str = "all time") -> str:
    """Overview: total files, avg score, red flags."""
    lines = [f"## Overview ({since_label})\n"]

    if not results:
        lines.append("No scores recorded.\n")
        return "\n".join(lines)

    total = len(results)
    avg = sum(r.composite_score for r in results) / total
    red_flags = sum(1 for r in results if r.composite_score < 0.5)
    above_70 = sum(1 for r in results if r.composite_score >= 0.7)

    lines.append(f"- **Files scored:** {total}")
    lines.append(f"- **Average composite score:** {avg:.2f}")
    lines.append(f"- **Scores >= 0.7 (green):** {above_70} ({above_70 * 100 // total}%)")
    lines.append(f"- **Scores < 0.5 (red flags):** {red_flags} ({red_flags * 100 // total}%)")
    lines.append("")

    return "\n".join(lines)


def view_by_user(results: list[ScoreResult]) -> str:
    """By-user breakdown table."""
    lines = ["## By User\n"]

    if not results:
        lines.append("No scores recorded.\n")
        return "\n".join(lines)

    # Group by user
    users: dict[str, list[float]] = {}
    for r in results:
        users.setdefault(r.user, []).append(r.composite_score)

    headers = ["User", "Files", "Avg Score", "Min Score", "Red Flags"]
    rows = []
    for user, scores in sorted(users.items()):
        red = sum(1 for s in scores if s < 0.5)
        rows.append(
            [
                user,
                str(len(scores)),
                f"{sum(scores) / len(scores):.2f}",
                f"{min(scores):.2f}",
                str(red),
            ]
        )

    # Sort by red flags descending
    rows.sort(key=lambda r: int(r[4]), reverse=True)
    lines.append(_format_table(headers, rows))
    return "\n".join(lines)


def view_by_skill(results: list[ScoreResult]) -> str:
    """By-skill breakdown table."""
    lines = ["## By Skill\n"]

    if not results:
        lines.append("No scores recorded.\n")
        return "\n".join(lines)

    # Group by skill
    skills: dict[str, list[ScoreResult]] = {}
    for r in results:
        skill = r.skill_used or "(none)"
        skills.setdefault(skill, []).append(r)

    headers = ["Skill", "Uses", "Avg Score", "Worst Dimension"]
    rows = []
    for skill, skill_results in sorted(skills.items()):
        avg = sum(r.composite_score for r in skill_results) / len(skill_results)

        # Find worst average dimension
        dim_totals: dict[str, list[float]] = {}
        for r in skill_results:
            for ds in r.dimension_scores:
                dim_totals.setdefault(ds.dimension.value, []).append(ds.score)

        worst_dim = ""
        worst_avg = 1.0
        for dim, scores in dim_totals.items():
            dim_avg = sum(scores) / len(scores)
            if dim_avg < worst_avg:
                worst_avg = dim_avg
                worst_dim = dim

        rows.append(
            [
                skill,
                str(len(skill_results)),
                f"{avg:.2f}",
                f"{worst_dim} ({worst_avg:.2f})" if worst_dim else "n/a",
            ]
        )

    lines.append(_format_table(headers, rows))
    return "\n".join(lines)


def view_red_flags(results: list[ScoreResult], threshold: float = 0.5) -> str:
    """Red flags: scores below threshold."""
    lines = ["## Red Flags\n"]

    flagged = [r for r in results if r.composite_score < threshold]

    if not flagged:
        lines.append(f"No scores below {threshold:.1f}. All clear!\n")
        return "\n".join(lines)

    lines.append(f"**{len(flagged)} scores below {threshold:.1f}:**\n")

    headers = ["File", "User", "Skill", "Score", "Worst Dim", "Timestamp"]
    rows = []
    for r in sorted(flagged, key=lambda x: x.composite_score):
        files = ", ".join(r.files_touched[:2]) or "(none)"
        dim_map = _dim_scores_map(r)
        worst = min(dim_map.items(), key=lambda x: x[1]) if dim_map else ("n/a", 0.0)

        rows.append(
            [
                files,
                r.user,
                r.skill_used or "(none)",
                f"{r.composite_score:.2f}",
                f"{worst[0]} ({worst[1]:.2f})",
                r.timestamp.strftime("%Y-%m-%d %H:%M"),
            ]
        )

    lines.append(_format_table(headers, rows))
    return "\n".join(lines)


def view_dimensions(results: list[ScoreResult]) -> str:
    """Dimension breakdown: which dimensions are consistently low?"""
    lines = ["## Dimension Breakdown\n"]

    if not results:
        lines.append("No scores recorded.\n")
        return "\n".join(lines)

    # Aggregate by dimension
    dim_scores: dict[str, list[float]] = {}
    for r in results:
        for ds in r.dimension_scores:
            dim_scores.setdefault(ds.dimension.value, []).append(ds.score)

    headers = ["Dimension", "Avg Score", "Min Score", "< 0.5 Count", "Assessment"]
    rows = []
    for dim in sorted(dim_scores.keys()):
        scores = dim_scores[dim]
        avg = sum(scores) / len(scores)
        low_count = sum(1 for s in scores if s < 0.5)

        if avg >= 0.8:
            assessment = "Strong"
        elif avg >= 0.6:
            assessment = "Adequate"
        elif avg >= 0.4:
            assessment = "Needs work"
        else:
            assessment = "Critical"

        rows.append(
            [
                dim,
                f"{avg:.2f}",
                f"{min(scores):.2f}",
                str(low_count),
                assessment,
            ]
        )

    # Sort by avg score ascending (worst first)
    rows.sort(key=lambda r: float(r[1]))
    lines.append(_format_table(headers, rows))
    return "\n".join(lines)


# --- CLI ---


def main() -> None:
    """CLI entry point for the monitoring dashboard."""
    parser = argparse.ArgumentParser(
        description="Rubric-gates monitoring dashboard",
    )
    subparsers = parser.add_subparsers(dest="command", help="Dashboard view")

    # Shared args
    def add_common_args(sp):
        sp.add_argument("--since", default="", help="Time range: 7d, 30d, 24h, 1w")
        sp.add_argument("--user", default="", help="Filter by user")
        sp.add_argument("--skill", default="", help="Filter by skill")

    # Overview
    sp_overview = subparsers.add_parser("overview", help="Score overview")
    add_common_args(sp_overview)

    # By user
    sp_users = subparsers.add_parser("users", help="Scores by user")
    add_common_args(sp_users)

    # By skill
    sp_skills = subparsers.add_parser("skills", help="Scores by skill")
    add_common_args(sp_skills)

    # Red flags
    sp_flags = subparsers.add_parser("red-flags", help="Low scores")
    add_common_args(sp_flags)
    sp_flags.add_argument(
        "--threshold", type=float, default=0.5, help="Red flag threshold (default: 0.5)"
    )

    # Dimensions
    sp_dims = subparsers.add_parser("dimensions", help="Dimension breakdown")
    add_common_args(sp_dims)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load data
    storage = create_storage()
    filters = _build_filters(args)
    results = storage.query(filters)

    since_label = f"since {args.since}" if args.since else "all time"

    # Render view
    if args.command == "overview":
        print(view_overview(results, since_label))
    elif args.command == "users":
        print(view_by_user(results))
    elif args.command == "skills":
        print(view_by_skill(results))
    elif args.command == "red-flags":
        threshold = args.threshold
        print(view_red_flags(results, threshold))
    elif args.command == "dimensions":
        print(view_dimensions(results))


if __name__ == "__main__":
    main()
