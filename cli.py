"""Unified CLI for rubric-gates.

Chains scorecard -> gate -> registry in a single pipeline.

Usage:
    python -m cli score <file.py> [--user USER] [--skill SKILL] [--store]
    python -m cli check <file.py> [--user USER]
    python -m cli status <tool-slug>
    python -m cli report [--since 7d] [--user USER]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gate.tiers.evaluator import TierEvaluator
from scorecard.engine import RubricEngine
from shared.config import load_config
from shared.models import GateTier, ScoreResult
from shared.storage import QueryFilters, create_storage


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="rubric-gates",
        description="Quality assurance pipeline for AI-generated code",
    )
    parser.add_argument("--config", default=None, help="Path to config file")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- score ---
    sp_score = subparsers.add_parser(
        "score", help="Score a file through the full pipeline (scorecard + gate)"
    )
    sp_score.add_argument("file", help="Python file to score")
    sp_score.add_argument("--user", default="", help="User who generated the code")
    sp_score.add_argument("--skill", default="", help="Skill used to generate the code")
    sp_score.add_argument("--store", action="store_true", help="Store result in JSONL backend")
    sp_score.add_argument("--tool", default="", help="Tool slug to update in registry")

    # --- check ---
    sp_check = subparsers.add_parser("check", help="Quick gate check (score + tier, no storage)")
    sp_check.add_argument("file", help="Python file to check")
    sp_check.add_argument("--user", default="", help="User who generated the code")

    # --- status ---
    sp_status = subparsers.add_parser("status", help="Show tool registry status")
    sp_status.add_argument("slug", help="Tool slug to look up")

    # --- report ---
    sp_report = subparsers.add_parser("report", help="Generate summary report from stored scores")
    sp_report.add_argument("--since", default="", help="Time range: 7d, 30d, 24h, 1w")
    sp_report.add_argument("--user", default="", help="Filter by user")
    sp_report.add_argument("--skill", default="", help="Filter by skill")

    return parser


def _read_file(file_path: str) -> tuple[str, str]:
    """Read a file and return (code, filename). Exits on error."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    return path.read_text(), path.name


def _format_tier(tier: GateTier) -> str:
    """Format a gate tier for display."""
    labels = {
        GateTier.GREEN: "GREEN  - Good to merge",
        GateTier.YELLOW: "YELLOW - Review recommended",
        GateTier.RED: "RED    - Blocked",
    }
    return labels.get(tier, tier.value)


def _print_score_result(result: ScoreResult) -> str:
    """Format a ScoreResult for display. Returns the formatted string."""
    lines = []
    files = ", ".join(result.files_touched) or "(none)"
    lines.append(f"File: {files}")
    lines.append(f"Composite Score: {result.composite_score:.2f}")

    for ds in result.dimension_scores:
        bar = "█" * int(ds.score * 20) + "░" * (20 - int(ds.score * 20))
        lines.append(f"  {ds.dimension.value:<18} {bar} {ds.score:.2f}")

    return "\n".join(lines)


def _print_gate_result(gate_result) -> str:
    """Format a GateResult for display. Returns the formatted string."""
    lines = []
    lines.append(f"Gate Tier: {_format_tier(gate_result.tier)}")

    if gate_result.blocked:
        lines.append("Status: BLOCKED")

    if gate_result.critical_patterns_found:
        lines.append(f"Critical Patterns: {', '.join(gate_result.critical_patterns_found)}")

    if gate_result.advisory_messages:
        lines.append("")
        lines.append("Advisories:")
        for msg in gate_result.advisory_messages:
            lines.append(f"  - {msg}")

    return "\n".join(lines)


def cmd_score(args: argparse.Namespace) -> int:
    """Run the full scoring pipeline: scorecard + gate + optional storage/registry."""
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path=config_path)

    code, filename = _read_file(args.file)

    # Score
    engine = RubricEngine(config=config)
    score_result = engine.score(
        code,
        filename,
        user=args.user,
        skill_used=args.skill,
    )

    # Gate
    evaluator = TierEvaluator(config=config.gate)
    gate_result = evaluator.evaluate(score_result, code, filename)

    # Output
    print(_print_score_result(score_result))
    print("")
    print(_print_gate_result(gate_result))

    # Store
    if args.store:
        storage = create_storage(config.storage)
        storage.append(score_result)
        print("\nResult stored.")

    # Registry update
    if args.tool:
        from registry.catalog.catalog import ToolCatalog

        catalog = ToolCatalog()
        try:
            catalog.update_scorecard(args.tool, score_result)
            print(f"\nTool '{args.tool}' scorecard updated.")
        except KeyError:
            print(f"\nWarning: Tool '{args.tool}' not found in registry.", file=sys.stderr)

    return 1 if gate_result.blocked else 0


def cmd_check(args: argparse.Namespace) -> int:
    """Quick gate check: score + tier, no storage."""
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path=config_path)

    code, filename = _read_file(args.file)

    engine = RubricEngine(config=config)
    score_result = engine.score(code, filename, user=args.user)

    evaluator = TierEvaluator(config=config.gate)
    gate_result = evaluator.evaluate(score_result, code, filename)

    print(f"Score: {score_result.composite_score:.2f}")
    print(f"Tier:  {_format_tier(gate_result.tier)}")

    if gate_result.advisory_messages:
        print("")
        for msg in gate_result.advisory_messages:
            print(f"  - {msg}")

    return 1 if gate_result.blocked else 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show tool registry status."""
    from registry.catalog.catalog import ToolCatalog

    catalog = ToolCatalog()
    tool = catalog.get(args.slug)

    if tool is None:
        print(f"Tool '{args.slug}' not found.", file=sys.stderr)
        return 1

    lines = []
    lines.append(f"Name:    {tool.name}")
    lines.append(f"Slug:    {tool.slug}")
    lines.append(f"Tier:    {tool.tier.value}")
    lines.append(f"Owner:   {tool.tech_owner or '(none)'}")
    lines.append(f"Users:   {len(tool.users)}")
    lines.append(f"Tags:    {', '.join(tool.tags) or '(none)'}")
    lines.append("")
    lines.append("Scorecard:")
    lines.append(f"  Composite: {tool.scorecard.latest_composite:.2f}")
    lines.append(f"  Trend:     {tool.scorecard.trend}")
    lines.append(f"  Scores:    {tool.scorecard.total_scores}")
    lines.append(f"  Red flags: {tool.scorecard.red_flags}")

    if tool.scorecard.latest_scores:
        lines.append("")
        lines.append("  Dimensions:")
        for dim, score in sorted(tool.scorecard.latest_scores.items()):
            lines.append(f"    {dim:<18} {score:.2f}")

    if tool.graduation_history:
        lines.append("")
        lines.append("Graduation History:")
        for event in tool.graduation_history:
            lines.append(
                f"  {event.from_tier.value} -> {event.to_tier.value}  "
                f"({event.date.strftime('%Y-%m-%d')})  {event.reason}"
            )

    print("\n".join(lines))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate summary report from stored scores."""
    from scorecard.dashboard.cli import (
        parse_since,
        view_by_skill,
        view_by_user,
        view_dimensions,
        view_overview,
        view_red_flags,
    )

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path=config_path)
    storage = create_storage(config.storage)

    filters = QueryFilters()
    if args.since:
        filters.start_date = parse_since(args.since)
    if args.user:
        filters.user = args.user
    if args.skill:
        filters.skill = args.skill

    results = storage.query(filters)
    since_label = f"since {args.since}" if args.since else "all time"

    print(view_overview(results, since_label))
    print(view_by_user(results))
    print(view_by_skill(results))
    print(view_dimensions(results))
    print(view_red_flags(results))

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point. Returns exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "score": cmd_score,
        "check": cmd_check,
        "status": cmd_status,
        "report": cmd_report,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
