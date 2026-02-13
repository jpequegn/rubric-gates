"""Post-edit hook for Claude Code integration.

Scores files after Write/Edit tool use. Two modes:
- Quick mode (default): rule-based only, instant feedback (<100ms)
- Full mode (--full): includes LLM scorers, runs async

Usage:
    # Quick score (non-blocking)
    python -m scorecard.hooks.post_edit --file path/to/file.py

    # Full score (async, logs to storage)
    python -m scorecard.hooks.post_edit --file path/to/file.py --full

    # Quiet mode (no output, just log)
    python -m scorecard.hooks.post_edit --file path/to/file.py --quiet

    # Specify user/skill
    python -m scorecard.hooks.post_edit --file path/to/file.py --user alice --skill scaffold-api
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

from scorecard.engine import RubricEngine
from shared.config import DimensionConfig, RubricGatesConfig, load_config
from shared.models import Dimension, ScoreResult
from shared.storage import create_storage


def _build_quick_config(base_config: RubricGatesConfig) -> RubricGatesConfig:
    """Build a config that disables LLM-based scorers for quick mode."""
    quick_dims = {}
    for name, dim_config in base_config.scorecard.dimensions.items():
        if name in ("documentation", "testability"):
            # Keep enabled but engine uses use_llm=False by default
            quick_dims[name] = DimensionConfig(
                weight=dim_config.weight,
                enabled=dim_config.enabled,
            )
        else:
            quick_dims[name] = dim_config.model_copy()
    config_data = base_config.model_dump()
    config_data["scorecard"]["dimensions"] = {k: v.model_dump() for k, v in quick_dims.items()}
    return RubricGatesConfig.model_validate(config_data)


def format_score_line(result: ScoreResult) -> str:
    """Format a one-line score summary.

    Example: "Score: 0.82 [C:0.9 S:0.8 M:0.7 D:0.8 T:0.9]"
    """
    dim_abbrev = {
        Dimension.CORRECTNESS: "C",
        Dimension.SECURITY: "S",
        Dimension.MAINTAINABILITY: "M",
        Dimension.DOCUMENTATION: "D",
        Dimension.TESTABILITY: "T",
    }
    parts = []
    for ds in result.dimension_scores:
        abbr = dim_abbrev.get(ds.dimension, ds.dimension.value[0].upper())
        parts.append(f"{abbr}:{ds.score:.1f}")
    dim_str = " ".join(parts)
    return f"Score: {result.composite_score:.2f} [{dim_str}]"


def run_scoring(
    file_path: str,
    *,
    user: str = "",
    skill_used: str = "",
    full: bool = False,
    quiet: bool = False,
    store: bool = True,
) -> ScoreResult | None:
    """Score a file and optionally store + print results.

    Args:
        file_path: Path to the Python file to score.
        user: User who generated the code.
        skill_used: Claude Code skill used.
        full: If True, include LLM-based scorers.
        quiet: If True, suppress output.
        store: If True, append result to storage.

    Returns:
        ScoreResult or None if the file can't be scored.
    """
    path = Path(file_path)
    if not path.exists():
        if not quiet:
            print(f"rubric-gates: file not found: {path}", file=sys.stderr)
        return None

    if not path.suffix == ".py":
        # Only score Python files
        return None

    try:
        code = path.read_text()
    except Exception as e:
        if not quiet:
            print(f"rubric-gates: read error: {e}", file=sys.stderr)
        return None

    if not code.strip():
        return None

    # Load config and build engine
    try:
        config = load_config()
        if not full:
            config = _build_quick_config(config)
        engine = RubricEngine(config=config)
    except Exception as e:
        if not quiet:
            print(f"rubric-gates: config error: {e}", file=sys.stderr)
        return None

    # Score
    try:
        result = engine.score(
            code,
            filename=path.name,
            user=user,
            skill_used=skill_used,
        )
    except Exception as e:
        if not quiet:
            print(f"rubric-gates: scoring error: {e}", file=sys.stderr)
        return None

    # Store
    if store:
        try:
            storage = create_storage()
            storage.append(result)
        except Exception as e:
            if not quiet:
                print(f"rubric-gates: storage error: {e}", file=sys.stderr)

    # Output
    if not quiet:
        print(f"rubric-gates: {format_score_line(result)}", file=sys.stderr)

    return result


def _run_full_async(file_path: str, user: str, skill_used: str) -> None:
    """Run full scoring in a background process."""
    run_scoring(
        file_path,
        user=user,
        skill_used=skill_used,
        full=True,
        quiet=True,
        store=True,
    )


def main() -> None:
    """CLI entry point for the post-edit hook."""
    parser = argparse.ArgumentParser(
        description="Rubric-gates post-edit hook: score a file after edit",
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the file to score",
    )
    parser.add_argument(
        "--user",
        default="",
        help="User who generated the code",
    )
    parser.add_argument(
        "--skill",
        default="",
        help="Claude Code skill used",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full scoring including LLM judges (slower)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output (just store results)",
    )
    parser.add_argument(
        "--async",
        dest="run_async",
        action="store_true",
        help="Run full scoring in background process",
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Don't store results (dry run)",
    )

    args = parser.parse_args()

    # Quick score (always runs synchronously for instant feedback)
    result = run_scoring(
        args.file,
        user=args.user,
        skill_used=args.skill,
        full=args.full and not args.run_async,
        quiet=args.quiet,
        store=not args.no_store,
    )

    # If --async, also kick off full scoring in background
    if args.run_async and result is not None:
        proc = multiprocessing.Process(
            target=_run_full_async,
            args=(args.file, args.user, args.skill),
            daemon=True,
        )
        proc.start()
        # Don't wait — let it run in background

    sys.exit(0 if result is not None else 1)


if __name__ == "__main__":
    main()
