"""CLI entry point for scoring files.

Usage:
    python -m scorecard.score path/to/file.py [path/to/file2.py ...]
"""

import sys
from pathlib import Path

from scorecard.engine import RubricEngine


def main() -> None:
    """Score files from the command line."""
    if len(sys.argv) < 2:
        print("Usage: python -m scorecard.score <file.py> [file2.py ...]")
        sys.exit(1)

    file_paths = [Path(p) for p in sys.argv[1:]]

    # Validate files exist
    for fp in file_paths:
        if not fp.exists():
            print(f"Error: {fp} not found")
            sys.exit(1)
        if not fp.suffix == ".py":
            print(f"Warning: {fp} is not a .py file, skipping")
            file_paths.remove(fp)

    if not file_paths:
        print("No valid Python files to score.")
        sys.exit(1)

    engine = RubricEngine()
    results = engine.score_files(file_paths)

    for result in results:
        _print_result(result)


def _print_result(result) -> None:
    """Pretty-print a ScoreResult."""
    files = ", ".join(result.files_touched) or "(none)"
    print(f"\n{'=' * 60}")
    print(f"File: {files}")
    print(f"Composite Score: {result.composite_score:.2f}")
    print(f"User: {result.user}")
    print(f"{'─' * 60}")

    for ds in result.dimension_scores:
        bar = "█" * int(ds.score * 20) + "░" * (20 - int(ds.score * 20))
        print(f"  {ds.dimension.value:<18} {bar} {ds.score:.2f}  ({ds.method.value})")
        if ds.details:
            # Truncate long details
            detail_text = ds.details[:100] + "..." if len(ds.details) > 100 else ds.details
            print(f"    {detail_text}")

    if result.metadata.get("scorer_errors"):
        print("\n  Errors:")
        for err in result.metadata["scorer_errors"]:
            print(f"    ⚠ {err}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
