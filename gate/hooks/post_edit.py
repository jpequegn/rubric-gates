"""Gate hook for Claude Code post-edit integration.

Runs after the scorecard hook. Evaluates code against tier thresholds
and pattern detectors, then outputs advisory or blocking messages.

Behavior by tier:
  GREEN  — silent, score logged
  YELLOW — advisory messages to stderr, non-blocking
  RED    — block message to stderr, non-zero exit code

Usage:
    python -m gate.hooks.post_edit --file path/to/file.py
    python -m gate.hooks.post_edit --file path/to/file.py --user alice
    python -m gate.hooks.post_edit --file path/to/file.py --no-store
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gate.patterns.anti_gaming import AntiGamingChecker
from gate.tiers.advisory import AdvisoryMessageGenerator
from gate.tiers.evaluator import TierEvaluator
from scorecard.engine import RubricEngine
from shared.config import load_config
from shared.models import GateResult, GateTier, ScoreResult
from shared.storage import create_storage


def _score_file(
    file_path: Path,
    user: str = "",
    skill_used: str = "",
) -> ScoreResult | None:
    """Score a file using the rubric engine.

    Returns None if the file can't be scored.
    """
    if not file_path.exists():
        return None
    if file_path.suffix != ".py":
        return None

    try:
        code = file_path.read_text()
    except Exception:
        return None

    if not code.strip():
        return None

    try:
        config = load_config()
        engine = RubricEngine(config=config)
        return engine.score(
            code,
            filename=file_path.name,
            user=user,
            skill_used=skill_used,
        )
    except Exception:
        return None


def run_gate(
    file_path: str,
    *,
    user: str = "",
    skill_used: str = "",
    quiet: bool = False,
    store: bool = True,
) -> GateResult | None:
    """Run the full gate evaluation on a file.

    Args:
        file_path: Path to the Python file.
        user: User who generated the code.
        skill_used: Claude Code skill used.
        quiet: If True, suppress output.
        store: If True, store results.

    Returns:
        GateResult or None if the file can't be evaluated.
    """
    path = Path(file_path)

    # Score the file
    score_result = _score_file(path, user=user, skill_used=skill_used)
    if score_result is None:
        return None

    # Read the code for pattern detection
    try:
        code = path.read_text()
    except Exception:
        return None

    # Apply anti-gaming adjustments
    gaming_checker = AntiGamingChecker()
    gaming_findings = gaming_checker.check(code, score_result=score_result)
    if gaming_findings:
        score_result = gaming_checker.apply_adjustments(score_result, gaming_findings)

    # Evaluate tier
    try:
        config = load_config()
        evaluator = TierEvaluator(config=config.gate)
        gate_result = evaluator.evaluate(score_result, code, path.name)
    except Exception as e:
        if not quiet:
            print(f"gate: evaluation error: {e}", file=sys.stderr)
        return None

    # Generate advisory messages for yellow tier
    if gate_result.tier == GateTier.YELLOW:
        advisor = AdvisoryMessageGenerator()
        advisories = advisor.generate(gate_result)
        gate_result = gate_result.model_copy(update={"advisory_messages": advisories})

    # Store results
    if store:
        try:
            storage = create_storage()
            storage.append(score_result)
        except Exception as e:
            if not quiet:
                print(f"gate: storage error: {e}", file=sys.stderr)

    # Output
    if not quiet:
        _print_output(gate_result)

    return gate_result


def _print_output(gate_result: GateResult) -> None:
    """Print gate output to stderr."""
    if gate_result.tier == GateTier.GREEN:
        # Silent for green
        return

    if gate_result.tier == GateTier.YELLOW:
        for msg in gate_result.advisory_messages:
            print(f"gate: advisory: {msg}", file=sys.stderr)
        return

    if gate_result.tier == GateTier.RED:
        for msg in gate_result.advisory_messages:
            print(f"gate: BLOCKED: {msg}", file=sys.stderr)

        # Show findings summary
        for finding in gate_result.pattern_findings:
            if finding.severity == "critical":
                print(
                    f"gate: BLOCKED: {finding.description} (line {finding.line_number})",
                    file=sys.stderr,
                )

        print("", file=sys.stderr)
        print("gate: Options:", file=sys.stderr)
        print("gate:   1. Fix the issues and re-run", file=sys.stderr)
        print(
            "gate:   2. Override with justification: "
            "python -m gate.overrides override --file <path> --reason '...'",
            file=sys.stderr,
        )
        print(
            "gate:   3. Escalate to tech team: "
            "python -m gate.overrides escalate --file <path> --reason '...'",
            file=sys.stderr,
        )


def format_gate_summary(gate_result: GateResult) -> str:
    """Format a one-line gate summary.

    Example: "GATE: GREEN (0.82)" or "GATE: RED [2 critical findings]"
    """
    tier = gate_result.tier.value.upper()
    score = gate_result.score_result.composite_score

    if gate_result.tier == GateTier.RED:
        n_critical = len(gate_result.critical_patterns_found)
        return f"GATE: {tier} ({score:.2f}) [{n_critical} critical finding(s)]"
    elif gate_result.tier == GateTier.YELLOW:
        n_advisories = len(gate_result.advisory_messages)
        return f"GATE: {tier} ({score:.2f}) [{n_advisories} advisory(ies)]"
    else:
        return f"GATE: {tier} ({score:.2f})"


def main() -> None:
    """CLI entry point for the gate hook."""
    parser = argparse.ArgumentParser(
        description="Rubric-gates gate hook: evaluate and optionally block code",
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the file to evaluate",
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
        "--quiet",
        action="store_true",
        help="Suppress output",
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Don't store results",
    )

    args = parser.parse_args()

    result = run_gate(
        args.file,
        user=args.user,
        skill_used=args.skill,
        quiet=args.quiet,
        store=not args.no_store,
    )

    if result is None:
        sys.exit(0)

    # Non-zero exit for red tier (signals Claude Code to pause)
    if result.tier == GateTier.RED:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
