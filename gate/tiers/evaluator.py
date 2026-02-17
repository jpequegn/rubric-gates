"""Tier evaluation engine.

Takes a ScoreResult + code and classifies output as green, yellow, or red.

Tier logic:
  RED if:
    - ANY critical pattern detected
    - Security dimension score < red threshold (default 0.3)

  YELLOW if:
    - Composite score between yellow and green thresholds
    - Any individual dimension below its yellow threshold
    - Any high-severity pattern detected

  GREEN if:
    - Composite score above green threshold
    - No critical/high patterns detected
    - All dimensions above their yellow thresholds
"""

from __future__ import annotations

from shared.config import GateConfig, load_config
from shared.models import Dimension, GateResult, GateTier, PatternFinding, ScoreResult

from gate.patterns import get_configured_detectors


# Default per-dimension yellow thresholds
_DEFAULT_DIM_YELLOW: dict[str, float] = {
    Dimension.CORRECTNESS.value: 0.4,
    Dimension.SECURITY.value: 0.4,
    Dimension.MAINTAINABILITY.value: 0.3,
    Dimension.DOCUMENTATION.value: 0.3,
    Dimension.TESTABILITY.value: 0.3,
}


class TierEvaluator:
    """Classifies code into green/yellow/red tiers.

    Args:
        config: Gate configuration with thresholds.
        dimension_yellow_thresholds: Per-dimension yellow thresholds.
            Defaults to sensible values if not provided.
    """

    def __init__(
        self,
        config: GateConfig | None = None,
        dimension_yellow_thresholds: dict[str, float] | None = None,
    ) -> None:
        if config is None:
            config = load_config().gate
        self.config = config
        self.dim_yellow = dimension_yellow_thresholds or _DEFAULT_DIM_YELLOW
        self.detectors = get_configured_detectors(config.patterns)

    def evaluate(
        self,
        score_result: ScoreResult,
        code: str,
        filename: str,
    ) -> GateResult:
        """Classify code into green/yellow/red tier.

        Args:
            score_result: The scoring result from the rubric engine.
            code: Source code to scan for critical patterns.
            filename: Name of the file being evaluated.

        Returns:
            GateResult with tier, findings, and advisory messages.
        """
        # Run all pattern detectors
        findings: list[PatternFinding] = []
        for detector in self.detectors:
            findings.extend(detector.detect(code, filename))

        # Classify findings by severity
        critical_findings = [f for f in findings if f.severity == "critical"]
        high_findings = [f for f in findings if f.severity == "high"]

        # Extract critical pattern names
        critical_patterns = list({f.pattern for f in critical_findings})

        # Build advisory messages
        advisories: list[str] = []

        # --- RED checks ---
        tier = GateTier.GREEN
        blocked = False

        # Critical patterns → RED
        if critical_findings:
            tier = GateTier.RED
            blocked = True
            for finding in critical_findings:
                advisories.append(
                    f"BLOCKED: {finding.description} (line {finding.line_number}). "
                    f"{finding.remediation}"
                )

        # Security dimension below red threshold → RED
        security_score = self._get_dimension_score(score_result, Dimension.SECURITY.value)
        if security_score is not None:
            if security_score < self.config.thresholds.red.security:
                tier = GateTier.RED
                blocked = True
                advisories.append(
                    f"BLOCKED: Security score {security_score:.2f} is below "
                    f"red threshold {self.config.thresholds.red.security:.2f}."
                )

        # If already RED, skip yellow/green checks
        if tier == GateTier.RED:
            return GateResult(
                tier=tier,
                score_result=score_result,
                critical_patterns_found=critical_patterns,
                pattern_findings=findings,
                advisory_messages=advisories,
                blocked=blocked,
            )

        # --- YELLOW checks ---

        # Composite score between yellow and green thresholds
        composite = score_result.composite_score
        green_min = self.config.thresholds.green.min_composite
        yellow_min = self.config.thresholds.yellow.min_composite

        if composite < green_min:
            tier = GateTier.YELLOW
            if composite >= yellow_min:
                advisories.append(
                    f"Composite score {composite:.2f} is below green threshold "
                    f"{green_min:.2f}. Consider improving before merging."
                )
            else:
                advisories.append(
                    f"Composite score {composite:.2f} is below yellow threshold "
                    f"{yellow_min:.2f}. Significant improvements needed."
                )

        # Any high-severity pattern → YELLOW
        if high_findings:
            tier = max(tier, GateTier.YELLOW, key=_tier_severity)
            for finding in high_findings:
                advisories.append(
                    f"WARNING: {finding.description} (line {finding.line_number}). "
                    f"{finding.remediation}"
                )

        # Any dimension below its yellow threshold → YELLOW
        for ds in score_result.dimension_scores:
            dim_threshold = self.dim_yellow.get(ds.dimension.value, 0.3)
            if ds.score < dim_threshold:
                tier = max(tier, GateTier.YELLOW, key=_tier_severity)
                advisories.append(
                    f"Dimension '{ds.dimension.value}' scored {ds.score:.2f}, "
                    f"below threshold {dim_threshold:.2f}."
                )

        return GateResult(
            tier=tier,
            score_result=score_result,
            critical_patterns_found=critical_patterns,
            pattern_findings=findings,
            advisory_messages=advisories,
            blocked=blocked,
        )

    @staticmethod
    def _get_dimension_score(result: ScoreResult, dimension: str) -> float | None:
        """Get score for a specific dimension, or None if not present."""
        for ds in result.dimension_scores:
            if ds.dimension.value == dimension:
                return ds.score
        return None


def _tier_severity(tier: GateTier) -> int:
    """Map tier to numeric severity for comparison."""
    return {GateTier.GREEN: 0, GateTier.YELLOW: 1, GateTier.RED: 2}[tier]
