"""Advisory message system for yellow-tier user feedback.

Generates friendly, actionable feedback when code hits yellow tier.
Messages are data-driven via templates, not hardcoded if-else chains.

Design principles:
- Friendly tone (users are colleagues, not adversaries)
- Specific (reference line numbers, function names, scores)
- Actionable (every advisory suggests a concrete fix)
- Brief (1-2 sentences max per advisory)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.models import Dimension, GateResult, GateTier, PatternFinding


# --- Message Templates ---


@dataclass
class AdvisoryTemplate:
    """A template for generating advisory messages.

    Attributes:
        category: The category this template applies to (e.g., "maintainability").
        condition: A callable that checks if this template applies.
        template: A format string with placeholders for context values.
    """

    category: str
    pattern_name: str = ""
    severity: str = ""
    dimension: str = ""
    template: str = ""


# Pattern-based templates: matched by pattern name from findings
_PATTERN_TEMPLATES: list[AdvisoryTemplate] = [
    # Credentials (high severity — connection strings)
    AdvisoryTemplate(
        category="security",
        pattern_name="hardcoded_credentials",
        severity="high",
        template=(
            "Connection string with embedded credentials on line {line_number}. "
            "Store credentials in environment variables instead."
        ),
    ),
    # File ops
    AdvisoryTemplate(
        category="security",
        pattern_name="unsafe_file_ops",
        severity="high",
        template=("Unsafe file operation on line {line_number}: {description}. {remediation}"),
    ),
    # Dependencies
    AdvisoryTemplate(
        category="dependencies",
        pattern_name="unvetted_dependencies",
        template=("Dependency concern on line {line_number}: {description}. {remediation}"),
    ),
    # Data exposure
    AdvisoryTemplate(
        category="security",
        pattern_name="data_exposure",
        severity="high",
        template=("Potential data exposure on line {line_number}: {description}. {remediation}"),
    ),
]

# Dimension-based templates: matched by dimension name from score results
_DIMENSION_TEMPLATES: dict[str, list[str]] = {
    Dimension.MAINTAINABILITY.value: [
        (
            "Maintainability scored {score:.2f} (target: >= {threshold:.2f}). "
            "Consider breaking complex functions into smaller pieces."
        ),
    ],
    Dimension.DOCUMENTATION.value: [
        (
            "Documentation scored {score:.2f} (target: >= {threshold:.2f}). "
            "Adding a module docstring and function docs helps others understand your code."
        ),
    ],
    Dimension.TESTABILITY.value: [
        (
            "Testability scored {score:.2f} (target: >= {threshold:.2f}). "
            "Even a few basic tests help catch regressions early."
        ),
    ],
    Dimension.CORRECTNESS.value: [
        (
            "Correctness scored {score:.2f} (target: >= {threshold:.2f}). "
            "Check for undefined variables or unreachable code paths."
        ),
    ],
    Dimension.SECURITY.value: [
        (
            "Security scored {score:.2f} (target: >= {threshold:.2f}). "
            "Review for common vulnerabilities like injection or unsafe operations."
        ),
    ],
}

# Composite score templates
_COMPOSITE_TEMPLATES = {
    "below_green": (
        "Overall quality score is {score:.2f} (green threshold: {green:.2f}). "
        "A few improvements will get this to green."
    ),
    "below_yellow": (
        "Overall quality score is {score:.2f} (yellow threshold: {yellow:.2f}). "
        "This needs significant improvements before merging."
    ),
}


# --- Generator ---


@dataclass
class AdvisoryConfig:
    """Configuration for advisory message generation.

    Attributes:
        dimension_thresholds: Per-dimension thresholds that trigger advisories.
        green_composite: Composite score threshold for green tier.
        yellow_composite: Composite score threshold for yellow tier.
        max_messages: Maximum number of advisory messages to return.
        custom_templates: Additional pattern templates to use.
    """

    dimension_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            Dimension.CORRECTNESS.value: 0.4,
            Dimension.SECURITY.value: 0.4,
            Dimension.MAINTAINABILITY.value: 0.3,
            Dimension.DOCUMENTATION.value: 0.3,
            Dimension.TESTABILITY.value: 0.3,
        }
    )
    green_composite: float = 0.7
    yellow_composite: float = 0.5
    max_messages: int = 5
    custom_templates: list[AdvisoryTemplate] = field(default_factory=list)


class AdvisoryMessageGenerator:
    """Generates user-friendly advisory messages for yellow-tier findings.

    Args:
        config: Advisory configuration. Uses defaults if not provided.
    """

    def __init__(self, config: AdvisoryConfig | None = None) -> None:
        self.config = config or AdvisoryConfig()

    def generate(self, gate_result: GateResult) -> list[str]:
        """Generate advisory messages for a gate result.

        Only generates messages for yellow-tier results. Green results
        need no advisories, and red results use the blocking messages
        from the evaluator.

        Args:
            gate_result: The gate evaluation result.

        Returns:
            List of friendly, actionable advisory messages.
        """
        if gate_result.tier == GateTier.GREEN:
            return []

        if gate_result.tier == GateTier.RED:
            # Red tier uses blocking messages from evaluator, not advisories
            return []

        messages: list[str] = []

        # 1. Composite score advisories
        messages.extend(self._composite_advisories(gate_result))

        # 2. Pattern-based advisories (high/medium severity only)
        messages.extend(self._pattern_advisories(gate_result))

        # 3. Dimension-based advisories
        messages.extend(self._dimension_advisories(gate_result))

        # Deduplicate and limit
        seen: set[str] = set()
        unique: list[str] = []
        for msg in messages:
            if msg not in seen:
                seen.add(msg)
                unique.append(msg)

        return unique[: self.config.max_messages]

    def _composite_advisories(self, gate_result: GateResult) -> list[str]:
        """Generate advisories based on composite score."""
        score = gate_result.score_result.composite_score
        messages: list[str] = []

        if score < self.config.yellow_composite:
            messages.append(
                _COMPOSITE_TEMPLATES["below_yellow"].format(
                    score=score, yellow=self.config.yellow_composite
                )
            )
        elif score < self.config.green_composite:
            messages.append(
                _COMPOSITE_TEMPLATES["below_green"].format(
                    score=score, green=self.config.green_composite
                )
            )

        return messages

    def _pattern_advisories(self, gate_result: GateResult) -> list[str]:
        """Generate advisories from pattern findings."""
        messages: list[str] = []
        all_templates = _PATTERN_TEMPLATES + self.config.custom_templates

        for finding in gate_result.pattern_findings:
            # Skip critical findings — those are handled by the evaluator
            if finding.severity == "critical":
                continue

            template = self._match_template(finding, all_templates)
            if template:
                msg = template.template.format(
                    line_number=finding.line_number,
                    description=finding.description,
                    remediation=finding.remediation,
                    line_content=finding.line_content,
                )
                messages.append(msg)
            else:
                # Fallback: use the finding's own description + remediation
                messages.append(
                    f"Line {finding.line_number}: {finding.description}. {finding.remediation}"
                )

        return messages

    def _dimension_advisories(self, gate_result: GateResult) -> list[str]:
        """Generate advisories from dimension scores below thresholds."""
        messages: list[str] = []

        for ds in gate_result.score_result.dimension_scores:
            threshold = self.config.dimension_thresholds.get(ds.dimension.value, 0.3)
            if ds.score < threshold:
                templates = _DIMENSION_TEMPLATES.get(ds.dimension.value, [])
                if templates:
                    msg = templates[0].format(score=ds.score, threshold=threshold)
                    messages.append(msg)

        return messages

    @staticmethod
    def _match_template(
        finding: PatternFinding,
        templates: list[AdvisoryTemplate],
    ) -> AdvisoryTemplate | None:
        """Find the best matching template for a finding."""
        for tmpl in templates:
            if tmpl.pattern_name and tmpl.pattern_name != finding.pattern:
                continue
            if tmpl.severity and tmpl.severity != finding.severity:
                continue
            return tmpl
        return None
