"""Custom pattern detector loaded from YAML config.

Allows organizations to add their own detection rules without code changes.
Custom patterns are regex-based and additive to built-in detectors.
"""

from __future__ import annotations

import re

from shared.config import PatternRuleConfig
from shared.models import PatternFinding

_VALID_SEVERITIES = {"critical", "high", "medium", "low"}


class CustomPatternDetector:
    """Detects patterns defined in YAML configuration.

    Each instance wraps a single PatternRuleConfig and compiles its
    regex pattern for efficient matching.
    """

    def __init__(self, rule: PatternRuleConfig) -> None:
        if rule.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{rule.severity}' for pattern '{rule.name}'. "
                f"Must be one of: {', '.join(sorted(_VALID_SEVERITIES))}"
            )
        try:
            self._compiled = re.compile(rule.pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern for '{rule.name}': {e}") from e

        self.rule = rule
        self.name = f"custom:{rule.name}"
        self.severity = rule.severity

    def detect(self, code: str, filename: str) -> list[PatternFinding]:
        """Scan code for this custom pattern."""
        findings: list[PatternFinding] = []

        for line_num, line in enumerate(code.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            if self._compiled.search(line):
                findings.append(
                    PatternFinding(
                        pattern=self.name,
                        severity=self.severity,
                        line_number=line_num,
                        line_content=stripped,
                        description=self.rule.message
                        or f"Custom pattern '{self.rule.name}' matched",
                        remediation=self.rule.remediation,
                    )
                )

        return findings


def load_custom_detectors(rules: list[PatternRuleConfig]) -> list[CustomPatternDetector]:
    """Create custom detectors from config rules.

    Validates each rule and raises ValueError on invalid config.
    """
    detectors: list[CustomPatternDetector] = []
    for rule in rules:
        detectors.append(CustomPatternDetector(rule))
    return detectors
