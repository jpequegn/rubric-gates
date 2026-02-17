"""Base classes and registry for pattern detectors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from shared.models import PatternFinding

if TYPE_CHECKING:
    from shared.config import PatternsConfig

# Global detector registry
_DETECTORS: list[PatternDetector] = []


class PatternDetector(Protocol):
    """Protocol for critical pattern detectors."""

    name: str
    severity: str  # "critical", "high", "medium"

    def detect(self, code: str, filename: str) -> list[PatternFinding]:
        """Return list of findings, empty if clean."""
        ...


def register_detector(detector: PatternDetector) -> PatternDetector:
    """Register a detector in the global registry."""
    _DETECTORS.append(detector)
    return detector


def get_all_detectors() -> list[PatternDetector]:
    """Return all registered (built-in) detectors."""
    return list(_DETECTORS)


def get_configured_detectors(patterns_config: PatternsConfig) -> list[PatternDetector]:
    """Return built-in + custom detectors, with disabled ones filtered out.

    Args:
        patterns_config: Patterns configuration with custom rules and disable list.

    Returns:
        List of active detectors.
    """
    from gate.patterns.custom import load_custom_detectors

    disabled = set(patterns_config.disabled)

    # Start with built-in detectors, excluding disabled ones
    detectors: list[PatternDetector] = [d for d in _DETECTORS if d.name not in disabled]

    # Add custom detectors (also check disabled, using custom:name format)
    custom = load_custom_detectors(patterns_config.custom)
    for cd in custom:
        if cd.name not in disabled and cd.rule.name not in disabled:
            detectors.append(cd)

    return detectors


def scan_all(code: str, filename: str) -> list[PatternFinding]:
    """Run all registered detectors and return combined findings."""
    findings: list[PatternFinding] = []
    for detector in _DETECTORS:
        findings.extend(detector.detect(code, filename))
    return findings
