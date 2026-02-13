"""Base classes and registry for pattern detectors."""

from __future__ import annotations

from typing import Protocol

from shared.models import PatternFinding

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
    """Return all registered detectors."""
    return list(_DETECTORS)


def scan_all(code: str, filename: str) -> list[PatternFinding]:
    """Run all registered detectors and return combined findings."""
    findings: list[PatternFinding] = []
    for detector in _DETECTORS:
        findings.extend(detector.detect(code, filename))
    return findings
