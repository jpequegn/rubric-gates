"""Unvetted dependencies detector.

Detects pip install with unpinned versions, installing from URLs,
and importing packages not in requirements files.
"""

from __future__ import annotations

import re

from shared.models import PatternFinding

from .base import register_detector

# pip install / uv add with no version pin
_PIP_INSTALL = re.compile(
    r"""(?:pip\s+install|uv\s+add)\s+([a-zA-Z0-9_-]+)(?:\s|$|&&|\|)""",
)

# Install from URL
_URL_INSTALL = re.compile(
    r"""(?:pip\s+install|uv\s+add)\s+(?:https?://|git\+|git://)\S+""",
)

# subprocess calls to pip/uv
_SUBPROCESS_PIP = re.compile(
    r"""(?:subprocess|os\.system)\S*.*(?:pip\s+install|uv\s+add)""",
)

# __import__ with dynamic string
_DYNAMIC_IMPORT = re.compile(
    r"""__import__\s*\((?!['"])""",
)


class DependenciesDetector:
    """Detects unvetted or unsafe dependency patterns."""

    name = "unvetted_dependencies"
    severity = "medium"

    def detect(self, code: str, filename: str) -> list[PatternFinding]:
        findings: list[PatternFinding] = []
        lines = code.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Check for URL-based installs (higher severity)
            if _URL_INSTALL.search(line):
                findings.append(
                    PatternFinding(
                        pattern=self.name,
                        severity="high",
                        line_number=line_num,
                        line_content=stripped,
                        description="Installing package from URL instead of PyPI",
                        remediation="Install from PyPI with a pinned version. "
                        "If a custom source is needed, use a private index.",
                    )
                )
                continue

            # Check for unpinned pip install
            match = _PIP_INSTALL.search(line)
            if match:
                pkg = match.group(1)
                # Check if version is pinned anywhere on the line
                version_pin = re.search(rf"""{re.escape(pkg)}[=<>!~]""", line)
                if not version_pin:
                    findings.append(
                        PatternFinding(
                            pattern=self.name,
                            severity=self.severity,
                            line_number=line_num,
                            line_content=stripped,
                            description=f"Unpinned package install: {pkg}",
                            remediation=f"Pin the version: pip install {pkg}==X.Y.Z "
                            "or add to pyproject.toml with a version constraint.",
                        )
                    )

            # Check for subprocess pip calls
            if _SUBPROCESS_PIP.search(line):
                findings.append(
                    PatternFinding(
                        pattern=self.name,
                        severity="high",
                        line_number=line_num,
                        line_content=stripped,
                        description="Runtime package installation via subprocess",
                        remediation="Declare dependencies in pyproject.toml or requirements.txt. "
                        "Avoid installing packages at runtime.",
                    )
                )

            # Check for dynamic __import__
            if _DYNAMIC_IMPORT.search(line):
                findings.append(
                    PatternFinding(
                        pattern=self.name,
                        severity=self.severity,
                        line_number=line_num,
                        line_content=stripped,
                        description="Dynamic __import__ with non-literal argument",
                        remediation="Use static imports or importlib.import_module() "
                        "with validated module names.",
                    )
                )

        return findings


register_detector(DependenciesDetector())
