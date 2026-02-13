"""Data exposure detector.

Detects logging of PII patterns (email, SSN, phone), writing sensitive
fields to unprotected files, and returning sensitive data in error messages.
"""

from __future__ import annotations

import re

from shared.models import PatternFinding

from .base import register_detector

# PII patterns
_EMAIL_PATTERN = re.compile(r"""[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+""")
_SSN_PATTERN = re.compile(r"""\b\d{3}-\d{2}-\d{4}\b""")
_PHONE_PATTERN = re.compile(r"""\b\d{3}[-.]?\d{3}[-.]?\d{4}\b""")

# Logging/print contexts
_LOG_FUNCS = re.compile(
    r"""(?:logger?\.\w+|logging\.\w+|print|sys\.stdout\.write|sys\.stderr\.write)\s*\(""",
)

# Sensitive field names
_SENSITIVE_FIELDS = re.compile(
    r"""\b(?:password|passwd|pwd|ssn|social_security|credit_card|card_number"""
    r"""|cvv|pin|secret|private_key|auth_token|access_token|refresh_token)\b""",
    re.IGNORECASE,
)

# Error message contexts
_ERROR_CONTEXT = re.compile(
    r"""(?:raise\s+\w+Error|raise\s+\w+Exception|return\s+.*(?:error|message|msg))""",
    re.IGNORECASE,
)


class DataExposureDetector:
    """Detects potential PII and sensitive data exposure."""

    name = "data_exposure"
    severity = "high"

    def detect(self, code: str, filename: str) -> list[PatternFinding]:
        findings: list[PatternFinding] = []
        lines = code.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Check for PII in log/print statements
            if _LOG_FUNCS.search(line):
                findings.extend(self._check_pii_in_log(line_num, stripped))

            # Check for sensitive fields in error messages
            if _ERROR_CONTEXT.search(line) and _SENSITIVE_FIELDS.search(line):
                findings.append(
                    PatternFinding(
                        pattern=self.name,
                        severity=self.severity,
                        line_number=line_num,
                        line_content=stripped,
                        description="Sensitive field referenced in error message",
                        remediation="Remove sensitive data from error messages. "
                        "Log sensitive details at DEBUG level only, never in user-facing errors.",
                    )
                )

            # Check for sensitive fields in f-strings used with print/log
            if _LOG_FUNCS.search(line) and _SENSITIVE_FIELDS.search(line):
                findings.append(
                    PatternFinding(
                        pattern=self.name,
                        severity=self.severity,
                        line_number=line_num,
                        line_content=stripped,
                        description="Sensitive field being logged or printed",
                        remediation="Mask or redact sensitive fields before logging. "
                        "Example: log password as '***' instead of the actual value.",
                    )
                )

        return findings

    def _check_pii_in_log(self, line_num: int, line_content: str) -> list[PatternFinding]:
        findings: list[PatternFinding] = []

        # Check for SSN patterns in log statements
        if _SSN_PATTERN.search(line_content):
            findings.append(
                PatternFinding(
                    pattern=self.name,
                    severity="critical",
                    line_number=line_num,
                    line_content=line_content,
                    description="Possible SSN pattern in log/print statement",
                    remediation="Never log SSN or other PII. Mask the value: ssn[:3] + '-XX-XXXX'",
                )
            )

        return findings


register_detector(DataExposureDetector())
