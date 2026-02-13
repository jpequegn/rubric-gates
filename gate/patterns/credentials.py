"""Hardcoded credentials detector.

Detects API keys, passwords, secrets, and connection strings with
embedded credentials in Python source code.
"""

from __future__ import annotations

import re

from shared.models import PatternFinding

from .base import register_detector

# Known API key prefixes and their services
_KEY_PREFIXES = [
    (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API key"),
    (r"sk-ant-[a-zA-Z0-9\-]{20,}", "Anthropic API key"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key ID"),
    (r"ghp_[a-zA-Z0-9]{36,}", "GitHub personal access token"),
    (r"gho_[a-zA-Z0-9]{36,}", "GitHub OAuth token"),
    (r"github_pat_[a-zA-Z0-9_]{20,}", "GitHub fine-grained token"),
    (r"glpat-[a-zA-Z0-9\-]{20,}", "GitLab personal access token"),
    (r"xoxb-[a-zA-Z0-9\-]{20,}", "Slack bot token"),
    (r"xoxp-[a-zA-Z0-9\-]{20,}", "Slack user token"),
    (r"SG\.[a-zA-Z0-9\-]{20,}", "SendGrid API key"),
    (r"sk_live_[a-zA-Z0-9]{20,}", "Stripe live key"),
    (r"pk_live_[a-zA-Z0-9]{20,}", "Stripe publishable key"),
]

# Password/secret assignment patterns
_SECRET_ASSIGN = re.compile(
    r"""(?:password|passwd|pwd|secret|api_key|apikey|token|auth_token|access_token"""
    r"""|secret_key|private_key)\s*=\s*["'][^"']{4,}["']""",
    re.IGNORECASE,
)

# Connection strings with embedded passwords
_CONN_STRING = re.compile(
    r"""(?:postgres|mysql|mongodb|redis|amqp)(?:ql)?://[^:\s]+:[^@\s]+@""",
    re.IGNORECASE,
)

# Bearer token in string literals
_BEARER_TOKEN = re.compile(
    r"""["']Bearer\s+[a-zA-Z0-9\-_.]{20,}["']""",
)


class CredentialsDetector:
    """Detects hardcoded credentials in source code."""

    name = "hardcoded_credentials"
    severity = "critical"

    def detect(self, code: str, filename: str) -> list[PatternFinding]:
        findings: list[PatternFinding] = []
        lines = code.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # Skip comments
            if stripped.startswith("#"):
                continue

            # Check API key prefixes
            for pattern, service in _KEY_PREFIXES:
                if re.search(pattern, line):
                    findings.append(
                        PatternFinding(
                            pattern=self.name,
                            severity=self.severity,
                            line_number=line_num,
                            line_content=stripped,
                            description=f"Possible {service} found in source code",
                            remediation="Move to environment variable or secrets manager. "
                            "Use os.environ.get() or a .env file.",
                        )
                    )

            # Check password/secret assignments
            if _SECRET_ASSIGN.search(line):
                # Exclude os.environ lookups and empty defaults
                if "os.environ" not in line and "getenv" not in line:
                    findings.append(
                        PatternFinding(
                            pattern=self.name,
                            severity=self.severity,
                            line_number=line_num,
                            line_content=stripped,
                            description="Hardcoded secret or password assignment",
                            remediation="Use os.environ.get() or a secrets manager "
                            "instead of hardcoding credentials.",
                        )
                    )

            # Check connection strings
            if _CONN_STRING.search(line):
                if "os.environ" not in line and "getenv" not in line:
                    findings.append(
                        PatternFinding(
                            pattern=self.name,
                            severity="high",
                            line_number=line_num,
                            line_content=stripped,
                            description="Connection string with embedded credentials",
                            remediation="Store connection credentials in environment variables. "
                            "Build the connection string from env vars at runtime.",
                        )
                    )

            # Check bearer tokens
            if _BEARER_TOKEN.search(line):
                findings.append(
                    PatternFinding(
                        pattern=self.name,
                        severity=self.severity,
                        line_number=line_num,
                        line_content=stripped,
                        description="Hardcoded Bearer token in source",
                        remediation="Load the token from an environment variable or secrets manager.",
                    )
                )

        return findings


register_detector(CredentialsDetector())
