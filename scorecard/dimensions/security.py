"""Security dimension scorer.

Detects common security vulnerabilities using rule-based pattern matching
and AST analysis — no LLM calls, fully deterministic.
"""

import ast
import re
from dataclasses import dataclass, field
from enum import Enum

from shared.models import Dimension, DimensionScore, ScoringMethod


# --- Severity ---


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"


SEVERITY_PENALTY = {
    Severity.CRITICAL: 0.3,
    Severity.HIGH: 0.2,
    Severity.MEDIUM: 0.1,
}


# --- Finding ---


@dataclass
class SecurityFinding:
    """A single security finding."""

    pattern: str
    severity: Severity
    line: int
    description: str


@dataclass
class SecurityReport:
    """Aggregated security findings."""

    findings: list[SecurityFinding] = field(default_factory=list)

    @property
    def score(self) -> float:
        total_penalty = sum(SEVERITY_PENALTY[f.severity] for f in self.findings)
        return max(0.0, 1.0 - total_penalty)

    @property
    def details_text(self) -> str:
        if not self.findings:
            return "No security issues detected"
        lines = []
        for f in self.findings:
            lines.append(
                f"[{f.severity.value.upper()}] {f.pattern}: {f.description} (line {f.line})"
            )
        return "; ".join(lines)


# --- Regex-based Detectors ---

# Common API key / secret patterns
_CREDENTIAL_PATTERNS = [
    # Generic API key assignments
    re.compile(
        r"""(?:api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?key|auth[_-]?token|password|passwd|pwd)\s*=\s*['"][A-Za-z0-9+/=_\-]{8,}['"]""",
        re.IGNORECASE,
    ),
    # AWS access key ID (starts with AKIA)
    re.compile(r"""['"]AKIA[0-9A-Z]{16}['"]"""),
    # Generic long hex/base64 tokens assigned to variables
    re.compile(
        r"""(?:token|secret|key|credential)\s*=\s*['"][a-f0-9]{32,}['"]""",
        re.IGNORECASE,
    ),
]

_SENSITIVE_PATH_PATTERNS = re.compile(
    r"""['"](?:/etc/(?:passwd|shadow|sudoers)|~?/\.ssh/|~?/\.aws/|~?/\.env)""",
)

_INSECURE_HTTP_PATTERN = re.compile(
    r"""['"]http://(?!localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])""",
)


def detect_hardcoded_credentials(code: str) -> list[SecurityFinding]:
    """Detect hardcoded API keys, passwords, and secrets."""
    findings: list[SecurityFinding] = []
    for i, line in enumerate(code.splitlines(), 1):
        # Skip comments
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern in _CREDENTIAL_PATTERNS:
            if pattern.search(line):
                findings.append(
                    SecurityFinding(
                        pattern="hardcoded_credential",
                        severity=Severity.CRITICAL,
                        line=i,
                        description="Possible hardcoded credential",
                    )
                )
                break  # One finding per line
    return findings


def detect_sensitive_paths(code: str) -> list[SecurityFinding]:
    """Detect hardcoded paths to sensitive system locations."""
    findings: list[SecurityFinding] = []
    for i, line in enumerate(code.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if _SENSITIVE_PATH_PATTERNS.search(line):
            findings.append(
                SecurityFinding(
                    pattern="sensitive_path",
                    severity=Severity.HIGH,
                    line=i,
                    description="Reference to sensitive system path",
                )
            )
    return findings


def detect_insecure_http(code: str) -> list[SecurityFinding]:
    """Detect non-localhost HTTP URLs (should use HTTPS)."""
    findings: list[SecurityFinding] = []
    for i, line in enumerate(code.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if _INSECURE_HTTP_PATTERN.search(line):
            findings.append(
                SecurityFinding(
                    pattern="insecure_http",
                    severity=Severity.MEDIUM,
                    line=i,
                    description="Insecure HTTP URL (use HTTPS)",
                )
            )
    return findings


# --- AST-based Detectors ---


def detect_sql_injection(tree: ast.Module) -> list[SecurityFinding]:
    """Detect string interpolation in SQL-like strings."""
    findings: list[SecurityFinding] = []
    sql_keywords = re.compile(
        r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC)\b",
        re.IGNORECASE,
    )

    for node in ast.walk(tree):
        # f-strings containing SQL keywords
        if isinstance(node, ast.JoinedStr):
            # Reconstruct a rough version to check for SQL keywords
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
            text = "".join(parts)
            if sql_keywords.search(text):
                findings.append(
                    SecurityFinding(
                        pattern="sql_injection",
                        severity=Severity.CRITICAL,
                        line=node.lineno,
                        description="SQL keyword in f-string (possible SQL injection)",
                    )
                )

        # str.format() or % formatting with SQL keywords
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "format"
                and isinstance(node.func.value, ast.Constant)
                and isinstance(node.func.value.value, str)
                and sql_keywords.search(node.func.value.value)
            ):
                findings.append(
                    SecurityFinding(
                        pattern="sql_injection",
                        severity=Severity.CRITICAL,
                        line=node.lineno,
                        description="SQL keyword in .format() call (possible SQL injection)",
                    )
                )

        # % string formatting with SQL
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            if (
                isinstance(node.left, ast.Constant)
                and isinstance(node.left.value, str)
                and sql_keywords.search(node.left.value)
            ):
                findings.append(
                    SecurityFinding(
                        pattern="sql_injection",
                        severity=Severity.CRITICAL,
                        line=node.lineno,
                        description="SQL keyword in %-format string (possible SQL injection)",
                    )
                )

    return findings


def detect_shell_injection(tree: ast.Module) -> list[SecurityFinding]:
    """Detect potentially unsafe subprocess/os.system calls."""
    findings: list[SecurityFinding] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func

        # os.system(...)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "system"
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
        ):
            findings.append(
                SecurityFinding(
                    pattern="shell_injection",
                    severity=Severity.CRITICAL,
                    line=node.lineno,
                    description="os.system() call (use subprocess with shell=False)",
                )
            )

        # subprocess.call/run/Popen with shell=True
        if isinstance(func, ast.Attribute) and func.attr in (
            "call",
            "run",
            "Popen",
            "check_output",
            "check_call",
        ):
            if isinstance(func.value, ast.Name) and func.value.id == "subprocess":
                for kw in node.keywords:
                    if (
                        kw.arg == "shell"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ):
                        findings.append(
                            SecurityFinding(
                                pattern="shell_injection",
                                severity=Severity.CRITICAL,
                                line=node.lineno,
                                description=f"subprocess.{func.attr}() with shell=True",
                            )
                        )
                        break

    return findings


def detect_eval_exec(tree: ast.Module) -> list[SecurityFinding]:
    """Detect eval() and exec() usage."""
    findings: list[SecurityFinding] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "eval":
                findings.append(
                    SecurityFinding(
                        pattern="eval_exec",
                        severity=Severity.HIGH,
                        line=node.lineno,
                        description="eval() usage (arbitrary code execution risk)",
                    )
                )
            elif node.func.id == "exec":
                findings.append(
                    SecurityFinding(
                        pattern="eval_exec",
                        severity=Severity.HIGH,
                        line=node.lineno,
                        description="exec() usage (arbitrary code execution risk)",
                    )
                )

    return findings


def detect_unsafe_deserialization(tree: ast.Module) -> list[SecurityFinding]:
    """Detect pickle/marshal imports that enable unsafe deserialization."""
    findings: list[SecurityFinding] = []
    unsafe_modules = {"pickle", "cPickle", "marshal", "shelve"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in unsafe_modules:
                    findings.append(
                        SecurityFinding(
                            pattern="unsafe_deserialization",
                            severity=Severity.HIGH,
                            line=node.lineno,
                            description=f"Import of {alias.name} (unsafe deserialization risk)",
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in unsafe_modules:
                findings.append(
                    SecurityFinding(
                        pattern="unsafe_deserialization",
                        severity=Severity.HIGH,
                        line=node.lineno,
                        description=f"Import from {node.module} (unsafe deserialization risk)",
                    )
                )

    return findings


# --- Scorer ---


class SecurityScorer:
    """Score code security via rule-based pattern matching and AST analysis.

    Deterministic, no LLM calls. Scoring:
    - Start at 1.0
    - Critical finding: -0.3
    - High finding: -0.2
    - Medium finding: -0.1
    - Floor at 0.0
    """

    def score(self, code: str, filename: str = "") -> DimensionScore:
        """Score code security. Returns 0.0-1.0 with finding details."""
        if not code or not code.strip():
            return DimensionScore(
                dimension=Dimension.SECURITY,
                score=0.0,
                method=ScoringMethod.RULE_BASED,
                details="Empty code",
            )

        report = SecurityReport()

        # Regex-based checks (work on raw text)
        report.findings.extend(detect_hardcoded_credentials(code))
        report.findings.extend(detect_sensitive_paths(code))
        report.findings.extend(detect_insecure_http(code))

        # AST-based checks (need valid syntax)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Can't run AST checks, just return regex findings
            return DimensionScore(
                dimension=Dimension.SECURITY,
                score=report.score,
                method=ScoringMethod.RULE_BASED,
                details=report.details_text,
            )

        report.findings.extend(detect_sql_injection(tree))
        report.findings.extend(detect_shell_injection(tree))
        report.findings.extend(detect_eval_exec(tree))
        report.findings.extend(detect_unsafe_deserialization(tree))

        return DimensionScore(
            dimension=Dimension.SECURITY,
            score=report.score,
            method=ScoringMethod.RULE_BASED,
            details=report.details_text,
            metadata={"finding_count": len(report.findings)},
        )
