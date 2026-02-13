"""Unsafe file operations detector.

Detects unrestricted path traversal, writing to sensitive system paths,
and reading from user-controlled paths without sanitization.
"""

from __future__ import annotations

import ast
import re

from shared.models import PatternFinding

from .base import register_detector

# Sensitive system paths
_SENSITIVE_PATHS = re.compile(
    r"""(?:/etc/(?:passwd|shadow|hosts|sudoers)|"""
    r"""/proc/|/sys/|/dev/|"""
    r"""~/.ssh/|\.env|\.git/config)""",
)

# Path traversal patterns
_TRAVERSAL = re.compile(r"""\.\./""")


class FileOpsDetector:
    """Detects unsafe file operation patterns."""

    name = "unsafe_file_ops"
    severity = "high"

    def detect(self, code: str, filename: str) -> list[PatternFinding]:
        findings: list[PatternFinding] = []
        lines = code.splitlines()

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return findings

        # Check for open() and Path operations with dynamic paths
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                finding = self._check_call(node, lines)
                if finding:
                    findings.append(finding)

        # Line-based: check for sensitive paths and traversal
        findings.extend(self._check_lines(lines))

        return findings

    def _check_call(self, node: ast.Call, lines: list[str]) -> PatternFinding | None:
        line_num = getattr(node, "lineno", 0)
        line_content = lines[line_num - 1].strip() if line_num > 0 else ""

        func = node.func

        # Check open() with user-controlled path (f-string or concatenation)
        if isinstance(func, ast.Name) and func.id == "open":
            if node.args and self._is_user_path(node.args[0]):
                return PatternFinding(
                    pattern=self.name,
                    severity=self.severity,
                    line_number=line_num,
                    line_content=line_content,
                    description="open() with dynamically constructed path",
                    remediation="Validate the path before opening. Use pathlib to resolve "
                    "and check it stays within an allowed directory: "
                    "resolved.relative_to(base_dir)",
                )

        # Check Path().write_text/write_bytes with dynamic path
        if isinstance(func, ast.Attribute) and func.attr in (
            "write_text",
            "write_bytes",
            "unlink",
            "rmdir",
        ):
            # Check if the path object is constructed dynamically
            if isinstance(func.value, ast.Call):
                inner = func.value
                if (
                    isinstance(inner.func, ast.Name)
                    and inner.func.id == "Path"
                    and inner.args
                    and self._is_user_path(inner.args[0])
                ):
                    return PatternFinding(
                        pattern=self.name,
                        severity=self.severity,
                        line_number=line_num,
                        line_content=line_content,
                        description=f"Path.{func.attr}() with dynamically constructed path",
                        remediation="Resolve the path and verify it stays within "
                        "the allowed base directory before writing.",
                    )

        return None

    def _check_lines(self, lines: list[str]) -> list[PatternFinding]:
        findings: list[PatternFinding] = []

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Check for path traversal in string literals
            if _TRAVERSAL.search(line):
                # Only flag if it's in a string context (quotes present)
                if '"../' in line or "'../" in line:
                    findings.append(
                        PatternFinding(
                            pattern=self.name,
                            severity=self.severity,
                            line_number=line_num,
                            line_content=stripped,
                            description="Path traversal pattern in string literal",
                            remediation="Use pathlib.Path.resolve() and verify the resolved "
                            "path is within the expected directory.",
                        )
                    )

            # Check for access to sensitive system paths
            if _SENSITIVE_PATHS.search(line):
                # Skip comments and docstrings
                if not stripped.startswith(("#", '"""', "'''")):
                    findings.append(
                        PatternFinding(
                            pattern=self.name,
                            severity="high",
                            line_number=line_num,
                            line_content=stripped,
                            description="Access to sensitive system path",
                            remediation="Avoid hardcoding sensitive system paths. "
                            "If access is necessary, use proper permissions checks.",
                        )
                    )

        return findings

    @staticmethod
    def _is_user_path(node: ast.expr) -> bool:
        """Check if a path argument looks user-controlled."""
        # f-strings are likely user-controlled
        if isinstance(node, ast.JoinedStr):
            return any(isinstance(v, ast.FormattedValue) for v in node.values)
        # String concatenation
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return True
        # Variable reference (not a literal string)
        if isinstance(node, ast.Name):
            return True
        return False


register_detector(FileOpsDetector())
