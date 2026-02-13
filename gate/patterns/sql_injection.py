"""SQL injection detector.

Detects string interpolation and concatenation inside SQL-like strings,
including f-strings, .format(), and % formatting with SQL keywords.
"""

from __future__ import annotations

import ast
import re

from shared.models import PatternFinding

from .base import register_detector

_SQL_KEYWORDS = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC|EXECUTE|UNION)\b",
    re.IGNORECASE,
)

# Patterns for string formatting in SQL context
_FORMAT_CALL = re.compile(r"""\.format\s*\(""")
_PERCENT_FORMAT = re.compile(r"""%\s*[sd]""")


class SQLInjectionDetector:
    """Detects potential SQL injection vulnerabilities."""

    name = "sql_injection"
    severity = "critical"

    def detect(self, code: str, filename: str) -> list[PatternFinding]:
        findings: list[PatternFinding] = []
        lines = code.splitlines()

        # AST-based detection for f-strings in SQL context
        try:
            tree = ast.parse(code)
        except SyntaxError:
            tree = None

        if tree:
            findings.extend(self._check_ast(tree, lines))

        # Line-based detection for .format() and % formatting
        findings.extend(self._check_lines(lines))

        return findings

    def _check_ast(self, tree: ast.Module, lines: list[str]) -> list[PatternFinding]:
        findings: list[PatternFinding] = []

        for node in ast.walk(tree):
            # Check f-strings (JoinedStr) that contain SQL keywords
            if isinstance(node, ast.JoinedStr):
                # Reconstruct the static parts to check for SQL keywords
                static_parts = []
                has_expression = False
                for value in node.values:
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        static_parts.append(value.value)
                    elif isinstance(value, ast.FormattedValue):
                        has_expression = True

                full_static = " ".join(static_parts)
                if has_expression and _SQL_KEYWORDS.search(full_static):
                    line_num = getattr(node, "lineno", 0)
                    line_content = lines[line_num - 1].strip() if line_num > 0 else ""
                    findings.append(
                        PatternFinding(
                            pattern=self.name,
                            severity=self.severity,
                            line_number=line_num,
                            line_content=line_content,
                            description="f-string with SQL keywords and interpolated values",
                            remediation="Use parameterized queries instead of f-strings. "
                            "Example: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
                        )
                    )

            # Check cursor.execute() with string concatenation
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "execute" and node.args:
                    arg = node.args[0]
                    # Check if the first arg is a BinOp (string concatenation)
                    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
                        line_num = getattr(node, "lineno", 0)
                        line_content = lines[line_num - 1].strip() if line_num > 0 else ""
                        findings.append(
                            PatternFinding(
                                pattern=self.name,
                                severity=self.severity,
                                line_number=line_num,
                                line_content=line_content,
                                description="String concatenation in SQL execute() call",
                                remediation="Use parameterized queries: "
                                "cursor.execute('SELECT * FROM t WHERE id = ?', (val,))",
                            )
                        )

        return findings

    def _check_lines(self, lines: list[str]) -> list[PatternFinding]:
        findings: list[PatternFinding] = []

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Check .format() with SQL keywords
            if _FORMAT_CALL.search(line) and _SQL_KEYWORDS.search(line):
                findings.append(
                    PatternFinding(
                        pattern=self.name,
                        severity=self.severity,
                        line_number=line_num,
                        line_content=stripped,
                        description=".format() used in SQL string",
                        remediation="Use parameterized queries instead of .format(). "
                        "Example: cursor.execute('SELECT * FROM users WHERE name = %s', (name,))",
                    )
                )

            # Check % formatting with SQL keywords
            if _PERCENT_FORMAT.search(line) and _SQL_KEYWORDS.search(line):
                # Exclude parameterized-style %s in execute() calls
                if ".execute(" not in line:
                    findings.append(
                        PatternFinding(
                            pattern=self.name,
                            severity="high",
                            line_number=line_num,
                            line_content=stripped,
                            description="%-formatting used in SQL string",
                            remediation="Use parameterized queries instead of % formatting.",
                        )
                    )

        return findings


register_detector(SQLInjectionDetector())
