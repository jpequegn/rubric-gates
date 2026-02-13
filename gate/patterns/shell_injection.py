"""Shell injection detector.

Detects os.system() with variable interpolation, subprocess calls with
shell=True and variable input, and other unsafe shell execution patterns.
"""

from __future__ import annotations

import ast

from shared.models import PatternFinding

from .base import register_detector


class ShellInjectionDetector:
    """Detects potential shell injection vulnerabilities."""

    name = "shell_injection"
    severity = "critical"

    def detect(self, code: str, filename: str) -> list[PatternFinding]:
        findings: list[PatternFinding] = []
        lines = code.splitlines()

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return findings

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            finding = self._check_call(node, lines)
            if finding:
                findings.append(finding)

        return findings

    def _check_call(self, node: ast.Call, lines: list[str]) -> PatternFinding | None:
        line_num = getattr(node, "lineno", 0)
        line_content = lines[line_num - 1].strip() if line_num > 0 else ""

        func = node.func

        # Check os.system() with any dynamic content
        if self._is_attr_call(func, "os", "system"):
            if node.args and self._is_dynamic(node.args[0]):
                return PatternFinding(
                    pattern=self.name,
                    severity=self.severity,
                    line_number=line_num,
                    line_content=line_content,
                    description="os.system() with dynamic input",
                    remediation="Use subprocess.run() with a list of arguments instead. "
                    "Example: subprocess.run(['ls', '-l', path], check=True)",
                )

        # Check subprocess calls with shell=True
        if self._is_subprocess_call(func):
            has_shell_true = any(
                isinstance(kw.value, ast.Constant) and kw.value.value is True and kw.arg == "shell"
                for kw in node.keywords
            )
            if has_shell_true and node.args and self._is_dynamic(node.args[0]):
                return PatternFinding(
                    pattern=self.name,
                    severity=self.severity,
                    line_number=line_num,
                    line_content=line_content,
                    description="subprocess call with shell=True and dynamic input",
                    remediation="Remove shell=True and pass command as a list. "
                    "Example: subprocess.run(['cmd', arg1, arg2], check=True)",
                )

        # Check os.popen()
        if self._is_attr_call(func, "os", "popen"):
            if node.args and self._is_dynamic(node.args[0]):
                return PatternFinding(
                    pattern=self.name,
                    severity=self.severity,
                    line_number=line_num,
                    line_content=line_content,
                    description="os.popen() with dynamic input",
                    remediation="Use subprocess.run() with a list of arguments instead.",
                )

        return None

    @staticmethod
    def _is_attr_call(func: ast.expr, module: str, attr: str) -> bool:
        """Check if func is module.attr (e.g., os.system)."""
        return (
            isinstance(func, ast.Attribute)
            and func.attr == attr
            and isinstance(func.value, ast.Name)
            and func.value.id == module
        )

    @staticmethod
    def _is_subprocess_call(func: ast.expr) -> bool:
        """Check if func is subprocess.run/call/Popen/check_output."""
        return (
            isinstance(func, ast.Attribute)
            and func.attr in ("run", "call", "Popen", "check_output", "check_call")
            and isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
        )

    @staticmethod
    def _is_dynamic(node: ast.expr) -> bool:
        """Check if an AST node contains dynamic (non-literal) content."""
        if isinstance(node, ast.Constant):
            return False
        if isinstance(node, ast.JoinedStr):
            # f-string — dynamic if it has FormattedValue nodes
            return any(isinstance(v, ast.FormattedValue) for v in node.values)
        # BinOp (concatenation), Name, Call, etc. are all dynamic
        return True


register_detector(ShellInjectionDetector())
