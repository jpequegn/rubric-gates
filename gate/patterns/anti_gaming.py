"""Anti-gaming measures for rubric scores.

Detects patterns where AI or users optimize for passing the rubric
without actually improving code quality. Each checker produces findings
with score adjustments applied before tier evaluation.

Gaming patterns detected:
1. Comment stuffing — meaningless or tautological comments
2. Try/catch wrapping — bare except with pass
3. Trivial file splitting — (checked at aggregate level, not here)
4. Test stub padding — test functions with no assertions
5. Length inflation — verbose code relative to purpose
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from shared.models import Dimension, ScoreResult


@dataclass
class GamingFinding:
    """A detected score gaming pattern."""

    pattern: str
    description: str
    score_adjustment: float  # Negative value to subtract
    dimension_affected: str  # Dimension to adjust
    line_number: int = 0


class AntiGamingChecker:
    """Detects potential score gaming and returns adjustment findings."""

    def check(
        self,
        code: str,
        score_result: ScoreResult | None = None,
        test_code: str = "",
    ) -> list[GamingFinding]:
        """Detect potential score gaming.

        Args:
            code: The source code to analyze.
            score_result: The scoring result (for context).
            test_code: Associated test code (for test stub detection).

        Returns:
            List of gaming findings with score adjustments.
        """
        findings: list[GamingFinding] = []

        findings.extend(self._check_comment_stuffing(code))
        findings.extend(self._check_try_catch_wrapping(code))
        findings.extend(self._check_test_stub_padding(test_code))
        findings.extend(self._check_length_inflation(code))

        return findings

    def apply_adjustments(
        self,
        score_result: ScoreResult,
        findings: list[GamingFinding],
    ) -> ScoreResult:
        """Apply gaming adjustments to a score result.

        Returns a new ScoreResult with adjusted dimension scores.
        Does not modify the original.
        """
        if not findings:
            return score_result

        # Build adjustment map: dimension -> total adjustment
        adjustments: dict[str, float] = {}
        for finding in findings:
            dim = finding.dimension_affected
            adjustments[dim] = adjustments.get(dim, 0.0) + finding.score_adjustment

        # Apply adjustments to dimension scores
        new_dims = []
        for ds in score_result.dimension_scores:
            adj = adjustments.get(ds.dimension.value, 0.0)
            new_score = max(0.0, min(1.0, ds.score + adj))
            new_ds = ds.model_copy(update={"score": new_score})
            new_dims.append(new_ds)

        # Recalculate composite (simple average for now)
        if new_dims:
            new_composite = sum(d.score for d in new_dims) / len(new_dims)
        else:
            new_composite = score_result.composite_score

        new_composite = max(0.0, min(1.0, new_composite))

        return score_result.model_copy(
            update={
                "dimension_scores": new_dims,
                "composite_score": new_composite,
            }
        )

    # --- Comment Stuffing ---

    def _check_comment_stuffing(self, code: str) -> list[GamingFinding]:
        """Detect meaningless or tautological comments."""
        findings: list[GamingFinding] = []
        lines = code.splitlines()
        tautological_count = 0
        total_comments = 0

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # Skip docstrings and non-comment lines
            if not stripped.startswith("#"):
                continue

            comment_text = stripped.lstrip("#").strip().lower()
            if not comment_text:
                continue

            total_comments += 1

            # Check for tautological comments (comment just describes the code)
            if self._is_tautological(comment_text, lines, line_num):
                tautological_count += 1

            # Check for generic filler comments
            if self._is_filler(comment_text):
                tautological_count += 1

        # Only flag if a significant portion of comments are stuffed
        if total_comments >= 3 and tautological_count / total_comments > 0.5:
            findings.append(
                GamingFinding(
                    pattern="comment_stuffing",
                    description=(
                        f"{tautological_count}/{total_comments} comments appear "
                        "tautological or generic. Comments should add context "
                        "beyond what the code already says."
                    ),
                    score_adjustment=-0.15,
                    dimension_affected=Dimension.DOCUMENTATION.value,
                )
            )

        return findings

    @staticmethod
    def _is_tautological(comment: str, lines: list[str], line_num: int) -> bool:
        """Check if a comment is a tautological restatement of adjacent code."""
        # Look at the next non-empty line
        for i in range(line_num, min(line_num + 2, len(lines))):
            next_line = lines[i].strip().lower()
            if not next_line or next_line.startswith("#"):
                continue

            # Extract identifiers from the code line
            code_tokens = set(re.findall(r"[a-z_][a-z0-9_]*", next_line))
            comment_tokens = set(re.findall(r"[a-z_][a-z0-9_]*", comment))

            # If the comment is just the code tokens restated
            if code_tokens and comment_tokens:
                overlap = code_tokens & comment_tokens
                if len(overlap) >= len(comment_tokens) * 0.7 and len(comment_tokens) >= 2:
                    return True

            break

        return False

    @staticmethod
    def _is_filler(comment: str) -> bool:
        """Check if a comment is generic filler."""
        filler_patterns = [
            r"^do the thing$",
            r"^process data$",
            r"^handle (?:the )?(?:data|input|output|result)$",
            r"^set \w+ to \w+$",
            r"^initialize \w+$",
            r"^create \w+$",
            r"^get \w+$",
            r"^return \w+$",
            r"^call \w+$",
            r"^update \w+$",
            r"^define \w+$",
            r"^assign \w+$",
            r"^import \w+$",
        ]
        for pattern in filler_patterns:
            if re.match(pattern, comment):
                return True
        return False

    # --- Try/Catch Wrapping ---

    def _check_try_catch_wrapping(self, code: str) -> list[GamingFinding]:
        """Detect bare except clauses with pass (exception swallowing)."""
        findings: list[GamingFinding] = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return findings

        swallowed = 0
        total_handlers = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                total_handlers += 1
                if self._is_swallowed(node):
                    swallowed += 1

        if swallowed >= 2:
            findings.append(
                GamingFinding(
                    pattern="try_catch_wrapping",
                    description=(
                        f"{swallowed} exception handlers just use 'pass' or do nothing. "
                        "Exceptions should be logged, re-raised, or handled specifically."
                    ),
                    score_adjustment=-0.1,
                    dimension_affected=Dimension.CORRECTNESS.value,
                )
            )

        return findings

    @staticmethod
    def _is_swallowed(handler: ast.ExceptHandler) -> bool:
        """Check if an exception handler swallows the exception."""
        body = handler.body
        if len(body) == 1:
            stmt = body[0]
            # `except: pass`
            if isinstance(stmt, ast.Pass):
                return True
            # `except: ...` (Ellipsis)
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if stmt.value.value is ...:
                    return True
        return False

    # --- Test Stub Padding ---

    def _check_test_stub_padding(self, test_code: str) -> list[GamingFinding]:
        """Detect test functions with no meaningful assertions."""
        if not test_code.strip():
            return []

        findings: list[GamingFinding] = []

        try:
            tree = ast.parse(test_code)
        except SyntaxError:
            return findings

        total_tests = 0
        empty_tests = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    total_tests += 1
                    if not self._has_assertion(node):
                        empty_tests += 1

        if total_tests >= 2 and empty_tests / total_tests > 0.5:
            findings.append(
                GamingFinding(
                    pattern="test_stub_padding",
                    description=(
                        f"{empty_tests}/{total_tests} test functions lack assertions. "
                        "Tests should verify behavior with assert statements."
                    ),
                    score_adjustment=-0.15,
                    dimension_affected=Dimension.TESTABILITY.value,
                )
            )

        return findings

    @staticmethod
    def _has_assertion(func_node: ast.FunctionDef) -> bool:
        """Check if a function contains any assertion."""
        for node in ast.walk(func_node):
            # assert statements
            if isinstance(node, ast.Assert):
                return True
            # pytest-style: self.assert*, self.assertEqual, etc.
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    if func.attr.startswith("assert"):
                        return True
                # pytest.raises context manager
                if isinstance(func, ast.Attribute) and func.attr == "raises":
                    return True
        return False

    # --- Length Inflation ---

    def _check_length_inflation(self, code: str) -> list[GamingFinding]:
        """Detect verbose code that could be significantly shorter."""
        findings: list[GamingFinding] = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return findings

        lines = code.splitlines()
        total_lines = len(lines)
        if total_lines < 50:
            return findings

        # Count functional elements vs boilerplate
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node)

        if not functions:
            return findings

        # Check for excessively verbose functions (lots of lines, few statements)
        verbose_functions = 0
        for func in functions:
            func_lines = (func.end_lineno or func.lineno) - func.lineno + 1
            stmt_count = sum(1 for _ in ast.walk(func) if isinstance(_, ast.stmt))
            if func_lines > 20 and stmt_count > 0:
                ratio = func_lines / stmt_count
                if ratio > 5:  # More than 5 lines per statement = very verbose
                    verbose_functions += 1

        if verbose_functions >= 2:
            findings.append(
                GamingFinding(
                    pattern="length_inflation",
                    description=(
                        f"{verbose_functions} functions appear excessively verbose "
                        "(high line-to-statement ratio). Consider more concise code."
                    ),
                    score_adjustment=-0.1,
                    dimension_affected=Dimension.MAINTAINABILITY.value,
                )
            )

        return findings
