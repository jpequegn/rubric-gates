"""Maintainability dimension scorer.

Measures whether generated code will be understandable and modifiable
later — cyclomatic complexity, function/file length, naming quality,
and nesting depth. All static analysis, no LLM calls.
"""

import ast
import re
from dataclasses import dataclass, field

from shared.models import Dimension, DimensionScore, ScoringMethod

# --- Defaults (overridable via config) ---

DEFAULT_COMPLEXITY_THRESHOLD = 10
DEFAULT_FUNCTION_LENGTH_THRESHOLD = 50
DEFAULT_FILE_LENGTH_THRESHOLD = 300
DEFAULT_NESTING_DEPTH_THRESHOLD = 4


# --- Check Results ---


@dataclass
class MaintainabilityCheck:
    """Result of a single maintainability check."""

    name: str
    score: float  # 0.0-1.0 for this check
    max_weight: float
    details: str


@dataclass
class MaintainabilityReport:
    """Aggregated maintainability results."""

    checks: list[MaintainabilityCheck] = field(default_factory=list)

    @property
    def total_score(self) -> float:
        total_weight = sum(c.max_weight for c in self.checks)
        if total_weight == 0:
            return 0.0
        weighted = sum(c.score * c.max_weight for c in self.checks)
        return min(weighted / total_weight, 1.0)

    @property
    def details_text(self) -> str:
        lines = []
        for c in self.checks:
            pct = int(c.score * 100)
            lines.append(f"[{pct}%] {c.name}: {c.details}")
        return "; ".join(lines)


# --- Cyclomatic Complexity ---


def _count_branches(node: ast.AST) -> int:
    """Count decision points in a function body (cyclomatic complexity).

    Counts: if, elif, for, while, except, with, and, or, assert,
    ternary (IfExp), comprehension conditions.
    Starts at 1 (base path).
    """
    complexity = 1

    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.IfExp)):
            complexity += 1
        elif isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
            complexity += 1
        elif isinstance(child, ast.ExceptHandler):
            complexity += 1
        elif isinstance(child, (ast.With, ast.AsyncWith)):
            complexity += 1
        elif isinstance(child, ast.Assert):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            # Each 'and'/'or' adds a path
            complexity += len(child.values) - 1
        elif isinstance(child, ast.comprehension):
            # Each 'if' clause in a comprehension
            complexity += len(child.ifs)

    return complexity


def check_cyclomatic_complexity(
    tree: ast.Module,
    threshold: int = DEFAULT_COMPLEXITY_THRESHOLD,
) -> MaintainabilityCheck:
    """Check cyclomatic complexity of each function."""
    high_complexity: list[str] = []
    max_complexity = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cc = _count_branches(node)
            max_complexity = max(max_complexity, cc)
            if cc > threshold:
                high_complexity.append(f"{node.name}()={cc}")

    if not high_complexity:
        if max_complexity == 0:
            detail = "No functions to check"
        else:
            detail = f"All functions below threshold ({threshold}), max={max_complexity}"
        return MaintainabilityCheck(
            name="complexity",
            score=1.0,
            max_weight=0.25,
            details=detail,
        )

    # Degrade score based on how many functions exceed threshold
    # and by how much the worst exceeds it
    worst_ratio = max_complexity / threshold
    penalty = min(1.0, (worst_ratio - 1.0) * 0.5 + len(high_complexity) * 0.1)
    score = max(0.0, 1.0 - penalty)

    return MaintainabilityCheck(
        name="complexity",
        score=score,
        max_weight=0.25,
        details=f"High complexity: {', '.join(high_complexity[:5])}",
    )


# --- Function Length ---


def check_function_length(
    tree: ast.Module,
    threshold: int = DEFAULT_FUNCTION_LENGTH_THRESHOLD,
) -> MaintainabilityCheck:
    """Check that functions aren't too long."""
    long_functions: list[str] = []
    max_length = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Calculate function length from first to last line
            end_lineno = _last_line(node)
            length = end_lineno - node.lineno + 1
            max_length = max(max_length, length)
            if length > threshold:
                long_functions.append(f"{node.name}()={length}L")

    if not long_functions:
        if max_length == 0:
            detail = "No functions to check"
        else:
            detail = f"All functions below {threshold} lines, max={max_length}L"
        return MaintainabilityCheck(
            name="function_length",
            score=1.0,
            max_weight=0.20,
            details=detail,
        )

    worst_ratio = max_length / threshold
    penalty = min(1.0, (worst_ratio - 1.0) * 0.3 + len(long_functions) * 0.15)
    score = max(0.0, 1.0 - penalty)

    return MaintainabilityCheck(
        name="function_length",
        score=score,
        max_weight=0.20,
        details=f"Long functions: {', '.join(long_functions[:5])}",
    )


def _last_line(node: ast.AST) -> int:
    """Get the last line number of an AST node (recursive)."""
    max_line = getattr(node, "lineno", 0)
    end = getattr(node, "end_lineno", None)
    if end is not None:
        max_line = max(max_line, end)
    for child in ast.iter_child_nodes(node):
        max_line = max(max_line, _last_line(child))
    return max_line


# --- File Length ---


def check_file_length(
    code: str,
    threshold: int = DEFAULT_FILE_LENGTH_THRESHOLD,
) -> MaintainabilityCheck:
    """Check that the file isn't too long."""
    lines = code.splitlines()
    # Count non-blank, non-comment lines
    effective = sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))
    total = len(lines)

    if total <= threshold:
        return MaintainabilityCheck(
            name="file_length",
            score=1.0,
            max_weight=0.15,
            details=f"{total} lines ({effective} effective), within {threshold} limit",
        )

    ratio = total / threshold
    penalty = min(1.0, (ratio - 1.0) * 0.5)
    score = max(0.0, 1.0 - penalty)

    return MaintainabilityCheck(
        name="file_length",
        score=score,
        max_weight=0.15,
        details=f"{total} lines ({effective} effective), exceeds {threshold} limit",
    )


# --- Naming Quality ---

_SNAKE_CASE = re.compile(r"^[a-z_][a-z0-9_]*$")
_UPPER_CASE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_PASCAL_CASE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
_DUNDER = re.compile(r"^__[a-z]+__$")

# Single-char names that are OK
_ACCEPTABLE_SHORT = {"i", "j", "k", "n", "x", "y", "z", "e", "f", "_"}


def check_naming_quality(tree: ast.Module) -> MaintainabilityCheck:
    """Check that names follow Python conventions and are descriptive."""
    total_names = 0
    violations: list[str] = []

    for node in ast.walk(tree):
        # Function names should be snake_case
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            total_names += 1
            name = node.name
            if not (_SNAKE_CASE.match(name) or _DUNDER.match(name)):
                violations.append(f"{name}() not snake_case (line {node.lineno})")

        # Class names should be PascalCase
        elif isinstance(node, ast.ClassDef):
            total_names += 1
            if not _PASCAL_CASE.match(node.name):
                violations.append(f"class {node.name} not PascalCase (line {node.lineno})")

        # Variable assignments — check for non-conventional names
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    total_names += 1
                    name = target.id
                    if name.startswith("_"):
                        continue  # Private names are fine
                    if not (
                        _SNAKE_CASE.match(name)
                        or _UPPER_CASE.match(name)
                        or _PASCAL_CASE.match(name)
                    ):
                        violations.append(f"'{name}' unconventional naming (line {node.lineno})")

    if total_names == 0:
        return MaintainabilityCheck(
            name="naming",
            score=1.0,
            max_weight=0.20,
            details="No names to check",
        )

    # Also check for overly short names in function defs
    short_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if len(node.name) <= 1 and node.name not in _ACCEPTABLE_SHORT:
                short_names.append(f"{node.name}() too short (line {node.lineno})")

    all_issues = violations + short_names

    if not all_issues:
        return MaintainabilityCheck(
            name="naming",
            score=1.0,
            max_weight=0.20,
            details=f"All {total_names} names follow conventions",
        )

    violation_rate = len(all_issues) / max(total_names, 1)
    score = max(0.0, 1.0 - violation_rate)

    shown = all_issues[:5]
    extra = f" (+{len(all_issues) - 5} more)" if len(all_issues) > 5 else ""
    return MaintainabilityCheck(
        name="naming",
        score=score,
        max_weight=0.20,
        details=f"{'; '.join(shown)}{extra}",
    )


# --- Nesting Depth ---


def check_nesting_depth(
    tree: ast.Module,
    threshold: int = DEFAULT_NESTING_DEPTH_THRESHOLD,
) -> MaintainabilityCheck:
    """Check maximum nesting depth in functions."""
    deep_spots: list[str] = []
    max_depth = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            depth = _max_nesting(node.body, current_depth=0)
            max_depth = max(max_depth, depth)
            if depth > threshold:
                deep_spots.append(f"{node.name}()={depth}")

    if not deep_spots:
        if max_depth == 0:
            detail = "No functions to check"
        else:
            detail = f"Max nesting depth {max_depth}, within {threshold} limit"
        return MaintainabilityCheck(
            name="nesting_depth",
            score=1.0,
            max_weight=0.20,
            details=detail,
        )

    worst_ratio = max_depth / threshold
    penalty = min(1.0, (worst_ratio - 1.0) * 0.5 + len(deep_spots) * 0.1)
    score = max(0.0, 1.0 - penalty)

    return MaintainabilityCheck(
        name="nesting_depth",
        score=score,
        max_weight=0.20,
        details=f"Deep nesting: {', '.join(deep_spots[:5])}",
    )


def _max_nesting(body: list[ast.stmt], current_depth: int) -> int:
    """Recursively calculate maximum nesting depth."""
    max_depth = current_depth
    nesting_nodes = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.With,
        ast.AsyncWith,
        ast.Try,
    )

    for stmt in body:
        if isinstance(stmt, nesting_nodes):
            # Check all sub-bodies
            for attr in ("body", "orelse", "finalbody", "handlers"):
                sub = getattr(stmt, attr, None)
                if sub:
                    if attr == "handlers":
                        for handler in sub:
                            d = _max_nesting(handler.body, current_depth + 1)
                            max_depth = max(max_depth, d)
                    else:
                        d = _max_nesting(sub, current_depth + 1)
                        max_depth = max(max_depth, d)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Nested function — don't count its depth against the parent
            pass

    return max_depth


# --- Scorer ---


class MaintainabilityScorer:
    """Score code maintainability via static analysis.

    Deterministic, no LLM calls. Checks:
    - Cyclomatic complexity (0.25)
    - Function length (0.20)
    - File length (0.15)
    - Naming quality (0.20)
    - Nesting depth (0.20)
    """

    def __init__(
        self,
        complexity_threshold: int = DEFAULT_COMPLEXITY_THRESHOLD,
        function_length_threshold: int = DEFAULT_FUNCTION_LENGTH_THRESHOLD,
        file_length_threshold: int = DEFAULT_FILE_LENGTH_THRESHOLD,
        nesting_depth_threshold: int = DEFAULT_NESTING_DEPTH_THRESHOLD,
    ):
        self.complexity_threshold = complexity_threshold
        self.function_length_threshold = function_length_threshold
        self.file_length_threshold = file_length_threshold
        self.nesting_depth_threshold = nesting_depth_threshold

    def score(self, code: str, filename: str = "") -> DimensionScore:
        """Score code maintainability. Returns 0.0-1.0."""
        if not code or not code.strip():
            return DimensionScore(
                dimension=Dimension.MAINTAINABILITY,
                score=0.0,
                method=ScoringMethod.RULE_BASED,
                details="Empty code",
            )

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return DimensionScore(
                dimension=Dimension.MAINTAINABILITY,
                score=0.0,
                method=ScoringMethod.RULE_BASED,
                details="SyntaxError — cannot analyze maintainability",
            )

        report = MaintainabilityReport()
        report.checks.append(check_cyclomatic_complexity(tree, self.complexity_threshold))
        report.checks.append(check_function_length(tree, self.function_length_threshold))
        report.checks.append(check_file_length(code, self.file_length_threshold))
        report.checks.append(check_naming_quality(tree))
        report.checks.append(check_nesting_depth(tree, self.nesting_depth_threshold))

        return DimensionScore(
            dimension=Dimension.MAINTAINABILITY,
            score=report.total_score,
            method=ScoringMethod.RULE_BASED,
            details=report.details_text,
        )
