"""Correctness dimension scorer.

Checks whether generated code is syntactically valid and free of obvious
structural errors using pure AST analysis — no LLM calls, fully deterministic.
"""

import ast
from dataclasses import dataclass, field

from shared.models import Dimension, DimensionScore, ScoringMethod

# --- AST Checks ---

PYTHON_BUILTINS = {
    "abs",
    "all",
    "any",
    "ascii",
    "bin",
    "bool",
    "breakpoint",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "classmethod",
    "compile",
    "complex",
    "copyright",
    "credits",
    "delattr",
    "dict",
    "dir",
    "divmod",
    "enumerate",
    "eval",
    "exec",
    "exit",
    "filter",
    "float",
    "format",
    "frozenset",
    "getattr",
    "globals",
    "hasattr",
    "hash",
    "help",
    "hex",
    "id",
    "input",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "license",
    "list",
    "locals",
    "map",
    "max",
    "memoryview",
    "min",
    "next",
    "object",
    "oct",
    "open",
    "ord",
    "pow",
    "print",
    "property",
    "quit",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "setattr",
    "slice",
    "sorted",
    "staticmethod",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "vars",
    "zip",
    "__name__",
    "__file__",
    "__doc__",
    "__builtins__",
    "__import__",
    "True",
    "False",
    "None",
    "NotImplemented",
    "Ellipsis",
    "__all__",
    "__annotations__",
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "BaseException",
    "BlockingIOError",
    "BrokenPipeError",
    "BufferError",
    "BytesWarning",
    "ChildProcessError",
    "ConnectionAbortedError",
    "ConnectionError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "DeprecationWarning",
    "EOFError",
    "EnvironmentError",
    "Exception",
    "FileExistsError",
    "FileNotFoundError",
    "FloatingPointError",
    "FutureWarning",
    "GeneratorExit",
    "IOError",
    "ImportError",
    "IndentationError",
    "IndexError",
    "InterruptedError",
    "IsADirectoryError",
    "KeyError",
    "KeyboardInterrupt",
    "LookupError",
    "MemoryError",
    "ModuleNotFoundError",
    "NameError",
    "NotADirectoryError",
    "NotImplementedError",
    "OSError",
    "OverflowError",
    "PendingDeprecationWarning",
    "PermissionError",
    "ProcessLookupError",
    "RecursionError",
    "ReferenceError",
    "ResourceWarning",
    "RuntimeError",
    "RuntimeWarning",
    "StopAsyncIteration",
    "StopIteration",
    "SyntaxError",
    "SyntaxWarning",
    "SystemError",
    "SystemExit",
    "TabError",
    "TimeoutError",
    "TypeError",
    "UnboundLocalError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "UnicodeError",
    "UnicodeTranslationError",
    "UnicodeWarning",
    "UserWarning",
    "ValueError",
    "Warning",
    "WindowsError",
    "ZeroDivisionError",
}


@dataclass
class CheckResult:
    """Result of a single correctness check."""

    name: str
    passed: bool
    max_points: float
    earned_points: float
    details: str = ""


@dataclass
class CorrectnessReport:
    """Aggregated correctness check results."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def total_score(self) -> float:
        total_possible = sum(c.max_points for c in self.checks)
        if total_possible == 0:
            return 0.0
        earned = sum(c.earned_points for c in self.checks)
        return min(earned / total_possible, 1.0)

    @property
    def details_text(self) -> str:
        lines = []
        for c in self.checks:
            status = "pass" if c.passed else "FAIL"
            lines.append(f"[{status}] {c.name}: {c.details}")
        return "; ".join(lines)


def check_syntax(code: str) -> tuple[CheckResult, ast.Module | None]:
    """Check if code parses to a valid AST."""
    try:
        tree = ast.parse(code)
        return (
            CheckResult(
                name="syntax",
                passed=True,
                max_points=0.5,
                earned_points=0.5,
                details="Valid syntax",
            ),
            tree,
        )
    except SyntaxError as e:
        msg = f"SyntaxError at line {e.lineno}: {e.msg}" if e.lineno else f"SyntaxError: {e.msg}"
        return (
            CheckResult(
                name="syntax",
                passed=False,
                max_points=0.5,
                earned_points=0.0,
                details=msg,
            ),
            None,
        )


def check_bare_excepts(tree: ast.Module) -> CheckResult:
    """Check for bare except clauses and empty except bodies."""
    findings: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            # Bare except (no exception type)
            if node.type is None:
                findings.append(f"Bare except at line {node.lineno}")
            # Empty except body (just pass or ...)
            if len(node.body) == 1:
                stmt = node.body[0]
                if isinstance(stmt, ast.Pass):
                    findings.append(f"Empty except (pass) at line {node.lineno}")
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    if stmt.value.value is ...:
                        findings.append(f"Empty except (...) at line {node.lineno}")

    if findings:
        return CheckResult(
            name="exception_handling",
            passed=False,
            max_points=0.2,
            earned_points=0.0,
            details="; ".join(findings),
        )
    return CheckResult(
        name="exception_handling",
        passed=True,
        max_points=0.2,
        earned_points=0.2,
        details="No bare/empty excepts",
    )


def check_unreachable_code(tree: ast.Module) -> CheckResult:
    """Check for code after return/raise/break/continue statements."""
    findings: list[str] = []

    for node in ast.walk(tree):
        # Check function/method bodies and loop bodies
        body: list[ast.stmt] | None = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            body = node.body
        elif isinstance(node, (ast.If,)):
            body = node.body
            # Also check the else branch
            if node.orelse:
                _check_body_for_unreachable(node.orelse, findings)

        if body:
            _check_body_for_unreachable(body, findings)

    if findings:
        return CheckResult(
            name="unreachable_code",
            passed=False,
            max_points=0.1,
            earned_points=0.0,
            details="; ".join(findings),
        )
    return CheckResult(
        name="unreachable_code",
        passed=True,
        max_points=0.1,
        earned_points=0.1,
        details="No unreachable code detected",
    )


def _check_body_for_unreachable(body: list[ast.stmt], findings: list[str]) -> None:
    """Check a statement list for code after terminal statements."""
    for i, stmt in enumerate(body):
        if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            if i < len(body) - 1:
                next_stmt = body[i + 1]
                findings.append(
                    f"Unreachable code after {type(stmt).__name__} at line {next_stmt.lineno}"
                )
                break  # Only report first unreachable per block


def check_undefined_names(tree: ast.Module) -> CheckResult:
    """Check for potentially undefined name usage (best-effort heuristic).

    This is a lightweight check — not a full scope analyzer. It tracks names
    defined at module/function scope and flags names used but never defined
    locally or imported.
    """
    # Collect all defined names (assignments, function defs, imports, class defs, for targets)
    defined: set[str] = set(PYTHON_BUILTINS)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
            # Add parameter names
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                defined.add(arg.arg)
            if node.args.vararg:
                defined.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined.add(node.args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    # Wildcard import — can't track, assume safe
                    return CheckResult(
                        name="undefined_names",
                        passed=True,
                        max_points=0.1,
                        earned_points=0.1,
                        details="Wildcard import present, skipping check",
                    )
                defined.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                _collect_names(target, defined)
        elif isinstance(node, ast.AnnAssign) and node.target:
            _collect_names(node.target, defined)
        elif isinstance(node, ast.AugAssign):
            _collect_names(node.target, defined)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            _collect_names(node.target, defined)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars:
                    _collect_names(item.optional_vars, defined)
        elif isinstance(node, ast.NamedExpr):
            defined.add(node.target.id)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                defined.add(node.name)
        elif isinstance(node, ast.Global):
            for name in node.names:
                defined.add(name)
        elif isinstance(node, ast.Nonlocal):
            for name in node.names:
                defined.add(name)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for generator in node.generators:
                _collect_names(generator.target, defined)

    # Collect all Name nodes used in Load context
    used_names: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used_names.append((node.id, node.lineno))

    # Find names used but never defined
    # Filter decorators and attribute access (conservative)
    undefined = []
    for name, lineno in used_names:
        if name not in defined and not name.startswith("_"):
            undefined.append(f"{name} (line {lineno})")

    if undefined:
        # Limit to first 5 to avoid noise
        shown = undefined[:5]
        extra = f" (+{len(undefined) - 5} more)" if len(undefined) > 5 else ""
        return CheckResult(
            name="undefined_names",
            passed=False,
            max_points=0.1,
            earned_points=0.0,
            details=f"Potentially undefined: {', '.join(shown)}{extra}",
        )
    return CheckResult(
        name="undefined_names",
        passed=True,
        max_points=0.1,
        earned_points=0.1,
        details="No undefined names detected",
    )


def _collect_names(node: ast.AST, names: set[str]) -> None:
    """Recursively collect assigned names from an AST target node."""
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            _collect_names(elt, names)
    elif isinstance(node, ast.Starred):
        _collect_names(node.value, names)


def check_return_consistency(tree: ast.Module) -> CheckResult:
    """Check for functions that sometimes return a value and sometimes don't."""
    findings: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        returns_value: list[int] = []
        returns_none: list[int] = []

        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                if child.value is None:
                    returns_none.append(child.lineno)
                else:
                    returns_value.append(child.lineno)

        # Flag if function has both return-with-value and bare-return
        if returns_value and returns_none:
            findings.append(
                f"{node.name}() has inconsistent returns (values at lines "
                f"{returns_value[:3]}, bare at lines {returns_none[:3]})"
            )

    if findings:
        return CheckResult(
            name="return_consistency",
            passed=False,
            max_points=0.1,
            earned_points=0.0,
            details="; ".join(findings),
        )
    return CheckResult(
        name="return_consistency",
        passed=True,
        max_points=0.1,
        earned_points=0.1,
        details="Return statements are consistent",
    )


# --- Scorer ---


class CorrectnessScorer:
    """Score code correctness via AST analysis.

    Deterministic, no LLM calls. Checks:
    - Syntax validity (0.5)
    - Exception handling (0.2)
    - Unreachable code (0.1)
    - Undefined names (0.1)
    - Return consistency (0.1)
    """

    def score(self, code: str, filename: str = "") -> DimensionScore:
        """Score code correctness. Returns 0.0-1.0."""
        if not code or not code.strip():
            return DimensionScore(
                dimension=Dimension.CORRECTNESS,
                score=0.0,
                method=ScoringMethod.AST_PARSE,
                details="Empty code",
            )

        # Only score Python files
        if filename and not filename.endswith((".py", "")):
            return DimensionScore(
                dimension=Dimension.CORRECTNESS,
                score=0.5,
                method=ScoringMethod.AST_PARSE,
                details=f"Unsupported file type: {filename}",
            )

        report = CorrectnessReport()

        # Check 1: Syntax
        syntax_result, tree = check_syntax(code)
        report.checks.append(syntax_result)

        if tree is None:
            # Can't run further checks without a valid AST
            return DimensionScore(
                dimension=Dimension.CORRECTNESS,
                score=report.total_score,
                method=ScoringMethod.AST_PARSE,
                details=report.details_text,
            )

        # Check 2: Bare/empty excepts
        report.checks.append(check_bare_excepts(tree))

        # Check 3: Unreachable code
        report.checks.append(check_unreachable_code(tree))

        # Check 4: Undefined names
        report.checks.append(check_undefined_names(tree))

        # Check 5: Return consistency
        report.checks.append(check_return_consistency(tree))

        return DimensionScore(
            dimension=Dimension.CORRECTNESS,
            score=report.total_score,
            method=ScoringMethod.AST_PARSE,
            details=report.details_text,
        )
