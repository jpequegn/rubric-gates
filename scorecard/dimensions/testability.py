"""Testability dimension scorer.

Evaluates whether generated code can be tested and whether tests exist.
Hybrid: rule-based structural checks + LLM judge for nuance.
"""

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from shared.models import Dimension, DimensionScore, ScoringMethod

# --- Rule-based Checks ---


@dataclass
class TestabilityCheck:
    """Result of a single testability check."""

    name: str
    score: float  # 0.0-1.0
    weight: float
    details: str


@dataclass
class TestabilityReport:
    """Aggregated testability results."""

    checks: list[TestabilityCheck] = field(default_factory=list)

    @property
    def total_score(self) -> float:
        total_weight = sum(c.weight for c in self.checks)
        if total_weight == 0:
            return 0.0
        weighted = sum(c.score * c.weight for c in self.checks)
        return min(weighted / total_weight, 1.0)

    @property
    def details_text(self) -> str:
        lines = []
        for c in self.checks:
            pct = int(c.score * 100)
            lines.append(f"[{pct}%] {c.name}: {c.details}")
        return "; ".join(lines)


def check_test_file_exists(
    filename: str,
    project_files: list[str] | None = None,
) -> TestabilityCheck:
    """Check if a corresponding test file exists."""
    if not filename or not project_files:
        return TestabilityCheck(
            name="test_file",
            score=0.0,
            weight=0.15,
            details="No project files provided to check for tests",
        )

    stem = Path(filename).stem
    # Common test file patterns
    test_names = {
        f"test_{stem}.py",
        f"{stem}_test.py",
    }

    for pf in project_files:
        pf_name = Path(pf).name
        if pf_name in test_names:
            return TestabilityCheck(
                name="test_file",
                score=1.0,
                weight=0.15,
                details=f"Test file found: {pf_name}",
            )

    return TestabilityCheck(
        name="test_file",
        score=0.0,
        weight=0.15,
        details=f"No test file found for {filename}",
    )


def check_test_assertions(test_code: str | None) -> TestabilityCheck:
    """Check that test code has real assertions, not just print statements."""
    if test_code is None:
        return TestabilityCheck(
            name="test_assertions",
            score=0.5,
            weight=0.10,
            details="No test code provided",
        )

    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return TestabilityCheck(
            name="test_assertions",
            score=0.0,
            weight=0.10,
            details="Test code has syntax errors",
        )

    assert_count = 0
    print_count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            assert_count += 1
        elif isinstance(node, ast.Call):
            # Count pytest.raises, assert methods, etc.
            if isinstance(node.func, ast.Attribute) and node.func.attr in (
                "assertEqual",
                "assertTrue",
                "assertFalse",
                "assertRaises",
                "assertIn",
                "assertIsNone",
                "assertIsNotNone",
            ):
                assert_count += 1
            # Count plain print() calls
            elif isinstance(node.func, ast.Name) and node.func.id == "print":
                print_count += 1

    if assert_count == 0 and print_count == 0:
        return TestabilityCheck(
            name="test_assertions",
            score=0.0,
            weight=0.10,
            details="No assertions or prints found in test code",
        )

    if assert_count == 0 and print_count > 0:
        return TestabilityCheck(
            name="test_assertions",
            score=0.2,
            weight=0.10,
            details=f"Only print statements ({print_count}), no assertions",
        )

    return TestabilityCheck(
        name="test_assertions",
        score=1.0,
        weight=0.10,
        details=f"{assert_count} assertions found",
    )


def check_function_purity(tree: ast.Module) -> TestabilityCheck:
    """Check that functions avoid global/nonlocal state mutation."""
    total_functions = 0
    impure_functions: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        total_functions += 1

        has_global = False
        has_nonlocal = False
        has_global_mutation = False

        for child in ast.walk(node):
            if isinstance(child, ast.Global):
                has_global = True
            elif isinstance(child, ast.Nonlocal):
                has_nonlocal = True
            # Attribute assignment on module-level names (heuristic)
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                        # Mutating a global dict/list: globals_dict[key] = val
                        pass  # Hard to tell if global, skip
                    elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == "self":
                            continue  # self.x = ... is fine
                        # Could be global state mutation
                        has_global_mutation = True

        if has_global or has_nonlocal or has_global_mutation:
            impure_functions.append(node.name)

    if total_functions == 0:
        return TestabilityCheck(
            name="function_purity",
            score=1.0,
            weight=0.15,
            details="No functions to check",
        )

    if not impure_functions:
        return TestabilityCheck(
            name="function_purity",
            score=1.0,
            weight=0.15,
            details=f"All {total_functions} functions are pure",
        )

    pure_ratio = 1.0 - len(impure_functions) / total_functions
    shown = impure_functions[:5]
    return TestabilityCheck(
        name="function_purity",
        score=max(0.0, pure_ratio),
        weight=0.15,
        details=f"Impure functions: {', '.join(f'{n}()' for n in shown)}",
    )


def check_modularity(tree: ast.Module) -> TestabilityCheck:
    """Check if code has clear input/output boundaries (functions, not top-level scripts)."""
    total_stmts = len(tree.body)
    if total_stmts == 0:
        return TestabilityCheck(
            name="modularity",
            score=1.0,
            weight=0.10,
            details="Empty module",
        )

    function_defs = 0
    class_defs = 0
    imports = 0
    executable = 0

    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_defs += 1
        elif isinstance(stmt, ast.ClassDef):
            class_defs += 1
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            imports += 1
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            pass  # Docstring
        elif isinstance(stmt, ast.If):
            if _is_main_guard(stmt):
                continue
            executable += 1
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            # Top-level assignments are executable (script-like)
            executable += 1
        else:
            executable += 1

    structured = function_defs + class_defs
    total_non_import = total_stmts - imports

    if total_non_import == 0:
        return TestabilityCheck(
            name="modularity",
            score=1.0,
            weight=0.10,
            details="Import-only module",
        )

    # High ratio of functions/classes = more testable
    if structured == 0 and executable > 0:
        return TestabilityCheck(
            name="modularity",
            score=0.2,
            weight=0.10,
            details=f"Script-style code: {executable} executable statements, no functions/classes",
        )

    structure_ratio = structured / max(structured + executable, 1)
    score = min(1.0, 0.3 + structure_ratio * 0.7)

    return TestabilityCheck(
        name="modularity",
        score=score,
        weight=0.10,
        details=f"{function_defs} functions, {class_defs} classes, {executable} top-level executable",
    )


def _is_main_guard(node: ast.If) -> bool:
    """Check if an If node is `if __name__ == '__main__':`."""
    test = node.test
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        if isinstance(test.ops[0], ast.Eq):
            left = test.left
            right = test.comparators[0] if test.comparators else None
            if (
                isinstance(left, ast.Name)
                and left.id == "__name__"
                and isinstance(right, ast.Constant)
                and right.value == "__main__"
            ):
                return True
    return False


# --- LLM Judge ---

_LLM_JUDGE_PROMPT = """\
You are evaluating Python code testability. Score 0.0-1.0.

Code to evaluate:
```python
{code}
```

Criteria:
1. isolation (weight 0.4): Can functions be tested independently? Are there \
clear input/output boundaries? Low coupling between components?
2. mockability (weight 0.3): Can external dependencies be easily mocked? \
Are there dependency injection points? Is I/O separated from logic?
3. edge_cases (weight 0.3): Does the code structure make edge cases testable? \
Are error paths accessible? Can boundary conditions be exercised?

Score 1.0 for simple, pure functions. Score lower for tightly-coupled, \
hard-to-isolate code.

Return ONLY valid JSON (no markdown fencing):
{{"isolation": 0.0, "mockability": 0.0, "edge_cases": 0.0, "reasoning": "..."}}
"""

_LLM_WEIGHTS = {
    "isolation": 0.4,
    "mockability": 0.3,
    "edge_cases": 0.3,
}


def _call_llm_judge(code: str, model: str = "claude-sonnet-4-5-20250514") -> dict | None:
    """Call the Claude API to judge testability. Returns None on failure."""
    try:
        import anthropic
    except ImportError:
        return None

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[
                {"role": "user", "content": _LLM_JUDGE_PROMPT.format(code=code[:4000])},
            ],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        data = json.loads(text)
        return data
    except Exception:
        return None


async def _call_llm_judge_async(
    code: str, model: str = "claude-sonnet-4-5-20250514"
) -> dict | None:
    """Async version of the LLM judge call using AsyncAnthropic."""
    try:
        import anthropic
    except ImportError:
        return None

    try:
        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=model,
            max_tokens=512,
            messages=[
                {"role": "user", "content": _LLM_JUDGE_PROMPT.format(code=code[:4000])},
            ],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return json.loads(text)
    except Exception:
        return None


def _clamp(value: float) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.5


# --- Cache ---

_score_cache: dict[str, DimensionScore] = {}


def _cache_key(code: str, filename: str, project_files: list[str] | None) -> str:
    parts = [code, filename, str(sorted(project_files or []))]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def clear_cache() -> None:
    """Clear the testability score cache."""
    _score_cache.clear()


# --- Scorer ---


class TestabilityScorer:
    """Score code testability via structural analysis and optional LLM judge.

    Rule-based checks (0.5 weight):
    - Test file existence (0.15)
    - Test assertions quality (0.10)
    - Function purity (0.15)
    - Modularity (0.10)

    LLM judge (0.5 weight, optional):
    - Isolation (0.4)
    - Mockability (0.3)
    - Edge case testability (0.3)
    """

    def __init__(self, use_llm: bool = True, model: str = "claude-sonnet-4-5-20250514"):
        self.use_llm = use_llm
        self.model = model

    async def score_async(
        self,
        code: str,
        filename: str = "",
        project_files: list[str] | None = None,
        test_code: str | None = None,
    ) -> DimensionScore:
        """Async version — native async for LLM, thread pool for rule-based."""
        if not code or not code.strip():
            return DimensionScore(
                dimension=Dimension.TESTABILITY,
                score=0.0,
                method=ScoringMethod.RULE_BASED,
                details="Empty code",
            )

        key = _cache_key(code, filename, project_files)
        if key in _score_cache:
            return _score_cache[key]

        try:
            tree = ast.parse(code)
        except SyntaxError:
            result = DimensionScore(
                dimension=Dimension.TESTABILITY,
                score=0.0,
                method=ScoringMethod.RULE_BASED,
                details="SyntaxError — cannot analyze testability",
            )
            _score_cache[key] = result
            return result

        report = TestabilityReport()
        report.checks.append(check_test_file_exists(filename, project_files))
        report.checks.append(check_test_assertions(test_code))
        report.checks.append(check_function_purity(tree))
        report.checks.append(check_modularity(tree))

        rule_score = report.total_score
        rule_details = report.details_text

        if self.use_llm:
            llm_result = await _call_llm_judge_async(code, model=self.model)
            if llm_result is not None:
                llm_score = sum(_clamp(llm_result.get(k, 0.5)) * w for k, w in _LLM_WEIGHTS.items())
                llm_score = min(max(llm_score, 0.0), 1.0)
                reasoning = llm_result.get("reasoning", "")

                combined = rule_score * 0.5 + llm_score * 0.5
                details = f"{rule_details}; [LLM] {reasoning}"

                result = DimensionScore(
                    dimension=Dimension.TESTABILITY,
                    score=combined,
                    method=ScoringMethod.HYBRID,
                    details=details,
                    metadata={
                        "rule_score": rule_score,
                        "llm_score": llm_score,
                        "isolation": _clamp(llm_result.get("isolation", 0.5)),
                        "mockability": _clamp(llm_result.get("mockability", 0.5)),
                        "edge_cases": _clamp(llm_result.get("edge_cases", 0.5)),
                    },
                )
                _score_cache[key] = result
                return result

        result = DimensionScore(
            dimension=Dimension.TESTABILITY,
            score=rule_score,
            method=ScoringMethod.RULE_BASED,
            details=f"(rule-based only) {rule_details}",
        )
        _score_cache[key] = result
        return result

    def score(
        self,
        code: str,
        filename: str = "",
        project_files: list[str] | None = None,
        test_code: str | None = None,
    ) -> DimensionScore:
        """Score testability. Returns 0.0-1.0."""
        if not code or not code.strip():
            return DimensionScore(
                dimension=Dimension.TESTABILITY,
                score=0.0,
                method=ScoringMethod.RULE_BASED,
                details="Empty code",
            )

        key = _cache_key(code, filename, project_files)
        if key in _score_cache:
            return _score_cache[key]

        try:
            tree = ast.parse(code)
        except SyntaxError:
            result = DimensionScore(
                dimension=Dimension.TESTABILITY,
                score=0.0,
                method=ScoringMethod.RULE_BASED,
                details="SyntaxError — cannot analyze testability",
            )
            _score_cache[key] = result
            return result

        # Rule-based checks
        report = TestabilityReport()
        report.checks.append(check_test_file_exists(filename, project_files))
        report.checks.append(check_test_assertions(test_code))
        report.checks.append(check_function_purity(tree))
        report.checks.append(check_modularity(tree))

        rule_score = report.total_score
        rule_details = report.details_text

        # LLM judge
        if self.use_llm:
            llm_result = _call_llm_judge(code, model=self.model)
            if llm_result is not None:
                llm_score = sum(_clamp(llm_result.get(k, 0.5)) * w for k, w in _LLM_WEIGHTS.items())
                llm_score = min(max(llm_score, 0.0), 1.0)
                reasoning = llm_result.get("reasoning", "")

                # Combine: 50% rule-based, 50% LLM
                combined = rule_score * 0.5 + llm_score * 0.5
                details = f"{rule_details}; [LLM] {reasoning}"

                result = DimensionScore(
                    dimension=Dimension.TESTABILITY,
                    score=combined,
                    method=ScoringMethod.HYBRID,
                    details=details,
                    metadata={
                        "rule_score": rule_score,
                        "llm_score": llm_score,
                        "isolation": _clamp(llm_result.get("isolation", 0.5)),
                        "mockability": _clamp(llm_result.get("mockability", 0.5)),
                        "edge_cases": _clamp(llm_result.get("edge_cases", 0.5)),
                    },
                )
                _score_cache[key] = result
                return result

        # Fallback: rule-based only
        result = DimensionScore(
            dimension=Dimension.TESTABILITY,
            score=rule_score,
            method=ScoringMethod.RULE_BASED,
            details=f"(rule-based only) {rule_details}",
        )
        _score_cache[key] = result
        return result
