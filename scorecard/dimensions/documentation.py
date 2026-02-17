"""Documentation dimension scorer.

Evaluates whether code is adequately documented for someone else to
understand it. Two modes:
- LLM judge (Claude API) for nuanced evaluation
- Rule-based fallback (docstring presence + comment density) when LLM unavailable
"""

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field

from shared.models import Dimension, DimensionScore, ScoringMethod

# --- LLM Judge ---

_LLM_JUDGE_PROMPT = """\
You are evaluating Python code documentation quality. Score each criterion 0.0-1.0.

IMPORTANT: Simple, obvious code (like `x = 1` or `return a + b`) does NOT need \
comments. Only penalize missing documentation when the logic is genuinely non-obvious.

Code to evaluate:
```python
{code}
```

Criteria:
1. logic_explanation (weight 0.35): Are complex/non-obvious parts explained? \
Score 1.0 if all code is simple/obvious OR if complex parts have comments.
2. function_docs (weight 0.25): Do public functions have docstrings describing \
their purpose? Score 1.0 if functions are trivially obvious or have docstrings.
3. purpose_docs (weight 0.25): Is there a module docstring or top-level comment \
explaining what this code does? Score 1.0 if the code's purpose is obvious from \
naming alone.
4. comment_accuracy (weight 0.15): Do existing comments match what the code \
actually does? Score 1.0 if there are no comments or all comments are accurate.

Return ONLY valid JSON (no markdown fencing):
{{"logic_explanation": 0.0, "function_docs": 0.0, "purpose_docs": 0.0, \
"comment_accuracy": 0.0, "reasoning": "..."}}
"""

_LLM_JUDGE_WEIGHTS = {
    "logic_explanation": 0.35,
    "function_docs": 0.25,
    "purpose_docs": 0.25,
    "comment_accuracy": 0.15,
}


@dataclass
class LLMJudgeResult:
    """Result from the LLM judge."""

    logic_explanation: float
    function_docs: float
    purpose_docs: float
    comment_accuracy: float
    reasoning: str

    @property
    def weighted_score(self) -> float:
        total = 0.0
        for key, weight in _LLM_JUDGE_WEIGHTS.items():
            total += getattr(self, key) * weight
        return min(max(total, 0.0), 1.0)


def _call_llm_judge(code: str, model: str = "claude-sonnet-4-5-20250514") -> LLMJudgeResult | None:
    """Call the Claude API to judge documentation quality.

    Returns None if the call fails for any reason.
    """
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
        # Strip markdown fencing if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        data = json.loads(text)
        return LLMJudgeResult(
            logic_explanation=_clamp(data.get("logic_explanation", 0.5)),
            function_docs=_clamp(data.get("function_docs", 0.5)),
            purpose_docs=_clamp(data.get("purpose_docs", 0.5)),
            comment_accuracy=_clamp(data.get("comment_accuracy", 0.5)),
            reasoning=data.get("reasoning", ""),
        )
    except Exception:
        return None


async def _call_llm_judge_async(
    code: str, model: str = "claude-sonnet-4-5-20250514"
) -> LLMJudgeResult | None:
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

        data = json.loads(text)
        return LLMJudgeResult(
            logic_explanation=_clamp(data.get("logic_explanation", 0.5)),
            function_docs=_clamp(data.get("function_docs", 0.5)),
            purpose_docs=_clamp(data.get("purpose_docs", 0.5)),
            comment_accuracy=_clamp(data.get("comment_accuracy", 0.5)),
            reasoning=data.get("reasoning", ""),
        )
    except Exception:
        return None


def _clamp(value: float) -> float:
    """Clamp a value to 0.0-1.0."""
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.5


# --- Rule-based Fallback ---


@dataclass
class FallbackCheck:
    """Result of a single fallback documentation check."""

    name: str
    score: float
    weight: float
    details: str


@dataclass
class FallbackReport:
    """Aggregated fallback results."""

    checks: list[FallbackCheck] = field(default_factory=list)

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


def check_module_docstring(tree: ast.Module) -> FallbackCheck:
    """Check for a module-level docstring."""
    has_docstring = (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    )
    if has_docstring:
        return FallbackCheck(
            name="module_docstring",
            score=1.0,
            weight=0.25,
            details="Module docstring present",
        )
    return FallbackCheck(
        name="module_docstring",
        score=0.0,
        weight=0.25,
        details="No module docstring",
    )


def check_function_docstrings(tree: ast.Module) -> FallbackCheck:
    """Check that public functions have docstrings."""
    total = 0
    documented = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip private/dunder methods
            if node.name.startswith("_"):
                continue
            total += 1
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                documented += 1

    if total == 0:
        return FallbackCheck(
            name="function_docstrings",
            score=1.0,
            weight=0.25,
            details="No public functions to check",
        )

    ratio = documented / total
    undocumented = total - documented
    if undocumented == 0:
        detail = f"All {total} public functions documented"
    else:
        detail = f"{undocumented}/{total} public functions lack docstrings"

    return FallbackCheck(
        name="function_docstrings",
        score=ratio,
        weight=0.25,
        details=detail,
    )


def check_comment_density(code: str) -> FallbackCheck:
    """Check comment density as a proxy for logic explanation.

    Targets 5-15% comment lines. Simple code with few comments is fine;
    complex code with zero comments is not.
    """
    lines = code.splitlines()
    code_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
    comment_lines = [line for line in lines if line.strip().startswith("#")]
    inline_comments = sum(1 for line in code_lines if "#" in line)
    total_comments = len(comment_lines) + inline_comments
    total_code = len(code_lines)

    if total_code == 0:
        return FallbackCheck(
            name="comment_density",
            score=1.0,
            weight=0.35,
            details="No code lines to check",
        )

    density = total_comments / total_code

    # Simple heuristic: 0% comments = 0.3 (some code is self-documenting),
    # 5%+ = 1.0 (adequate), scale linearly between
    if density >= 0.05:
        score = 1.0
    else:
        score = 0.3 + (density / 0.05) * 0.7

    return FallbackCheck(
        name="comment_density",
        score=score,
        weight=0.35,
        details=f"{total_comments} comments in {total_code} code lines ({density:.0%} density)",
    )


def check_class_docstrings(tree: ast.Module) -> FallbackCheck:
    """Check that classes have docstrings."""
    total = 0
    documented = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            total += 1
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                documented += 1

    if total == 0:
        return FallbackCheck(
            name="class_docstrings",
            score=1.0,
            weight=0.15,
            details="No public classes to check",
        )

    ratio = documented / total
    undocumented = total - documented
    if undocumented == 0:
        detail = f"All {total} public classes documented"
    else:
        detail = f"{undocumented}/{total} public classes lack docstrings"

    return FallbackCheck(
        name="class_docstrings",
        score=ratio,
        weight=0.15,
        details=detail,
    )


def _score_fallback(code: str, tree: ast.Module) -> tuple[float, str]:
    """Score documentation using rule-based fallback."""
    report = FallbackReport()
    report.checks.append(check_module_docstring(tree))
    report.checks.append(check_function_docstrings(tree))
    report.checks.append(check_comment_density(code))
    report.checks.append(check_class_docstrings(tree))
    return report.total_score, report.details_text


# --- Cache ---

_score_cache: dict[str, DimensionScore] = {}


def _cache_key(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def clear_cache() -> None:
    """Clear the documentation score cache."""
    _score_cache.clear()


# --- Scorer ---


class DocumentationScorer:
    """Score code documentation quality.

    Primary: LLM judge (Claude API) for nuanced evaluation.
    Fallback: Rule-based heuristic when LLM is unavailable.
    Results are cached by code content hash.
    """

    def __init__(self, use_llm: bool = True, model: str = "claude-sonnet-4-5-20250514"):
        self.use_llm = use_llm
        self.model = model

    async def score_async(self, code: str, filename: str = "") -> DimensionScore:
        """Async version — native async for LLM, thread pool for fallback."""
        import asyncio

        if not code or not code.strip():
            return DimensionScore(
                dimension=Dimension.DOCUMENTATION,
                score=0.0,
                method=ScoringMethod.LLM_JUDGE,
                details="Empty code",
            )

        key = _cache_key(code)
        if key in _score_cache:
            return _score_cache[key]

        try:
            tree = ast.parse(code)
        except SyntaxError:
            result = DimensionScore(
                dimension=Dimension.DOCUMENTATION,
                score=0.0,
                method=ScoringMethod.RULE_BASED,
                details="SyntaxError — cannot analyze documentation",
            )
            _score_cache[key] = result
            return result

        if self.use_llm:
            llm_result = await _call_llm_judge_async(code, model=self.model)
            if llm_result is not None:
                result = DimensionScore(
                    dimension=Dimension.DOCUMENTATION,
                    score=llm_result.weighted_score,
                    method=ScoringMethod.LLM_JUDGE,
                    details=llm_result.reasoning,
                    metadata={
                        "logic_explanation": llm_result.logic_explanation,
                        "function_docs": llm_result.function_docs,
                        "purpose_docs": llm_result.purpose_docs,
                        "comment_accuracy": llm_result.comment_accuracy,
                    },
                )
                _score_cache[key] = result
                return result

        fallback_score, fallback_details = await asyncio.to_thread(_score_fallback, code, tree)
        result = DimensionScore(
            dimension=Dimension.DOCUMENTATION,
            score=fallback_score,
            method=ScoringMethod.RULE_BASED,
            details=f"(fallback) {fallback_details}",
        )
        _score_cache[key] = result
        return result

    def score(self, code: str, filename: str = "") -> DimensionScore:
        """Score documentation quality. Returns 0.0-1.0."""
        if not code or not code.strip():
            return DimensionScore(
                dimension=Dimension.DOCUMENTATION,
                score=0.0,
                method=ScoringMethod.LLM_JUDGE,
                details="Empty code",
            )

        # Check cache
        key = _cache_key(code)
        if key in _score_cache:
            return _score_cache[key]

        try:
            tree = ast.parse(code)
        except SyntaxError:
            result = DimensionScore(
                dimension=Dimension.DOCUMENTATION,
                score=0.0,
                method=ScoringMethod.RULE_BASED,
                details="SyntaxError — cannot analyze documentation",
            )
            _score_cache[key] = result
            return result

        # Try LLM judge first
        if self.use_llm:
            llm_result = _call_llm_judge(code, model=self.model)
            if llm_result is not None:
                result = DimensionScore(
                    dimension=Dimension.DOCUMENTATION,
                    score=llm_result.weighted_score,
                    method=ScoringMethod.LLM_JUDGE,
                    details=llm_result.reasoning,
                    metadata={
                        "logic_explanation": llm_result.logic_explanation,
                        "function_docs": llm_result.function_docs,
                        "purpose_docs": llm_result.purpose_docs,
                        "comment_accuracy": llm_result.comment_accuracy,
                    },
                )
                _score_cache[key] = result
                return result

        # Fallback to rule-based
        fallback_score, fallback_details = _score_fallback(code, tree)
        result = DimensionScore(
            dimension=Dimension.DOCUMENTATION,
            score=fallback_score,
            method=ScoringMethod.RULE_BASED,
            details=f"(fallback) {fallback_details}",
        )
        _score_cache[key] = result
        return result
