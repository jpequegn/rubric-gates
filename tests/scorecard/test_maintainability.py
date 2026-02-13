"""Tests for maintainability dimension scorer."""

import ast

import pytest

from scorecard.dimensions.maintainability import (
    MaintainabilityScorer,
    check_cyclomatic_complexity,
    check_file_length,
    check_function_length,
    check_naming_quality,
    check_nesting_depth,
)
from shared.models import Dimension, ScoringMethod


@pytest.fixture()
def scorer():
    return MaintainabilityScorer()


def _parse(code: str) -> ast.Module:
    return ast.parse(code)


# --- Cyclomatic Complexity ---


class TestCyclomaticComplexity:
    def test_simple_function(self):
        code = """\
def add(a, b):
    return a + b
"""
        result = check_cyclomatic_complexity(_parse(code))
        assert result.score == 1.0

    def test_branching_function(self):
        code = """\
def categorize(x):
    if x > 100:
        return "high"
    elif x > 50:
        return "medium"
    elif x > 10:
        return "low"
    else:
        return "minimal"
"""
        result = check_cyclomatic_complexity(_parse(code))
        # CC = 1 + 4 (if + 3 elif) = 5, within threshold
        assert result.score == 1.0

    def test_complex_function_exceeds(self):
        # Build a function with CC > 10
        code = """\
def complex_handler(data):
    if data is None:
        return None
    if data.type == "a":
        if data.value > 0:
            for item in data.items:
                if item.active:
                    if item.priority > 5:
                        while item.retry > 0:
                            try:
                                process(item)
                            except ValueError:
                                if item.fallback:
                                    use_fallback(item)
                            except TypeError:
                                log_error(item)
    elif data.type == "b":
        assert data.valid
    return data
"""
        result = check_cyclomatic_complexity(_parse(code))
        assert result.score < 1.0
        assert "complex_handler" in result.details

    def test_no_functions(self):
        code = "x = 1\ny = 2"
        result = check_cyclomatic_complexity(_parse(code))
        assert result.score == 1.0
        assert "No functions" in result.details

    def test_custom_threshold(self):
        code = """\
def moderate(x):
    if x > 0:
        if x > 10:
            return "big"
        return "small"
    return "zero"
"""
        # With low threshold, this should flag
        result = check_cyclomatic_complexity(_parse(code), threshold=2)
        assert result.score < 1.0

    def test_boolean_operators_counted(self):
        code = """\
def check(a, b, c, d, e, f, g, h, i, j, k):
    if a and b and c and d and e and f and g and h and i and j and k:
        return True
    return False
"""
        # CC = 1 (base) + 1 (if) + 10 (and operators) = 12
        result = check_cyclomatic_complexity(_parse(code), threshold=10)
        assert result.score < 1.0


# --- Function Length ---


class TestFunctionLength:
    def test_short_function(self):
        code = """\
def greet(name):
    return f"Hello, {name}!"
"""
        result = check_function_length(_parse(code))
        assert result.score == 1.0

    def test_long_function(self):
        # Generate a function with 60+ lines
        body_lines = [f"    x{i} = {i}" for i in range(60)]
        code = "def long_func():\n" + "\n".join(body_lines) + "\n    return x0\n"
        result = check_function_length(_parse(code))
        assert result.score < 1.0
        assert "long_func" in result.details

    def test_no_functions(self):
        code = "x = 1"
        result = check_function_length(_parse(code))
        assert result.score == 1.0

    def test_custom_threshold(self):
        body_lines = [f"    x{i} = {i}" for i in range(15)]
        code = "def medium():\n" + "\n".join(body_lines) + "\n"
        # With low threshold of 10, this should flag
        result = check_function_length(_parse(code), threshold=10)
        assert result.score < 1.0

    def test_multiple_short_functions(self):
        code = """\
def a():
    return 1

def b():
    return 2

def c():
    return 3
"""
        result = check_function_length(_parse(code))
        assert result.score == 1.0


# --- File Length ---


class TestFileLength:
    def test_short_file(self):
        code = "x = 1\ny = 2\nprint(x + y)"
        result = check_file_length(code)
        assert result.score == 1.0
        assert "3 lines" in result.details

    def test_long_file(self):
        lines = [f"x{i} = {i}" for i in range(400)]
        code = "\n".join(lines)
        result = check_file_length(code)
        assert result.score < 1.0
        assert "exceeds" in result.details

    def test_custom_threshold(self):
        lines = [f"x{i} = {i}" for i in range(20)]
        code = "\n".join(lines)
        result = check_file_length(code, threshold=10)
        assert result.score < 1.0

    def test_blanks_and_comments_in_count(self):
        code = "x = 1\n\n# comment\ny = 2\n\n"
        result = check_file_length(code)
        assert result.score == 1.0
        # Total is 5 lines, effective is 2 (non-blank, non-comment)
        assert "effective" in result.details

    def test_exactly_at_threshold(self):
        lines = [f"x{i} = {i}" for i in range(300)]
        code = "\n".join(lines)
        result = check_file_length(code, threshold=300)
        assert result.score == 1.0

    def test_very_long_file_floors(self):
        lines = [f"x{i} = {i}" for i in range(2000)]
        code = "\n".join(lines)
        result = check_file_length(code, threshold=300)
        assert result.score < 0.2


# --- Naming Quality ---


class TestNamingQuality:
    def test_good_names(self):
        code = """\
class UserProfile:
    pass

def get_user_name(user_id):
    return "test"

MAX_RETRIES = 3
total_count = 0
"""
        result = check_naming_quality(_parse(code))
        assert result.score == 1.0

    def test_bad_function_name(self):
        code = """\
def GetUserName():
    return "test"
"""
        result = check_naming_quality(_parse(code))
        assert result.score < 1.0
        assert "not snake_case" in result.details

    def test_bad_class_name(self):
        code = """\
class my_class:
    pass
"""
        result = check_naming_quality(_parse(code))
        assert result.score < 1.0
        assert "not PascalCase" in result.details

    def test_dunder_ok(self):
        code = """\
class Foo:
    def __init__(self):
        pass

    def __repr__(self):
        return "Foo"
"""
        result = check_naming_quality(_parse(code))
        assert result.score == 1.0

    def test_private_names_ok(self):
        code = """\
_internal = 1
_cache = {}
"""
        result = check_naming_quality(_parse(code))
        assert result.score == 1.0

    def test_no_names(self):
        code = "1 + 2"
        result = check_naming_quality(_parse(code))
        assert result.score == 1.0
        assert "No names" in result.details

    def test_mixed_violations(self):
        code = """\
def badName():
    pass

class another_bad:
    pass
"""
        result = check_naming_quality(_parse(code))
        assert result.score < 1.0


# --- Nesting Depth ---


class TestNestingDepth:
    def test_shallow_function(self):
        code = """\
def simple(x):
    if x > 0:
        return x
    return 0
"""
        result = check_nesting_depth(_parse(code))
        assert result.score == 1.0

    def test_deep_nesting(self):
        code = """\
def deep(data):
    if data:
        for item in data:
            if item.active:
                for sub in item.children:
                    if sub.valid:
                        process(sub)
"""
        result = check_nesting_depth(_parse(code))
        assert result.score < 1.0
        assert "deep" in result.details

    def test_no_functions(self):
        code = "x = 1"
        result = check_nesting_depth(_parse(code))
        assert result.score == 1.0
        assert "No functions" in result.details

    def test_custom_threshold(self):
        code = """\
def two_deep(x):
    if x:
        for i in x:
            print(i)
"""
        result = check_nesting_depth(_parse(code), threshold=1)
        assert result.score < 1.0

    def test_try_except_counted(self):
        code = """\
def handle(x):
    if x:
        try:
            for item in x:
                if item:
                    process(item)
        except Exception:
            pass
"""
        result = check_nesting_depth(_parse(code), threshold=3)
        assert result.score < 1.0

    def test_nested_functions_independent(self):
        code = """\
def outer():
    if True:
        if True:
            pass

    def inner():
        if True:
            if True:
                pass
"""
        # inner's nesting shouldn't count against outer
        result = check_nesting_depth(_parse(code), threshold=3)
        assert result.score == 1.0


# --- MaintainabilityScorer Integration ---


class TestMaintainabilityScorer:
    def test_clean_code(self, scorer):
        code = """\
def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

result = add(1, 2)
print(multiply(result, 3))
"""
        score = scorer.score(code, "main.py")
        assert score.dimension == Dimension.MAINTAINABILITY
        assert score.method == ScoringMethod.RULE_BASED
        assert score.score == 1.0

    def test_empty_code(self, scorer):
        score = scorer.score("", "empty.py")
        assert score.score == 0.0
        assert "Empty" in score.details

    def test_syntax_error(self, scorer):
        score = scorer.score("def broken( return", "bad.py")
        assert score.score == 0.0
        assert "SyntaxError" in score.details

    def test_complex_code_lower_score(self, scorer):
        # Build a complex function
        lines = ["def monster(data):"]
        for i in range(60):
            indent = "    "
            if i % 5 == 0:
                lines.append(f"{indent}if data.field{i}:")
                indent = "        "
            lines.append(f"{indent}x{i} = data.value{i}")
        lines.append("    return x0")
        code = "\n".join(lines)
        score = scorer.score(code, "complex.py")
        assert score.score < 1.0

    def test_deterministic(self, scorer):
        code = "def foo(x):\n    return x + 1"
        s1 = scorer.score(code, "t.py")
        s2 = scorer.score(code, "t.py")
        assert s1.score == s2.score

    def test_configurable_thresholds(self):
        scorer = MaintainabilityScorer(
            complexity_threshold=3,
            function_length_threshold=10,
            file_length_threshold=50,
            nesting_depth_threshold=2,
        )
        code = """\
def moderate(x, y):
    if x > 0:
        if y > 0:
            for i in range(x):
                print(i)
    return x + y
"""
        score = scorer.score(code, "test.py")
        # With stricter thresholds, this should score lower
        assert score.score < 1.0

    def test_scope_creep_detector(self, scorer):
        # A 2000-line file should be penalized on file_length (weight 0.15)
        lines = [f"x{i} = {i}" for i in range(2000)]
        code = "\n".join(lines)
        score = scorer.score(code, "huge.py")
        assert score.score < 1.0
        assert "exceeds" in score.details

    def test_details_contain_percentages(self, scorer):
        code = "def foo():\n    return 1"
        score = scorer.score(code, "test.py")
        assert "%" in score.details
