"""Tests for correctness dimension scorer."""

import pytest

from scorecard.dimensions.correctness import (
    CorrectnessScorer,
    check_bare_excepts,
    check_return_consistency,
    check_syntax,
    check_undefined_names,
    check_unreachable_code,
)
from shared.models import Dimension, ScoringMethod


@pytest.fixture()
def scorer():
    return CorrectnessScorer()


# --- Syntax Check ---


class TestSyntaxCheck:
    def test_valid_code(self):
        result, tree = check_syntax("x = 1\nprint(x)")
        assert result.passed
        assert tree is not None

    def test_syntax_error(self):
        result, tree = check_syntax("def foo( return")
        assert not result.passed
        assert tree is None
        assert "SyntaxError" in result.details

    def test_empty_string(self):
        result, tree = check_syntax("")
        assert result.passed  # Empty is valid Python
        assert tree is not None

    def test_complex_valid_code(self):
        code = """
import os
from pathlib import Path

class MyClass:
    def __init__(self, name: str):
        self.name = name

    async def process(self, data: list[int]) -> int:
        return sum(data)

def main():
    obj = MyClass("test")
    print(obj.name)
"""
        result, tree = check_syntax(code)
        assert result.passed


# --- Bare Excepts ---


class TestBareExcepts:
    def _parse(self, code: str):
        import ast

        return ast.parse(code)

    def test_no_excepts(self):
        tree = self._parse("x = 1")
        result = check_bare_excepts(tree)
        assert result.passed

    def test_bare_except(self):
        code = """
try:
    x = 1
except:
    pass
"""
        tree = self._parse(code)
        result = check_bare_excepts(tree)
        assert not result.passed
        assert "Bare except" in result.details

    def test_empty_except_pass(self):
        code = """
try:
    x = 1
except ValueError:
    pass
"""
        tree = self._parse(code)
        result = check_bare_excepts(tree)
        assert not result.passed
        assert "Empty except (pass)" in result.details

    def test_empty_except_ellipsis(self):
        code = """
try:
    x = 1
except TypeError:
    ...
"""
        tree = self._parse(code)
        result = check_bare_excepts(tree)
        assert not result.passed
        assert "Empty except (...)" in result.details

    def test_proper_except(self):
        code = """
try:
    x = 1
except ValueError as e:
    print(f"Error: {e}")
"""
        tree = self._parse(code)
        result = check_bare_excepts(tree)
        assert result.passed

    def test_except_with_logging(self):
        code = """
try:
    risky()
except Exception as e:
    logger.error(e)
    raise
"""
        tree = self._parse(code)
        result = check_bare_excepts(tree)
        assert result.passed


# --- Unreachable Code ---


class TestUnreachableCode:
    def _parse(self, code: str):
        import ast

        return ast.parse(code)

    def test_no_unreachable(self):
        code = """
def foo():
    return 1
"""
        tree = self._parse(code)
        result = check_unreachable_code(tree)
        assert result.passed

    def test_code_after_return(self):
        code = """
def foo():
    return 1
    x = 2
"""
        tree = self._parse(code)
        result = check_unreachable_code(tree)
        assert not result.passed
        assert "Unreachable code after Return" in result.details

    def test_code_after_raise(self):
        code = """
def foo():
    raise ValueError("bad")
    cleanup()
"""
        tree = self._parse(code)
        result = check_unreachable_code(tree)
        assert not result.passed
        assert "Unreachable code after Raise" in result.details

    def test_return_in_if_branch_ok(self):
        code = """
def foo(x):
    if x > 0:
        return x
    return -x
"""
        tree = self._parse(code)
        result = check_unreachable_code(tree)
        assert result.passed

    def test_code_after_break(self):
        code = """
for i in range(10):
    break
    print(i)
"""
        tree = self._parse(code)
        result = check_unreachable_code(tree)
        assert not result.passed
        assert "Break" in result.details


# --- Undefined Names ---


class TestUndefinedNames:
    def _parse(self, code: str):
        import ast

        return ast.parse(code)

    def test_all_defined(self):
        code = """
x = 1
y = x + 2
print(y)
"""
        tree = self._parse(code)
        result = check_undefined_names(tree)
        assert result.passed

    def test_undefined_variable(self):
        code = """
x = 1
y = x + undefined_var
"""
        tree = self._parse(code)
        result = check_undefined_names(tree)
        assert not result.passed
        assert "undefined_var" in result.details

    def test_imported_names_ok(self):
        code = """
import os
from pathlib import Path
p = Path(os.getcwd())
"""
        tree = self._parse(code)
        result = check_undefined_names(tree)
        assert result.passed

    def test_function_params_ok(self):
        code = """
def foo(a, b, *args, **kwargs):
    return a + b
"""
        tree = self._parse(code)
        result = check_undefined_names(tree)
        assert result.passed

    def test_class_names_ok(self):
        code = """
class Foo:
    pass
x = Foo()
"""
        tree = self._parse(code)
        result = check_undefined_names(tree)
        assert result.passed

    def test_for_target_defined(self):
        code = """
for item in [1, 2, 3]:
    print(item)
"""
        tree = self._parse(code)
        result = check_undefined_names(tree)
        assert result.passed

    def test_with_target_defined(self):
        code = """
with open("f") as fh:
    data = fh.read()
"""
        tree = self._parse(code)
        result = check_undefined_names(tree)
        assert result.passed

    def test_wildcard_import_skips(self):
        code = """
from os.path import *
join("a", "b")
"""
        tree = self._parse(code)
        result = check_undefined_names(tree)
        assert result.passed
        assert "Wildcard" in result.details

    def test_walrus_operator(self):
        code = """
data = [1, 2, 3, 4]
if (n := len(data)) > 3:
    print(n)
"""
        tree = self._parse(code)
        result = check_undefined_names(tree)
        assert result.passed

    def test_comprehension_target(self):
        code = """
squares = [x * x for x in range(10)]
"""
        tree = self._parse(code)
        result = check_undefined_names(tree)
        assert result.passed

    def test_tuple_unpacking(self):
        code = """
a, b = 1, 2
print(a + b)
"""
        tree = self._parse(code)
        result = check_undefined_names(tree)
        assert result.passed


# --- Return Consistency ---


class TestReturnConsistency:
    def _parse(self, code: str):
        import ast

        return ast.parse(code)

    def test_consistent_returns(self):
        code = """
def foo(x):
    if x > 0:
        return x
    return 0
"""
        tree = self._parse(code)
        result = check_return_consistency(tree)
        assert result.passed

    def test_inconsistent_returns(self):
        code = """
def foo(x):
    if x > 0:
        return x
    return
"""
        tree = self._parse(code)
        result = check_return_consistency(tree)
        assert not result.passed
        assert "inconsistent returns" in result.details

    def test_no_returns(self):
        code = """
def foo():
    print("hello")
"""
        tree = self._parse(code)
        result = check_return_consistency(tree)
        assert result.passed

    def test_only_bare_returns(self):
        code = """
def foo(x):
    if x:
        return
    return
"""
        tree = self._parse(code)
        result = check_return_consistency(tree)
        assert result.passed

    def test_only_value_returns(self):
        code = """
def foo(x):
    if x > 0:
        return 1
    return -1
"""
        tree = self._parse(code)
        result = check_return_consistency(tree)
        assert result.passed


# --- CorrectnessScorer Integration ---


class TestCorrectnessScorer:
    def test_perfect_code(self, scorer):
        code = """
def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

result = add(1, 2)
print(multiply(result, 3))
"""
        score = scorer.score(code, "main.py")
        assert score.dimension == Dimension.CORRECTNESS
        assert score.method == ScoringMethod.AST_PARSE
        assert score.score == 1.0

    def test_syntax_error_code(self, scorer):
        score = scorer.score("def broken( return", "bad.py")
        assert score.score == 0.0
        assert "SyntaxError" in score.details

    def test_code_with_bare_except(self, scorer):
        code = """
try:
    x = 1
except:
    pass
"""
        score = scorer.score(code, "main.py")
        assert score.score < 1.0
        assert "Bare except" in score.details

    def test_code_with_unreachable(self, scorer):
        code = """
def foo():
    return 1
    x = 2
"""
        score = scorer.score(code, "main.py")
        assert score.score < 1.0
        assert "Unreachable" in score.details

    def test_empty_code(self, scorer):
        score = scorer.score("", "empty.py")
        assert score.score == 0.0
        assert "Empty" in score.details

    def test_whitespace_only(self, scorer):
        score = scorer.score("   \n\n  ", "empty.py")
        assert score.score == 0.0

    def test_score_is_deterministic(self, scorer):
        code = "x = 1\nprint(x)"
        s1 = scorer.score(code, "test.py")
        s2 = scorer.score(code, "test.py")
        assert s1.score == s2.score

    def test_details_are_human_readable(self, scorer):
        code = """
def foo():
    return 1
    dead_code = True
"""
        score = scorer.score(code, "test.py")
        # Should have multiple [pass]/[FAIL] entries
        assert "[" in score.details
        assert "]" in score.details

    def test_complex_clean_code(self, scorer):
        code = """
import os
from pathlib import Path
from typing import Optional

class FileProcessor:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def process(self, filename: str) -> Optional[str]:
        path = self.base_dir / filename
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return f.read()
        except OSError as e:
            print(f"Error reading {path}: {e}")
            return None

def main():
    processor = FileProcessor(os.getcwd())
    content = processor.process("test.txt")
    if content is not None:
        print(f"Read {len(content)} bytes")
"""
        score = scorer.score(code, "processor.py")
        assert score.score == 1.0

    def test_multiple_issues(self, scorer):
        code = """
def foo():
    try:
        x = 1
    except:
        pass
    return x
    dead = True
"""
        score = scorer.score(code, "bad.py")
        # Should have lower score due to both bare except and unreachable code
        assert score.score < 0.8
        assert "Bare except" in score.details
        assert "Unreachable" in score.details

    def test_no_filename(self, scorer):
        score = scorer.score("x = 1", "")
        assert score.score > 0.0  # Should still work without filename
