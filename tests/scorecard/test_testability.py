"""Tests for testability dimension scorer."""

import ast
from unittest.mock import patch

import pytest

from scorecard.dimensions.testability import (
    TestabilityScorer,
    check_function_purity,
    check_modularity,
    check_test_assertions,
    check_test_file_exists,
    clear_cache,
)
from shared.models import Dimension, ScoringMethod


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_cache()
    yield
    clear_cache()


def _parse(code: str) -> ast.Module:
    return ast.parse(code)


# --- Test File Exists ---


class TestTestFileExists:
    def test_test_file_found(self):
        result = check_test_file_exists(
            "utils.py",
            project_files=["utils.py", "test_utils.py"],
        )
        assert result.score == 1.0

    def test_test_file_in_tests_dir(self):
        result = check_test_file_exists(
            "utils.py",
            project_files=["utils.py", "tests/test_utils.py"],
        )
        assert result.score == 1.0

    def test_no_test_file(self):
        result = check_test_file_exists(
            "utils.py",
            project_files=["utils.py", "main.py"],
        )
        assert result.score == 0.0

    def test_no_project_files(self):
        result = check_test_file_exists("utils.py", project_files=None)
        assert result.score == 0.0

    def test_no_filename(self):
        result = check_test_file_exists("", project_files=["test_foo.py"])
        assert result.score == 0.0

    def test_suffix_pattern(self):
        result = check_test_file_exists(
            "utils.py",
            project_files=["utils.py", "utils_test.py"],
        )
        assert result.score == 1.0


# --- Test Assertions ---


class TestTestAssertions:
    def test_has_assertions(self):
        test_code = """\
def test_add():
    assert add(1, 2) == 3
    assert add(0, 0) == 0
"""
        result = check_test_assertions(test_code)
        assert result.score == 1.0
        assert "2 assertions" in result.details

    def test_only_prints(self):
        test_code = """\
def test_add():
    print(add(1, 2))
    print(add(0, 0))
"""
        result = check_test_assertions(test_code)
        assert result.score == pytest.approx(0.2)

    def test_no_assertions_or_prints(self):
        test_code = """\
def test_add():
    result = add(1, 2)
"""
        result = check_test_assertions(test_code)
        assert result.score == 0.0

    def test_no_test_code(self):
        result = check_test_assertions(None)
        assert result.score == 0.5

    def test_syntax_error(self):
        result = check_test_assertions("def broken( return")
        assert result.score == 0.0

    def test_unittest_assertions(self):
        test_code = """\
class TestFoo(unittest.TestCase):
    def test_bar(self):
        self.assertEqual(1, 1)
        self.assertTrue(True)
"""
        result = check_test_assertions(test_code)
        assert result.score == 1.0


# --- Function Purity ---


class TestFunctionPurity:
    def test_pure_functions(self):
        code = """\
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
"""
        result = check_function_purity(_parse(code))
        assert result.score == 1.0

    def test_global_usage(self):
        code = """\
counter = 0

def increment():
    global counter
    counter += 1
"""
        result = check_function_purity(_parse(code))
        assert result.score < 1.0
        assert "increment" in result.details

    def test_nonlocal_usage(self):
        code = """\
def outer():
    x = 0
    def inner():
        nonlocal x
        x += 1
    return inner
"""
        result = check_function_purity(_parse(code))
        assert result.score < 1.0

    def test_self_mutation_ok(self):
        code = """\
class Foo:
    def set_value(self, v):
        self.value = v
"""
        result = check_function_purity(_parse(code))
        assert result.score == 1.0

    def test_no_functions(self):
        code = "x = 1"
        result = check_function_purity(_parse(code))
        assert result.score == 1.0


# --- Modularity ---


class TestModularity:
    def test_well_structured(self):
        code = """\
import os

def process(data):
    return data

class Handler:
    def handle(self):
        pass
"""
        result = check_modularity(_parse(code))
        assert result.score > 0.8

    def test_script_style(self):
        code = """\
x = 1
y = 2
print(x + y)
for i in range(10):
    print(i)
result = x * y
"""
        result = check_modularity(_parse(code))
        assert result.score < 0.5
        assert "Script-style" in result.details

    def test_main_guard_ok(self):
        code = """\
def main():
    print("hello")

if __name__ == "__main__":
    main()
"""
        result = check_modularity(_parse(code))
        assert result.score > 0.8

    def test_empty_module(self):
        result = check_modularity(_parse(""))
        assert result.score == 1.0

    def test_import_only(self):
        code = "import os\nimport sys\nfrom pathlib import Path"
        result = check_modularity(_parse(code))
        assert result.score == 1.0


# --- TestabilityScorer Integration (Rule-based only) ---


class TestTestabilityScorerRuleBased:
    def test_clean_modular_code(self):
        scorer = TestabilityScorer(use_llm=False)
        code = """\
def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b
"""
        score = scorer.score(code, "math_utils.py")
        assert score.dimension == Dimension.TESTABILITY
        assert score.method == ScoringMethod.RULE_BASED
        assert score.score > 0.0

    def test_empty_code(self):
        scorer = TestabilityScorer(use_llm=False)
        score = scorer.score("", "empty.py")
        assert score.score == 0.0
        assert "Empty" in score.details

    def test_syntax_error(self):
        scorer = TestabilityScorer(use_llm=False)
        score = scorer.score("def broken( return", "bad.py")
        assert score.score == 0.0

    def test_with_test_file(self):
        scorer = TestabilityScorer(use_llm=False)
        code = "def add(a, b):\n    return a + b"
        score = scorer.score(
            code,
            "utils.py",
            project_files=["utils.py", "test_utils.py"],
        )
        assert score.score > 0.0

    def test_with_test_code(self):
        scorer = TestabilityScorer(use_llm=False)
        code = "def add(a, b):\n    return a + b"
        test_code = "def test_add():\n    assert add(1, 2) == 3"
        score = scorer.score(code, "utils.py", test_code=test_code)
        assert score.score > 0.0

    def test_deterministic(self):
        scorer = TestabilityScorer(use_llm=False)
        code = "def foo():\n    return 1"
        s1 = scorer.score(code, "t.py")
        clear_cache()
        s2 = scorer.score(code, "t.py")
        assert s1.score == s2.score

    def test_caching(self):
        scorer = TestabilityScorer(use_llm=False)
        code = "def foo():\n    return 1"
        s1 = scorer.score(code, "t.py")
        s2 = scorer.score(code, "t.py")
        assert s1.score == s2.score

    def test_rule_based_details(self):
        scorer = TestabilityScorer(use_llm=False)
        code = "def foo():\n    return 1"
        score = scorer.score(code, "t.py")
        assert "(rule-based only)" in score.details


# --- TestabilityScorer with LLM Mock ---


class TestTestabilityScorerLLM:
    @patch("scorecard.dimensions.testability._call_llm_judge")
    def test_llm_combined_score(self, mock_judge):
        mock_judge.return_value = {
            "isolation": 0.9,
            "mockability": 0.8,
            "edge_cases": 0.7,
            "reasoning": "Good testability",
        }
        scorer = TestabilityScorer(use_llm=True)
        code = "def add(a, b):\n    return a + b"
        score = scorer.score(code, "utils.py")

        assert score.method == ScoringMethod.HYBRID
        assert "[LLM]" in score.details
        assert score.metadata["llm_score"] > 0.0
        assert score.metadata["rule_score"] > 0.0

    @patch("scorecard.dimensions.testability._call_llm_judge")
    def test_llm_failure_fallback(self, mock_judge):
        mock_judge.return_value = None
        scorer = TestabilityScorer(use_llm=True)
        code = "def foo():\n    return 1"
        score = scorer.score(code, "t.py")

        assert score.method == ScoringMethod.RULE_BASED
        assert "(rule-based only)" in score.details

    @patch("scorecard.dimensions.testability._call_llm_judge")
    def test_llm_caching(self, mock_judge):
        mock_judge.return_value = {
            "isolation": 0.8,
            "mockability": 0.8,
            "edge_cases": 0.8,
            "reasoning": "Cached",
        }
        scorer = TestabilityScorer(use_llm=True)
        code = "def foo():\n    return 1"

        scorer.score(code, "t.py")
        scorer.score(code, "t.py")
        assert mock_judge.call_count == 1
