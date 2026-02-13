"""Tests for documentation dimension scorer."""

import ast
import json
from unittest.mock import MagicMock, patch

import pytest

from scorecard.dimensions.documentation import (
    DocumentationScorer,
    LLMJudgeResult,
    _clamp,
    check_class_docstrings,
    check_comment_density,
    check_function_docstrings,
    check_module_docstring,
    clear_cache,
)
from shared.models import Dimension, ScoringMethod


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear cache before each test."""
    clear_cache()
    yield
    clear_cache()


def _parse(code: str) -> ast.Module:
    return ast.parse(code)


# --- Module Docstring ---


class TestModuleDocstring:
    def test_has_docstring(self):
        code = '"""This module does stuff."""\nx = 1'
        result = check_module_docstring(_parse(code))
        assert result.score == 1.0

    def test_no_docstring(self):
        code = "x = 1\ny = 2"
        result = check_module_docstring(_parse(code))
        assert result.score == 0.0

    def test_empty_module(self):
        result = check_module_docstring(_parse(""))
        assert result.score == 0.0


# --- Function Docstrings ---


class TestFunctionDocstrings:
    def test_all_documented(self):
        code = '''\
def foo():
    """Does foo."""
    return 1

def bar():
    """Does bar."""
    return 2
'''
        result = check_function_docstrings(_parse(code))
        assert result.score == 1.0

    def test_none_documented(self):
        code = """\
def foo():
    return 1

def bar():
    return 2
"""
        result = check_function_docstrings(_parse(code))
        assert result.score == 0.0
        assert "2/2" in result.details

    def test_partially_documented(self):
        code = '''\
def foo():
    """Has docstring."""
    return 1

def bar():
    return 2
'''
        result = check_function_docstrings(_parse(code))
        assert result.score == pytest.approx(0.5)

    def test_private_skipped(self):
        code = """\
def _private():
    return 1

def __dunder__():
    return 2
"""
        result = check_function_docstrings(_parse(code))
        assert result.score == 1.0
        assert "No public functions" in result.details

    def test_no_functions(self):
        code = "x = 1"
        result = check_function_docstrings(_parse(code))
        assert result.score == 1.0


# --- Comment Density ---


class TestCommentDensity:
    def test_well_commented(self):
        code = """\
# Initialize configuration
config = load_config()
# Process the data
result = process(config)
"""
        result = check_comment_density(code)
        assert result.score == 1.0

    def test_no_comments(self):
        code = """\
x = 1
y = 2
z = x + y
print(z)
"""
        result = check_comment_density(code)
        # 0% density = 0.3 (self-documenting baseline)
        assert result.score == pytest.approx(0.3)

    def test_inline_comments_counted(self):
        code = """\
x = 1  # first value
y = 2  # second value
z = x + y
"""
        result = check_comment_density(code)
        assert result.score > 0.3

    def test_empty_code(self):
        result = check_comment_density("")
        assert result.score == 1.0

    def test_density_in_details(self):
        code = "x = 1\n# comment\ny = 2"
        result = check_comment_density(code)
        assert "density" in result.details


# --- Class Docstrings ---


class TestClassDocstrings:
    def test_documented_class(self):
        code = '''\
class Foo:
    """A foo class."""
    pass
'''
        result = check_class_docstrings(_parse(code))
        assert result.score == 1.0

    def test_undocumented_class(self):
        code = """\
class Foo:
    pass
"""
        result = check_class_docstrings(_parse(code))
        assert result.score == 0.0

    def test_private_class_skipped(self):
        code = """\
class _Internal:
    pass
"""
        result = check_class_docstrings(_parse(code))
        assert result.score == 1.0

    def test_no_classes(self):
        code = "x = 1"
        result = check_class_docstrings(_parse(code))
        assert result.score == 1.0


# --- LLM Judge Result ---


class TestLLMJudgeResult:
    def test_weighted_score(self):
        result = LLMJudgeResult(
            logic_explanation=1.0,
            function_docs=1.0,
            purpose_docs=1.0,
            comment_accuracy=1.0,
            reasoning="All good",
        )
        assert result.weighted_score == 1.0

    def test_weighted_score_mixed(self):
        result = LLMJudgeResult(
            logic_explanation=0.5,
            function_docs=0.5,
            purpose_docs=0.5,
            comment_accuracy=0.5,
            reasoning="Meh",
        )
        assert result.weighted_score == pytest.approx(0.5)

    def test_weighted_score_zero(self):
        result = LLMJudgeResult(
            logic_explanation=0.0,
            function_docs=0.0,
            purpose_docs=0.0,
            comment_accuracy=0.0,
            reasoning="Bad",
        )
        assert result.weighted_score == 0.0


class TestClamp:
    def test_normal(self):
        assert _clamp(0.5) == 0.5

    def test_below(self):
        assert _clamp(-1.0) == 0.0

    def test_above(self):
        assert _clamp(2.0) == 1.0

    def test_invalid(self):
        assert _clamp("bad") == 0.5


# --- DocumentationScorer with LLM Mock ---


class TestDocumentationScorerLLM:
    def _mock_anthropic_response(self, data: dict):
        """Create a mock Anthropic API response."""
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = json.dumps(data)
        mock_response.content = [mock_content]
        return mock_response

    @patch("scorecard.dimensions.documentation.anthropic", create=True)
    def test_llm_judge_success(self, mock_anthropic_module):
        mock_client = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = self._mock_anthropic_response(
            {
                "logic_explanation": 0.9,
                "function_docs": 0.8,
                "purpose_docs": 0.7,
                "comment_accuracy": 1.0,
                "reasoning": "Good documentation overall",
            }
        )

        with patch("scorecard.dimensions.documentation._call_llm_judge") as mock_judge:
            mock_judge.return_value = LLMJudgeResult(
                logic_explanation=0.9,
                function_docs=0.8,
                purpose_docs=0.7,
                comment_accuracy=1.0,
                reasoning="Good documentation overall",
            )
            scorer = DocumentationScorer(use_llm=True)
            score = scorer.score("def foo():\n    return 1", "test.py")

            assert score.dimension == Dimension.DOCUMENTATION
            assert score.method == ScoringMethod.LLM_JUDGE
            assert 0.0 <= score.score <= 1.0
            assert "Good documentation" in score.details

    @patch("scorecard.dimensions.documentation._call_llm_judge")
    def test_llm_failure_falls_back(self, mock_judge):
        mock_judge.return_value = None

        scorer = DocumentationScorer(use_llm=True)
        score = scorer.score("def foo():\n    return 1", "test.py")

        assert score.method == ScoringMethod.RULE_BASED
        assert "(fallback)" in score.details

    @patch("scorecard.dimensions.documentation._call_llm_judge")
    def test_caching(self, mock_judge):
        mock_judge.return_value = LLMJudgeResult(
            logic_explanation=0.8,
            function_docs=0.8,
            purpose_docs=0.8,
            comment_accuracy=0.8,
            reasoning="Cached",
        )
        scorer = DocumentationScorer(use_llm=True)
        code = "def foo():\n    return 1"

        s1 = scorer.score(code, "test.py")
        s2 = scorer.score(code, "test.py")

        assert s1.score == s2.score
        # LLM should only be called once due to caching
        assert mock_judge.call_count == 1


# --- DocumentationScorer Fallback Mode ---


class TestDocumentationScorerFallback:
    def test_fallback_mode(self):
        scorer = DocumentationScorer(use_llm=False)
        code = '''\
"""Module docstring."""

def foo():
    """Does foo."""
    # Important logic
    return 1
'''
        score = scorer.score(code, "test.py")
        assert score.method == ScoringMethod.RULE_BASED
        assert "(fallback)" in score.details
        assert score.score > 0.5

    def test_empty_code(self):
        scorer = DocumentationScorer(use_llm=False)
        score = scorer.score("", "empty.py")
        assert score.score == 0.0
        assert "Empty" in score.details

    def test_whitespace_only(self):
        scorer = DocumentationScorer(use_llm=False)
        score = scorer.score("   \n\n  ", "empty.py")
        assert score.score == 0.0

    def test_syntax_error(self):
        scorer = DocumentationScorer(use_llm=False)
        score = scorer.score("def broken( return", "bad.py")
        assert score.score == 0.0
        assert "SyntaxError" in score.details

    def test_well_documented_code(self):
        scorer = DocumentationScorer(use_llm=False)
        code = '''\
"""User management module."""

class UserService:
    """Handles user operations."""

    def get_user(self, user_id: int):
        """Fetch a user by ID."""
        # Look up in database
        return self.db.find(user_id)

    def create_user(self, name: str):
        """Create a new user."""
        # Validate input first
        if not name:
            raise ValueError("Name required")
        return self.db.insert(name)
'''
        score = scorer.score(code, "users.py")
        assert score.score > 0.8

    def test_poorly_documented_code(self):
        scorer = DocumentationScorer(use_llm=False)
        code = """\
class Foo:
    def bar(self):
        return 1

    def baz(self, x):
        if x > 0:
            return x * 2
        return 0

def process(data):
    result = []
    for item in data:
        if item.valid:
            result.append(item.value)
    return result
"""
        score = scorer.score(code, "messy.py")
        assert score.score < 0.5

    def test_deterministic(self):
        scorer = DocumentationScorer(use_llm=False)
        code = "def foo():\n    return 1"
        s1 = scorer.score(code, "t.py")
        clear_cache()
        s2 = scorer.score(code, "t.py")
        assert s1.score == s2.score
