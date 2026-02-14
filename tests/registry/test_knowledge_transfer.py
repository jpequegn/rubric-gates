"""Tests for knowledge transfer document generator."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from shared.models import ScorecardSummary, ScoreResult, ToolRegistryEntry, ToolTier

from registry.workflows.knowledge_transfer import (
    CodeAnalysis,
    KnowledgeTransferGenerator,
)


# --- Helpers ---


def _make_tool(
    name: str = "Expense Categorizer",
    tier: ToolTier = ToolTier.T2,
    description: str = "Categorizes expense reports by department",
    created_by: str = "jane",
    tech_owner: str = "dave",
    users: list[str] | None = None,
    tags: list[str] | None = None,
    source_path: str = "",
    composite: float = 0.72,
    scores: dict[str, float] | None = None,
    red_flags: int = 0,
    total_scores: int = 23,
    trend: str = "improving",
) -> ToolRegistryEntry:
    return ToolRegistryEntry(
        name=name,
        tier=tier,
        description=description,
        created_by=created_by,
        tech_owner=tech_owner,
        users=users or ["jane", "bob"],
        tags=tags or ["finance", "automation"],
        source_path=source_path,
        scorecard=ScorecardSummary(
            latest_composite=composite,
            latest_scores=scores
            or {
                "correctness": 0.85,
                "security": 0.90,
                "maintainability": 0.55,
                "documentation": 0.60,
                "testability": 0.40,
            },
            trend=trend,
            total_scores=total_scores,
            red_flags=red_flags,
        ),
    )


# --- Header ---


class TestHeader:
    def test_includes_tool_name(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(name="My Tool")
        doc = gen.generate(tool)
        assert "# Knowledge Transfer: My Tool" in doc


# --- Overview ---


class TestOverview:
    def test_includes_purpose(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(description="Handles finance reports")
        doc = gen.generate(tool)
        assert "Handles finance reports" in doc

    def test_includes_tier(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(tier=ToolTier.T2)
        doc = gen.generate(tool)
        assert "T2" in doc

    def test_includes_tech_owner(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(tech_owner="dave")
        doc = gen.generate(tool)
        assert "dave" in doc

    def test_includes_users(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(users=["alice", "bob"])
        doc = gen.generate(tool)
        assert "alice" in doc
        assert "bob" in doc

    def test_includes_tags(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(tags=["finance", "automation"])
        doc = gen.generate(tool)
        assert "finance" in doc

    def test_no_description_shows_not_documented(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(description="")
        doc = gen.generate(tool)
        assert "Not documented" in doc

    def test_no_tech_owner_shows_unassigned(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(tech_owner=None)
        doc = gen.generate(tool)
        assert "Unassigned" in doc


# --- Architecture ---


class TestArchitecture:
    def test_no_source_shows_not_available(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(source_path="")
        doc = gen.generate(tool)
        assert "not available" in doc.lower()

    def test_with_source_file(self, tmp_path):
        source = tmp_path / "tool.py"
        source.write_text(
            "import requests\n"
            "import json\n"
            "\n"
            "class Categorizer:\n"
            "    def categorize(self, data):\n"
            "        return data\n"
            "\n"
            "def helper():\n"
            "    pass\n"
        )
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(source_path=str(source))
        doc = gen.generate(tool)
        assert "Categorizer" in doc
        assert "requests" in doc


# --- Quality Profile ---


class TestQualityProfile:
    def test_includes_composite_score(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(composite=0.72)
        doc = gen.generate(tool)
        assert "0.72" in doc

    def test_includes_trend(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(trend="improving")
        doc = gen.generate(tool)
        assert "improving" in doc

    def test_includes_strongest_dimensions(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(
            scores={
                "correctness": 0.85,
                "security": 0.90,
                "maintainability": 0.55,
                "documentation": 0.60,
                "testability": 0.40,
            }
        )
        doc = gen.generate(tool)
        assert "security" in doc.lower()
        assert "Strongest" in doc

    def test_red_flags_shown(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(red_flags=2)
        doc = gen.generate(tool)
        assert "2" in doc
        assert "red-tier" in doc.lower()

    def test_no_red_flags(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(red_flags=0)
        doc = gen.generate(tool)
        assert "None" in doc


# --- Risk Assessment ---


class TestRiskAssessment:
    def test_test_coverage_comprehensive(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(scores={"testability": 0.9})
        doc = gen.generate(tool)
        assert "Comprehensive" in doc

    def test_test_coverage_basic(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(scores={"testability": 0.6})
        doc = gen.generate(tool)
        assert "Basic" in doc

    def test_test_coverage_insufficient(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(scores={"testability": 0.2})
        doc = gen.generate(tool)
        assert "Insufficient" in doc

    def test_bus_factor_single(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(users=["alice"])
        doc = gen.generate(tool)
        assert "high risk" in doc.lower()

    def test_bus_factor_multiple(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(users=["alice", "bob", "charlie"])
        doc = gen.generate(tool)
        assert "3" in doc


# --- Maintenance Guide ---


class TestMaintenanceGuide:
    def test_includes_source_location(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(source_path="/tools/categorizer.py")
        doc = gen.generate(tool)
        assert "/tools/categorizer.py" in doc

    def test_known_issues_from_low_scores(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(
            scores={
                "correctness": 0.85,
                "testability": 0.3,
            }
        )
        doc = gen.generate(tool)
        assert "testability" in doc.lower()
        assert "needs improvement" in doc.lower()

    def test_detects_main_block(self, tmp_path):
        source = tmp_path / "tool.py"
        source.write_text(
            'import argparse\n\ndef main():\n    pass\n\nif __name__ == "__main__":\n    main()\n'
        )
        gen = KnowledgeTransferGenerator()
        tool = _make_tool(source_path=str(source))
        doc = gen.generate(tool)
        assert "How to run" in doc


# --- Scorecard History ---


class TestScorecardHistory:
    def test_no_storage_shows_not_available(self):
        gen = KnowledgeTransferGenerator(storage=None)
        tool = _make_tool()
        doc = gen.generate(tool)
        assert "not available" in doc.lower()

    def test_no_scores_shows_no_events(self):
        storage = MagicMock()
        storage.query.return_value = []
        gen = KnowledgeTransferGenerator(storage=storage)
        tool = _make_tool(source_path="/tools/test.py")
        doc = gen.generate(tool)
        assert "No scoring events" in doc

    def test_renders_score_table(self):
        now = datetime.now()
        scores = [
            ScoreResult(
                user="alice",
                composite_score=0.75,
                files_touched=["/tools/test.py"],
                timestamp=now - timedelta(days=1),
            ),
            ScoreResult(
                user="bob",
                composite_score=0.80,
                files_touched=["/tools/test.py"],
                timestamp=now,
            ),
        ]
        storage = MagicMock()
        storage.query.return_value = scores
        gen = KnowledgeTransferGenerator(storage=storage)
        tool = _make_tool(source_path="/tools/test.py")
        doc = gen.generate(tool)
        assert "| Date |" in doc
        assert "0.75" in doc
        assert "0.80" in doc
        assert "alice" in doc
        assert "bob" in doc


# --- CodeAnalysis ---


class TestCodeAnalysis:
    def test_basic_analysis(self, tmp_path):
        source = tmp_path / "test.py"
        code = (
            "import os\nimport requests\n\nclass MyClass:\n    pass\n\ndef my_func():\n    pass\n"
        )
        source.write_text(code)
        analysis = CodeAnalysis(source, code)

        assert analysis.line_count == 8
        assert "MyClass" in analysis.classes
        assert "my_func" in analysis.functions
        assert "os" in analysis.stdlib_imports
        assert "requests" in analysis.third_party_imports

    def test_syntax_error_handled(self, tmp_path):
        source = tmp_path / "bad.py"
        code = "def broken(\n"
        source.write_text(code)
        analysis = CodeAnalysis(source, code)
        assert analysis.tree is None
        assert analysis.line_count == 1


# --- Full Document ---


class TestFullDocument:
    def test_all_sections_present(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool()
        doc = gen.generate(tool)

        assert "# Knowledge Transfer" in doc
        assert "## Overview" in doc
        assert "## Architecture" in doc
        assert "## Quality Profile" in doc
        assert "## Risk Assessment" in doc
        assert "## Maintenance Guide" in doc
        assert "## Scorecard History" in doc

    def test_document_is_valid_markdown(self):
        gen = KnowledgeTransferGenerator()
        tool = _make_tool()
        doc = gen.generate(tool)
        # Should start with a heading and end with newline
        assert doc.startswith("# ")
        assert doc.endswith("\n")
