"""Tests for tool catalog CRUD and promotion."""

import pytest

from shared.models import (
    Dimension,
    DimensionScore,
    ScorecardSummary,
    ScoringMethod,
    ScoreResult,
    ToolRegistryEntry,
    ToolTier,
)

from registry.catalog.catalog import ToolCatalog, _slugify


# --- Helpers ---


def _make_tool(
    name: str = "Test Tool",
    tier: ToolTier = ToolTier.T0,
    description: str = "A test tool",
    users: list[str] | None = None,
    tags: list[str] | None = None,
    tech_owner: str | None = None,
) -> ToolRegistryEntry:
    return ToolRegistryEntry(
        name=name,
        tier=tier,
        description=description,
        users=users or ["alice"],
        tags=tags or [],
        tech_owner=tech_owner,
    )


# --- Slugify ---


class TestSlugify:
    def test_basic_name(self):
        assert _slugify("My Tool") == "my-tool"

    def test_special_chars(self):
        assert _slugify("Expense Categorizer (v2)") == "expense-categorizer-v2"

    def test_leading_trailing(self):
        assert _slugify("  My Tool  ") == "my-tool"

    def test_multiple_spaces(self):
        assert _slugify("My   Great   Tool") == "my-great-tool"

    def test_already_slug(self):
        assert _slugify("my-tool") == "my-tool"


# --- Register ---


class TestRegister:
    def test_register_new_tool(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        tool = _make_tool(name="Expense Categorizer")
        slug = catalog.register(tool)
        assert slug == "expense-categorizer"
        assert (tmp_path / "expense-categorizer.yaml").exists()

    def test_register_with_explicit_slug(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        tool = _make_tool(name="My Tool")
        tool = tool.model_copy(update={"slug": "custom-slug"})
        slug = catalog.register(tool)
        assert slug == "custom-slug"

    def test_register_duplicate_raises(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        tool = _make_tool(name="My Tool")
        catalog.register(tool)
        with pytest.raises(ValueError, match="already exists"):
            catalog.register(tool)

    def test_register_returns_slug(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        tool = _make_tool(name="Data Pipeline")
        slug = catalog.register(tool)
        assert slug == "data-pipeline"


# --- Get ---


class TestGet:
    def test_get_existing(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        tool = _make_tool(name="My Tool", description="Does stuff")
        catalog.register(tool)

        loaded = catalog.get("my-tool")
        assert loaded is not None
        assert loaded.name == "My Tool"
        assert loaded.description == "Does stuff"

    def test_get_nonexistent(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        assert catalog.get("nope") is None

    def test_get_preserves_all_fields(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        tool = _make_tool(
            name="Full Tool",
            tier=ToolTier.T1,
            users=["alice", "bob"],
            tags=["finance", "automation"],
            tech_owner="dave",
        )
        catalog.register(tool)

        loaded = catalog.get("full-tool")
        assert loaded is not None
        assert loaded.tier == ToolTier.T1
        assert loaded.users == ["alice", "bob"]
        assert loaded.tags == ["finance", "automation"]
        assert loaded.tech_owner == "dave"


# --- List ---


class TestList:
    def test_list_all(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool B"))
        catalog.register(_make_tool(name="Tool A"))
        catalog.register(_make_tool(name="Tool C"))

        tools = catalog.list()
        assert len(tools) == 3
        assert tools[0].name == "Tool A"  # Sorted by name

    def test_list_empty(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        assert catalog.list() == []

    def test_filter_by_tier(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="T0 Tool", tier=ToolTier.T0))
        catalog.register(_make_tool(name="T1 Tool", tier=ToolTier.T1))

        t1_tools = catalog.list(tier=ToolTier.T1)
        assert len(t1_tools) == 1
        assert t1_tools[0].name == "T1 Tool"

    def test_filter_by_owner(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool A", tech_owner="alice"))
        catalog.register(_make_tool(name="Tool B", tech_owner="bob"))

        alice_tools = catalog.list(owner="alice")
        assert len(alice_tools) == 1
        assert alice_tools[0].tech_owner == "alice"

    def test_filter_by_tag(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Finance Tool", tags=["finance"]))
        catalog.register(_make_tool(name="HR Tool", tags=["hr"]))

        finance = catalog.list(tag="finance")
        assert len(finance) == 1
        assert finance[0].name == "Finance Tool"

    def test_filter_combined(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(
            _make_tool(name="A", tier=ToolTier.T1, tech_owner="alice", tags=["finance"])
        )
        catalog.register(_make_tool(name="B", tier=ToolTier.T1, tech_owner="bob", tags=["finance"]))
        catalog.register(
            _make_tool(name="C", tier=ToolTier.T2, tech_owner="alice", tags=["finance"])
        )

        results = catalog.list(tier=ToolTier.T1, owner="alice")
        assert len(results) == 1
        assert results[0].name == "A"


# --- Update ---


class TestUpdate:
    def test_update_description(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="My Tool"))

        updated = catalog.update("my-tool", {"description": "New description"})
        assert updated.description == "New description"

        # Persisted
        reloaded = catalog.get("my-tool")
        assert reloaded is not None
        assert reloaded.description == "New description"

    def test_update_tech_owner(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="My Tool"))

        updated = catalog.update("my-tool", {"tech_owner": "bob"})
        assert updated.tech_owner == "bob"

    def test_update_nonexistent_raises(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        with pytest.raises(KeyError, match="not found"):
            catalog.update("nope", {"description": "x"})


# --- Delete ---


class TestDelete:
    def test_delete_existing(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="My Tool"))

        assert catalog.delete("my-tool")
        assert catalog.get("my-tool") is None

    def test_delete_nonexistent(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        assert not catalog.delete("nope")


# --- Promote ---


class TestPromote:
    def test_promote_t0_to_t1(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="My Tool", tier=ToolTier.T0))

        promoted = catalog.promote("my-tool", ToolTier.T1, reason="Second user", approved_by="auto")
        assert promoted.tier == ToolTier.T1
        assert len(promoted.graduation_history) == 1
        assert promoted.graduation_history[0].from_tier == ToolTier.T0
        assert promoted.graduation_history[0].to_tier == ToolTier.T1
        assert promoted.graduation_history[0].reason == "Second user"

    def test_promote_persisted(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="My Tool", tier=ToolTier.T0))
        catalog.promote("my-tool", ToolTier.T1)

        reloaded = catalog.get("my-tool")
        assert reloaded is not None
        assert reloaded.tier == ToolTier.T1

    def test_promote_invalid_transition_raises(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="My Tool", tier=ToolTier.T0))

        with pytest.raises(ValueError, match="Cannot promote"):
            catalog.promote("my-tool", ToolTier.T3)

    def test_promote_nonexistent_raises(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        with pytest.raises(KeyError, match="not found"):
            catalog.promote("nope", ToolTier.T1)

    def test_promote_accumulates_history(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="My Tool", tier=ToolTier.T0))

        catalog.promote("my-tool", ToolTier.T1, reason="Shared")
        catalog.promote("my-tool", ToolTier.T2, reason="Team adoption")

        tool = catalog.get("my-tool")
        assert tool is not None
        assert tool.tier == ToolTier.T2
        assert len(tool.graduation_history) == 2


# --- Update Scorecard ---


class TestUpdateScorecard:
    def test_update_scorecard(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="My Tool"))

        score = ScoreResult(
            user="alice",
            composite_score=0.85,
            dimension_scores=[
                DimensionScore(
                    dimension=Dimension.CORRECTNESS,
                    score=0.9,
                    method=ScoringMethod.RULE_BASED,
                ),
            ],
        )
        catalog.update_scorecard("my-tool", score)

        tool = catalog.get("my-tool")
        assert tool is not None
        assert tool.scorecard.latest_composite == 0.85
        assert tool.scorecard.latest_scores["correctness"] == 0.9
        assert tool.scorecard.total_scores == 1

    def test_scorecard_trend_improving(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        tool = _make_tool(name="My Tool")
        tool = tool.model_copy(update={"scorecard": ScorecardSummary(latest_composite=0.5)})
        catalog.register(tool)

        score = ScoreResult(user="alice", composite_score=0.8)
        catalog.update_scorecard("my-tool", score)

        reloaded = catalog.get("my-tool")
        assert reloaded is not None
        assert reloaded.scorecard.trend == "improving"

    def test_scorecard_trend_declining(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        tool = _make_tool(name="My Tool")
        tool = tool.model_copy(update={"scorecard": ScorecardSummary(latest_composite=0.9)})
        catalog.register(tool)

        score = ScoreResult(user="alice", composite_score=0.5)
        catalog.update_scorecard("my-tool", score)

        reloaded = catalog.get("my-tool")
        assert reloaded is not None
        assert reloaded.scorecard.trend == "declining"

    def test_scorecard_trend_stable(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        tool = _make_tool(name="My Tool")
        tool = tool.model_copy(update={"scorecard": ScorecardSummary(latest_composite=0.8)})
        catalog.register(tool)

        score = ScoreResult(user="alice", composite_score=0.82)
        catalog.update_scorecard("my-tool", score)

        reloaded = catalog.get("my-tool")
        assert reloaded is not None
        assert reloaded.scorecard.trend == "stable"

    def test_scorecard_increments_total(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="My Tool"))

        for _ in range(3):
            score = ScoreResult(user="alice", composite_score=0.7)
            catalog.update_scorecard("my-tool", score)

        tool = catalog.get("my-tool")
        assert tool is not None
        assert tool.scorecard.total_scores == 3

    def test_scorecard_nonexistent_raises(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        score = ScoreResult(user="alice", composite_score=0.7)
        with pytest.raises(KeyError, match="not found"):
            catalog.update_scorecard("nope", score)


# --- Search ---


class TestSearch:
    def test_search_by_name(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Expense Categorizer"))
        catalog.register(_make_tool(name="Data Pipeline"))

        results = catalog.search("expense")
        assert len(results) == 1
        assert results[0].name == "Expense Categorizer"

    def test_search_by_description(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool A", description="Handles finance reports"))
        catalog.register(_make_tool(name="Tool B", description="Processes HR data"))

        results = catalog.search("finance")
        assert len(results) == 1
        assert results[0].name == "Tool A"

    def test_search_case_insensitive(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="My Great Tool"))

        results = catalog.search("GREAT")
        assert len(results) == 1

    def test_search_no_results(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Something"))

        assert catalog.search("xyz") == []
