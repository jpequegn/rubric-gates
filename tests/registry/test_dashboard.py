"""Tests for registry dashboard views."""

from shared.models import ScorecardSummary, ToolRegistryEntry, ToolTier

from registry.catalog.catalog import ToolCatalog
from registry.dashboard.views import (
    render_heatmap,
    render_inventory,
    render_ownership,
    render_pipeline,
)
from registry.graduation.triggers import GraduationSuggestion
from registry.workflows.engine import GraduationWorkflow


# --- Helpers ---


def _make_tool(
    name: str = "Tool A",
    tier: ToolTier = ToolTier.T1,
    tech_owner: str | None = None,
    users: list[str] | None = None,
    composite: float = 0.7,
    scores: dict[str, float] | None = None,
    trend: str = "stable",
    red_flags: int = 0,
) -> ToolRegistryEntry:
    return ToolRegistryEntry(
        name=name,
        tier=tier,
        description=f"Description of {name}",
        tech_owner=tech_owner,
        users=users or ["alice"],
        scorecard=ScorecardSummary(
            latest_composite=composite,
            latest_scores=scores or {"correctness": 0.8, "security": 0.7},
            trend=trend,
            red_flags=red_flags,
        ),
    )


def _register(catalog: ToolCatalog, *tools: ToolRegistryEntry) -> None:
    for t in tools:
        catalog.register(t)


# --- Inventory ---


class TestInventory:
    def test_empty_catalog(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        output = render_inventory(catalog)
        assert "No tools registered" in output

    def test_lists_tools(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        _register(catalog, _make_tool(name="Alpha"), _make_tool(name="Beta"))
        output = render_inventory(catalog)
        assert "Alpha" in output
        assert "Beta" in output
        assert "| Tool |" in output

    def test_sorted_by_tier_descending(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        _register(
            catalog,
            _make_tool(name="T0 Tool", tier=ToolTier.T0),
            _make_tool(name="T3 Tool", tier=ToolTier.T3),
            _make_tool(name="T1 Tool", tier=ToolTier.T1),
        )
        output = render_inventory(catalog)
        t3_pos = output.index("T3 Tool")
        t1_pos = output.index("T1 Tool")
        t0_pos = output.index("T0 Tool")
        assert t3_pos < t1_pos < t0_pos

    def test_warns_unowned_t2(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        _register(catalog, _make_tool(name="Orphan", tier=ToolTier.T2, tech_owner=None))
        output = render_inventory(catalog)
        assert "no tech owner" in output.lower()

    def test_warns_declining(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        _register(catalog, _make_tool(name="Falling", trend="declining"))
        output = render_inventory(catalog)
        assert "declining" in output.lower()

    def test_shows_score(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        _register(catalog, _make_tool(name="Scored", composite=0.85))
        output = render_inventory(catalog)
        assert "0.85" in output


# --- Pipeline ---


class TestPipeline:
    def test_no_suggestions_or_workflows(self, tmp_path):
        workflow = GraduationWorkflow(
            catalog=ToolCatalog(data_dir=tmp_path / "cat"),
            storage_path=tmp_path / "wf",
        )
        output = render_pipeline([], workflow)
        assert "No pending" in output

    def test_shows_suggestions(self, tmp_path):
        workflow = GraduationWorkflow(
            catalog=ToolCatalog(data_dir=tmp_path / "cat"),
            storage_path=tmp_path / "wf",
        )
        suggestions = [
            GraduationSuggestion(
                tool_slug="my-tool",
                current_tier=ToolTier.T0,
                suggested_tier=ToolTier.T1,
                trigger_reason="Second user detected",
            ),
        ]
        output = render_pipeline(suggestions, workflow)
        assert "my-tool" in output
        assert "Second user" in output
        assert "Pending Suggestions" in output

    def test_shows_active_workflows(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path / "cat")
        catalog.register(_make_tool(name="My Tool", tier=ToolTier.T0, users=["a", "b"]))
        workflow = GraduationWorkflow(
            catalog=catalog,
            storage_path=tmp_path / "wf",
            auto_approve_t0_t1=False,
        )
        workflow.nominate("my-tool", ToolTier.T1, nominated_by="alice")

        output = render_pipeline([], workflow)
        assert "Active Workflows" in output
        assert "my-tool" in output
        assert "alice" in output


# --- Ownership ---


class TestOwnership:
    def test_no_t2_tools(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        _register(catalog, _make_tool(name="T1", tier=ToolTier.T1))
        output = render_ownership(catalog)
        assert "No T2+" in output

    def test_coverage_stats(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        _register(
            catalog,
            _make_tool(name="Owned", tier=ToolTier.T2, tech_owner="dave"),
            _make_tool(name="Orphan", tier=ToolTier.T2, tech_owner=None),
        )
        output = render_ownership(catalog)
        assert "1/2" in output

    def test_owner_load(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        _register(
            catalog,
            _make_tool(name="Tool A", tier=ToolTier.T2, tech_owner="dave"),
            _make_tool(name="Tool B", tier=ToolTier.T2, tech_owner="dave"),
            _make_tool(name="Tool C", tier=ToolTier.T2, tech_owner="alice"),
        )
        output = render_ownership(catalog)
        assert "dave" in output
        assert "alice" in output
        assert "Owner Load" in output

    def test_unowned_listed(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        _register(catalog, _make_tool(name="Orphan", tier=ToolTier.T2, tech_owner=None))
        output = render_ownership(catalog)
        assert "Unowned" in output
        assert "Orphan" in output


# --- Heatmap ---


class TestHeatmap:
    def test_empty_catalog(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        output = render_heatmap(catalog)
        assert "No tools registered" in output

    def test_renders_matrix(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        _register(
            catalog,
            _make_tool(
                name="Tool A",
                scores={"correctness": 0.9, "security": 0.3, "documentation": 0.6},
            ),
        )
        output = render_heatmap(catalog)
        assert "Tool A" in output
        assert "[G]" in output  # correctness 0.9
        assert "[R]" in output  # security 0.3
        assert "[Y]" in output  # documentation 0.6

    def test_sorted_by_tier(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        _register(
            catalog,
            _make_tool(name="T0", tier=ToolTier.T0, scores={"correctness": 0.8}),
            _make_tool(name="T2", tier=ToolTier.T2, scores={"correctness": 0.8}),
        )
        output = render_heatmap(catalog)
        t2_pos = output.index("T2")
        t0_pos = output.index("T0")
        assert t2_pos < t0_pos

    def test_systemic_weaknesses(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        _register(
            catalog,
            _make_tool(name="A", scores={"documentation": 0.3}),
            _make_tool(name="B", scores={"documentation": 0.4}),
        )
        output = render_heatmap(catalog)
        assert "Systemic Weaknesses" in output
        assert "documentation" in output

    def test_no_weaknesses_when_scores_high(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        _register(
            catalog,
            _make_tool(name="A", scores={"correctness": 0.9}),
            _make_tool(name="B", scores={"correctness": 0.8}),
        )
        output = render_heatmap(catalog)
        assert "Systemic Weaknesses" not in output

    def test_legend_present(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        _register(catalog, _make_tool(name="A", scores={"correctness": 0.8}))
        output = render_heatmap(catalog)
        assert "Legend" in output

    def test_no_scores_shows_message(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        tool = ToolRegistryEntry(
            name="A",
            description="No scores",
            scorecard=ScorecardSummary(latest_scores={}),
        )
        catalog.register(tool)
        output = render_heatmap(catalog)
        assert "No dimension scores" in output
