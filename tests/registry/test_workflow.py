"""Tests for graduation workflow engine."""

import pytest

from shared.models import ToolRegistryEntry, ToolTier

from registry.catalog.catalog import ToolCatalog
from registry.workflows.engine import (
    GraduationWorkflow,
    WorkflowStatus,
)


# --- Helpers ---


def _make_tool(
    name: str = "test-tool",
    tier: ToolTier = ToolTier.T0,
    description: str = "A test tool",
    users: list[str] | None = None,
    tech_owner: str | None = None,
    metadata: dict | None = None,
) -> ToolRegistryEntry:
    return ToolRegistryEntry(
        name=name,
        tier=tier,
        description=description,
        users=users or ["alice"],
        tech_owner=tech_owner,
        metadata=metadata or {},
    )


def _ready_t0_tool() -> ToolRegistryEntry:
    """A T0 tool that meets all T0→T1 requirements."""
    return _make_tool(
        name="Ready Tool",
        tier=ToolTier.T0,
        description="A well-documented tool",
        users=["alice", "bob"],
    )


def _make_workflow(
    tmp_path, auto_approve_t0_t1: bool = True
) -> tuple[ToolCatalog, GraduationWorkflow]:
    catalog = ToolCatalog(data_dir=tmp_path / "catalog")
    workflow = GraduationWorkflow(
        catalog=catalog,
        storage_path=tmp_path / "workflows",
        auto_approve_t0_t1=auto_approve_t0_t1,
    )
    return catalog, workflow


# --- Nominate ---


class TestNominate:
    def test_nominate_creates_workflow(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path, auto_approve_t0_t1=False)
        catalog.register(_ready_t0_tool())

        state = workflow.nominate("ready-tool", ToolTier.T1, nominated_by="alice")
        assert state.tool_slug == "ready-tool"
        assert state.from_tier == ToolTier.T0
        assert state.to_tier == ToolTier.T1
        assert state.nominated_by == "alice"

    def test_nominate_evaluates_readiness(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path, auto_approve_t0_t1=False)
        catalog.register(_ready_t0_tool())

        state = workflow.nominate("ready-tool", ToolTier.T1)
        assert state.ready
        assert state.overall_readiness > 0
        assert len(state.action_items) > 0

    def test_nominate_not_ready_tool(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path, auto_approve_t0_t1=False)
        catalog.register(_make_tool(name="Bare Tool", description="", users=["alice"]))

        state = workflow.nominate("bare-tool", ToolTier.T1)
        assert not state.ready
        assert state.status == WorkflowStatus.PENDING

    def test_nominate_nonexistent_raises(self, tmp_path):
        _, workflow = _make_workflow(tmp_path)
        with pytest.raises(KeyError, match="not found"):
            workflow.nominate("nope", ToolTier.T1)

    def test_nominate_invalid_transition_raises(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path)
        catalog.register(_make_tool(name="My Tool", tier=ToolTier.T0))

        with pytest.raises(ValueError, match="Cannot graduate"):
            workflow.nominate("my-tool", ToolTier.T3)

    def test_nominate_duplicate_pending_raises(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path, auto_approve_t0_t1=False)
        catalog.register(_make_tool(name="My Tool", description="A tool", users=["a", "b"]))

        workflow.nominate("my-tool", ToolTier.T1)
        with pytest.raises(ValueError, match="already has a pending"):
            workflow.nominate("my-tool", ToolTier.T1)


# --- Auto-Approve T0→T1 ---


class TestAutoApprove:
    def test_auto_approve_ready_t0(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path, auto_approve_t0_t1=True)
        catalog.register(_ready_t0_tool())

        state = workflow.nominate("ready-tool", ToolTier.T1)
        assert state.status == WorkflowStatus.PROMOTED
        assert state.reviewed_by == "auto"

        # Tool should be promoted in catalog
        tool = catalog.get("ready-tool")
        assert tool is not None
        assert tool.tier == ToolTier.T1

    def test_auto_approve_disabled(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path, auto_approve_t0_t1=False)
        catalog.register(_ready_t0_tool())

        state = workflow.nominate("ready-tool", ToolTier.T1)
        assert state.status == WorkflowStatus.PENDING

    def test_no_auto_approve_if_not_ready(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path, auto_approve_t0_t1=True)
        catalog.register(_make_tool(name="Bare Tool", description="", users=["alice"]))

        state = workflow.nominate("bare-tool", ToolTier.T1)
        assert state.status == WorkflowStatus.PENDING

    def test_no_auto_approve_for_t1_t2(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path, auto_approve_t0_t1=True)
        catalog.register(
            _make_tool(
                name="T1 Tool",
                tier=ToolTier.T1,
                tech_owner="dave",
                metadata={"dependencies_pinned": True, "code_reviewed": True},
            )
        )
        # Even with auto_approve_t0_t1=True, T1→T2 should not auto-approve
        state = workflow.nominate("t1-tool", ToolTier.T2)
        assert state.status == WorkflowStatus.PENDING


# --- Action Items ---


class TestActionItems:
    def test_get_action_items_for_pending(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path, auto_approve_t0_t1=False)
        catalog.register(_make_tool(name="Bare Tool", description="", users=["alice"]))

        workflow.nominate("bare-tool", ToolTier.T1)
        items = workflow.get_action_items("bare-tool")
        assert len(items) > 0
        assert all(not item.met for item in items)

    def test_no_action_items_if_no_pending(self, tmp_path):
        _, workflow = _make_workflow(tmp_path)
        assert workflow.get_action_items("nope") == []

    def test_action_items_include_details(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path, auto_approve_t0_t1=False)
        catalog.register(_make_tool(name="My Tool", description="", users=["alice"]))

        workflow.nominate("my-tool", ToolTier.T1)
        items = workflow.get_action_items("my-tool")
        assert all(item.details for item in items)


# --- Approve ---


class TestApprove:
    def test_approve_promotes_tool(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path, auto_approve_t0_t1=False)
        catalog.register(_ready_t0_tool())

        state = workflow.nominate("ready-tool", ToolTier.T1)
        result = workflow.approve(state.id, approved_by="dave")

        assert result.status == WorkflowStatus.PROMOTED
        assert result.reviewed_by == "dave"
        assert result.promoted_at is not None

        tool = catalog.get("ready-tool")
        assert tool is not None
        assert tool.tier == ToolTier.T1

    def test_approve_nonexistent_raises(self, tmp_path):
        _, workflow = _make_workflow(tmp_path)
        with pytest.raises(KeyError, match="not found"):
            workflow.approve("nope", approved_by="dave")

    def test_approve_already_approved_raises(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path, auto_approve_t0_t1=False)
        catalog.register(_ready_t0_tool())

        state = workflow.nominate("ready-tool", ToolTier.T1)
        workflow.approve(state.id, approved_by="dave")

        with pytest.raises(ValueError, match="not pending"):
            workflow.approve(state.id, approved_by="another")


# --- Reject ---


class TestReject:
    def test_reject_workflow(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path, auto_approve_t0_t1=False)
        catalog.register(_ready_t0_tool())

        state = workflow.nominate("ready-tool", ToolTier.T1)
        result = workflow.reject(state.id, rejected_by="dave", reason="Not yet")

        assert result.status == WorkflowStatus.REJECTED
        assert result.reviewed_by == "dave"
        assert result.reject_reason == "Not yet"

        # Tool should NOT be promoted
        tool = catalog.get("ready-tool")
        assert tool is not None
        assert tool.tier == ToolTier.T0

    def test_reject_nonexistent_raises(self, tmp_path):
        _, workflow = _make_workflow(tmp_path)
        with pytest.raises(KeyError, match="not found"):
            workflow.reject("nope", rejected_by="dave")

    def test_reject_already_rejected_raises(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path, auto_approve_t0_t1=False)
        catalog.register(_ready_t0_tool())

        state = workflow.nominate("ready-tool", ToolTier.T1)
        workflow.reject(state.id, rejected_by="dave")

        with pytest.raises(ValueError, match="not pending"):
            workflow.reject(state.id, rejected_by="another")


# --- List Pending ---


class TestListPending:
    def test_list_pending(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path, auto_approve_t0_t1=False)
        catalog.register(_make_tool(name="Tool A", description="A", users=["a", "b"]))
        catalog.register(_make_tool(name="Tool B", description="B", users=["a", "b"]))

        workflow.nominate("tool-a", ToolTier.T1)
        workflow.nominate("tool-b", ToolTier.T1)

        pending = workflow.list_pending()
        assert len(pending) == 2

    def test_approved_not_in_pending(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path, auto_approve_t0_t1=False)
        catalog.register(_ready_t0_tool())

        state = workflow.nominate("ready-tool", ToolTier.T1)
        workflow.approve(state.id, approved_by="dave")

        assert workflow.list_pending() == []

    def test_empty_pending(self, tmp_path):
        _, workflow = _make_workflow(tmp_path)
        assert workflow.list_pending() == []


# --- Persistence ---


class TestPersistence:
    def test_state_survives_reload(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path / "catalog")
        catalog.register(_make_tool(name="My Tool", description="A tool", users=["a", "b"]))

        # Create workflow and nominate
        wf1 = GraduationWorkflow(
            catalog=catalog,
            storage_path=tmp_path / "workflows",
            auto_approve_t0_t1=False,
        )
        state = wf1.nominate("my-tool", ToolTier.T1)

        # Create new workflow instance (simulates restart)
        wf2 = GraduationWorkflow(
            catalog=catalog,
            storage_path=tmp_path / "workflows",
            auto_approve_t0_t1=False,
        )
        pending = wf2.list_pending()
        assert len(pending) == 1
        assert pending[0].id == state.id


# --- Get Workflow ---


class TestGetWorkflow:
    def test_get_existing(self, tmp_path):
        catalog, workflow = _make_workflow(tmp_path, auto_approve_t0_t1=False)
        catalog.register(_ready_t0_tool())

        state = workflow.nominate("ready-tool", ToolTier.T1)
        loaded = workflow.get_workflow(state.id)
        assert loaded is not None
        assert loaded.tool_slug == "ready-tool"

    def test_get_nonexistent(self, tmp_path):
        _, workflow = _make_workflow(tmp_path)
        assert workflow.get_workflow("nope") is None
