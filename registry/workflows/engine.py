"""Graduation workflow engine.

Manages the full lifecycle: nominate → evaluate → action items →
approve/reject → promote. Workflow state is persisted as JSONL
so it survives process restarts.

Approval routing:
  T0→T1: auto-approve (configurable) or tech team
  T1→T2: requires tech team approval
  T2→T3: requires tech team + management approval
"""

from __future__ import annotations

import fcntl
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from shared.models import ToolTier

from registry.catalog.catalog import ToolCatalog
from registry.graduation.rubrics import GraduationResult, GraduationRubric


class WorkflowStatus(str, Enum):
    """Status of a graduation workflow."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    PROMOTED = "promoted"


class ActionItem(BaseModel):
    """A specific action needed for graduation readiness."""

    requirement: str
    met: bool = False
    details: str = ""
    blocking: bool = True


class WorkflowState(BaseModel):
    """Persisted state of a graduation workflow request."""

    id: str = ""
    tool_slug: str
    from_tier: ToolTier
    to_tier: ToolTier
    nominated_by: str = ""
    nominated_at: datetime = Field(default_factory=datetime.now)
    status: WorkflowStatus = WorkflowStatus.PENDING
    action_items: list[ActionItem] = Field(default_factory=list)
    ready: bool = False
    overall_readiness: float = 0.0
    reviewed_by: str = ""
    reviewed_at: datetime | None = None
    reject_reason: str = ""
    promoted_at: datetime | None = None


class GraduationWorkflow:
    """Manages graduation workflow lifecycle.

    Args:
        catalog: Tool catalog for reading/promoting tools.
        rubric: Graduation rubric for evaluation.
        storage_path: Path to persist workflow state.
        auto_approve_t0_t1: Whether to auto-approve T0→T1 promotions.
    """

    def __init__(
        self,
        catalog: ToolCatalog,
        rubric: GraduationRubric | None = None,
        storage_path: str | Path | None = None,
        auto_approve_t0_t1: bool = True,
    ) -> None:
        self._catalog = catalog
        self._rubric = rubric or GraduationRubric()
        if storage_path is None:
            storage_path = Path("./rubric-gates-data/workflows")
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._auto_approve_t0_t1 = auto_approve_t0_t1

    def _log_file(self) -> Path:
        return self._storage_path / "workflows.jsonl"

    def _read_all(self) -> list[WorkflowState]:
        target = self._log_file()
        if not target.exists():
            return []
        states: list[WorkflowState] = []
        with open(target) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    states.append(WorkflowState.model_validate_json(line))
                except Exception:
                    continue
        return states

    def _write_all(self, states: list[WorkflowState]) -> None:
        target = self._log_file()
        with open(target, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                for s in states:
                    f.write(s.model_dump_json() + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _append(self, state: WorkflowState) -> None:
        target = self._log_file()
        with open(target, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(state.model_dump_json() + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _update_state(self, workflow_id: str, updates: dict[str, Any]) -> WorkflowState | None:
        states = self._read_all()
        updated: WorkflowState | None = None
        for i, s in enumerate(states):
            if s.id == workflow_id:
                states[i] = s.model_copy(update=updates)
                updated = states[i]
                break
        if updated is not None:
            self._write_all(states)
        return updated

    def _get_state(self, workflow_id: str) -> WorkflowState | None:
        for s in self._read_all():
            if s.id == workflow_id:
                return s
        return None

    def nominate(
        self,
        slug: str,
        target_tier: ToolTier,
        nominated_by: str = "",
    ) -> WorkflowState:
        """Start a graduation workflow for a tool.

        Evaluates readiness immediately and auto-approves T0→T1
        if configured and the tool is ready.

        Args:
            slug: Tool slug.
            target_tier: Target tier.
            nominated_by: Who nominated.

        Returns:
            The workflow state.

        Raises:
            KeyError: If tool not found.
            ValueError: If invalid transition or workflow already pending.
        """
        tool = self._catalog.get(slug)
        if tool is None:
            raise KeyError(f"Tool '{slug}' not found.")

        from registry.graduation.tiers import is_valid_transition

        if not is_valid_transition(tool.tier, target_tier):
            raise ValueError(f"Cannot graduate from {tool.tier.value} to {target_tier.value}.")

        # Check for existing pending workflow
        for s in self._read_all():
            if s.tool_slug == slug and s.status == WorkflowStatus.PENDING:
                raise ValueError(f"Tool '{slug}' already has a pending workflow.")

        # Evaluate readiness
        grad_result = self._rubric.evaluate(tool, target_tier)
        action_items = self._build_action_items(grad_result)

        state = WorkflowState(
            id=str(uuid.uuid4()),
            tool_slug=slug,
            from_tier=tool.tier,
            to_tier=target_tier,
            nominated_by=nominated_by,
            action_items=action_items,
            ready=grad_result.ready,
            overall_readiness=grad_result.overall_readiness,
        )

        # Auto-approve T0→T1 if ready and configured
        if (
            self._auto_approve_t0_t1
            and tool.tier == ToolTier.T0
            and target_tier == ToolTier.T1
            and grad_result.ready
        ):
            state = state.model_copy(
                update={
                    "status": WorkflowStatus.APPROVED,
                    "reviewed_by": "auto",
                    "reviewed_at": datetime.now(),
                }
            )
            self._append(state)
            return self._do_promote(state)

        self._append(state)
        return state

    def get_action_items(self, slug: str) -> list[ActionItem]:
        """Get remaining action items for a tool's pending graduation.

        Args:
            slug: Tool slug.

        Returns:
            List of action items (empty if no pending workflow).
        """
        for s in self._read_all():
            if s.tool_slug == slug and s.status == WorkflowStatus.PENDING:
                return [a for a in s.action_items if not a.met]
        return []

    def approve(self, workflow_id: str, approved_by: str) -> WorkflowState:
        """Approve a graduation request and promote the tool.

        Args:
            workflow_id: Workflow ID.
            approved_by: Who approved.

        Returns:
            Updated workflow state.

        Raises:
            KeyError: If workflow not found.
            ValueError: If workflow is not pending.
        """
        state = self._get_state(workflow_id)
        if state is None:
            raise KeyError(f"Workflow '{workflow_id}' not found.")
        if state.status != WorkflowStatus.PENDING:
            raise ValueError(f"Workflow is not pending (status: {state.status.value}).")

        updated = self._update_state(
            workflow_id,
            {
                "status": WorkflowStatus.APPROVED,
                "reviewed_by": approved_by,
                "reviewed_at": datetime.now(),
            },
        )
        if updated is None:
            raise KeyError(f"Workflow '{workflow_id}' not found.")

        return self._do_promote(updated)

    def reject(
        self,
        workflow_id: str,
        rejected_by: str,
        reason: str = "",
    ) -> WorkflowState:
        """Reject a graduation request.

        Args:
            workflow_id: Workflow ID.
            rejected_by: Who rejected.
            reason: Rejection reason.

        Returns:
            Updated workflow state.

        Raises:
            KeyError: If workflow not found.
            ValueError: If workflow is not pending.
        """
        state = self._get_state(workflow_id)
        if state is None:
            raise KeyError(f"Workflow '{workflow_id}' not found.")
        if state.status != WorkflowStatus.PENDING:
            raise ValueError(f"Workflow is not pending (status: {state.status.value}).")

        updated = self._update_state(
            workflow_id,
            {
                "status": WorkflowStatus.REJECTED,
                "reviewed_by": rejected_by,
                "reviewed_at": datetime.now(),
                "reject_reason": reason,
            },
        )
        if updated is None:
            raise KeyError(f"Workflow '{workflow_id}' not found.")
        return updated

    def list_pending(self) -> list[WorkflowState]:
        """List all pending graduation requests."""
        return [s for s in self._read_all() if s.status == WorkflowStatus.PENDING]

    def get_workflow(self, workflow_id: str) -> WorkflowState | None:
        """Get a workflow by ID."""
        return self._get_state(workflow_id)

    def _do_promote(self, state: WorkflowState) -> WorkflowState:
        """Execute the actual promotion in the catalog."""
        self._catalog.promote(
            state.tool_slug,
            state.to_tier,
            reason=f"Graduated via workflow (nominated by {state.nominated_by})",
            approved_by=state.reviewed_by,
        )
        updated = self._update_state(
            state.id,
            {
                "status": WorkflowStatus.PROMOTED,
                "promoted_at": datetime.now(),
            },
        )
        return updated or state

    @staticmethod
    def _build_action_items(grad_result: GraduationResult) -> list[ActionItem]:
        """Convert graduation checklist to action items."""
        items: list[ActionItem] = []
        for check in grad_result.checklist:
            items.append(
                ActionItem(
                    requirement=check.requirement,
                    met=check.met,
                    details=check.details,
                    blocking=check.blocking,
                )
            )
        return items
