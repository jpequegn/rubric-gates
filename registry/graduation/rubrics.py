"""Graduation rubrics for tier transitions.

Evaluates whether a tool is ready to graduate from one tier to the next.
Rubric criteria are data-driven: loaded from config dicts that can be
provided via YAML configuration.

Each transition (T0→T1, T1→T2, T2→T3) has a rubric defining required
and advisory checklist items. Items are evaluated against the tool's
registry entry and scorecard summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shared.models import ToolRegistryEntry, ToolTier

from registry.graduation.tiers import is_valid_transition


# --- Data Models ---


@dataclass
class ChecklistItem:
    """A single graduation checklist item."""

    requirement: str
    met: bool = False
    details: str = ""
    blocking: bool = True  # If True, must be met to graduate


@dataclass
class GraduationResult:
    """Result of evaluating a tool against graduation criteria."""

    ready: bool
    from_tier: ToolTier
    to_tier: ToolTier
    checklist: list[ChecklistItem] = field(default_factory=list)
    blocking_items: list[ChecklistItem] = field(default_factory=list)
    advisory_items: list[ChecklistItem] = field(default_factory=list)
    overall_readiness: float = 0.0  # 0.0-1.0


# --- Default Rubric Configs ---

# These can be overridden via YAML config. Each criterion has:
#   requirement: human-readable description
#   check: name of the check function to run
#   blocking: whether it blocks graduation
#   params: optional parameters for the check

DEFAULT_T0_TO_T1: list[dict[str, Any]] = [
    {
        "requirement": "Tool has a description (README or header comment)",
        "check": "has_description",
        "blocking": True,
    },
    {
        "requirement": "No unresolved red-tier security issues",
        "check": "no_red_flags",
        "blocking": True,
    },
    {
        "requirement": "At least 2 users registered",
        "check": "min_users",
        "blocking": True,
        "params": {"min": 2},
    },
    {
        "requirement": "Source path is set",
        "check": "has_source_path",
        "blocking": False,
    },
]

DEFAULT_T1_TO_T2: list[dict[str, Any]] = [
    {
        "requirement": "Composite quality score >= 0.7",
        "check": "min_composite_score",
        "blocking": True,
        "params": {"min": 0.7},
    },
    {
        "requirement": "Zero unresolved red-tier issues",
        "check": "no_red_flags",
        "blocking": True,
    },
    {
        "requirement": "Basic test coverage exists",
        "check": "has_test_coverage",
        "blocking": True,
    },
    {
        "requirement": "Tech owner assigned",
        "check": "has_tech_owner",
        "blocking": True,
    },
    {
        "requirement": "Dependencies vetted and pinned",
        "check": "dependencies_pinned",
        "blocking": True,
    },
    {
        "requirement": "Code review completed by tech team member",
        "check": "has_code_review",
        "blocking": False,
    },
]

DEFAULT_T2_TO_T3: list[dict[str, Any]] = [
    {
        "requirement": "Full test coverage (unit + integration)",
        "check": "full_test_coverage",
        "blocking": True,
    },
    {
        "requirement": "Security review completed",
        "check": "security_review_done",
        "blocking": True,
    },
    {
        "requirement": "Error handling and logging implemented",
        "check": "has_error_handling",
        "blocking": True,
    },
    {
        "requirement": "Rollback plan documented",
        "check": "has_rollback_plan",
        "blocking": True,
    },
    {
        "requirement": "Monitoring and alerting configured",
        "check": "has_monitoring",
        "blocking": True,
    },
    {
        "requirement": "Knowledge transfer completed",
        "check": "knowledge_transfer_done",
        "blocking": True,
    },
    {
        "requirement": "Composite quality score >= 0.8",
        "check": "min_composite_score",
        "blocking": True,
        "params": {"min": 0.8},
    },
    {
        "requirement": "At least 3 users",
        "check": "min_users",
        "blocking": False,
        "params": {"min": 3},
    },
]

# Map transitions to their default configs
_DEFAULT_RUBRICS: dict[tuple[ToolTier, ToolTier], list[dict[str, Any]]] = {
    (ToolTier.T0, ToolTier.T1): DEFAULT_T0_TO_T1,
    (ToolTier.T1, ToolTier.T2): DEFAULT_T1_TO_T2,
    (ToolTier.T2, ToolTier.T3): DEFAULT_T2_TO_T3,
}


# --- Check Functions ---


def _check_has_description(tool: ToolRegistryEntry, **_: Any) -> tuple[bool, str]:
    """Check if the tool has a description."""
    if tool.description.strip():
        return True, f"Description: {tool.description[:60]}"
    return False, "No description set. Add a README or description."


def _check_no_red_flags(tool: ToolRegistryEntry, **_: Any) -> tuple[bool, str]:
    """Check if the tool has no red flags."""
    if tool.scorecard.red_flags == 0:
        return True, "No red-tier issues found."
    return False, f"{tool.scorecard.red_flags} red-tier issue(s) unresolved."


def _check_min_users(tool: ToolRegistryEntry, *, min: int = 2, **_: Any) -> tuple[bool, str]:
    """Check if the tool has enough users."""
    count = len(tool.users)
    if count >= min:
        return True, f"{count} user(s) registered (minimum: {min})."
    return False, f"Only {count} user(s) registered (minimum: {min})."


def _check_has_source_path(tool: ToolRegistryEntry, **_: Any) -> tuple[bool, str]:
    """Check if source path is set."""
    if tool.source_path:
        return True, f"Source: {tool.source_path}"
    return False, "No source path configured."


def _check_min_composite_score(
    tool: ToolRegistryEntry, *, min: float = 0.7, **_: Any
) -> tuple[bool, str]:
    """Check if composite score meets minimum."""
    score = tool.scorecard.latest_composite
    if score >= min:
        return True, f"Composite score: {score:.2f} (minimum: {min:.2f})."
    return False, f"Composite score: {score:.2f} (minimum: {min:.2f})."


def _check_has_tech_owner(tool: ToolRegistryEntry, **_: Any) -> tuple[bool, str]:
    """Check if a tech owner is assigned."""
    if tool.tech_owner:
        return True, f"Tech owner: {tool.tech_owner}"
    return False, "No tech owner assigned."


def _check_has_test_coverage(tool: ToolRegistryEntry, **_: Any) -> tuple[bool, str]:
    """Check if basic test coverage exists."""
    test_score = tool.scorecard.latest_scores.get("testability", 0.0)
    if test_score >= 0.5:
        return True, f"Testability score: {test_score:.2f}."
    return False, f"Testability score: {test_score:.2f} (minimum: 0.50)."


def _check_dependencies_pinned(tool: ToolRegistryEntry, **_: Any) -> tuple[bool, str]:
    """Check if dependencies are pinned (via metadata flag)."""
    if tool.metadata.get("dependencies_pinned", False):
        return True, "Dependencies are pinned."
    return False, "Dependencies not confirmed as pinned."


def _check_has_code_review(tool: ToolRegistryEntry, **_: Any) -> tuple[bool, str]:
    """Check if code review has been completed (via metadata flag)."""
    if tool.metadata.get("code_reviewed", False):
        return True, "Code review completed."
    return False, "Code review not completed."


def _check_full_test_coverage(tool: ToolRegistryEntry, **_: Any) -> tuple[bool, str]:
    """Check if full test coverage exists."""
    test_score = tool.scorecard.latest_scores.get("testability", 0.0)
    if test_score >= 0.8:
        return True, f"Testability score: {test_score:.2f} (full coverage)."
    return False, f"Testability score: {test_score:.2f} (minimum: 0.80 for full coverage)."


def _check_security_review_done(tool: ToolRegistryEntry, **_: Any) -> tuple[bool, str]:
    """Check if security review has been completed."""
    if tool.metadata.get("security_reviewed", False):
        return True, "Security review completed."
    return False, "Security review not completed."


def _check_has_error_handling(tool: ToolRegistryEntry, **_: Any) -> tuple[bool, str]:
    """Check if error handling and logging is implemented."""
    if tool.metadata.get("error_handling", False):
        return True, "Error handling and logging confirmed."
    return False, "Error handling and logging not confirmed."


def _check_has_rollback_plan(tool: ToolRegistryEntry, **_: Any) -> tuple[bool, str]:
    """Check if a rollback plan is documented."""
    if tool.metadata.get("rollback_plan", False):
        return True, "Rollback plan documented."
    return False, "No rollback plan documented."


def _check_has_monitoring(tool: ToolRegistryEntry, **_: Any) -> tuple[bool, str]:
    """Check if monitoring and alerting is configured."""
    if tool.metadata.get("monitoring_configured", False):
        return True, "Monitoring and alerting configured."
    return False, "Monitoring and alerting not configured."


def _check_knowledge_transfer_done(tool: ToolRegistryEntry, **_: Any) -> tuple[bool, str]:
    """Check if knowledge transfer has been completed."""
    if tool.metadata.get("knowledge_transfer", False):
        return True, "Knowledge transfer completed."
    return False, "Knowledge transfer not completed."


# Check function registry
_CHECK_FUNCTIONS: dict[str, Any] = {
    "has_description": _check_has_description,
    "no_red_flags": _check_no_red_flags,
    "min_users": _check_min_users,
    "has_source_path": _check_has_source_path,
    "min_composite_score": _check_min_composite_score,
    "has_tech_owner": _check_has_tech_owner,
    "has_test_coverage": _check_has_test_coverage,
    "dependencies_pinned": _check_dependencies_pinned,
    "has_code_review": _check_has_code_review,
    "full_test_coverage": _check_full_test_coverage,
    "security_review_done": _check_security_review_done,
    "has_error_handling": _check_has_error_handling,
    "has_rollback_plan": _check_has_rollback_plan,
    "has_monitoring": _check_has_monitoring,
    "knowledge_transfer_done": _check_knowledge_transfer_done,
}


# --- Graduation Rubric ---


class GraduationRubric:
    """Evaluates tool readiness for tier graduation.

    Rubric criteria can be customized via config_overrides, which maps
    transition tuples to lists of criterion dicts matching the default format.
    """

    def __init__(
        self,
        config_overrides: dict[tuple[ToolTier, ToolTier], list[dict[str, Any]]] | None = None,
    ) -> None:
        self._rubrics = dict(_DEFAULT_RUBRICS)
        if config_overrides:
            self._rubrics.update(config_overrides)

    def evaluate(
        self,
        tool: ToolRegistryEntry,
        target_tier: ToolTier,
    ) -> GraduationResult:
        """Evaluate whether a tool is ready to graduate to target tier.

        Args:
            tool: The tool registry entry to evaluate.
            target_tier: The tier to graduate to.

        Returns:
            GraduationResult with checklist and readiness assessment.
        """
        from_tier = tool.tier
        transition = (from_tier, target_tier)

        if not is_valid_transition(from_tier, target_tier):
            return GraduationResult(
                ready=False,
                from_tier=from_tier,
                to_tier=target_tier,
                checklist=[
                    ChecklistItem(
                        requirement="Valid tier transition",
                        met=False,
                        details=(
                            f"Cannot transition from {from_tier.value} to "
                            f"{target_tier.value}. Must be one step up."
                        ),
                        blocking=True,
                    )
                ],
                blocking_items=[
                    ChecklistItem(
                        requirement="Valid tier transition",
                        met=False,
                        details="Invalid transition.",
                        blocking=True,
                    )
                ],
                overall_readiness=0.0,
            )

        criteria = self._rubrics.get(transition, [])
        checklist: list[ChecklistItem] = []
        blocking: list[ChecklistItem] = []
        advisory: list[ChecklistItem] = []

        for criterion in criteria:
            check_name = criterion["check"]
            params = criterion.get("params", {})
            is_blocking = criterion.get("blocking", True)

            check_fn = _CHECK_FUNCTIONS.get(check_name)
            if check_fn is None:
                item = ChecklistItem(
                    requirement=criterion["requirement"],
                    met=False,
                    details=f"Unknown check: {check_name}",
                    blocking=is_blocking,
                )
            else:
                met, details = check_fn(tool, **params)
                item = ChecklistItem(
                    requirement=criterion["requirement"],
                    met=met,
                    details=details,
                    blocking=is_blocking,
                )

            checklist.append(item)

            if not item.met and item.blocking:
                blocking.append(item)
            elif not item.met and not item.blocking:
                advisory.append(item)

        # Calculate readiness
        total = len(checklist)
        met_count = sum(1 for c in checklist if c.met)
        readiness = met_count / total if total > 0 else 0.0

        return GraduationResult(
            ready=len(blocking) == 0,
            from_tier=from_tier,
            to_tier=target_tier,
            checklist=checklist,
            blocking_items=blocking,
            advisory_items=advisory,
            overall_readiness=readiness,
        )

    def get_criteria(self, from_tier: ToolTier, to_tier: ToolTier) -> list[dict[str, Any]]:
        """Get the raw criteria config for a transition."""
        return list(self._rubrics.get((from_tier, to_tier), []))
