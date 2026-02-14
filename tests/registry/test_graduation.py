"""Tests for graduation tier model and rubrics."""

from shared.models import ScorecardSummary, ToolRegistryEntry, ToolTier

from registry.graduation.rubrics import (
    ChecklistItem,
    GraduationResult,
    GraduationRubric,
)
from registry.graduation.tiers import (
    TIER_DEFINITIONS,
    VALID_TRANSITIONS,
    get_tier_definition,
    is_valid_transition,
    tier_index,
)


# --- Helpers ---


def _make_tool(
    name: str = "test-tool",
    tier: ToolTier = ToolTier.T0,
    description: str = "",
    users: list[str] | None = None,
    tech_owner: str | None = None,
    composite: float = 0.0,
    scores: dict[str, float] | None = None,
    red_flags: int = 0,
    source_path: str = "",
    metadata: dict | None = None,
) -> ToolRegistryEntry:
    return ToolRegistryEntry(
        name=name,
        tier=tier,
        description=description,
        users=users or [],
        tech_owner=tech_owner,
        source_path=source_path,
        scorecard=ScorecardSummary(
            latest_composite=composite,
            latest_scores=scores or {},
            red_flags=red_flags,
        ),
        metadata=metadata or {},
    )


# --- Tier Definitions ---


class TestTierDefinitions:
    def test_all_tiers_defined(self):
        for tier in ToolTier:
            assert tier in TIER_DEFINITIONS

    def test_get_tier_definition(self):
        t0 = get_tier_definition(ToolTier.T0)
        assert t0.label == "Personal"
        assert t0.tier == ToolTier.T0

    def test_t3_is_critical(self):
        t3 = get_tier_definition(ToolTier.T3)
        assert "Critical" in t3.label
        assert "production" in t3.description.lower() or "business" in t3.description.lower()

    def test_tier_index_ordering(self):
        assert tier_index(ToolTier.T0) < tier_index(ToolTier.T1)
        assert tier_index(ToolTier.T1) < tier_index(ToolTier.T2)
        assert tier_index(ToolTier.T2) < tier_index(ToolTier.T3)


# --- Valid Transitions ---


class TestTransitions:
    def test_valid_transitions(self):
        assert is_valid_transition(ToolTier.T0, ToolTier.T1)
        assert is_valid_transition(ToolTier.T1, ToolTier.T2)
        assert is_valid_transition(ToolTier.T2, ToolTier.T3)

    def test_skip_transition_invalid(self):
        assert not is_valid_transition(ToolTier.T0, ToolTier.T2)
        assert not is_valid_transition(ToolTier.T0, ToolTier.T3)
        assert not is_valid_transition(ToolTier.T1, ToolTier.T3)

    def test_downgrade_invalid(self):
        assert not is_valid_transition(ToolTier.T1, ToolTier.T0)
        assert not is_valid_transition(ToolTier.T3, ToolTier.T0)

    def test_same_tier_invalid(self):
        assert not is_valid_transition(ToolTier.T0, ToolTier.T0)
        assert not is_valid_transition(ToolTier.T2, ToolTier.T2)

    def test_all_valid_transitions_listed(self):
        assert len(VALID_TRANSITIONS) == 3


# --- T0 → T1 Rubric ---


class TestT0ToT1:
    def setup_method(self):
        self.rubric = GraduationRubric()

    def test_ready_tool_passes(self):
        tool = _make_tool(
            tier=ToolTier.T0,
            description="A useful tool for data processing",
            users=["alice", "bob"],
            source_path="tools/data.py",
        )
        result = self.rubric.evaluate(tool, ToolTier.T1)
        assert result.ready
        assert result.overall_readiness == 1.0
        assert len(result.blocking_items) == 0

    def test_no_description_blocks(self):
        tool = _make_tool(
            tier=ToolTier.T0,
            description="",
            users=["alice", "bob"],
        )
        result = self.rubric.evaluate(tool, ToolTier.T1)
        assert not result.ready
        assert any("description" in b.requirement.lower() for b in result.blocking_items)

    def test_single_user_blocks(self):
        tool = _make_tool(
            tier=ToolTier.T0,
            description="A tool",
            users=["alice"],
        )
        result = self.rubric.evaluate(tool, ToolTier.T1)
        assert not result.ready
        assert any("user" in b.requirement.lower() for b in result.blocking_items)

    def test_red_flags_block(self):
        tool = _make_tool(
            tier=ToolTier.T0,
            description="A tool",
            users=["alice", "bob"],
            red_flags=1,
        )
        result = self.rubric.evaluate(tool, ToolTier.T1)
        assert not result.ready

    def test_missing_source_path_is_advisory(self):
        tool = _make_tool(
            tier=ToolTier.T0,
            description="A tool",
            users=["alice", "bob"],
            source_path="",
        )
        result = self.rubric.evaluate(tool, ToolTier.T1)
        # Source path is advisory (non-blocking), so tool should still be ready
        assert result.ready
        assert len(result.advisory_items) == 1
        assert "source" in result.advisory_items[0].requirement.lower()


# --- T1 → T2 Rubric ---


class TestT1ToT2:
    def setup_method(self):
        self.rubric = GraduationRubric()

    def test_ready_tool_passes(self):
        tool = _make_tool(
            tier=ToolTier.T1,
            description="Team tool",
            users=["alice", "bob", "charlie"],
            tech_owner="dave",
            composite=0.8,
            scores={"testability": 0.7},
            metadata={"dependencies_pinned": True, "code_reviewed": True},
        )
        result = self.rubric.evaluate(tool, ToolTier.T2)
        assert result.ready
        assert len(result.blocking_items) == 0

    def test_low_score_blocks(self):
        tool = _make_tool(
            tier=ToolTier.T1,
            description="Team tool",
            users=["alice", "bob"],
            tech_owner="dave",
            composite=0.5,
            scores={"testability": 0.6},
            metadata={"dependencies_pinned": True},
        )
        result = self.rubric.evaluate(tool, ToolTier.T2)
        assert not result.ready
        assert any("score" in b.requirement.lower() for b in result.blocking_items)

    def test_no_tech_owner_blocks(self):
        tool = _make_tool(
            tier=ToolTier.T1,
            composite=0.8,
            scores={"testability": 0.7},
            metadata={"dependencies_pinned": True},
        )
        result = self.rubric.evaluate(tool, ToolTier.T2)
        assert not result.ready
        assert any("tech owner" in b.requirement.lower() for b in result.blocking_items)

    def test_no_test_coverage_blocks(self):
        tool = _make_tool(
            tier=ToolTier.T1,
            tech_owner="dave",
            composite=0.8,
            scores={"testability": 0.3},
            metadata={"dependencies_pinned": True},
        )
        result = self.rubric.evaluate(tool, ToolTier.T2)
        assert not result.ready
        assert any("test" in b.requirement.lower() for b in result.blocking_items)

    def test_unpinned_deps_block(self):
        tool = _make_tool(
            tier=ToolTier.T1,
            tech_owner="dave",
            composite=0.8,
            scores={"testability": 0.7},
            metadata={"dependencies_pinned": False},
        )
        result = self.rubric.evaluate(tool, ToolTier.T2)
        assert not result.ready

    def test_code_review_is_advisory(self):
        tool = _make_tool(
            tier=ToolTier.T1,
            tech_owner="dave",
            composite=0.8,
            scores={"testability": 0.7},
            metadata={"dependencies_pinned": True, "code_reviewed": False},
        )
        result = self.rubric.evaluate(tool, ToolTier.T2)
        # Code review is advisory, not blocking
        assert result.ready
        assert len(result.advisory_items) == 1


# --- T2 → T3 Rubric ---


class TestT2ToT3:
    def setup_method(self):
        self.rubric = GraduationRubric()

    def test_ready_tool_passes(self):
        tool = _make_tool(
            tier=ToolTier.T2,
            users=["a", "b", "c"],
            composite=0.9,
            scores={"testability": 0.9},
            metadata={
                "security_reviewed": True,
                "error_handling": True,
                "rollback_plan": True,
                "monitoring_configured": True,
                "knowledge_transfer": True,
            },
        )
        result = self.rubric.evaluate(tool, ToolTier.T3)
        assert result.ready
        assert len(result.blocking_items) == 0

    def test_missing_security_review_blocks(self):
        tool = _make_tool(
            tier=ToolTier.T2,
            composite=0.9,
            scores={"testability": 0.9},
            metadata={
                "error_handling": True,
                "rollback_plan": True,
                "monitoring_configured": True,
                "knowledge_transfer": True,
            },
        )
        result = self.rubric.evaluate(tool, ToolTier.T3)
        assert not result.ready
        assert any("security" in b.requirement.lower() for b in result.blocking_items)

    def test_low_test_coverage_blocks(self):
        tool = _make_tool(
            tier=ToolTier.T2,
            composite=0.9,
            scores={"testability": 0.5},
            metadata={
                "security_reviewed": True,
                "error_handling": True,
                "rollback_plan": True,
                "monitoring_configured": True,
                "knowledge_transfer": True,
            },
        )
        result = self.rubric.evaluate(tool, ToolTier.T3)
        assert not result.ready

    def test_missing_monitoring_blocks(self):
        tool = _make_tool(
            tier=ToolTier.T2,
            composite=0.9,
            scores={"testability": 0.9},
            metadata={
                "security_reviewed": True,
                "error_handling": True,
                "rollback_plan": True,
                "knowledge_transfer": True,
            },
        )
        result = self.rubric.evaluate(tool, ToolTier.T3)
        assert not result.ready

    def test_high_composite_required(self):
        tool = _make_tool(
            tier=ToolTier.T2,
            composite=0.7,  # Below 0.8 threshold for T3
            scores={"testability": 0.9},
            metadata={
                "security_reviewed": True,
                "error_handling": True,
                "rollback_plan": True,
                "monitoring_configured": True,
                "knowledge_transfer": True,
            },
        )
        result = self.rubric.evaluate(tool, ToolTier.T3)
        assert not result.ready

    def test_min_users_is_advisory(self):
        tool = _make_tool(
            tier=ToolTier.T2,
            users=["alice"],  # Only 1, below advisory threshold of 3
            composite=0.9,
            scores={"testability": 0.9},
            metadata={
                "security_reviewed": True,
                "error_handling": True,
                "rollback_plan": True,
                "monitoring_configured": True,
                "knowledge_transfer": True,
            },
        )
        result = self.rubric.evaluate(tool, ToolTier.T3)
        assert result.ready  # Advisory only
        assert len(result.advisory_items) == 1


# --- Invalid Transitions ---


class TestInvalidTransitions:
    def setup_method(self):
        self.rubric = GraduationRubric()

    def test_skip_tier_rejected(self):
        tool = _make_tool(tier=ToolTier.T0)
        result = self.rubric.evaluate(tool, ToolTier.T2)
        assert not result.ready
        assert "transition" in result.checklist[0].requirement.lower()

    def test_downgrade_rejected(self):
        tool = _make_tool(tier=ToolTier.T2)
        result = self.rubric.evaluate(tool, ToolTier.T1)
        assert not result.ready

    def test_same_tier_rejected(self):
        tool = _make_tool(tier=ToolTier.T1)
        result = self.rubric.evaluate(tool, ToolTier.T1)
        assert not result.ready


# --- Readiness Score ---


class TestReadiness:
    def setup_method(self):
        self.rubric = GraduationRubric()

    def test_full_readiness(self):
        tool = _make_tool(
            tier=ToolTier.T0,
            description="A tool",
            users=["alice", "bob"],
            source_path="tools/data.py",
        )
        result = self.rubric.evaluate(tool, ToolTier.T1)
        assert result.overall_readiness == 1.0

    def test_partial_readiness(self):
        tool = _make_tool(
            tier=ToolTier.T0,
            description="",
            users=["alice"],
        )
        result = self.rubric.evaluate(tool, ToolTier.T1)
        assert 0.0 < result.overall_readiness < 1.0

    def test_zero_readiness(self):
        tool = _make_tool(
            tier=ToolTier.T0,
            description="",
            users=[],
            red_flags=3,
        )
        result = self.rubric.evaluate(tool, ToolTier.T1)
        assert result.overall_readiness == 0.0


# --- Custom Config ---


class TestCustomConfig:
    def test_override_criteria(self):
        custom = {
            (ToolTier.T0, ToolTier.T1): [
                {
                    "requirement": "Must have 5 users",
                    "check": "min_users",
                    "blocking": True,
                    "params": {"min": 5},
                },
            ],
        }
        rubric = GraduationRubric(config_overrides=custom)
        tool = _make_tool(tier=ToolTier.T0, users=["a", "b"])
        result = rubric.evaluate(tool, ToolTier.T1)
        assert not result.ready

    def test_custom_does_not_affect_other_transitions(self):
        custom = {
            (ToolTier.T0, ToolTier.T1): [
                {
                    "requirement": "Custom",
                    "check": "has_description",
                    "blocking": True,
                },
            ],
        }
        rubric = GraduationRubric(config_overrides=custom)
        # T1→T2 should still use defaults
        criteria = rubric.get_criteria(ToolTier.T1, ToolTier.T2)
        assert len(criteria) > 1

    def test_get_criteria_returns_copy(self):
        rubric = GraduationRubric()
        criteria = rubric.get_criteria(ToolTier.T0, ToolTier.T1)
        criteria.clear()
        # Original should be unaffected
        assert len(rubric.get_criteria(ToolTier.T0, ToolTier.T1)) > 0

    def test_unknown_check_handled(self):
        custom = {
            (ToolTier.T0, ToolTier.T1): [
                {
                    "requirement": "Mystery check",
                    "check": "does_not_exist",
                    "blocking": True,
                },
            ],
        }
        rubric = GraduationRubric(config_overrides=custom)
        tool = _make_tool(tier=ToolTier.T0)
        result = rubric.evaluate(tool, ToolTier.T1)
        assert not result.ready
        assert "Unknown" in result.checklist[0].details


# --- GraduationResult ---


class TestGraduationResult:
    def test_result_fields(self):
        result = GraduationResult(
            ready=True,
            from_tier=ToolTier.T0,
            to_tier=ToolTier.T1,
            overall_readiness=1.0,
        )
        assert result.ready
        assert result.from_tier == ToolTier.T0
        assert result.to_tier == ToolTier.T1

    def test_checklist_item_fields(self):
        item = ChecklistItem(
            requirement="Has description",
            met=True,
            details="Looks good",
            blocking=True,
        )
        assert item.requirement == "Has description"
        assert item.met
        assert item.blocking
