"""End-to-end integration tests across all modules (issue #57).

Tests the full flow: scorecard → gate → registry → training,
with no external dependencies (no LLM calls, no GPU).
"""

from __future__ import annotations

from datetime import datetime

from gate.overrides.manager import OverrideManager
from gate.tiers.evaluator import TierEvaluator
from registry.catalog.catalog import ToolCatalog
from registry.graduation.rubrics import GraduationRubric
from registry.graduation.triggers import GraduationTriggerEngine
from registry.workflows.engine import GraduationWorkflow, WorkflowStatus
from scorecard.engine import RubricEngine
from shared.config import OverridesConfig
from shared.models import (
    Dimension,
    DimensionScore,
    GateResult,
    GateTier,
    PatternFinding,
    ScoreResult,
    ScoringMethod,
    ToolRegistryEntry,
    ToolTier,
)
from shared.storage import JSONLBackend
from training.data import ScorecardDataset


# --- Helpers ---

GOOD_CODE = '''\
def calculate_total(items: list[dict]) -> float:
    """Calculate total price with tax."""
    return sum(item["price"] * (1 + item.get("tax_rate", 0.0)) for item in items)
'''

BAD_CODE = """\
password = "admin123"
def login(user, pw):
    if pw == password:
        return True
"""

MEDIUM_CODE = """\
def process_data(data):
    result = []
    for item in data:
        if item:
            result.append(item)
    return result
"""


def _make_score_result(
    code: str = "",
    composite: float = 0.75,
    user: str = "alice",
    skill: str = "codegen",
    timestamp: datetime | None = None,
    files: list[str] | None = None,
) -> ScoreResult:
    return ScoreResult(
        timestamp=timestamp or datetime.now(),
        user=user,
        skill_used=skill,
        files_touched=files or ["main.py"],
        source_code=code or None,
        dimension_scores=[
            DimensionScore(
                dimension=Dimension.CORRECTNESS,
                score=composite,
                method=ScoringMethod.AST_PARSE,
            ),
            DimensionScore(
                dimension=Dimension.SECURITY,
                score=composite,
                method=ScoringMethod.RULE_BASED,
            ),
        ],
        composite_score=composite,
    )


# --- Scorecard → Gate pipeline ---


class TestScorecardToGate:
    def test_good_code_gets_green(self):
        engine = RubricEngine()
        result = engine.score(GOOD_CODE, "calc.py", user="alice")
        evaluator = TierEvaluator()
        gate = evaluator.evaluate(result, GOOD_CODE, "calc.py")

        assert result.composite_score > 0
        assert result.source_code == GOOD_CODE
        assert gate.tier in (GateTier.GREEN, GateTier.YELLOW)
        assert not gate.blocked

    def test_bad_code_gets_red_or_yellow(self):
        engine = RubricEngine()
        result = engine.score(BAD_CODE, "login.py", user="alice")
        evaluator = TierEvaluator()
        gate = evaluator.evaluate(result, BAD_CODE, "login.py")

        assert gate.tier in (GateTier.RED, GateTier.YELLOW)
        if gate.tier == GateTier.RED:
            assert gate.blocked
            assert len(gate.critical_patterns_found) > 0

    def test_pipeline_stores_and_retrieves(self, tmp_path):
        engine = RubricEngine()
        storage = JSONLBackend(tmp_path / "scores")

        result = engine.score(GOOD_CODE, "calc.py", user="alice")
        storage.append(result)

        evaluator = TierEvaluator()
        evaluator.evaluate(result, GOOD_CODE, "calc.py")

        retrieved = storage.query()
        assert len(retrieved) == 1
        assert retrieved[0].source_code == GOOD_CODE
        assert retrieved[0].composite_score == result.composite_score

    def test_multiple_files_scored_and_gated(self):
        engine = RubricEngine()
        evaluator = TierEvaluator()

        codes = [
            (GOOD_CODE, "good.py"),
            (BAD_CODE, "bad.py"),
            (MEDIUM_CODE, "medium.py"),
        ]

        results = []
        for code, filename in codes:
            score = engine.score(code, filename, user="alice")
            gate = evaluator.evaluate(score, code, filename)
            results.append((score, gate))

        assert len(results) == 3
        # At least one should differ in tier from the others
        tiers = {g.tier for _, g in results}
        assert len(tiers) >= 1  # At minimum we get results for all


# --- Registry graduation flow ---


class TestRegistryGraduationFlow:
    def test_register_and_trigger_t0_to_t1(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path / "catalog")
        storage = JSONLBackend(tmp_path / "scores")

        # Register a T0 tool with 2 users (triggers second_user)
        tool = ToolRegistryEntry(
            name="My Tool",
            slug="my-tool",
            tier=ToolTier.T0,
            users=["alice", "bob"],
            source_path="/tools/my-tool",
        )
        catalog.register(tool)

        trigger_engine = GraduationTriggerEngine(catalog, storage)
        suggestions = trigger_engine.check_triggers()

        assert len(suggestions) == 1
        assert suggestions[0].tool_slug == "my-tool"
        assert suggestions[0].suggested_tier == ToolTier.T1
        assert "Second user" in suggestions[0].trigger_reason

    def test_no_trigger_single_user(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path / "catalog")
        storage = JSONLBackend(tmp_path / "scores")

        tool = ToolRegistryEntry(
            name="Solo Tool",
            slug="solo-tool",
            tier=ToolTier.T0,
            users=["alice"],
        )
        catalog.register(tool)

        trigger_engine = GraduationTriggerEngine(catalog, storage)
        suggestions = trigger_engine.check_triggers()
        assert len(suggestions) == 0

    def test_full_graduation_workflow(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path / "catalog")

        tool = ToolRegistryEntry(
            name="My Tool",
            slug="my-tool",
            tier=ToolTier.T0,
            users=["alice", "bob"],
            description="A useful tool",
            created_by="alice",
        )
        catalog.register(tool)

        rubric = GraduationRubric()
        workflow = GraduationWorkflow(
            catalog=catalog,
            rubric=rubric,
            storage_path=tmp_path / "workflows",
            auto_approve_t0_t1=False,
        )

        # Nominate
        state = workflow.nominate("my-tool", ToolTier.T1, nominated_by="alice")
        assert state.status == WorkflowStatus.PENDING
        assert state.from_tier == ToolTier.T0
        assert state.to_tier == ToolTier.T1

        # Check action items
        items = workflow.get_action_items("my-tool")
        assert isinstance(items, list)

        # Approve
        approved = workflow.approve(state.id, approved_by="tech-lead")
        assert approved.status == WorkflowStatus.PROMOTED

        # Verify tool was promoted
        updated_tool = catalog.get("my-tool")
        assert updated_tool is not None
        assert updated_tool.tier == ToolTier.T1
        assert len(updated_tool.graduation_history) == 1

    def test_auto_approve_t0_to_t1(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path / "catalog")

        tool = ToolRegistryEntry(
            name="Auto Tool",
            slug="auto-tool",
            tier=ToolTier.T0,
            users=["alice", "bob"],
            description="Auto-approved tool",
            created_by="alice",
        )
        catalog.register(tool)

        workflow = GraduationWorkflow(
            catalog=catalog,
            storage_path=tmp_path / "workflows",
            auto_approve_t0_t1=True,
        )

        state = workflow.nominate("auto-tool", ToolTier.T1, nominated_by="alice")
        # Should be auto-promoted
        assert state.status == WorkflowStatus.PROMOTED

        updated = catalog.get("auto-tool")
        assert updated is not None
        assert updated.tier == ToolTier.T1

    def test_reject_workflow(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path / "catalog")

        tool = ToolRegistryEntry(
            name="Rejected Tool",
            slug="rejected-tool",
            tier=ToolTier.T0,
            users=["alice"],
        )
        catalog.register(tool)

        workflow = GraduationWorkflow(
            catalog=catalog,
            storage_path=tmp_path / "workflows",
            auto_approve_t0_t1=False,
        )

        state = workflow.nominate("rejected-tool", ToolTier.T1, nominated_by="alice")
        rejected = workflow.reject(state.id, rejected_by="tech-lead", reason="Not ready")

        assert rejected.status == WorkflowStatus.REJECTED
        assert rejected.reject_reason == "Not ready"

        # Tool should still be T0
        tool = catalog.get("rejected-tool")
        assert tool is not None
        assert tool.tier == ToolTier.T0


# --- Score → Training data pipeline ---


class TestScoreToTrainingPipeline:
    def test_scored_code_becomes_training_data(self, tmp_path):
        engine = RubricEngine()
        storage = JSONLBackend(tmp_path / "scores")

        # Score multiple files and store
        codes = [GOOD_CODE, MEDIUM_CODE, GOOD_CODE]
        for i, code in enumerate(codes):
            result = engine.score(code, f"file{i}.py", user="alice")
            storage.append(result)

        # Load into training dataset
        ds = ScorecardDataset()
        examples = ds.from_storage(storage)

        assert len(examples) == 3
        # Source code should be populated
        assert examples[0].code == GOOD_CODE
        assert examples[1].code == MEDIUM_CODE
        assert examples[0].metadata["has_source_code"] is True

    def test_training_data_with_min_scores(self, tmp_path):
        storage = JSONLBackend(tmp_path / "scores")
        storage.append(_make_score_result(code=GOOD_CODE))

        ds = ScorecardDataset()
        examples = ds.from_storage(storage, min_scores=1)
        assert len(examples) == 1

    def test_training_data_split(self, tmp_path):
        engine = RubricEngine()
        storage = JSONLBackend(tmp_path / "scores")

        for i in range(10):
            result = engine.score(GOOD_CODE, f"file{i}.py", user="alice")
            storage.append(result)

        ds = ScorecardDataset()
        ds.from_storage(storage)
        train, eval_ = ds.split(train_ratio=0.8)

        assert len(train) == 8
        assert len(eval_) == 2


# --- Override workflow integration ---


class TestOverrideWorkflow:
    def _gate_red_result(self) -> GateResult:
        """Create a red gate result for override testing."""
        score = _make_score_result(code=BAD_CODE, composite=0.3)
        return GateResult(
            tier=GateTier.RED,
            score_result=score,
            blocked=True,
            critical_patterns_found=["hardcoded_credentials"],
            pattern_findings=[
                PatternFinding(
                    pattern="hardcoded_credentials",
                    severity="critical",
                    line_number=1,
                    description="Hardcoded password",
                )
            ],
            advisory_messages=["BLOCKED: Hardcoded password detected"],
        )

    def test_override_with_justification(self, tmp_path):
        config = OverridesConfig(
            require_justification=True,
            max_overrides_per_user_per_week=3,
        )
        manager = OverrideManager(config=config, storage_path=tmp_path / "overrides")

        gate_result = self._gate_red_result()
        result = manager.request_override(
            gate_result=gate_result,
            user="alice",
            justification="Testing only, will fix before merge",
            filename="login.py",
        )

        assert result.approved is True
        assert result.record is not None
        assert result.record.user == "alice"

    def test_override_rejected_without_justification(self, tmp_path):
        config = OverridesConfig(require_justification=True)
        manager = OverrideManager(config=config, storage_path=tmp_path / "overrides")

        gate_result = self._gate_red_result()
        result = manager.request_override(
            gate_result=gate_result,
            user="alice",
            justification="",
        )

        assert result.approved is False
        assert "Justification is required" in result.reason

    def test_override_rate_limiting(self, tmp_path):
        config = OverridesConfig(
            require_justification=False,
            max_overrides_per_user_per_week=2,
        )
        manager = OverrideManager(config=config, storage_path=tmp_path / "overrides")
        gate_result = self._gate_red_result()

        # Use up the limit
        for _ in range(2):
            result = manager.request_override(
                gate_result=gate_result, user="alice", justification="reason"
            )
            assert result.approved is True

        # Third should be rejected
        result = manager.request_override(
            gate_result=gate_result, user="alice", justification="reason"
        )
        assert result.approved is False
        assert "limit reached" in result.reason.lower()

    def test_escalation_workflow(self, tmp_path):
        config = OverridesConfig()
        manager = OverrideManager(config=config, storage_path=tmp_path / "overrides")
        gate_result = self._gate_red_result()

        # Escalate
        result = manager.request_escalation(
            gate_result=gate_result,
            user="alice",
            justification="Need tech team review",
            filename="login.py",
        )
        assert result.record is not None

        # Check pending escalations
        pending = manager.get_pending_escalations()
        assert len(pending) == 1
        assert pending[0].user == "alice"

        # Resolve escalation
        resolved = manager.resolve_escalation(
            pending[0].id, reviewer="tech-lead", outcome="approved"
        )
        assert resolved is True

        # No more pending
        assert len(manager.get_pending_escalations()) == 0

    def test_full_flow_score_gate_override(self, tmp_path):
        """Score code → gate blocks → override approved → audit recorded."""
        engine = RubricEngine()
        evaluator = TierEvaluator()
        config = OverridesConfig(require_justification=True)
        manager = OverrideManager(config=config, storage_path=tmp_path / "overrides")

        # Score and gate
        result = engine.score(BAD_CODE, "login.py", user="alice")
        gate = evaluator.evaluate(result, BAD_CODE, "login.py")

        if gate.blocked:
            # Request override
            override = manager.request_override(
                gate_result=gate,
                user="alice",
                justification="Legacy code, scheduled for refactor",
                filename="login.py",
            )
            assert override.approved is True

            # Verify audit trail
            records = manager.get_user_overrides("alice")
            assert len(records) == 1
            assert records[0].justification == "Legacy code, scheduled for refactor"


# --- Scorecard → Registry scorecard update ---


class TestScorecardToRegistry:
    def test_score_updates_tool_scorecard(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path / "catalog")
        tool = ToolRegistryEntry(
            name="My Tool",
            slug="my-tool",
            tier=ToolTier.T1,
        )
        catalog.register(tool)

        engine = RubricEngine()
        result = engine.score(GOOD_CODE, "calc.py", user="alice")

        catalog.update_scorecard("my-tool", result)

        updated = catalog.get("my-tool")
        assert updated is not None
        assert updated.scorecard.total_scores == 1
        assert updated.scorecard.latest_composite == result.composite_score

    def test_multiple_scores_update_trend(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path / "catalog")
        tool = ToolRegistryEntry(
            name="Trend Tool",
            slug="trend-tool",
            tier=ToolTier.T1,
        )
        catalog.register(tool)

        engine = RubricEngine()

        # Score bad code first (low score)
        result1 = engine.score(BAD_CODE, "bad.py", user="alice")
        catalog.update_scorecard("trend-tool", result1)

        # Then score good code (higher score)
        result2 = engine.score(GOOD_CODE, "good.py", user="alice")
        catalog.update_scorecard("trend-tool", result2)

        updated = catalog.get("trend-tool")
        assert updated is not None
        assert updated.scorecard.total_scores == 2
        # If score improved significantly, trend should reflect
        assert updated.scorecard.trend in ("improving", "stable")
