"""Tests for the override workflow manager."""

from shared.config import OverridesConfig
from shared.models import (
    GateResult,
    GateTier,
    ScoreResult,
)

from gate.overrides.manager import OverrideManager


def _make_gate_result(blocked: bool = True) -> GateResult:
    return GateResult(
        tier=GateTier.RED,
        score_result=ScoreResult(user="test", composite_score=0.3),
        blocked=blocked,
        critical_patterns_found=["hardcoded_credentials"],
        advisory_messages=["BLOCKED: Hardcoded password found."],
    )


# --- Override requests ---


class TestOverrideRequest:
    def test_basic_override(self, tmp_path):
        mgr = OverrideManager(storage_path=tmp_path / "overrides")
        gate = _make_gate_result()
        result = mgr.request_override(
            gate, user="alice", justification="False positive, this is a test fixture."
        )
        assert result.approved
        assert result.record is not None
        assert result.record.user == "alice"
        assert result.record.override_type == "proceed"
        assert result.record.justification == "False positive, this is a test fixture."

    def test_override_creates_audit_record(self, tmp_path):
        mgr = OverrideManager(storage_path=tmp_path / "overrides")
        gate = _make_gate_result()
        mgr.request_override(gate, user="alice", justification="test reason")

        records = mgr._read_all_records()
        assert len(records) == 1
        assert records[0].user == "alice"
        assert records[0].id != ""

    def test_override_with_filename(self, tmp_path):
        mgr = OverrideManager(storage_path=tmp_path / "overrides")
        gate = _make_gate_result()
        result = mgr.request_override(
            gate, user="alice", justification="test", filename="config.py"
        )
        assert result.record.filename == "config.py"


# --- Justification required ---


class TestJustificationRequired:
    def test_empty_justification_denied(self, tmp_path):
        config = OverridesConfig(require_justification=True)
        mgr = OverrideManager(config=config, storage_path=tmp_path / "overrides")
        gate = _make_gate_result()
        result = mgr.request_override(gate, user="alice", justification="")
        assert not result.approved
        assert "justification" in result.reason.lower()

    def test_whitespace_justification_denied(self, tmp_path):
        config = OverridesConfig(require_justification=True)
        mgr = OverrideManager(config=config, storage_path=tmp_path / "overrides")
        gate = _make_gate_result()
        result = mgr.request_override(gate, user="alice", justification="   ")
        assert not result.approved

    def test_justification_not_required(self, tmp_path):
        config = OverridesConfig(require_justification=False)
        mgr = OverrideManager(config=config, storage_path=tmp_path / "overrides")
        gate = _make_gate_result()
        result = mgr.request_override(gate, user="alice", justification="")
        assert result.approved


# --- Rate limiting ---


class TestRateLimiting:
    def test_under_limit(self, tmp_path):
        config = OverridesConfig(max_overrides_per_user_per_week=3)
        mgr = OverrideManager(config=config, storage_path=tmp_path / "overrides")
        gate = _make_gate_result()

        r1 = mgr.request_override(gate, user="alice", justification="reason 1")
        r2 = mgr.request_override(gate, user="alice", justification="reason 2")
        assert r1.approved
        assert r2.approved

    def test_at_limit_denied(self, tmp_path):
        config = OverridesConfig(max_overrides_per_user_per_week=2)
        mgr = OverrideManager(config=config, storage_path=tmp_path / "overrides")
        gate = _make_gate_result()

        mgr.request_override(gate, user="alice", justification="reason 1")
        mgr.request_override(gate, user="alice", justification="reason 2")
        r3 = mgr.request_override(gate, user="alice", justification="reason 3")
        assert not r3.approved
        assert "limit" in r3.reason.lower()

    def test_different_users_independent(self, tmp_path):
        config = OverridesConfig(max_overrides_per_user_per_week=1)
        mgr = OverrideManager(config=config, storage_path=tmp_path / "overrides")
        gate = _make_gate_result()

        mgr.request_override(gate, user="alice", justification="reason")
        r_bob = mgr.request_override(gate, user="bob", justification="reason")
        assert r_bob.approved

    def test_escalations_dont_count_toward_limit(self, tmp_path):
        config = OverridesConfig(max_overrides_per_user_per_week=1)
        mgr = OverrideManager(config=config, storage_path=tmp_path / "overrides")
        gate = _make_gate_result()

        # Escalate (should not count)
        mgr.request_escalation(gate, user="alice", justification="escalate reason")
        # Override should still work
        result = mgr.request_override(gate, user="alice", justification="override reason")
        assert result.approved


# --- Escalation ---


class TestEscalation:
    def test_basic_escalation(self, tmp_path):
        mgr = OverrideManager(storage_path=tmp_path / "overrides")
        gate = _make_gate_result()
        result = mgr.request_escalation(gate, user="alice", justification="Need tech team review.")
        assert not result.approved  # Escalation is not an approval
        assert result.record is not None
        assert result.record.override_type == "escalate"
        assert result.record.review_outcome == "pending"

    def test_pending_escalations(self, tmp_path):
        mgr = OverrideManager(storage_path=tmp_path / "overrides")
        gate = _make_gate_result()
        mgr.request_escalation(gate, user="alice", justification="reason 1")
        mgr.request_escalation(gate, user="bob", justification="reason 2")

        pending = mgr.get_pending_escalations()
        assert len(pending) == 2

    def test_resolve_escalation_approved(self, tmp_path):
        mgr = OverrideManager(storage_path=tmp_path / "overrides")
        gate = _make_gate_result()
        result = mgr.request_escalation(gate, user="alice", justification="need help")
        override_id = result.record.id

        success = mgr.resolve_escalation(override_id, reviewer="tech_lead", outcome="approved")
        assert success

        # Verify the record was updated
        pending = mgr.get_pending_escalations()
        assert len(pending) == 0

        records = mgr._read_all_records()
        resolved = [r for r in records if r.id == override_id]
        assert len(resolved) == 1
        assert resolved[0].reviewed_by == "tech_lead"
        assert resolved[0].review_outcome == "approved"

    def test_resolve_escalation_rejected(self, tmp_path):
        mgr = OverrideManager(storage_path=tmp_path / "overrides")
        gate = _make_gate_result()
        result = mgr.request_escalation(gate, user="alice", justification="reason")
        override_id = result.record.id

        success = mgr.resolve_escalation(override_id, reviewer="tech_lead", outcome="rejected")
        assert success

        records = mgr._read_all_records()
        resolved = [r for r in records if r.id == override_id]
        assert resolved[0].review_outcome == "rejected"

    def test_resolve_invalid_outcome(self, tmp_path):
        mgr = OverrideManager(storage_path=tmp_path / "overrides")
        gate = _make_gate_result()
        result = mgr.request_escalation(gate, user="alice", justification="reason")

        success = mgr.resolve_escalation(result.record.id, reviewer="tech_lead", outcome="maybe")
        assert not success

    def test_resolve_nonexistent_id(self, tmp_path):
        mgr = OverrideManager(storage_path=tmp_path / "overrides")
        success = mgr.resolve_escalation("nonexistent-id", reviewer="tech_lead", outcome="approved")
        assert not success


# --- User queries ---


class TestUserQueries:
    def test_get_user_overrides(self, tmp_path):
        mgr = OverrideManager(storage_path=tmp_path / "overrides")
        gate = _make_gate_result()
        mgr.request_override(gate, user="alice", justification="reason 1")
        mgr.request_override(gate, user="alice", justification="reason 2")
        mgr.request_override(gate, user="bob", justification="reason 3")

        alice_records = mgr.get_user_overrides("alice")
        assert len(alice_records) == 2

        bob_records = mgr.get_user_overrides("bob")
        assert len(bob_records) == 1

    def test_get_override_stats(self, tmp_path):
        config = OverridesConfig(max_overrides_per_user_per_week=5)
        mgr = OverrideManager(config=config, storage_path=tmp_path / "overrides")
        gate = _make_gate_result()
        mgr.request_override(gate, user="alice", justification="reason 1")
        mgr.request_override(gate, user="alice", justification="reason 2")
        mgr.request_escalation(gate, user="alice", justification="escalate")

        stats = mgr.get_override_stats("alice")
        assert stats["total_overrides"] == 3
        assert stats["weekly_overrides"] == 2  # Only "proceed" type
        assert stats["weekly_limit"] == 5
        assert stats["remaining_this_week"] == 3
        assert stats["pending_escalations"] == 1


# --- Persistence ---


class TestPersistence:
    def test_records_persist(self, tmp_path):
        storage = tmp_path / "overrides"
        gate = _make_gate_result()

        # Create and write
        mgr1 = OverrideManager(storage_path=storage)
        mgr1.request_override(gate, user="alice", justification="reason")

        # New manager reads from same path
        mgr2 = OverrideManager(storage_path=storage)
        records = mgr2._read_all_records()
        assert len(records) == 1
        assert records[0].user == "alice"

    def test_empty_storage(self, tmp_path):
        mgr = OverrideManager(storage_path=tmp_path / "overrides")
        records = mgr._read_all_records()
        assert records == []
        pending = mgr.get_pending_escalations()
        assert pending == []
