"""Override workflow manager with audit logging.

Handles override requests when code hits red tier, tracks audit records,
enforces rate limits, and manages escalation workflow.
"""

from __future__ import annotations

import fcntl
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from shared.config import OverridesConfig, load_config
from shared.models import GateResult, OverrideRecord


class OverrideResult:
    """Result of an override request."""

    def __init__(
        self,
        approved: bool,
        record: OverrideRecord | None = None,
        reason: str = "",
    ) -> None:
        self.approved = approved
        self.record = record
        self.reason = reason


class OverrideManager:
    """Manages gate override requests with audit logging and rate limiting.

    Args:
        config: Overrides configuration. Uses defaults if not provided.
        storage_path: Path to store override audit logs.
    """

    def __init__(
        self,
        config: OverridesConfig | None = None,
        storage_path: str | Path | None = None,
    ) -> None:
        if config is None:
            config = load_config().gate.overrides
        self.config = config

        if storage_path is None:
            storage_path = Path("./rubric-gates-data/overrides")
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _log_file(self) -> Path:
        """Get the override audit log file path."""
        return self.storage_path / "overrides.jsonl"

    def _append_record(self, record: OverrideRecord) -> None:
        """Append an override record to the audit log."""
        target = self._log_file()
        line = record.model_dump_json() + "\n"
        with open(target, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _read_all_records(self) -> list[OverrideRecord]:
        """Read all override records from the audit log."""
        target = self._log_file()
        if not target.exists():
            return []

        records: list[OverrideRecord] = []
        with open(target) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(OverrideRecord.model_validate_json(line))
                except Exception:
                    continue
        return records

    def _update_record(self, record_id: str, **updates: str) -> bool:
        """Update a record in the audit log by rewriting the file."""
        target = self._log_file()
        if not target.exists():
            return False

        records = self._read_all_records()
        found = False
        updated_records: list[OverrideRecord] = []

        for rec in records:
            if rec.id == record_id:
                for key, val in updates.items():
                    setattr(rec, key, val)
                found = True
            updated_records.append(rec)

        if not found:
            return False

        # Rewrite file
        with open(target, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                for rec in updated_records:
                    f.write(rec.model_dump_json() + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return True

    def _count_user_overrides_this_week(self, user: str) -> int:
        """Count how many overrides a user has made in the current week."""
        week_start = datetime.now() - timedelta(days=7)
        records = self._read_all_records()
        return sum(
            1
            for r in records
            if r.user == user and r.override_type == "proceed" and r.timestamp >= week_start
        )

    def request_override(
        self,
        gate_result: GateResult,
        user: str,
        justification: str,
        filename: str = "",
    ) -> OverrideResult:
        """Process an override request.

        Args:
            gate_result: The gate result that was blocked.
            user: The user requesting the override.
            justification: Reason for the override.
            filename: The file being overridden.

        Returns:
            OverrideResult with approved status and audit record.
        """
        # Check justification requirement
        if self.config.require_justification and not justification.strip():
            return OverrideResult(
                approved=False,
                reason="Justification is required for overrides.",
            )

        # Check rate limit
        weekly_count = self._count_user_overrides_this_week(user)
        if weekly_count >= self.config.max_overrides_per_user_per_week:
            return OverrideResult(
                approved=False,
                reason=(
                    f"Override limit reached ({weekly_count}/"
                    f"{self.config.max_overrides_per_user_per_week} this week). "
                    "Please escalate to the tech team instead."
                ),
            )

        # Create override record
        record = OverrideRecord(
            id=str(uuid.uuid4()),
            user=user,
            filename=filename,
            gate_result=gate_result,
            justification=justification,
            override_type="proceed",
            review_outcome="pending",
        )

        # Store audit record
        self._append_record(record)

        return OverrideResult(
            approved=True,
            record=record,
            reason="Override approved. Audit record created.",
        )

    def request_escalation(
        self,
        gate_result: GateResult,
        user: str,
        justification: str,
        filename: str = "",
    ) -> OverrideResult:
        """Escalate to tech team for review.

        Args:
            gate_result: The gate result that was blocked.
            user: The user requesting escalation.
            justification: Reason for the escalation.
            filename: The file being escalated.

        Returns:
            OverrideResult with the escalation record.
        """
        record = OverrideRecord(
            id=str(uuid.uuid4()),
            user=user,
            filename=filename,
            gate_result=gate_result,
            justification=justification,
            override_type="escalate",
            review_outcome="pending",
        )

        self._append_record(record)

        return OverrideResult(
            approved=False,
            record=record,
            reason="Escalated to tech team. Awaiting review.",
        )

    def get_pending_escalations(self) -> list[OverrideRecord]:
        """Get all escalations awaiting tech team review."""
        records = self._read_all_records()
        return [
            r for r in records if r.override_type == "escalate" and r.review_outcome == "pending"
        ]

    def get_user_overrides(self, user: str, since: datetime | None = None) -> list[OverrideRecord]:
        """Get all overrides for a specific user."""
        records = self._read_all_records()
        return [r for r in records if r.user == user and (since is None or r.timestamp >= since)]

    def resolve_escalation(
        self,
        override_id: str,
        reviewer: str,
        outcome: str,
    ) -> bool:
        """Tech team resolves an escalation.

        Args:
            override_id: The ID of the override record.
            reviewer: The tech team member resolving.
            outcome: "approved" or "rejected".

        Returns:
            True if the record was found and updated.
        """
        if outcome not in ("approved", "rejected"):
            return False

        return self._update_record(
            override_id,
            reviewed_by=reviewer,
            review_outcome=outcome,
        )

    def get_override_stats(self, user: str) -> dict:
        """Get override statistics for a user."""
        records = self._read_all_records()
        user_records = [r for r in records if r.user == user]
        week_start = datetime.now() - timedelta(days=7)
        weekly = [r for r in user_records if r.timestamp >= week_start]

        return {
            "total_overrides": len(user_records),
            "weekly_overrides": len([r for r in weekly if r.override_type == "proceed"]),
            "weekly_limit": self.config.max_overrides_per_user_per_week,
            "remaining_this_week": max(
                0,
                self.config.max_overrides_per_user_per_week
                - len([r for r in weekly if r.override_type == "proceed"]),
            ),
            "pending_escalations": len(
                [
                    r
                    for r in user_records
                    if r.override_type == "escalate" and r.review_outcome == "pending"
                ]
            ),
        }
