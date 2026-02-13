"""Storage backends for rubric-gates score data.

Append-only storage with query support. JSONL is the default backend;
SQLite is available as an optional upgrade path.
"""

from __future__ import annotations

import fcntl
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from shared.config import StorageConfig
from shared.models import ScoreResult


# --- Query Filters ---


class QueryFilters(BaseModel):
    """Filters for querying stored score results."""

    start_date: datetime | None = None
    end_date: datetime | None = None
    user: str | None = None
    skill: str | None = None
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    max_score: float | None = Field(default=None, ge=0.0, le=1.0)
    files_touched: str | None = None  # Match any file containing this substring


def _matches(result: ScoreResult, filters: QueryFilters) -> bool:
    """Check if a ScoreResult matches the given filters."""
    if filters.start_date and result.timestamp < filters.start_date:
        return False
    if filters.end_date and result.timestamp > filters.end_date:
        return False
    if filters.user and result.user != filters.user:
        return False
    if filters.skill and result.skill_used != filters.skill:
        return False
    if filters.min_score is not None and result.composite_score < filters.min_score:
        return False
    if filters.max_score is not None and result.composite_score > filters.max_score:
        return False
    if filters.files_touched and not any(filters.files_touched in f for f in result.files_touched):
        return False
    return True


# --- Storage Protocol ---


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for score storage backends."""

    def append(self, result: ScoreResult) -> None:
        """Append a score result to storage."""
        ...

    def query(self, filters: QueryFilters | None = None) -> list[ScoreResult]:
        """Query stored results with optional filters."""
        ...

    def count(self, filters: QueryFilters | None = None) -> int:
        """Count results matching filters."""
        ...

    def aggregate(
        self, filters: QueryFilters | None = None, group_by: str = "user"
    ) -> dict[str, Any]:
        """Aggregate results grouped by a field."""
        ...


# --- JSONL Backend ---


class JSONLBackend:
    """Append-only JSONL storage with daily file rotation.

    Files are stored as: {base_path}/scores-YYYY-MM-DD.jsonl
    Thread-safe via file locking on append.
    """

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _file_for_date(self, dt: datetime) -> Path:
        """Get the JSONL file path for a given date."""
        return self.base_path / f"scores-{dt.strftime('%Y-%m-%d')}.jsonl"

    def _all_files(self) -> list[Path]:
        """Get all JSONL score files, sorted by date."""
        return sorted(self.base_path.glob("scores-*.jsonl"))

    def _files_in_range(self, filters: QueryFilters | None) -> list[Path]:
        """Get JSONL files that could contain results matching the date range."""
        all_files = self._all_files()
        if not filters or (not filters.start_date and not filters.end_date):
            return all_files

        result = []
        for f in all_files:
            # Extract date from filename: scores-YYYY-MM-DD.jsonl
            try:
                date_str = f.stem.replace("scores-", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            if filters.start_date and file_date < filters.start_date.date():
                continue
            if filters.end_date and file_date > filters.end_date.date():
                continue
            result.append(f)
        return result

    def append(self, result: ScoreResult) -> None:
        """Append a score result. Thread-safe via file locking."""
        target_file = self._file_for_date(result.timestamp)
        line = result.model_dump_json() + "\n"

        with open(target_file, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _read_file(self, path: Path) -> list[ScoreResult]:
        """Read all ScoreResults from a single JSONL file."""
        results = []
        if not path.exists():
            return results
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(ScoreResult.model_validate_json(line))
                except Exception:
                    continue  # Skip malformed lines
        return results

    def query(self, filters: QueryFilters | None = None) -> list[ScoreResult]:
        """Query results with optional filters."""
        files = self._files_in_range(filters)
        results = []
        for f in files:
            for result in self._read_file(f):
                if filters is None or _matches(result, filters):
                    results.append(result)
        return results

    def count(self, filters: QueryFilters | None = None) -> int:
        """Count results matching filters."""
        return len(self.query(filters))

    def aggregate(
        self, filters: QueryFilters | None = None, group_by: str = "user"
    ) -> dict[str, Any]:
        """Aggregate results grouped by a field.

        Returns dict mapping group key to aggregate stats:
        {group_value: {"count": N, "avg_score": X, "min_score": X, "max_score": X}}
        """
        results = self.query(filters)
        groups: dict[str, list[float]] = {}

        for r in results:
            key = getattr(r, group_by, None)
            if key is None:
                key = "unknown"
            key = str(key)
            groups.setdefault(key, []).append(r.composite_score)

        aggregated: dict[str, Any] = {}
        for key, scores in groups.items():
            aggregated[key] = {
                "count": len(scores),
                "avg_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
            }
        return aggregated


# --- Factory ---


def create_storage(config: StorageConfig | None = None) -> JSONLBackend:
    """Create a storage backend from configuration.

    Args:
        config: Storage configuration. If None, uses defaults.

    Returns:
        A configured storage backend.
    """
    if config is None:
        config = StorageConfig()

    if config.backend == "jsonl":
        return JSONLBackend(base_path=config.path)

    raise ValueError(f"Unknown storage backend: {config.backend!r}. Supported: 'jsonl'")
