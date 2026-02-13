"""Tests for storage backends."""

from datetime import datetime, timedelta

import pytest

from shared.config import StorageConfig
from shared.models import Dimension, DimensionScore, ScoreResult, ScoringMethod
from shared.storage import JSONLBackend, QueryFilters, StorageBackend, create_storage


# --- Helpers ---


def make_result(
    user: str = "alice",
    skill: str = "scaffold-api",
    score: float = 0.75,
    timestamp: datetime | None = None,
    files: list[str] | None = None,
) -> ScoreResult:
    """Create a ScoreResult for testing."""
    return ScoreResult(
        timestamp=timestamp or datetime.now(),
        user=user,
        skill_used=skill,
        files_touched=files or ["main.py"],
        dimension_scores=[
            DimensionScore(
                dimension=Dimension.CORRECTNESS,
                score=score,
                method=ScoringMethod.AST_PARSE,
            ),
        ],
        composite_score=score,
    )


# --- Protocol ---


class TestProtocol:
    def test_jsonl_implements_protocol(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        assert isinstance(backend, StorageBackend)


# --- JSONL Append ---


class TestJSONLAppend:
    def test_append_creates_file(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        result = make_result()
        backend.append(result)

        files = list(tmp_path.glob("scores-*.jsonl"))
        assert len(files) == 1

    def test_append_file_named_by_date(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        ts = datetime(2026, 3, 15, 10, 30, 0)
        result = make_result(timestamp=ts)
        backend.append(result)

        expected = tmp_path / "scores-2026-03-15.jsonl"
        assert expected.exists()

    def test_append_multiple_same_day(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        ts = datetime(2026, 3, 15, 10, 0, 0)
        backend.append(make_result(user="alice", timestamp=ts))
        backend.append(make_result(user="bob", timestamp=ts))

        files = list(tmp_path.glob("scores-*.jsonl"))
        assert len(files) == 1

        lines = (tmp_path / "scores-2026-03-15.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

    def test_append_different_days(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        backend.append(make_result(timestamp=datetime(2026, 3, 15)))
        backend.append(make_result(timestamp=datetime(2026, 3, 16)))

        files = list(tmp_path.glob("scores-*.jsonl"))
        assert len(files) == 2

    def test_append_creates_base_dir(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "dir"
        backend = JSONLBackend(nested)
        backend.append(make_result())
        assert nested.exists()

    def test_appended_data_is_valid_json(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        result = make_result(user="test", score=0.82)
        backend.append(result)

        file = list(tmp_path.glob("scores-*.jsonl"))[0]
        import json

        line = file.read_text().strip()
        data = json.loads(line)
        assert data["user"] == "test"
        assert data["composite_score"] == 0.82


# --- JSONL Query ---


class TestJSONLQuery:
    @pytest.fixture()
    def populated_backend(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        base = datetime(2026, 3, 15, 10, 0, 0)
        backend.append(make_result(user="alice", skill="api", score=0.9, timestamp=base))
        backend.append(
            make_result(
                user="bob",
                skill="scaffold",
                score=0.5,
                timestamp=base + timedelta(hours=1),
            )
        )
        backend.append(
            make_result(
                user="alice",
                skill="api",
                score=0.7,
                timestamp=base + timedelta(days=1),
            )
        )
        backend.append(
            make_result(
                user="charlie",
                skill="test-gen",
                score=0.3,
                timestamp=base + timedelta(days=2),
                files=["utils.py", "test_utils.py"],
            )
        )
        return backend

    def test_query_all(self, populated_backend):
        results = populated_backend.query()
        assert len(results) == 4

    def test_query_no_filters(self, populated_backend):
        results = populated_backend.query(QueryFilters())
        assert len(results) == 4

    def test_filter_by_user(self, populated_backend):
        results = populated_backend.query(QueryFilters(user="alice"))
        assert len(results) == 2
        assert all(r.user == "alice" for r in results)

    def test_filter_by_skill(self, populated_backend):
        results = populated_backend.query(QueryFilters(skill="api"))
        assert len(results) == 2
        assert all(r.skill_used == "api" for r in results)

    def test_filter_by_min_score(self, populated_backend):
        results = populated_backend.query(QueryFilters(min_score=0.7))
        assert len(results) == 2
        assert all(r.composite_score >= 0.7 for r in results)

    def test_filter_by_max_score(self, populated_backend):
        results = populated_backend.query(QueryFilters(max_score=0.5))
        assert len(results) == 2
        assert all(r.composite_score <= 0.5 for r in results)

    def test_filter_by_score_range(self, populated_backend):
        results = populated_backend.query(QueryFilters(min_score=0.5, max_score=0.7))
        assert len(results) == 2

    def test_filter_by_date_range(self, populated_backend):
        start = datetime(2026, 3, 16)
        end = datetime(2026, 3, 17, 23, 59, 59)
        results = populated_backend.query(QueryFilters(start_date=start, end_date=end))
        assert len(results) == 2

    def test_filter_by_start_date_only(self, populated_backend):
        start = datetime(2026, 3, 17)
        results = populated_backend.query(QueryFilters(start_date=start))
        assert len(results) == 1
        assert results[0].user == "charlie"

    def test_filter_by_files_touched(self, populated_backend):
        results = populated_backend.query(QueryFilters(files_touched="utils.py"))
        assert len(results) == 1
        assert results[0].user == "charlie"

    def test_combined_filters(self, populated_backend):
        results = populated_backend.query(QueryFilters(user="alice", min_score=0.8))
        assert len(results) == 1
        assert results[0].composite_score == 0.9

    def test_no_matches(self, populated_backend):
        results = populated_backend.query(QueryFilters(user="nobody"))
        assert len(results) == 0

    def test_query_empty_storage(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        results = backend.query()
        assert results == []


# --- JSONL Count ---


class TestJSONLCount:
    def test_count_all(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        for i in range(5):
            backend.append(make_result(score=i * 0.2))
        assert backend.count() == 5

    def test_count_with_filter(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        backend.append(make_result(user="alice"))
        backend.append(make_result(user="bob"))
        backend.append(make_result(user="alice"))
        assert backend.count(QueryFilters(user="alice")) == 2

    def test_count_empty(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        assert backend.count() == 0


# --- JSONL Aggregate ---


class TestJSONLAggregate:
    def test_aggregate_by_user(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        backend.append(make_result(user="alice", score=0.9))
        backend.append(make_result(user="alice", score=0.7))
        backend.append(make_result(user="bob", score=0.5))

        agg = backend.aggregate(group_by="user")
        assert "alice" in agg
        assert "bob" in agg
        assert agg["alice"]["count"] == 2
        assert agg["alice"]["avg_score"] == pytest.approx(0.8)
        assert agg["alice"]["min_score"] == 0.7
        assert agg["alice"]["max_score"] == 0.9
        assert agg["bob"]["count"] == 1

    def test_aggregate_by_skill(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        backend.append(make_result(skill="api", score=0.8))
        backend.append(make_result(skill="api", score=0.6))
        backend.append(make_result(skill="scaffold", score=0.9))

        agg = backend.aggregate(group_by="skill_used")
        assert "api" in agg
        assert "scaffold" in agg
        assert agg["api"]["avg_score"] == pytest.approx(0.7)

    def test_aggregate_with_filters(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        backend.append(make_result(user="alice", score=0.9))
        backend.append(make_result(user="alice", score=0.3))
        backend.append(make_result(user="bob", score=0.5))

        agg = backend.aggregate(filters=QueryFilters(min_score=0.5), group_by="user")
        assert "alice" in agg
        assert agg["alice"]["count"] == 1  # Only the 0.9 score
        assert "bob" in agg

    def test_aggregate_empty(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        agg = backend.aggregate()
        assert agg == {}


# --- File Rotation ---


class TestFileRotation:
    def test_daily_rotation(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        for day in range(5):
            ts = datetime(2026, 3, 10 + day, 12, 0, 0)
            backend.append(make_result(timestamp=ts))

        files = sorted(tmp_path.glob("scores-*.jsonl"))
        assert len(files) == 5
        assert files[0].name == "scores-2026-03-10.jsonl"
        assert files[4].name == "scores-2026-03-14.jsonl"

    def test_date_range_skips_irrelevant_files(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        for day in range(5):
            ts = datetime(2026, 3, 10 + day, 12, 0, 0)
            backend.append(make_result(timestamp=ts))

        # Query only days 12-13
        results = backend.query(
            QueryFilters(
                start_date=datetime(2026, 3, 12),
                end_date=datetime(2026, 3, 13, 23, 59, 59),
            )
        )
        assert len(results) == 2


# --- Malformed Data ---


class TestMalformedData:
    def test_skips_empty_lines(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        backend.append(make_result())

        # Inject an empty line
        file = list(tmp_path.glob("scores-*.jsonl"))[0]
        with open(file, "a") as f:
            f.write("\n\n")

        results = backend.query()
        assert len(results) == 1

    def test_skips_malformed_json(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        backend.append(make_result())

        file = list(tmp_path.glob("scores-*.jsonl"))[0]
        with open(file, "a") as f:
            f.write("this is not json\n")
            f.write('{"incomplete": true}\n')

        results = backend.query()
        assert len(results) == 1  # Only the valid one


# --- Factory ---


class TestFactory:
    def test_create_jsonl(self, tmp_path):
        config = StorageConfig(backend="jsonl", path=str(tmp_path / "data"))
        backend = create_storage(config)
        assert isinstance(backend, JSONLBackend)

    def test_create_default(self):
        backend = create_storage()
        assert isinstance(backend, JSONLBackend)

    def test_create_unknown_raises(self):
        config = StorageConfig(backend="postgres", path="/tmp")
        with pytest.raises(ValueError, match="Unknown storage backend"):
            create_storage(config)

    def test_factory_created_backend_works(self, tmp_path):
        config = StorageConfig(backend="jsonl", path=str(tmp_path))
        backend = create_storage(config)
        backend.append(make_result(user="test"))
        assert backend.count() == 1
        assert backend.query(QueryFilters(user="test"))[0].user == "test"


# --- Roundtrip ---


class TestRoundtrip:
    def test_full_roundtrip(self, tmp_path):
        """Append → query → verify all fields survive serialization."""
        backend = JSONLBackend(tmp_path)
        original = ScoreResult(
            timestamp=datetime(2026, 6, 15, 14, 30, 0),
            user="jane.doe",
            skill_used="generate-tool",
            files_touched=["app.py", "utils.py"],
            dimension_scores=[
                DimensionScore(
                    dimension=Dimension.CORRECTNESS,
                    score=0.9,
                    method=ScoringMethod.AST_PARSE,
                    details="All checks passed",
                ),
                DimensionScore(
                    dimension=Dimension.SECURITY,
                    score=0.7,
                    method=ScoringMethod.RULE_BASED,
                    details="HTTP URL found",
                    metadata={"finding_count": 1},
                ),
            ],
            composite_score=0.8,
            metadata={"session_id": "abc123", "duration_ms": 450},
        )

        backend.append(original)
        results = backend.query()
        assert len(results) == 1

        restored = results[0]
        assert restored.user == "jane.doe"
        assert restored.skill_used == "generate-tool"
        assert restored.files_touched == ["app.py", "utils.py"]
        assert restored.composite_score == 0.8
        assert len(restored.dimension_scores) == 2
        assert restored.dimension_scores[0].dimension == Dimension.CORRECTNESS
        assert restored.dimension_scores[1].metadata["finding_count"] == 1
        assert restored.metadata["session_id"] == "abc123"
