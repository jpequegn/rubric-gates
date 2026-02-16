"""Tests for source code storage in ScoreResult (issue #56)."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

from shared.models import Dimension, DimensionScore, ScoreResult, ScoringMethod
from shared.storage import JSONLBackend
from training.data import ScorecardDataset


SAMPLE_CODE = 'def hello():\n    """Say hello."""\n    return "world"\n'


# --- Model tests ---


class TestScoreResultSourceCode:
    def test_default_is_none(self):
        result = ScoreResult(user="test")
        assert result.source_code is None

    def test_set_source_code(self):
        result = ScoreResult(user="test", source_code=SAMPLE_CODE)
        assert result.source_code == SAMPLE_CODE

    def test_empty_string_is_valid(self):
        result = ScoreResult(user="test", source_code="")
        assert result.source_code == ""

    def test_json_roundtrip_with_code(self):
        result = ScoreResult(user="test", source_code=SAMPLE_CODE, composite_score=0.8)
        json_str = result.model_dump_json()
        restored = ScoreResult.model_validate_json(json_str)
        assert restored.source_code == SAMPLE_CODE

    def test_json_roundtrip_without_code(self):
        result = ScoreResult(user="test")
        json_str = result.model_dump_json()
        restored = ScoreResult.model_validate_json(json_str)
        assert restored.source_code is None

    def test_dict_roundtrip_with_code(self):
        result = ScoreResult(user="test", source_code=SAMPLE_CODE)
        d = result.model_dump()
        assert d["source_code"] == SAMPLE_CODE
        restored = ScoreResult.model_validate(d)
        assert restored.source_code == SAMPLE_CODE


# --- Backward compatibility ---


class TestBackwardCompatibility:
    def test_json_without_source_code_field(self):
        """Old JSONL records without source_code should still load."""
        raw = json.dumps(
            {
                "timestamp": "2026-01-15T10:00:00",
                "user": "alice",
                "skill_used": "scaffold",
                "files_touched": ["main.py"],
                "dimension_scores": [],
                "composite_score": 0.75,
                "metadata": {},
            }
        )
        result = ScoreResult.model_validate_json(raw)
        assert result.user == "alice"
        assert result.source_code is None
        assert result.composite_score == 0.75

    def test_dict_without_source_code_field(self):
        """Dict without source_code should default to None."""
        d = {
            "user": "bob",
            "composite_score": 0.6,
        }
        result = ScoreResult.model_validate(d)
        assert result.source_code is None


# --- Storage roundtrip ---


class TestStorageSourceCode:
    def test_append_and_query_with_code(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        original = ScoreResult(
            timestamp=datetime(2026, 6, 15, 14, 30, 0),
            user="jane",
            source_code=SAMPLE_CODE,
            composite_score=0.8,
        )
        backend.append(original)
        results = backend.query()
        assert len(results) == 1
        assert results[0].source_code == SAMPLE_CODE

    def test_append_and_query_without_code(self, tmp_path):
        backend = JSONLBackend(tmp_path)
        original = ScoreResult(
            timestamp=datetime(2026, 6, 15, 14, 30, 0),
            user="jane",
            composite_score=0.8,
        )
        backend.append(original)
        results = backend.query()
        assert len(results) == 1
        assert results[0].source_code is None

    def test_mixed_records(self, tmp_path):
        """Storage with both old (no code) and new (with code) records."""
        backend = JSONLBackend(tmp_path)
        ts = datetime(2026, 6, 15)
        backend.append(ScoreResult(user="a", timestamp=ts, composite_score=0.5))
        backend.append(
            ScoreResult(user="b", timestamp=ts, source_code=SAMPLE_CODE, composite_score=0.8)
        )

        results = backend.query()
        assert len(results) == 2
        assert results[0].source_code is None
        assert results[1].source_code == SAMPLE_CODE

    def test_large_source_code(self, tmp_path):
        """Large code strings survive storage roundtrip."""
        backend = JSONLBackend(tmp_path)
        large_code = "x = 1\n" * 10000  # ~60KB
        original = ScoreResult(
            user="test",
            source_code=large_code,
            composite_score=0.5,
        )
        backend.append(original)
        results = backend.query()
        assert results[0].source_code == large_code

    def test_code_with_special_chars(self, tmp_path):
        """Code with quotes, newlines, and unicode survives roundtrip."""
        backend = JSONLBackend(tmp_path)
        code = 'def greet(name):\n    return f"Hello, {name}! 🎉"\n'
        original = ScoreResult(user="test", source_code=code, composite_score=0.7)
        backend.append(original)
        results = backend.query()
        assert results[0].source_code == code


# --- Engine integration ---


class TestEngineSourceCode:
    def test_engine_populates_source_code(self):
        from scorecard.engine import RubricEngine

        engine = RubricEngine()
        result = engine.score("x = 1", "test.py", user="test")
        assert result.source_code == "x = 1"

    def test_engine_none_for_empty_code(self):
        from scorecard.engine import RubricEngine

        engine = RubricEngine()
        result = engine.score("", "test.py", user="test")
        assert result.source_code is None


# --- Training data pipeline ---


class TestTrainingDataSourceCode:
    def test_from_storage_uses_source_code(self):
        mock_storage = MagicMock()
        mock_storage.query.return_value = [
            ScoreResult(
                user="alice",
                composite_score=0.75,
                source_code=SAMPLE_CODE,
                dimension_scores=[
                    DimensionScore(
                        dimension=Dimension.CORRECTNESS,
                        score=0.8,
                        method=ScoringMethod.RULE_BASED,
                    ),
                ],
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]

        ds = ScorecardDataset()
        examples = ds.from_storage(mock_storage)
        assert len(examples) == 1
        assert examples[0].code == SAMPLE_CODE
        assert examples[0].metadata["has_source_code"] is True

    def test_from_storage_empty_code_without_source(self):
        mock_storage = MagicMock()
        mock_storage.query.return_value = [
            ScoreResult(
                user="bob",
                composite_score=0.5,
                dimension_scores=[],
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]

        ds = ScorecardDataset()
        examples = ds.from_storage(mock_storage)
        assert examples[0].code == ""
        assert examples[0].metadata["has_source_code"] is False

    def test_from_storage_mixed_with_and_without_code(self):
        mock_storage = MagicMock()
        mock_storage.query.return_value = [
            ScoreResult(
                user="a",
                composite_score=0.5,
                dimension_scores=[],
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            ScoreResult(
                user="b",
                composite_score=0.8,
                source_code="x = 1",
                dimension_scores=[],
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]

        ds = ScorecardDataset()
        examples = ds.from_storage(mock_storage)
        assert examples[0].code == ""
        assert examples[1].code == "x = 1"
