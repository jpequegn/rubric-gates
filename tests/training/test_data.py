"""Tests for training data pipeline."""

from unittest.mock import MagicMock
from datetime import datetime, timezone

from shared.models import Dimension, DimensionScore, ScoreResult, ScoringMethod

from training.data import ScorecardDataset, TrainingExample

import pytest


class TestTrainingExample:
    def test_defaults(self):
        ex = TrainingExample(code="x = 1", composite_score=0.5)
        assert ex.code == "x = 1"
        assert ex.composite_score == 0.5
        assert ex.dimension_scores == {}
        assert ex.metadata == {}

    def test_with_scores(self):
        ex = TrainingExample(
            code="pass",
            composite_score=0.8,
            dimension_scores={"correctness": 0.9, "security": 0.7},
            metadata={"tier": "high"},
        )
        assert ex.dimension_scores["correctness"] == 0.9
        assert ex.metadata["tier"] == "high"


class TestScorecardDataset:
    def test_empty_by_default(self):
        ds = ScorecardDataset()
        assert len(ds) == 0
        assert ds.examples == []

    def test_from_storage(self):
        mock_storage = MagicMock()
        mock_storage.query.return_value = [
            ScoreResult(
                composite_score=0.75,
                dimension_scores=[
                    DimensionScore(
                        dimension=Dimension.CORRECTNESS,
                        score=0.8,
                        method=ScoringMethod.RULE_BASED,
                    ),
                    DimensionScore(
                        dimension=Dimension.SECURITY,
                        score=0.7,
                        method=ScoringMethod.RULE_BASED,
                    ),
                ],
                user="alice",
                timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                skill_used="test-skill",
            ),
        ]

        ds = ScorecardDataset()
        examples = ds.from_storage(mock_storage)

        assert len(examples) == 1
        assert examples[0].composite_score == 0.75
        assert examples[0].dimension_scores["correctness"] == 0.8
        assert examples[0].metadata["user"] == "alice"

    def test_from_storage_min_scores_raises(self):
        mock_storage = MagicMock()
        mock_storage.query.return_value = []

        ds = ScorecardDataset()
        with pytest.raises(ValueError, match="Insufficient data"):
            ds.from_storage(mock_storage, min_scores=10)

    def test_from_storage_min_scores_passes(self):
        mock_storage = MagicMock()
        mock_storage.query.return_value = [
            ScoreResult(
                composite_score=0.5,
                dimension_scores=[],
                user="bob",
                timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                skill_used="s",
            )
        ]

        ds = ScorecardDataset()
        examples = ds.from_storage(mock_storage, min_scores=1)
        assert len(examples) == 1

    def test_from_storage_stores_examples(self):
        mock_storage = MagicMock()
        mock_storage.query.return_value = [
            ScoreResult(
                composite_score=0.6,
                dimension_scores=[],
                user="c",
                timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                skill_used="s",
            )
        ]

        ds = ScorecardDataset()
        ds.from_storage(mock_storage)
        assert len(ds) == 1
        assert ds.examples[0].composite_score == 0.6


class TestSyntheticGeneration:
    def test_generates_correct_count(self):
        ds = ScorecardDataset()
        examples = ds.generate_synthetic(num_examples=50, seed=42)
        assert len(examples) == 50
        assert len(ds) == 50

    def test_reproducible_with_seed(self):
        ds1 = ScorecardDataset()
        ex1 = ds1.generate_synthetic(num_examples=20, seed=123)

        ds2 = ScorecardDataset()
        ex2 = ds2.generate_synthetic(num_examples=20, seed=123)

        for a, b in zip(ex1, ex2):
            assert a.composite_score == b.composite_score
            assert a.code == b.code

    def test_different_seeds_differ(self):
        ds1 = ScorecardDataset()
        ex1 = ds1.generate_synthetic(num_examples=20, seed=1)

        ds2 = ScorecardDataset()
        ex2 = ds2.generate_synthetic(num_examples=20, seed=2)

        # At least some should differ
        differ = any(a.code != b.code for a, b in zip(ex1, ex2))
        assert differ

    def test_scores_in_valid_range(self):
        ds = ScorecardDataset()
        examples = ds.generate_synthetic(num_examples=100, seed=42)

        for ex in examples:
            assert 0.0 <= ex.composite_score <= 1.0
            for score in ex.dimension_scores.values():
                assert 0.0 <= score <= 1.0

    def test_metadata_present(self):
        ds = ScorecardDataset()
        examples = ds.generate_synthetic(num_examples=5, seed=42)

        for i, ex in enumerate(examples):
            assert ex.metadata["synthetic"] is True
            assert ex.metadata["tier"] in ("high", "medium", "low")
            assert ex.metadata["index"] == i

    def test_code_not_empty(self):
        ds = ScorecardDataset()
        examples = ds.generate_synthetic(num_examples=10, seed=42)

        for ex in examples:
            assert len(ex.code) > 0

    def test_dimension_scores_cover_all_dimensions(self):
        ds = ScorecardDataset()
        examples = ds.generate_synthetic(num_examples=5, seed=42)

        expected_dims = {d.value for d in Dimension}
        for ex in examples:
            assert set(ex.dimension_scores.keys()) == expected_dims


class TestSplit:
    def test_default_split_ratio(self):
        ds = ScorecardDataset()
        ds.generate_synthetic(num_examples=100, seed=42)
        train, eval_ = ds.split()
        assert len(train) == 80
        assert len(eval_) == 20

    def test_custom_ratio(self):
        ds = ScorecardDataset()
        ds.generate_synthetic(num_examples=100, seed=42)
        train, eval_ = ds.split(train_ratio=0.6)
        assert len(train) == 60
        assert len(eval_) == 40

    def test_reproducible(self):
        ds = ScorecardDataset()
        ds.generate_synthetic(num_examples=50, seed=42)
        train1, eval1 = ds.split(seed=99)
        train2, eval2 = ds.split(seed=99)

        assert [e.composite_score for e in train1] == [e.composite_score for e in train2]
        assert [e.composite_score for e in eval1] == [e.composite_score for e in eval2]

    def test_all_examples_present(self):
        ds = ScorecardDataset()
        ds.generate_synthetic(num_examples=50, seed=42)
        train, eval_ = ds.split()

        all_scores = sorted(e.composite_score for e in train + eval_)
        orig_scores = sorted(e.composite_score for e in ds.examples)
        assert all_scores == orig_scores


class TestToDicts:
    def test_empty(self):
        ds = ScorecardDataset()
        assert ds.to_dicts() == []

    def test_converts_to_dicts(self):
        ds = ScorecardDataset()
        ds.generate_synthetic(num_examples=5, seed=42)
        dicts = ds.to_dicts()

        assert len(dicts) == 5
        for d in dicts:
            assert "code" in d
            assert "composite_score" in d
            # Dimension scores are flattened with score_ prefix
            score_keys = [k for k in d if k.startswith("score_")]
            assert len(score_keys) > 0
