"""Tests for the distilled scorer."""

import json

import pytest

from shared.models import Dimension
from training.distilled_scorer import CalibrationMetrics, DistilledScorer


class TestCalibrationMetrics:
    def test_defaults(self):
        cal = CalibrationMetrics()
        assert cal.expected_calibration_error == 0.0
        assert cal.temperature == 1.0
        assert cal.platt_a == 1.0
        assert cal.platt_b == 0.0


class TestDistilledScorerInit:
    def test_not_found(self):
        with pytest.raises(FileNotFoundError, match="Model not found"):
            DistilledScorer("/nonexistent/path")

    def test_basic_init(self, tmp_path):
        scorer = DistilledScorer(str(tmp_path))
        assert not scorer.is_loaded
        assert len(scorer.dimensions) == len(Dimension)

    def test_loads_distill_config(self, tmp_path):
        config = {
            "dimensions": ["correctness", "security"],
            "prompt_template": "Rate {dimension}: {code}",
        }
        (tmp_path / "distill_config.json").write_text(json.dumps(config))

        scorer = DistilledScorer(str(tmp_path))
        assert scorer.dimensions == ["correctness", "security"]

    def test_loads_calibration(self, tmp_path):
        cal = {
            "expected_calibration_error": 0.03,
            "temperature": 1.2,
            "platt_a": 0.95,
            "platt_b": 0.02,
        }
        (tmp_path / "calibration.json").write_text(json.dumps(cal))

        scorer = DistilledScorer(str(tmp_path))
        assert scorer.calibration.expected_calibration_error == 0.03
        assert scorer.calibration.temperature == 1.2
        assert scorer.calibration.platt_a == 0.95
        assert scorer.calibration.platt_b == 0.02


class TestScorerProperties:
    def test_is_loaded_false(self, tmp_path):
        scorer = DistilledScorer(str(tmp_path))
        assert scorer.is_loaded is False

    def test_dimensions_returns_copy(self, tmp_path):
        scorer = DistilledScorer(str(tmp_path))
        dims = scorer.dimensions
        dims.append("fake")
        assert "fake" not in scorer.dimensions


class TestScore:
    def test_raises_when_not_loaded(self, tmp_path):
        scorer = DistilledScorer(str(tmp_path))
        with pytest.raises(RuntimeError, match="Model not loaded"):
            scorer.score("x = 1", "correctness")

    def test_raises_unknown_dimension(self, tmp_path):
        config = {"dimensions": ["correctness"]}
        (tmp_path / "distill_config.json").write_text(json.dumps(config))

        scorer = DistilledScorer(str(tmp_path))
        scorer._model = "fake"
        scorer._tokenizer = "fake"

        with pytest.raises(ValueError, match="Unknown dimension"):
            scorer.score("x = 1", "nonexistent")


class TestScoreAll:
    def test_raises_when_not_loaded(self, tmp_path):
        scorer = DistilledScorer(str(tmp_path))
        with pytest.raises(RuntimeError, match="Model not loaded"):
            scorer.score_all("x = 1")


class TestCalibrate:
    def test_identity_calibration(self, tmp_path):
        scorer = DistilledScorer(str(tmp_path))
        # Default calibration is identity: temp=1, a=1, b=0
        assert scorer._calibrate(0.5) == 0.5
        assert scorer._calibrate(0.0) == 0.0
        assert scorer._calibrate(1.0) == 1.0

    def test_temperature_scaling(self, tmp_path):
        scorer = DistilledScorer(str(tmp_path))
        scorer._calibration = CalibrationMetrics(temperature=2.0)
        # 0.8 / 2.0 * 1.0 + 0.0 = 0.4
        assert scorer._calibrate(0.8) == 0.4

    def test_platt_scaling(self, tmp_path):
        scorer = DistilledScorer(str(tmp_path))
        scorer._calibration = CalibrationMetrics(platt_a=0.9, platt_b=0.05)
        # 0.5 / 1.0 * 0.9 + 0.05 = 0.5
        assert scorer._calibrate(0.5) == 0.5

    def test_clamps_to_range(self, tmp_path):
        scorer = DistilledScorer(str(tmp_path))
        scorer._calibration = CalibrationMetrics(platt_a=2.0, platt_b=0.5)
        # Should clamp to 1.0
        assert scorer._calibrate(1.0) == 1.0

        scorer._calibration = CalibrationMetrics(platt_b=-0.5)
        # Should clamp to 0.0
        assert scorer._calibrate(0.0) == 0.0


class TestSaveCalibration:
    def test_saves_and_updates(self, tmp_path):
        scorer = DistilledScorer(str(tmp_path))

        metrics = CalibrationMetrics(
            expected_calibration_error=0.04,
            temperature=1.1,
            platt_a=0.98,
            platt_b=0.01,
        )
        scorer.save_calibration(metrics)

        assert scorer.calibration.temperature == 1.1
        cal_path = tmp_path / "calibration.json"
        assert cal_path.exists()

        saved = json.loads(cal_path.read_text())
        assert saved["temperature"] == 1.1
        assert saved["platt_a"] == 0.98

    def test_roundtrip(self, tmp_path):
        scorer1 = DistilledScorer(str(tmp_path))
        metrics = CalibrationMetrics(
            expected_calibration_error=0.05,
            temperature=1.3,
            platt_a=0.92,
            platt_b=0.03,
        )
        scorer1.save_calibration(metrics)

        scorer2 = DistilledScorer(str(tmp_path))
        assert scorer2.calibration.temperature == 1.3
        assert scorer2.calibration.platt_a == 0.92


class TestGetMetadata:
    def test_no_meta_file(self, tmp_path):
        scorer = DistilledScorer(str(tmp_path))
        meta = scorer.get_metadata()

        assert "calibration" in meta
        assert "dimensions" in meta
        assert meta["calibration"]["ece"] == 0.0

    def test_with_meta_file(self, tmp_path):
        (tmp_path / "rubric_meta.json").write_text(
            json.dumps({"model_name": "test-model", "type": "distilled_scorer"})
        )

        scorer = DistilledScorer(str(tmp_path))
        meta = scorer.get_metadata()

        assert meta["model_name"] == "test-model"
        assert meta["type"] == "distilled_scorer"
        assert "calibration" in meta
        assert "dimensions" in meta
