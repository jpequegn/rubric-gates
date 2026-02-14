"""Tests for the rubric code generator."""

import json

import pytest

from training.rubric_generator import GenerationResult, RubricCodeGenerator


class TestGenerationResult:
    def test_defaults(self):
        result = GenerationResult()
        assert result.code == ""
        assert result.score is None
        assert result.gate_tier is None
        assert result.metadata is None


class TestRubricCodeGeneratorInit:
    def test_not_found(self):
        with pytest.raises(FileNotFoundError, match="Model not found"):
            RubricCodeGenerator("/nonexistent/path")

    def test_basic_init(self, tmp_path):
        gen = RubricCodeGenerator(str(tmp_path))
        assert not gen.is_loaded
        assert gen.model_name == ""
        assert gen._max_completion_length == 1024

    def test_loads_config(self, tmp_path):
        config = {
            "max_completion_length": 2048,
            "model_name": "test-model",
        }
        (tmp_path / "generator_config.json").write_text(json.dumps(config))

        gen = RubricCodeGenerator(str(tmp_path))
        assert gen._max_completion_length == 2048
        assert gen.model_name == "test-model"


class TestProperties:
    def test_is_loaded_false(self, tmp_path):
        gen = RubricCodeGenerator(str(tmp_path))
        assert gen.is_loaded is False


class TestGenerate:
    def test_raises_when_not_loaded(self, tmp_path):
        gen = RubricCodeGenerator(str(tmp_path))
        with pytest.raises(RuntimeError, match="Model not loaded"):
            gen.generate("Write a function")


class TestGenerateAndScore:
    def test_raises_when_not_loaded(self, tmp_path):
        gen = RubricCodeGenerator(str(tmp_path))
        with pytest.raises(RuntimeError, match="Model not loaded"):
            gen.generate_and_score("Write a function")


class TestGetMetadata:
    def test_no_meta_file(self, tmp_path):
        gen = RubricCodeGenerator(str(tmp_path))
        meta = gen.get_metadata()
        assert "max_completion_length" in meta
        assert meta["max_completion_length"] == 1024

    def test_with_meta_file(self, tmp_path):
        (tmp_path / "rubric_meta.json").write_text(
            json.dumps({"type": "rubric_code_generator", "grpo_config": {}})
        )
        (tmp_path / "generator_config.json").write_text(
            json.dumps({"max_completion_length": 512, "model_name": "my-model"})
        )

        gen = RubricCodeGenerator(str(tmp_path))
        meta = gen.get_metadata()

        assert meta["type"] == "rubric_code_generator"
        assert meta["max_completion_length"] == 512
        assert meta["model_name"] == "my-model"
