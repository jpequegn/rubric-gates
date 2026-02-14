"""Tests for skill optimization runner."""

import json

from training.skill_optimizer import (
    OptimizerConfig,
    OptimizedSkill,
    SkillOptimizer,
)
from training.skill_runner import (
    BatchReport,
    SkillCandidate,
    SkillOptimizationRunner,
)


def _make_runner(tmp_path):
    optimizer = SkillOptimizer(config=OptimizerConfig(max_iterations=2))
    return SkillOptimizationRunner(
        optimizer=optimizer,
        skills_dir=tmp_path / "skills",
        backup_dir=tmp_path / "backups",
    )


class TestSkillCandidate:
    def test_defaults(self):
        c = SkillCandidate()
        assert c.name == ""
        assert c.avg_score == 0.0
        assert c.weak_dimensions == []


class TestBatchReport:
    def test_defaults(self):
        r = BatchReport()
        assert r.total_skills == 0
        assert r.results == []
        assert r.comparisons == []


class TestIdentifyCandidates:
    def test_finds_low_scoring_skills(self, tmp_path):
        runner = _make_runner(tmp_path)
        histories = {
            "good-skill": [
                {"correctness": 0.9, "security": 0.8},
            ]
            * 10,
            "bad-skill": [
                {"correctness": 0.4, "security": 0.3},
            ]
            * 10,
        }
        candidates = runner.identify_candidates(score_histories=histories, min_score_history=5)
        names = [c.name for c in candidates]
        assert "bad-skill" in names
        assert "good-skill" not in names

    def test_respects_min_history(self, tmp_path):
        runner = _make_runner(tmp_path)
        histories = {
            "too-few": [{"correctness": 0.3}] * 2,
        }
        candidates = runner.identify_candidates(score_histories=histories, min_score_history=5)
        assert len(candidates) == 0

    def test_sorted_by_score(self, tmp_path):
        runner = _make_runner(tmp_path)
        histories = {
            "medium": [{"correctness": 0.5}] * 10,
            "worst": [{"correctness": 0.3}] * 10,
            "okay": [{"correctness": 0.6}] * 10,
        }
        candidates = runner.identify_candidates(score_histories=histories, min_score_history=5)
        if len(candidates) >= 2:
            assert candidates[0].avg_score <= candidates[1].avg_score


class TestRunBatch:
    def test_runs_batch(self, tmp_path):
        runner = _make_runner(tmp_path)
        skills = {
            "skill-a": "Write Python code",
            "skill-b": "Build a tool",
        }
        report = runner.run_batch(
            skills=skills,
            test_tasks=["Sort a list", "Parse JSON"],
        )
        assert isinstance(report, BatchReport)
        assert report.total_skills == 2
        assert len(report.results) == 2

    def test_counts_correct(self, tmp_path):
        runner = _make_runner(tmp_path)
        skills = {"skill": "Basic"}
        report = runner.run_batch(
            skills=skills,
            test_tasks=["task"],
        )
        total = report.optimized + report.no_improvement + report.rejected
        assert total == 1


class TestDeploy:
    def test_deploys_with_backup(self, tmp_path):
        runner = _make_runner(tmp_path)
        optimized = OptimizedSkill(
            skill_name="test-skill",
            original_prompt="Old prompt",
            optimized_prompt="New prompt",
            original_avg_score=0.5,
            optimized_avg_score=0.7,
            improvement=0.2,
        )

        backup_path = runner.deploy("test-skill", optimized, backup=True)

        assert backup_path is not None
        assert backup_path.exists()

        backup_data = json.loads(backup_path.read_text())
        assert backup_data["original_prompt"] == "Old prompt"

        skill_path = tmp_path / "skills" / "test-skill.txt"
        assert skill_path.exists()
        assert skill_path.read_text() == "New prompt"

    def test_deploys_without_backup(self, tmp_path):
        runner = SkillOptimizationRunner(
            optimizer=SkillOptimizer(),
            skills_dir=tmp_path / "skills",
        )
        optimized = OptimizedSkill(
            skill_name="test",
            optimized_prompt="New",
        )
        backup_path = runner.deploy("test", optimized, backup=True)
        assert backup_path is None  # No backup dir configured

    def test_tracks_deployed(self, tmp_path):
        runner = _make_runner(tmp_path)
        optimized = OptimizedSkill(
            skill_name="my-skill",
            improvement=0.15,
        )
        runner.deploy("my-skill", optimized)
        assert "my-skill" in runner.list_deployed()
        assert runner.list_deployed()["my-skill"] == 0.15


class TestRollback:
    def test_rollback_success(self, tmp_path):
        runner = _make_runner(tmp_path)
        optimized = OptimizedSkill(
            skill_name="test-skill",
            original_prompt="Original prompt text",
            optimized_prompt="Optimized text",
        )
        runner.deploy("test-skill", optimized, backup=True)

        result = runner.rollback("test-skill")
        assert result is True

        skill_path = tmp_path / "skills" / "test-skill.txt"
        assert skill_path.read_text() == "Original prompt text"

    def test_rollback_no_backup(self, tmp_path):
        runner = _make_runner(tmp_path)
        result = runner.rollback("nonexistent")
        assert result is False

    def test_rollback_removes_from_deployed(self, tmp_path):
        runner = _make_runner(tmp_path)
        optimized = OptimizedSkill(skill_name="test")
        runner.deploy("test", optimized, backup=True)
        assert "test" in runner.list_deployed()

        runner.rollback("test")
        assert "test" not in runner.list_deployed()


class TestListDeployed:
    def test_empty(self, tmp_path):
        runner = _make_runner(tmp_path)
        assert runner.list_deployed() == {}


class TestGenerateReport:
    def test_generates_markdown(self, tmp_path):
        runner = _make_runner(tmp_path)
        report = BatchReport(
            total_skills=2,
            optimized=1,
            no_improvement=1,
            rejected=0,
            results=[
                OptimizedSkill(
                    skill_name="skill-a",
                    original_avg_score=0.5,
                    optimized_avg_score=0.65,
                    improvement=0.15,
                    strategy_used="instruction_refinement",
                ),
                OptimizedSkill(
                    skill_name="skill-b",
                    original_avg_score=0.7,
                    optimized_avg_score=0.7,
                    improvement=0.0,
                ),
            ],
        )
        output = runner.generate_report(report)
        assert "## Skill Optimization Report" in output
        assert "skill-a" in output
        assert "skill-b" in output
        assert "instruction_refinement" in output
        assert "+0.1500" in output

    def test_empty_report(self, tmp_path):
        runner = _make_runner(tmp_path)
        report = BatchReport()
        output = runner.generate_report(report)
        assert "## Skill Optimization Report" in output
        assert "**Skills analyzed:** 0" in output
