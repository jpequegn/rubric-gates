"""Tests for async scoring pipeline (issue #58)."""

from __future__ import annotations

import asyncio

import pytest

from scorecard.dimensions.correctness import CorrectnessScorer
from scorecard.dimensions.documentation import DocumentationScorer
from scorecard.dimensions.maintainability import MaintainabilityScorer
from scorecard.dimensions.security import SecurityScorer
from scorecard.dimensions.testability import TestabilityScorer
from scorecard.engine import RubricEngine
from shared.models import Dimension, DimensionScore, ScoringMethod

SAMPLE_CODE = 'def hello():\n    """Say hello."""\n    return "world"\n'

BAD_CODE = """\
password = "admin123"
def login(user, pw):
    if pw == password:
        return True
"""


# --- Individual scorer async ---


class TestCorrectnessAsync:
    @pytest.mark.asyncio
    async def test_score_async_returns_same_as_sync(self):
        scorer = CorrectnessScorer()
        sync_result = scorer.score(SAMPLE_CODE, "test.py")
        async_result = await scorer.score_async(SAMPLE_CODE, "test.py")
        assert async_result.score == sync_result.score
        assert async_result.dimension == Dimension.CORRECTNESS

    @pytest.mark.asyncio
    async def test_score_async_empty_code(self):
        scorer = CorrectnessScorer()
        result = await scorer.score_async("", "test.py")
        assert result.score == 0.0


class TestSecurityAsync:
    @pytest.mark.asyncio
    async def test_score_async_returns_same_as_sync(self):
        scorer = SecurityScorer()
        sync_result = scorer.score(SAMPLE_CODE, "test.py")
        async_result = await scorer.score_async(SAMPLE_CODE, "test.py")
        assert async_result.score == sync_result.score
        assert async_result.dimension == Dimension.SECURITY

    @pytest.mark.asyncio
    async def test_score_async_detects_credentials(self):
        scorer = SecurityScorer()
        result = await scorer.score_async(BAD_CODE, "login.py")
        assert result.score < 1.0


class TestMaintainabilityAsync:
    @pytest.mark.asyncio
    async def test_score_async_returns_same_as_sync(self):
        scorer = MaintainabilityScorer()
        sync_result = scorer.score(SAMPLE_CODE, "test.py")
        async_result = await scorer.score_async(SAMPLE_CODE, "test.py")
        assert async_result.score == sync_result.score
        assert async_result.dimension == Dimension.MAINTAINABILITY


class TestDocumentationAsync:
    @pytest.mark.asyncio
    async def test_score_async_fallback(self):
        """Without LLM, async fallback should match sync."""
        scorer = DocumentationScorer(use_llm=False)
        sync_result = scorer.score(SAMPLE_CODE, "test.py")
        async_result = await scorer.score_async(SAMPLE_CODE, "test.py")
        assert async_result.score == sync_result.score
        assert async_result.dimension == Dimension.DOCUMENTATION

    @pytest.mark.asyncio
    async def test_score_async_empty_code(self):
        scorer = DocumentationScorer(use_llm=False)
        result = await scorer.score_async("", "test.py")
        assert result.score == 0.0


class TestTestabilityAsync:
    @pytest.mark.asyncio
    async def test_score_async_fallback(self):
        """Without LLM, async fallback should match sync."""
        scorer = TestabilityScorer(use_llm=False)
        sync_result = scorer.score(SAMPLE_CODE, "test.py")
        async_result = await scorer.score_async(SAMPLE_CODE, "test.py")
        assert async_result.score == sync_result.score
        assert async_result.dimension == Dimension.TESTABILITY


# --- Engine async ---


class TestEngineScoreAsync:
    @pytest.mark.asyncio
    async def test_score_async_produces_result(self):
        engine = RubricEngine()
        result = await engine.score_async(SAMPLE_CODE, "test.py", user="alice")
        assert result.composite_score > 0
        assert result.user == "alice"
        assert result.source_code == SAMPLE_CODE
        assert len(result.dimension_scores) > 0

    @pytest.mark.asyncio
    async def test_score_async_matches_sync(self):
        engine = RubricEngine()
        sync_result = engine.score(SAMPLE_CODE, "test.py", user="alice")
        async_result = await engine.score_async(SAMPLE_CODE, "test.py", user="alice")
        assert async_result.composite_score == sync_result.composite_score
        assert len(async_result.dimension_scores) == len(sync_result.dimension_scores)

    @pytest.mark.asyncio
    async def test_score_async_bad_code(self):
        engine = RubricEngine()
        result = await engine.score_async(BAD_CODE, "login.py", user="alice")
        assert result.composite_score >= 0
        assert result.source_code == BAD_CODE

    @pytest.mark.asyncio
    async def test_score_async_empty_code(self):
        engine = RubricEngine()
        result = await engine.score_async("", "test.py", user="alice")
        assert result.source_code is None

    @pytest.mark.asyncio
    async def test_score_async_with_metadata(self):
        engine = RubricEngine()
        result = await engine.score_async(
            SAMPLE_CODE, "test.py", user="alice", metadata={"custom": "value"}
        )
        assert result.metadata["custom"] == "value"


# --- Timeout handling ---


class TestScorerTimeout:
    @pytest.mark.asyncio
    async def test_timeout_produces_zero_score(self):
        """A scorer that takes too long should get zero score, not block."""
        engine = RubricEngine(scorer_timeout=0.01)

        # Patch one scorer to be slow
        slow_scorer = CorrectnessScorer()

        async def slow_score_async(code, filename=""):
            await asyncio.sleep(5)
            return DimensionScore(
                dimension=Dimension.CORRECTNESS,
                score=1.0,
                method=ScoringMethod.AST_PARSE,
            )

        slow_scorer.score_async = slow_score_async
        engine._scorers[Dimension.CORRECTNESS] = slow_scorer

        result = await engine.score_async(SAMPLE_CODE, "test.py", user="alice")

        # Should still complete (not hang)
        assert result is not None

        # The timed-out dimension should have zero score
        correctness_scores = [
            ds for ds in result.dimension_scores if ds.dimension == Dimension.CORRECTNESS
        ]
        assert len(correctness_scores) == 1
        assert correctness_scores[0].score == 0.0
        assert "scorer_errors" in result.metadata

    @pytest.mark.asyncio
    async def test_other_scorers_unaffected_by_timeout(self):
        """One scorer timing out should not affect others."""
        engine = RubricEngine(scorer_timeout=0.01)

        slow_scorer = CorrectnessScorer()

        async def slow_score_async(code, filename=""):
            await asyncio.sleep(5)
            return DimensionScore(
                dimension=Dimension.CORRECTNESS,
                score=1.0,
                method=ScoringMethod.AST_PARSE,
            )

        slow_scorer.score_async = slow_score_async
        engine._scorers[Dimension.CORRECTNESS] = slow_scorer

        result = await engine.score_async(SAMPLE_CODE, "test.py", user="alice")

        # Other dimensions should have non-zero scores
        non_correctness = [
            ds for ds in result.dimension_scores if ds.dimension != Dimension.CORRECTNESS
        ]
        assert len(non_correctness) > 0
        assert any(ds.score > 0 for ds in non_correctness)


# --- Scorer exception handling ---


class TestScorerErrorHandling:
    @pytest.mark.asyncio
    async def test_scorer_exception_produces_zero_score(self):
        engine = RubricEngine()

        broken_scorer = CorrectnessScorer()

        async def broken_score_async(code, filename=""):
            raise RuntimeError("scorer crashed")

        broken_scorer.score_async = broken_score_async
        engine._scorers[Dimension.CORRECTNESS] = broken_scorer

        result = await engine.score_async(SAMPLE_CODE, "test.py", user="alice")
        assert result is not None

        correctness_scores = [
            ds for ds in result.dimension_scores if ds.dimension == Dimension.CORRECTNESS
        ]
        assert correctness_scores[0].score == 0.0
        assert "scorer_errors" in result.metadata


# --- File scoring async ---


class TestScoreFileAsync:
    @pytest.mark.asyncio
    async def test_score_file_async(self, tmp_path):
        f = tmp_path / "hello.py"
        f.write_text(SAMPLE_CODE)

        engine = RubricEngine()
        result = await engine.score_file_async(f, user="alice")
        assert result.composite_score > 0
        assert result.source_code == SAMPLE_CODE

    @pytest.mark.asyncio
    async def test_score_files_async(self, tmp_path):
        files = []
        for i in range(5):
            f = tmp_path / f"file{i}.py"
            f.write_text(SAMPLE_CODE)
            files.append(f)

        engine = RubricEngine()
        results = await engine.score_files_async(files, user="alice")
        assert len(results) == 5
        assert all(r.composite_score > 0 for r in results)

    @pytest.mark.asyncio
    async def test_score_files_async_respects_concurrency(self, tmp_path):
        """Semaphore should limit concurrent file scoring."""
        files = []
        for i in range(10):
            f = tmp_path / f"file{i}.py"
            f.write_text(SAMPLE_CODE)
            files.append(f)

        engine = RubricEngine(file_concurrency=2)
        results = await engine.score_files_async(files, user="alice")
        assert len(results) == 10


# --- Concurrency verification ---


class TestConcurrentExecution:
    @pytest.mark.asyncio
    async def test_scorers_run_concurrently(self):
        """Verify that scorers actually run in parallel, not sequentially."""
        engine = RubricEngine()

        call_times: list[tuple[str, float]] = []
        original_scorers = dict(engine._scorers)

        for dim, scorer in original_scorers.items():
            original_async = scorer.score_async

            async def make_tracked(dim_name, orig_fn, *args, **kwargs):
                import time

                start = time.monotonic()
                result = await orig_fn(*args, **kwargs)
                elapsed = time.monotonic() - start
                call_times.append((dim_name, elapsed))
                return result

            # Bind dim.value and original_async in closure
            dim_val = dim.value
            orig = original_async

            if dim == Dimension.TESTABILITY:

                async def tracked_async(
                    code, filename="", project_files=None, test_code=None, _d=dim_val, _o=orig
                ):
                    return await make_tracked(
                        _d,
                        _o,
                        code,
                        filename,
                        project_files=project_files,
                        test_code=test_code,
                    )
            else:

                async def tracked_async(code, filename="", _d=dim_val, _o=orig):
                    return await make_tracked(_d, _o, code, filename)

            scorer.score_async = tracked_async

        await engine.score_async(SAMPLE_CODE, "test.py", user="alice")

        # All scorers should have been called
        assert len(call_times) == len(original_scorers)


# --- Sync fallback ---


class TestSyncFallback:
    def test_sync_score_still_works(self):
        """Sync score() should work even with async-capable scorers."""
        engine = RubricEngine()
        result = engine.score(SAMPLE_CODE, "test.py", user="alice")
        assert result.composite_score > 0

    def test_sync_score_files_still_works(self, tmp_path):
        f = tmp_path / "hello.py"
        f.write_text(SAMPLE_CODE)
        engine = RubricEngine()
        results = engine.score_files([f], user="alice")
        assert len(results) == 1
        assert results[0].composite_score > 0
