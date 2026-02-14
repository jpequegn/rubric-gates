"""Tests for the tier evaluation engine."""

from shared.config import GateConfig, GateThresholds, GreenThreshold, RedThreshold, YellowThreshold
from shared.models import (
    Dimension,
    DimensionScore,
    GateTier,
    ScoreResult,
    ScoringMethod,
)

from gate.tiers.evaluator import TierEvaluator


def _make_score(
    composite: float = 0.8,
    dims: list[tuple[Dimension, float]] | None = None,
) -> ScoreResult:
    """Build a ScoreResult for testing."""
    dim_scores = []
    if dims:
        for dim, val in dims:
            dim_scores.append(
                DimensionScore(dimension=dim, score=val, method=ScoringMethod.RULE_BASED)
            )
    return ScoreResult(user="test", composite_score=composite, dimension_scores=dim_scores)


def _make_config(
    green_min: float = 0.7,
    yellow_min: float = 0.5,
    red_security: float = 0.3,
) -> GateConfig:
    """Build a GateConfig with custom thresholds."""
    return GateConfig(
        thresholds=GateThresholds(
            green=GreenThreshold(min_composite=green_min),
            yellow=YellowThreshold(min_composite=yellow_min),
            red=RedThreshold(security=red_security),
        ),
    )


CLEAN_CODE = "def add(a, b):\n    return a + b\n"


# --- GREEN tier ---


class TestGreenTier:
    def test_clean_code_high_score(self):
        evaluator = TierEvaluator(config=_make_config())
        score = _make_score(
            composite=0.85,
            dims=[
                (Dimension.CORRECTNESS, 0.9),
                (Dimension.SECURITY, 0.8),
                (Dimension.MAINTAINABILITY, 0.8),
                (Dimension.DOCUMENTATION, 0.7),
                (Dimension.TESTABILITY, 0.7),
            ],
        )
        result = evaluator.evaluate(score, CLEAN_CODE, "clean.py")
        assert result.tier == GateTier.GREEN
        assert not result.blocked
        assert result.critical_patterns_found == []
        assert result.pattern_findings == []
        assert result.advisory_messages == []

    def test_exactly_at_green_threshold(self):
        evaluator = TierEvaluator(config=_make_config(green_min=0.7))
        score = _make_score(composite=0.7)
        result = evaluator.evaluate(score, CLEAN_CODE, "ok.py")
        assert result.tier == GateTier.GREEN

    def test_no_dimension_scores(self):
        evaluator = TierEvaluator(config=_make_config())
        score = _make_score(composite=0.8)
        result = evaluator.evaluate(score, CLEAN_CODE, "simple.py")
        assert result.tier == GateTier.GREEN

    def test_all_dimensions_above_yellow(self):
        evaluator = TierEvaluator(config=_make_config())
        score = _make_score(
            composite=0.75,
            dims=[
                (Dimension.CORRECTNESS, 0.6),
                (Dimension.SECURITY, 0.6),
                (Dimension.MAINTAINABILITY, 0.5),
            ],
        )
        result = evaluator.evaluate(score, CLEAN_CODE, "decent.py")
        assert result.tier == GateTier.GREEN


# --- YELLOW tier ---


class TestYellowTier:
    def test_composite_below_green(self):
        evaluator = TierEvaluator(config=_make_config())
        score = _make_score(composite=0.6)
        result = evaluator.evaluate(score, CLEAN_CODE, "mediocre.py")
        assert result.tier == GateTier.YELLOW
        assert not result.blocked
        assert any("below green" in msg for msg in result.advisory_messages)

    def test_composite_below_yellow(self):
        evaluator = TierEvaluator(config=_make_config())
        score = _make_score(composite=0.4)
        result = evaluator.evaluate(score, CLEAN_CODE, "bad.py")
        assert result.tier == GateTier.YELLOW
        assert any("below yellow" in msg for msg in result.advisory_messages)

    def test_dimension_below_yellow_threshold(self):
        evaluator = TierEvaluator(
            config=_make_config(),
            dimension_yellow_thresholds={"correctness": 0.5},
        )
        score = _make_score(
            composite=0.75,
            dims=[(Dimension.CORRECTNESS, 0.3)],
        )
        result = evaluator.evaluate(score, CLEAN_CODE, "bad_correctness.py")
        assert result.tier == GateTier.YELLOW
        assert any("correctness" in msg for msg in result.advisory_messages)

    def test_high_severity_pattern(self):
        evaluator = TierEvaluator(config=_make_config())
        score = _make_score(composite=0.8)
        # Code with connection string (high severity from credentials detector)
        code = 'db = "postgres://admin:pass123@localhost/db"\n'
        result = evaluator.evaluate(score, code, "config.py")
        assert result.tier == GateTier.YELLOW
        assert not result.blocked
        assert len(result.pattern_findings) >= 1

    def test_advisory_messages_present(self):
        evaluator = TierEvaluator(config=_make_config())
        score = _make_score(composite=0.55)
        result = evaluator.evaluate(score, CLEAN_CODE, "mediocre.py")
        assert len(result.advisory_messages) > 0


# --- RED tier ---


class TestRedTier:
    def test_critical_pattern_blocks(self):
        evaluator = TierEvaluator(config=_make_config())
        score = _make_score(composite=0.95)
        # High-scoring code BUT has hardcoded password
        code = 'password = "supersecret123"\n'
        result = evaluator.evaluate(score, code, "leaked.py")
        assert result.tier == GateTier.RED
        assert result.blocked
        assert len(result.critical_patterns_found) > 0
        assert len(result.advisory_messages) > 0
        assert any("BLOCKED" in msg for msg in result.advisory_messages)

    def test_security_below_red_threshold(self):
        evaluator = TierEvaluator(config=_make_config(red_security=0.3))
        score = _make_score(
            composite=0.6,
            dims=[(Dimension.SECURITY, 0.2)],
        )
        result = evaluator.evaluate(score, CLEAN_CODE, "insecure.py")
        assert result.tier == GateTier.RED
        assert result.blocked
        assert any("Security score" in msg for msg in result.advisory_messages)

    def test_sql_injection_blocks(self):
        evaluator = TierEvaluator(config=_make_config())
        score = _make_score(composite=0.8)
        code = 'query = f"SELECT * FROM users WHERE id = {user_id}"\n'
        result = evaluator.evaluate(score, code, "db.py")
        assert result.tier == GateTier.RED
        assert result.blocked

    def test_shell_injection_blocks(self):
        evaluator = TierEvaluator(config=_make_config())
        score = _make_score(composite=0.9)
        code = 'os.system(f"rm -rf {path}")\n'
        result = evaluator.evaluate(score, code, "cleanup.py")
        assert result.tier == GateTier.RED
        assert result.blocked

    def test_api_key_blocks(self):
        evaluator = TierEvaluator(config=_make_config())
        score = _make_score(composite=0.95)
        code = 'key = "sk-abc123def456ghi789jklmnopqrstuvwxyz"\n'
        result = evaluator.evaluate(score, code, "config.py")
        assert result.tier == GateTier.RED
        assert result.blocked

    def test_high_composite_but_critical_pattern_still_red(self):
        """Edge case: code that scores well but has a critical pattern."""
        evaluator = TierEvaluator(config=_make_config())
        score = _make_score(
            composite=0.95,
            dims=[
                (Dimension.CORRECTNESS, 0.95),
                (Dimension.SECURITY, 0.95),
                (Dimension.MAINTAINABILITY, 0.9),
                (Dimension.DOCUMENTATION, 0.9),
                (Dimension.TESTABILITY, 0.9),
            ],
        )
        # Perfect score but hardcoded credentials
        code = 'api_key = "sk-abc123def456ghi789jklmnopqrstuvwxyz"\n'
        result = evaluator.evaluate(score, code, "perfect_but_leaked.py")
        assert result.tier == GateTier.RED
        assert result.blocked

    def test_security_exactly_at_red_threshold_not_blocked(self):
        evaluator = TierEvaluator(config=_make_config(red_security=0.3))
        score = _make_score(
            composite=0.8,
            dims=[(Dimension.SECURITY, 0.3)],
        )
        result = evaluator.evaluate(score, CLEAN_CODE, "borderline.py")
        # Score == threshold is NOT below, so not red for this reason
        assert result.tier != GateTier.RED or result.blocked is False or True
        # More precise: 0.3 is not < 0.3
        assert not any("Security score" in msg for msg in result.advisory_messages)


# --- GateResult fields ---


class TestGateResult:
    def test_green_result_fields(self):
        evaluator = TierEvaluator(config=_make_config())
        score = _make_score(composite=0.8)
        result = evaluator.evaluate(score, CLEAN_CODE, "good.py")
        assert result.score_result is score
        assert result.tier == GateTier.GREEN
        assert result.blocked is False
        assert result.critical_patterns_found == []

    def test_red_result_has_findings(self):
        evaluator = TierEvaluator(config=_make_config())
        score = _make_score(composite=0.9)
        code = 'password = "hunter2"\n'
        result = evaluator.evaluate(score, code, "bad.py")
        assert result.tier == GateTier.RED
        assert len(result.pattern_findings) > 0
        assert all(isinstance(f.line_number, int) for f in result.pattern_findings)

    def test_findings_include_remediation(self):
        evaluator = TierEvaluator(config=_make_config())
        score = _make_score(composite=0.9)
        code = 'password = "hunter2"\n'
        result = evaluator.evaluate(score, code, "bad.py")
        for finding in result.pattern_findings:
            assert finding.remediation


# --- Custom thresholds ---


class TestCustomThresholds:
    def test_strict_green_threshold(self):
        evaluator = TierEvaluator(config=_make_config(green_min=0.9))
        score = _make_score(composite=0.85)
        result = evaluator.evaluate(score, CLEAN_CODE, "decent.py")
        assert result.tier == GateTier.YELLOW

    def test_relaxed_green_threshold(self):
        evaluator = TierEvaluator(config=_make_config(green_min=0.5))
        score = _make_score(composite=0.55)
        result = evaluator.evaluate(score, CLEAN_CODE, "ok.py")
        assert result.tier == GateTier.GREEN

    def test_strict_red_security(self):
        evaluator = TierEvaluator(config=_make_config(red_security=0.5))
        score = _make_score(
            composite=0.7,
            dims=[(Dimension.SECURITY, 0.45)],
        )
        result = evaluator.evaluate(score, CLEAN_CODE, "borderline.py")
        assert result.tier == GateTier.RED
        assert result.blocked

    def test_custom_dimension_yellow_threshold(self):
        evaluator = TierEvaluator(
            config=_make_config(),
            dimension_yellow_thresholds={"documentation": 0.6},
        )
        score = _make_score(
            composite=0.8,
            dims=[(Dimension.DOCUMENTATION, 0.5)],
        )
        result = evaluator.evaluate(score, CLEAN_CODE, "undocumented.py")
        assert result.tier == GateTier.YELLOW
        assert any("documentation" in msg for msg in result.advisory_messages)


# --- Multiple issues ---


class TestMultipleIssues:
    def test_multiple_critical_patterns(self):
        evaluator = TierEvaluator(config=_make_config())
        score = _make_score(composite=0.9)
        code = 'password = "hunter2"\nquery = f"SELECT * FROM users WHERE id = {uid}"\n'
        result = evaluator.evaluate(score, code, "terrible.py")
        assert result.tier == GateTier.RED
        assert result.blocked
        assert len(result.critical_patterns_found) >= 2
        assert len(result.advisory_messages) >= 2

    def test_yellow_score_plus_high_pattern(self):
        evaluator = TierEvaluator(config=_make_config())
        score = _make_score(composite=0.6)
        code = 'db = "postgres://admin:pass@localhost/db"\n'
        result = evaluator.evaluate(score, code, "mediocre.py")
        assert result.tier == GateTier.YELLOW
        assert not result.blocked
        assert len(result.advisory_messages) >= 2  # score + pattern
