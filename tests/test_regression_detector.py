"""Tests for auto-downgrade regression detection (issue #61)."""

from __future__ import annotations

from datetime import datetime, timedelta

from shared.config import RegressionConfig, RegistryConfig
from shared.models import (
    DemotionSuggestion,
    Dimension,
    DimensionScore,
    GraduationEvent,
    ScoreResult,
    ScoringMethod,
    ToolRegistryEntry,
    ToolTier,
)

from registry.graduation.regression import RegressionDetector
from registry.graduation.tiers import (
    VALID_DEMOTIONS,
    demotion_target,
    is_valid_demotion,
)


# --- Helpers ---


def _tool(
    slug: str = "test-tool",
    tier: ToolTier = ToolTier.T2,
    graduation_date: datetime | None = None,
) -> ToolRegistryEntry:
    """Create a test tool entry."""
    history = []
    if graduation_date is not None:
        prev_tier = {
            ToolTier.T1: ToolTier.T0,
            ToolTier.T2: ToolTier.T1,
            ToolTier.T3: ToolTier.T2,
        }.get(tier, ToolTier.T0)
        history.append(
            GraduationEvent(
                from_tier=prev_tier,
                to_tier=tier,
                date=graduation_date,
                reason="test promotion",
            )
        )
    return ToolRegistryEntry(
        name="Test Tool",
        slug=slug,
        tier=tier,
        users=["alice", "bob"],
        source_path="test.py",
        graduation_history=history,
    )


def _scores(
    n: int,
    composite: float = 0.8,
    security: float = 0.8,
    start: datetime | None = None,
) -> list[ScoreResult]:
    """Create a list of N score results with given composite and security scores."""
    if start is None:
        start = datetime(2025, 1, 1)
    results = []
    for i in range(n):
        results.append(
            ScoreResult(
                user="alice",
                composite_score=composite,
                timestamp=start + timedelta(hours=i),
                dimension_scores=[
                    DimensionScore(
                        dimension=Dimension.SECURITY,
                        score=security,
                        method=ScoringMethod.RULE_BASED,
                    ),
                    DimensionScore(
                        dimension=Dimension.CORRECTNESS,
                        score=composite,
                        method=ScoringMethod.RULE_BASED,
                    ),
                ],
            )
        )
    return results


# --- Demotion Tiers ---


class TestDemotionTiers:
    def test_valid_demotions(self):
        assert is_valid_demotion(ToolTier.T3, ToolTier.T2)
        assert is_valid_demotion(ToolTier.T2, ToolTier.T1)
        assert is_valid_demotion(ToolTier.T1, ToolTier.T0)

    def test_invalid_demotions(self):
        assert not is_valid_demotion(ToolTier.T0, ToolTier.T1)  # promotion, not demotion
        assert not is_valid_demotion(ToolTier.T3, ToolTier.T0)  # skip tiers

    def test_demotion_target(self):
        assert demotion_target(ToolTier.T3) == ToolTier.T2
        assert demotion_target(ToolTier.T2) == ToolTier.T1
        assert demotion_target(ToolTier.T1) == ToolTier.T0
        assert demotion_target(ToolTier.T0) is None

    def test_valid_demotions_list(self):
        assert len(VALID_DEMOTIONS) == 3


# --- RegressionDetector ---


class TestRegressionDetector:
    def test_no_regression_good_scores(self):
        detector = RegressionDetector()
        tool = _tool(tier=ToolTier.T2)
        scores = _scores(15, composite=0.85)
        assert detector.check(tool, scores) is None

    def test_t0_cannot_be_demoted(self):
        detector = RegressionDetector()
        tool = _tool(tier=ToolTier.T0)
        scores = _scores(15, composite=0.1)
        assert detector.check(tool, scores) is None

    def test_not_enough_scores(self):
        detector = RegressionDetector()
        tool = _tool(tier=ToolTier.T2)
        scores = _scores(5, composite=0.1)
        assert detector.check(tool, scores) is None

    def test_consecutive_low_composite_triggers(self):
        detector = RegressionDetector(RegressionConfig(consecutive_scores=5))
        tool = _tool(tier=ToolTier.T2)
        # T2 minimum is 0.7, all scores below that
        scores = _scores(5, composite=0.5)
        result = detector.check(tool, scores)
        assert result is not None
        assert result.tool_slug == "test-tool"
        assert result.current_tier == ToolTier.T2
        assert result.suggested_tier == ToolTier.T1
        assert "Composite score below" in result.reason
        assert result.evidence["consecutive_scores"] == 5

    def test_mixed_scores_no_trigger(self):
        """Not ALL consecutive scores are below threshold."""
        detector = RegressionDetector(RegressionConfig(consecutive_scores=5))
        tool = _tool(tier=ToolTier.T2)
        scores = _scores(4, composite=0.5) + _scores(1, composite=0.8)
        assert detector.check(tool, scores) is None

    def test_low_dimension_triggers(self):
        detector = RegressionDetector(RegressionConfig(consecutive_scores=3))
        tool = _tool(tier=ToolTier.T2)
        # Security dimension consistently below 0.3
        scores = _scores(3, composite=0.8, security=0.1)
        result = detector.check(tool, scores)
        assert result is not None
        assert "security" in result.reason
        assert result.evidence["dimension"] == "security"

    def test_dimension_not_all_below_no_trigger(self):
        detector = RegressionDetector(RegressionConfig(consecutive_scores=3))
        tool = _tool(tier=ToolTier.T2)
        # 2 below + 1 above = no trigger
        scores = _scores(2, composite=0.8, security=0.1) + _scores(1, composite=0.8, security=0.5)
        assert detector.check(tool, scores) is None

    def test_red_tier_count_triggers(self):
        detector = RegressionDetector(
            RegressionConfig(
                consecutive_scores=100,  # disable composite check
                rolling_window=10,
                red_tier_threshold=3,
            )
        )
        tool = _tool(tier=ToolTier.T2)
        # 7 good + 3 red (composite < 0.3)
        scores = _scores(7, composite=0.8) + _scores(3, composite=0.2)
        result = detector.check(tool, scores)
        assert result is not None
        assert "red-tier scores" in result.reason
        assert result.evidence["red_count"] == 3

    def test_red_tier_from_security(self):
        """Red tier count counts security < 0.3 even if composite is fine."""
        detector = RegressionDetector(
            RegressionConfig(
                consecutive_scores=100,
                rolling_window=10,
                red_tier_threshold=3,
            )
        )
        tool = _tool(tier=ToolTier.T2)
        scores = _scores(7, composite=0.8, security=0.8) + _scores(3, composite=0.8, security=0.1)
        result = detector.check(tool, scores)
        assert result is not None
        assert result.evidence["red_count"] == 3

    def test_red_tier_below_threshold_no_trigger(self):
        detector = RegressionDetector(
            RegressionConfig(
                consecutive_scores=100,
                rolling_window=10,
                red_tier_threshold=5,
            )
        )
        tool = _tool(tier=ToolTier.T2)
        scores = _scores(8, composite=0.8) + _scores(2, composite=0.2)
        assert detector.check(tool, scores) is None


# --- Grace Period ---


class TestGracePeriod:
    def test_within_grace_period_no_trigger(self):
        detector = RegressionDetector(RegressionConfig(consecutive_scores=3, grace_period_days=7))
        now = datetime(2025, 6, 10)
        tool = _tool(tier=ToolTier.T2, graduation_date=datetime(2025, 6, 5))
        scores = _scores(5, composite=0.3, start=datetime(2025, 6, 5))
        # Within 7-day grace period
        assert detector.check(tool, scores, now=now) is None

    def test_after_grace_period_triggers(self):
        detector = RegressionDetector(RegressionConfig(consecutive_scores=3, grace_period_days=7))
        now = datetime(2025, 6, 20)
        tool = _tool(tier=ToolTier.T2, graduation_date=datetime(2025, 6, 1))
        scores = _scores(5, composite=0.3, start=datetime(2025, 6, 10))
        # Past grace period
        result = detector.check(tool, scores, now=now)
        assert result is not None

    def test_no_graduation_history_no_grace(self):
        """Tools with no graduation history are not in grace period."""
        detector = RegressionDetector(RegressionConfig(consecutive_scores=3, grace_period_days=7))
        tool = _tool(tier=ToolTier.T2)
        scores = _scores(5, composite=0.3)
        result = detector.check(tool, scores)
        assert result is not None

    def test_exact_grace_period_boundary(self):
        """At exactly the grace period end, no longer in grace."""
        detector = RegressionDetector(RegressionConfig(consecutive_scores=3, grace_period_days=7))
        graduation_date = datetime(2025, 6, 1)
        now = graduation_date + timedelta(days=7)
        tool = _tool(tier=ToolTier.T2, graduation_date=graduation_date)
        scores = _scores(5, composite=0.3, start=datetime(2025, 6, 5))
        result = detector.check(tool, scores, now=now)
        assert result is not None


# --- DemotionSuggestion Model ---


class TestDemotionSuggestion:
    def test_create(self):
        suggestion = DemotionSuggestion(
            tool_slug="my-tool",
            current_tier=ToolTier.T2,
            suggested_tier=ToolTier.T1,
            reason="Quality regression",
        )
        assert suggestion.tool_slug == "my-tool"
        assert suggestion.current_tier == ToolTier.T2
        assert suggestion.suggested_tier == ToolTier.T1
        assert suggestion.evidence == {}

    def test_with_evidence(self):
        suggestion = DemotionSuggestion(
            tool_slug="my-tool",
            current_tier=ToolTier.T3,
            suggested_tier=ToolTier.T2,
            reason="Red flags",
            evidence={"red_count": 5},
        )
        assert suggestion.evidence["red_count"] == 5


# --- Tier-specific regression ---


class TestTierSpecificRegression:
    def test_t1_regression_to_t0(self):
        detector = RegressionDetector(RegressionConfig(consecutive_scores=3))
        tool = _tool(tier=ToolTier.T1)
        # T1 min is 0.5
        scores = _scores(3, composite=0.3)
        result = detector.check(tool, scores)
        assert result is not None
        assert result.suggested_tier == ToolTier.T0

    def test_t3_regression_to_t2(self):
        detector = RegressionDetector(RegressionConfig(consecutive_scores=3))
        tool = _tool(tier=ToolTier.T3)
        # T3 min is 0.8
        scores = _scores(3, composite=0.6)
        result = detector.check(tool, scores)
        assert result is not None
        assert result.suggested_tier == ToolTier.T2

    def test_t2_borderline_no_trigger(self):
        """Score at exactly the tier minimum should not trigger."""
        detector = RegressionDetector(RegressionConfig(consecutive_scores=3))
        tool = _tool(tier=ToolTier.T2)
        # T2 min is 0.7 — scores at exactly 0.7 should not trigger
        scores = _scores(3, composite=0.7)
        assert detector.check(tool, scores) is None


# --- Config ---


class TestRegressionConfig:
    def test_defaults(self):
        config = RegressionConfig()
        assert config.consecutive_scores == 10
        assert config.red_tier_threshold == 5
        assert config.rolling_window == 20
        assert config.grace_period_days == 7

    def test_custom_values(self):
        config = RegressionConfig(
            consecutive_scores=5,
            red_tier_threshold=3,
            rolling_window=15,
            grace_period_days=14,
        )
        assert config.consecutive_scores == 5
        assert config.grace_period_days == 14

    def test_registry_config_has_regression(self):
        config = RegistryConfig()
        assert config.regression.consecutive_scores == 10

    def test_composite_check_priority(self):
        """Composite check fires before dimension check."""
        detector = RegressionDetector(RegressionConfig(consecutive_scores=3))
        tool = _tool(tier=ToolTier.T2)
        # Both composite and security are bad
        scores = _scores(3, composite=0.3, security=0.1)
        result = detector.check(tool, scores)
        assert result is not None
        # Composite check fires first
        assert "Composite score" in result.reason
