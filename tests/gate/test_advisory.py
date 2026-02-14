"""Tests for the advisory message system."""

from shared.models import (
    Dimension,
    DimensionScore,
    GateResult,
    GateTier,
    PatternFinding,
    ScoreResult,
    ScoringMethod,
)

from gate.tiers.advisory import AdvisoryConfig, AdvisoryMessageGenerator, AdvisoryTemplate


def _make_score(
    composite: float = 0.6,
    dims: list[tuple[Dimension, float]] | None = None,
) -> ScoreResult:
    dim_scores = []
    if dims:
        for dim, val in dims:
            dim_scores.append(
                DimensionScore(dimension=dim, score=val, method=ScoringMethod.RULE_BASED)
            )
    return ScoreResult(user="test", composite_score=composite, dimension_scores=dim_scores)


def _make_gate_result(
    tier: GateTier = GateTier.YELLOW,
    composite: float = 0.6,
    dims: list[tuple[Dimension, float]] | None = None,
    findings: list[PatternFinding] | None = None,
) -> GateResult:
    return GateResult(
        tier=tier,
        score_result=_make_score(composite=composite, dims=dims),
        pattern_findings=findings or [],
    )


def _make_finding(
    pattern: str = "test_pattern",
    severity: str = "high",
    line_number: int = 10,
    description: str = "test issue",
    remediation: str = "fix it",
) -> PatternFinding:
    return PatternFinding(
        pattern=pattern,
        severity=severity,
        line_number=line_number,
        line_content="x = bad()",
        description=description,
        remediation=remediation,
    )


# --- Green tier: no messages ---


class TestGreenTier:
    def test_green_returns_empty(self):
        gen = AdvisoryMessageGenerator()
        result = _make_gate_result(tier=GateTier.GREEN, composite=0.9)
        messages = gen.generate(result)
        assert messages == []


# --- Red tier: no advisories (uses blocking messages) ---


class TestRedTier:
    def test_red_returns_empty(self):
        gen = AdvisoryMessageGenerator()
        result = _make_gate_result(tier=GateTier.RED, composite=0.2)
        messages = gen.generate(result)
        assert messages == []


# --- Composite score advisories ---


class TestCompositeAdvisories:
    def test_below_green(self):
        gen = AdvisoryMessageGenerator()
        result = _make_gate_result(composite=0.6)
        messages = gen.generate(result)
        assert any("0.60" in msg for msg in messages)
        assert any("green" in msg.lower() for msg in messages)

    def test_below_yellow(self):
        gen = AdvisoryMessageGenerator()
        result = _make_gate_result(composite=0.4)
        messages = gen.generate(result)
        assert any("0.40" in msg for msg in messages)
        assert any("significant" in msg.lower() for msg in messages)

    def test_friendly_tone(self):
        gen = AdvisoryMessageGenerator()
        result = _make_gate_result(composite=0.6)
        messages = gen.generate(result)
        # Should not contain aggressive language
        for msg in messages:
            assert "FAIL" not in msg
            assert "REJECT" not in msg
            assert "BLOCKED" not in msg


# --- Pattern-based advisories ---


class TestPatternAdvisories:
    def test_high_severity_finding_generates_message(self):
        gen = AdvisoryMessageGenerator()
        finding = _make_finding(
            pattern="unsafe_file_ops",
            severity="high",
            line_number=42,
            description="open() with dynamically constructed path",
            remediation="Validate the path before opening.",
        )
        result = _make_gate_result(composite=0.8, findings=[finding])
        messages = gen.generate(result)
        assert any("42" in msg for msg in messages)

    def test_medium_severity_finding(self):
        gen = AdvisoryMessageGenerator()
        finding = _make_finding(
            pattern="unvetted_dependencies",
            severity="medium",
            line_number=5,
            description="Unpinned package install: requests",
            remediation="Pin the version.",
        )
        result = _make_gate_result(composite=0.8, findings=[finding])
        messages = gen.generate(result)
        assert any("5" in msg for msg in messages)

    def test_critical_findings_skipped(self):
        gen = AdvisoryMessageGenerator()
        finding = _make_finding(severity="critical", line_number=1)
        result = _make_gate_result(composite=0.8, findings=[finding])
        messages = gen.generate(result)
        # Critical findings should NOT generate advisories
        assert not any("line 1" in msg.lower() for msg in messages)

    def test_fallback_for_unknown_pattern(self):
        gen = AdvisoryMessageGenerator()
        finding = _make_finding(
            pattern="unknown_pattern",
            severity="high",
            line_number=99,
            description="Something unusual",
            remediation="Review the code.",
        )
        result = _make_gate_result(composite=0.8, findings=[finding])
        messages = gen.generate(result)
        assert any("99" in msg for msg in messages)
        assert any("Something unusual" in msg for msg in messages)

    def test_connection_string_template(self):
        gen = AdvisoryMessageGenerator()
        finding = _make_finding(
            pattern="hardcoded_credentials",
            severity="high",
            line_number=15,
        )
        result = _make_gate_result(composite=0.8, findings=[finding])
        messages = gen.generate(result)
        assert any("15" in msg for msg in messages)
        assert any("credentials" in msg.lower() or "environment" in msg.lower() for msg in messages)


# --- Dimension-based advisories ---


class TestDimensionAdvisories:
    def test_low_maintainability(self):
        gen = AdvisoryMessageGenerator()
        result = _make_gate_result(
            composite=0.8,
            dims=[(Dimension.MAINTAINABILITY, 0.2)],
        )
        messages = gen.generate(result)
        assert any("maintainability" in msg.lower() for msg in messages)
        assert any("0.20" in msg for msg in messages)

    def test_low_documentation(self):
        gen = AdvisoryMessageGenerator()
        result = _make_gate_result(
            composite=0.8,
            dims=[(Dimension.DOCUMENTATION, 0.1)],
        )
        messages = gen.generate(result)
        assert any("documentation" in msg.lower() for msg in messages)

    def test_low_testability(self):
        gen = AdvisoryMessageGenerator()
        result = _make_gate_result(
            composite=0.8,
            dims=[(Dimension.TESTABILITY, 0.2)],
        )
        messages = gen.generate(result)
        assert any("testability" in msg.lower() or "test" in msg.lower() for msg in messages)

    def test_above_threshold_no_advisory(self):
        gen = AdvisoryMessageGenerator()
        result = _make_gate_result(
            composite=0.8,
            dims=[(Dimension.MAINTAINABILITY, 0.5)],
        )
        messages = gen.generate(result)
        assert not any("maintainability" in msg.lower() for msg in messages)

    def test_multiple_low_dimensions(self):
        gen = AdvisoryMessageGenerator()
        result = _make_gate_result(
            composite=0.6,
            dims=[
                (Dimension.MAINTAINABILITY, 0.2),
                (Dimension.DOCUMENTATION, 0.1),
                (Dimension.TESTABILITY, 0.15),
            ],
        )
        messages = gen.generate(result)
        assert len(messages) >= 3  # composite + at least 2 dimensions


# --- Configuration ---


class TestAdvisoryConfig:
    def test_max_messages(self):
        config = AdvisoryConfig(max_messages=2)
        gen = AdvisoryMessageGenerator(config=config)
        result = _make_gate_result(
            composite=0.4,
            dims=[
                (Dimension.MAINTAINABILITY, 0.1),
                (Dimension.DOCUMENTATION, 0.1),
                (Dimension.TESTABILITY, 0.1),
            ],
        )
        messages = gen.generate(result)
        assert len(messages) <= 2

    def test_custom_dimension_thresholds(self):
        config = AdvisoryConfig(dimension_thresholds={Dimension.DOCUMENTATION.value: 0.8})
        gen = AdvisoryMessageGenerator(config=config)
        result = _make_gate_result(
            composite=0.8,
            dims=[(Dimension.DOCUMENTATION, 0.7)],
        )
        messages = gen.generate(result)
        assert any("documentation" in msg.lower() for msg in messages)

    def test_custom_template(self):
        custom = AdvisoryTemplate(
            category="custom",
            pattern_name="custom_check",
            template="Custom issue on line {line_number}: {description}.",
        )
        config = AdvisoryConfig(custom_templates=[custom])
        gen = AdvisoryMessageGenerator(config=config)
        finding = _make_finding(
            pattern="custom_check",
            severity="high",
            line_number=7,
            description="custom problem",
        )
        result = _make_gate_result(composite=0.8, findings=[finding])
        messages = gen.generate(result)
        assert any("Custom issue on line 7" in msg for msg in messages)

    def test_custom_composite_thresholds(self):
        config = AdvisoryConfig(green_composite=0.9, yellow_composite=0.6)
        gen = AdvisoryMessageGenerator(config=config)
        result = _make_gate_result(composite=0.8)
        messages = gen.generate(result)
        assert any("0.80" in msg for msg in messages)
        assert any("0.90" in msg or "green" in msg.lower() for msg in messages)


# --- Deduplication ---


class TestDeduplication:
    def test_no_duplicate_messages(self):
        gen = AdvisoryMessageGenerator()
        result = _make_gate_result(composite=0.6)
        messages = gen.generate(result)
        assert len(messages) == len(set(messages))


# --- Message quality ---


class TestMessageQuality:
    def test_messages_are_strings(self):
        gen = AdvisoryMessageGenerator()
        result = _make_gate_result(
            composite=0.5,
            dims=[(Dimension.SECURITY, 0.2)],
        )
        messages = gen.generate(result)
        assert all(isinstance(msg, str) for msg in messages)
        assert all(len(msg) > 10 for msg in messages)

    def test_messages_end_with_period(self):
        gen = AdvisoryMessageGenerator()
        result = _make_gate_result(
            composite=0.5,
            dims=[(Dimension.MAINTAINABILITY, 0.1)],
        )
        messages = gen.generate(result)
        for msg in messages:
            assert msg.rstrip().endswith(".")

    def test_actionable_language(self):
        gen = AdvisoryMessageGenerator()
        result = _make_gate_result(
            composite=0.6,
            dims=[(Dimension.DOCUMENTATION, 0.1)],
        )
        messages = gen.generate(result)
        # At least one message should contain actionable words
        actionable_words = ["consider", "adding", "help", "improve", "fix", "review", "break"]
        has_actionable = any(
            any(word in msg.lower() for word in actionable_words) for msg in messages
        )
        assert has_actionable
