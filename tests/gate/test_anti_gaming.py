"""Tests for anti-gaming measures."""

from shared.models import Dimension, DimensionScore, ScoreResult, ScoringMethod

from gate.patterns.anti_gaming import AntiGamingChecker, GamingFinding


def _make_score(
    composite: float = 0.8,
    dims: list[tuple[Dimension, float]] | None = None,
) -> ScoreResult:
    dim_scores = []
    if dims:
        for dim, val in dims:
            dim_scores.append(
                DimensionScore(dimension=dim, score=val, method=ScoringMethod.RULE_BASED)
            )
    return ScoreResult(user="test", composite_score=composite, dimension_scores=dim_scores)


# --- Comment Stuffing ---


class TestCommentStuffing:
    def setup_method(self):
        self.checker = AntiGamingChecker()

    def test_tautological_comments_detected(self):
        code = (
            "# set x to 5\n"
            "x = 5\n"
            "# set y to 10\n"
            "y = 10\n"
            "# return result\n"
            "return result\n"
            "# update count\n"
            "count += 1\n"
        )
        findings = self.checker.check(code)
        assert any(f.pattern == "comment_stuffing" for f in findings)

    def test_meaningful_comments_ok(self):
        code = (
            "# Calculate the weighted average using exponential decay\n"
            "avg = sum(w * v for w, v in zip(weights, values))\n"
            "# Clamp to valid range to prevent overflow in downstream calcs\n"
            "result = max(0.0, min(1.0, avg))\n"
            "# Edge case: empty input returns sentinel value per API contract\n"
            "if not values:\n"
            "    return -1.0\n"
        )
        findings = self.checker.check(code)
        assert not any(f.pattern == "comment_stuffing" for f in findings)

    def test_filler_comments_detected(self):
        code = (
            "# process data\n"
            "data = load()\n"
            "# handle the input\n"
            "result = process(data)\n"
            "# do the thing\n"
            "execute(result)\n"
        )
        findings = self.checker.check(code)
        assert any(f.pattern == "comment_stuffing" for f in findings)

    def test_few_comments_not_flagged(self):
        # Less than 3 comments shouldn't trigger
        code = "# set x to 5\nx = 5\ny = 10\n"
        findings = self.checker.check(code)
        assert not any(f.pattern == "comment_stuffing" for f in findings)

    def test_adjustment_targets_documentation(self):
        code = "# set x to 5\nx = 5\n# set y to 10\ny = 10\n# return result\nreturn result\n"
        findings = self.checker.check(code)
        stuffing = [f for f in findings if f.pattern == "comment_stuffing"]
        if stuffing:
            assert stuffing[0].dimension_affected == Dimension.DOCUMENTATION.value
            assert stuffing[0].score_adjustment < 0


# --- Try/Catch Wrapping ---


class TestTryCatchWrapping:
    def setup_method(self):
        self.checker = AntiGamingChecker()

    def test_swallowed_exceptions_detected(self):
        code = (
            "try:\n"
            "    do_something()\n"
            "except Exception:\n"
            "    pass\n"
            "try:\n"
            "    do_another()\n"
            "except:\n"
            "    pass\n"
        )
        findings = self.checker.check(code)
        assert any(f.pattern == "try_catch_wrapping" for f in findings)

    def test_handled_exceptions_ok(self):
        code = (
            "try:\n"
            "    result = risky_operation()\n"
            "except ValueError as e:\n"
            "    logger.error(f'Failed: {e}')\n"
            "    raise\n"
            "try:\n"
            "    data = load_file(path)\n"
            "except FileNotFoundError:\n"
            "    data = default_data()\n"
        )
        findings = self.checker.check(code)
        assert not any(f.pattern == "try_catch_wrapping" for f in findings)

    def test_single_swallow_not_flagged(self):
        # One swallowed exception is tolerable
        code = "try:\n    optional_cleanup()\nexcept Exception:\n    pass\n"
        findings = self.checker.check(code)
        assert not any(f.pattern == "try_catch_wrapping" for f in findings)

    def test_ellipsis_handler_detected(self):
        code = "try:\n    a()\nexcept:\n    ...\ntry:\n    b()\nexcept:\n    ...\n"
        findings = self.checker.check(code)
        assert any(f.pattern == "try_catch_wrapping" for f in findings)

    def test_adjustment_targets_correctness(self):
        code = "try:\n    a()\nexcept:\n    pass\ntry:\n    b()\nexcept:\n    pass\n"
        findings = self.checker.check(code)
        wrapping = [f for f in findings if f.pattern == "try_catch_wrapping"]
        assert wrapping[0].dimension_affected == Dimension.CORRECTNESS.value
        assert wrapping[0].score_adjustment < 0


# --- Test Stub Padding ---


class TestTestStubPadding:
    def setup_method(self):
        self.checker = AntiGamingChecker()

    def test_empty_tests_detected(self):
        test_code = (
            "def test_one():\n    pass\ndef test_two():\n    pass\ndef test_three():\n    pass\n"
        )
        findings = self.checker.check("", test_code=test_code)
        assert any(f.pattern == "test_stub_padding" for f in findings)

    def test_tests_with_assertions_ok(self):
        test_code = (
            "def test_add():\n"
            "    assert add(1, 2) == 3\n"
            "def test_subtract():\n"
            "    assert subtract(5, 3) == 2\n"
        )
        findings = self.checker.check("", test_code=test_code)
        assert not any(f.pattern == "test_stub_padding" for f in findings)

    def test_no_test_functions_ok(self):
        test_code = "def helper():\n    return 42\n"
        findings = self.checker.check("", test_code=test_code)
        assert not any(f.pattern == "test_stub_padding" for f in findings)

    def test_mixed_tests_threshold(self):
        # 1 empty out of 3 = 33%, below 50% threshold
        test_code = (
            "def test_a():\n"
            "    assert True\n"
            "def test_b():\n"
            "    assert 1 == 1\n"
            "def test_c():\n"
            "    pass\n"
        )
        findings = self.checker.check("", test_code=test_code)
        assert not any(f.pattern == "test_stub_padding" for f in findings)

    def test_call_without_assert_detected(self):
        test_code = (
            "def test_one():\n"
            "    result = do_something()\n"
            "def test_two():\n"
            "    process()\n"
            "def test_three():\n"
            "    run()\n"
        )
        findings = self.checker.check("", test_code=test_code)
        assert any(f.pattern == "test_stub_padding" for f in findings)

    def test_adjustment_targets_testability(self):
        test_code = "def test_a():\n    pass\ndef test_b():\n    pass\n"
        findings = self.checker.check("", test_code=test_code)
        padding = [f for f in findings if f.pattern == "test_stub_padding"]
        assert padding[0].dimension_affected == Dimension.TESTABILITY.value
        assert padding[0].score_adjustment < 0

    def test_empty_test_code_ok(self):
        findings = self.checker.check("x = 1", test_code="")
        assert not any(f.pattern == "test_stub_padding" for f in findings)


# --- Length Inflation ---


class TestLengthInflation:
    def setup_method(self):
        self.checker = AntiGamingChecker()

    def test_short_code_not_flagged(self):
        code = "x = 1\ny = 2\n"
        findings = self.checker.check(code)
        assert not any(f.pattern == "length_inflation" for f in findings)

    def test_concise_code_ok(self):
        # Normal density code
        lines = []
        for i in range(10):
            lines.append(f"def func_{i}(x):")
            lines.append(f"    return x + {i}")
            lines.append("")
        code = "\n".join(lines)
        findings = self.checker.check(code)
        assert not any(f.pattern == "length_inflation" for f in findings)

    def test_clean_code_no_findings(self):
        code = "def add(a, b):\n    return a + b\n"
        findings = self.checker.check(code)
        assert len(findings) == 0


# --- Score Adjustment ---


class TestScoreAdjustment:
    def setup_method(self):
        self.checker = AntiGamingChecker()

    def test_apply_no_findings(self):
        score = _make_score(
            composite=0.8,
            dims=[(Dimension.DOCUMENTATION, 0.8)],
        )
        result = self.checker.apply_adjustments(score, [])
        assert result.composite_score == 0.8

    def test_apply_single_adjustment(self):
        score = _make_score(
            composite=0.8,
            dims=[(Dimension.DOCUMENTATION, 0.8)],
        )
        finding = GamingFinding(
            pattern="comment_stuffing",
            description="test",
            score_adjustment=-0.15,
            dimension_affected=Dimension.DOCUMENTATION.value,
        )
        result = self.checker.apply_adjustments(score, [finding])
        doc_score = next(
            d.score for d in result.dimension_scores if d.dimension == Dimension.DOCUMENTATION
        )
        assert abs(doc_score - 0.65) < 0.01

    def test_adjustment_floors_at_zero(self):
        score = _make_score(
            composite=0.1,
            dims=[(Dimension.CORRECTNESS, 0.05)],
        )
        finding = GamingFinding(
            pattern="try_catch_wrapping",
            description="test",
            score_adjustment=-0.2,
            dimension_affected=Dimension.CORRECTNESS.value,
        )
        result = self.checker.apply_adjustments(score, [finding])
        corr_score = next(
            d.score for d in result.dimension_scores if d.dimension == Dimension.CORRECTNESS
        )
        assert corr_score == 0.0

    def test_multiple_adjustments(self):
        score = _make_score(
            composite=0.8,
            dims=[
                (Dimension.DOCUMENTATION, 0.8),
                (Dimension.CORRECTNESS, 0.7),
            ],
        )
        findings = [
            GamingFinding(
                pattern="comment_stuffing",
                description="test",
                score_adjustment=-0.15,
                dimension_affected=Dimension.DOCUMENTATION.value,
            ),
            GamingFinding(
                pattern="try_catch_wrapping",
                description="test",
                score_adjustment=-0.1,
                dimension_affected=Dimension.CORRECTNESS.value,
            ),
        ]
        result = self.checker.apply_adjustments(score, findings)
        assert result.composite_score < 0.8

    def test_original_not_modified(self):
        score = _make_score(
            composite=0.8,
            dims=[(Dimension.DOCUMENTATION, 0.8)],
        )
        finding = GamingFinding(
            pattern="comment_stuffing",
            description="test",
            score_adjustment=-0.15,
            dimension_affected=Dimension.DOCUMENTATION.value,
        )
        self.checker.apply_adjustments(score, [finding])
        # Original should be unchanged
        assert score.composite_score == 0.8
        assert score.dimension_scores[0].score == 0.8


# --- Integration ---


class TestIntegration:
    def test_check_and_apply(self):
        checker = AntiGamingChecker()
        code = "# set x to 5\nx = 5\n# set y to 10\ny = 10\n# return result\nreturn result\n"
        score = _make_score(
            composite=0.8,
            dims=[(Dimension.DOCUMENTATION, 0.9)],
        )
        findings = checker.check(code)
        if findings:
            adjusted = checker.apply_adjustments(score, findings)
            assert adjusted.composite_score <= score.composite_score

    def test_clean_code_no_adjustment(self):
        checker = AntiGamingChecker()
        code = "def add(a, b):\n    return a + b\n"
        findings = checker.check(code)
        assert len(findings) == 0
