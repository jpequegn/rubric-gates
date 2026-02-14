"""Tests for automatic graduation triggers."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from shared.config import AutoTriggers, RegistryConfig, T0ToT1Triggers
from shared.models import ScoreResult, ToolRegistryEntry, ToolTier

from registry.catalog.catalog import ToolCatalog
from registry.graduation.triggers import GraduationTriggerEngine


# --- Helpers ---


def _make_tool(
    name: str = "test-tool",
    tier: ToolTier = ToolTier.T0,
    users: list[str] | None = None,
    source_path: str = "/tools/test.py",
    metadata: dict | None = None,
) -> ToolRegistryEntry:
    return ToolRegistryEntry(
        name=name,
        tier=tier,
        description="A test tool",
        users=users or ["alice"],
        source_path=source_path,
        metadata=metadata or {},
    )


def _make_score(
    user: str = "alice",
    composite: float = 0.7,
    files: list[str] | None = None,
    timestamp: datetime | None = None,
    metadata: dict | None = None,
) -> ScoreResult:
    return ScoreResult(
        user=user,
        composite_score=composite,
        files_touched=files or ["/tools/test.py"],
        timestamp=timestamp or datetime.now(),
        metadata=metadata or {},
    )


def _make_engine(
    catalog: ToolCatalog,
    scores: list[ScoreResult] | None = None,
    config: RegistryConfig | None = None,
) -> GraduationTriggerEngine:
    storage = MagicMock()
    storage.query.return_value = scores or []
    if config is None:
        config = RegistryConfig()
    return GraduationTriggerEngine(catalog=catalog, storage=storage, config=config)


# --- T0 → T1 Triggers ---


class TestT0ToT1SecondUser:
    def test_two_users_triggers(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool A", users=["alice", "bob"]))

        engine = _make_engine(catalog)
        suggestion = engine.check_tool("tool-a")

        assert suggestion is not None
        assert suggestion.suggested_tier == ToolTier.T1
        assert "user" in suggestion.trigger_reason.lower()
        assert suggestion.evidence["user_count"] == 2

    def test_single_user_no_trigger(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool A", users=["alice"]))

        engine = _make_engine(catalog)
        suggestion = engine.check_tool("tool-a")
        assert suggestion is None

    def test_second_user_disabled(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool A", users=["alice", "bob"]))

        config = RegistryConfig(
            auto_triggers=AutoTriggers(
                t0_to_t1=T0ToT1Triggers(second_user=False),
            )
        )
        engine = _make_engine(catalog, config=config)
        suggestion = engine.check_tool("tool-a")
        assert suggestion is None


class TestT0ToT1ActiveDevelopment:
    def test_ten_scores_triggers(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool A"))

        scores = [_make_score() for _ in range(10)]
        engine = _make_engine(catalog, scores=scores)
        suggestion = engine.check_tool("tool-a")

        assert suggestion is not None
        assert suggestion.suggested_tier == ToolTier.T1
        assert "scoring events" in suggestion.trigger_reason.lower()

    def test_nine_scores_no_trigger(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool A"))

        scores = [_make_score() for _ in range(9)]
        engine = _make_engine(catalog, scores=scores)
        suggestion = engine.check_tool("tool-a")
        assert suggestion is None


class TestT0ToT1LinesOfCode:
    def test_exceeds_threshold(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool A", metadata={"lines_of_code": 600}))

        engine = _make_engine(catalog)
        suggestion = engine.check_tool("tool-a")

        assert suggestion is not None
        assert suggestion.suggested_tier == ToolTier.T1
        assert "lines" in suggestion.trigger_reason.lower()

    def test_below_threshold(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool A", metadata={"lines_of_code": 200}))

        engine = _make_engine(catalog)
        suggestion = engine.check_tool("tool-a")
        assert suggestion is None


class TestT0ToT1RisingTrend:
    def test_rising_scores_triggers(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool A"))

        now = datetime.now()
        scores = [
            _make_score(composite=0.5, timestamp=now - timedelta(hours=3)),
            _make_score(composite=0.6, timestamp=now - timedelta(hours=2)),
            _make_score(composite=0.7, timestamp=now - timedelta(hours=1)),
        ]
        engine = _make_engine(catalog, scores=scores)
        suggestion = engine.check_tool("tool-a")

        assert suggestion is not None
        assert "trend" in suggestion.trigger_reason.lower()

    def test_flat_scores_no_trigger(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool A"))

        now = datetime.now()
        scores = [
            _make_score(composite=0.7, timestamp=now - timedelta(hours=3)),
            _make_score(composite=0.7, timestamp=now - timedelta(hours=2)),
            _make_score(composite=0.7, timestamp=now - timedelta(hours=1)),
        ]
        engine = _make_engine(catalog, scores=scores)
        suggestion = engine.check_tool("tool-a")
        assert suggestion is None


# --- T1 → T2 Triggers ---


class TestT1ToT2DailyUsage:
    def test_multi_user_multi_day_triggers(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool A", tier=ToolTier.T1))

        now = datetime.now()
        scores = []
        for day in range(14):
            for user in ["alice", "bob", "charlie"]:
                scores.append(
                    _make_score(
                        user=user,
                        timestamp=now - timedelta(days=day),
                    )
                )
        engine = _make_engine(catalog, scores=scores)
        suggestion = engine.check_tool("tool-a")

        assert suggestion is not None
        assert suggestion.suggested_tier == ToolTier.T2
        assert "users" in suggestion.trigger_reason.lower()

    def test_too_few_users_no_trigger(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool A", tier=ToolTier.T1))

        now = datetime.now()
        scores = [_make_score(user="alice", timestamp=now - timedelta(days=d)) for d in range(14)]
        engine = _make_engine(catalog, scores=scores)
        suggestion = engine.check_tool("tool-a")
        assert suggestion is None

    def test_too_few_days_no_trigger(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool A", tier=ToolTier.T1))

        now = datetime.now()
        scores = []
        for day in range(5):
            for user in ["alice", "bob", "charlie"]:
                scores.append(_make_score(user=user, timestamp=now - timedelta(days=day)))
        engine = _make_engine(catalog, scores=scores)
        suggestion = engine.check_tool("tool-a")
        assert suggestion is None


class TestT1ToT2Metadata:
    def test_pii_triggers(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(
            _make_tool(name="Tool A", tier=ToolTier.T1, metadata={"handles_pii": True})
        )

        engine = _make_engine(catalog)
        suggestion = engine.check_tool("tool-a")

        assert suggestion is not None
        assert "pii" in suggestion.trigger_reason.lower()

    def test_external_apis_triggers(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(
            _make_tool(name="Tool A", tier=ToolTier.T1, metadata={"external_apis": True})
        )

        engine = _make_engine(catalog)
        suggestion = engine.check_tool("tool-a")

        assert suggestion is not None
        assert "api" in suggestion.trigger_reason.lower()

    def test_manual_nomination_triggers(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(
            _make_tool(
                name="Tool A",
                tier=ToolTier.T1,
                metadata={"nominated_for_t2": True, "nominated_by": "dave"},
            )
        )

        engine = _make_engine(catalog)
        suggestion = engine.check_tool("tool-a")

        assert suggestion is not None
        assert "nomination" in suggestion.trigger_reason.lower()


# --- T2 → T3 (Manual Only) ---


class TestT2ToT3:
    def test_no_auto_trigger(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool A", tier=ToolTier.T2))

        engine = _make_engine(catalog)
        suggestion = engine.check_tool("tool-a")
        assert suggestion is None


# --- Check All ---


class TestCheckAll:
    def test_scans_all_tools(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool A", users=["alice", "bob"]))
        catalog.register(_make_tool(name="Tool B", users=["alice"]))
        catalog.register(_make_tool(name="Tool C", users=["alice", "charlie"]))

        engine = _make_engine(catalog)
        suggestions = engine.check_triggers()

        # Tool A and Tool C have 2 users → should trigger
        assert len(suggestions) == 2
        slugs = {s.tool_slug for s in suggestions}
        assert "tool-a" in slugs
        assert "tool-c" in slugs

    def test_empty_catalog(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        engine = _make_engine(catalog)
        assert engine.check_triggers() == []


# --- Suggestion Fields ---


class TestSuggestionFields:
    def test_includes_graduation_result(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        catalog.register(_make_tool(name="Tool A", users=["alice", "bob"]))

        engine = _make_engine(catalog)
        suggestion = engine.check_tool("tool-a")

        assert suggestion is not None
        assert suggestion.graduation_result is not None
        assert suggestion.graduation_result.from_tier == ToolTier.T0
        assert suggestion.graduation_result.to_tier == ToolTier.T1

    def test_nonexistent_tool_returns_none(self, tmp_path):
        catalog = ToolCatalog(data_dir=tmp_path)
        engine = _make_engine(catalog)
        assert engine.check_tool("nope") is None


# --- Rising Trend Helper ---


class TestRisingTrend:
    def test_rising(self):
        now = datetime.now()
        scores = [
            _make_score(composite=0.5, timestamp=now - timedelta(hours=2)),
            _make_score(composite=0.6, timestamp=now - timedelta(hours=1)),
            _make_score(composite=0.7, timestamp=now),
        ]
        assert GraduationTriggerEngine._has_rising_trend(scores, window=3)

    def test_not_rising(self):
        now = datetime.now()
        scores = [
            _make_score(composite=0.7, timestamp=now - timedelta(hours=2)),
            _make_score(composite=0.6, timestamp=now - timedelta(hours=1)),
            _make_score(composite=0.5, timestamp=now),
        ]
        assert not GraduationTriggerEngine._has_rising_trend(scores, window=3)

    def test_too_few_scores(self):
        scores = [_make_score(composite=0.5), _make_score(composite=0.6)]
        assert not GraduationTriggerEngine._has_rising_trend(scores, window=3)
