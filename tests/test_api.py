"""Tests for REST API layer (issue #66).

Uses httpx.AsyncClient with ASGI transport — no server needed.
Patches scoring engine and catalog to avoid real scorer dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from api.app import APIConfig, RateLimiter, create_app
from api.schemas import (
    ErrorResponse,
    GateRequest,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
    ToolSummary,
)
from shared.models import (
    DimensionScore,
    GateResult,
    GateTier,
    ScoreResult,
    ScorecardSummary,
    ToolRegistryEntry,
    ToolTier,
)

# --- Fixtures ---

try:
    from httpx import ASGITransport, AsyncClient

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

pytestmark = pytest.mark.skipif(not HAS_HTTPX, reason="httpx not installed")


def _mock_score_result(composite: float = 0.85) -> ScoreResult:
    return ScoreResult(
        user="api",
        composite_score=composite,
        dimension_scores=[
            DimensionScore(
                dimension="correctness",
                score=0.9,
                method="rule_based",
            ),
            DimensionScore(
                dimension="security",
                score=0.8,
                method="rule_based",
            ),
        ],
    )


def _mock_gate_result() -> GateResult:
    return GateResult(
        tier=GateTier.GREEN,
        score_result=_mock_score_result(),
        blocked=False,
        advisory_messages=["All checks passed."],
    )


def _mock_tool() -> ToolRegistryEntry:
    return ToolRegistryEntry(
        name="Expense Categorizer",
        slug="expense-categorizer",
        description="Categorizes expenses automatically",
        tier=ToolTier.T1,
        created_by="alice",
        tech_owner="bob",
        users=["alice", "bob"],
        scorecard=ScorecardSummary(
            latest_composite=0.85,
            latest_scores={"correctness": 0.9, "security": 0.8},
            total_scores=5,
        ),
        tags=["finance"],
    )


@pytest.fixture
def app():
    """Create app with mocked dependencies."""
    config = APIConfig(catalog_dir="/tmp/test-catalog")
    test_app = create_app(config)

    mock_engine = MagicMock()
    mock_engine.score.return_value = _mock_score_result()

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate.return_value = _mock_gate_result()

    mock_catalog = MagicMock()
    mock_catalog.list.return_value = [_mock_tool()]
    mock_catalog.get.return_value = _mock_tool()
    mock_catalog.update.return_value = _mock_tool()

    test_app.state._engine = mock_engine
    test_app.state._catalog = mock_catalog
    test_app.state._evaluator = mock_evaluator

    return test_app


@pytest.fixture
def app_with_auth():
    """Create app with API key authentication enabled."""
    config = APIConfig(api_key="test-secret-key")
    test_app = create_app(config)

    mock_engine = MagicMock()
    mock_engine.score.return_value = _mock_score_result()
    test_app.state._engine = mock_engine

    return test_app


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def auth_client(app_with_auth):
    transport = ASGITransport(app=app_with_auth)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# --- Health ---


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, auth_client: AsyncClient):
        """Health endpoint should not require API key."""
        resp = await auth_client.get("/health")
        assert resp.status_code == 200


# --- Score ---


class TestScore:
    @pytest.mark.asyncio
    async def test_score_basic(self, client: AsyncClient):
        resp = await client.post(
            "/score",
            json={"code": "print('hello')", "filename": "hello.py"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["composite_score"] == 0.85
        assert "correctness" in data["dimensions"]
        assert "result" in data

    @pytest.mark.asyncio
    async def test_score_minimal(self, client: AsyncClient):
        resp = await client.post("/score", json={"code": "x = 1"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_score_empty_code_rejected(self, client: AsyncClient):
        resp = await client.post("/score", json={"code": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_score_missing_code_rejected(self, client: AsyncClient):
        resp = await client.post("/score", json={})
        assert resp.status_code == 422


# --- Score File ---


class TestScoreFile:
    @pytest.mark.asyncio
    async def test_score_file_upload(self, client: AsyncClient):
        resp = await client.post(
            "/score-file",
            files={"file": ("test.py", b"print('hello')", "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["composite_score"] == 0.85


# --- Gate ---


class TestGate:
    @pytest.mark.asyncio
    async def test_gate_basic(self, client: AsyncClient):
        resp = await client.post(
            "/gate",
            json={"code": "print('hello')", "filename": "hello.py"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "green"
        assert data["blocked"] is False
        assert "advisory_messages" in data

    @pytest.mark.asyncio
    async def test_gate_with_profile(self, client: AsyncClient):
        resp = await client.post(
            "/gate",
            json={
                "code": "x = 1",
                "filename": "x.py",
                "profile": "strict",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_gate_empty_code_rejected(self, client: AsyncClient):
        resp = await client.post("/gate", json={"code": ""})
        assert resp.status_code == 422


# --- Tools ---


class TestToolsList:
    @pytest.mark.asyncio
    async def test_list_tools(self, client: AsyncClient):
        resp = await client.get("/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["slug"] == "expense-categorizer"
        assert data[0]["tier"] == "T1"

    @pytest.mark.asyncio
    async def test_list_tools_empty(self, app, client: AsyncClient):
        app.state._catalog.list.return_value = []
        resp = await client.get("/tools")
        assert resp.status_code == 200
        assert resp.json() == []


class TestToolDetail:
    @pytest.mark.asyncio
    async def test_get_tool(self, client: AsyncClient):
        resp = await client.get("/tools/expense-categorizer")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Expense Categorizer"
        assert data["tier"] == "T1"
        assert data["scorecard"]["latest_composite"] == 0.85

    @pytest.mark.asyncio
    async def test_get_tool_not_found(self, app, client: AsyncClient):
        app.state._catalog.get.return_value = None
        resp = await client.get("/tools/nonexistent")
        assert resp.status_code == 404


class TestToolScore:
    @pytest.mark.asyncio
    async def test_score_tool(self, client: AsyncClient):
        resp = await client.post(
            "/tools/expense-categorizer/score",
            json={"code": "def categorize(): pass"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["slug"] == "expense-categorizer"
        assert "score" in data
        assert "updated_scorecard" in data

    @pytest.mark.asyncio
    async def test_score_tool_not_found(self, app, client: AsyncClient):
        app.state._catalog.get.return_value = None
        resp = await client.post(
            "/tools/nonexistent/score",
            json={"code": "x = 1"},
        )
        assert resp.status_code == 404


# --- Authentication ---


class TestAuthentication:
    @pytest.mark.asyncio
    async def test_no_key_returns_401(self, auth_client: AsyncClient):
        resp = await auth_client.post("/score", json={"code": "x = 1"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_key_returns_401(self, auth_client: AsyncClient):
        resp = await auth_client.post(
            "/score",
            json={"code": "x = 1"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_correct_key_allowed(self, auth_client: AsyncClient):
        resp = await auth_client.post(
            "/score",
            json={"code": "x = 1"},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_disabled_allows_all(self, client: AsyncClient):
        resp = await client.post("/score", json={"code": "x = 1"})
        assert resp.status_code == 200


# --- Rate Limiting ---


class TestRateLimiter:
    def test_allows_within_limit(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        assert limiter.check("client1") is True
        assert limiter.check("client1") is True
        assert limiter.check("client1") is True

    def test_blocks_over_limit(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.check("client1") is True
        assert limiter.check("client1") is True
        assert limiter.check("client1") is False

    def test_independent_clients(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.check("client1") is True
        assert limiter.check("client2") is True
        assert limiter.check("client1") is False

    def test_window_expiry(self):
        limiter = RateLimiter(max_requests=1, window_seconds=0)
        assert limiter.check("client1") is True
        # Window of 0 seconds means everything is expired immediately
        assert limiter.check("client1") is True


class TestRateLimitEndpoint:
    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self):
        config = APIConfig(rate_limit=2, rate_window=60)
        test_app = create_app(config)
        mock_engine = MagicMock()
        mock_engine.score.return_value = _mock_score_result()
        test_app.state._engine = mock_engine

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            await c.post("/score", json={"code": "x = 1"})
            await c.post("/score", json={"code": "x = 2"})
            resp = await c.post("/score", json={"code": "x = 3"})
            assert resp.status_code == 429


# --- Schema models ---


class TestSchemas:
    def test_score_request_validation(self):
        req = ScoreRequest(code="x = 1")
        assert req.filename == "untitled.py"

    def test_gate_request_validation(self):
        req = GateRequest(code="x = 1")
        assert req.profile == ""

    def test_health_response(self):
        resp = HealthResponse(status="ok", version="1.0.0")
        assert resp.status == "ok"

    def test_error_response(self):
        resp = ErrorResponse(detail="something went wrong")
        assert resp.detail == "something went wrong"

    def test_tool_summary(self):
        ts = ToolSummary(name="Test", slug="test", tier=ToolTier.T0)
        assert ts.latest_composite == 0.0

    def test_score_response_model(self):
        resp = ScoreResponse(
            composite_score=0.85,
            dimensions={"correctness": 0.9},
            result=_mock_score_result(),
        )
        assert resp.composite_score == 0.85


# --- OpenAPI docs ---


class TestOpenAPI:
    @pytest.mark.asyncio
    async def test_openapi_available(self, client: AsyncClient):
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["info"]["title"] == "rubric-gates API"
        assert "/score" in data["paths"]
        assert "/gate" in data["paths"]
        assert "/tools" in data["paths"]
        assert "/health" in data["paths"]

    @pytest.mark.asyncio
    async def test_docs_available(self, client: AsyncClient):
        resp = await client.get("/docs")
        assert resp.status_code == 200


# --- CORS ---


class TestCORS:
    @pytest.mark.asyncio
    async def test_cors_headers_present(self, client: AsyncClient):
        resp = await client.options(
            "/score",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" in resp.headers


# --- App factory ---


class TestAppFactory:
    def test_create_app_defaults(self):
        app = create_app()
        assert app.title == "rubric-gates API"

    def test_create_app_custom_config(self):
        config = APIConfig(api_key="secret", rate_limit=10)
        app = create_app(config)
        assert app.state.config.api_key == "secret"
        assert app.state.config.rate_limit == 10
