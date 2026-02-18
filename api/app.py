"""FastAPI application for rubric-gates REST API.

Exposes scoring, gate evaluation, and tool registry endpoints.
Supports optional API key authentication and rate limiting.

Usage:
    uvicorn api.app:create_app --factory
"""

from __future__ import annotations

import importlib.metadata
import time
from collections import defaultdict
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ErrorResponse,
    GateRequest,
    GateResponse,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
    ToolDetailResponse,
    ToolScoreRequest,
    ToolScoreResponse,
    ToolSummary,
)


# --- Configuration ---


class APIConfig:
    """API configuration with sensible defaults.

    Attributes:
        api_key: Optional API key for authentication. Empty string disables auth.
        rate_limit: Max requests per window per client.
        rate_window: Rate limit window in seconds.
        cors_origins: Allowed CORS origins.
        catalog_dir: Path to tool catalog data directory.
    """

    def __init__(
        self,
        api_key: str = "",
        rate_limit: int = 60,
        rate_window: int = 60,
        cors_origins: list[str] | None = None,
        catalog_dir: str = "",
    ) -> None:
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.rate_window = rate_window
        self.cors_origins = cors_origins or ["*"]
        self.catalog_dir = catalog_dir


# --- Rate limiter ---


class RateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, client_id: str) -> bool:
        """Return True if the request is allowed."""
        now = time.time()
        cutoff = now - self.window
        # Prune old entries
        self._requests[client_id] = [t for t in self._requests[client_id] if t > cutoff]
        if len(self._requests[client_id]) >= self.max_requests:
            return False
        self._requests[client_id].append(now)
        return True


# --- App factory ---


def create_app(config: APIConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: API configuration. Uses defaults if not provided.

    Returns:
        Configured FastAPI app.
    """
    if config is None:
        config = APIConfig()

    try:
        version = importlib.metadata.version("rubric-gates")
    except importlib.metadata.PackageNotFoundError:
        version = "0.1.0"

    app = FastAPI(
        title="rubric-gates API",
        description="REST API for rubric-based code quality scoring and gate evaluation.",
        version=version,
        responses={
            401: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
        },
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State
    app.state.config = config
    app.state.rate_limiter = RateLimiter(config.rate_limit, config.rate_window)

    # --- Dependencies ---

    async def verify_api_key(
        request: Request,
        x_api_key: str | None = Header(default=None),
    ) -> None:
        """Verify API key if authentication is configured."""
        cfg: APIConfig = request.app.state.config
        if not cfg.api_key:
            return  # Auth disabled
        if x_api_key != cfg.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    async def check_rate_limit(request: Request) -> None:
        """Enforce per-client rate limiting."""
        limiter: RateLimiter = request.app.state.rate_limiter
        client_ip = request.client.host if request.client else "unknown"
        if not limiter.check(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    # --- Lazy loaders for scoring/gate/catalog ---

    def _get_engine() -> Any:
        from scorecard.engine import RubricEngine

        if not hasattr(app.state, "_engine"):
            app.state._engine = RubricEngine()
        return app.state._engine

    def _get_evaluator(profile: str = "") -> Any:
        if hasattr(app.state, "_evaluator"):
            return app.state._evaluator
        from gate.tiers.evaluator import TierEvaluator

        return TierEvaluator(profile=profile)

    def _get_catalog() -> Any:
        if hasattr(app.state, "_catalog"):
            return app.state._catalog
        from registry.catalog.catalog import ToolCatalog

        catalog = ToolCatalog(data_dir=config.catalog_dir) if config.catalog_dir else ToolCatalog()
        app.state._catalog = catalog
        return catalog

    # --- Routes ---

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="ok", version=version)

    @app.post(
        "/score",
        response_model=ScoreResponse,
        dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    )
    async def score(req: ScoreRequest) -> ScoreResponse:
        """Score source code across all quality dimensions."""
        engine = _get_engine()
        result = engine.score(
            code=req.code,
            filename=req.filename,
            user=req.user or "api",
            skill_used=req.skill_used,
        )
        dimensions = {ds.dimension.value: ds.score for ds in result.dimension_scores}
        return ScoreResponse(
            composite_score=result.composite_score,
            dimensions=dimensions,
            result=result,
        )

    @app.post(
        "/score-file",
        response_model=ScoreResponse,
        dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    )
    async def score_file(file: UploadFile) -> ScoreResponse:
        """Score an uploaded file."""
        content = await file.read()
        code = content.decode("utf-8")
        filename = file.filename or "uploaded.py"
        engine = _get_engine()
        result = engine.score(code=code, filename=filename, user="api")
        dimensions = {ds.dimension.value: ds.score for ds in result.dimension_scores}
        return ScoreResponse(
            composite_score=result.composite_score,
            dimensions=dimensions,
            result=result,
        )

    @app.post(
        "/gate",
        response_model=GateResponse,
        dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    )
    async def gate(req: GateRequest) -> GateResponse:
        """Evaluate code through the quality gate."""
        engine = _get_engine()
        score_result = engine.score(
            code=req.code,
            filename=req.filename,
            user=req.user or "api",
        )
        evaluator = _get_evaluator(profile=req.profile)
        gate_result = evaluator.evaluate(score_result, req.code, req.filename)
        return GateResponse(
            tier=gate_result.tier,
            blocked=gate_result.blocked,
            findings_count=len(gate_result.pattern_findings),
            advisory_messages=gate_result.advisory_messages,
            result=gate_result,
        )

    @app.get(
        "/tools",
        response_model=list[ToolSummary],
        dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    )
    async def list_tools() -> list[ToolSummary]:
        """List all registered tools with tier info."""
        catalog = _get_catalog()
        tools = catalog.list()
        return [
            ToolSummary(
                name=t.name,
                slug=t.slug,
                tier=t.tier,
                latest_composite=t.scorecard.latest_composite,
                tags=t.tags,
            )
            for t in tools
        ]

    @app.get(
        "/tools/{slug}",
        response_model=ToolDetailResponse,
        dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    )
    async def get_tool(slug: str) -> ToolDetailResponse:
        """Get detailed info for a specific tool."""
        catalog = _get_catalog()
        tool = catalog.get(slug)
        if tool is None:
            raise HTTPException(status_code=404, detail=f"Tool '{slug}' not found.")
        return ToolDetailResponse(
            name=tool.name,
            slug=tool.slug,
            description=tool.description,
            tier=tool.tier,
            created_by=tool.created_by,
            tech_owner=tool.tech_owner,
            users=tool.users,
            scorecard=tool.scorecard,
            tags=tool.tags,
            metadata=tool.metadata,
        )

    @app.post(
        "/tools/{slug}/score",
        response_model=ToolScoreResponse,
        dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    )
    async def score_tool(slug: str, req: ToolScoreRequest) -> ToolScoreResponse:
        """Score code and update a tool's scorecard."""
        catalog = _get_catalog()
        tool = catalog.get(slug)
        if tool is None:
            raise HTTPException(status_code=404, detail=f"Tool '{slug}' not found.")

        engine = _get_engine()
        result = engine.score(
            code=req.code,
            filename=req.filename,
            user=req.user or "api",
        )

        # Update scorecard summary
        dim_scores = {ds.dimension.value: ds.score for ds in result.dimension_scores}
        updated_scorecard = tool.scorecard.model_copy(
            update={
                "latest_composite": result.composite_score,
                "latest_scores": dim_scores,
                "total_scores": tool.scorecard.total_scores + 1,
            }
        )
        catalog.update(slug, {"scorecard": updated_scorecard.model_dump(mode="json")})

        return ToolScoreResponse(
            slug=slug,
            score=result,
            updated_scorecard=updated_scorecard,
        )

    return app
