"""FastAPI application.

Start (development, local only):

    cd backend
    .venv/bin/uvicorn app.api.main:create_app --factory --host 127.0.0.1 --port 8000

There is no authentication: bind to 127.0.0.1 only.

Safety of responses and logs:
* Validation errors return field locations and messages only - never the
  submitted values (which could contain secrets).
* Unexpected errors return a generic message; details go to the server log.
* Stored data is redacted before it is written, so API responses read from the
  database never contain the API key.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.schemas import (
    ActiveRuleRef,
    CacheCleared,
    DatasetOut,
    EvaluationCreated,
    EvaluationOut,
    EvaluationRunOut,
    Health,
    LLMCallOut,
    MiningOut,
    RuleDetailOut,
    RuleEventOut,
    RuleOut,
    RunCreated,
    RunDetailsOut,
    RunOut,
    ToolCallOut,
    ToolsOut,
    TraceEventOut,
    TraceOut,
    ValidationCreated,
    ValidationOut,
)
from app.config import Settings, load_settings
from app.evaluation.dataset import load_dataset
from app.evaluation.metrics import load_pricing
from app.evaluation.runner import EvaluationConfig
from app.learning.miner import not_eligible_reason
from app.learning.models import ValidationConfig
from app.llm.base import LLMClient
from app.llm.errors import LLMConfigurationError
from app.llm.gemini import GeminiClient
from app.services.evaluations import EvaluationService
from app.services.learning import LearnedRules, LearningService
from app.services.runs import RequestRejected, RunService
from app.services.schemas import (
    SESSION_ID_PATTERN,
    ApproveRequest,
    DisableRequest,
    MineRequest,
    RunRequest,
    ShadowRequest,
)
from app.storage.db import Database
from app.storage.evaluations import EvaluationNotFoundError, EvaluationRepository
from app.storage.learning import RULE_STATUSES, LearningRepository, RuleNotFoundError
from app.storage.redaction import Redactor
from app.storage.repository import RunNotFoundError, RunRepository, SqlTraceRecorder
from app.telemetry.clock import Clock, SystemClock

logger = logging.getLogger(__name__)
HEX_ID = r"^[0-9a-f]{32}$"


def _hex_id():
    """A fresh Path() per route: FastAPI binds a Path object to one parameter name."""
    return Path(pattern=HEX_ID)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(
    settings: Settings | None = None,
    *,
    gemini_factory: Callable[[], LLMClient] | None = None,
    clock: Clock | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    clock = clock or SystemClock()
    gemini_factory = gemini_factory or (lambda: GeminiClient.from_settings(settings))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db = Database(settings.database_url)
        pricing = load_pricing(settings.pricing_file)
        db.migrate()
        redactor = Redactor(settings.secret_values())
        repo = RunRepository(db, redactor)
        eval_repo = EvaluationRepository(db, redactor)
        now = clock.utcnow().isoformat()
        interrupted = repo.mark_interrupted_runs(now) + eval_repo.mark_interrupted_all(now)
        if interrupted:
            logger.warning(
                "%d unfinished run(s)/evaluation(s) from a previous process marked interrupted",
                interrupted,
            )
        recorder = SqlTraceRecorder(repo)
        # Learned rules: loaded from the database (and re-checked) before any
        # run can start, so a restart keeps exactly the approved rules.
        rules = LearnedRules()
        learning = LearningService(
            settings=settings,
            repo=LearningRepository(db),
            runs=repo,
            recorder=recorder,
            gemini_factory=gemini_factory,
            rules=rules,
            pricing=pricing,
            clock=clock,
            redactor=redactor,
        )
        learning.start()
        service = RunService(
            settings=settings,
            repository=repo,
            recorder=recorder,
            gemini_factory=gemini_factory,
            clock=clock,
            policy=rules.policy,
        )
        evaluations = EvaluationService(
            settings=settings,
            evaluations=eval_repo,
            runs=repo,
            recorder=recorder,
            gemini_factory=gemini_factory,
            pricing=pricing,
            clock=clock,
            active_rules=rules.active_only,
        )
        app.state.db, app.state.repo, app.state.service = db, repo, service
        app.state.eval_repo, app.state.evaluations = eval_repo, evaluations
        app.state.pricing = pricing
        app.state.learning = learning
        try:
            yield
        finally:
            await learning.shutdown()
            await evaluations.shutdown()
            await service.shutdown()
            db.dispose()

    app = FastAPI(
        title="AgentFlow Optimizer API",
        version="0.7.0",
        lifespan=lifespan,
    )
    # Browsers may call the API only from the configured local origins, without
    # cookies/credentials, and only with the methods the API uses.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    # ---------------------------------------------------------------- errors
    @app.exception_handler(RequestValidationError)
    async def _validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {
                "loc": [str(p) for p in e.get("loc", ())],
                "msg": e.get("msg", ""),
                "type": e.get("type", ""),
            }
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "invalid request",
                    "details": details,
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(RunNotFoundError)
    async def _not_found(_request: Request, _exc: RunNotFoundError) -> JSONResponse:
        return _error(404, "run_not_found", "no run with this id")

    @app.exception_handler(EvaluationNotFoundError)
    async def _eval_not_found(_request: Request, _exc: EvaluationNotFoundError) -> JSONResponse:
        return _error(404, "evaluation_not_found", "no evaluation with this id")

    @app.exception_handler(RuleNotFoundError)
    async def _rule_not_found(_request: Request, _exc: RuleNotFoundError) -> JSONResponse:
        return _error(404, "not_found", "no such learned rule, mining or validation")

    @app.exception_handler(RequestRejected)
    async def _rejected(_request: Request, exc: RequestRejected) -> JSONResponse:
        return _error(exc.status_code, exc.code, exc.message)

    @app.exception_handler(LLMConfigurationError)
    async def _llm_config(_request: Request, exc: LLMConfigurationError) -> JSONResponse:
        return _error(503, exc.code, exc.message)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, _exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return _error(500, "internal_error", "an unexpected error occurred")

    # ---------------------------------------------------------------- routes
    def service(request: Request) -> RunService:
        return request.app.state.service

    def repo(request: Request) -> RunRepository:
        return request.app.state.repo

    def learning(request: Request) -> LearningService:
        return request.app.state.learning

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        try:
            with request.app.state.db.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            database = "ok"
        except Exception:
            logger.exception("database health check failed")
            database = "unavailable"
        return Health(
            status="ok" if database == "ok" else "degraded",
            database=database,
            gemini_configured=settings.has_gemini_key,
            gemini_model=settings.gemini_model if settings.has_gemini_key else None,
            pricing_configured=request.app.state.pricing is not None,
            active_runs=len(service(request).active_run_ids),
            active_evaluations=len(request.app.state.evaluations.active_ids),
            active_validations=len(learning(request).active_validation_ids),
            active_learned_rules=len(learning(request).rules.policy().active),
        )

    @app.get("/tools", response_model=ToolsOut)
    def list_tools(request: Request) -> dict:
        ls = learning(request)
        current = ls.current_settings()
        rows = {r.id: r for r in ls.rules.active_rows()}
        tools = []
        for tool in service(request).available_tools():
            spec = ls.spec(tool["name"])
            now = current.get(tool["name"], {})
            refs = [
                ActiveRuleRef(rule_id=rid, kind=rows[rid].kind, setting=rows[rid].proposed_setting)
                for rid in (now.get("ttl_rule"), now.get("scope_rule"))
                if rid and rid in rows
            ]
            tools.append(
                {
                    **tool,
                    "max_ttl_seconds": spec.max_ttl_seconds if spec else None,
                    "effective_ttl_seconds": now.get("ttl_seconds", tool["ttl_seconds"]),
                    "effective_cache_scope": now.get("cache_scope", tool["cache_scope"]),
                    "learned_rules": refs,
                    "learning_eligible": spec is not None and not_eligible_reason(spec) is None,
                }
            )
        return {"tools": tools}

    @app.post("/runs", status_code=202, response_model=RunCreated)
    async def create_run(body: RunRequest, request: Request) -> RunCreated:
        run_id = service(request).submit(body)
        return RunCreated(
            run_id=run_id,
            status="queued",
            links={
                "self": f"/runs/{run_id}",
                "details": f"/runs/{run_id}/details",
                "trace": f"/runs/{run_id}/trace",
            },
        )

    @app.get("/runs", response_model=list[RunOut])
    def list_runs(
        request: Request,
        session_id: str | None = Query(default=None, pattern=SESSION_ID_PATTERN),
        status: str | None = Query(default=None, max_length=30),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> list[RunOut]:
        rows = repo(request).list_runs(session_id=session_id, status=status, limit=limit)
        return [RunOut.from_row(r, request.app.state.pricing) for r in rows]

    @app.get("/runs/{run_id}", response_model=RunOut)
    def get_run(request: Request, run_id: str = _hex_id()) -> RunOut:
        return RunOut.from_row(repo(request).get(run_id), request.app.state.pricing)

    @app.get("/runs/{run_id}/details", response_model=RunDetailsOut)
    def get_run_details(request: Request, run_id: str = _hex_id()) -> RunDetailsOut:
        r = repo(request)
        return RunDetailsOut(
            run=RunOut.from_row(r.get(run_id), request.app.state.pricing),
            llm_calls=[LLMCallOut.from_row(c) for c in r.llm_calls(run_id)],
            tool_calls=[ToolCallOut.from_row(c) for c in r.tool_calls(run_id)],
        )

    @app.get("/runs/{run_id}/trace", response_model=TraceOut)
    def get_trace(
        request: Request,
        run_id: str = _hex_id(),
        after_id: int = Query(default=0, ge=0),
        limit: int = Query(default=500, ge=1, le=2000),
    ) -> TraceOut:
        events = repo(request).trace(run_id, after_id=after_id, limit=limit)
        return TraceOut(
            run_id=run_id,
            events=[TraceEventOut.from_row(e) for e in events],
            next_after_id=events[-1].id if len(events) == limit else None,
        )

    @app.delete("/sessions/{session_id}/cache", response_model=CacheCleared)
    def clear_cache(
        request: Request, session_id: str = Path(pattern=SESSION_ID_PATTERN)
    ) -> CacheCleared:
        cleared = service(request).clear_session_cache(session_id)
        return CacheCleared(session_id=session_id, cleared_entries=cleared)

    # ---------------------------------------------------------------- evaluations
    @app.get("/datasets/{name}/{version}", response_model=DatasetOut)
    def get_dataset(
        name: str = Path(pattern=r"^[a-z][a-z0-9_]{0,40}$"),
        version: str = Path(pattern=r"^[0-9a-z]{1,10}$"),
    ) -> dict:
        """Task list WITHOUT expected answers or scripts (those stay server-side)."""
        try:
            dataset = load_dataset(name, version)
        except FileNotFoundError:
            return _error(404, "unknown_dataset", "no such dataset")
        return {
            "name": dataset.name,
            "version": dataset.version,
            "checksum": dataset.checksum,
            "description": dataset.description,
            "tasks": [
                {
                    "id": t.id,
                    "category": t.category,
                    "split": t.split,
                    "task": t.task,
                    "tools": t.tools,
                    "allow_side_effects": t.allow_side_effects,
                    "scripted_only": t.scripted_only,
                }
                for t in dataset.tasks
            ],
        }

    @app.post("/evaluations", status_code=202, response_model=EvaluationCreated)
    async def create_evaluation(body: EvaluationConfig, request: Request) -> EvaluationCreated:
        evaluation_id = request.app.state.evaluations.submit(body)
        row = request.app.state.eval_repo.get(evaluation_id)
        return EvaluationCreated(
            evaluation_id=evaluation_id,
            status=row.status,
            planned_runs=row.planned_runs,
            links={
                "self": f"/evaluations/{evaluation_id}",
                "runs": f"/evaluations/{evaluation_id}/runs",
            },
        )

    @app.get("/evaluations", response_model=list[EvaluationOut])
    def list_evaluations(
        request: Request, limit: int = Query(default=20, ge=1, le=100)
    ) -> list[EvaluationOut]:
        rows = request.app.state.eval_repo.list(limit=limit)
        return [EvaluationOut.from_row(r, include_report=False) for r in rows]

    @app.get("/evaluations/{evaluation_id}", response_model=EvaluationOut)
    def get_evaluation(request: Request, evaluation_id: str = _hex_id()) -> EvaluationOut:
        return EvaluationOut.from_row(request.app.state.eval_repo.get(evaluation_id))

    @app.get("/evaluations/{evaluation_id}/runs", response_model=list[EvaluationRunOut])
    def get_evaluation_runs(request: Request, evaluation_id: str = _hex_id()) -> list:
        return [
            EvaluationRunOut.model_validate(r)
            for r in request.app.state.eval_repo.runs(evaluation_id)
        ]

    # ---------------------------------------------------------------- learning
    @app.post("/learning/minings", status_code=201, response_model=MiningOut)
    def create_mining(body: MineRequest, request: Request) -> MiningOut:
        """Mine proposals from completed evaluations (training split only)."""
        ls = learning(request)
        mining_id = ls.mine(
            dataset=body.dataset,
            dataset_version=body.dataset_version,
            evaluation_ids=body.evaluation_ids,
            thresholds=body.thresholds,
        )
        return MiningOut.from_row(ls.get_mining(mining_id))

    @app.get("/learning/minings", response_model=list[MiningOut])
    def list_minings(
        request: Request, limit: int = Query(default=10, ge=1, le=50)
    ) -> list[MiningOut]:
        rows = learning(request).list_minings(limit=limit)
        return [MiningOut.from_row(r) for r in rows]

    @app.get("/learning/minings/{mining_id}", response_model=MiningOut)
    def get_mining(request: Request, mining_id: str = _hex_id()) -> MiningOut:
        return MiningOut.from_row(learning(request).get_mining(mining_id))

    @app.get("/learning/rules", response_model=list[RuleOut])
    def list_rules(
        request: Request,
        status: str | None = Query(default=None, max_length=20),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> list[RuleOut]:
        if status is not None and status not in RULE_STATUSES:
            return _error(422, "invalid_request", f"status must be one of {list(RULE_STATUSES)}")
        rows = learning(request).list_rules(statuses=(status,) if status else None, limit=limit)
        return [RuleOut.from_row(r) for r in rows]

    @app.get("/learning/rules/{rule_id}", response_model=RuleDetailOut)
    def get_rule(request: Request, rule_id: str = _hex_id()) -> RuleDetailOut:
        d = learning(request).detail(rule_id)
        return RuleDetailOut(
            rule=RuleOut.from_row(d["rule"]),
            events=[RuleEventOut.model_validate(e) for e in d["events"]],
            validations=[ValidationOut.from_row(v, include_result=False) for v in d["validations"]],
            shadow_live=d["shadow_live"],
        )

    @app.post("/learning/rules/{rule_id}/shadow", response_model=RuleOut)
    def set_shadow(body: ShadowRequest, request: Request, rule_id: str = _hex_id()) -> RuleOut:
        return RuleOut.from_row(learning(request).set_shadow(rule_id, body.enabled))

    @app.post(
        "/learning/rules/{rule_id}/validations",
        status_code=202,
        response_model=ValidationCreated,
    )
    async def validate_rule(
        body: ValidationConfig, request: Request, rule_id: str = _hex_id()
    ) -> ValidationCreated:
        validation_id = learning(request).submit_validation(rule_id, body)
        return ValidationCreated(
            validation_id=validation_id,
            rule_id=rule_id,
            status="running",
            links={
                "self": f"/learning/validations/{validation_id}",
                "rule": f"/learning/rules/{rule_id}",
            },
        )

    @app.get("/learning/validations/{validation_id}", response_model=ValidationOut)
    def get_validation(request: Request, validation_id: str = _hex_id()) -> ValidationOut:
        row = learning(request).get_validation(validation_id)
        return ValidationOut.from_row(row)

    @app.post("/learning/rules/{rule_id}/approve", response_model=RuleOut)
    def approve_rule(body: ApproveRequest, request: Request, rule_id: str = _hex_id()) -> RuleOut:
        row = learning(request).approve(
            rule_id, approved_by=body.approved_by, note=body.note, confirm=body.confirm
        )
        return RuleOut.from_row(row)

    @app.post("/learning/rules/{rule_id}/disable", response_model=RuleOut)
    def disable_rule(body: DisableRequest, request: Request, rule_id: str = _hex_id()) -> RuleOut:
        row = learning(request).disable(rule_id, reason=body.reason, disabled_by=body.disabled_by)
        return RuleOut.from_row(row)

    return app
