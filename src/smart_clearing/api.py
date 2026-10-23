"""HTTP interface for the simulation.

The service validates inputs, stores simulated cycles, and never sends a
payment instruction. Run it against PostgreSQL from Docker Compose.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, sessionmaker

from smart_clearing import __version__
from smart_clearing.config import SIMULATION_DISCLAIMER, Settings, configure_logging, get_settings
from smart_clearing.model import load_model, unfitted_rules_model
from smart_clearing.schemas import (
    ClearingReport,
    CycleReportResponse,
    CycleSummary,
    HealthResponse,
    IngestResponse,
    NetPosition,
    OptimizeRequest,
    OptimizeResponse,
    PredictRequest,
    PredictResponse,
    TransactionBatch,
)
from smart_clearing.storage import (
    InvalidStateError,
    NotFoundError,
    PayloadConflictError,
    Repository,
    check_database,
    create_db_engine,
    init_database,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the database once per process."""
    settings: Settings = app.state.settings
    configure_logging(settings.log_level)
    engine = create_db_engine(settings)
    init_database(engine)
    app.state.engine = engine
    app.state.session_factory = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True
    )
    logger.info("Smart Clearing API ready (simulation only)")
    yield
    engine.dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application."""
    resolved = settings if settings is not None else get_settings()
    application = FastAPI(
        title="Smart Clearing & Settlement Engine",
        version=__version__,
        description=SIMULATION_DISCLAIMER,
        lifespan=lifespan,
    )
    application.state.settings = resolved
    _register_routes(application)
    return application


def get_session(request: Request) -> Iterator[Session]:
    """Yield a session and commit only when the handler returns."""
    session = request.app.state.session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _register_routes(application: FastAPI) -> None:
    @application.exception_handler(NotFoundError)
    async def _missing(_request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @application.exception_handler(PayloadConflictError)
    async def _conflict(_request: Request, exc: PayloadConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @application.exception_handler(InvalidStateError)
    async def _state(_request: Request, exc: InvalidStateError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        database_ok = check_database(request.app.state.engine)
        return HealthResponse(
            status="ok" if database_ok else "degraded",
            simulation_only=True,
            database="ok" if database_ok else "unavailable",
            disclaimer=SIMULATION_DISCLAIMER,
        )

    @application.post("/transactions", response_model=IngestResponse)
    def post_transactions(
        batch: TransactionBatch,
        session: Session = Depends(get_session),
    ) -> IngestResponse:
        created, existing, identifiers = Repository(session).ingest(batch.transactions)
        return IngestResponse(
            created=created,
            existing=existing,
            transaction_ids=identifiers,
            disclaimer=SIMULATION_DISCLAIMER,
        )

    @application.get("/cycles", response_model=list[CycleSummary])
    def get_cycles(session: Session = Depends(get_session)) -> list[CycleSummary]:
        return Repository(session).list_cycles()

    @application.post("/cycles/{cycle_id}/clear", response_model=ClearingReport)
    def clear_cycle_route(cycle_id: str, session: Session = Depends(get_session)) -> ClearingReport:
        return Repository(session).clear(cycle_id)

    @application.get("/cycles/{cycle_id}/positions", response_model=list[NetPosition])
    def get_positions(cycle_id: str, session: Session = Depends(get_session)) -> list[NetPosition]:
        return Repository(session).positions(cycle_id)

    @application.post("/cycles/{cycle_id}/predict-risk", response_model=PredictResponse)
    def predict_risk(
        cycle_id: str,
        body: PredictRequest,
        request: Request,
        session: Session = Depends(get_session),
    ) -> PredictResponse:
        settings = request.app.state.settings
        if settings.model_path.exists():
            model = load_model(settings.model_path)
            logger.info("Loaded shortfall model from %s", settings.model_path)
        else:
            model = unfitted_rules_model()
            logger.warning(
                "Model artifact %s is missing; using the fixed-scale rules baseline",
                settings.model_path,
            )
        return Repository(session).predict(cycle_id, body.snapshots, model)

    @application.post("/cycles/{cycle_id}/optimize-liquidity", response_model=OptimizeResponse)
    def optimize_liquidity_route(
        cycle_id: str,
        body: OptimizeRequest,
        session: Session = Depends(get_session),
    ) -> OptimizeResponse:
        return Repository(session).optimize(cycle_id, body)

    @application.get("/cycles/{cycle_id}/report", response_model=CycleReportResponse)
    def get_report(cycle_id: str, session: Session = Depends(get_session)) -> CycleReportResponse:
        return Repository(session).report(cycle_id)


app = create_app()
